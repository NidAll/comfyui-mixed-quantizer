#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Model-level BF16-relative quality validation for quantized checkpoints.

Loads the ORIGINAL (BF16/FP16) checkpoint and a quantized checkpoint (w4a8
or mixed) through the real ComfyUI path, runs the diffusion model on
identical synthetic inputs at several timesteps, and compares the denoiser
outputs directly. This is the model gate behind the "model-verified"
certification level: layer-level calibration chooses candidates, this harness
decides whether the whole model stays inside the quality budget.

Primary metrics per timestep (denoiser/model output):

    relative L2 (worst across seeds)
    cosine similarity (mean across seeds)
    SNR (dB, 20*log10(1/rel_l2), capped at 300 dB)
    max normalized error (worst across seeds)

Inputs are regenerated from fixed seeds (--seeds, default "0,1,2") so the
measurement is reproducible. The reference and quantized models are never
loaded at the same time: the reference runs all forwards first and its
outputs are kept on CPU, then it is deleted, then the quantized model runs
and all metrics are computed on CPU. This halves the peak VRAM footprint of
the gate.

Usage (on the target machine, ComfyUI >= v0.31.0, real checkpoints):

    PYTHONPATH=<comfyui-src> python testdata/model_quality.py \
        --source original_bf16.safetensors \
        --model converted_mixed.safetensors \
        --timesteps 50,500,900 \
        --seeds 0,1,2 \
        --threshold 0.05 \
        --json model_quality.json

On a passing run a sidecar attestation is written next to --json (or next to
--model when --json is omitted); --attest overrides the path. --stamp-metadata
additionally rewrites the checkpoint's _quantization_metadata
quality_validation.level to "model-verified" in place (this re-reads every
tensor, roughly doubling the I/O).

Exit codes: 0 = all timesteps inside the threshold, 1 = any timestep exceeds
it, 2 = could not run (missing ComfyUI/checkpoints).

The synthetic-input defaults match the generic latent format
(1, 16, 1, 32, 32); pass --latent-shape / --context-shape / --context-len for
the architecture under test, or use --arch-preset for the documented presets
(wan, sdxl, flux). Final-image generation (E2E level) is a separate workflow;
this script intentionally measures the denoiser output, which is the cleaner
engineering signal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone

FORMATS = ("convrot_w4a4", "asym_w4a8_int8", "int8_tensorwise")

# (latent_shape, context_shape) presets. "auto" is the historical default.
ARCH_PRESETS = {
    "wan": ((1, 16, 1, 32, 32), (1, 77, 4096)),
    "sdxl": ((1, 4, 64, 64), (1, 77, 2048)),
    "flux": ((1, 16, 1, 64, 64), (1, 512, 4096)),
    "auto": ((1, 16, 1, 32, 32), (1, 77, 4096)),
}

SNR_DB_CAP = 300.0
SNR_FLOOR = 1e-15


def load_model(path: str, metadata: dict):
    from comfy.utils import load_torch_file, convert_old_quants
    from comfy import model_detection
    ckpt = load_torch_file(path)
    ckpt, _ = convert_old_quants(ckpt, metadata=metadata)
    prefix = next((k[:k.index(".") + 1] for k in ckpt
                   if k.startswith("model.diffusion_model.")), "")
    mc = model_detection.model_config_from_unet(ckpt, prefix)
    if mc is None:
        raise RuntimeError(f"model_config_from_unet failed for {path}")
    model = mc.get_model(ckpt)
    model.load_model_weights(ckpt, prefix)
    return model, type(mc).__name__


def apply_eval(model) -> None:
    """Switch a loaded ComfyUI model to eval mode when it supports it.

    Some ComfyUI model wrappers do not implement .eval(); those are skipped.
    """
    for m in (model, getattr(model, "diffusion_model", None)):
        if m is None:
            continue
        try:
            m.eval()
        except Exception:
            pass


def snr_db(rel_l2: float) -> float:
    """Amplitude-ratio SNR in dB: 20*log10(1/rel_l2), capped at 300 dB."""
    if rel_l2 < SNR_FLOOR:
        return SNR_DB_CAP
    return 20.0 * math.log10(1.0 / max(rel_l2, SNR_FLOOR))


def run_forwards(model, timesteps, seeds, latent_shape, context_shape,
                 timestep_dtype, dev, torch) -> dict:
    """Forward the diffusion model for every (timestep, seed) pair.

    Inputs are regenerated per seed after torch.manual_seed(seed); outputs
    move to CPU immediately, so at most one activation batch lives on the
    device at a time. Returns {t_val: {seed: cpu tensor}}.
    """
    outputs = {}
    with torch.inference_mode():
        for t_val in timesteps:
            if timestep_dtype == "long":
                t = torch.full((1,), int(t_val), dtype=torch.long, device=dev)
            else:
                t = torch.full((1,), t_val, dtype=torch.float32, device=dev)
            per_seed = {}
            for seed in seeds:
                torch.manual_seed(seed)
                x = torch.randn(*latent_shape, dtype=torch.bfloat16,
                                device=dev)
                context = torch.randn(*context_shape, dtype=torch.bfloat16,
                                      device=dev)
                y = model.diffusion_model(
                    x, t, context=context, transformer_options={})
                per_seed[seed] = y.detach().to("cpu")
                del x, context, y
            outputs[t_val] = per_seed
    return outputs


def compare_outputs(ref_outputs, q_outputs, timesteps, seeds, torch):
    """Compute per-(timestep, seed) metrics on CPU, aggregate per timestep.

    Returns (results, worst_rel_l2). The per-row norm mean is batch-safe and
    is the aggregation used for every seed; per-timestep entries report the
    worst rel_l2, the mean cosine, and the worst max error across seeds.
    """
    results = {}
    worst = 0.0
    for t_val in timesteps:
        per_seed = {}
        worst_rel_l2 = 0.0
        cos_sum = 0.0
        worst_max_err = 0.0
        for seed in seeds:
            y_ref = ref_outputs[t_val][seed]
            y_q = q_outputs[t_val][seed]
            if tuple(y_ref.shape) != tuple(y_q.shape):
                raise RuntimeError(
                    f"shape mismatch at t={t_val} seed={seed}: "
                    f"{tuple(y_ref.shape)} vs {tuple(y_q.shape)}")
            ref_flat = y_ref.float().reshape(y_ref.shape[0], -1)
            q_flat = y_q.float().reshape(y_q.shape[0], -1)
            rel_l2 = float(((q_flat - ref_flat).norm(dim=1)
                            / ref_flat.norm(dim=1).clamp(min=1e-8)).mean())
            cos = float(((q_flat * ref_flat).sum(dim=1)
                         / (q_flat.norm(dim=1) * ref_flat.norm(dim=1))
                         .clamp(min=1e-12)).mean())
            max_err = float(((q_flat - ref_flat).abs().max(dim=1).values
                             / ref_flat.abs().max(dim=1).values
                             .clamp(min=1e-8)).mean())
            per_seed[str(seed)] = {
                "rel_l2": rel_l2,
                "cosine": cos,
                "snr_db": round(snr_db(rel_l2), 2),
                "max_normalized_error": max_err,
            }
            worst_rel_l2 = max(worst_rel_l2, rel_l2)
            cos_sum += cos
            worst_max_err = max(worst_max_err, max_err)
        mean_cosine = cos_sum / len(seeds)
        results[str(int(t_val))] = {
            "worst_rel_l2": worst_rel_l2,
            "mean_cosine": mean_cosine,
            "snr_db": round(snr_db(worst_rel_l2), 2),
            "worst_max_normalized_error": worst_max_err,
            "per_seed": per_seed,
            # legacy aliases kept for JSON backward compatibility
            "rel_l2": worst_rel_l2,
            "cosine": mean_cosine,
            "max_normalized_error": worst_max_err,
        }
        worst = max(worst, worst_rel_l2)
    return results, worst


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def default_attest_path(json_path, model_path) -> str:
    if json_path:
        base = json_path[:-5] if json_path.endswith(".json") else json_path
        return base + ".attest.json"
    return model_path + ".quality-attest.json"


def build_attestation(seeds, threshold, worst, arch, formats, model_sha,
                      source_sha, torch_version) -> dict:
    return {
        "level": "model-verified",
        "seeds": seeds,
        "torch_version": torch_version,
        "threshold": threshold,
        "worst_rel_l2": worst,
        "model_sha256": model_sha,
        "source_sha256": source_sha,
        "architecture": arch,
        "quantized_formats": formats,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def stamp_metadata(path: str, meta: dict) -> None:
    """Set quality_validation.level="model-verified" in the checkpoint.

    safetensors metadata cannot be edited in place, so every tensor is
    re-read and the file is rewritten (roughly doubling the I/O). The
    original metadata dict is preserved except for the level field.
    """
    from safetensors import safe_open
    import safetensors.torch

    if not path.endswith(".safetensors"):
        print(f"WARN: --stamp-metadata skipped: {path} is not a safetensors "
              f"file")
        return
    if "_quantization_metadata" not in meta:
        print(f"WARN: --stamp-metadata skipped: {path} has no "
              f"_quantization_metadata")
        return
    print(f"WARN: --stamp-metadata rewrites {path} in place; all tensors are "
          f"re-read and written back (roughly doubles the I/O)")
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = dict(f.metadata() or {})
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    qm = json.loads(meta["_quantization_metadata"])
    qm.setdefault("quality_validation", {})["level"] = "model-verified"
    meta["_quantization_metadata"] = json.dumps(qm, separators=(",", ":"))
    tmp = f"{path}.stamp.{os.getpid()}.tmp"
    safetensors.torch.save_file(tensors, tmp, metadata=meta)
    os.replace(tmp, path)
    print(f"stamped {path}: quality_validation.level=model-verified")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="quantized checkpoint")
    ap.add_argument("--source", required=True, help="original BF16 checkpoint")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--timesteps", default="50,500,900",
                    help="comma-separated timestep values")
    ap.add_argument("--timestep-dtype", choices=["long", "float"],
                    default="long")
    ap.add_argument("--seeds", default="0,1,2",
                    help="comma-separated RNG seeds; inputs are regenerated "
                         "per seed per timestep")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="max allowed denoiser relative L2 per timestep")
    ap.add_argument("--arch-preset", choices=tuple(ARCH_PRESETS),
                    default="auto",
                    help="input shape preset: wan latent 1,16,1,32,32 "
                         "context 1,77,4096; sdxl 1,4,64,64 / 1,77,2048; "
                         "flux 1,16,1,64,64 / 1,512,4096; auto = "
                         "1,16,1,32,32 / 1,77,4096. Explicit "
                         "--latent-shape/--context-shape override it.")
    ap.add_argument("--latent-shape", default=None,
                    help="comma-separated latent shape (default: the "
                         "--arch-preset latent shape)")
    ap.add_argument("--context-shape", default=None,
                    help="comma-separated context shape (default: the "
                         "--arch-preset context shape)")
    ap.add_argument("--context-len", type=int, default=77)
    ap.add_argument("--json", default=None, metavar="PATH",
                    help="write the results summary JSON to PATH")
    ap.add_argument("--attest", default=None, metavar="PATH",
                    help="attestation sidecar path (default: --json path "
                         "with '.attest.json' suffix, or "
                         "<model>.quality-attest.json)")
    ap.add_argument("--stamp-metadata", action="store_true",
                    help="on PASS, rewrite the quantized checkpoint's "
                         "_quantization_metadata quality_validation.level to "
                         "'model-verified' in place (re-reads every tensor; "
                         "roughly doubles the I/O)")
    args = ap.parse_args()

    import torch

    dev = torch.device(args.device)
    try:
        import comfy  # noqa: F401
        from safetensors import safe_open
    except ImportError as e:  # pragma: no cover
        print(f"FAIL: ComfyUI/safetensors not importable ({e})")
        return 2

    for p in (args.model, args.source):
        if not os.path.isfile(p):  # pragma: no cover
            print(f"FAIL: checkpoint not found: {p}")
            return 2

    try:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    except ValueError:
        print(f"FAIL: invalid --seeds value: {args.seeds}")
        return 2
    if not seeds:
        print("FAIL: --seeds must contain at least one seed")
        return 2
    seeds = list(dict.fromkeys(seeds))  # dedupe, preserve order

    with safe_open(args.model, framework="pt", device="cpu") as f:
        meta = dict(f.metadata() or {})
    try:
        qm = json.loads(meta.get("_quantization_metadata", "{}"))
        layer_formats = {
            layer: conf.get("format")
            for layer, conf in (qm.get("layers") or {}).items()
            if isinstance(conf, dict) and conf.get("format") in FORMATS
        }
    except (TypeError, json.JSONDecodeError):  # pragma: no cover
        layer_formats = {}

    preset_latent, preset_context = ARCH_PRESETS[args.arch_preset]
    latent_shape = tuple(
        int(x) for x in (args.latent_shape if args.latent_shape is not None
                         else ",".join(str(v) for v in preset_latent))
        .split(","))
    context_shape = tuple(
        int(x) for x in (args.context_shape if args.context_shape is not None
                         else ",".join(str(v) for v in preset_context))
        .split(","))
    timesteps = [float(x) for x in args.timesteps.split(",") if x.strip()]

    print(f"loading reference model {args.source} ...")
    ref_model, arch = load_model(args.source, {})
    apply_eval(ref_model)
    ref_model.to(dev)
    print(f"architecture: {arch}; quantized formats in use: "
          f"{sorted(set(layer_formats.values())) or 'none'}")
    print(f"running reference forwards: {len(timesteps)} timesteps x "
          f"{len(seeds)} seeds")
    ref_outputs = run_forwards(ref_model, timesteps, seeds, latent_shape,
                               context_shape, args.timestep_dtype, dev, torch)
    del ref_model
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    print("reference done; model unloaded")

    print(f"loading quantized model {args.model} ...")
    q_model, _ = load_model(args.model, meta)
    apply_eval(q_model)
    q_model.to(dev)
    print(f"running quantized forwards: {len(timesteps)} timesteps x "
          f"{len(seeds)} seeds")
    q_outputs = run_forwards(q_model, timesteps, seeds, latent_shape,
                             context_shape, args.timestep_dtype, dev, torch)
    del q_model
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    print("quantized done; model unloaded")

    try:
        results, worst = compare_outputs(ref_outputs, q_outputs, timesteps,
                                         seeds, torch)
    except RuntimeError as e:
        print(f"FAIL: {e}")
        return 2

    for t_val in timesteps:
        entry = results[str(int(t_val))]
        seed_l2s = [round(s["rel_l2"], 4)
                    for s in entry["per_seed"].values()]
        print(f"t={t_val:>5}: worst relL2 {entry['worst_rel_l2']:.4f} "
              f"(seeds {seed_l2s})  cosine {entry['mean_cosine']:.6f}  "
              f"SNR {entry['snr_db']:.1f} dB  "
              f"max {entry['worst_max_normalized_error']:.4f}")

    ok = worst <= args.threshold
    attest_path = None
    if ok:
        if args.stamp_metadata:
            stamp_metadata(args.model, meta)
        # Hash after stamping so the attestation matches the file on disk.
        model_sha = sha256_file(args.model)
        source_sha = sha256_file(args.source)
        attestation = build_attestation(
            seeds, args.threshold, worst, arch,
            sorted(set(layer_formats.values())), model_sha, source_sha,
            torch.__version__)
        attest_path = (args.attest if args.attest
                       else default_attest_path(args.json, args.model))
        with open(attest_path, "w", encoding="utf-8") as f:
            json.dump(attestation, f, indent=2)
        print(f"attestation written: {attest_path}")
    elif args.attest:
        print("WARN: quality gate failed; no attestation written")

    summary = {
        "level": "model-verified" if ok else "model-failed",
        "reference_precision": "bf16/fp16",
        "architecture": arch,
        "quantized_formats": sorted(set(layer_formats.values())),
        "seeds": seeds,
        "per_timestep": results,
        "worst_rel_l2": worst,
        "threshold": args.threshold,
        "ok": ok,
        "attestation": attest_path,
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    print(f"model quality: worst relL2 {worst:.4f} vs threshold "
          f"{args.threshold} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
