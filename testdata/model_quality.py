#!/usr/bin/env python3
"""Model-level BF16-relative quality validation for quantized checkpoints.

Loads the ORIGINAL (BF16/FP16) checkpoint and a quantized checkpoint (w4a8
or mixed) through the real ComfyUI path, runs the diffusion model on
identical synthetic inputs at several timesteps, and compares the denoiser
outputs directly. This is the model gate behind the "model-verified"
certification level: layer-level calibration chooses candidates, this harness
decides whether the whole model stays inside the quality budget.

Primary metrics per timestep (denoiser/model output):

    relative L2
    cosine similarity
    SNR (dB)
    max normalized error

Usage (on the target machine, ComfyUI >= v0.31.0, real checkpoints):

    PYTHONPATH=<comfyui-src> python testdata/model_quality.py \
        --source original_bf16.safetensors \
        --model converted_mixed.safetensors \
        --timesteps 50,500,900 \
        --threshold 0.05

Exit codes: 0 = all timesteps inside the threshold, 1 = any timestep exceeds
it, 2 = could not run (missing ComfyUI/checkpoints).

The synthetic-input defaults match the generic latent format
(1, 16, 1, 32, 32); pass --latent-shape / --context-shape / --context-len for
the architecture under test. Final-image generation (E2E level) is a separate
workflow; this script intentionally measures the denoiser output, which is
the cleaner engineering signal.
"""
from __future__ import annotations

import argparse
import json
import sys

FORMATS = ("convrot_w4a4", "asym_w4a8_int8", "int8_tensorwise")


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="quantized checkpoint")
    ap.add_argument("--source", required=True, help="original BF16 checkpoint")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--timesteps", default="50,500,900",
                    help="comma-separated timestep values")
    ap.add_argument("--timestep-dtype", choices=["long", "float"],
                    default="long")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="max allowed denoiser relative L2 per timestep")
    ap.add_argument("--latent-shape", default="1,16,1,32,32")
    ap.add_argument("--context-shape", default="1,77,4096")
    ap.add_argument("--context-len", type=int, default=77)
    ap.add_argument("--json", default=None, metavar="PATH")
    args = ap.parse_args()

    import torch

    dev = torch.device(args.device)
    try:
        import comfy  # noqa: F401
        from safetensors import safe_open
    except ImportError as e:  # pragma: no cover
        print(f"FAIL: ComfyUI/safetensors not importable ({e})")
        return 2
    import os

    for p in (args.model, args.source):
        if not os.path.isfile(p):  # pragma: no cover
            print(f"FAIL: checkpoint not found: {p}")
            return 2

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

    ref_model, arch = load_model(args.source, {})
    q_model, _ = load_model(args.model, meta)
    print(f"architecture: {arch}; quantized formats in use: "
          f"{sorted(set(layer_formats.values())) or 'none'}")
    for m in (ref_model, q_model):
        m = m.to(dev)

    latent_shape = tuple(int(x) for x in args.latent_shape.split(","))
    context_shape = tuple(int(x) for x in args.context_shape.split(","))
    timesteps = [float(x) for x in args.timesteps.split(",") if x.strip()]

    results = {}
    worst = 0.0
    for t_val in timesteps:
        x = torch.randn(*latent_shape, dtype=torch.bfloat16, device=dev)
        t = (torch.full((1,), int(t_val), dtype=torch.long, device=dev)
             if args.timestep_dtype == "long" else
             torch.full((1,), t_val, dtype=torch.float32, device=dev))
        context = torch.randn(*context_shape, dtype=torch.bfloat16,
                              device=dev)
        kwargs = {"context": context, "transformer_options": {}}
        y_ref = ref_model.diffusion_model(x, t, **kwargs)
        y_q = q_model.diffusion_model(x, t, **kwargs)
        if tuple(y_ref.shape) != tuple(y_q.shape):
            print(f"FAIL: shape mismatch at t={t_val}: "
                  f"{tuple(y_ref.shape)} vs {tuple(y_q.shape)}")
            return 2
        ref_flat = y_ref.float().reshape(y_ref.shape[0], -1)
        q_flat = y_q.float().reshape(y_q.shape[0], -1)
        rel_l2 = float((q_flat - ref_flat).norm(dim=1)
                       / ref_flat.norm(dim=1).clamp(min=1e-8))
        rel_l2 = float(rel_l2.mean())
        cos = float((q_flat * ref_flat).sum(dim=1) /
                    (q_flat.norm(dim=1) * ref_flat.norm(dim=1))
                    .clamp(min=1e-12))
        cos = float(cos.mean())
        snr = (300.0 if rel_l2 < 1e-15 else
               10.0 * float(torch.log10(torch.tensor(1.0 / max(rel_l2, 1e-15)))))
        max_err = float(((q_flat - ref_flat).abs().max(dim=1).values
                         / ref_flat.abs().max(dim=1).values.clamp(min=1e-8)).mean())
        results[str(int(t_val))] = {
            "rel_l2": rel_l2, "cosine": cos, "snr_db": round(snr, 2),
            "max_normalized_error": max_err,
        }
        worst = max(worst, rel_l2)
        print(f"t={t_val:>5}: relL2 {rel_l2:.4f}  cosine {cos:.6f}  "
              f"SNR {snr:.1f} dB  max {max_err:.4f}")

    ok = worst <= args.threshold
    summary = {
        "level": "model-verified" if ok else "model-failed",
        "reference_precision": "bf16/fp16",
        "architecture": arch,
        "quantized_formats": sorted(set(layer_formats.values())),
        "per_timestep": results,
        "worst_rel_l2": worst,
        "threshold": args.threshold,
        "ok": ok,
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    print(f"model quality: worst relL2 {worst:.4f} vs threshold "
          f"{args.threshold} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
