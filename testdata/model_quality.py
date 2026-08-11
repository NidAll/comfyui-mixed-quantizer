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

Text-encoder mode (quantized TE embedding quality vs the original TE):

    PYTHONPATH=<comfyui-src> python testdata/model_quality.py \
        --te --source original_t5xxl.safetensors \
        --model converted_t5xxl.safetensors --te-clip-type sd3 \
        --te-threshold 0.05

Both modes load through the real ComfyUI path; the TE mode encodes fixed
prompts with both TEs and compares the embedding tensors (cosine + relative
L2) that the CLIP loader exposes (pooled_output / hidden_states).

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
    from comfy.utils import (  # pyright: ignore[reportMissingImports]
        load_torch_file, convert_old_quants)
    from comfy import model_detection  # pyright: ignore[reportMissingImports]
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
    ap.add_argument("--te", action="store_true",
                    help="text-encoder mode: compare TE embeddings of the "
                         "original vs converted TE checkpoints")
    ap.add_argument("--te-clip-type", default="sd1",
                    help="CLIPType for the TE files (sd1, sd2, sdxl, sd3, "
                         "flux, wan, hunyuan_video, krea, ...); default sd1")
    ap.add_argument("--te-threshold", type=float, default=0.05,
                    help="max allowed relative L2 of the TE embeddings")
    ap.add_argument("--te-prompts",
                    default="a red cat sitting on a mat,a blue sky over a calm "
                            "ocean,a bustling city street at night,an astronaut "
                            "on the moon,a bowl of fresh fruit on a wooden table",
                    help="comma-separated prompts for the TE comparison")
    args = ap.parse_args()

    import torch  # pyright: ignore[reportMissingImports]

    dev = torch.device(args.device)
    try:
        import comfy  # pyright: ignore[reportMissingImports]  # noqa: F401
        from safetensors import safe_open  # pyright: ignore[reportMissingImports]
    except ImportError as e:  # pragma: no cover
        print(f"FAIL: ComfyUI/safetensors not importable ({e})")
        return 2
    import os

    for p in (args.model, args.source):
        if not os.path.isfile(p):  # pragma: no cover
            print(f"FAIL: checkpoint not found: {p}")
            return 2

    if args.te:
        return run_te_quality(args)

    with safe_open(args.model, framework="pt", device="cpu") as f:
        meta = dict(f.metadata() or {})
    try:
        qm = json.loads(meta.get("_quantization_metadata", "{}"))
        layer_formats = {
            str(layer): str(conf.get("format"))
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
        rel_l2 = float(((q_flat - ref_flat).norm(dim=1)
                        / ref_flat.norm(dim=1).clamp(min=1e-8)).mean())
        cos = float(((q_flat * ref_flat).sum(dim=1) /
                     (q_flat.norm(dim=1) * ref_flat.norm(dim=1))
                     .clamp(min=1e-12)).mean())
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


def run_te_quality(args) -> int:
    """Text-encoder quality mode: encode fixed prompts with the original and
    the converted TE and compare the embedding tensors the CLIP loader
    exposes (pooled_output / hidden_states / lg_hidden_states)."""
    import torch  # pyright: ignore[reportMissingImports]
    dev = torch.device(args.device)
    try:
        from comfy.sd import load_clip, CLIPType  # pyright: ignore[reportMissingImports]
    except ImportError as e:  # pragma: no cover
        print(f"FAIL: ComfyUI not importable ({e})")
        return 2

    clip_type_name = args.te_clip_type
    member = {"sd1": "STABLE_DIFFUSION", "sd2": "STABLE_DIFFUSION_2",
              "sdxl": "SDXL", "sd3": "SD3", "flux": "FLUX",
              "wan": "WAN", "hunyuan_video": "HUNYUAN_VIDEO",
              "krea": "KREA"}.get(clip_type_name)
    if member is None or not hasattr(CLIPType, member):  # pragma: no cover
        print(f"FAIL: unknown/unsupported --te-clip-type {clip_type_name!r}")
        return 2
    clip_type = getattr(CLIPType, member)

    ref_clip = load_clip(ckpt_paths=[args.source], embedding_directory=None,
                         clip_type=clip_type)
    q_clip = load_clip(ckpt_paths=[args.model], embedding_directory=None,
                       clip_type=clip_type)
    for m in (ref_clip, q_clip):
        m.load_device = dev
    prompts = [p.strip() for p in args.te_prompts.split(",") if p.strip()]

    def embed(clip, prompt):
        tokens = clip.tokenize([prompt])
        cond = clip.encode_tokens(tokens)
        out = {}
        for entry in cond:
            d = entry[1] if isinstance(entry, (list, tuple)) else entry
            if isinstance(d, dict):
                out.update({k: v for k, v in d.items()
                            if hasattr(v, "shape") and v is not None})
        return out

    results = {}
    worst = 0.0
    worst_cos = 1.0
    for prompt in prompts:
        ref_out = embed(ref_clip, prompt)
        q_out = embed(q_clip, prompt)
        common = [k for k in ref_out if k in q_out
                  and ref_out[k].shape == q_out[k].shape]
        if not common:  # pragma: no cover
            print(f"FAIL: no comparable embeddings for prompt {prompt!r}; "
                  f"ref keys {sorted(ref_out)} vs q keys {sorted(q_out)}")
            return 2
        for key in common:
            a = ref_out[key].float().reshape(1, -1)
            b = q_out[key].float().reshape(1, -1)
            rel_l2 = float(((b - a).norm(dim=1) /
                            a.norm(dim=1).clamp(min=1e-8)).mean())
            cos = float(((b * a).sum(dim=1) /
                         (b.norm(dim=1) * a.norm(dim=1))
                         .clamp(min=1e-12)).mean())
            worst = max(worst, rel_l2)
            worst_cos = min(worst_cos, cos)
            results.setdefault(key, []).append(
                {"prompt": prompt, "rel_l2": rel_l2, "cosine": cos})
    ok = worst <= args.te_threshold
    summary = {
        "level": "te-model-verified" if ok else "te-model-failed",
        "mode": "text_encoder",
        "clip_type": clip_type_name,
        "prompts": prompts,
        "worst_rel_l2": worst,
        "worst_cosine": worst_cos,
        "threshold": args.te_threshold,
        "ok": ok,
        "per_key": {k: v for k, v in results.items()},
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    print(f"TE quality: worst relL2 {worst:.4f} cosine {worst_cos:.6f} vs "
          f"threshold {args.te_threshold} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
