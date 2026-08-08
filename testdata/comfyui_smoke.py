#!/usr/bin/env python3
"""End-to-end ComfyUI smoke inference for a converted W4A8 checkpoint (P0).

Loads the checkpoint through the real ComfyUI path (load_torch_file ->
convert_old_quants -> model_config_from_unet -> get_model ->
load_model_weights), asserts quantized layers are AsymW4A8Int8Layout
QuantizedTensors, then runs one diffusion-model forward on the selected
device. The forward uses synthetic latents/context (no text encoder), so it
reaches the attention/FFN linears including the fused W4A8 CUDA kernels.

Usage (on the user's CUDA machine, ComfyUI >= v0.31.0):
    PYTHONPATH=<comfyui-src> python testdata/comfyui_smoke.py \
        --model boogu_w4a8.safetensors [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="converted W4A8 checkpoint")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from safetensors import safe_open

    dev = torch.device(args.device)
    try:
        import comfy  # noqa: F401
        from comfy.utils import load_torch_file, convert_old_quants
        from comfy import model_detection
    except ImportError as e:  # pragma: no cover
        print(f"FAIL: ComfyUI not importable ({e}); run with "
              "PYTHONPATH pointing at a ComfyUI checkout >= v0.31.0")
        return 2

    with safe_open(args.model, framework="pt", device="cpu") as f:
        meta = dict(f.metadata() or {})
    ckpt = load_torch_file(args.model)
    ckpt, _ = convert_old_quants(ckpt, metadata=meta)
    # prefix-less checkpoints (Comfy-Org repacks) use an empty key prefix;
    # prefixed checkpoints are detected automatically by their keys.
    prefix = next((k[:k.index(".") + 1] for k in ckpt
                   if k.startswith("model.diffusion_model.")), "")
    mc = model_detection.model_config_from_unet(ckpt, prefix)
    if mc is None:  # pragma: no cover
        print("FAIL: model_config_from_unet returned None; is the checkpoint "
              "a supported ComfyUI diffusion model?")
        return 2
    model = mc.get_model(ckpt)
    model.load_model_weights(ckpt, prefix)
    print(f"loaded {type(mc).__name__}; weights on {model.diffusion_model.device}")

    n_qt = 0
    first_conf = None
    for name, m in model.diffusion_model.named_modules():
        w = getattr(m, "weight", None)
        if w is not None and type(w).__name__ == "QuantizedTensor":
            n_qt += 1
            if first_conf is None:
                first_conf = (name, getattr(w._params, "convrot_groupsize", None),
                              getattr(w._params, "group_size", None))
    print(f"QuantizedTensor layers: {n_qt}  (first: {first_conf})")
    if n_qt == 0:  # pragma: no cover
        print("FAIL: no quantized layers loaded; the checkpoint may be all passthrough")
        return 2
    if first_conf is not None and first_conf[1] != 256:  # pragma: no cover
        print(f"FAIL: convrot_groupsize {first_conf[1]} != 256")
        return 2

    # ---- one model evaluation with synthetic inputs ----
    model = model.to(dev)
    dm = model.diffusion_model
    try:
        x = torch.randn(1, 16, 1, 32, 32, dtype=torch.bfloat16, device=dev)
        t = torch.full((1,), 500, dtype=torch.long, device=dev)
        context = torch.randn(1, 77, 4096, dtype=torch.bfloat16, device=dev)
        out = dm(x, t, context=context, transformer_options={})
        print(f"forward OK: {tuple(out.shape)}")
    except Exception as e:  # pragma: no cover
        print(f"FAIL: first forward raised {type(e).__name__}: {e}")
        print("If the shape was wrong, adjust x/context in this script to the "
              "model's latent format; a convrot exception would be: "
              "'convrot fused kernel only supports group_size 256'")
        return 2
    print("comfyui-smoke: pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
