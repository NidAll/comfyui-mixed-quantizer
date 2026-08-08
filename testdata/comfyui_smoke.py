#!/usr/bin/env python3
"""End-to-end ComfyUI smoke inference for a converted checkpoint (W4A8 or mixed).

Loads the checkpoint through the real ComfyUI path (load_torch_file ->
convert_old_quants -> model_config_from_unet -> get_model ->
load_model_weights), asserts every quantized layer is a QuantizedTensor whose
layout matches its per-layer metadata format, then runs one diffusion-model
forward on the selected device. The forward uses synthetic latents/context
(no text encoder), so it reaches the attention/FFN linears including the
fused CUDA kernels.

Expected layouts per metadata format:

    convrot_w4a4     -> TensorCoreConvRotW4A4Layout
    asym_w4a8_int8   -> AsymW4A8Int8Layout
    int8_tensorwise  -> TensorWiseINT8Layout

Usage (on the user's CUDA machine, ComfyUI >= v0.31.0):
    PYTHONPATH=<comfyui-src> python testdata/comfyui_smoke.py \
        --model checkpoint.safetensors [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import sys

EXPECTED_LAYOUTS = {
    "convrot_w4a4": "TensorCoreConvRotW4A4Layout",
    "asym_w4a8_int8": "AsymW4A8Int8Layout",
    "int8_tensorwise": "TensorWiseINT8Layout",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    help="converted W4A8 or mixed checkpoint")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--require-format", action="append", default=[],
                    choices=["convrot_w4a4", "asym_w4a8_int8",
                             "int8_tensorwise"],
                    help="strict mode: the checkpoint must contain this "
                         "format (repeatable) and it must load with the "
                         "matching layout")
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
    try:
        qm = json.loads(meta.get("_quantization_metadata", "{}"))
        layer_formats = {
            layer: conf.get("format")
            for layer, conf in (qm.get("layers") or {}).items()
            if isinstance(conf, dict)
        }
    except (TypeError, json.JSONDecodeError):  # pragma: no cover
        layer_formats = {}
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

    # ---- per-layer layout assertions against the metadata ----
    qt_by_layer = {}
    for name, m in model.diffusion_model.named_modules():
        w = getattr(m, "weight", None)
        if w is not None and type(w).__name__ == "QuantizedTensor":
            qt_by_layer[name] = w
    n_qt = len(qt_by_layer)
    print(f"QuantizedTensor layers: {n_qt}")
    if n_qt == 0:  # pragma: no cover
        print("FAIL: no quantized layers loaded; the checkpoint may be all passthrough")
        return 2

    layout_counts: dict[str, int] = {}
    failures: list[str] = []
    for layer, w in qt_by_layer.items():
        layout = type(w._params).__name__
        layout_counts[layout] = layout_counts.get(layout, 0) + 1
        expected = EXPECTED_LAYOUTS.get(layer_formats.get(layer, ""))
        if expected is not None and layout != expected:
            failures.append(f"{layer}: metadata {layer_formats.get(layer)} "
                            f"but layout {layout}")
    for fmt, layout in EXPECTED_LAYOUTS.items():
        if fmt in layer_formats.values() and layout not in layout_counts:
            failures.append(f"metadata format {fmt} produced no {layout} layers")
    if failures:  # pragma: no cover
        print("FAIL: layout/metadata mismatches:")
        for f in failures[:10]:
            print(f"  - {f}")
        return 2
    print(f"layouts: {layout_counts}")
    # strict mode: every required format must actually be present and loaded
    if args.require_format:
        present = {fmt for fmt in args.require_format
                   if fmt in layer_formats.values()}
        missing = [fmt for fmt in args.require_format if fmt not in present]
        if missing:  # pragma: no cover
            print(f"FAIL: --require-format missing from the checkpoint: "
                  f"{missing}")
            return 2
    if set(layer_formats.values()) - set(EXPECTED_LAYOUTS):
        print(f"WARN: metadata contains unhandled formats: "
              f"{set(layer_formats.values()) - set(EXPECTED_LAYOUTS)}")
    if n_qt != len(layer_formats):
        print(f"WARN: {n_qt} quantized modules vs {len(layer_formats)} "
              "metadata entries (expected when shapes mismatch the model)")

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
