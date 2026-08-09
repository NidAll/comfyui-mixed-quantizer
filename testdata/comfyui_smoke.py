#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end ComfyUI smoke inference for a converted checkpoint (W4A8 or mixed).

Loads the checkpoint through the real ComfyUI path (load_torch_file ->
convert_old_quants(metadata=meta) -> model_config_from_unet(ckpt, prefix) ->
get_model -> load_model_weights), using ComfyUI's own
model_detection.unet_prefix_from_state_dict for the state-dict prefix
(fallback: the old key heuristic with a warning). It then asserts every
quantized layer is a QuantizedTensor whose layout matches its per-layer
metadata format in BOTH directions (every metadata layer with a format must
be loaded with the expected layout, and every loaded QuantizedTensor layer
must exist in the metadata), and runs one diffusion-model forward on the
selected device. The forward uses synthetic latents/context (no text
encoder), so it reaches the attention/FFN linears including the fused CUDA
kernels.

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
        # raw metadata keys carry the unet prefix (full state-dict keys);
        # they are normalized against the derived prefix below.
        raw_layer_formats = {
            layer: conf.get("format")
            for layer, conf in (qm.get("layers") or {}).items()
            if isinstance(conf, dict)
        }
    except (TypeError, json.JSONDecodeError):  # pragma: no cover
        raw_layer_formats = {}
    ckpt = load_torch_file(args.model)
    ckpt, _ = convert_old_quants(ckpt, metadata=meta)
    # ComfyUI's own prefix detection (exists since v0.30). The old key
    # heuristic returned "model." instead of "model.diffusion_model." for
    # prefixed checkpoints; keep it only as a fallback for older checkouts.
    try:
        from comfy.model_detection import unet_prefix_from_state_dict
        prefix = unet_prefix_from_state_dict(ckpt)
    except (ImportError, AttributeError):
        print("WARN: comfy.model_detection.unet_prefix_from_state_dict "
              "unavailable (ComfyUI < v0.30?); falling back to the key "
              "heuristic")
        prefix = next((k[:k.index(".") + 1] for k in ckpt
                       if k.startswith("model.diffusion_model.")), "")
    mc = model_detection.model_config_from_unet(ckpt, prefix)
    if mc is None:  # pragma: no cover
        print("FAIL: model_config_from_unet returned None; is the checkpoint "
              "a supported ComfyUI diffusion model?")
        return 2
    model = mc.get_model(ckpt)
    model.load_model_weights(ckpt, prefix)
    print(f"loaded {type(mc).__name__} (prefix {prefix!r}); "
          f"weights on {model.diffusion_model.device}")

    # ---- per-layer layout assertions against the metadata ----
    # Metadata layer names carry the unet prefix (the converter stores the
    # full state-dict key); module names from named_modules() do not.
    def strip_prefix(layer: str) -> str:
        if prefix and layer.startswith(prefix):
            return layer[len(prefix):]
        return layer

    layer_formats = {strip_prefix(layer): fmt
                     for layer, fmt in raw_layer_formats.items()}

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
        fmt = layer_formats.get(layer)
        if fmt is None:
            failures.append(f"{layer}: QuantizedTensor loaded but missing "
                            f"from metadata")
            continue
        expected = EXPECTED_LAYOUTS.get(fmt)
        if expected is not None and layout != expected:
            failures.append(f"{layer}: metadata {fmt} but layout {layout}")
    # metadata -> loaded: every metadata layer with a known format must be
    # present as a QuantizedTensor with the expected layout
    for layer, fmt in layer_formats.items():
        if fmt not in EXPECTED_LAYOUTS:
            continue
        if layer not in qt_by_layer:
            failures.append(f"{layer}: metadata format {fmt} but layer not "
                            f"loaded as QuantizedTensor")
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
    unhandled = set(layer_formats.values()) - set(EXPECTED_LAYOUTS)
    if unhandled:
        print(f"WARN: metadata contains unhandled formats: {unhandled}")
    if n_qt != len(layer_formats):
        print(f"WARN: {n_qt} quantized modules vs {len(layer_formats)} "
              "metadata entries (expected for unhandled formats or layers "
              "outside the unet)")

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
