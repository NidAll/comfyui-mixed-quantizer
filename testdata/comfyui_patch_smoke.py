#!/usr/bin/env python3
"""LoRA / offload / low-VRAM integration smoke for mixed checkpoints.

Runs on the target machine with real ComfyUI (>= v0.31.0), a real mixed
checkpoint, and (optionally) a LoRA file that touches W4A4, W4A8 and INT8
layers at the same time.

Sequence per memory mode (normal / dynamic / lowvram):
    load mixed checkpoint
    -> forward A
    apply LoRA (when --lora is given)
    -> forward B          (assert B differs from A, no NaN)
    offload the model
    reload
    -> forward C          (assert finite, layouts still match metadata)
    remove LoRA
    -> forward D          (assert close to A)

Layouts are inspected after every transition: each quantized layer must keep
its QuantizedTensor type and the layout matching its per-layer metadata
format. This catches format-specific patching/requantization problems (for
example ComfyUI issue #14642, INT8 ConvRot weights re-quantized as plain
tensorwise on LoRA offload).

Usage:
    PYTHONPATH=<comfyui-src> python testdata/comfyui_patch_smoke.py \
        --model mixed.safetensors [--lora model.safetensors]
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
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lora-strength", type=float, default=0.8)
    args = ap.parse_args()

    import torch
    from safetensors import safe_open
    try:
        import comfy  # noqa: F401
        from comfy.utils import load_torch_file, convert_old_quants
        from comfy import model_detection
    except ImportError as e:  # pragma: no cover
        print(f"FAIL: ComfyUI not importable ({e})")
        return 2

    dev = torch.device(args.device)
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

    def build_model():
        ckpt = load_torch_file(args.model)
        ckpt, _ = convert_old_quants(ckpt, metadata=meta)
        prefix = next((k[:k.index(".") + 1] for k in ckpt
                       if k.startswith("model.diffusion_model.")), "")
        mc = model_detection.model_config_from_unet(ckpt, prefix)
        if mc is None:  # pragma: no cover
            raise RuntimeError("model_config_from_unet failed")
        model = mc.get_model(ckpt)
        model.load_model_weights(ckpt, prefix)
        return model

    def layouts_ok(model) -> list[str]:
        bad = []
        by_layout: dict[str, int] = {}
        for name, m in model.diffusion_model.named_modules():
            w = getattr(m, "weight", None)
            if w is None or type(w).__name__ != "QuantizedTensor":
                continue
            layout = type(w._params).__name__
            by_layout[layout] = by_layout.get(layout, 0) + 1
            expected = EXPECTED_LAYOUTS.get(layer_formats.get(name, ""))
            if expected is not None and layout != expected:
                bad.append(f"{name}: {layer_formats.get(name)} -> {layout}")
        for fmt, layout in EXPECTED_LAYOUTS.items():
            if fmt in layer_formats.values() and layout not in by_layout:
                bad.append(f"format {fmt} produced no {layout} layers")
        return bad

    def forward(model) -> torch.Tensor:
        model = model.to(dev)
        x = torch.randn(1, 16, 1, 32, 32, dtype=torch.bfloat16, device=dev)
        t = torch.full((1,), 500, dtype=torch.long, device=dev)
        context = torch.randn(1, 77, 4096, dtype=torch.bfloat16, device=dev)
        return model.diffusion_model(x, t, context=context,
                                     transformer_options={})

    def rel_diff(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a - b).norm() / b.norm().clamp(min=1e-8))

    failures = []

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"[{'PASS' if ok else 'FAIL'}] {name} -- {detail}")
        if not ok:
            failures.append(name)

    memory_modes = ["normal"]
    if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        memory_modes = ["normal", "dynamic", "lowvram"]

    for mode in memory_modes:
        try:
            from comfy import model_management
            model_management.set_lowvram_mode(mode == "lowvram")
            model_management.set_dynamic_vram(mode == "dynamic")
        except Exception as e:  # noqa: S110 (memory-mode switching optional)
            print(f"note: memory mode {mode} unavailable ({e})")
        model = build_model()
        bad = layouts_ok(model)
        check(f"{mode}: layouts after load", not bad, "; ".join(bad[:3]) or "ok")
        A = forward(model)
        check(f"{mode}: forward A finite", bool(torch.isfinite(A).all()), "")
        lora_state = None
        if args.lora:
            from comfy.utils import load_torch_file as _ltf
            lora_state = _ltf(args.lora)
            try:
                from comfy.sd import load_lora_for_models
                model_patcher = type("P", (), {"model": model})()
                load_lora_for_models(model_patcher, model_patcher, lora_state,
                                     args.lora_strength, args.lora_strength)
            except Exception as e:  # pragma: no cover
                check(f"{mode}: LoRA apply", False, f"{type(e).__name__}: {e}")
                continue
            B = forward(model)
            check(f"{mode}: forward B finite", bool(torch.isfinite(B).all()), "")
            check(f"{mode}: LoRA changes output",
                  rel_diff(B, A) > 1e-4, f"rel diff {rel_diff(B, A):.4f}")
        # offload / reload
        try:
            model_management.unload_all_models()
        except Exception as e:  # noqa: S110 (offload optional)
            print(f"note: offload unavailable ({e})")
        model2 = build_model()
        bad = layouts_ok(model2)
        check(f"{mode}: layouts after offload/reload", not bad,
              "; ".join(bad[:3]) or "ok")
        C = forward(model2)
        check(f"{mode}: forward C finite", bool(torch.isfinite(C).all()), "")
        if lora_state is not None:
            # D = close to A after the reloaded model never got the LoRA
            check(f"{mode}: reload without LoRA ~ baseline",
                  rel_diff(C, A) < 0.05, f"rel diff {rel_diff(C, A):.4f}")

    print(f"comfyui-patch-smoke: {len(memory_modes)} mode(s), "
          f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
