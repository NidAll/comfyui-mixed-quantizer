#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""LoRA / offload / low-VRAM integration smoke for mixed checkpoints.

Runs on the target machine with real ComfyUI (>= v0.30.0), a real mixed
checkpoint, and (optionally) a LoRA file that touches W4A4, W4A8 and INT8
layers at the same time.

Sequence per memory mode (normal / dynamic / lowvram):
    load mixed checkpoint
    -> forward A          (fixed seeded inputs, reused for A/B/C/D)
    apply LoRA through a real comfy.model_patcher.ModelPatcher and
      comfy.sd.load_lora_for_models
    -> forward B          (assert B differs from A, no NaN)
    offload the model
    remove the LoRA (patcher.unpatch_model), reload
    -> forward C          (assert finite, layouts still match metadata)
    -> forward D          (assert close to A)

Inputs are constructed ONCE per mode with torch.manual_seed(SEED) and the
same tensors are reused for forwards A/B/C/D, so only the LoRA or the mode
can change the output.

Layouts are inspected after every transition, in both directions: every
metadata layer with a format must be loaded with its expected layout, and
every loaded QuantizedTensor layer must exist in the metadata. This catches
format-specific patching/requantization problems (for example ComfyUI issue
#14642, INT8 ConvRot weights re-quantized as plain tensorwise on LoRA
offload).

Memory modes that cannot be enabled (set_lowvram_mode/set_dynamic_vram
raise) are recorded as SKIP with the exception text, excluded from the
pass/fail denominator, and reported separately in the summary.

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

# Fixed seed for the synthetic latents/context; inputs are built once per
# memory mode and reused for forwards A/B/C/D.
SEED = 1234


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
        # raw metadata keys carry the unet prefix (full state-dict keys);
        # they are normalized against the derived prefix in layouts_ok.
        raw_layer_formats = {
            layer: conf.get("format")
            for layer, conf in (qm.get("layers") or {}).items()
            if isinstance(conf, dict)
        }
    except (TypeError, json.JSONDecodeError):  # pragma: no cover
        raw_layer_formats = {}

    def detect_prefix(ckpt) -> str:
        # ComfyUI's own prefix detection (exists since v0.30). The old key
        # heuristic returned "model." instead of "model.diffusion_model."
        # for prefixed checkpoints; keep it only as a fallback for older
        # checkouts.
        try:
            from comfy.model_detection import unet_prefix_from_state_dict
            return unet_prefix_from_state_dict(ckpt)
        except (ImportError, AttributeError):
            print("WARN: comfy.model_detection.unet_prefix_from_state_dict "
                  "unavailable (ComfyUI < v0.30?); falling back to the key "
                  "heuristic")
            return next((k[:k.index(".") + 1] for k in ckpt
                         if k.startswith("model.diffusion_model.")), "")

    def build_model():
        ckpt = load_torch_file(args.model)
        ckpt, _ = convert_old_quants(ckpt, metadata=meta)
        prefix = detect_prefix(ckpt)
        mc = model_detection.model_config_from_unet(ckpt, prefix)
        if mc is None:  # pragma: no cover
            raise RuntimeError("model_config_from_unet failed")
        model = mc.get_model(ckpt)
        model.load_model_weights(ckpt, prefix)
        return model, prefix

    def layouts_ok(model, prefix) -> list[str]:
        # Metadata layer names carry the unet prefix (the converter stores
        # the full state-dict key); module names from named_modules() do not.
        def strip(layer: str) -> str:
            if prefix and layer.startswith(prefix):
                return layer[len(prefix):]
            return layer

        layer_formats = {strip(layer): fmt
                         for layer, fmt in raw_layer_formats.items()}
        bad = []
        qt_by_layer = {}
        for name, m in model.diffusion_model.named_modules():
            w = getattr(m, "weight", None)
            if w is None or type(w).__name__ != "QuantizedTensor":
                continue
            qt_by_layer[name] = w
            fmt = layer_formats.get(name)
            if fmt is None:
                bad.append(f"{name}: QuantizedTensor loaded but missing "
                           f"from metadata")
                continue
            layout = type(w._params).__name__
            expected = EXPECTED_LAYOUTS.get(fmt)
            if expected is not None and layout != expected:
                bad.append(f"{name}: {fmt} -> {layout}")
        # metadata -> loaded: every metadata layer with a known format must
        # be present as a QuantizedTensor with the expected layout
        for layer, fmt in layer_formats.items():
            if fmt not in EXPECTED_LAYOUTS:
                continue
            w = qt_by_layer.get(layer)
            if w is None:
                bad.append(f"{layer}: metadata {fmt} but layer not loaded "
                           f"as QuantizedTensor")
        return bad

    def forward(model, x, t, context) -> torch.Tensor:
        model = model.to(dev)
        return model.diffusion_model(x, t, context=context,
                                     transformer_options={})

    def rel_diff(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a - b).norm() / b.norm().clamp(min=1e-8))

    checks = 0
    failures = 0
    skips = 0

    def check(name: str, ok: bool, detail: str) -> None:
        nonlocal checks, failures
        checks += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name} -- {detail}")
        if not ok:
            failures += 1

    memory_modes = ["normal"]
    if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        memory_modes = ["normal", "dynamic", "lowvram"]

    for mode in memory_modes:
        try:
            from comfy import model_management
            model_management.set_lowvram_mode(mode == "lowvram")
            model_management.set_dynamic_vram(mode == "dynamic")
        except Exception as e:
            print(f"[SKIP] mode: {mode} -- {type(e).__name__}: {e}")
            skips += 1
            continue

        # fixed inputs, constructed once per mode and reused for A/B/C/D
        torch.manual_seed(SEED)
        x = torch.randn(1, 16, 1, 32, 32, dtype=torch.bfloat16, device=dev)
        t = torch.full((1,), 500, dtype=torch.long, device=dev)
        context = torch.randn(1, 77, 4096, dtype=torch.bfloat16, device=dev)

        model, prefix = build_model()
        bad = layouts_ok(model, prefix)
        check(f"{mode}: layouts after load", not bad, "; ".join(bad[:3]) or "ok")
        A = forward(model, x, t, context)
        check(f"{mode}: forward A finite", bool(torch.isfinite(A).all()), "")

        lora_patcher = None
        if args.lora:
            from comfy.utils import load_torch_file as _ltf
            lora_state = _ltf(args.lora)
            try:
                from comfy.model_patcher import ModelPatcher
                from comfy.sd import load_lora_for_models
                # Real ModelPatcher over the same module we forward; the
                # patcher's model attribute must be that exact object.
                model = model.to(dev)
                model.device = dev
                patcher = ModelPatcher(model, load_device=dev,
                                       offload_device=torch.device("cpu"))
                result = load_lora_for_models(
                    patcher, None, lora_state, args.lora_strength,
                    args.lora_strength)
                # ComfyUI >= v0.30 returns (model_patcher, clip_patcher);
                # older versions may legitimately return None instead, in
                # which case the LoRA cannot be applied through this path.
                if result is None or len(result) == 0 or result[0] is None:
                    check(f"{mode}: LoRA apply", False,
                          "load_lora_for_models returned no patcher; this "
                          "ComfyUI version does not return patchers")
                    continue
                lora_patcher = result[0]
                if lora_patcher.model is not model:
                    check(f"{mode}: LoRA patcher identity", False,
                          "returned patcher wraps a different model object")
                    lora_patcher = None
                    continue
                n_patches = sum(len(v)
                                for v in lora_patcher.patches.values())
                if n_patches == 0:
                    print(f"WARN: {mode}: LoRA matched no weights (wrong "
                          f"architecture?); forward B will not change")
                lora_patcher.patch_model()
                print(f"{mode}: LoRA patched through ModelPatcher "
                      f"({n_patches} patch entries)")
            except Exception as e:  # pragma: no cover
                check(f"{mode}: LoRA apply", False, f"{type(e).__name__}: {e}")
                continue
            B = forward(model, x, t, context)
            check(f"{mode}: forward B finite", bool(torch.isfinite(B).all()), "")
            check(f"{mode}: LoRA changes output",
                  rel_diff(B, A) > 1e-4, f"rel diff {rel_diff(B, A):.4f}")

        # offload / reload
        try:
            model_management.unload_all_models()
        except Exception as e:  # noqa: S110 (offload optional)
            print(f"note: offload unavailable ({e})")
        if lora_patcher is not None:
            try:
                lora_patcher.unpatch_model()
            except Exception as e:  # noqa: S110 (unpatch optional)
                print(f"note: LoRA unpatch unavailable ({e})")
            bad = layouts_ok(model, prefix)
            check(f"{mode}: layouts after LoRA remove", not bad,
                  "; ".join(bad[:3]) or "ok")
        model2, prefix2 = build_model()
        bad = layouts_ok(model2, prefix2)
        check(f"{mode}: layouts after offload/reload", not bad,
              "; ".join(bad[:3]) or "ok")
        C = forward(model2, x, t, context)
        check(f"{mode}: forward C finite", bool(torch.isfinite(C).all()), "")
        if lora_patcher is not None:
            # D = same inputs on the model whose LoRA was removed; the
            # reloaded model never saw the LoRA, so both must match A.
            D = forward(model, x, t, context)
            check(f"{mode}: reload without LoRA ~ baseline",
                  rel_diff(C, A) < 0.05, f"rel diff {rel_diff(C, A):.4f}")
            check(f"{mode}: LoRA removed ~ baseline",
                  rel_diff(D, A) < 0.05, f"rel diff {rel_diff(D, A):.4f}")

    n_pass = checks - failures
    print(f"comfyui-patch-smoke: {n_pass} PASS, {skips} SKIP, {failures} FAIL "
          f"({checks} checks across {len(memory_modes)} mode(s))")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
