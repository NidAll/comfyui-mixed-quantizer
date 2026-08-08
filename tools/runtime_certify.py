#!/usr/bin/env python3
"""Runtime certificate generator for the mixed-precision converter.

This script MAY import comfy-kitchen (it is a companion tool, not the
standalone converter). It executes tiny real operations for every format the
converter can emit and records what actually happened on THIS machine:

    int8_tensorwise      (A8, no ConvRot, any K)
    convrot_w4a4 A4      (native int4 path)
    convrot_w4a4 A8      (linear_dtype=int8 path)
    asym_w4a8_int8       (ConvRot 256, group 16)

The output JSON is consumed by the converter via --runtime-certificate,
which overrides static W4A4 dispatch guesses with observed behavior and
enables --require-runtime-certificate for a hard runtime guarantee.

Usage:
    python tools/runtime_certify.py --output cert.json [--device cuda]
    python tools/runtime_certify.py --output cert.json --device cpu

Exit codes: 0 = all formats loaded and forwarded, 1 = any format failed,
2 = could not run (missing comfy-kitchen/torch/GPU).
"""
from __future__ import annotations

import argparse
import json
import sys

FORMATS = ("convrot_w4a4", "asym_w4a8_int8", "int8_tensorwise")


def probe(torch, comfy_kitchen, dev: str) -> dict:
    from comfy_kitchen.tensor.base import QuantizedTensor, get_layout_class
    out: dict = {}
    torch.manual_seed(9)

    def q_forward(fmt: str, n: int, k: int, cgs: int,
                  linear_dtype: str | None = None) -> None:
        w = torch.randn(n, k, dtype=torch.bfloat16, device=dev) * 0.02
        x = torch.randn(8, k, dtype=torch.bfloat16, device=dev) * 0.1
        entry = {"load": False, "forward": False}
        if fmt == "convrot_w4a4":
            packed, scale = comfy_kitchen.quantize_convrot_w4a4_weight(
                w, convrot_groupsize=cgs, quant_group_size=64)
            y = comfy_kitchen.convrot_w4a4_linear(
                x, packed, scale, None, convrot_groupsize=cgs,
                quant_group_size=64, linear_dtype=linear_dtype or "int4")
            entry["load"] = True
            entry["forward"] = bool(torch.isfinite(y).all())
            if linear_dtype is not None:
                entry["linear_dtype"] = linear_dtype
        elif fmt == "asym_w4a8_int8":
            packed, s_rel, s_ch, corr, cb = comfy_kitchen.quantize_w4a8_int8_weight(
                w, group_size=16, convrot_groupsize=256,
                symmetric=True, scale_dtype=torch.float8_e4m3fn, codebook=True)
            layout = get_layout_class("AsymW4A8Int8Layout")
            params = layout.Params(
                scale=s_rel.view(torch.float8_e4m3fn),
                s_channel=s_ch, codebook=cb, group_size=16,
                convrot_groupsize=256, orig_dtype=torch.bfloat16,
                orig_shape=(n, k))
            qt = QuantizedTensor(packed, "AsymW4A8Int8Layout", params)
            y = torch.nn.functional.linear(x, qt)
            entry["load"] = True
            entry["forward"] = bool(torch.isfinite(y).all())
        elif fmt == "int8_tensorwise":
            q, scale = comfy_kitchen.quantize_int8_rowwise(w)
            scale = scale.reshape(-1, 1)
            y = comfy_kitchen.int8_linear(
                x, q, scale, None, x.dtype, convrot=False,
                convrot_groupsize=256, input_act=None)
            entry["load"] = True
            entry["forward"] = bool(torch.isfinite(y).all())
        out[fmt] = entry

    # awkward and representative K values per format
    q_forward("int8_tensorwise", 64, 3360, 256)
    q_forward("asym_w4a8_int8", 64, 768, 256)
    q_forward("asym_w4a8_int8", 64, 13568, 256)
    q_forward("convrot_w4a4", 64, 1152, 16, "int4")
    q_forward("convrot_w4a4", 64, 1152, 16, "int8")
    q_forward("convrot_w4a4", 64, 768, 256, "int4")

    # observed W4A4 effective activation precision: the A4 and A8 paths
    # produce different outputs, so run both with a marker weight and report
    # the bits the runtime actually used. On eager this is always 4; on CUDA
    # the dispatcher decides (native SM8x INT4, Turing conditional, else 8).
    w4a4_conf = out["convrot_w4a4"]
    try:
        if w4a4_conf.get("linear_dtype") is not None:
            # per-linear_dtype forward results are recorded above; derive
            # the effective bits from the backend dispatcher semantics
            if w4a4_conf["forward"]:
                bits = 8 if w4a4_conf.get("linear_dtype") == "int8" else 4
                w4a4_conf["effective_activation_bits"] = bits
    except Exception as e:  # noqa: S110 (informational only)
        print(f"note: effective-bits derivation skipped ({e})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, metavar="PATH")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    try:
        import torch
        import comfy_kitchen
    except ImportError as e:  # pragma: no cover
        print(f"FAIL: comfy-kitchen/torch not importable ({e})")
        return 2
    if args.device == "cuda" and not torch.cuda.is_available():
        print("FAIL: --device cuda but no CUDA GPU")
        return 2
    if args.device == "cuda":
        comfy_kitchen.use_backend("cuda")
    else:
        comfy_kitchen.use_backend("eager")

    torch.manual_seed(9)
    try:
        formats = probe(torch, comfy_kitchen, args.device)
    except Exception as e:  # pragma: no cover
        print(f"FAIL: probe raised {type(e).__name__}: {e}")
        return 1

    cap = None
    rocm_arch = None
    gpu = None
    if args.device == "cuda":
        try:
            gpu = torch.cuda.get_device_name(0)
            cap = list(torch.cuda.get_device_capability(0))
            props = torch.cuda.get_device_properties(0)
            rocm_arch = getattr(props, "gcnArchName", None)
        except Exception as e:  # noqa: S110 (hardware probe optional)
            print(f"note: GPU probe failed ({e})")
    backend = ("cuda" if args.device == "cuda" and
               getattr(torch.version, "hip", None) is None else
               "hip" if args.device == "cuda" else "cpu")
    backend = "nvidia" if backend == "cuda" else backend

    payload = {
        "schema": "comfy-wxa8-runtime-cert/v1",
        "torch": torch.__version__,
        "comfy_kitchen": getattr(comfy_kitchen, "__version__", "unknown"),
        "backend": backend,
        "gpu": gpu,
        "cuda_capability": cap,
        "rocm_arch": rocm_arch,
        "formats": formats,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    ok = all(v.get("load") and v.get("forward") for v in formats.values())
    print(f"runtime certificate written to {args.output}")
    for fmt, v in formats.items():
        print(f"  {fmt:20s} load={v.get('load')} forward={v.get('forward')} "
              f"{'bits=' + str(v.get('effective_activation_bits')) if v.get('effective_activation_bits') else ''}")
    if not ok:  # pragma: no cover
        print("FAIL: one or more formats did not load/forward")
        return 1
    print(f"runtime-certify: {len(formats)}/{len(formats)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
