#!/usr/bin/env python3
"""CUDA regression smoke test for the W4A8 runtime contract (P0).

Runs three checks on a machine with an NVIDIA GPU:
  1. Quantize K=256/768/13568 tensors with the converter math on CUDA, build
     comfy-kitchen AsymW4A8Int8Layout tensors, and run the fused CUDA W4A8
     linear; compare against the dequantized matmul reference.
  2. Prove a deliberately invalid ConvRot configuration (convrot_groupsize 64,
     the historical v1.2.1 bug) is rejected by --validation-only before any
     runtime kernel is reached.
  3. (optional, --model) Load a mixed W4A8 checkpoint through real ComfyUI and
     run one diffusion-model forward; see testdata/comfyui_smoke.py.

Exit codes: 0 = passed (or skipped when CUDA is unavailable), 2 = failed.
The CPU CI matrix skips this script; run it on the user's CUDA machine or a
self-hosted GPU runner (see .github/workflows/cuda-smoke.yml).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONVERTER = os.path.join(REPO, "comfyui_wxa8_quantizer.py")
sys.path.insert(0, REPO)

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name} -- {detail}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None,
                    help="mixed W4A8 checkpoint for the ComfyUI forward check")
    ap.add_argument("--source", default=None,
                    help="original (unquantized) checkpoint matching --model")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("SKIPPED: CUDA unavailable (CPU CI); run on a GPU machine")
        return 0

    from comfy_kitchen.tensor.base import QuantizedTensor, get_layout_class
    import comfy_kitchen.backends.cuda  # noqa: F401  (registers the CUDA backend)
    import comfy_kitchen
    comfy_kitchen.use_backend("cuda")

    # ---- check 1: fused CUDA W4A8 linear on compatible K values ----
    dev = torch.device(args.device)
    try:
        for k in (256, 768, 13568):
            torch.manual_seed(k)
            weight = torch.randn(384, k, dtype=torch.bfloat16, device=dev) * 0.02
            packed, s_rel, s_ch, corr, cb = __import__("comfyui_wxa8_quantizer",
                                                       fromlist=["quantize_w4a8_weight"]).quantize_w4a8_weight(
                weight, group_size=16, convrot_groupsize=256,
                symmetric=True, scale_dtype=torch.float8_e4m3fn, codebook=True)
            assert corr is None
            layout = get_layout_class("AsymW4A8Int8Layout")
            params = layout.Params(
                scale=s_rel.view(torch.float8_e4m3fn),
                s_channel=s_ch, codebook=cb,
                group_size=16, convrot_groupsize=256,
                orig_dtype=torch.bfloat16, orig_shape=(384, k))
            qt = QuantizedTensor(packed, "AsymW4A8Int8Layout", params)
            x = torch.randn(8, k, dtype=torch.bfloat16, device=dev) * 0.1
            y = torch.nn.functional.linear(x, qt)
            y_ref = torch.nn.functional.linear(x, qt.dequantize())
            err = (y - y_ref).abs().max().item()
            ok = err < 0.1 and tuple(y.shape) == (8, 384)
            check(f"cuda-w4a8-linear-K{k}", ok,
                  f"max err {err:.4f} vs dequant reference" if ok else
                  f"err {err:.4f} / shape {tuple(y.shape)}")
    except Exception as e:  # pragma: no cover
        check("cuda-w4a8-linear", False, f"{type(e).__name__}: {e}")

    # ---- check 2: invalid ConvRot config rejected before runtime ----
    try:
        with tempfile.TemporaryDirectory() as d:
            bad = os.path.join(d, "cgs64.safetensors")
            torch.manual_seed(3)
            w = torch.randn(64, 320, dtype=torch.float16)
            meta = {"_quantization_metadata": json.dumps({"layers": {
                "model.diffusion_model.test_layer": {
                    "format": "asym_w4a8_int8", "group_size": 16,
                    "convrot": True, "convrot_groupsize": 64}}})}
            import safetensors.torch
            safetensors.torch.save_file({
                "model.diffusion_model.test_layer.weight": w,
                "model.diffusion_model.test_layer.weight_s_rel": torch.zeros(64, 20, dtype=torch.uint8),
                "model.diffusion_model.test_layer.weight_s_channel": torch.ones(64, dtype=torch.float32),
                "model.diffusion_model.test_layer.weight_codebook": torch.zeros(16, dtype=torch.float32),
            }, bad, metadata=meta)
            bad_out = bad + ".out"
            import shutil
            shutil.copyfile(bad, bad_out)
            r = subprocess.run(  # noqa: S603 (test-only invocation)
                [sys.executable, CONVERTER, bad, "--output", bad_out,
                 "--validation-only", "--architecture", "sdxl"],
                capture_output=True, text=True, timeout=300)
            ok = r.returncode != 0 and "convrot_groupsize" in (r.stderr + r.stdout)
            check("invalid-convrot-rejected", ok,
                  "v1.2.1-style cgs=64 file refused by --validation-only"
                  if ok else r.stderr[-300:])
    except Exception as e:  # pragma: no cover
        check("invalid-convrot-rejected", False, f"{type(e).__name__}: {e}")

    # ---- check 2.5: mixed Boogu-like checkpoint through the fused kernels ----
    # The converted boogu_real fixture has real K=3360 passthrough layers and
    # real K=13568 W4A8 layers. Run both through the CUDA linear path exactly
    # as ComfyUI would: QuantizedTensors for the W4A8 layers, plain bf16 for
    # the rest. This is the failure class from the original Boogu crash.
    fixture = os.path.join(REPO, "testdata", "boogu_real_fixture_w4a8.safetensors")
    if os.path.exists(fixture):
        try:
            import safetensors.torch
            import torch.nn.functional as F
            sd = safetensors.torch.load_file(fixture, device="cpu")
            with __import__("safetensors").safe_open(fixture, framework="pt", device="cpu") as f:
                qm = json.loads(f.metadata()["_quantization_metadata"])
            layout = get_layout_class("AsymW4A8Int8Layout")
            n_qt = 0
            errs = []
            for layer, conf in qm["layers"].items():
                k = int(conf["group_size"]); cgs = int(conf["convrot_groupsize"])
                # orig_shape needs (N, K); packed is [N, K/2]
                packed = sd[layer + ".weight"].to(dev)
                params = layout.Params(
                    scale=sd[layer + ".weight_s_rel"].to(dev).view(torch.float8_e4m3fn),
                    s_channel=sd[layer + ".weight_s_channel"].to(dev),
                    codebook=sd[layer + ".weight_codebook"].to(dev),
                    group_size=k, convrot_groupsize=cgs,
                    orig_dtype=torch.bfloat16,
                    orig_shape=(packed.shape[0], packed.shape[1] * 2))
                qt = QuantizedTensor(packed, "AsymW4A8Int8Layout", params)
                x = torch.randn(4, packed.shape[1] * 2, dtype=torch.bfloat16, device=dev)
                y = F.linear(x, qt)
                y_ref = F.linear(x, qt.dequantize())
                denom = max(float(y.abs().max()), 1e-6)
                errs.append((y - y_ref).abs().max().item() / denom)
                n_qt += 1
            ok = n_qt > 0 and max(errs) < 0.1
            check("mixed-checkpoint-cuda-forward", ok,
                  f"{n_qt} W4A8 layers ran through the fused CUDA kernels, "
                  f"max rel err {max(errs):.4f}"
                  if ok else f"n_qt={n_qt} rel errs={errs[:3]}")
        except Exception as e:  # pragma: no cover
            check("mixed-checkpoint-cuda-forward", False, f"{type(e).__name__}: {e}")
    else:
        print("SKIP mixed-checkpoint-cuda-forward (fixture not generated)")

    # ---- check 2.6: int8_tensorwise (no ConvRot) at Boogu K=3360 on CUDA ----
    try:
        torch.manual_seed(3360)
        w = torch.randn(384, 3360, dtype=torch.bfloat16, device=dev) * 0.02
        q, scale = __import__("comfyui_wxa8_quantizer",
                              fromlist=["quantize_int8_tensorwise_weight"]).quantize_int8_tensorwise_weight(
            w)
        x = torch.randn(8, 3360, dtype=torch.bfloat16, device=dev) * 0.1
        y = comfy_kitchen.int8_linear(x, q, scale, None, x.dtype,
                                      convrot=False, convrot_groupsize=256,
                                      input_act=None)
        y_ref = torch.nn.functional.linear(x, w)
        err = (y - y_ref).abs().max().item() / max(float(y_ref.abs().max()), 1e-6)
        ok = err < 0.05 and tuple(y.shape) == (8, 384)
        check("cuda-int8-tensorwise-K3360", ok,
              f"max rel err {err:.4f} (dynamic int8 activations)"
              if ok else f"err {err:.4f} / shape {tuple(y.shape)}")
    except Exception as e:  # pragma: no cover
        check("cuda-int8-tensorwise-K3360", False, f"{type(e).__name__}: {e}")

    # ---- check 2.7: convrot_w4a4 at awkward K (1152, cgs=16) on CUDA ----
    try:
        from comfy_kitchen.tensor.convrot_w4a4 import (
            quantize_convrot_w4a4_weight)
        torch.manual_seed(1152)
        w = torch.randn(384, 1152, dtype=torch.bfloat16, device=dev) * 0.02
        qw, ws = quantize_convrot_w4a4_weight(w, convrot_groupsize=16,
                                              quant_group_size=64)
        x = torch.randn(8, 1152, dtype=torch.bfloat16, device=dev) * 0.1
        for ld in ("int8", "int4"):
            y = comfy_kitchen.convrot_w4a4_linear(
                x, qw, ws, None, convrot_groupsize=16,
                quant_group_size=64, linear_dtype=ld)
            y_ref = torch.nn.functional.linear(x, w)
            err = (y - y_ref).abs().max().item() / max(float(y_ref.abs().max()), 1e-6)
            ok = err < 0.35 and tuple(y.shape) == (8, 384)
            check(f"cuda-w4a4-K1152-cgs16-{ld}", ok,
                  f"max rel err {err:.4f}" if ok else f"err {err:.4f} / shape {tuple(y.shape)}")
    except Exception as e:  # pragma: no cover
        check("cuda-w4a4-K1152-cgs16", False, f"{type(e).__name__}: {e}")

    # ---- check 2.8: mixed checkpoint through the CUDA kernels ----
    mixed_fixture = os.path.join(REPO, "testdata", "boogu_real_fixture_mixed.safetensors")
    if os.path.exists(mixed_fixture):
        try:
            import safetensors.torch
            sd = safetensors.torch.load_file(mixed_fixture, device="cpu")
            with __import__("safetensors").safe_open(mixed_fixture, framework="pt", device="cpu") as f:
                qm = json.loads(f.metadata()["_quantization_metadata"])
            n_int8 = n_w4a8 = 0
            errs = []
            for layer, conf in qm["layers"].items():
                lfmt = conf["format"]
                packed = sd[layer + ".weight"].to(dev)
                if lfmt == "int8_tensorwise":
                    scale = sd[layer + ".weight_scale"].to(dev)
                    x = torch.randn(4, packed.shape[1], dtype=torch.bfloat16, device=dev)
                    y = comfy_kitchen.int8_linear(
                        x, packed, scale, None, x.dtype, convrot=False,
                        convrot_groupsize=256, input_act=None)
                    w_ref = (packed.float() * scale).to(torch.bfloat16)
                    y_ref = torch.nn.functional.linear(x, w_ref)
                    n_int8 += 1
                else:
                    layout = get_layout_class("AsymW4A8Int8Layout")
                    params = layout.Params(
                        scale=sd[layer + ".weight_s_rel"].to(dev).view(torch.float8_e4m3fn),
                        s_channel=sd[layer + ".weight_s_channel"].to(dev),
                        codebook=sd[layer + ".weight_codebook"].to(dev),
                        group_size=int(conf["group_size"]),
                        convrot_groupsize=int(conf["convrot_groupsize"]),
                        orig_dtype=torch.bfloat16,
                        orig_shape=(packed.shape[0], packed.shape[1] * 2))
                    qt = QuantizedTensor(packed, "AsymW4A8Int8Layout", params)
                    x = torch.randn(4, packed.shape[1] * 2, dtype=torch.bfloat16, device=dev)
                    y = F.linear(x, qt)
                    y_ref = F.linear(x, qt.dequantize())
                    n_w4a8 += 1
                denom = max(float(y.abs().max()), 1e-6)
                errs.append((y - y_ref).abs().max().item() / denom)
            ok = n_int8 > 0 and n_w4a8 > 0 and max(errs) < 0.1
            check("mixed-checkpoint-cuda-forward", ok,
                  f"{n_int8} INT8 + {n_w4a8} W4A8 layers through the CUDA kernels, "
                  f"max rel err {max(errs):.4f}"
                  if ok else f"n_int8={n_int8} n_w4a8={n_w4a8} errs={errs[:3]}")
        except Exception as e:  # pragma: no cover
            check("mixed-checkpoint-cuda-forward", False, f"{type(e).__name__}: {e}")
    else:
        print("SKIP mixed-checkpoint-cuda-forward (mixed fixture not generated)")

    # ---- check 3 (optional): ComfyUI one-step forward ----
    if args.model:
        r = subprocess.run(  # noqa: S603 (optional user-supplied model path)
            [sys.executable, os.path.join(REPO, "testdata", "comfyui_smoke.py"),
             "--model", args.model] +
            (["--source", args.source] if args.source else []),
            capture_output=True, text=True, timeout=1800)
        ok = r.returncode == 0
        check("comfyui-one-step-forward", ok,
              "checkpoint loaded and one model forward completed"
              if ok else (r.stdout + r.stderr)[-800:])

    failed = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\ncuda-smoke: {len(RESULTS) - failed}/{len(RESULTS)} passed")
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
