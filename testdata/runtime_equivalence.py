#!/usr/bin/env python3
"""Runtime equivalence: our calibration simulators vs the real comfy-kitchen.

For every format the planner can select, this script quantizes the same
deterministic weight with OUR converter math, runs the ACTUAL comfy-kitchen
eager operation, runs OUR runtime-output simulation, and requires them to
agree to 1e-4 relative (they measure 0 to 5e-8 on the reference platform;
the bound exists for fp32 accumulation noise only).

Formats and dimensions exercised (the awkward real-model K values):

    W4A4 (A4 mode):  K 320, 640, 1152, 1408, 1920, 3072, 4096
    W4A8:            K 256, 768, 1024, 3072, 4096
    INT8:            K 2520, 3360, 4096

Plus the text-encoder (TE) dimension set added for the TE quantization
support: T5-XXL (4096/10240), Qwen2-0.5B (896/4864), Gemma-2-2B
(2304/9216). K=896 is deliberately absent from the W4A8 list because
896 % 256 != 0 makes W4A8 ineligible (the INT8/W4A4 fallback rows cover it).

The W4A4 A8 (linear_dtype=int8) execution mode is CUDA-only in comfy-kitchen
(the eager backend always executes A4); its simulator agreement is checked in
testdata/cuda_smoke.py against the CUDA kernels instead.

Usage: python testdata/runtime_equivalence.py [--seeds 3] [--n 64] [--rows 16]
Exit codes: 0 = all simulators agree with comfy-kitchen, 2 = failed or the
comfy-kitchen import is unavailable.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} -- {detail}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--rows", type=int, default=16)
    args = ap.parse_args()

    try:
        import comfy_kitchen  # pyright: ignore[reportMissingImports]
        import torch  # pyright: ignore[reportMissingImports]
        from comfy_kitchen.backends.eager.w4a8_int8 import (  # pyright: ignore[reportMissingImports]
            w4a8_int8_linear)
    except ImportError as e:  # pragma: no cover
        print(f"SKIPPED: comfy-kitchen not installed ({e}); "
              "run with the optional requirements")
        return 2

    comfy_kitchen.use_backend("eager")
    spec = importlib.util.spec_from_file_location(
        "wxa8_eq", os.path.join(REPO, "comfyui_wxa8_quantizer.py"))
    assert spec is not None and spec.loader is not None  # pyright: ignore[reportOptionalMemberAccess]
    m = importlib.util.module_from_spec(spec)
    sys.modules["wxa8_eq"] = m
    spec.loader.exec_module(m)

    def disagreement(y1: torch.Tensor, y2: torch.Tensor) -> float:
        return float((y1 - y2).norm() / y2.norm().clamp(min=1e-8))

    w4a4_cases = [(320, 64), (640, 64), (1152, 16), (1408, 64), (1920, 64),
                  (3072, 256), (4096, 256),
                  # TE dims (k, convrot_groupsize)
                  (10240, 256), (896, 64), (4864, 256), (2304, 256), (9216, 256)]
    w4a8_cases = [256, 768, 1024, 3072, 4096,
                  # TE dims (all K%256==0; 896 is correctly absent)
                  10240, 2304, 9216, 4864]
    int8_cases = [2520, 3360, 4096,
                  # TE dims (K=896 exercises the W4A8-ineligible fallback)
                  896, 4864, 2304, 9216, 10240]
    worst = {"w4a4": 0.0, "w4a8": 0.0, "int8": 0.0}

    for seed in range(args.seeds):
        for k, cgs in w4a4_cases:
            torch.manual_seed(seed * 1000 + k)
            w = torch.randn(args.n, k, dtype=torch.float32) * 0.02
            x = torch.randn(args.rows, k, dtype=torch.float32) * 0.1
            packed, scale = m.quantize_w4a4_weight(w, cgs)
            tens = {"": packed, "_scale": scale}
            h = m.build_hadamard(cgs, device=x.device, dtype=torch.float32)
            act_q, act_scale = m._act_quant_int4(m.rotate_activation(x, h, cgs))
            y_sim = m._simulate_quantized_chunk(
                w, tens, m.FORMAT_W4A4, 16, cgs, act_q, act_scale, "int4")
            qw, ws = comfy_kitchen.quantize_convrot_w4a4_weight(
                w, convrot_groupsize=cgs, quant_group_size=64)
            y_ck = comfy_kitchen.convrot_w4a4_linear(
                x, qw, ws, None, convrot_groupsize=cgs,
                quant_group_size=64, linear_dtype="int4")
            d = disagreement(y_sim, y_ck)
            worst["w4a4"] = max(worst["w4a4"], d)
            ok = (y_sim.shape == y_ck.shape and torch.isfinite(y_sim).all()
                  and d < 1e-4)
            check(f"w4a4-a4-K{k}-cgs{cgs}-seed{seed}", ok,
                  f"sim vs eager disagreement {d:.2e}")
        for k in w4a8_cases:
            torch.manual_seed(seed * 1000 + k)
            w = torch.randn(args.n, k, dtype=torch.float32) * 0.02
            x = torch.randn(args.rows, k, dtype=torch.float32) * 0.1
            packed, s_rel, s_ch, corr, cb = m.quantize_w4a8_weight(
                w, group_size=16, convrot_groupsize=256,
                symmetric=True, scale_dtype=torch.float8_e4m3fn, codebook=True)
            assert corr is None
            tens = {"": packed, "_s_rel": s_rel, "_s_channel": s_ch,
                    "_codebook": cb}
            h = m.build_hadamard(256, device=x.device, dtype=torch.float32)
            act_q, act_scale = m._act_quant_int8(m.rotate_activation(x, h, 256))
            y_sim = m._simulate_quantized_chunk(
                w, tens, m.FORMAT_W4A8, 16, 256, act_q, act_scale, "int8")
            y_ck = w4a8_int8_linear(
                x, packed, s_rel, s_ch, codebook=cb, group_size=16,
                convrot_groupsize=256, out_dtype=torch.float32)
            d = disagreement(y_sim, y_ck)
            worst["w4a8"] = max(worst["w4a8"], d)
            ok = (y_sim.shape == y_ck.shape and torch.isfinite(y_sim).all()
                  and d < 1e-4)
            check(f"w4a8-K{k}-seed{seed}", ok,
                  f"sim vs eager disagreement {d:.2e}")
        for k in int8_cases:
            torch.manual_seed(seed * 1000 + k)
            w = torch.randn(args.n, k, dtype=torch.float32) * 0.02
            x = torch.randn(args.rows, k, dtype=torch.float32) * 0.1
            q, scale = m.quantize_int8_tensorwise_weight(w)
            tens = {"": q, "_scale": scale}
            act_q, act_scale = m._act_quant_int8(x)
            y_sim = m._simulate_quantized_chunk(
                w, tens, m.FORMAT_INT8, 16, 256, act_q, act_scale, "int8")
            y_ck = comfy_kitchen.int8_linear(
                x, q, scale, None, x.dtype, convrot=False,
                convrot_groupsize=256, input_act=None)
            d = disagreement(y_sim, y_ck)
            worst["int8"] = max(worst["int8"], d)
            ok = (y_sim.shape == y_ck.shape and torch.isfinite(y_sim).all()
                  and d < 1e-4)
            check(f"int8-K{k}-seed{seed}", ok,
                  f"sim vs eager disagreement {d:.2e}")

    failed = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\nruntime-equivalence: worst disagreements "
          f"w4a4={worst['w4a4']:.2e} w4a8={worst['w4a8']:.2e} "
          f"int8={worst['int8']:.2e}; {len(RESULTS) - failed}/{len(RESULTS)} passed")
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
