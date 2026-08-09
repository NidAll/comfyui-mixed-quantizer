#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
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

Numeric pass/fail (v2): every probe runs the eager backend as the reference
and the target backend on the same fixed-seed inputs, then records the
relative L2 error. A probe passes only when that error is under the bound
documented in the record (REL_L2_BOUND, or REL_L2_BOUND_W4A4_A4 for the
int4-requested W4A4 variants; see the constants below for why the A4 bound
is looser). Per-probe records are keyed by (format, K, convrot_groupsize,
linear_dtype): repeated probes never overwrite each other.

Effective W4A4 activation bits (v2): observed from runtime behavior, never
inferred from the requested linear_dtype. Every W4A4 probe runs BOTH
variants (linear_dtype int4 and int8) on the target backend and compares
each against the eager A4 reference (the eager convrot_w4a4_linear op) and
the eager A8 reference (comfy-kitchen's eager int8_linear with ConvRot on
the unpacked int4 codes; eager itself always executes A4, which is observed
and recorded). The closer reference decides effective_activation_bits; the
`certain` flag says whether the winner was at least 2x closer and within
its bound. On eager the observation is always A4, for both requests.

The schema is comfy-wxa8-runtime-cert/v2 (v1 had one record per format, so
repeated probes silently overwrote each other). The v2 payload keeps the
v1 shape the converter reads (formats keyed by format name with at least
load/forward booleans and the W4A4 effective_activation_bits); the
per-probe records and package binds are additive.

Usage:
    python tools/runtime_certify.py --output cert.json [--device cuda]
    python tools/runtime_certify.py --output cert.json --device cpu

Exit codes: 0 = every probe passed, 1 = one or more probes failed,
2 = could not run (missing comfy-kitchen/torch/GPU or unexpected error).
A failure JSON (with any failing probe records) is written on every exit,
including early exits and exceptions.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCHEMA = "comfy-wxa8-runtime-cert/v2"

FORMATS = ("convrot_w4a4", "asym_w4a8_int8", "int8_tensorwise")

# Fixed seed for every probe: deterministic inputs, same for the eager
# reference and the target backend. Measured on the reference machine
# (RTX 3050, torch 2.13.0+cu130, comfy-kitchen 0.2.28): W4A8 rel_l2
# 0.0025-0.0096, INT8 0.0, W4A4-A8 0.0-0.0095, W4A4-A4 0.038-0.055.
FIXED_SEED = 0

# Numeric pass/fail bound for probe outputs vs the eager reference. 1e-2 is
# loose enough for tiny random probes (worst measured 0.0096) and tight
# enough to reject broken kernels, NaNs, and wrong-scale outputs.
REL_L2_BOUND = 1e-2

# W4A4 int4-requested probes compare against the eager A4 op, which computes
# its matmul in bf16 while the CUDA kernel accumulates exactly. That
# implementation difference alone measures 0.038-0.055 on correct kernels
# (the shared int4-activation quantization error is about 0.037 on these
# probes), so REL_L2_BOUND would falsely fail a correct kernel. 0.1 keeps a
# >= 1.9x margin while still rejecting NaN/garbage/scale errors.
REL_L2_BOUND_W4A4_A4 = 0.1

# Certainty margin for the observed W4A4 activation bits: the winning
# reference must be at least 2x closer than the loser (and within its
# bound) for the observation to count as certain.
CERTAIN_RATIO = 0.5

# (format, N, K, convrot_groupsize, linear_dtype): awkward and
# representative real-model dimensions per format.
PROBES = (
    ("int8_tensorwise", 64, 3360, 256, None),
    ("asym_w4a8_int8", 64, 768, 256, None),
    ("asym_w4a8_int8", 64, 13568, 256, None),
    ("convrot_w4a4", 64, 1152, 16, "int4"),
    ("convrot_w4a4", 64, 1152, 16, "int8"),
    ("convrot_w4a4", 64, 768, 256, "int4"),
)


def _rel_l2(a, b) -> float:
    return float((a - b).norm() / b.norm().clamp(min=1e-8))


def _converter_commit() -> str:
    """Best-effort HEAD of the converter repo; 'unknown' when git is not
    available or the file is outside a checkout."""
    try:
        r = subprocess.run(  # noqa: S603 (explicit local git invocation)
            ["git", "-C", REPO, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10)
        head = r.stdout.strip()
        return head if r.returncode == 0 and head else "unknown"
    except Exception:  # noqa: S110 (best-effort bind)
        return "unknown"


def _package_versions(torch, comfy_kitchen) -> dict:
    out = {
        "torch": torch.__version__,
        "torch_cuda": getattr(torch.version, "cuda", None),
        "torch_hip": getattr(torch.version, "hip", None),
        "python": sys.version.split()[0],
    }
    try:
        out["comfy_kitchen"] = importlib.metadata.version("comfy-kitchen")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        out["comfy_kitchen"] = getattr(comfy_kitchen, "__version__", "unknown")
    for name in ("safetensors", "numpy"):
        try:
            out[name] = importlib.import_module(name).__version__
        except Exception:  # noqa: S110 (informational only)
            out[name] = "unknown"
    return out


def _probe_int8(torch, comfy_kitchen, dev: str, n: int, k: int, cgs: int,
                rec: dict) -> None:
    w = torch.randn(n, k, dtype=torch.bfloat16) * 0.02
    x = torch.randn(8, k, dtype=torch.bfloat16) * 0.1
    q, scale = comfy_kitchen.quantize_int8_rowwise(w)
    scale = scale.reshape(-1, 1)
    y_ref = comfy_kitchen.int8_linear(
        x, q, scale, None, x.dtype, convrot=False,
        convrot_groupsize=cgs, input_act=None)
    y_t = comfy_kitchen.int8_linear(
        x.to(dev), q.to(dev), scale.to(dev), None, x.dtype, convrot=False,
        convrot_groupsize=cgs, input_act=None)
    d = _rel_l2(y_t.cpu(), y_ref)
    rec.update(load=True, forward=d < REL_L2_BOUND, rel_l2=d,
               bound=REL_L2_BOUND,
               reference="eager int8_linear (rowwise int8)",
               effective_activation_bits=8, certain=True)


def _probe_w4a8(torch, comfy_kitchen, dev: str, n: int, k: int,
                rec: dict) -> None:
    from comfy_kitchen.tensor.base import QuantizedTensor, get_layout_class
    w = torch.randn(n, k, dtype=torch.bfloat16) * 0.02
    x = torch.randn(8, k, dtype=torch.bfloat16) * 0.1
    packed, s_rel, s_ch, _corr, cb = comfy_kitchen.quantize_w4a8_int8_weight(
        w, group_size=16, convrot_groupsize=256, symmetric=True,
        scale_dtype=torch.float8_e4m3fn, codebook=True)
    layout = get_layout_class("AsymW4A8Int8Layout")
    params = layout.Params(
        scale=s_rel.view(torch.float8_e4m3fn),
        s_channel=s_ch, codebook=cb, group_size=16,
        convrot_groupsize=256, orig_dtype=torch.bfloat16,
        orig_shape=(n, k))
    qt = QuantizedTensor(packed, "AsymW4A8Int8Layout", params)
    y_ref = torch.nn.functional.linear(x, qt)
    y_t = torch.nn.functional.linear(x.to(dev), qt.to(dev))
    d = _rel_l2(y_t.cpu(), y_ref)
    rec.update(load=True, forward=d < REL_L2_BOUND, rel_l2=d,
               bound=REL_L2_BOUND,
               reference="eager F.linear(x, AsymW4A8Int8Layout "
                         "QuantizedTensor)",
               effective_activation_bits=8, certain=True)


def _probe_w4a4(torch, comfy_kitchen, dev: str, n: int, k: int, cgs: int,
                linear_dtype: str, rec: dict) -> None:
    from comfy_kitchen.backends.eager.convrot_w4a4 import (
        prepare_int4_weight_for_int8_linear)
    w = torch.randn(n, k, dtype=torch.bfloat16) * 0.02
    x = torch.randn(8, k, dtype=torch.bfloat16) * 0.1
    packed, ws = comfy_kitchen.quantize_convrot_w4a4_weight(
        w, convrot_groupsize=cgs, quant_group_size=64)
    # Eager references. The eager convrot op always executes A4 (it ignores
    # linear_dtype, which is itself observed below); the A8 reference is
    # comfy-kitchen's eager int8_linear with ConvRot applied to the unpacked
    # int4 codes, i.e. the same computation the CUDA A8 kernels implement.
    y_ref_a4 = comfy_kitchen.convrot_w4a4_linear(
        x, packed, ws, None, convrot_groupsize=cgs,
        quant_group_size=64, linear_dtype="int4")
    y_ref_a8 = comfy_kitchen.int8_linear(
        x, prepare_int4_weight_for_int8_linear(packed), ws, None, x.dtype,
        convrot=True, convrot_groupsize=cgs)
    variants: dict = {}
    for ld in ("int4", "int8"):
        y_t = comfy_kitchen.convrot_w4a4_linear(
            x.to(dev), packed.to(dev), ws.to(dev), None,
            convrot_groupsize=cgs, quant_group_size=64, linear_dtype=ld)
        d_a4 = _rel_l2(y_t.cpu(), y_ref_a4)
        d_a8 = _rel_l2(y_t.cpu(), y_ref_a8)
        bits = 4 if d_a4 <= d_a8 else 8
        win, lose = (d_a4, d_a8) if bits == 4 else (d_a8, d_a4)
        bound = REL_L2_BOUND_W4A4_A4 if bits == 4 else REL_L2_BOUND
        variants[ld] = {
            "effective_activation_bits": bits,
            "certain": bool(win <= lose * CERTAIN_RATIO and win < bound),
            "forward": bool(win < bound),
            "bound": bound,
            "rel_l2": win,
            "rel_l2_vs_a4_reference": d_a4,
            "rel_l2_vs_a8_reference": d_a8,
        }
    v = variants[linear_dtype]
    rec.update(
        load=True, forward=v["forward"], rel_l2=v["rel_l2"],
        bound=v["bound"], effective_activation_bits=v["effective_activation_bits"],
        certain=v["certain"], variants=variants,
        reference="eager convrot_w4a4_linear (A4) / eager int8_linear "
                  "convrot on unpacked int4 codes (A8)")


def probe(torch, comfy_kitchen, dev: str) -> list[dict]:
    records: list[dict] = []
    for fmt, n, k, cgs, linear_dtype in PROBES:
        rec = {
            "format": fmt, "k": k, "convrot_groupsize": cgs,
            "linear_dtype": linear_dtype, "seed": FIXED_SEED,
            "load": False, "forward": False,
        }
        torch.manual_seed(FIXED_SEED)
        try:
            if fmt == "int8_tensorwise":
                _probe_int8(torch, comfy_kitchen, dev, n, k, cgs, rec)
            elif fmt == "asym_w4a8_int8":
                _probe_w4a8(torch, comfy_kitchen, dev, n, k, rec)
            else:
                assert linear_dtype in ("int4", "int8"), linear_dtype
                _probe_w4a4(torch, comfy_kitchen, dev, n, k, cgs,
                            linear_dtype, rec)
        except Exception as e:  # per-probe failure, keep the other probes
            rec["error"] = f"{type(e).__name__}: {e}"
        records.append(rec)
    return records


def _aggregate(records: list[dict]) -> dict:
    """Format-level view for the converter: load/forward booleans plus the
    observed W4A4 activation bits. The converter reads one bits value per
    format (it plans with a single linear_dtype, default int8), so the W4A4
    value is the observation for the int8-requested probe (matching the
    default request); per-probe records keep every per-request observation.
    Fall back to the least optimistic (minimum) observed bits when no
    int8-requested probe exists."""
    formats: dict = {}
    for fmt in FORMATS:
        probes = [r for r in records if r["format"] == fmt]
        entry = {
            "load": bool(probes) and all(p.get("load") for p in probes),
            "forward": bool(probes) and all(p.get("forward") for p in probes),
            "probes": probes,
        }
        if fmt == "convrot_w4a4":
            int8_probe = next(
                (p for p in probes if p.get("linear_dtype") == "int8"), None)
            if int8_probe is not None:
                v = int8_probe["variants"]["int8"]
                entry["effective_activation_bits"] = v["effective_activation_bits"]
                entry["certain"] = v["certain"]
            else:  # pragma: no cover (probe set always includes int8)
                observed = {
                    p["variants"][p["linear_dtype"]]["effective_activation_bits"]
                    for p in probes if p.get("variants")}
                entry["effective_activation_bits"] = min(observed) if observed else None
                entry["certain"] = False
        else:
            entry["effective_activation_bits"] = 8
            entry["certain"] = True
        formats[fmt] = entry
    return formats


def _write_payload(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _payload(records: list[dict], *, ok: bool, exit_code: int,
             failure: str | None = None, torch=None, comfy_kitchen=None,
             backend: str | None = None, gpu=None, cap=None,
             rocm_arch=None) -> dict:
    payload: dict = {
        "schema": SCHEMA,
        "ok": ok,
        "exit_code": exit_code,
        "probe_bounds": {
            "rel_l2": REL_L2_BOUND,
            "rel_l2_w4a4_a4": REL_L2_BOUND_W4A4_A4,
            "certain_ratio": CERTAIN_RATIO,
        },
        "converter_commit": _converter_commit(),
        "backend": backend,
        "gpu": gpu,
        "cuda_capability": cap,
        "rocm_arch": rocm_arch,
        "formats": _aggregate(records) if records else {},
    }
    if torch is not None:
        payload["torch"] = torch.__version__
    if comfy_kitchen is not None:
        try:
            payload["comfy_kitchen"] = importlib.metadata.version(
                "comfy-kitchen")
        except importlib.metadata.PackageNotFoundError:  # pragma: no cover
            payload["comfy_kitchen"] = getattr(
                comfy_kitchen, "__version__", "unknown")
    if torch is not None and comfy_kitchen is not None:
        payload["package_versions"] = _package_versions(torch, comfy_kitchen)
    if failure is not None:
        payload["failure"] = failure
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, metavar="PATH")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    records: list[dict] = []
    try:
        import torch
        import comfy_kitchen
    except ImportError as e:  # pragma: no cover
        print(f"FAIL: comfy-kitchen/torch not importable ({e})")
        _write_payload(args.output, _payload(
            records, ok=False, exit_code=2,
            failure=f"comfy-kitchen/torch not importable ({e})"))
        return 2
    if args.device == "cuda" and not torch.cuda.is_available():
        print("FAIL: --device cuda but no CUDA GPU")
        _write_payload(args.output, _payload(
            records, ok=False, exit_code=2,
            failure="--device cuda but no CUDA GPU",
            torch=torch, comfy_kitchen=comfy_kitchen))
        return 2

    backend = ("amd" if getattr(torch.version, "hip", None) else "nvidia"
               if args.device == "cuda" else "cpu")
    comfy_kitchen.use_backend("cuda" if args.device == "cuda" else "eager")

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

    try:
        records = probe(torch, comfy_kitchen, args.device)
    except Exception as e:  # pragma: no cover
        print(f"FAIL: probe raised {type(e).__name__}: {e}")
        _write_payload(args.output, _payload(
            records, ok=False, exit_code=2,
            failure=f"probe raised {type(e).__name__}: {e}",
            torch=torch, comfy_kitchen=comfy_kitchen, backend=backend,
            gpu=gpu, cap=cap, rocm_arch=rocm_arch))
        return 2

    formats = _aggregate(records)
    ok = all(v.get("load") and v.get("forward") for v in formats.values())
    payload = _payload(
        records, ok=ok, exit_code=0 if ok else 1,
        failure=None if ok else "one or more probes failed "
                               "(see formats.<fmt>.probes)",
        torch=torch, comfy_kitchen=comfy_kitchen, backend=backend,
        gpu=gpu, cap=cap, rocm_arch=rocm_arch)
    _write_payload(args.output, payload)

    print(f"runtime certificate written to {args.output}")
    for fmt, entry in formats.items():
        bits = entry.get("effective_activation_bits")
        print(f"  {fmt:20s} load={entry.get('load')} "
              f"forward={entry.get('forward')} "
              f"bits={bits} certain={entry.get('certain')} "
              f"({len(entry['probes'])} probe(s))")
        for p in entry["probes"]:
            print(f"      K={p['k']:<6} cgs={p['convrot_groupsize']:<4} "
                  f"ld={str(p.get('linear_dtype')):<5} "
                  f"load={p.get('load')} forward={p.get('forward')} "
                  f"rel_l2={p.get('rel_l2')} "
                  f"eff_bits={p.get('effective_activation_bits')} "
                  f"certain={p.get('certain')}"
                  + (f" error={p['error']}" if p.get("error") else ""))
    if not ok:
        print("FAIL: one or more probes did not pass the rel_l2 bound")
        return 1
    print(f"runtime-certify: {len(formats)}/{len(formats)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
