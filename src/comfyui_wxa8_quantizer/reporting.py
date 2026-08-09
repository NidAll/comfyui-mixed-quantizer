"""Compression statistics and text/console reports."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import dataclasses
import os
import sys
from comfyui_wxa8_quantizer.constants import COMFYUI_PR, COMFY_KITCHEN_REV, CONVERTER_NAME, CONVERTER_VERSION, FORMAT_INT8, FORMAT_MIXED, FORMAT_MIXED_REVISION, FORMAT_W4A8, FORMAT_W4A8_REVISION, MIXED_FORMATS, get_converter_version
from comfyui_wxa8_quantizer.engine import FORMAT_TO_KITCHEN_LAYOUT
from comfyui_wxa8_quantizer.io import CheckpointInfo, KNOWN_GATE_DIMS, TensorMeta
from comfyui_wxa8_quantizer.planning import CalibrationStats, ConversionPlan, DecisionKind, TensorDecision, TensorMetrics
from comfyui_wxa8_quantizer.policies import DetectionResult
from comfyui_wxa8_quantizer.runtime import EnvironmentInfo
from comfyui_wxa8_quantizer.utils import FLOAT_DTYPES, _peak_rss_bytes, human_bytes, json_dumps
_COMPRESSION_BUCKETS = (
    "quantized", "convrot_rejected", "shape_rejected", "small_tensor",
    "keep_precision", "sensitivity_kept", "policy_kept", "user_forced",
    "user_excluded", "not_in_quantize_set", "unrecognized", "outside_prefix",
)

def compression_stats(info: CheckpointInfo, plan: ConversionPlan,
                     detection: DetectionResult) -> Dict[str, Any]:
    """How much of the family's policy-targeted 2D linear weights actually got
    quantized, bucketed by the reason each layer stayed unquantized.

    policy-targeted = 2D float weights under the unet prefix matched by the
    family quantize patterns, minus user overrides (--include / --exclude /
    --keep-precision) and minus intentional policy keeps outside the quantize
    set (policy_kept bucket). The fraction is bytes, so big FFN layers
    dominate. A low fraction means the checkpoint stays mostly at original
    precision; the bucket counts show whether ConvRot-256, K%16 shape rules,
    small tensors, sensitivity analysis, or user filters caused it.
    """
# SPDX-License-Identifier: Apache-2.0
    prefix = detection.unet_prefix
    if not any(k.startswith(prefix) for k in info.key_set()):
        prefix = ""
    policy = detection.policy
    q_re = policy.quantize_re()
    k_re = policy.keep_re()
    x_re = policy.exclude_re()
    by_name = {d.name: d for d in plan.decisions}

    buckets: Dict[str, Dict[str, Any]] = {
        name: {"layers": 0, "bytes": 0, "k_values": set()}
        for name in _COMPRESSION_BUCKETS
    }

    def add(bucket: str, meta: TensorMeta, decision: Optional[TensorDecision]) -> None:
        buckets[bucket]["layers"] += 1
        buckets[bucket]["bytes"] += meta.nbytes
        if len(meta.shape) == 2:
            buckets[bucket]["k_values"].add(int(meta.shape[1]))

    for meta in info.tensors:
        if len(meta.shape) != 2 or meta.dtype not in FLOAT_DTYPES:
            continue
        rel = meta.name[len(prefix):] if prefix and meta.name.startswith(prefix) else meta.name
        decision = by_name.get(meta.name)
        reason = decision.reason or "" if decision is not None else ""
        if decision is not None and decision.kind == DecisionKind.QUANTIZE:
            add("user_forced" if "user-forced" in reason else "quantized", meta, decision)
            continue
        if not meta.name.startswith(prefix):
            add("outside_prefix", meta, decision)
            continue
        in_qset = bool(q_re.search(rel)) if policy.quantize else True
        if "matched --exclude" in reason or "matched --keep-precision" in reason:
            add("user_excluded", meta, decision)
        elif "convrot_groupsize" in reason:
            add("convrot_rejected", meta, decision)
        elif "not quantizable" in reason:
            add("shape_rejected", meta, decision)
        elif "small tensor" in reason:
            add("small_tensor", meta, decision)
        elif "sensitivity" in reason or "retained at" in reason:
            add("sensitivity_kept", meta, decision)
        elif in_qset:
            add("keep_precision", meta, decision)
        elif "policy keep" in reason or "policy exclude" in reason                 or k_re.search(rel) or x_re.search(rel):
            add("policy_kept", meta, decision)
        elif "not in" in reason and "quantize set" in reason:
            add("not_in_quantize_set", meta, decision)
        else:
            add("unrecognized", meta, decision)

    policy_targeted = {n: buckets[n] for n in
                       ("quantized", "convrot_rejected", "shape_rejected",
                        "small_tensor", "keep_precision", "sensitivity_kept")}
    targeted_bytes = sum(b["bytes"] for b in policy_targeted.values())
    targeted_layers = sum(b["layers"] for b in policy_targeted.values())
    quantized_bytes = buckets["quantized"]["bytes"]
    fraction = (min(1.0, quantized_bytes / targeted_bytes)
                if targeted_bytes else None)
    return {
        "quantized_2d_layers": buckets["quantized"]["layers"],
        "targeted_2d_layers": targeted_layers,
        "quantized_2d_bytes": quantized_bytes,
        "targeted_2d_bytes": targeted_bytes,
        "quantized_fraction": fraction,
        "failing_k_values": sorted(buckets["convrot_rejected"]["k_values"]),
        "buckets": {
            name: {"layers": b["layers"], "bytes": b["bytes"],
                   "k_values": sorted(b["k_values"])}
            for name, b in buckets.items()
        },
    }

def policy_miss_warning(stats: Dict[str, Any],
                         architecture: str) -> Optional[str]:
    """Warn when large 2D float weights under the detected prefix match neither
    quantize nor keep/exclude patterns: the architecture policy may be stale."""
    buckets = stats.get("buckets", {})
    unrec = buckets.get("unrecognized", {})
    not_in_set = buckets.get("not_in_quantize_set", {})
    layers = unrec.get("layers", 0) + not_in_set.get("layers", 0)
    if not layers:
        return None
    bytes_ = unrec.get("bytes", 0) + not_in_set.get("bytes", 0)
    in_prefix_2d = (stats.get("targeted_2d_bytes", 0) + bytes_
                    + buckets.get("outside_prefix", {}).get("bytes", 0))
    if bytes_ <= 0.05 * max(1, in_prefix_2d) or bytes_ < 16 * 1024 * 1024:
        return None
    return (
        "policy may be stale: {} unrecognized 2D linear weights ({} bytes) "
        "under the {} prefix match neither quantize nor keep/exclude patterns; "
        "review the architecture policy"
    ).format(layers, bytes_, architecture)

def low_compression_warning(stats: Dict[str, Any],
                            family: Optional[str] = None) -> Optional[str]:
    """Warning text when most policy-targeted linear bytes stay unquantized.

    The message names the real cause: ConvRot-256 gate victims (with an
    "expected for <family>" phrasing when the failing K values are documented
    for that architecture), K%16 shape failures, or sensitivity/user filters.
    """
    frac = stats.get("quantized_fraction")
    targeted = stats.get("targeted_2d_bytes") or 0
    if frac is None or targeted == 0 or frac >= 0.5:
        return None
    buckets = stats.get("buckets", {})
    conv = buckets.get("convrot_rejected", {})
    shape = buckets.get("shape_rejected", {})
    q = stats["quantized_2d_layers"]
    t = stats["targeted_2d_layers"]
    head = ("low compression: only {:.0f}% of policy-targeted 2D linear bytes "
            "were quantized ({} of {} layers)".format(100 * frac, q, t))
    if conv.get("layers") and conv.get("bytes", 0) >= (shape.get("bytes") or 0):
        ks = ", ".join(str(k) for k in conv.get("k_values", []))
        known = KNOWN_GATE_DIMS.get(family or "", ())
        if known and ks and set(conv.get("k_values", [])) <= set(known):
            return (head + ". Expected for {}: K in {} is incompatible with "
                    "ConvRot-256, so those layers stay at original precision. "
                    "The output remains CUDA-runnable, just larger."
                    .format(family, ks))
        return (head + ". The CUDA ConvRot-256 gate keeps layers with "
                "K % 256 != 0 at original precision (failing K values: {})".format(ks))
    if shape.get("layers"):
        ks = ", ".join(str(k) for k in shape.get("k_values", []))
        return (head + ". Layers with K in {} fail the W4A8 K % 16 / group "
                "divisibility rules and stay at original precision.".format(ks))
    return (head + ". Most of the loss comes from sensitivity analysis or user "
            "filters, not the ConvRot gate; review --sensitivity-threshold and "
            "--keep-precision/--exclude.")

def render_text_report(report: Dict[str, Any]) -> str:
    L: List[str] = []
    a = L.append
    a("=" * 78)
    a(f"{CONVERTER_NAME} {CONVERTER_VERSION} -- conversion report")
    a("=" * 78)
    a(f"format            : {report.get('format')} ({report.get('format_revision')})")
    a(f"architecture      : {report.get('architecture')} (confidence {report.get('detection_confidence')})")
    a(f"unet prefix       : {report.get('unet_prefix')}")
    a(f"input             : {report.get('input_kind')} ({', '.join(report.get('input_files', []))})")
    a(f"output            : {report.get('output_path')} ({human_bytes(report.get('output_bytes', 0))})")
    a(f"elapsed           : {report.get('elapsed_seconds')} s")
    a(f"peak RSS          : {human_bytes(report.get('peak_rss_bytes', 0))}")
    if report.get("warnings"):
        a("warnings:")
        for w in report["warnings"]:
            a(f"  - {w}")
    a("")
    a("-- detection evidence --")
    for e in report.get("detection_evidence", []):
        a(f"  + {e}")
    for e in report.get("detection_hints", []):
        a(f"  ~ {e} (hint)")
    if report.get("competing"):
        a("  competing matches: " + ", ".join(report["competing"]))
    a("")
    a("-- tensor statistics --")
    a(f"  input tensors   : {report.get('n_input_tensors')}")
    a(f"  quantized layers: {report.get('n_quantized')}")
    a(f"  kept tensors    : {report.get('n_kept')}")
    a("")
    a("-- compression --")
    comp = report.get("compression") or {}
    if comp.get("targeted_2d_bytes"):
        frac = comp.get("quantized_fraction")
        a(f"  quantized      : {comp.get('quantized_2d_layers')} / "
          f"{comp.get('targeted_2d_layers')} policy-targeted 2D layers")
        a(f"  byte share     : {100*frac:.1f}% of targeted 2D bytes"
          if frac is not None else "  byte share     : n/a")
        if comp.get("failing_k_values"):
            a("  gate victims   : K in %s not divisible by 256 (ConvRot-256); "
              "kept at original precision" % comp["failing_k_values"])
        gh = report.get("group_histogram") or {}
        a("  convrot groups : " + str(gh.get("convrot_groupsize", {})))
        a("  group sizes    : " + str(gh.get("group_size", {})))
        b = comp.get("buckets", {})
        for name in _COMPRESSION_BUCKETS:
            bl = b.get(name)
            if bl and bl.get("layers"):
                ktxt = ("; K in %s" % bl["k_values"]) if bl.get("k_values") else ""
                a("  %-16s: %d layers, %s%s" % (
                    name, bl["layers"], human_bytes(bl["bytes"]), ktxt))
    else:
        a("  no policy-targeted 2D linear layers")
    mixed = report.get("mixed_plan") or {}
    if mixed:
        a("")
        a("-- mixed precision --")
        a(f"  profile        : {mixed.get('profile')} (layer gate "
          f"{mixed.get('layer_gate')}, global gate {mixed.get('global_gate')})")
        a("  global mean err: "
          + (f"{mixed.get('global_mean_error'):.4f}" if mixed.get("global_mean_error") is not None else "n/a"))
        a(f"  promotions     : {len(mixed.get('promotions') or [])}")
        counts = mixed.get("counts") or {}
        params = mixed.get("layer_params") or {}
        bytes_ = mixed.get("layer_bytes") or {}
        total_p = sum(params.values()) or 1
        for fmt in MIXED_FORMATS:
            if counts.get(fmt):
                a(f"  {fmt:22s}: {counts[fmt]:4d} layers, "
                  f"{100*params.get(fmt,0)/total_p:5.1f}% of quantized params, "
                  f"{human_bytes(bytes_.get(fmt,0))}")
        a(f"  quantized-only payload  : "
          f"{sum(bytes_.values())/total_p:.3f} bytes/parameter")
        if mixed.get("effective_bpp") is not None:
            a(f"  effective targeted      : {mixed['effective_bpp']:.3f} "
              f"bytes/parameter (target {mixed.get('compression_target_bpp')})")
            a(f"  original precision      : "
              f"{100*(mixed.get('original_precision_parameter_fraction') or 0):.1f}% "
              f"of params, "
              f"{100*(mixed.get('original_precision_output_byte_fraction') or 0):.1f}% "
              f"of output bytes (max "
              f"{100*(mixed.get('max_bf16_fraction') or 0):.1f}%)")
        runtime = mixed.get("runtime") or {}
        if runtime.get("formats"):
            a("  runtime target          : " + runtime.get("backend", "?") +
              " [" + ", ".join(
                  f"{f}={s.get('status', s) if isinstance(s, dict) else s}"
                  for f, s in runtime["formats"].items()) + "]")
            if runtime.get("w4a4_effective_activation_bits"):
                a("  w4a4 effective act     : A" +
                  str(runtime["w4a4_effective_activation_bits"]))
        cands = mixed.get("candidates") or {}
        if cands:
            a("")
            a("-- per-layer candidates --")
            for name in sorted(cands):
                parts = []
                for fmt, c in cands[name].items():
                    err = c.get("act_rel_l2")
                    if err is None:
                        err = c.get("weight_rel_l2")
                    err_txt = f"err={err:.4f}" if err is not None else ""
                    if c.get("eligible"):
                        parts.append(f"{fmt}({c.get('bytes')}B {err_txt})")
                    else:
                        parts.append(
                            f"{fmt}(rejected: {c.get('reason', '?')})")
                a(f"  {name}: " + ", ".join(parts))
    a("")
    a("-- quantization --")
    for row in report.get("quantization_rows", []):
        a(f"  {row}")
    a("")
    a("-- calibration --")
    a("  " + report.get("calibration_summary", "none"))
    a("")
    a("-- validation --")
    for c in report.get("validation_checks", []):
        a(f"  [{c['status']:22s}] {c['name']}" + (f" -- {c['detail']}" if c.get("detail") else ""))
    a("")
    a("-- compatibility requirements --")
    a("  " + report.get("compatibility_summary", ""))
    a("")
    a("-- output integrity --")
    a(f"  sha256          : {report.get('output_sha256')}")
    for f, h in (report.get("input_hashes") or {}).items():
        a(f"  input {os.path.basename(f)} : {h}")
    a("=" * 78)
    return "\n".join(L)

def _group_histogram(plan: ConversionPlan) -> Dict[str, Dict[int, int]]:
    """convrot_groupsize / group_size counts over the quantized layers. One-line
    proof in the report that every emitted layer uses the production config."""
    convrot: Dict[int, int] = {}
    group: Dict[int, int] = {}
    for d in plan.quantized_layers():
        convrot[d.convrot_groupsize] = convrot.get(d.convrot_groupsize, 0) + 1
        group[d.group_size] = group.get(d.group_size, 0) + 1
    return {"convrot_groupsize": convrot, "group_size": group}

def build_report(info: CheckpointInfo, plan: ConversionPlan, env: EnvironmentInfo,
                 args: Any, result: DetectionResult, calibration: Optional[CalibrationStats],
                 metrics: Dict[str, TensorMetrics], validation: Dict[str, Any],
                 input_hashes: Dict[str, str], output_sha256: str,
                 elapsed: float, warnings: List[str],
                 quant_rows: List[str]) -> Dict[str, Any]:
    required_layouts = sorted({
        FORMAT_TO_KITCHEN_LAYOUT[d.format]
        for d in plan.quantized_layers() if d.format in MIXED_FORMATS})
    comp = [
        "comfy-kitchen >= %s (PR #90, merged) with %s" % (
            COMFY_KITCHEN_REV,
            ", ".join(required_layouts) or "AsymW4A8Int8Layout"),
        "ComfyUI >= v0.31.0 (PR #%d merged as 344b43989e; older builds need "
        "patches/comfyui_w4a8_loader.patch)" % COMFYUI_PR,
        "CUDA: PyTorch cu130+, SM >= 8.0; ROCm: triton >= 3.7; eager works anywhere",
    ]
    if plan.fmt == FORMAT_MIXED:
        mp = plan.mixed_plan or {}
        runtime = mp.get("runtime") or {}
        comp.append(
            "mixed runtime target: %s (%s)" % (
                runtime.get("backend", "unknown"),
                "; ".join(f"{f}={status}" for f, status in
                          (runtime.get("formats") or {}).items())))
        if mp.get("effective_bpp") is not None:
            comp.append(
                "mixed gates: effective %.3f bytes/param (target %.3f), "
                "%.1f%% original precision (max %.1f%%)" % (
                    mp["effective_bpp"], mp.get("compression_target_bpp", 0),
                    100 * (mp.get("bf16_fraction") or 0),
                    100 * (mp.get("max_bf16_fraction") or 0)))
    return {
        "converter": CONVERTER_NAME, "converter_version": get_converter_version(),
        "format": plan.fmt, "format_revision": (
            FORMAT_MIXED_REVISION if plan.fmt == FORMAT_MIXED
            else FORMAT_W4A8_REVISION),
        "architecture": result.architecture, "detection_confidence": result.confidence,
        "unet_prefix": result.unet_prefix,
        "detection_evidence": result.evidence, "detection_hints": result.hints,
        "competing": result.competing,
        "input_kind": info.kind, "input_files": info.files, "input_bytes": info.total_bytes,
        "output_path": args.output, "output_bytes": plan.total_out_bytes,
        "n_input_tensors": len(info.tensors), "n_quantized": plan.n_quantized,
        "n_kept": plan.n_kept,
        "compression": compression_stats(info, plan, result),
        "group_histogram": _group_histogram(plan),
        "mixed_plan": plan.mixed_plan,
        "quantization_rows": quant_rows,
        "tensor_decisions": [
            {"name": item.name, "decision": item.kind.value,
             "reason": item.reason, "layer": item.layer,
             "group_size": item.group_size if item.kind == DecisionKind.QUANTIZE else None,
             "convrot_groupsize": item.convrot_groupsize
             if item.kind == DecisionKind.QUANTIZE else None}
            for item in plan.decisions
        ],
        "sensitivity_metrics": {
            name: dataclasses.asdict(metric) for name, metric in metrics.items()
        },
        "calibration_summary": (
            f"source={calibration.source}, files={calibration.files}, "
            f"layers={len(calibration.layers)}, provenance={json_dumps(calibration.provenance)}"
            if calibration else "calibration-free (reference format; per-group absmax scales)"),
        "validation_checks": validation.get("checks", []),
        "compatibility_summary": "; ".join(comp),
        "input_hashes": input_hashes, "output_sha256": output_sha256,
        "elapsed_seconds": round(elapsed, 3), "peak_rss_bytes": _peak_rss_bytes(),
        "environment": env.to_dict(), "warnings": warnings,
    }

def _console_color(code: str, text: str) -> str:
    """ANSI color when stdout is a terminal, plain text otherwise."""
    if not sys.stdout.isatty():
        return text
    return f"\x1b[{code}m{text}\x1b[0m"

def print_console_summary(*, out_path: str, input_bytes: int, output_bytes: int,
                          architecture: str, confidence: str, n_quantized: int,
                          n_kept: int, comp: Dict[str, Any], group_size: int,
                          validation: Optional[Dict[str, Any]],
                          elapsed: float, warnings: List[str],
                          report_path: Optional[str] = None,
                          dry_run: bool = False,
                          mixed_plan: Optional[Dict[str, Any]] = None) -> None:
    """Concise, human-friendly terminal summary printed after a conversion
    (or dry run). Plain ASCII when piped; colored when interactive."""
    saved = (1.0 - output_bytes / input_bytes) if input_bytes else 0.0
    saved_txt = f"saved {100*saved:.1f}%" if saved >= 0 else "larger than input"
    saved_color = _console_color("32", saved_txt) if saved >= 0.5 else (
        _console_color("33", saved_txt) if saved >= 0 else saved_txt)
    frac = comp.get("quantized_fraction")
    frac_txt = (f"{100*frac:.0f}%" if frac is not None else "n/a")
    mode = "dry run (nothing written)" if dry_run else "conversion complete"
    title = _console_color("1", f" {mode} ") + f"  elapsed {elapsed:.1f} s"
    n_failed = (validation or {}).get("n_failed", 0)
    if validation:
        v_txt = (f"{validation.get('n_passed', 0)} passed, "
                 f"{validation.get('n_passed_with_warnings', 0)} with warnings, "
                 f"{n_failed} failed, {validation.get('n_skipped', 0)} skipped")
        v_txt = (_console_color("31", v_txt) if n_failed else
                 _console_color("32", v_txt))
    else:
        v_txt = "not run (dry run)"
    lines = []
    a = lines.append
    a("=" * 78)
    a(title)
    a("-" * 78)
    a(f"  input       : {human_bytes(input_bytes):>10s}")
    a(f"  output      : {human_bytes(output_bytes):>10s}   {saved_color}")
    a(f"  architecture: {architecture} (confidence {confidence})")
    if frac is not None:
        a(f"  quantized   : {n_quantized} layers ({frac_txt} of policy-targeted "
          "2D linear bytes); " + f"{n_kept} kept")
    else:
        a(f"  quantized   : {n_quantized} layers; {n_kept} kept")
    if mixed_plan:
        counts = mixed_plan.get("counts") or {}
        dist = ", ".join(
            f"{fmt.split('_')[0].upper() if fmt == FORMAT_INT8 else fmt.split('_')[0].upper()}"
            f"x{counts.get(fmt, 0)}"
            for fmt in MIXED_FORMATS if counts.get(fmt))
        a(f"  format      : {FORMAT_MIXED}  profile={mixed_plan.get('profile')}  "
          f"[{dist}]")
        mean = mixed_plan.get("global_mean_error")
        a(f"  gates       : layer<={mixed_plan.get('layer_gate')}  "
          f"global_mean<={mixed_plan.get('global_gate')}"
          + (f"  (mean {mean:.4f})" if mean is not None else ""))
        if mixed_plan.get("effective_bpp") is not None:
            a(f"  compression : {mixed_plan['effective_bpp']:.3f} bytes/param "
              f"(target {mixed_plan.get('compression_target_bpp')}); "
              f"{100*(mixed_plan.get('original_precision_parameter_fraction') or 0):.1f}% "
              f"params original (max "
              f"{100*(mixed_plan.get('max_bf16_fraction') or 0):.1f}% bytes)")
    else:
        a(f"  format      : {FORMAT_W4A8}  group_size={group_size}  convrot_groupsize=256")
    a(f"  validation  : {v_txt}")
    if report_path:
        a(f"  report      : {report_path}")
    if warnings:
        a("-" * 78)
        wl = _console_color("33", "WARNING") if len(warnings) == 1 else             _console_color("33", f"WARNINGS ({len(warnings)})")
        a(f"  {wl}")
        for w in warnings:
            a(f"    - {w}")
    a("=" * 78)
    print("\n".join(lines))
