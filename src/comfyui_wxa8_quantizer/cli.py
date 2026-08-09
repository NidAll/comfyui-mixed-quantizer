"""Argument parsing, main entry point and report writing."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from pathlib import Path
import argparse
import contextlib
import dataclasses
import json
import math
import os
import tempfile
import time
import torch
from comfyui_wxa8_quantizer.constants import (
    FORMAT_INT8, FORMAT_MIXED, FORMAT_W4A4, FORMAT_W4A8, LAYER_CONF_KEY,
    METADATA_KEY_EXT, METADATA_KEY_QUANT, MIXED_FORMATS, MIXED_PROFILES,
    get_converter_version,
)
from comfyui_wxa8_quantizer.engine import ConversionEngine, _check_runtime_compatibility, hash_checkpoint_files
from comfyui_wxa8_quantizer.errors import InputError, OutputError, PolicyError, RuntimeCompatibilityError, UnknownArchitectureError, UnsupportedArchitectureError, UsageError
from comfyui_wxa8_quantizer.io import CheckpointReader, RawSafetensorsFile, _validate_destination_paths, discover_checkpoint, republish_with_metadata
from comfyui_wxa8_quantizer.logging_utils import JsonLogHandler, log, setup_logging
from comfyui_wxa8_quantizer.metadata import build_extension_metadata, build_quant_metadata
from comfyui_wxa8_quantizer.planner import MixedPlanner
from comfyui_wxa8_quantizer.planning import ConversionPlan, DecisionKind, SensitivityAnalyzer, build_output_entries, classify_tensors, load_calibration
from comfyui_wxa8_quantizer.policies import DetectionResult, detect_architecture, family_names, get_family, unet_prefix_from_keys
from comfyui_wxa8_quantizer.quantize import _quant_work_bytes, apply_sensitivity_prepass
from comfyui_wxa8_quantizer.reporting import build_report, compression_stats, low_compression_warning, policy_miss_warning, print_console_summary, render_text_report
from comfyui_wxa8_quantizer.runtime import _check_runtime_certificate, inspect_environment, load_runtime_certificate, runtime_capabilities_for
from comfyui_wxa8_quantizer.selftests import run_self_tests
from comfyui_wxa8_quantizer.utils import _atomic_write_json, _fsync_parent, _remove_temp_path, human_bytes, json_dumps, parse_size, sha256_file, sha256_safetensors_payload
from comfyui_wxa8_quantizer.validation import Validator, _refresh_validation_summary, plan_from_output, verify_output
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="comfyui_wxa8_quantizer.py",
        description=(
            "Standalone W4A8 checkpoint converter for ComfyUI-compatible generative "
            "models. See the module docstring for the verified format specification "
            "and exact source revisions."),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version",
                   version=f"%(prog)s {get_converter_version()}")
    p.add_argument("model", nargs="?", metavar="ORIGINAL_MODEL",
                   help="path to the original checkpoint: a .safetensors file, a "
                        "sharded safetensors directory, an HF-style model directory, "
                        "or (with --trust-pickle) a torch pickle checkpoint")
    p.add_argument("--output", metavar="PATH", default=None,
                   help="output checkpoint path (required for conversion)")
    p.add_argument("--format", choices=["w4a8", "mixed"], default="w4a8",
                   help="quantization format: w4a8 (single-format reference "
                        "path) or mixed (experimental per-layer optimizer "
                        "over convrot_w4a4 / asym_w4a8_int8 / int8_tensorwise)")
    p.add_argument("--profile", choices=list(MIXED_PROFILES), default="auto",
                   help="mixed-mode profile: auto detects the runtime (GPU -> "
                        "balanced, CPU -> conservative); balanced, conservative "
                        "and size-first set the quality/compression tradeoff")
    p.add_argument("--target-runtime", choices=["auto", "nvidia", "amd", "cpu"],
                   default="auto",
                   help="target inference runtime used for format eligibility; "
                        "auto probes torch (CUDA/ROCm/CPU)")
    p.add_argument("--runtime-certificate", metavar="PATH", default=None,
                   help="mixed mode: JSON certificate produced by "
                        "tools/runtime_certify.py on the target inference "
                        "machine; overrides static W4A4 dispatch guesses with "
                        "observed behavior")
    p.add_argument("--require-runtime-certificate", action="store_true",
                   help="mixed mode: refuse conversion unless every selected "
                        "format is runtime-certified on the target machine")
    p.add_argument("--strip-gpu-identity", action="store_true",
                   help="omit GPU identity (device name, compute capability, "
                        "ROCm architecture) from the checkpoint extension "
                        "metadata, for publishing converted checkpoints "
                        "without revealing the target or conversion machine")
    p.add_argument("--quality-gate", type=float, default=None, metavar="F",
                   help="mixed mode: per-layer error gate override (relL2 "
                        "fraction; default from the profile)")
    p.add_argument("--global-error-gate", type=float, default=None, metavar="F",
                   help="mixed mode: global mean-error gate override; the "
                        "planner promotes layers until this passes")
    p.add_argument("--max-linear-bytes-per-param", type=float, default=None,
                   metavar="F",
                   help="mixed mode: hard compression target in bytes/parameter "
                        "for the targeted linear payload (overrides the profile; "
                        "conversion fails when exceeded)")
    p.add_argument("--max-bf16-fraction", "--max-original-byte-fraction",
                   type=float, default=None, dest="max_bf16_fraction",
                   metavar="F",
                   help="mixed mode: hard limit on the fraction of final "
                        "targeted serialized bytes occupied by "
                        "original-precision tensors (legacy option name "
                        "--max-bf16-fraction; the alias "
                        "--max-original-byte-fraction is preferred)")
    p.add_argument("--w4a4-linear-dtype", choices=["int4", "int8"], default="int8",
                   help="convrot_w4a4 execution variant: int4 (true W4A4) or "
                        "int8 (4-bit weights through the int8 path; the "
                        "default). Execution property only, never a quality "
                        "fallback")
    p.add_argument("--disable-w4a4", action="store_true",
                   help="mixed mode: never select convrot_w4a4 layers")
    p.add_argument("--disable-w4a8", action="store_true",
                   help="mixed mode: never select asym_w4a8_int8 layers")
    p.add_argument("--disable-int8", action="store_true",
                   help="mixed mode: never select int8_tensorwise layers")
    p.add_argument("--require-calibration", action="store_true",
                   help="mixed mode: refuse planning without calibration "
                        "activations (--calibration-source)")
    p.add_argument("--architecture", default="auto",
                   help="auto or an architecture name from --list-architectures")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "rocm"], default="auto",
                   help="device for quantization compute (default auto -> cpu for "
                        "determinism and bounded memory; cuda offloads per tensor)")
    p.add_argument("--compute-dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto",
                   help="precision of the quantization math (default fp32, matching "
                        "the reference implementation)")
    p.add_argument("--output-dtype", choices=["auto", "fp16", "bf16"], default="auto",
                   help="cast passthrough (non-quantized) float tensors to this "
                        "dtype; auto keeps the original dtype")
    p.add_argument("--group-size", type=int, default=None,
                   help="quantization group size; only 16 is supported (the "
                        "validated W4A8 configuration with convrot 256)")
    p.add_argument("--min-quantized-byte-fraction", type=float, default=None,
                   metavar="F",
                   help="abort when the quantized share of policy-targeted 2D "
                        "linear bytes is below F (0..1); used with "
                        "--fail-on-low-compression or alone as the threshold")
    p.add_argument("--fail-on-low-compression", action="store_true",
                   help="refuse to convert when the quantized byte fraction is "
                        "below 0.10 (or --min-quantized-byte-fraction)")
    p.add_argument("--calibration-source", metavar="PATH", default=None,
                   help="local calibration data (.npz/.pt/.npy files or a directory; "
                        "arrays named exactly like the layer keys, shape [S, K])")
    p.add_argument("--calibration-samples", type=int, default=None,
                   help="limit calibration rows used per layer")
    p.add_argument("--calibration-cache", metavar="PATH", default=None,
                   help="read/write a compressed cache of the activation rows")
    p.add_argument("--seed", type=int, default=0, help="reproducibility seed")
    p.add_argument("--include", action="append", default=[],
                   metavar="PATTERN", help="regex; select matching eligible layer weights "
                        "for quantization (shape/dtype safety gates still apply)")
    p.add_argument("--exclude", action="append", default=[],
                   metavar="PATTERN", help="regex; never quantize matching tensors")
    p.add_argument("--keep-precision", action="append", default=[],
                   metavar="PATTERN", help="regex; keep matching weights at original precision")
    p.add_argument("--sensitivity-threshold", type=float, default=None,
                   help="keep layers at original precision when their (activation-aware "
                        "if calibration given, else weight-only) error exceeds this")
    p.add_argument("--error-threshold", type=float, default=None,
                   help="hard reconstruction relL2 fallback during a calibration/"
                        "sensitivity prepass (default: architecture policy bound)")
    p.add_argument("--max-memory", default="2G", metavar="SIZE",
                   help="per-tensor working-memory budget (e.g. 512M, 2G); larger "
                        "tensors are quantized in chunks")
    p.add_argument("--streaming", action="store_true", default=True,
                   help="stream the conversion with bounded memory (always enabled)")
    p.add_argument("--resume", action="store_true",
                   help="resume an interrupted conversion from its state file")
    p.add_argument("--overwrite", action="store_true",
                   help="allow replacing an existing output file")
    p.add_argument("--dry-run", action="store_true",
                   help="detect, plan and report without writing the output")
    p.add_argument("--inspect", action="store_true",
                   help="inspect the input checkpoint and exit")
    p.add_argument("--list-architectures", action="store_true",
                   help="list the embedded architecture registry and exit")
    p.add_argument("--validate", action="store_true",
                   help="run full standalone validation after conversion (all layers, "
                        "output hash, optional runtime compatibility probe)")
    p.add_argument("--validation-only", action="store_true",
                   help="validate an existing output checkpoint (with --model and --output)")
    p.add_argument("--verify-output", metavar="PATH", default=None,
                   help="source-free verification of an existing output "
                        "checkpoint: structural, metadata, packing and "
                        "payload-hash checks with no original model required")
    p.add_argument("--metadata-only", action="store_true",
                   help="generate the metadata and report only; do not write the model")
    p.add_argument("--report", metavar="PATH", default=None,
                   help="write the human-readable report to PATH")
    p.add_argument("--log-level", default="info",
                   choices=["debug", "info", "warning", "error"])
    p.add_argument("--json-log", metavar="PATH", default=None,
                   help="also emit structured JSON log lines to PATH")
    p.add_argument("--trust-pickle", action="store_true",
                   help="allow deserializing pickle-based checkpoints (unsafe for "
                        "untrusted files)")
    p.add_argument("--yes", action="store_true", help="assume yes for confirmations")
    p.add_argument("--self-test", action="store_true",
                   help="run the embedded engineering self-tests and exit")
    return p

def _write_reports(args: Any, report: Dict[str, Any]) -> None:
    text = render_text_report(report)
    print(text)
    if args.report:
        report_path = os.path.abspath(args.report)
        parent = os.path.dirname(report_path) or "."
        os.makedirs(parent, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{Path(report_path).name}.",
                                   suffix=".tmp", dir=parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, report_path)
            _fsync_parent(report_path)
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        _atomic_write_json(args.report + ".json", report, indent=1)
        log().info("reports written to %s and %s.json", args.report, args.report)

def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    operation_modes = {
        "--inspect": args.inspect,
        "--validation-only": args.validation_only,
        "--verify-output": args.verify_output is not None,
        "--metadata-only": args.metadata_only,
        "--dry-run": args.dry_run,
    }
    selected_modes = [name for name, selected in operation_modes.items() if selected]
    if len(selected_modes) > 1:
        raise UsageError(
            "operation modes are mutually exclusive: " + ", ".join(selected_modes))
    if args.list_architectures and args.self_test:
        raise UsageError("--list-architectures and --self-test are mutually exclusive")
    if (args.list_architectures or args.self_test) and selected_modes:
        special = "--list-architectures" if args.list_architectures else "--self-test"
        raise UsageError(
            f"{special} cannot be combined with " + ", ".join(selected_modes))
    if args.resume and args.overwrite:
        raise UsageError("--resume and --overwrite are mutually exclusive")
    if args.group_size is not None and args.group_size != 16:
        raise UsageError(
            "--group-size must be 16: only the validated W4A8 configuration "
            "(group_size=16, convrot_groupsize=256) is supported by the "
            "production runtime")
    setup_logging(args.log_level,
                  args.json_log if (args.list_architectures or args.self_test) else None)
    env = inspect_environment()

    if args.list_architectures:
        print(f"{'family':18s} {'runtime':12s} classes")
        print("-" * 100)
        for name in family_names():
            pol = get_family(name)
            print(f"{name:18s} {pol.runtime_status:12s} {', '.join(pol.comfyui_classes)}")
        print()
        print("W4A8 = reference 'asym_w4a8_int8' format (comfy-kitchen PR #90, "
              "ComfyUI PR #15308).")
        return 0

    if args.self_test:
        return run_self_tests()

    if args.verify_output is not None:
        if not os.path.exists(args.verify_output):
            parser.error(f"--verify-output file not found: {args.verify_output}")
        summary = verify_output(args.verify_output)
        print("=" * 70)
        print("output checkpoint verification (source-free)")
        print("=" * 70)
        print(f"tensors         : {summary['n_tensors']} "
              f"({human_bytes(summary['total_bytes'])})")
        print(f"quantized layers: {summary['quantized_layers']} "
              f"{summary['formats'] or ''}")
        print(f"payload sha256  : {summary['output_sha256']}")
        print("-" * 70)
        for c in summary["checks"]:
            print(f"[{'PASS' if c['ok'] else 'FAIL'}] {c['name']} -- {c['detail']}")
        print(f"verify-output: {summary['n_failed']} failure(s)")
        return 0 if summary["ok"] else 1

    if args.model is None and args.verify_output is None:
        parser.error("ORIGINAL_MODEL is required")

    t_start = time.time()
    warnings: List[str] = []
    if isinstance(args.max_memory, str):
        args.max_memory = parse_size(args.max_memory)
    if args.max_memory <= 0:
        raise UsageError("--max-memory must be positive")
    if args.group_size is not None and args.group_size <= 0:
        raise UsageError("--group-size must be positive")
    if args.calibration_samples is not None and args.calibration_samples <= 0:
        raise UsageError("--calibration-samples must be positive")
    for option, value in (("--sensitivity-threshold", args.sensitivity_threshold),
                          ("--error-threshold", args.error_threshold)):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise UsageError(f"{option} must be a finite non-negative number")
    info = discover_checkpoint(args.model, trust_pickle=args.trust_pickle)
    _validate_destination_paths(info, args)
    if args.json_log:
        log().addHandler(JsonLogHandler(args.json_log))
    if info.kind == "pickle":
        warnings.append("pickle input loaded fully into RAM (streaming not possible); "
                        "only convert checkpoints you trust")

    if info.is_quantized_input and not (args.inspect or args.validation_only):
        raise InputError(
            "input checkpoint already contains quantization markers "
            f"('{METADATA_KEY_QUANT}' metadata or '{LAYER_CONF_KEY}' tensors); "
            "refusing to re-quantize. Use --inspect to review it.")

    def _shape_lookup(name: str) -> Optional[Tuple[int, ...]]:
        m = info.by_name(name)
        return tuple(m.shape) if m is not None else None

    try:
        detection = detect_architecture(
            info,
            override=None if args.architecture in (None, "auto") else args.architecture,
            shape_lookup=_shape_lookup)
        warnings.extend(detection.warnings)
    except UnknownArchitectureError:
        if args.inspect:
            # inspection must still work for unknown checkpoints
            detection = DetectionResult(
                architecture="unknown", confidence="none",
                policy=get_family("sd15"), unet_prefix=unet_prefix_from_keys(info.key_set()),
                warnings=["architecture not identified (inspection mode)"])
        else:
            raise

    if args.inspect:
        print("=" * 70)
        print("input checkpoint inspection")
        print("=" * 70)
        print(f"kind            : {info.kind}")
        print(f"files           : {info.files}")
        print(f"tensors         : {len(info.tensors)}  ({human_bytes(info.total_bytes)})")
        print(f"metadata keys   : {sorted(info.metadata.keys())}")
        print(f"config.json     : {'present' if info.config else 'absent'}")
        print(f"quantized input : {info.is_quantized_input}")
        print(f"detected arch   : {detection.architecture} (confidence {detection.confidence})")
        print(f"unet prefix     : {detection.unet_prefix!r}")
        print("evidence        : " + "; ".join(detection.evidence or ["-"]))
        print("hints           : " + "; ".join(detection.hints or ["-"]))
        print("competing       : " + "; ".join(detection.competing or ["-"]))
        print("policy          : " + detection.policy.family)
        print()
        print("first 40 tensors:")
        for t in info.tensors[:40]:
            print(f"  {t.name:70s} {t.dtype} {tuple(t.shape)}")
        if len(info.tensors) > 40:
            print(f"  ... {len(info.tensors) - 40} more")
        return 0

    if detection.policy.runtime_status == "unsupported" and args.architecture in (None, "auto"):
        raise UnsupportedArchitectureError(
            f"architecture {detection.architecture!r} has no ComfyUI quantized-loading "
            "path; conversion would not be consumable. If you still want to convert it "
            "for research, pass --architecture " + detection.architecture + " explicitly.")

    if args.validation_only:
        if not args.output or not os.path.exists(args.output):
            parser.error("--validation-only requires an existing --output file")
        if args.format == "w4a8":
            fmt = FORMAT_W4A8
        elif args.format == "mixed":
            fmt = FORMAT_MIXED
        else:
            parser.error(f"--format must be w4a8 or mixed, got {args.format!r}")
        plan = plan_from_output(args.output, detection, fmt, info)
        validator = Validator(info, plan, args.output, args, env)
        validation_input_hashes = hash_checkpoint_files(info)
        with CheckpointReader(info) as validation_reader:
            summary = validator.run(reader=validation_reader,
                                    input_hashes=validation_input_hashes)
        quant_rows = []
        for decision in plan.quantized_layers():
            tensor = info.by_name(decision.name)
            shape = tuple(tensor.shape) if tensor is not None else "missing"
            quant_rows.append(
                f"{decision.layer}: {shape} gs={decision.group_size} "
                f"cgs={decision.convrot_groupsize} "
                f"fmt={decision.format}")
        report = build_report(info, plan, env, args, detection, None, {},
                              summary, validation_input_hashes,
                              summary.get("output_sha256", ""),
                              time.time() - t_start, warnings, quant_rows)
        _write_reports(args, report)
        return 0 if summary["n_failed"] == 0 else 2

    if args.format == "w4a8":
        fmt = FORMAT_W4A8
    elif args.format == "mixed":
        fmt = FORMAT_MIXED
    else:
        parser.error(f"--format must be w4a8 or mixed, got {args.format!r}")

    # ---- mixed-mode runtime + profile resolution (auto) ----
    # Auto detects the GPU (NVIDIA CUDA, AMD ROCm, or CPU) and picks the
    # balanced profile on an accelerator, conservative on CPU.  The detected
    # architecture is already known here; a calibration-free balanced run is
    # allowed but reported as weight-gated rather than activation-gated.
    effective_runtime = "auto"
    if args.target_runtime != "auto":
        effective_runtime = args.target_runtime
    elif torch.cuda.is_available():
        if getattr(torch.version, "hip", None) is not None:
            effective_runtime = "amd"
        else:
            effective_runtime = "nvidia"
    else:
        effective_runtime = "cpu"
    if fmt == FORMAT_MIXED:
        profile_name = args.profile
        if profile_name == "auto":
            profile_name = "balanced" if effective_runtime != "cpu" else "conservative"
            warnings.append(
                f"auto profile: {profile_name} for target runtime "
                f"{effective_runtime} (set --profile to override)")
        if args.require_calibration and not args.calibration_source:
            parser.error("--require-calibration needs --calibration-source")
        disabled_formats = []
        if args.disable_w4a4:
            disabled_formats.append(FORMAT_W4A4)
        if args.disable_w4a8:
            disabled_formats.append(FORMAT_W4A8)
        if args.disable_int8:
            disabled_formats.append(FORMAT_INT8)
        if len(disabled_formats) == 3:
            parser.error("at least one quantized format must stay enabled in mixed mode")
        if effective_runtime == "cpu":
            warnings.append(
                "W4A4 linear_dtype note: the comfy-kitchen eager backend "
                "accepts int4/int8 but always executes the int4 activation "
                "path; linear_dtype=int8 only changes the CUDA kernels")
        args._mixed_profile_name = profile_name
        args._mixed_disabled_formats = disabled_formats
        args._effective_runtime = effective_runtime

    if args.output is None:
        parser.error("--output is required")

    # ---- output path safety ----
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.isdir(out_path):
        raise OutputError(f"output path is a directory: {out_path}")
    for f in info.files:
        same_file = os.path.abspath(f) == out_path
        if os.path.exists(out_path):
            with contextlib.suppress(OSError):
                same_file = same_file or os.path.samefile(f, out_path)
        if same_file:
            raise OutputError("output path must not be the same as an input file")
    if os.path.exists(out_path) and not args.overwrite and not args.resume:
        raise OutputError(f"output already exists: {out_path} (use --overwrite)")
    tmp_path = out_path + ".tmp"
    staged_path = out_path + ".staged"
    validation_path = out_path + ".validation"
    state_path = out_path + ".state.json"
    if os.path.exists(tmp_path) and not args.overwrite and not args.resume:
        raise OutputError(f"temp output already exists: {tmp_path} (use --overwrite or --resume)")
    for label, internal_path in (("staged output", staged_path),
                                 ("validation output", validation_path)):
        if not os.path.exists(internal_path):
            continue
        if args.overwrite:
            _remove_temp_path(internal_path)
        elif args.resume:
            # A validation copy is never resume authority.  A staged output is
            # trusted only together with its checksummed state file.
            if internal_path == validation_path or not os.path.exists(state_path):
                _remove_temp_path(internal_path)
        else:
            raise OutputError(
                f"{label} already exists: {internal_path} "
                "(use --overwrite or --resume)")

    # ---- planning ----
    compute_dtype = {"auto": torch.float32, "fp32": torch.float32,
                     "fp16": torch.float16, "bf16": torch.bfloat16}[args.compute_dtype]
    args._compute_dtype_tensor = compute_dtype
    out_dtype = {"auto": None, "fp16": torch.float16, "bf16": torch.bfloat16}[args.output_dtype]
    decisions = classify_tensors(info, detection, fmt, args.group_size,
                                 args.include, args.exclude, args.keep_precision,
                                 out_dtype, None)

    effective_device = torch.device("cpu")
    effective_backend = "cpu"
    if args.device == "cuda":
        if torch.cuda.is_available() and getattr(torch.version, "hip", None) is None:
            effective_device = torch.device("cuda")
            effective_backend = "cuda"
        else:
            warnings.append("--device cuda requested but a CUDA backend is "
                            "unavailable; CPU used")
    elif args.device == "rocm":
        if torch.cuda.is_available() and getattr(torch.version, "hip", None) is not None:
            effective_device = torch.device("cuda")
            effective_backend = "rocm"
        else:
            warnings.append("--device rocm requested but a ROCm backend is "
                            "unavailable; CPU used")

    # Sensitivity / mixed planning is a planning operation.  It must be
    # complete before output shapes and offsets are frozen.
    needs_sensitivity_prepass = (
        fmt == FORMAT_W4A8
        and (args.sensitivity_threshold is not None
             or args.error_threshold is not None
             or args.calibration_source is not None)
    )
    prepass_source_hashes = (
        hash_checkpoint_files(info, refresh=True)
        if needs_sensitivity_prepass else None
    )
    calibration = None
    if args.calibration_source:
        calibration = load_calibration(args.calibration_source, info,
                                       args.calibration_samples, args.calibration_cache,
                                       args.max_memory)
    sensitivity = None
    mixed_planner = None
    if fmt == FORMAT_MIXED:
        profile_name = getattr(args, "_mixed_profile_name", "balanced")
        layer_gate = args.quality_gate
        if layer_gate is None and args.error_threshold is not None:
            layer_gate = args.error_threshold
        runtime_caps = runtime_capabilities_for(effective_runtime, env)
        certificate = None
        if args.runtime_certificate:
            certificate = load_runtime_certificate(args.runtime_certificate)
            warnings.append(
                f"runtime certificate loaded: backend={certificate.backend}, "
                f"gpu={certificate.gpu or 'unknown'}, formats="
                f"{sorted(certificate.formats)}")
        mixed_planner = MixedPlanner(
            profile_name, calibration, args.max_memory, effective_device,
            compute_dtype,
            disabled_formats=getattr(args, "_mixed_disabled_formats", ()),
            layer_gate=layer_gate,
            global_gate=args.global_error_gate,
            runtime=runtime_caps,
            compression_target_bpp=args.max_linear_bytes_per_param,
            max_bf16_fraction=args.max_bf16_fraction,
            linear_dtype=args.w4a4_linear_dtype,
            runtime_certificate=certificate)
        if args.require_runtime_certificate and certificate is None:
            raise RuntimeCompatibilityError(
                "--require-runtime-certificate needs --runtime-certificate "
                "(produce it with tools/runtime_certify.py on the target "
                "inference machine)")
        mixed_plan = mixed_planner.plan(info, decisions)
        args._runtime_gpu_name = runtime_caps.gpu_name
        _check_runtime_compatibility(env, mixed_planner, decisions, warnings)
        if certificate is not None:
            required = sorted({
                d.format for d in decisions
                if d.kind == DecisionKind.QUANTIZE
                and d.format in MIXED_FORMATS})
            _check_runtime_certificate(certificate, runtime_caps, required)
            # mark only the formats the certificate actually exercised
            cap_map = {FORMAT_W4A4: ("w4a4", runtime_caps.w4a4),
                       FORMAT_W4A8: ("w4a8", runtime_caps.w4a8),
                       FORMAT_INT8: ("int8", runtime_caps.int8)}
            certed = {
                field: dataclasses.replace(cap, certified=True)
                for fmt, (field, cap) in cap_map.items()
                if fmt in certificate.formats}
            runtime_caps = dataclasses.replace(
                runtime_caps, runtime_certified=True, **certed)
            mixed_planner.runtime = runtime_caps
        if not calibration:
            warnings.append(
                "mixed planner: no calibration activations, quality gates use "
                "weight-only reconstruction error (pass --calibration-source "
                "for activation-aware gates)")
        if mixed_plan["promotions"]:
            warnings.append(
                f"mixed planner: global gate required {len(mixed_plan['promotions'])} "
                "promotion(s): " + "; ".join(mixed_plan["promotions"][:3]) +
                (" ..." if len(mixed_plan["promotions"]) > 3 else ""))
        if mixed_plan["kept"]:
            warnings.append(
                f"mixed planner kept {mixed_plan['kept']} layer(s) at original "
                "precision (no candidate within the quality gate)")
        if profile_name != "balanced" and args.quality_gate is None \
                and args.global_error_gate is None:
            warnings.append(
                f"mixed profile {profile_name}: layer gate "
                f"{mixed_planner.layer_gate}, global gate "
                f"{mixed_planner.global_gate} (override with --quality-gate / "
                "--global-error-gate)")
    elif needs_sensitivity_prepass:
        sensitivity = SensitivityAnalyzer(
            args.sensitivity_threshold,
            args.error_threshold if args.error_threshold is not None
            else detection.policy.max_rel_l2,
            calibration)
        apply_sensitivity_prepass(info, decisions, sensitivity, args.max_memory,
                                  effective_device, compute_dtype)
        kept_by_sensitivity = sum(1 for m in sensitivity.results.values() if m.kept)
        if kept_by_sensitivity:
            warnings.append(
                f"sensitivity prepass retained {kept_by_sensitivity} layer(s) at "
                "original precision")
        post_prepass_hashes = hash_checkpoint_files(info, refresh=True)
        if post_prepass_hashes != prepass_source_hashes:
            raise InputError(
                "one or more source files changed during sensitivity planning; "
                "refusing to build an output inventory from mixed data")

    plan = ConversionPlan(fmt=fmt, detection=detection, decisions=decisions,
                          metadata_quant={}, metadata_ext={}, output_entries=[])
    plan.device = effective_backend
    if mixed_planner is not None:
        plan.mixed_plan = mixed_planner.summary
    entries, total_out = build_output_entries(info, decisions, fmt, out_dtype)
    plan.output_entries = entries
    plan.total_out_bytes = total_out
    plan.n_quantized = len(plan.quantized_layers())
    plan.n_kept = len(decisions) - plan.n_quantized
    plan.chunked_layers = {
        d.name for d in plan.quantized_layers()
        if _quant_work_bytes(info.by_name(d.name)) > args.max_memory
    }
    plan.metadata_quant = build_quant_metadata(info, plan)

    if plan.n_quantized == 0:
        raise PolicyError(
            "no tensors selected for quantization under the "
            f"{detection.architecture!r} policy after sensitivity analysis "
            "(adjust thresholds or use --include to force layers)")

    quant_rows = []
    for d in plan.quantized_layers():
        m = info.by_name(d.name)
        quant_rows.append(
            f"{d.layer}: {tuple(m.shape)} gs={d.group_size} "
            f"cgs={d.convrot_groupsize} fmt={d.format}")

    comp_stats = compression_stats(info, plan, detection)
    effective_group_size = (plan.quantized_layers()[0].group_size
                            if plan.quantized_layers() else
                            (args.group_size or 16))
    low_comp = low_compression_warning(comp_stats, detection.architecture)
    if low_comp:
        warnings.append(low_comp)
    miss = policy_miss_warning(comp_stats, detection.architecture)
    if miss:
        warnings.append(miss)
    fail_threshold = args.min_quantized_byte_fraction
    if fail_threshold is None and args.fail_on_low_compression:
        fail_threshold = 0.10
    frac = comp_stats.get("quantized_fraction")
    if fail_threshold is not None and frac is not None and frac < fail_threshold:
        flag = ("--min-quantized-byte-fraction" if args.min_quantized_byte_fraction
                else "--fail-on-low-compression")
        raise UsageError(
            f"quantized byte fraction {100*frac:.1f}% is below the required "
            f"{100*fail_threshold:.0f}% ({flag}); nothing was written. Lower "
            "the threshold or drop the flag to allow the conversion.")

    if args.metadata_only:
        metadata_input_hashes = hash_checkpoint_files(info)
        meta = dict(info.metadata)
        meta[METADATA_KEY_QUANT] = json_dumps(plan.metadata_quant)
        meta[METADATA_KEY_EXT] = json_dumps(build_extension_metadata(
            info, plan, env, args, calibration, sensitivity,
            metadata_input_hashes, "", {"status": "metadata-only"},
            warnings))
        if args.output:
            _atomic_write_json(args.output + ".metadata.json", meta, indent=1)
            log().info("metadata written to %s.metadata.json", args.output)
        else:
            print(json_dumps(meta))
        report = build_report(info, plan, env, args, detection, None, {}, {},
                              {}, "", time.time() - t_start, warnings, quant_rows)
        _write_reports(args, report)
        return 0

    if args.dry_run:
        report = build_report(info, plan, env, args, detection, None, {}, {},
                              {}, "", time.time() - t_start, warnings, quant_rows)
        _write_reports(args, report)
        print_console_summary(
            out_path=out_path, input_bytes=info.total_bytes,
            output_bytes=total_out, architecture=detection.architecture,
            confidence=detection.confidence, n_quantized=plan.n_quantized,
            n_kept=plan.n_kept, comp=comp_stats, group_size=effective_group_size,
            validation=None, elapsed=time.time() - t_start, warnings=warnings,
            report_path=os.path.abspath(args.report) if args.report else None,
            dry_run=True, mixed_plan=plan.mixed_plan)
        return 0

    # ---- conversion ----
    engine = ConversionEngine(info, plan, args, state_path, tmp_path, staged_path)
    try:
        engine.run()
    except Exception:
        try:
            engine.save_state()
        finally:
            engine.close()
        log().error("conversion failed; state saved to %s (rerun with --resume)",
                    state_path)
        raise
    engine.close()
    input_hashes = engine.input_hashes
    metrics = sensitivity.results if sensitivity else {}

    # ---- provisional metadata + validation ----
    tensor_payload_sha = sha256_safetensors_payload(staged_path)
    # Preserve benign source metadata, overriding only the two keys owned by
    # this converter.  Quantized inputs are refused earlier, so replacement is
    # defensive rather than a re-quantization path.
    qm_meta = dict(info.metadata)
    qm_meta[METADATA_KEY_QUANT] = json_dumps(plan.metadata_quant)
    ext_meta = build_extension_metadata(
        info, plan, env, args, calibration, sensitivity, input_hashes,
        tensor_payload_sha, {"status": "pending"}, warnings)
    qm_meta[METADATA_KEY_EXT] = json_dumps(ext_meta)
    log().info("building validation copy at %s", validation_path)
    republish_with_metadata(staged_path, validation_path, qm_meta, entries)

    # ---- validation ----
    validator = Validator(info, plan, validation_path, args, env)
    with CheckpointReader(info) as validation_reader:
        summary = validator.run(reader=validation_reader, metrics=metrics,
                                input_hashes=input_hashes)

    # Never expose a newly generated checkpoint at the requested output path
    # when standalone validation failed.  The checksummed staged file and state
    # remain available for --resume; an older --overwrite target is untouched.
    if summary["n_failed"]:
        warnings.append(
            f"validation failed {summary['n_failed']} checks; new output was not "
            "published and any pre-existing output remains unchanged")
        report = build_report(
            info, plan, env, args, detection, calibration, metrics, summary,
            input_hashes, "", time.time() - t_start, warnings, quant_rows)
        _write_reports(args, report)
        if os.path.exists(validation_path):
            _remove_temp_path(validation_path)
        return 2

    # Embed the completed tensor/schema validation.  Full-file SHA256 is not
    # embedded because changing the header changes that hash; the stable tensor
    # payload hash is embedded instead.
    embedded_summary = json.loads(json_dumps(summary))
    embedded_summary["checks"] = [
        check for check in embedded_summary.get("checks", [])
        if check.get("name") != "output-hash"
    ]
    embedded_summary.pop("output_sha256", None)
    embedded_summary["scope"] = (
        "tensor payload, reconstruction, policy and metadata schema before final "
        "metadata publication; tensor payload is unchanged by publication")
    _refresh_validation_summary(embedded_summary)
    ext_meta = build_extension_metadata(
        info, plan, env, args, calibration, sensitivity, input_hashes,
        tensor_payload_sha, embedded_summary, warnings)
    qm_meta[METADATA_KEY_EXT] = json_dumps(ext_meta)
    # Build and inspect the exact final checkpoint at the private validation
    # path.  Only a fully reopened, payload-bound candidate may replace the
    # requested output (including an existing --overwrite target).
    republish_with_metadata(staged_path, validation_path, qm_meta, entries)

    final_ok = False
    try:
        with RawSafetensorsFile(validation_path) as final_raw:
            final_meta = final_raw.metadata
        final_ext = json.loads(final_meta[METADATA_KEY_EXT])
        final_ok = (final_ext.get("output", {}).get("tensor_data_sha256") ==
                    sha256_safetensors_payload(validation_path))
        summary.setdefault("checks", []).append({
            "name": "final-publication", "status": "passed" if final_ok else "failed",
            "detail": "final file reopens; embedded tensor payload hash matches"
                      if final_ok else "final tensor payload hash mismatch",
            "reason": "",
        })
    except Exception as e:
        summary.setdefault("checks", []).append({
            "name": "final-publication", "status": "failed",
            "detail": f"final reopen failed: {e}", "reason": "",
        })
    _refresh_validation_summary(summary)
    if not final_ok:
        warnings.append(
            "final checkpoint candidate failed integrity checks; requested output "
            "was not replaced")
        report = build_report(
            info, plan, env, args, detection, calibration, metrics, summary,
            input_hashes, "", time.time() - t_start, warnings, quant_rows)
        _write_reports(args, report)
        if os.path.exists(validation_path):
            _remove_temp_path(validation_path)
        return 2

    output_sha = sha256_file(validation_path)
    os.replace(validation_path, out_path)
    _fsync_parent(out_path)
    summary["output_sha256"] = output_sha
    output_hash_check = next(
        (check for check in summary.get("checks", [])
         if check.get("name") == "output-hash"), None)
    if output_hash_check is not None:
        output_hash_check["detail"] = f"sha256={output_sha}"
    elif args.validate:
        summary.setdefault("checks", []).append({
            "name": "output-hash", "status": "passed",
            "detail": f"sha256={output_sha}", "reason": "",
        })
    _refresh_validation_summary(summary)

    # ---- report ----
    report = build_report(info, plan, env, args, detection, calibration, metrics,
                          summary, input_hashes, output_sha,
                          time.time() - t_start, warnings, quant_rows)
    _write_reports(args, report)
    print_console_summary(
        out_path=out_path, input_bytes=info.total_bytes,
        output_bytes=os.path.getsize(out_path),
        architecture=detection.architecture, confidence=detection.confidence,
        n_quantized=plan.n_quantized, n_kept=plan.n_kept, comp=comp_stats,
        group_size=effective_group_size, validation=summary,
        elapsed=time.time() - t_start, warnings=warnings,
        report_path=os.path.abspath(args.report) if args.report else None,
        dry_run=False, mixed_plan=plan.mixed_plan)

    # ---- cleanup ----
    if os.path.exists(tmp_path):
        _remove_temp_path(tmp_path)
    if os.path.exists(validation_path):
        _remove_temp_path(validation_path)
    if os.path.exists(staged_path):
        _remove_temp_path(staged_path)
    if os.path.exists(state_path):
        _remove_temp_path(state_path)
    return 0 if summary["n_failed"] == 0 else 2
