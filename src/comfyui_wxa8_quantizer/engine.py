"""Conversion engine: state, plan hashing, runtime compatibility and orchestration."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import dataclasses
import hashlib
import math
import os
import re
import time
import torch
from comfyui_wxa8_quantizer.constants import FORMAT_INT8, FORMAT_W4A4, FORMAT_W4A8, FORMAT_W4A8_REVISION, MIXED_FORMATS, QUANT_ALGORITHM_REVISION, get_converter_version
from comfyui_wxa8_quantizer.errors import InputError, OutputError, PolicyError, RuntimeCompatibilityError
from comfyui_wxa8_quantizer.formats import format_scale_suffixes, quantize_weight_by_format
from comfyui_wxa8_quantizer.io import CheckpointInfo, CheckpointReader, RawSafetensorsFile, SafetensorsStreamWriter, _load_json_object, tensor_to_bytes
from comfyui_wxa8_quantizer.logging_utils import log
from comfyui_wxa8_quantizer.planner import MixedPlanner
from comfyui_wxa8_quantizer.planning import ConversionPlan, DecisionKind, SensitivityAnalyzer, TensorDecision, TensorMetrics
from comfyui_wxa8_quantizer.quantize import _chunk_rows_for_budget, _codebook_sample_size, _gather_codebook_samples, _quant_work_bytes, _quantize_row_chunk, quantize_tensor_bounded
from comfyui_wxa8_quantizer.runtime import EnvironmentInfo
from comfyui_wxa8_quantizer.utils import FP8_DTYPES, _atomic_write_json, _open_regular_nofollow, _remove_temp_path, human_bytes, json_dumps, sha256_file, torch_dtype_name
@dataclass
class ConversionState:
    version: int = 2
    plan_hash: str = ""
    output: str = ""
    tmp: str = ""
    fmt: str = ""
    entries: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # name -> {offset,size,status}
    temp_device: Optional[int] = None
    temp_inode: Optional[int] = None
    done: bool = False

def hash_checkpoint_files(info: CheckpointInfo, *, refresh: bool = False) -> Dict[str, str]:
    if refresh or not info.source_hashes:
        info.source_hashes = {path: sha256_file(path) for path in sorted(info.files)}
    return dict(info.source_hashes)

def _portable_file_labels(paths: Sequence[str]) -> List[str]:
    """Stable, non-secret file labels for metadata embedded in moved models."""
# SPDX-License-Identifier: Apache-2.0
    basenames = [Path(path).name for path in paths]
    if len(set(basenames)) == len(basenames):
        return basenames
    absolute = [os.path.abspath(path) for path in paths]
    try:
        common = os.path.commonpath([os.path.dirname(path) for path in absolute])
        relative = [os.path.relpath(path, common).replace(os.sep, "/")
                    for path in absolute]
        if len(set(relative)) == len(relative) and not any(
                label == ".." or label.startswith("../") for label in relative):
            return relative
    except ValueError:
        pass
    # Different volumes or adversarial duplicate names: retain deterministic
    # identity without embedding machine-specific absolute paths.
    return [f"{index:04d}-{name}" for index, name in enumerate(basenames)]

def _portable_hash_manifest(paths: Sequence[str], hashes: Dict[str, str]) -> Dict[str, str]:
    by_absolute = {os.path.abspath(path): digest for path, digest in hashes.items()}
    manifest: Dict[str, str] = {}
    for path, label in zip(paths, _portable_file_labels(paths), strict=True):
        absolute = os.path.abspath(path)
        if absolute not in by_absolute:
            raise OutputError(f"missing source hash for metadata: {path}")
        manifest[label] = by_absolute[absolute]
    return manifest

def _plan_hash(info: CheckpointInfo, plan: ConversionPlan, args: Any,
               source_hashes: Dict[str, str]) -> str:
    h = hashlib.sha256()
    options = {
        "converter_version": get_converter_version(),
        "format": plan.fmt,
        "format_revision": FORMAT_W4A8_REVISION,
        "algorithm_revision": QUANT_ALGORITHM_REVISION,
        "architecture": plan.detection.architecture,
        "prefix": plan.detection.unet_prefix,
        "group_size": getattr(args, "group_size", None),
        "compute_dtype": getattr(args, "compute_dtype", "auto"),
        "output_dtype": getattr(args, "output_dtype", "auto"),
        "min_numel": getattr(args, "min_numel_override", None),
        "seed": getattr(args, "seed", 0),
        "device": getattr(args, "device", "auto"),
        "max_memory": getattr(args, "max_memory", None),
        "include": getattr(args, "include", []),
        "exclude": getattr(args, "exclude", []),
        "keep_precision": getattr(args, "keep_precision", []),
        "sensitivity_threshold": getattr(args, "sensitivity_threshold", None),
        "error_threshold": getattr(args, "error_threshold", None),
        "calibration_source": getattr(args, "calibration_source", None),
        "calibration_samples": getattr(args, "calibration_samples", None),
        "calibration_cache": getattr(args, "calibration_cache", None),
        "profile": getattr(args, "profile", "auto"),
        "target_runtime": getattr(args, "target_runtime", "auto"),
        "quality_gate": getattr(args, "quality_gate", None),
        "global_error_gate": getattr(args, "global_error_gate", None),
        "max_linear_bytes_per_param": getattr(args, "max_linear_bytes_per_param", None),
        "max_bf16_fraction": getattr(args, "max_bf16_fraction", None),
        "w4a4_linear_dtype": getattr(args, "w4a4_linear_dtype", "int8"),
        "strip_gpu_identity": getattr(args, "strip_gpu_identity", False),
        "runtime_backend": getattr(args, "_effective_runtime", "auto"),
        "runtime_gpu": (getattr(args, "_runtime_gpu_name", None)
                        or getattr(args, "_effective_runtime", None)),
        "runtime_certificate_sha256": (
            sha256_file(args.runtime_certificate)
            if getattr(args, "runtime_certificate", None) else None),
        "disable_w4a4": getattr(args, "disable_w4a4", False),
        "disable_w4a8": getattr(args, "disable_w4a8", False),
        "disable_int8": getattr(args, "disable_int8", False),
        "source_sha256": source_hashes,
        "decisions": [{
            "name": d.name, "kind": d.kind.value, "layer": d.layer,
            "group_size": d.group_size,
            "convrot_groupsize": d.convrot_groupsize,
            "format": d.format,
            "out_dtype": torch_dtype_name(d.out_dtype) if d.out_dtype else None,
        } for d in plan.decisions],
        "entries": [{
            "name": e["name"], "dtype": torch_dtype_name(e["dtype"]),
            "shape": list(e["shape"]), "nbytes": e["nbytes"],
        } for e in plan.output_entries],
    }
    h.update(json_dumps(options).encode("utf-8"))
    return h.hexdigest()

FORMAT_TO_KITCHEN_LAYOUT = {
    FORMAT_W4A8: "AsymW4A8Int8Layout",
    FORMAT_W4A4: "TensorCoreConvRotW4A4Layout",
    FORMAT_INT8: "TensorWiseINT8Layout",
}

def _check_runtime_compatibility(env: EnvironmentInfo,
                                 planner: MixedPlanner,
                                 decisions: List[TensorDecision],
                                 warnings: List[str]) -> None:
    """Fail closed before conversion when the target machine demonstrably
    lacks a runtime path for a selected format.

    The probe is static (installed package source), never a runtime import.
    When neither comfy-kitchen nor ComfyUI is installed the probe is
    unavailable and the per-format runtime-contract validators remain the
    verification layer (with a warning)."""
    required = sorted({
        d.format for d in decisions
        if d.kind == DecisionKind.QUANTIZE and d.format in MIXED_FORMATS})
    if not required:
        return
    if env.has_comfy_quant_ops:
        missing = [f for f in required if f not in env.comfyui_quant_algos]
        if missing:
            raise RuntimeCompatibilityError(
                "mixed checkpoint requires ComfyUI formats "
                f"{', '.join(required)} but the installed ComfyUI "
                f"quant_ops.py registers only "
                f"{', '.join(env.comfyui_quant_algos) or 'none'}; missing: "
                f"{', '.join(missing)}. Update ComfyUI (>= v0.31.0) or avoid "
                "the affected formats with --disable-*.")
    if env.has_comfy_kitchen:
        layout_attr = {
            FORMAT_W4A8: "comfy_kitchen_has_w4a8_layout",
            FORMAT_W4A4: "comfy_kitchen_has_w4a4_layout",
            FORMAT_INT8: "comfy_kitchen_has_int8_layout",
        }
        missing = [f for f in required
                   if not getattr(env, layout_attr[f], False)]
        if missing:
            raise RuntimeCompatibilityError(
                "mixed checkpoint requires comfy-kitchen layouts "
                f"{', '.join(FORMAT_TO_KITCHEN_LAYOUT[f] for f in missing)} "
                "but the installed comfy-kitchen does not contain them; "
                "update comfy-kitchen or avoid the affected formats with "
                "--disable-*.")
    if not env.has_comfy_kitchen and not env.has_comfy_quant_ops:
        warnings.append(
            "runtime compatibility probe unavailable (neither ComfyUI nor "
            "comfy-kitchen installed); mixed output is verified by the "
            "per-format runtime-contract validators only")

class _SimulatedCrash(Exception):
    """Test-only hook: simulated interruption for resume-state self-tests."""

class ConversionEngine:
    def __init__(self, info: CheckpointInfo, plan: ConversionPlan, args: Any,
                 state_path: str, tmp_path: str, final_path: str):
        self.info = info
        self.plan = plan
        self.args = args
        self.state_path = state_path
        self.tmp_path = tmp_path
        self.final_path = final_path
        self.reader = CheckpointReader(info)
        self.input_hashes = hash_checkpoint_files(info)
        self.state = ConversionState(
            plan_hash=_plan_hash(info, plan, args, self.input_hashes),
            output=final_path, tmp=tmp_path,
            fmt=plan.fmt)
        self.writer: Optional[SafetensorsStreamWriter] = None
        self.metrics: Dict[str, TensorMetrics] = {}
        self.output_hash = ""

    # -- state persistence -------------------------------------------------
    def save_state(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            if self.writer.identity is not None:
                self.state.temp_device, self.state.temp_inode = self.writer.identity
        _atomic_write_json(self.state_path, dataclasses.asdict(self.state))

    def load_state(self) -> bool:
        if not os.path.exists(self.state_path):
            return False
        try:
            data = _load_json_object(
                self.state_path, "resume state", nofollow=True)
            st = ConversionState(**{k: v for k, v in data.items() if k in ConversionState.__dataclass_fields__})
            if (type(st.version) is not int
                    or not all(isinstance(value, str) for value in (
                        st.plan_hash, st.output, st.tmp, st.fmt))
                    or not isinstance(st.entries, dict)
                    or type(st.done) is not bool
                    or any(value is not None and type(value) is not int
                           for value in (st.temp_device, st.temp_inode))
                    or any(value is not None and value < 0
                           for value in (st.temp_device, st.temp_inode))):
                raise OutputError("resume state has invalid field types")
            for name, record in st.entries.items():
                if (not isinstance(name, str) or not name
                        or not isinstance(record, dict)
                        or type(record.get("offset")) is not int
                        or record["offset"] < 0
                        or type(record.get("size")) is not int
                        or record["size"] < 0
                        or record.get("status") != "done"
                        or not isinstance(record.get("sha256"), str)
                        or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])):
                    raise OutputError(
                        f"resume state has an invalid tensor record for {name!r}")
        except Exception as e:
            raise OutputError(
                f"cannot parse resume state {self.state_path}: {e}") from e
        if st.version != ConversionState().version:
            raise OutputError(
                f"resume state version {st.version} is unsupported; remove it and "
                "start a fresh conversion")
        if st.plan_hash != self.state.plan_hash:
            raise OutputError(
                "resume state does not match the current conversion parameters "
                "(input files, format, options changed). Remove the state file to start fresh.")
        if st.fmt != self.plan.fmt or os.path.abspath(st.output) != os.path.abspath(self.final_path):
            raise OutputError("resume state mismatch (output/format changed)")
        if os.path.abspath(st.tmp) != os.path.abspath(self.tmp_path):
            raise OutputError("resume state references a different temp path")
        if not os.path.exists(st.tmp) and not os.path.exists(self.final_path):
            raise OutputError(
                f"resume data missing: neither {st.tmp} nor {self.final_path} exists")
        self.state = st
        return True

    def completed_offsets(self) -> Dict[str, int]:
        return {k: v["offset"] for k, v in self.state.entries.items()
                if v.get("status") == "done"}

    def _verify_resume_entries(self, entries_by_name: Dict[str, Dict[str, Any]]) -> None:
        if self.writer is None:
            raise OutputError("resume verification requires an open output writer")
        for name, record in self.state.entries.items():
            if record.get("status") != "done":
                continue
            if name not in entries_by_name:
                raise OutputError(f"resume state contains unknown tensor {name!r}")
            expected = entries_by_name[name]
            if record.get("offset") != self.writer.offset_for(name) or \
                    record.get("size") != expected["nbytes"]:
                raise OutputError(f"resume state range mismatch for {name!r}")
            digest = record.get("sha256")
            if not isinstance(digest, str) or self.writer.tensor_sha256(name) != digest:
                raise OutputError(f"resume temp data checksum mismatch for {name!r}")

    def _verify_completed_output(
            self, entries_by_name: Dict[str, Dict[str, Any]]) -> None:
        """Verify a fully written staged output after a post-conversion crash."""
        completed = {name for name, record in self.state.entries.items()
                     if record.get("status") == "done"}
        if completed != set(entries_by_name):
            missing = sorted(set(entries_by_name) - completed)
            extra = sorted(completed - set(entries_by_name))
            raise OutputError(
                f"staged resume inventory mismatch: missing={missing[:5]} "
                f"extra={extra[:5]}")
        fd, file_stat = _open_regular_nofollow(self.final_path, os.O_RDONLY)
        identity = (int(file_stat.st_dev), int(file_stat.st_ino))
        os.close(fd)
        expected_identity = None
        if self.state.temp_device is not None and self.state.temp_inode is not None:
            expected_identity = (self.state.temp_device, self.state.temp_inode)
        if expected_identity is not None and identity != expected_identity:
            raise OutputError(
                "staged output identity changed; refusing possible replacement")
        with RawSafetensorsFile(self.final_path) as staged:
            if set(staged.entries) != set(entries_by_name):
                raise OutputError("staged output tensor inventory changed")
            for name, expected in entries_by_name.items():
                dtype, shape, _, _ = staged.get(name)
                if dtype != expected["dtype"] or tuple(shape) != tuple(expected["shape"]):
                    raise OutputError(f"staged output shape/dtype mismatch for {name!r}")
                digest = hashlib.sha256(staged.read_bytes(name)).hexdigest()
                if digest != self.state.entries[name].get("sha256"):
                    raise OutputError(f"staged output checksum mismatch for {name!r}")
        post_stat = os.stat(self.final_path, follow_symlinks=False)
        if (int(post_stat.st_dev), int(post_stat.st_ino)) != identity:
            raise OutputError("staged output was replaced during resume verification")

    _SUFFIX_TO_OUTPUT = {
        "": ".weight",
        "_s_rel": ".weight_s_rel",
        "_s_channel": ".weight_s_channel",
        "_codebook": ".weight_codebook",
        "_correction": ".weight_correction",
        "_scale": ".weight_scale",
    }

    @staticmethod
    def _decision_output_names(d: TensorDecision) -> List[str]:
        if d.kind == DecisionKind.QUANTIZE:
            if d.layer is None:
                raise PolicyError(f"quantized tensor {d.name!r} has no layer name")
            return [d.layer + ConversionEngine._SUFFIX_TO_OUTPUT[suffix]
                    for suffix in format_scale_suffixes(d.format)]
        return [d.name]

    def _record_entry(self, name: str, data_sha256: str,
                      entries_by_name: Dict[str, Dict[str, Any]]) -> None:
        if self.writer is None:
            raise OutputError("cannot record output before the writer is open")
        self.state.entries[name] = {
            "offset": self.writer.offset_for(name),
            "size": entries_by_name[name]["nbytes"],
            "status": "done",
            "sha256": data_sha256,
        }

    # -- main loop ---------------------------------------------------------
    def run(self, sensitivity: Optional[SensitivityAnalyzer] = None) -> None:
        args = self.args
        max_mem = args.max_memory or (2 * 1024**3)
        # Device selection was resolved once during planning.  PyTorch exposes
        # ROCm accelerators through the "cuda" device type, while plan.device
        # retains the user-facing backend identity for metadata and resume.
        device = torch.device(
            "cuda" if self.plan.device in ("cuda", "rocm") else "cpu")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise PolicyError(
                f"planned {self.plan.device} conversion but the accelerator is no "
                "longer available")

        entries_by_name = {e["name"]: e for e in self.plan.output_entries}
        resumed = self.load_state() if args.resume else False
        if resumed and not os.path.exists(self.tmp_path) \
                and os.path.exists(self.final_path):
            log().info("recovering completed staged output %s", self.final_path)
            self._verify_completed_output(entries_by_name)
            post_hashes = {
                path: sha256_file(path) for path in sorted(self.info.files)}
            if post_hashes != self.input_hashes:
                raise InputError(
                    "one or more source files changed since the staged output was "
                    "created")
            self.state.done = True
            _atomic_write_json(self.state_path, dataclasses.asdict(self.state))
            self.output_hash = sha256_file(self.final_path)
            return
        if resumed:
            log().info("resuming conversion from %s", self.tmp_path)
        else:
            if os.path.exists(self.tmp_path):
                if args.resume:
                    log().warning("no resume state found; starting fresh and removing "
                                  "stale temp output %s", self.tmp_path)
                    _remove_temp_path(self.tmp_path)
                elif args.overwrite:
                    log().info("removing stale temp output %s", self.tmp_path)
                    _remove_temp_path(self.tmp_path)
                else:
                    raise OutputError(
                        f"temp output already exists: {self.tmp_path} (use --overwrite or --resume)")
        resume_offsets = self.completed_offsets() if resumed else {}
        expected_identity = None
        if resumed and self.state.temp_device is not None and self.state.temp_inode is not None:
            expected_identity = (self.state.temp_device, self.state.temp_inode)
        self.writer = SafetensorsStreamWriter(self.tmp_path, self.plan.output_entries,
                                              resume_offsets=resume_offsets,
                                              resume_mode=resumed,
                                              expected_identity=expected_identity)
        self.writer.open()
        if resumed:
            self._verify_resume_entries(entries_by_name)
        else:
            self.save_state()

        total = len(self.plan.decisions)
        t0 = time.time()
        done = 0
        for d in self.plan.decisions:
            done += 1
            output_names = self._decision_output_names(d)
            completed_outputs = [
                name for name in output_names
                if self.state.entries.get(name, {}).get("status") == "done"
            ]
            if len(completed_outputs) == len(output_names):
                continue
            # A signal can land between the per-layer writes.  Recompute and
            # rewrite that entire logical layer so stored bytes and checksums
            # cannot mix two quantization attempts/backends.
            if completed_outputs:
                if self.writer is None:
                    raise OutputError("resume writer is not initialized")
                for output_name in output_names:
                    self.state.entries.pop(output_name, None)
                    self.writer.invalidate_resume_tensor(output_name)
            if d.kind == DecisionKind.QUANTIZE:
                compute_dtype = getattr(args, "_compute_dtype_tensor", torch.float32)
                meta = self.info.by_name(d.name)
                if meta is None:
                    raise PolicyError(f"conversion plan references missing tensor {d.name!r}")
                if _quant_work_bytes(meta) > max_mem:
                    self.plan.chunked_layers.add(d.name)
                    self._write_quantized_chunked(
                        d, entries_by_name, max_mem, device, compute_dtype)
                else:
                    quant_tensors = quantize_tensor_bounded(
                        self.reader, d.name, d.format, d.group_size,
                        d.convrot_groupsize, max_mem, device,
                        compute_dtype=compute_dtype)
                    self._write_quantized(d, quant_tensors, entries_by_name)
                    del quant_tensors
            else:
                self._write_passthrough(d, entries_by_name)

            if getattr(self, "_crash_after", None) is not None and done >= self._crash_after:
                raise _SimulatedCrash(f"simulated interruption after {done} tensors")
            if done == 1 or done % 10 == 0 or done == total:
                self.save_state()
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-9)
                log().info("progress %d/%d tensors (%.1f/s)", done, total, rate)

        if self.writer is None:
            raise OutputError("conversion ended without an output writer")
        expected_names = set(entries_by_name)
        completed_names = {name for name, record in self.state.entries.items()
                           if record.get("status") == "done"}
        if completed_names != expected_names:
            missing = sorted(expected_names - completed_names)
            raise OutputError(f"conversion ended with incomplete tensors: {missing[:5]}")
        post_hashes = {path: sha256_file(path) for path in sorted(self.info.files)}
        if post_hashes != self.input_hashes:
            raise InputError(
                "one or more source files changed during conversion; refusing to "
                "publish a mixed checkpoint")
        self.save_state()
        self.writer.finalize(self.final_path)
        self.output_hash = sha256_file(self.final_path)
        self.state.done = True
        _atomic_write_json(self.state_path, dataclasses.asdict(self.state))
        log().info("conversion complete: %s", self.final_path)

    # -- writers -----------------------------------------------------------
    def _write_quantized(self, d: TensorDecision, tensors: Dict[str, torch.Tensor],
                         entries_by_name: Dict[str, Dict[str, Any]]) -> None:
        if d.layer is None:
            raise PolicyError(f"quantized tensor {d.name!r} has no layer name")
        if self.writer is None:
            raise OutputError("cannot write quantized tensor before writer initialization")
        suffix_map = dict(ConversionEngine._SUFFIX_TO_OUTPUT)
        for suffix, t in tensors.items():
            out_name = d.layer + suffix_map[suffix]
            data = tensor_to_bytes(t)
            self.writer.write_tensor_bytes(out_name, data)
            self._record_entry(out_name, hashlib.sha256(data).hexdigest(),
                               entries_by_name)

    def _write_quantized_chunked(self, d: TensorDecision,
                                 entries_by_name: Dict[str, Dict[str, Any]],
                                 max_mem: int, device: torch.device,
                                 compute_dtype: Optional[torch.dtype]) -> None:
        if d.layer is None:
            raise PolicyError(f"quantized tensor {d.name!r} has no layer name")
        if self.writer is None:
            raise OutputError("cannot write quantized tensor before writer initialization")
        meta = self.info.by_name(d.name)
        if meta is None:
            raise PolicyError(f"conversion plan references missing tensor {d.name!r}")
        n, k = int(meta.shape[0]), int(meta.shape[1])
        if d.format != FORMAT_W4A8:
            self._write_quantized_rowwise_chunked(d, entries_by_name, max_mem,
                                                  device, compute_dtype)
            return
        chunk_rows = _chunk_rows_for_budget(k, n, max_mem)
        sample_size = _codebook_sample_size(max_mem, n * k)
        codebook = _gather_codebook_samples(
            self.reader, d.name, k, d.group_size, d.convrot_groupsize,
            sample_size, chunk_rows, compute_dtype=compute_dtype)
        suffix_map = {"": ".weight", "_s_rel": ".weight_s_rel",
                      "_s_channel": ".weight_s_channel"}
        hashers = {suffix: hashlib.sha256() for suffix in suffix_map}
        row_bytes = {
            "": entries_by_name[d.layer + ".weight"]["nbytes"] // n,
            "_s_rel": entries_by_name[d.layer + ".weight_s_rel"]["nbytes"] // n,
            "_s_channel": entries_by_name[d.layer + ".weight_s_channel"]["nbytes"] // n,
        }
        for r0 in range(0, n, chunk_rows):
            r1 = min(n, r0 + chunk_rows)
            part = _quantize_row_chunk(
                self.reader, d.name, r0, r1, d.group_size,
                d.convrot_groupsize, codebook, device, compute_dtype)
            for suffix, output_suffix in suffix_map.items():
                name = d.layer + output_suffix
                data = tensor_to_bytes(part[suffix])
                self.writer.write_tensor_slice(name, r0 * row_bytes[suffix], data)
                hashers[suffix].update(data)
            del part
        codebook_name = d.layer + ".weight_codebook"
        codebook_data = tensor_to_bytes(codebook)
        self.writer.write_tensor_bytes(codebook_name, codebook_data)
        for suffix, output_suffix in suffix_map.items():
            self._record_entry(d.layer + output_suffix, hashers[suffix].hexdigest(),
                               entries_by_name)
        self._record_entry(codebook_name, hashlib.sha256(codebook_data).hexdigest(),
                           entries_by_name)

    def _write_quantized_rowwise_chunked(
            self, d: TensorDecision,
            entries_by_name: Dict[str, Dict[str, Any]],
            max_mem: int, device: torch.device,
            compute_dtype: Optional[torch.dtype]) -> None:
        """Chunked writer for the rowwise formats (INT8 / W4A4): rows are
        independent, so each chunk quantizes exactly and scales stream out."""
        if d.layer is None or self.writer is None:
            raise PolicyError("chunked rowwise write needs a layer and open writer")
        meta = self.info.by_name(d.name)
        if meta is None:
            raise PolicyError(f"conversion plan references missing tensor {d.name!r}")
        n, k = int(meta.shape[0]), int(meta.shape[1])
        chunk_rows = _chunk_rows_for_budget(k, n, max_mem)
        suffixes = format_scale_suffixes(d.format)
        output_suffixes = {s: ConversionEngine._SUFFIX_TO_OUTPUT[s] for s in suffixes}
        hashers = {s: hashlib.sha256() for s in suffixes}
        row_bytes = {
            s: entries_by_name[d.layer + output_suffixes[s]]["nbytes"] // n
            for s in suffixes
        }
        for r0 in range(0, n, chunk_rows):
            r1 = min(n, r0 + chunk_rows)
            chunk = self.reader.read_tensor(d.name)[r0:r1]
            if compute_dtype is not None and chunk.dtype not in FP8_DTYPES:
                chunk = chunk.to(compute_dtype)
            if device.type == "cuda":
                chunk = chunk.to(device)
            part = quantize_weight_by_format(chunk, d.format, d.group_size,
                                             d.convrot_groupsize)
            for suffix in suffixes:
                name = d.layer + output_suffixes[suffix]
                data = tensor_to_bytes(part[suffix].cpu() if device.type != "cpu"
                                       else part[suffix])
                self.writer.write_tensor_slice(name, r0 * row_bytes[suffix], data)
                hashers[suffix].update(data)
            del chunk, part
        for suffix in suffixes:
            self._record_entry(d.layer + output_suffixes[suffix],
                               hashers[suffix].hexdigest(), entries_by_name)

    def _write_passthrough(self, d: TensorDecision,
                           entries_by_name: Dict[str, Dict[str, Any]]) -> None:
        if self.writer is None:
            raise OutputError("cannot write passthrough tensor before writer initialization")
        meta = self.info.by_name(d.name)
        if meta is None:
            raise PolicyError(f"conversion plan references missing tensor {d.name!r}")
        target_dtype = entries_by_name[d.name]["dtype"]
        max_mem = int(getattr(self.args, "max_memory", 2 * 1024**3))
        # Keep I/O buffers modest even when the user's quantization budget is
        # very large.  The output was preallocated, so slices may be written in
        # any order without ever materializing the full passthrough tensor.
        io_bytes = min(16 * 1024 * 1024, max(1, max_mem // 4))
        digest = hashlib.sha256()
        if target_dtype == meta.dtype:
            raw = self.reader.read_bytes(d.name)
            for byte_offset in range(0, meta.nbytes, io_bytes):
                data = bytes(raw[byte_offset:byte_offset + io_bytes])
                self.writer.write_tensor_slice(d.name, byte_offset, data)
                digest.update(data)
            del raw
        else:
            bytes_per_element = meta.dtype.itemsize + 2 * target_dtype.itemsize
            if max_mem < bytes_per_element:
                raise PolicyError(
                    f"--max-memory {human_bytes(max_mem)} cannot cast one "
                    f"{meta.dtype} element to {target_dtype}")
            elements_per_chunk = max(1, min(
                math.prod(meta.shape),
                max_mem // bytes_per_element,
                max(1, io_bytes // target_dtype.itemsize),
            ))
            source = self.reader.read_tensor(d.name).reshape(-1)
            numel = math.prod(meta.shape)
            for element_offset in range(0, numel, elements_per_chunk):
                converted = source[element_offset:element_offset + elements_per_chunk].to(
                    target_dtype)
                data = tensor_to_bytes(converted)
                self.writer.write_tensor_slice(
                    d.name, element_offset * target_dtype.itemsize, data)
                digest.update(data)
                del converted, data
            del source
        self._record_entry(d.name, digest.hexdigest(), entries_by_name)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
        self.reader.close()
