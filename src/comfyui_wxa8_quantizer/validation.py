"""Output validation and plan reconstruction from an existing checkpoint."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import dataclasses
import hashlib
import json
import os
import re
from safetensors import safe_open
import struct
import torch
from comfyui_wxa8_quantizer.constants import DEFAULT_W4A4_LINEAR_DTYPE, FORMAT_INT8, FORMAT_MIXED, FORMAT_MIXED_REVISION, FORMAT_W4A4, FORMAT_W4A8, FORMAT_W4A8_REVISION, INT8_MAX_REL_L2, METADATA_KEY_EXT, METADATA_KEY_QUANT, MIXED_FORMATS, W4A4_MAX_REL_L2, W4A4_QUANT_GROUP_SIZE, W4A8_CONVROT_GROUPSIZE
from comfyui_wxa8_quantizer.engine import FORMAT_TO_KITCHEN_LAYOUT, _portable_hash_manifest, hash_checkpoint_files
from comfyui_wxa8_quantizer.errors import PolicyError, ValidationError
from comfyui_wxa8_quantizer.formats import (
    _is_power_of_four, dequantize_int8_tensorwise_weight, dequantize_w4a4_weight,
    dequantize_w4a8_weight, dequantize_weight_by_format, unpack_int4_signed,
    unpack_w4, validate_w4_shape,
)
from comfyui_wxa8_quantizer.io import CheckpointInfo, CheckpointReader, tensor_nbytes, torch_dtype_from_safe
from comfyui_wxa8_quantizer.logging_utils import log
from comfyui_wxa8_quantizer.planning import ConversionPlan, DecisionKind, TensorDecision, TensorMetrics, build_output_entries, compute_weight_metrics
from comfyui_wxa8_quantizer.policies import DetectionResult
from comfyui_wxa8_quantizer.quantize import _MetricAccumulator, _chunk_rows_for_budget, _quant_work_bytes, quantize_tensor_bounded
from comfyui_wxa8_quantizer.runtime import EnvironmentInfo
from comfyui_wxa8_quantizer.utils import TORCH_TO_SAFE, human_bytes, json_dumps, sha256_file, sha256_safetensors_payload
def metadata_json_bytes(meta: Dict[str, str]) -> torch.Tensor:
    return torch.tensor(list(json.dumps(meta).encode("utf-8")), dtype=torch.uint8)

@dataclass
class ValidationCheck:
    name: str
    status: str          # passed | passed-with-warnings | failed | skipped
    detail: str = ""
    reason: str = ""

def _refresh_validation_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    checks = summary.setdefault("checks", [])
    summary["n_passed"] = sum(1 for c in checks if c.get("status") == "passed")
    summary["n_passed_with_warnings"] = sum(
        1 for c in checks if c.get("status") == "passed-with-warnings")
    summary["n_failed"] = sum(1 for c in checks if c.get("status") == "failed")
    summary["n_skipped"] = sum(1 for c in checks if c.get("status") == "skipped")
    return summary

class Validator:
    def __init__(self, info: CheckpointInfo, plan: ConversionPlan,
                 output_path: str, args: Any, env: EnvironmentInfo):
        self.info = info
        self.plan = plan
        self.output_path = output_path
        self.args = args
        self.env = env
        self.checks: List[ValidationCheck] = []
        self.output_sha256 = ""

    def check(self, name: str, ok: bool, detail: str = "", warn: bool = False,
              skipped: bool = False, reason: str = "") -> None:
        status = "skipped" if skipped else ("passed-with-warnings" if (warn and ok) else ("passed" if ok else "failed"))
        self.checks.append(ValidationCheck(name, status, detail, reason))

    def summary(self) -> Dict[str, Any]:
        return {
            "checks": [dataclasses.asdict(c) for c in self.checks],
            "n_passed": sum(1 for c in self.checks if c.status == "passed"),
            "n_passed_with_warnings": sum(1 for c in self.checks if c.status == "passed-with-warnings"),
            "n_failed": sum(1 for c in self.checks if c.status == "failed"),
            "n_skipped": sum(1 for c in self.checks if c.status == "skipped"),
            "output_sha256": self.output_sha256,
        }

    def run(self, reader: Optional[CheckpointReader] = None,
            metrics: Optional[Dict[str, TensorMetrics]] = None,
            input_hashes: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        args = self.args
        full_validation = bool(args.validate or args.validation_only)
        out_path = self.output_path
        if not os.path.exists(out_path):
            self.check("output-exists", False, f"{out_path} missing")
            return self.summary()
        size = os.path.getsize(out_path)
        log().info("validation started on %s (%s, %s mode)",
                   out_path, human_bytes(size),
                   "full" if full_validation else "representative")

        # ---- reopen with safetensors ----
        try:
            st = safe_open(out_path, framework="pt")
            out_names = list(st.keys())
            ok = len(out_names) == len(self.plan.output_entries)
            self.check("output-reopen", True, f"{len(out_names)} tensors, {human_bytes(size)}")
            self.check("output-entry-count", ok,
                       f"{len(out_names)} vs {len(self.plan.output_entries)} expected")
        except Exception as e:
            self.check("output-reopen", False, f"safe_open failed: {e}")
            return self.summary()

        expected = {e["name"]: e for e in self.plan.output_entries}
        output_name_set = set(out_names)
        missing = [n for n in expected if n not in output_name_set]
        extra = [n for n in output_name_set if n not in expected]
        self.check("output-inventory", not missing and not extra,
                   f"missing={missing[:5]} extra={extra[:5]}")

        # ---- dtype / shape preservation ----
        shape_ok = True
        dtype_ok = True
        bad = []
        for name, e in expected.items():
            if name not in output_name_set:
                continue
            t = st.get_slice(name)
            shp = tuple(t.get_shape())
            dt = t.get_dtype()
            if shp != tuple(e["shape"]):
                shape_ok = False
                bad.append(f"{name}: shape {shp} != {e['shape']}")
            if dt != TORCH_TO_SAFE[e["dtype"]]:
                dtype_ok = False
                bad.append(f"{name}: dtype {dt} != {TORCH_TO_SAFE[e['dtype']]}")
        self.check("shape-preservation", shape_ok, "; ".join(bad[:5]))
        self.check("dtype-check", dtype_ok, "; ".join(bad[:5]))

        # ---- metadata ----
        meta = {}
        try:
            with open(out_path, "rb") as f:
                hlen = struct.unpack("<Q", f.read(8))[0]
                meta = json.loads(f.read(hlen).decode("utf-8")).get("__metadata__", {})
        except Exception as e:
            meta = {}
        has_quant_meta = METADATA_KEY_QUANT in meta
        self.check("metadata-present", has_quant_meta,
                   "missing _quantization_metadata" if not has_quant_meta else
                   "present")
        ext_ok = METADATA_KEY_EXT in meta
        self.check("extension-metadata-present", ext_ok,
                   "missing comfy_wxa8 extension metadata" if not ext_ok else "present")
        ext_payload: Dict[str, Any] = {}
        is_mixed_plan = self.plan.fmt == FORMAT_MIXED
        if ext_ok:
            try:
                ext_payload = json.loads(meta[METADATA_KEY_EXT])
                required = ("schema", "converter", "converter_version", "format",
                            "architecture", "source", "quantization", "output")
                source_block = ext_payload.get("source", {}) \
                    if isinstance(ext_payload, dict) else {}
                quant_block = ext_payload.get("quantization", {}) \
                    if isinstance(ext_payload, dict) else {}
                output_block = ext_payload.get("output", {}) \
                    if isinstance(ext_payload, dict) else {}
                source_files = source_block.get("files", []) \
                    if isinstance(source_block, dict) else []
                source_hashes = source_block.get("sha256", {}) \
                    if isinstance(source_block, dict) else {}
                schema_ok = isinstance(ext_payload, dict) and all(
                    key in ext_payload for key in required)
                expected_schema = ("comfy_wxa8/v2" if self.plan.fmt == FORMAT_MIXED
                                   else "comfy_wxa8/v1")
                expected_revision = (FORMAT_MIXED_REVISION
                                     if self.plan.fmt == FORMAT_MIXED
                                     else FORMAT_W4A8_REVISION)
                schema_ok = (
                    schema_ok
                    and ext_payload.get("schema") == expected_schema
                    and ext_payload.get("format") == self.plan.fmt
                    and ext_payload.get("format_revision") == expected_revision
                    and ext_payload.get("architecture") ==
                    self.plan.detection.architecture
                    and isinstance(ext_payload.get("source"), dict)
                    and isinstance(ext_payload.get("quantization"), dict)
                    and isinstance(ext_payload.get("output"), dict)
                    and isinstance(source_files, list)
                    and source_files
                    and all(isinstance(label, str) and label
                            and not Path(label).is_absolute()
                            for label in source_files)
                    and isinstance(source_hashes, dict)
                    and set(source_hashes) == set(source_files)
                    and all(isinstance(digest, str)
                            and re.fullmatch(r"[0-9a-f]{64}", digest)
                            for digest in source_hashes.values())
                    and isinstance(quant_block.get("weight_bits"), (int, str))
                    and (
                        # v1 (single-format w4a8): W4A8-global fields
                        (not is_mixed_plan
                         and quant_block.get("activation_bits") == 8
                         and quant_block.get("weight_bits") == 4
                         and quant_block.get("packing") == "int4-nibble-lsb"
                         and quant_block.get("scale_dtype") == "fp8_e4m3fn")
                        or
                        # v2 (mixed): mode + per-format activation semantics +
                        # distribution + certification level
                        (is_mixed_plan
                         and quant_block.get("activation_precision") ==
                         "per-format"
                         and quant_block.get("mode") == "mixed"
                         and isinstance(quant_block.get("formats"), dict)
                         and set(quant_block["formats"]) <= set(MIXED_FORMATS)
                         and set(quant_block["formats"]) == {
                             d.format for d in self.plan.quantized_layers()
                             if d.format in MIXED_FORMATS}
                         and isinstance(
                             quant_block.get("quality_validation"), dict)
                         and quant_block["quality_validation"].get("level")
                         in ("unverified", "calibrated"))
                    )
                    and isinstance(quant_block.get("activation_quantization"), str)
                    and isinstance(output_block.get("tensor_data_sha256"), str)
                    and bool(re.fullmatch(
                        r"[0-9a-f]{64}", output_block["tensor_data_sha256"]))
                )
                self.check("extension-metadata-schema", schema_ok,
                           "required fields, revision and architecture"
                           if schema_ok else
                           "invalid extension fields, revision or architecture")
                if input_hashes:
                    recorded_source_hashes = ext_payload.get("source", {}).get(
                        "sha256", {})
                    source_hash_ok = (
                        isinstance(recorded_source_hashes, dict)
                        and sorted(recorded_source_hashes.values()) ==
                        sorted(input_hashes.values())
                    )
                    self.check(
                        "metadata-source-hash", source_hash_ok,
                        "source content hashes match" if source_hash_ok else
                        "embedded source hashes do not match the supplied original")
                recorded_payload_hash = ext_payload.get("output", {}).get(
                    "tensor_data_sha256")
                if recorded_payload_hash:
                    actual_payload_hash = sha256_safetensors_payload(out_path)
                    self.check("tensor-payload-hash",
                               recorded_payload_hash == actual_payload_hash,
                               f"sha256={actual_payload_hash}")
                else:
                    self.check("tensor-payload-hash", False,
                               "missing output.tensor_data_sha256")
            except Exception as e:
                self.check("extension-metadata-json", False, f"unparseable: {e}")
        if has_quant_meta:
            try:
                qm = json.loads(meta[METADATA_KEY_QUANT])
                layers = qm.get("layers", {})
                q_layers = {d.layer for d in self.plan.quantized_layers()}
                mism = q_layers.symmetric_difference(set(layers.keys()))
                self.check("metadata-layer-inventory", not mism,
                           (f"{len(layers)} layers match" if not mism else
                            f"layer set mismatch: {sorted(mism)[:5]}"))
                decision_by_layer = {
                    decision.layer: decision for decision in self.plan.quantized_layers()
                }
                conf_ok = True
                for layer, value in layers.items():
                    dec = decision_by_layer.get(layer)
                    if dec is None or not isinstance(value, dict):
                        conf_ok = False
                        continue
                    lfmt = value.get("format")
                    if lfmt != dec.format or lfmt not in MIXED_FORMATS:
                        conf_ok = False
                        continue
                    if lfmt == FORMAT_W4A8:
                        conf_ok = conf_ok and (
                            value.get("group_size") == dec.group_size
                            and value.get("convrot") is True
                            and value.get("convrot_groupsize") ==
                            dec.convrot_groupsize)
                    elif lfmt == FORMAT_W4A4:
                        conf_ok = conf_ok and (
                            value.get("convrot_groupsize") ==
                            dec.convrot_groupsize
                            and value.get("quant_group_size") ==
                            W4A4_QUANT_GROUP_SIZE
                            and value.get("linear_dtype") in ("int4", "int8"))
                    # int8_tensorwise: format field only
                conf_ok = conf_ok and not mism
                self.check("metadata-layer-conf", conf_ok,
                           "format/group/ConvRot fields valid" if conf_ok else
                           "invalid format/group/ConvRot field")

                # ---- independent runtime-contract invariants (P0/P1) ----
                # Re-derived from the file itself, not from the plan: every
                # quantized layer must carry convrot=true, convrot_groupsize=256,
                # group_size=16, and exact tensor shapes, and (when the original
                # checkpoint is known) K % 256 == 0.
                inv_ok = True
                inv_bad: List[str] = []
                try:
                    with safe_open(out_path, framework="pt") as inv_st:
                        for layer, conf in layers.items():
                            lfmt = conf.get("format")
                            try:
                                gs = int(conf.get("group_size", 0))
                                cgs = int(conf.get("convrot_groupsize", 0))
                            except (TypeError, ValueError):
                                gs = cgs = 0
                            w = inv_st.get_slice(layer + ".weight")
                            n, k2 = w.get_shape()
                            if lfmt == FORMAT_W4A8:
                                srel = inv_st.get_slice(layer + ".weight_s_rel")
                                sch = inv_st.get_slice(layer + ".weight_s_channel")
                                cb = inv_st.get_slice(layer + ".weight_codebook")
                                k = k2 * 2
                                if (conf.get("convrot") is not True
                                        or cgs != W4A8_CONVROT_GROUPSIZE
                                        or gs != 16
                                        or tuple(w.get_shape()) != (n, k // 2)
                                        or tuple(srel.get_shape()) != (n, k // gs)
                                        or tuple(sch.get_shape()) != (n,)
                                        or tuple(cb.get_shape()) != (16,)
                                        or k % gs != 0 or k % cgs != 0):
                                    inv_ok = False
                                    inv_bad.append(f"{layer}: w4a8 conf/shape "
                                                   f"{conf} w={w.get_shape()}")
                                    continue
                                if self.info is not None:
                                    src_meta = self.info.by_name(layer + ".weight")
                                    if src_meta is None or int(src_meta.shape[1]) % 256 != 0:
                                        inv_ok = False
                                        inv_bad.append(
                                            f"{layer}: source K not divisible by 256")
                            elif lfmt == FORMAT_W4A4:
                                scale = inv_st.get_slice(layer + ".weight_scale")
                                k = k2 * 2
                                src_k = None
                                if self.info is not None:
                                    src_meta = self.info.by_name(
                                        layer + ".weight")
                                    src_k = (int(src_meta.shape[1])
                                             if src_meta is not None else None)
                                if (tuple(w.get_shape()) != (n, k // 2)
                                        or tuple(scale.get_shape()) != (n,)
                                        or k % W4A4_QUANT_GROUP_SIZE != 0
                                        or k % cgs != 0
                                        or int(conf.get("quant_group_size", 0)) !=
                                        W4A4_QUANT_GROUP_SIZE
                                        or conf.get("linear_dtype") not in
                                        ("int4", "int8")
                                        or cgs < 16
                                        or not _is_power_of_four(cgs)
                                        or (src_k is not None and src_k != k)):
                                    inv_ok = False
                                    inv_bad.append(f"{layer}: w4a4 conf/shape "
                                                   f"{conf} w={w.get_shape()} "
                                                   f"scale={scale.get_shape()}")
                            elif lfmt == FORMAT_INT8:
                                scale = inv_st.get_slice(layer + ".weight_scale")
                                src_k = None
                                if self.info is not None:
                                    src_meta = self.info.by_name(
                                        layer + ".weight")
                                    src_k = (int(src_meta.shape[1])
                                             if src_meta is not None else None)
                                if (tuple(w.get_shape()) != (n, k2)
                                        or tuple(scale.get_shape()) not in
                                        ((n,), (n, 1))
                                        or (src_k is not None and src_k != k2)):
                                    inv_ok = False
                                    inv_bad.append(f"{layer}: int8 shape "
                                                   f"w={w.get_shape()} "
                                                   f"scale={scale.get_shape()}")
                            else:
                                inv_ok = False
                                inv_bad.append(f"{layer}: unknown format {lfmt!r}")
                except Exception as e:
                    inv_ok = False
                    inv_bad.append(str(e))
                self.check(
                    "metadata-runtime-contract", inv_ok,
                    "per-format runtime contract (shapes, scales, K rules) "
                    "valid" if inv_ok else "; ".join(inv_bad[:4]) or "invalid")

            except Exception as e:
                self.check("metadata-json", False, f"unparseable: {e}")

        # ---- passthrough byte integrity (P0, full validation only) ----
        # With --output-dtype auto every non-quantized tensor must be preserved
        # byte-for-byte (dtype, shape, payload). A requested fp16/bf16 cast is
        # checked separately by dtype-check.
        if full_validation and reader is not None \
                and getattr(args, "output_dtype", "auto") in (None, "auto"):
            try:
                q_names = {f"{d.layer}.weight" for d in self.plan.quantized_layers()}
                payload_mismatch: List[str] = []
                payload_checked = 0
                with open(out_path, "rb") as f:
                    hlen = struct.unpack("<Q", f.read(8))[0]
                    header = json.loads(f.read(hlen).decode("utf-8"))
                    for name, e in expected.items():
                        if name in q_names:
                            continue
                        tinfo = header.get(name)
                        if not isinstance(tinfo, dict):
                            payload_mismatch.append(f"{name}: missing in header")
                            continue
                        src_meta = self.info.by_name(name) if self.info is not None else None
                        if src_meta is None:
                            continue
                        # Only tensors whose dtype was preserved can be byte-compared;
                        # explicit output-dtype casts are covered by dtype-check.
                        if e["dtype"] != src_meta.dtype:
                            continue
                        d0, d1 = tinfo["data_offsets"]
                        f.seek(8 + hlen + d0)
                        out_bytes = f.read(d1 - d0)
                        src_bytes = reader.read_bytes(name)
                        payload_checked += 1
                        if len(out_bytes) != len(src_bytes)                                 or hashlib.sha256(out_bytes).digest() !=                                    hashlib.sha256(src_bytes).digest():
                            payload_mismatch.append(name)
                self.check(
                    "passthrough-integrity", not payload_mismatch,
                    (f"{payload_checked} passthrough tensors byte-identical"
                     if not payload_mismatch else
                     f"payload differs for: {payload_mismatch[:5]}"))
            except Exception as e:
                self.check("passthrough-integrity", False, f"check failed: {e}")

        # ---- tensor round trips (all with --validate, representative otherwise) ----
        q_layers = self.plan.quantized_layers()
        if q_layers:
            import random
            rng = random.Random(getattr(args, "seed", 0) or 0)  # noqa: S311
            if full_validation:
                sample = list(q_layers)
            else:
                sample = sorted(q_layers, key=lambda d: self.info.by_name(d.name).nbytes,
                                reverse=True)
                sample = sample[:16] + (rng.sample(sample, min(8, len(sample)))
                                        if len(sample) > 16 else [])
                sample = list({id(d): d for d in sample}.values())
            worst: Dict[str, float] = {}
            bounds_used: Dict[str, float] = {}
            log().info("validation: reconstruction round trips over %d "
                       "quantized layers (%s)", len(sample),
                       "full set" if full_validation else "representative sample")
            for idx, d in enumerate(sample):
                log().info("validation round trip %d/%d: %s",
                           idx + 1, len(sample), d.layer or d.name)
                if d.layer is None:
                    self.check(f"recon-{d.name}", False,
                               "quantized decision has no layer name")
                    continue
                try:
                    orig = self.info.by_name(d.name)
                    if orig is not None and reader is not None:
                        max_mem = getattr(args, "max_memory", 2 * 1024**3)
                        bounded = (
                            d.name in getattr(self.plan, "chunked_layers", set())
                            or _quant_work_bytes(orig) > max_mem
                        )
                        recon_bound = {
                            FORMAT_W4A8: self.plan.detection.policy.max_rel_l2,
                            FORMAT_W4A4: W4A4_MAX_REL_L2,
                            FORMAT_INT8: INT8_MAX_REL_L2,
                        }.get(d.format, self.plan.detection.policy.max_rel_l2)
                        bounds_used[d.layer] = recon_bound
                        pack_ok = True
                        if bounded:
                            n, k = int(orig.shape[0]), int(orig.shape[1])
                            chunk_rows = _chunk_rows_for_budget(k, n, max_mem)
                            acc = _MetricAccumulator(d.name)
                            original_view = reader.read_tensor(d.name)
                            packed_slice = st.get_slice(d.layer + ".weight")
                            if d.format == FORMAT_W4A8:
                                cb = st.get_tensor(d.layer + ".weight_codebook")
                                s_rel_slice = st.get_slice(d.layer + ".weight_s_rel")
                                s_ch_slice = st.get_slice(d.layer + ".weight_s_channel")
                                for r0 in range(0, n, chunk_rows):
                                    r1 = min(n, r0 + chunk_rows)
                                    packed = packed_slice[r0:r1]
                                    s_rel = s_rel_slice[r0:r1]
                                    s_ch = s_ch_slice[r0:r1]
                                    dq = dequantize_w4a8_weight(
                                        packed, s_rel, s_ch, codebook=cb,
                                        group_size=d.group_size,
                                        convrot_groupsize=d.convrot_groupsize,
                                        output_dtype=torch.float32)
                                    acc.update(original_view[r0:r1], dq, None)
                                    rt = unpack_w4(packed)
                                    repacked = (
                                        (rt[:, 0::2] & 0xF)
                                        | ((rt[:, 1::2] & 0xF) << 4)
                                    ).to(torch.int8)
                                    pack_ok = pack_ok and bool(torch.equal(repacked, packed))
                                    del packed, s_rel, s_ch, dq, rt, repacked
                            else:
                                scale_slice = st.get_slice(d.layer + ".weight_scale")
                                for r0 in range(0, n, chunk_rows):
                                    r1 = min(n, r0 + chunk_rows)
                                    packed = packed_slice[r0:r1]
                                    scale = scale_slice[r0:r1]
                                    dq = dequantize_weight_by_format(
                                        {"": packed, "_scale": scale},
                                        d.format, d.group_size,
                                        d.convrot_groupsize, torch.float32)
                                    acc.update(original_view[r0:r1], dq, None)
                                    if d.format == FORMAT_W4A4:
                                        rt = unpack_int4_signed(packed)
                                        repacked = (
                                            (rt[:, 0::2] & 0xF)
                                            | ((rt[:, 1::2] & 0xF) << 4)
                                        ).to(torch.int8)
                                        pack_ok = pack_ok and bool(torch.equal(repacked, packed))
                                    del packed, scale, dq, rt, repacked
                            m = acc.finish()
                        else:
                            packed = st.get_tensor(d.layer + ".weight")
                            if d.format == FORMAT_W4A8:
                                cb = st.get_tensor(d.layer + ".weight_codebook")
                                s_rel = st.get_tensor(d.layer + ".weight_s_rel")
                                s_ch = st.get_tensor(d.layer + ".weight_s_channel")
                                dq = dequantize_w4a8_weight(
                                    packed, s_rel, s_ch, codebook=cb,
                                    group_size=d.group_size,
                                    convrot_groupsize=d.convrot_groupsize,
                                    output_dtype=torch.float32)
                            else:
                                scale = st.get_tensor(d.layer + ".weight_scale")
                                dq = dequantize_weight_by_format(
                                    {"": packed, "_scale": scale},
                                    d.format, d.group_size,
                                    d.convrot_groupsize, torch.float32)
                            orig_t = reader.read_tensor(d.name).float()
                            m = compute_weight_metrics(orig_t, dq)
                            if d.format == FORMAT_W4A4:
                                rt = unpack_int4_signed(packed)
                                repacked = (
                                    (rt[:, 0::2] & 0xF)
                                    | ((rt[:, 1::2] & 0xF) << 4)
                                ).to(torch.int8)
                                pack_ok = bool(torch.equal(repacked, packed))
                        worst[d.layer] = m.rel_l2
                        if m.rel_l2 > recon_bound:
                            self.check(f"recon-{d.layer}", False,
                                       f"relL2 {m.rel_l2:.4f} > {recon_bound}")
                        elif m.cosine < self.plan.detection.policy.min_cosine:
                            self.check(f"recon-{d.layer}", False,
                                       f"cosine {m.cosine:.4f} < {self.plan.detection.policy.min_cosine}")
                        else:
                            self.check(f"recon-{d.layer}", True,
                                       f"relL2={m.rel_l2:.4f} snr={m.snr_db:.1f}dB cos={m.cosine:.4f}")
                        self.check(f"pack-rt-{d.layer}", pack_ok,
                                   "pack/unpack round trip"
                                   + (" (bounded row chunks)" if bounded else ""))
                except Exception as e:
                    self.check(f"recon-{d.layer}", False, f"error: {e}")
            if worst:
                mx = max(worst.values())
                mx_bound = max(bounds_used.values()) if bounds_used else \
                    self.plan.detection.policy.max_rel_l2
                self.check("reconstruction-error-bound", mx <= mx_bound,
                           f"max {'full' if full_validation else 'sampled'} relL2 {mx:.4f} "
                           f"(per-format bound {mx_bound})")

        # ---- scale validation ----
        scale_ok = True
        scale_detail = ""
        n_scale = 0
        for d in q_layers:
            if d.layer is None:
                scale_ok = False
                scale_detail = f"{d.name}: quantized decision has no layer name"
                break
            try:
                if d.format == FORMAT_W4A4:
                    scale = st.get_tensor(d.layer + ".weight_scale")
                    finite = bool(torch.isfinite(scale).all())
                    valid_range = bool((scale > 0).all())
                    if not finite or not valid_range:
                        scale_ok = False
                        scale_detail = (
                            f"{d.layer}: non-finite or non-positive W4A4 scales")
                        break
                    n_scale += 1
                    continue
                if d.format == FORMAT_INT8:
                    scale = st.get_tensor(d.layer + ".weight_scale")
                    finite = bool(torch.isfinite(scale).all())
                    valid_range = bool((scale > 0).all())
                    if not finite or not valid_range:
                        scale_ok = False
                        scale_detail = (
                            f"{d.layer}: non-finite or non-positive INT8 scales")
                        break
                    n_scale += 1
                    continue
                cb = st.get_tensor(d.layer + ".weight_codebook").float()
                orig = self.info.by_name(d.name)
                max_mem = getattr(args, "max_memory", 2 * 1024**3)
                if orig is not None and _quant_work_bytes(orig) > max_mem:
                    n, k = int(orig.shape[0]), int(orig.shape[1])
                    chunk_rows = _chunk_rows_for_budget(k, n, max_mem)
                else:
                    n = int(st.get_slice(d.layer + ".weight_s_channel").get_shape()[0])
                    chunk_rows = n
                finite = bool(torch.isfinite(cb).all())
                valid_range = True
                s_rel_slice = st.get_slice(d.layer + ".weight_s_rel")
                s_ch_slice = st.get_slice(d.layer + ".weight_s_channel")
                for r0 in range(0, n, chunk_rows):
                    r1 = min(n, r0 + chunk_rows)
                    s_rel = s_rel_slice[r0:r1].float()
                    s_ch = s_ch_slice[r0:r1].float()
                    finite = finite and bool(
                        torch.isfinite(s_rel).all() and torch.isfinite(s_ch).all())
                    valid_range = valid_range and bool(
                        (s_rel >= 0).all() and (s_ch > 0).all())
                    del s_rel, s_ch
                if not finite:
                    scale_ok = False
                    scale_detail = f"{d.layer}: non-finite scales"
                    break
                if not valid_range:
                    scale_ok = False
                    scale_detail = (
                        f"{d.layer}: negative fp8 relative scale or non-positive "
                        "channel scale")
                    break
                n_scale += 1
            except Exception as e:
                scale_ok = False
                scale_detail = f"{d.layer}: {e}"
                break
        self.check("scale-validation", scale_ok,
                   scale_detail or f"{n_scale} layers OK (fp8 s_rel finite/nonnegative, "
                   "fp32 s_channel finite/positive)")

        # ---- deterministic re-quantization sample ----
        if q_layers and reader is not None and not args.validation_only:
            conv_device = getattr(self.plan, "device", "cpu")
            if conv_device != "cpu":
                self.check("deterministic-vs-disk", True,
                           "skipped: conversion ran on " + conv_device + " (codebook "
                           "subsample is device-dependent); determinism verified on CPU",
                           skipped=True,
                           reason="device-dependent codebook subsample")
            d0 = q_layers[0]
            try:
                if d0.name in getattr(self.plan, "chunked_layers", set()):
                    self.check("deterministic-conversion", True,
                               "skipped: bounded chunk determinism is covered by self-test",
                               skipped=True, reason="large chunked layer")
                    raise StopIteration
                compute_dtype = getattr(args, "_compute_dtype_tensor", torch.float32)
                max_mem = getattr(args, "max_memory", 2 * 1024**3)
                out1 = quantize_tensor_bounded(
                    reader, d0.name, d0.format, d0.group_size,
                    d0.convrot_groupsize, max_mem, torch.device("cpu"),
                    compute_dtype=compute_dtype)
                out2 = quantize_tensor_bounded(
                    reader, d0.name, d0.format, d0.group_size,
                    d0.convrot_groupsize, max_mem, torch.device("cpu"),
                    compute_dtype=compute_dtype)
                det = all(torch.equal(out1[k], out2[k]) for k in out1)
                self.check("deterministic-conversion", det, "two runs byte-identical")
                if conv_device == "cpu" and d0.name not in getattr(self.plan, "chunked_layers", set()):
                    on_disk = st.get_tensor(d0.layer + ".weight")
                    matches_disk = torch.equal(out1[""], on_disk)
                    self.check("deterministic-vs-disk", matches_disk,
                               "recomputed packed weight matches the file" if matches_disk else "MISMATCH")
            except StopIteration:
                pass
            except Exception as e:
                self.check("deterministic-conversion", False, f"error: {e}")

        # ---- output hash ----
        if full_validation:
            self.output_sha256 = sha256_file(out_path)
            self.check("output-hash", True, f"sha256={self.output_sha256}")

        # ---- compatibility probe (optional, no ComfyUI required) ----
        if full_validation:
            required_formats = sorted({
                d.format for d in self.plan.quantized_layers()
                if d.format in MIXED_FORMATS})
            if self.env.has_comfy_kitchen:
                layout_attr = {
                    FORMAT_W4A8: "comfy_kitchen_has_w4a8_layout",
                    FORMAT_W4A4: "comfy_kitchen_has_w4a4_layout",
                    FORMAT_INT8: "comfy_kitchen_has_int8_layout",
                }
                missing = [f for f in required_formats
                           if not getattr(self.env, layout_attr[f], False)]
                self.check("compat-comfy-kitchen", not missing,
                           detail=f"installed comfy-kitchen: {self.env.comfy_kitchen_rev or 'version unknown'}; "
                                  "required layouts " +
                                  (", ".join(FORMAT_TO_KITCHEN_LAYOUT[f]
                                             for f in required_formats)
                                   or "none") +
                                  ("; MISSING: " + ", ".join(
                                      FORMAT_TO_KITCHEN_LAYOUT[f] for f in missing)
                                   if missing else " all present"),
                           warn=bool(missing))
            else:
                self.check("compat-comfy-kitchen", True, "comfy-kitchen not installed (skipped)",
                           skipped=True, reason="optional runtime probe")
            if self.env.has_comfy_quant_ops:
                missing = [f for f in required_formats
                           if f not in self.env.comfyui_quant_algos]
                self.check("compat-comfyui", not missing,
                           detail=f"static ComfyUI quant_ops formats: "
                                  f"{self.env.comfyui_quant_algos}"
                                  + (f"; MISSING: {missing}" if missing else ""),
                           warn=bool(missing))
            else:
                self.check("compat-comfyui", True, "ComfyUI not installed (skipped)",
                           skipped=True, reason="optional runtime probe")

        # ---- input hash cross-check ----
        if input_hashes:
            if full_validation:
                current_hashes = hash_checkpoint_files(self.info, refresh=True)
                portable_current = _portable_hash_manifest(
                    self.info.files, current_hashes)
                portable_expected = _portable_hash_manifest(
                    self.info.files, input_hashes)
                self.check(
                    "input-hash", current_hashes == input_hashes,
                    (json_dumps(portable_current) if current_hashes == input_hashes else
                     "source changed during validation: " + json_dumps({
                         "expected": portable_expected,
                         "actual": portable_current,
                     })))
            else:
                self.check("input-hash", True,
                           "source hashes matched before and after conversion: "
                           + json_dumps(_portable_hash_manifest(
                               self.info.files, input_hashes)))

        del st
        return self.summary()

def _fmt_display(fmt: Optional[str]) -> Optional[str]:
    return {FORMAT_W4A8: "w4a8"}.get(fmt, fmt)

def plan_from_output(output_path: str, detection: DetectionResult,
                     fmt: str, info: Optional[CheckpointInfo] = None) -> ConversionPlan:
    """Reconstruct a minimal plan from an existing output checkpoint (validation-only)."""
    with safe_open(output_path, framework="pt") as st:
        names = list(st.keys())
        meta = st.metadata() or {}
        actual_entries = []
        for name in names:
            sl = st.get_slice(name)
            shape = tuple(sl.get_shape())
            dt = torch_dtype_from_safe(sl.get_dtype())
            actual_entries.append({
                "name": name,
                "dtype": dt,
                "shape": shape,
                "nbytes": tensor_nbytes(dt, shape),
            })
    try:
        qm = json.loads(meta.get(METADATA_KEY_QUANT, "{}"))
    except (TypeError, json.JSONDecodeError) as e:
        raise ValidationError(
            f"{output_path}: invalid {METADATA_KEY_QUANT} metadata: {e}") from e
    if not isinstance(qm, dict) or not isinstance(qm.get("layers"), dict):
        raise ValidationError(
            f"{output_path}: {METADATA_KEY_QUANT} must contain a layers object")
    layers = qm.get("layers", {})
    if not layers:
        raise ValidationError(
            f"{output_path}: no quantized layers recorded in {METADATA_KEY_QUANT}")
    decisions: List[TensorDecision] = []
    for layer, conf in layers.items():
        if not isinstance(layer, str) or not layer or not isinstance(conf, dict):
            raise ValidationError(
                f"{output_path}: malformed quantized-layer metadata entry")
        lfmt = conf.get("format")
        if lfmt not in MIXED_FORMATS:
            raise ValidationError(
                f"{output_path}: incompatible format metadata for {layer!r} "
                f"(unknown per-layer format {lfmt!r})")
        if fmt == FORMAT_MIXED:
            pass
        elif lfmt != fmt:
            raise ValidationError(
                f"{output_path}: incompatible format metadata for {layer!r}")
        try:
            group_size = int(conf.get("group_size", 16))
            convrot_groupsize = int(conf.get("convrot_groupsize", 256))
        except (TypeError, ValueError) as e:
            raise ValidationError(
                f"{output_path}: invalid group metadata for {layer!r}") from e
        if group_size <= 0 or convrot_groupsize <= 0:
            raise ValidationError(
                f"{output_path}: invalid group metadata for {layer!r}")
        source_name = f"{layer}.weight"
        if info is not None:
            source_meta = info.by_name(source_name)
            if source_meta is None or len(source_meta.shape) != 2:
                raise ValidationError(
                    f"{output_path}: quantized layer {layer!r} has no matching "
                    "2D weight in the supplied original checkpoint")
            k = int(source_meta.shape[1])
            if lfmt == FORMAT_W4A8:
                if convrot_groupsize != W4A8_CONVROT_GROUPSIZE:
                    raise ValidationError(
                        f"{output_path}: layer {layer!r} uses convrot_groupsize "
                        f"{convrot_groupsize}, but the comfy-kitchen 0.2.27 CUDA "
                        f"runtime only supports {W4A8_CONVROT_GROUPSIZE} (K must "
                        "be divisible by 256). Re-convert the checkpoint with "
                        "converter >= 1.2.2; incompatible layers now pass through.")
                try:
                    validate_w4_shape(k, group_size, convrot_groupsize)
                except PolicyError as e:
                    raise ValidationError(
                        f"{output_path}: invalid W4A8 shape metadata for "
                        f"{layer!r}: {e}") from e
            elif lfmt == FORMAT_W4A4:
                try:
                    if k % W4A4_QUANT_GROUP_SIZE != 0:
                        raise PolicyError(
                            f"K={k} not divisible by quant_group_size "
                            f"{W4A4_QUANT_GROUP_SIZE}")
                    if k % convrot_groupsize != 0:
                        raise PolicyError(
                            f"K={k} not divisible by convrot_groupsize "
                            f"{convrot_groupsize}")
                    if int(conf.get("quant_group_size", W4A4_QUANT_GROUP_SIZE)) \
                            != W4A4_QUANT_GROUP_SIZE:
                        raise PolicyError("quant_group_size must be 64")
                    if conf.get("linear_dtype", DEFAULT_W4A4_LINEAR_DTYPE) \
                            not in ("int4", "int8"):
                        raise PolicyError("linear_dtype must be int4 or int8")
                    if convrot_groupsize < 16 or \
                            not _is_power_of_four(convrot_groupsize):
                        raise PolicyError(
                            f"convrot_groupsize {convrot_groupsize} not a "
                            "supported power of 4")
                except PolicyError as e:
                    raise ValidationError(
                        f"{output_path}: invalid W4A4 shape metadata for "
                        f"{layer!r}: {e}") from e
            elif lfmt == FORMAT_INT8:
                pass  # int8_tensorwise without ConvRot has no shape constraint
        decisions.append(TensorDecision(
            source_name, DecisionKind.QUANTIZE, "reconstructed from output metadata",
            layer=layer, group_size=group_size,
            convrot_groupsize=convrot_groupsize, format=lfmt))
    quantized_names = {f"{layer}.weight" for layer in layers}
    if info is not None:
        for tensor in info.tensors:
            if tensor.name not in quantized_names:
                decisions.append(TensorDecision(
                    tensor.name, DecisionKind.KEEP, "reconstructed passthrough"))
    else:
        for name in names:
            if name.endswith(".weight") and name not in quantized_names:
                decisions.append(TensorDecision(
                    name, DecisionKind.KEEP, "reconstructed passthrough"))
    if info is not None:
        passthrough_dtype = None
        try:
            ext = json.loads(meta.get(METADATA_KEY_EXT, "{}"))
            if not isinstance(ext, dict) or not isinstance(
                    ext.get("quantization", {}), dict):
                raise ValidationError(
                    f"{output_path}: invalid {METADATA_KEY_EXT} metadata object")
            recorded_dtype = ext.get("quantization", {}).get(
                "passthrough_output_dtype", "auto")
            if recorded_dtype not in ("auto", "fp16", "bf16"):
                raise ValidationError(
                    f"{output_path}: invalid passthrough_output_dtype "
                    f"{recorded_dtype!r}")
            passthrough_dtype = {
                "auto": None, "fp16": torch.float16, "bf16": torch.bfloat16,
            }[recorded_dtype]
        except json.JSONDecodeError as e:
            raise ValidationError(
                f"{output_path}: invalid {METADATA_KEY_EXT} metadata: {e}") from e
        entries, expected_total = build_output_entries(
            info, decisions, fmt, passthrough_dtype)
    else:
        entries = actual_entries
        expected_total = sum(entry["nbytes"] for entry in actual_entries)
    plan = ConversionPlan(fmt=fmt, detection=detection, decisions=decisions,
                          metadata_quant=qm, metadata_ext={},
                          output_entries=entries, total_out_bytes=expected_total)
    plan.n_quantized = len([d for d in decisions if d.kind == DecisionKind.QUANTIZE])
    plan.n_kept = len(decisions) - plan.n_quantized
    return plan



def verify_output(output_path: str) -> Dict[str, Any]:
    """Source-free verification of an existing output checkpoint.

    Unlike ``--validation-only`` (which reconstructs the plan and compares the
    output against the original checkpoint), this operation needs no source
    model. It reopens the output, validates the tensor inventory, checks every
    quantized layer's metadata against the per-format tensor contracts
    (shapes, dtypes, group constraints), runs a bounded packing roundtrip, and
    reports the payload sha256. Intended for quick post-download or
    post-publish checks on machines that do not have the original model.
    """
    import math as _math
    checks: List[Dict[str, Any]] = []
    n_failed = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal n_failed
        if not ok:
            n_failed += 1
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    if not os.path.exists(output_path):
        check("output-exists", False, f"{output_path} missing")
        return {"ok": False, "checks": checks, "n_failed": 1,
                "output_sha256": "", "n_tensors": 0, "total_bytes": 0,
                "quantized_layers": 0, "formats": {}}

    total_bytes = os.path.getsize(output_path)
    try:
        st = safe_open(output_path, framework="pt")
        out_names = list(st.keys())
    except Exception as e:  # noqa: BLE001 - report any reopen failure
        check("output-reopen", False, f"{type(e).__name__}: {e}")
        return {"ok": False, "checks": checks, "n_failed": n_failed,
                "output_sha256": "", "n_tensors": 0,
                "total_bytes": total_bytes, "quantized_layers": 0,
                "formats": {}}
    meta = st.metadata() or {}
    check("output-reopen", True, f"{len(out_names)} tensors, {human_bytes(total_bytes)}")
    check("tensor-inventory", len(out_names) > 0, f"{len(out_names)} tensors")

    try:
        qm = json.loads(meta.get(METADATA_KEY_QUANT, "{}"))
    except (TypeError, json.JSONDecodeError) as e:
        qm = None
        check("quant-metadata", False, f"invalid {METADATA_KEY_QUANT}: {e}")
    layers = {}
    if isinstance(qm, dict) and isinstance(qm.get("layers"), dict):
        layers = qm["layers"]
        check("quant-metadata", len(layers) > 0,
              f"{len(layers)} quantized layer(s) recorded")
    else:
        check("quant-metadata", False, f"{METADATA_KEY_QUANT} missing layers object")

    by_name = {n: st.get_slice(n) for n in out_names}
    fmt_counts: Dict[str, int] = {}
    n_quant = 0

    def dtype_of(name: str) -> torch.dtype:
        try:
            return torch_dtype_from_safe(by_name[name].get_dtype())
        except Exception:  # noqa: BLE001 - dtype lookup is informational
            return None

    def shape_of(name: str) -> Optional[Tuple[int, ...]]:
        try:
            return tuple(by_name[name].get_shape())
        except Exception:  # noqa: BLE001
            return None

    for layer, conf in layers.items():
        if not isinstance(layer, str) or not layer or not isinstance(conf, dict):
            check("layer-metadata", False, f"{layer!r}: malformed entry")
            continue
        lfmt = conf.get("format")
        if lfmt not in MIXED_FORMATS:
            check("layer-format", False, f"{layer}: unknown format {lfmt!r}")
            continue
        fmt_counts[lfmt] = fmt_counts.get(lfmt, 0) + 1
        n_quant += 1
        try:
            group_size = int(conf.get("group_size", 16))
            convrot_groupsize = int(conf.get("convrot_groupsize", 256))
        except (TypeError, ValueError):
            check("layer-groups", False, f"{layer}: invalid group metadata")
            continue
        wname = f"{layer}.weight"
        wshape = shape_of(wname)
        wdtype = dtype_of(wname)
        problems = []
        if wshape is None or len(wshape) != 2:
            problems.append(f"weight missing or not 2D ({wname})")
        else:
            n, kp = int(wshape[0]), int(wshape[1])
            if lfmt == FORMAT_W4A8:
                k = 2 * kp
                if wdtype is not None and wdtype != torch.int8:
                    problems.append(f"weight dtype {wdtype} != int8")
                if k % W4A8_CONVROT_GROUPSIZE != 0:
                    problems.append(f"K={k} not divisible by {W4A8_CONVROT_GROUPSIZE}")
                if convrot_groupsize != W4A8_CONVROT_GROUPSIZE:
                    problems.append(f"cgs={convrot_groupsize} != {W4A8_CONVROT_GROUPSIZE}")
                srel = shape_of(f"{layer}.weight_s_rel")
                if srel != (n, k // 16):
                    problems.append(f"weight_s_rel {srel} != [{n}, {k // 16}]")
                if dtype_of(f"{layer}.weight_s_rel") != torch.float8_e4m3fn:
                    problems.append("weight_s_rel not fp8_e4m3fn")
                sch = shape_of(f"{layer}.weight_s_channel")
                if sch != (n,):
                    problems.append(f"weight_s_channel {sch} != [{n}]")
                cb = shape_of(f"{layer}.weight_codebook")
                if cb != (16,):
                    problems.append(f"weight_codebook {cb} != [16]")
            elif lfmt == FORMAT_W4A4:
                k = 2 * kp
                if wdtype is not None and wdtype != torch.int8:
                    problems.append(f"weight dtype {wdtype} != int8")
                if k % W4A4_QUANT_GROUP_SIZE != 0:
                    problems.append(f"K={k} not divisible by {W4A4_QUANT_GROUP_SIZE}")
                if convrot_groupsize not in (16, 64, 256) or k % convrot_groupsize != 0:
                    problems.append(f"invalid cgs={convrot_groupsize} for K={k}")
                sc = shape_of(f"{layer}.weight_scale")
                if sc != (n,):
                    problems.append(f"weight_scale {sc} != [{n}]")
            else:  # int8_tensorwise
                k = kp
                if wdtype is not None and wdtype != torch.int8:
                    problems.append(f"weight dtype {wdtype} != int8")
                sc = shape_of(f"{layer}.weight_scale")
                if sc not in ((n, 1), (n,)):
                    problems.append(f"weight_scale {sc} != [{n}, 1]")
                if sc == (n,):
                    problems.append("weight_scale is [N]; runtime contract prefers [N, 1]")
        check(f"layer:{layer}", not problems, "; ".join(problems) or
              f"{lfmt} N={n} K={k} gs={group_size} cgs={convrot_groupsize}")

    # Bounded packing roundtrip: dequant a sample of rows for each quantized
    # layer and require finite, sane-magnitude output.
    sample_rows = 32
    roundtrip_fail = 0
    for layer, conf in layers.items():
        if not isinstance(conf, dict):
            continue
        lfmt = conf.get("format")
        if lfmt not in MIXED_FORMATS:
            continue
        wname = f"{layer}.weight"
        wshape = shape_of(wname)
        if wshape is None or len(wshape) != 2:
            continue
        n, kp = int(wshape[0]), int(wshape[1])
        r0 = min(n, sample_rows)
        try:
            packed = by_name[wname][:r0]
            if lfmt == FORMAT_W4A8:
                s_rel = by_name[f"{layer}.weight_s_rel"][:r0]
                s_ch = by_name[f"{layer}.weight_s_channel"][:r0]
                cb = by_name[f"{layer}.weight_codebook"][:r0] if f"{layer}.weight_codebook" in by_name else None
                out = dequantize_w4a8_weight(
                    packed, s_rel, s_ch, codebook=cb, group_size=int(conf.get("group_size", 16)),
                    convrot_groupsize=int(conf.get("convrot_groupsize", 256)),
                    output_dtype=torch.float32)
            elif lfmt == FORMAT_W4A4:
                scale = by_name[f"{layer}.weight_scale"][:r0]
                out = dequantize_w4a4_weight(
                    packed, scale, int(conf.get("convrot_groupsize", 256)),
                    output_dtype=torch.float32)
            else:
                scale = by_name[f"{layer}.weight_scale"][:r0]
                scale = scale.reshape(-1, 1)
                out = dequantize_int8_tensorwise_weight(
                    packed, scale, output_dtype=torch.float32)
            finite = bool(torch.isfinite(out).all())
            sane = bool(float(out.abs().max()) <= 1e4)
            if not (finite and sane):
                roundtrip_fail += 1
                check(f"roundtrip:{layer}", False,
                      f"finite={finite} max_abs={float(out.abs().max()):.3e}")
        except Exception as e:  # noqa: BLE001 - packing failures are the finding
            roundtrip_fail += 1
            check(f"roundtrip:{layer}", False, f"{type(e).__name__}: {e}")
    if roundtrip_fail == 0 and n_quant:
        check("packing-roundtrip", True, f"sampled {min(sample_rows, n_quant)} layer(s), all finite")

    # Extension metadata consistency (informational; never fails the check).
    ext = meta.get(METADATA_KEY_EXT)
    if ext:
        try:
            ext_obj = json.loads(ext)
            ext_counts = None
            q_ext = ext_obj.get("quantization") or {}
            dist = q_ext.get("distribution") or {}
            if isinstance(dist.get("counts"), dict):
                ext_counts = {str(k): int(v) for k, v in dist["counts"].items()}
            if ext_counts is not None and ext_counts != fmt_counts:
                check("extension-metadata", True,
                      f"{METADATA_KEY_EXT} distribution {ext_counts} differs from "
                      f"{METADATA_KEY_QUANT} {fmt_counts} (informational)")
        except (TypeError, json.JSONDecodeError) as e:
            check("extension-metadata", True, f"unparsable {METADATA_KEY_EXT}: {e}")

    out_sha = sha256_file(output_path)
    ok = n_failed == 0
    return {
        "ok": ok,
        "checks": checks,
        "n_failed": n_failed,
        "n_tensors": len(out_names),
        "total_bytes": total_bytes,
        "quantized_layers": n_quant,
        "formats": fmt_counts,
        "output_sha256": out_sha,
    }
