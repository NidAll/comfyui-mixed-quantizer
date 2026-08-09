"""Mixed-precision planner: per-layer selection, promotion loop and gates."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from dataclasses import dataclass, field
import torch
from comfyui_wxa8_quantizer.constants import DEFAULT_W4A4_LINEAR_DTYPE, FORMAT_INT8, FORMAT_ORIGINAL, FORMAT_W4A4, FORMAT_W4A8, MIXED_FORMATS, W4A4_QUANT_GROUP_SIZE, W4A8_CONVROT_GROUPSIZE
from comfyui_wxa8_quantizer.errors import CompressionGateError, PolicyError, QualityGateError
from comfyui_wxa8_quantizer.formats import _pick_w4a4_convrot_group, build_hadamard, dequantize_weight_by_format, quantize_weight_by_format, quantized_format_bytes, rotate_activation
from comfyui_wxa8_quantizer.io import CheckpointInfo, CheckpointReader, TensorMeta
from comfyui_wxa8_quantizer.planning import CalibrationStats, DecisionKind, TensorDecision, TensorMetrics, _OutputErrorAccumulator, _act_quant_int4, _act_quant_int8, _simulate_quantized_chunk, compute_weight_metrics, runtime_output_rel_l2
from comfyui_wxa8_quantizer.quantize import _MetricAccumulator, _chunk_rows_for_budget, _codebook_sample_size, _gather_codebook_samples, _quant_work_bytes, _quantize_row_chunk
from comfyui_wxa8_quantizer.runtime import RuntimeCapabilities, W4A4ExecutionMode, resolve_w4a4_execution_mode, runtime_capabilities_for
from comfyui_wxa8_quantizer.utils import FLOAT_DTYPES, FP8_DTYPES, human_bytes
@dataclass(frozen=True)
class MixedProfile:
    name: str
    layer_gate: float            # per-layer error gate (relL2 fraction)
    global_gate: float           # mean error gate over selected layers
    compression_target_bpp: float  # bytes/parameter target for quantized linear bytes
    max_bf16_fraction: float     # max bf16 share of targeted linear bytes

MIXED_PROFILE_DEFAULTS = {
    # Gate defaults are set against the measured W4A8 reference: weight-only
    # relL2 of the codebook path is ~0.073 on real dims, so a balanced profile
    # keeps W4A8 as the workhorse (global gate just above 0.073), conservative
    # forces promotion toward INT8 (gate below it), and size-first admits the
    # ~0.14 W4A4 error for layers that can take it (per-layer gate 0.15).
    "balanced": MixedProfile("balanced", 0.10, 0.080, 0.90, 0.05),
    "conservative": MixedProfile("conservative", 0.05, 0.040, 1.05, 0.02),
    "size-first": MixedProfile("size-first", 0.15, 0.100, 0.75, 0.10),
}

@dataclass
class CandidateResult:
    format: str
    eligible: bool
    reason: str = ""
    estimated_bytes: int = 0
    weight_rel_l2: Optional[float] = None
    act_rel_l2: Optional[float] = None
    # reproducibility record for the W4A4 execution variant
    requested_linear_dtype: Optional[str] = None
    effective_activation_bits: Optional[int] = None
    effective_runtime_path: Optional[str] = None
    runtime_certified: bool = False
    runtime_certain: bool = False
    backend: Optional[str] = None

class MixedPlanner:
    """Evaluate per-format candidates, select cheapest acceptable per layer,
    then promote greedily (best error reduction per extra byte, with original
    precision as the final rescue) until the weighted global error gate
    passes. Quality and compression gates are HARD: a plan that cannot meet
    them raises QualityGateError / CompressionGateError instead of silently
    publishing a checkpoint that misses its targets."""

    def __init__(self, profile_name: str,
                 calibration: Optional[CalibrationStats],
                 max_mem: int, device: torch.device,
                 compute_dtype: Optional[torch.dtype],
                 disabled_formats: Sequence[str] = (),
                 layer_gate: Optional[float] = None,
                 global_gate: Optional[float] = None,
                 runtime: Optional[RuntimeCapabilities] = None,
                 compression_target_bpp: Optional[float] = None,
                 max_bf16_fraction: Optional[float] = None,
                 linear_dtype: str = DEFAULT_W4A4_LINEAR_DTYPE,
                 runtime_certificate: Optional["RuntimeCertificate"] = None):
        if profile_name not in MIXED_PROFILE_DEFAULTS:
            raise PolicyError(f"unknown mixed profile {profile_name!r}")
        self.profile = MIXED_PROFILE_DEFAULTS[profile_name]
        self.calibration = calibration
        self.max_mem = max_mem
        self.device = device
        self.compute_dtype = compute_dtype
        self.disabled = set(disabled_formats)
        self.runtime = runtime if runtime is not None \
            else runtime_capabilities_for("cpu")
        self.linear_dtype = linear_dtype
        self.runtime_certificate = runtime_certificate
        self.layer_gate = (layer_gate if layer_gate is not None
                           else self.profile.layer_gate)
        self.global_gate = (global_gate if global_gate is not None
                            else self.profile.global_gate)
        self.compression_target_bpp = (
            compression_target_bpp if compression_target_bpp is not None
            else self.profile.compression_target_bpp)
        self.max_bf16_fraction = (
            max_bf16_fraction if max_bf16_fraction is not None
            else self.profile.max_bf16_fraction)
        self.candidates: Dict[str, Dict[str, CandidateResult]] = {}
        self.promotions: List[str] = []
        self.summary: Dict[str, Any] = {}
        self._targeted: List[TensorDecision] = []

    # -- eligibility -------------------------------------------------------
    def eligible_formats(self, meta: TensorMeta) -> List[str]:
        if len(meta.shape) != 2 or meta.dtype not in FLOAT_DTYPES:
            return []
        k = int(meta.shape[1])
        out: List[str] = []
        if k % 256 == 0 and self.runtime.supports(FORMAT_W4A8):
            out.append(FORMAT_W4A8)
        if k % W4A4_QUANT_GROUP_SIZE == 0 and self.runtime.supports(FORMAT_W4A4):
            try:
                _pick_w4a4_convrot_group(k)
                out.append(FORMAT_W4A4)
            except PolicyError:
                pass
        if self.runtime.supports(FORMAT_INT8):
            out.append(FORMAT_INT8)
        return [f for f in out if f not in self.disabled]

    def runtime_reason(self, fmt: str) -> str:
        if not self.runtime.supports(fmt):
            return f"not supported on target runtime {self.runtime.target}"
        return f"{self.runtime.describe(fmt)} on {self.runtime.target}"

    def cgs_for(self, k: int, fmt: str) -> int:
        if fmt == FORMAT_W4A4:
            return _pick_w4a4_convrot_group(k)
        return W4A8_CONVROT_GROUPSIZE

    # -- evaluation --------------------------------------------------------
    def _w4a4_scoring_modes(self) -> Tuple[List[int], W4A4ExecutionMode]:
        """Activation-bit modes to score a W4A4 candidate with. Certain
        modes evaluate one value; uncertain modes evaluate BOTH and the
        caller takes the worst (max) error, so the planner never
        optimistically assumes a dispatch it cannot prove."""
        mode = self.w4a4_execution_mode_for_candidate()
        if mode.certain:
            return [mode.activation_bits], mode
        return [4, 8], mode

    def w4a4_execution_mode_for_candidate(self) -> W4A4ExecutionMode:
        """Effective W4A4 execution mode for candidate evaluation: a runtime
        certificate (observed on the target machine) wins over the static
        dispatch model. Uncertain static modes are handled by the caller
        through worst-case evaluation of both A4 and A8."""
        if (self.runtime_certificate is not None
                and self.runtime_certificate.formats.get(FORMAT_W4A4)):
            conf = self.runtime_certificate.formats[FORMAT_W4A4]
            bits = (conf.get("effective_activation_bits")
                    or conf.get("activation_bits"))
            if bits in (4, 8):
                return W4A4ExecutionMode(
                    int(bits), "certified", True,
                    "observed on the target machine "
                    f"({self.runtime_certificate.gpu or 'gpu'})")
        return resolve_w4a4_execution_mode(self.runtime, self.linear_dtype)

    def _activations(self, name: str) -> Optional[torch.Tensor]:
        if self.calibration is None:
            return None
        for key in (name, name[:-len(".weight")]):
            stats = self.calibration.layers.get(key)
            if stats is not None:
                return stats["samples"]
        return None

    def _evaluate_format(self, reader: CheckpointReader, d: TensorDecision,
                         fmt: str) -> TensorMetrics:
        meta = reader.info.by_name(d.name)
        if meta is None:
            raise PolicyError(f"planner references missing tensor {d.name!r}")
        cgs = self.cgs_for(int(meta.shape[1]), fmt)
        activations = self._activations(d.name)
        if _quant_work_bytes(meta) <= self.max_mem:
            w = reader.read_tensor(d.name)
            if self.compute_dtype is not None and w.dtype not in FP8_DTYPES:
                w = w.to(self.compute_dtype)
            if self.device.type == "cuda":
                w = w.to(self.device)
            try:
                q = quantize_weight_by_format(w, fmt, d.group_size, cgs)
                dq = dequantize_weight_by_format(q, fmt, d.group_size, cgs,
                                                 torch.float32)
                metrics = compute_weight_metrics(w.float().cpu(), dq.cpu())
                if activations is not None:
                    if fmt == FORMAT_W4A4:
                        modes, _mode = self._w4a4_scoring_modes()
                        errs = [
                            runtime_output_rel_l2(
                                w.cpu(),
                                {k: v.cpu() for k, v in q.items()}, fmt,
                                d.group_size, cgs, activations,
                                w4a4_activation_bits=bits)
                            for bits in modes
                        ]
                        metrics.act_rel_l2 = max(errs)  # worst case
                    else:
                        metrics.act_rel_l2 = runtime_output_rel_l2(
                            w.cpu(), {k: v.cpu() for k, v in q.items()}, fmt,
                            d.group_size, cgs, activations,
                            w4a4_activation_bits=8)
            finally:
                del w
            return metrics
        # bounded-memory chunked path: rowwise formats chunk exactly; w4a8
        # uses the pre-fit codebook row chunks, identical to the reference
        # path. Runtime output error accumulates per sample across chunks.
        n, k = int(meta.shape[0]), int(meta.shape[1])
        chunk_rows = _chunk_rows_for_budget(k, n, self.max_mem)
        acc = _MetricAccumulator(d.name)
        out_acc = None
        out_accs = None
        act_q = act_scale = None
        act_qs = act_scales = None
        if activations is not None:
            x = activations.float()
            if fmt in (FORMAT_W4A4, FORMAT_W4A8):
                h = build_hadamard(cgs, device=x.device, dtype=torch.float32)
                x_rot = rotate_activation(x, h, cgs)
            else:
                x_rot = x
            n_samples = int(activations.shape[0])
            if fmt == FORMAT_W4A4:
                modes, _mode = self._w4a4_scoring_modes()
                out_accs = {bits: _OutputErrorAccumulator(n_samples)
                            for bits in modes}
                act_qs = {}
                act_scales = {}
                for bits in modes:
                    if bits == 4:
                        act_qs[bits], act_scales[bits] = _act_quant_int4(x_rot)
                    else:
                        act_qs[bits], act_scales[bits] = _act_quant_int8(x_rot)
            else:
                act_q, act_scale = _act_quant_int8(x_rot)
                out_acc = _OutputErrorAccumulator(n_samples)
        if fmt == FORMAT_W4A8:
            sample_size = _codebook_sample_size(self.max_mem, n * k)
            codebook = _gather_codebook_samples(
                reader, d.name, k, d.group_size, cgs, sample_size, chunk_rows,
                compute_dtype=self.compute_dtype)
            for r0 in range(0, n, chunk_rows):
                r1 = min(n, r0 + chunk_rows)
                q = _quantize_row_chunk(reader, d.name, r0, r1, d.group_size,
                                        cgs, codebook, self.device,
                                        self.compute_dtype)
                dq = dequantize_weight_by_format(q, FORMAT_W4A8, d.group_size,
                                                 cgs, torch.float32)
                orig_chunk = reader.read_tensor(d.name)[r0:r1]
                acc.update(orig_chunk, dq, activations)
                if out_acc is not None:
                    y_q = _simulate_quantized_chunk(
                        orig_chunk, q, fmt, d.group_size, cgs, act_q,
                        act_scale, "int8")
                    y_ref = activations.float() @ orig_chunk.float().T
                    out_acc.update(y_q.cpu(), y_ref.cpu())
                del q, dq
        else:
            for r0 in range(0, n, chunk_rows):
                r1 = min(n, r0 + chunk_rows)
                chunk = reader.read_tensor(d.name)[r0:r1]
                if self.compute_dtype is not None and chunk.dtype not in FP8_DTYPES:
                    chunk = chunk.to(self.compute_dtype)
                if self.device.type == "cuda":
                    chunk = chunk.to(self.device)
                q = quantize_weight_by_format(chunk, fmt, d.group_size, cgs)
                dq = dequantize_weight_by_format(q, fmt, d.group_size, cgs,
                                                 torch.float32)
                acc.update(chunk.cpu().float(), dq.cpu(), activations)
                if out_acc is not None:
                    y_q = _simulate_quantized_chunk(
                        chunk.cpu(), {k: v.cpu() for k, v in q.items()}, fmt,
                        d.group_size, cgs, act_q, act_scale, "int8")
                    y_ref = activations.float() @ chunk.cpu().float().T
                    out_acc.update(y_q, y_ref)
                if out_accs is not None:
                    y_ref = activations.float() @ chunk.cpu().float().T
                    for bits, oacc in out_accs.items():
                        y_q = _simulate_quantized_chunk(
                            chunk.cpu(), {k: v.cpu() for k, v in q.items()},
                            fmt, d.group_size, cgs, act_qs[bits],
                            act_scales[bits],
                            "int4" if bits == 4 else "int8")
                        oacc.update(y_q, y_ref)
                del chunk, q, dq
        metrics = acc.finish()
        if out_acc is not None:
            metrics.act_rel_l2 = out_acc.finish()
        if out_accs is not None:
            metrics.act_rel_l2 = max(oacc.finish() for oacc in out_accs.values())
        return metrics

    # -- planning ----------------------------------------------------------
    def plan(self, info: CheckpointInfo,
             decisions: List[TensorDecision]) -> Dict[str, Any]:
        self._targeted = [d for d in decisions
                          if d.kind == DecisionKind.QUANTIZE]
        with CheckpointReader(info) as reader:
            for d in self._targeted:
                meta = info.by_name(d.name)
                if meta is None:
                    raise PolicyError(
                        f"mixed planner references missing tensor {d.name!r}")
                n, k = int(meta.shape[0]), int(meta.shape[1])
                cands: Dict[str, CandidateResult] = {}
                w4a4_mode = self.w4a4_execution_mode_for_candidate()
                for fmt in self.eligible_formats(meta):
                    mode = (w4a4_mode if fmt == FORMAT_W4A4 else None)
                    cand = CandidateResult(
                        format=fmt, eligible=True,
                        estimated_bytes=quantized_format_bytes(n, k, fmt),
                        reason=self.runtime_reason(fmt),
                        requested_linear_dtype=(
                            self.linear_dtype if fmt == FORMAT_W4A4 else None),
                        effective_activation_bits=(
                            mode.activation_bits if mode is not None and
                            mode.certain else None),
                        effective_runtime_path=(
                            mode.path if mode is not None else None),
                        runtime_certified=(
                            self.runtime_certificate is not None),
                        runtime_certain=(
                            mode.certain if mode is not None else None),
                        backend=(self.runtime.target if fmt == FORMAT_W4A4
                                 else None))
                    metrics = self._evaluate_format(reader, d, fmt)
                    cand.weight_rel_l2 = metrics.rel_l2
                    cand.act_rel_l2 = metrics.act_rel_l2
                    cands[fmt] = cand
                # record every eligible format even when the runtime cannot
                # run it, so the report explains why it was skipped
                for fmt in (FORMAT_W4A4, FORMAT_W4A8, FORMAT_INT8):
                    if fmt in cands:
                        continue
                    if fmt in self.disabled:
                        cands[fmt] = CandidateResult(
                            format=fmt, eligible=False,
                            reason="disabled by CLI flag",
                            estimated_bytes=quantized_format_bytes(n, k, fmt))
                    elif not self.runtime.supports(fmt):
                        cands[fmt] = CandidateResult(
                            format=fmt, eligible=False,
                            reason=self.runtime_reason(fmt),
                            estimated_bytes=quantized_format_bytes(n, k, fmt))
                    else:
                        cands[fmt] = CandidateResult(
                            format=fmt, eligible=False,
                            reason="shape ineligible",
                            estimated_bytes=quantized_format_bytes(n, k, fmt))
                # original precision is a real candidate: zero error, source
                # bytes. Selection only picks it when every quantized format
                # fails the layer gate; promotion may also upgrade to it.
                cands[FORMAT_ORIGINAL] = CandidateResult(
                    format=FORMAT_ORIGINAL, eligible=True,
                    reason="original precision (source bytes, zero error)",
                    estimated_bytes=meta.nbytes,
                    weight_rel_l2=0.0, act_rel_l2=0.0)
                self.candidates[d.name] = cands

        selected, kept = self.select(info, decisions)
        promoted = self.promote(info, decisions) if self.global_gate is not None else []
        self._enforce_gates(info, decisions)
        self.summary = self._summarize(info, decisions)
        self.summary.update({
            "selected": selected, "kept": kept, "promotions": promoted,
            "runtime_backend": self.runtime.target,
            "w4a4_linear_dtype": self.linear_dtype,
        })
        return self.summary

    def _error_of(self, cand: CandidateResult) -> Optional[float]:
        if cand is None:
            return None
        return (cand.act_rel_l2 if cand.act_rel_l2 is not None
                else cand.weight_rel_l2)

    def select(self, info: CheckpointInfo,
               decisions: List[TensorDecision]) -> Tuple[int, int]:
        selected = 0
        kept = 0
        for d in self._targeted:
            cands = self.candidates.get(d.name, {})
            ordered = sorted(
                (c for c in cands.values() if c.eligible),
                key=lambda c: (c.estimated_bytes,
                                c.format != FORMAT_INT8))
            chosen = None
            for cand in ordered:
                err = self._error_of(cand)
                if err is not None and err <= self.layer_gate:
                    chosen = cand
                    break
            if chosen is None:
                # unreachable: ORIGINAL (error 0) always satisfies the gate
                d.kind = DecisionKind.KEEP_PRECISION
                d.reason = ("mixed planner: no candidate within quality gate "
                            f"(gate {self.layer_gate})")
                kept += 1
                continue
            meta = info.by_name(d.name)
            if chosen.format == FORMAT_ORIGINAL:
                d.kind = DecisionKind.KEEP_PRECISION
                d.format = FORMAT_ORIGINAL
                d.reason = ("mixed planner: original precision (no quantized "
                            "format within quality gate "
                            f"{self.layer_gate})")
                kept += 1
                continue
            d.format = chosen.format
            d.convrot_groupsize = self.cgs_for(int(meta.shape[1]), chosen.format)
            d.linear_dtype = (self.linear_dtype if chosen.format == FORMAT_W4A4
                              else None)
            d.reason = f"mixed planner: {chosen.format}"
            selected += 1
        return selected, kept

    def _layer_params(self, info: CheckpointInfo, d: TensorDecision) -> int:
        meta = info.by_name(d.name)
        if meta is None or len(meta.shape) != 2:
            return 0
        return int(meta.shape[0]) * int(meta.shape[1])

    def targeted_weighted_error(self, info: CheckpointInfo,
                                decisions: List[TensorDecision]) -> Optional[float]:
        """Param-weighted mean error over the ENTIRE targeted set.

        Layers kept at original precision contribute error 0 but their
        parameters stay in the denominator, so the metric represents the
        whole targeted linear payload, not only the quantized subset."""
        num = 0.0
        den = 0
        for d in self._targeted:
            w = self._layer_params(info, d)
            if w == 0:
                continue
            if d.kind == DecisionKind.QUANTIZE:
                cand = self.candidates.get(d.name, {}).get(d.format)
                err = self._error_of(cand)
                if err is None:
                    continue
                num += err * w
            # KEEP_PRECISION layers contribute error 0 but stay in the mean
            den += w
        if den == 0:
            return None
        return num / den

    def quantized_weighted_error(self, info: CheckpointInfo,
                                 decisions: List[TensorDecision]) -> Optional[float]:
        """Diagnostic: param-weighted mean error over the quantized subset
        only (excludes original-precision layers from the denominator)."""
        num = 0.0
        den = 0
        for d in decisions:
            if d.kind != DecisionKind.QUANTIZE:
                continue
            cand = self.candidates.get(d.name, {}).get(d.format)
            err = self._error_of(cand)
            if err is None:
                continue
            w = self._layer_params(info, d)
            num += err * w
            den += w
        if den == 0:
            return None
        return num / den

    def global_mean_error(self, info: CheckpointInfo,
                          decisions: List[TensorDecision]) -> Optional[float]:
        """The optimizer/report metric: targeted_weighted_error (BF16 layers
        included with error 0)."""
        return self.targeted_weighted_error(info, decisions)

    def promote(self, info: CheckpointInfo,
                decisions: List[TensorDecision]) -> List[str]:
        """Greedy: repeatedly promote the layer with the best error reduction
        per extra byte until the weighted global gate passes or no promotion
        helps. Original precision (BF16/FP16) is a real candidate, so a
        sensitive layer can be rescued all the way to its source format."""
        while True:
            mean = self.global_mean_error(info, decisions)
            if mean is None or mean <= self.global_gate:
                break
            best: Optional[Tuple[float, TensorDecision, CandidateResult]] = None
            for d in self._targeted:
                if d.kind != DecisionKind.QUANTIZE:
                    continue
                meta = info.by_name(d.name)
                n, k = int(meta.shape[0]), int(meta.shape[1])
                cur = self.candidates.get(d.name, {}).get(d.format)
                cur_err = self._error_of(cur)
                if cur_err is None:
                    continue
                cur_bytes = quantized_format_bytes(n, k, d.format)
                # all upgrades, original precision included as a candidate:
                # the optimizer picks whichever gives the best marginal
                # error reduction per byte (W4A4->W4A8, W4A4->INT8,
                # W4A8->INT8, any -> original)
                for fmt, cand in self.candidates.get(d.name, {}).items():
                    if fmt == d.format or not cand.eligible:
                        continue
                    cand_err = self._error_of(cand)
                    if cand_err is None or cand_err >= cur_err:
                        continue
                    extra = max(cand.estimated_bytes - cur_bytes, 1)
                    ratio = (cur_err - cand_err) / float(extra)
                    if best is None or ratio > best[0]:
                        best = (ratio, d, cand)
            if best is None:
                break
            _, d, cand = best
            if cand.format == FORMAT_ORIGINAL:
                d.kind = DecisionKind.KEEP_PRECISION
                d.format = FORMAT_ORIGINAL
                d.reason = "mixed planner (promoted): original precision"
            else:
                meta = info.by_name(d.name)
                d.format = cand.format
                d.convrot_groupsize = self.cgs_for(
                    int(meta.shape[1]), cand.format)
                d.linear_dtype = (self.linear_dtype
                                  if cand.format == FORMAT_W4A4 else None)
                d.reason = f"mixed planner (promoted): {cand.format}"
            # measure AFTER applying, so the log shows the new mean
            mean_after = self.global_mean_error(info, decisions)
            mean_txt = ("n/a" if mean_after is None
                        else f"{mean_after:.4f}")
            self.promotions.append(
                f"{d.name}:{cand.format} (weighted error -> {mean_txt})")
        return self.promotions

    # -- hard gates --------------------------------------------------------
    def _top_contributors(self, info: CheckpointInfo, key: Callable[
            [TensorDecision], float], limit: int = 3) -> List[str]:
        ranked = sorted(self._targeted, key=key, reverse=True)[:limit]
        out = []
        for d in ranked:
            meta = info.by_name(d.name)
            out.append(f"{d.name} ({human_bytes(meta.nbytes)})")
        return out

    def _enforce_gates(self, info: CheckpointInfo,
                       decisions: List[TensorDecision]) -> None:
        if not any(d.kind == DecisionKind.QUANTIZE
                   for d in self._targeted):
            raise QualityGateError(
                "quality gate failed: no candidate layer could be quantized "
                "within the configured gates (layer gate "
                f"{self.layer_gate}); refusing to publish a passthrough-only "
                "checkpoint. Loosen --quality-gate or the profile.")
        final_error = self.global_mean_error(info, decisions)
        if (final_error is not None and self.global_gate is not None
                and final_error > self.global_gate):
            worst = self._top_contributors(
                info, key=lambda d: (self._error_of(
                    self.candidates.get(d.name, {}).get(d.format)) or 0.0)
                    * self._layer_params(info, d))
            quant_bits = sorted({
                self.candidates.get(d.name, {}).get(d.format).format
                for d in self._targeted
                if d.kind == DecisionKind.QUANTIZE
                and self.candidates.get(d.name, {}).get(d.format) is not None})
            raise QualityGateError(
                f"quality gate failed: targeted weighted mean error "
                f"{final_error:.6f} > global gate {self.global_gate:.6f} "
                f"(profile {self.profile.name}). All remaining quantized "
                f"formats: {', '.join(quant_bits) or 'none'}. Largest error "
                f"contributors: {', '.join(worst)}. Raise "
                "--global-error-gate, provide calibration for "
                "activation-aware gates, or use a larger profile.")
        quant_bytes = 0
        kept_bytes = 0
        params = 0
        kept_params = 0
        kept_by_name: List[Tuple[str, int]] = []
        for d in self._targeted:
            meta = info.by_name(d.name)
            n, k = int(meta.shape[0]), int(meta.shape[1])
            params += n * k
            if d.kind == DecisionKind.QUANTIZE:
                quant_bytes += quantized_format_bytes(n, k, d.format)
            else:
                kept_bytes += meta.nbytes
                kept_params += n * k
                kept_by_name.append((d.name, meta.nbytes))
        effective_bpp = (quant_bytes + kept_bytes) / max(params, 1)
        original_byte_fraction = kept_bytes / max(quant_bytes + kept_bytes, 1)
        original_param_fraction = kept_params / max(params, 1)
        if self.compression_target_bpp is not None \
                and effective_bpp > self.compression_target_bpp:
            kept_share = 100 * original_byte_fraction
            top_kept = ", ".join(name for name, _ in
                                 sorted(kept_by_name, key=lambda kv: -kv[1])[:3])
            raise CompressionGateError(
                f"compression gate failed: effective targeted payload "
                f"{effective_bpp:.4f} bytes/parameter > target "
                f"{self.compression_target_bpp:.4f} (profile "
                f"{self.profile.name}). {kept_share:.1f}% of the targeted "
                f"output bytes are at original precision "
                f"({100*original_param_fraction:.1f}% of parameters); "
                f"largest kept layers: {top_kept or 'none'}. Raise "
                "--max-linear-bytes-per-param or "
                "--max-original-byte-fraction, or loosen the quality gates.")
        if original_byte_fraction > self.max_bf16_fraction:
            top_kept = ", ".join(name for name, _ in
                                 sorted(kept_by_name, key=lambda kv: -kv[1])[:3])
            raise CompressionGateError(
                f"compression gate failed: "
                f"{100*original_byte_fraction:.1f}% of the targeted output "
                f"bytes kept at original precision > "
                f"{100*self.max_bf16_fraction:.1f}% (profile "
                f"{self.profile.name}); {100*original_param_fraction:.1f}% "
                f"of parameters. Largest kept layers: {top_kept or 'none'}. "
                "Raise --max-original-byte-fraction or loosen the quality "
                "gates.")
        self._effective_bpp = effective_bpp
        self._original_byte_fraction = original_byte_fraction
        self._original_param_fraction = original_param_fraction

    def _summarize(self, info: CheckpointInfo,
                   decisions: List[TensorDecision]) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        layer_params: Dict[str, int] = {}
        layer_bytes: Dict[str, int] = {}
        kept_params = 0
        kept_bytes = 0
        for d in self._targeted:
            meta = info.by_name(d.name)
            n, k = int(meta.shape[0]), int(meta.shape[1])
            if d.kind == DecisionKind.QUANTIZE:
                counts[d.format] = counts.get(d.format, 0) + 1
                layer_params[d.format] = layer_params.get(d.format, 0) + n * k
                layer_bytes[d.format] = layer_bytes.get(
                    d.format, 0) + quantized_format_bytes(n, k, d.format)
            else:
                kept_params += n * k
                kept_bytes += meta.nbytes
        mean = self.targeted_weighted_error(info, decisions)
        q_mean = self.quantized_weighted_error(info, decisions)
        return {
            "profile": self.profile.name,
            "layer_gate": self.layer_gate,
            "global_gate": self.global_gate,
            "global_mean_error": mean,
            "targeted_weighted_error": mean,
            "quantized_weighted_error": q_mean,
            "counts": counts,
            "layer_params": layer_params,
            "layer_bytes": layer_bytes,
            "kept_params": kept_params,
            "kept_bytes": kept_bytes,
            "effective_bpp": getattr(self, "_effective_bpp", None),
            "original_precision_parameter_fraction": getattr(
                self, "_original_param_fraction", None),
            "original_precision_output_byte_fraction": getattr(
                self, "_original_byte_fraction", None),
            "compression_target_bpp": self.compression_target_bpp,
            "max_bf16_fraction": self.max_bf16_fraction,
            "candidates": {
                name: {
                    fmt: {
                        "eligible": cand.eligible,
                        "reason": cand.reason,
                        "bytes": cand.estimated_bytes,
                        "weight_rel_l2": cand.weight_rel_l2,
                        "act_rel_l2": cand.act_rel_l2,
                        "requested_linear_dtype": cand.requested_linear_dtype,
                        "effective_activation_bits": cand.effective_activation_bits,
                        "effective_runtime_path": cand.effective_runtime_path,
                        "runtime_certified": cand.runtime_certified,
                        "runtime_certain": cand.runtime_certain,
                        "backend": cand.backend,
                    } for fmt, cand in cands.items()
                } for name, cands in self.candidates.items()
            },
            "runtime": {
                "backend": self.runtime.target,
                "gpu_name": self.runtime.gpu_name,
                "cuda_capability": list(self.runtime.cuda_capability)
                if self.runtime.cuda_capability else None,
                "rocm_arch": self.runtime.rocm_arch,
                "runtime_certified": self.runtime.runtime_certified,
                "formats": {
                    fmt: {
                        "loadable": self.runtime.capability(fmt).loadable,
                        "executable": self.runtime.capability(fmt).executable,
                        "accelerated": self.runtime.capability(fmt).accelerated,
                        "certified": self.runtime.capability(fmt).certified,
                        "backend": self.runtime.capability(fmt).backend,
                        "reason": self.runtime.capability(fmt).reason,
                        "status": self.runtime.describe(fmt),
                    } for fmt in MIXED_FORMATS
                },
            },
        }
