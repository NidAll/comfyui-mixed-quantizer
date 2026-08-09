"""Bounded-memory quantization and the sensitivity prepass."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from dataclasses import dataclass, field
import math
import torch
from comfyui_wxa8_quantizer.constants import FORMAT_W4A8
from comfyui_wxa8_quantizer.errors import PolicyError
from comfyui_wxa8_quantizer.formats import assign_codes, assign_grid, build_hadamard, dequantize_weight_by_format, fit_codebook, format_scale_suffixes, quantize_weight_by_format, rotate_int8_convrot_weight
from comfyui_wxa8_quantizer.io import CheckpointInfo, CheckpointReader, TensorMeta
from comfyui_wxa8_quantizer.logging_utils import log
from comfyui_wxa8_quantizer.planning import DecisionKind, SensitivityAnalyzer, TensorDecision, TensorMetrics
from comfyui_wxa8_quantizer.utils import FP8_DTYPES, human_bytes
QUANT_WORK_BYTES_PER_ELEMENT = 48

MIN_CHUNK_MEMORY = 32 * 1024 * 1024

def _quant_work_bytes(meta: TensorMeta) -> int:
    return math.prod(meta.shape) * QUANT_WORK_BYTES_PER_ELEMENT

def _chunk_rows_for_budget(k: int, n: int, max_mem: int) -> int:
    if max_mem < MIN_CHUNK_MEMORY:
        raise PolicyError(
            f"--max-memory must be at least {human_bytes(MIN_CHUNK_MEMORY)} for "
            "Lloyd-Max W4A8 quantization")
    row_work = k * QUANT_WORK_BYTES_PER_ELEMENT
    if row_work > max_mem:
        raise PolicyError(
            f"--max-memory {human_bytes(max_mem)} cannot hold one {k}-element "
            f"working row (estimated {human_bytes(row_work)})")
    return max(1, min(n, max_mem // row_work))

def _codebook_sample_size(max_mem: int, total_elements: int) -> int:
    # Lloyd-Max temporarily materializes distances/assignments.  Keep that
    # working set below roughly half the user budget.
    budgeted = max(4096, max_mem // 128)
    return min(300000, total_elements, budgeted)

def _quantize_rotated_w4a8_with_codebook(weight: torch.Tensor, group_size: int,
                                         codebook: torch.Tensor,
                                         scale_dtype: torch.dtype = torch.float8_e4m3fn,
                                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """W4A8 quantization of a rotated weight chunk with a PRE-FIT codebook."""
    n, k = weight.shape
    groups = k // group_size
    grouped = weight.float().view(n, groups, group_size)
    group_scale = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    quantized = assign_codes(grouped / group_scale, codebook)
    for _ in range(3):
        qc = codebook[quantized]
        group_scale = ((grouped * qc).sum(-1, keepdim=True)
                       / (qc * qc).sum(-1, keepdim=True).clamp(min=1e-8)).clamp(min=1e-8)
        quantized = assign_codes(grouped / group_scale, codebook)
    shifted = codebook[quantized] * group_scale
    s_channel = (shifted.abs().amax(dim=(1, 2)) / 127.0).clamp(min=1e-8)
    s_rel = (group_scale.squeeze(-1) / s_channel.unsqueeze(1)).float().contiguous()
    if scale_dtype != torch.float32:
        s_rel = s_rel.to(scale_dtype).contiguous()
    levels = (codebook.view(1, 1, 16) * s_rel.float().unsqueeze(-1)).round_().clamp_(-127, 127)
    unsigned = assign_grid(grouped, levels, s_channel).view(n, k)
    packed = ((unsigned[:, 0::2] & 0xF) | ((unsigned[:, 1::2] & 0xF) << 4)).to(torch.int8).contiguous()
    return packed, s_rel, s_channel.float().contiguous(), codebook

def _gather_codebook_samples(reader: CheckpointReader, name: str, k: int,
                             group_size: int, convrot_groupsize: int,
                             sample_size: int, chunk_rows: int,
                             device: Any = "cpu",
                             compute_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Deterministic subsample of normalized rotated weights for codebook fitting.

    Draws `sample_size` flattened indices with the reference seed-0 generator,
    then gathers those elements chunk by chunk.  Used only when a tensor does
    not fit in the memory budget; identical distribution to the reference path.
    """
    meta = reader.info.by_name(name)
    n = int(meta.shape[0])
    total = n * k
    if total > sample_size:
        gen = torch.Generator(device="cpu").manual_seed(0)
        idx = torch.randint(0, total, (sample_size,), device="cpu", generator=gen)
    else:
        idx = torch.arange(total, device="cpu")
    work_dtype = compute_dtype or torch.float32
    h = build_hadamard(convrot_groupsize, device="cpu", dtype=work_dtype)
    h_t = h.T
    samples = torch.empty(idx.numel(), dtype=torch.float32)
    n_conv_groups = k // convrot_groupsize
    n_quant_groups = k // group_size
    for r0 in range(0, n, chunk_rows):
        r1 = min(n, r0 + chunk_rows)
        chunk = reader.read_tensor(name)[r0:r1].to(work_dtype)   # [rows, k]
        rot = torch.matmul(
            chunk.view(r1 - r0, n_conv_groups, convrot_groupsize), h_t
        ).reshape(r1 - r0, k)
        # per-group normalization (identical to the in-memory reference path)
        grouped = rot.float().view(r1 - r0, n_quant_groups, group_size)
        gs = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        normalized = (grouped / gs).reshape(r1 - r0, k)
        start = r0 * k
        end = r1 * k
        mask = (idx >= start) & (idx < end)
        if mask.any():
            local = idx[mask] - start
            samples[mask] = normalized.flatten()[local]
    if samples.numel() == 0:
        raise PolicyError("codebook sampling produced no elements")
    return fit_codebook(samples, levels=16, iterations=25,
                        sample_size=sample_size).contiguous()

def quantize_tensor_bounded(reader: CheckpointReader, name: str, fmt: str,
                            group_size: int, convrot_groupsize: int,
                            max_mem: int, device: Any,
                            compute_dtype: Optional[torch.dtype] = None,
                            ) -> Dict[str, torch.Tensor]:
    """Quantize one tensor with a bounded working set (chunked when needed).

    compute_dtype: fp16/bf16 lowers the precision of the rotation + fit math
    (deviation from the fp32 reference path; recorded in metadata).
    """
    device = torch.device(device)
    meta = reader.info.by_name(name)
    n, k = int(meta.shape[0]), int(meta.shape[1])
    full_bytes = meta.nbytes
    work_bytes = _quant_work_bytes(meta)
    if work_bytes <= max_mem:
        w = reader.read_tensor(name)
        if compute_dtype is not None and w.dtype not in FP8_DTYPES:
            w = w.to(compute_dtype)
        if device.type == "cuda":
            w = w.to(device)
        try:
            out = quantize_weight_by_format(w, fmt, group_size, convrot_groupsize)
        finally:
            del w
        if device.type != "cpu":
            out = {kk: vv.cpu() for kk, vv in out.items()}
        return out

    log().info("chunked quantization for %s (%s > budget %s)",
               name, human_bytes(full_bytes), human_bytes(max_mem))
    chunk_rows = _chunk_rows_for_budget(k, n, max_mem)
    if fmt != FORMAT_W4A8:
        # rowwise formats chunk exactly: no codebook, no cross-row state
        parts: Dict[str, List[torch.Tensor]] = {
            s: [] for s in format_scale_suffixes(fmt)}
        for r0 in range(0, n, chunk_rows):
            r1 = min(n, r0 + chunk_rows)
            chunk = reader.read_tensor(name)[r0:r1]
            if compute_dtype is not None and chunk.dtype not in FP8_DTYPES:
                chunk = chunk.to(compute_dtype)
            if device.type == "cuda":
                chunk = chunk.to(device)
            part = quantize_weight_by_format(chunk, fmt, group_size,
                                             convrot_groupsize)
            for s in parts:
                t = part[s]
                parts[s].append(t.cpu() if device.type != "cpu" else t)
            del chunk, part
        return {s: torch.cat(v, dim=0).contiguous() for s, v in parts.items()}
    sample_size = _codebook_sample_size(max_mem, n * k)
    codebook = _gather_codebook_samples(reader, name, k, group_size,
                                        convrot_groupsize, sample_size, chunk_rows,
                                        device="cpu", compute_dtype=compute_dtype)
    # per-chunk processing
    packed_parts: List[torch.Tensor] = []
    s_rel_parts: List[torch.Tensor] = []
    s_ch_parts: List[torch.Tensor] = []
    for r0 in range(0, n, chunk_rows):
        r1 = min(n, r0 + chunk_rows)
        chunk = reader.read_tensor(name)[r0:r1]
        if compute_dtype is not None and chunk.dtype not in FP8_DTYPES:
            chunk = chunk.to(compute_dtype)
        if device.type == "cuda":
            chunk = chunk.to(device)
        rot = rotate_int8_convrot_weight(chunk, convrot_groupsize)
        p, s_rel, s_ch, _ = _quantize_rotated_w4a8_with_codebook(
            rot, group_size, codebook.to(device=rot.device))
        packed_parts.append(p.cpu() if device.type != "cpu" else p)
        s_rel_parts.append(s_rel.cpu() if device.type != "cpu" else s_rel)
        s_ch_parts.append(s_ch.cpu() if device.type != "cpu" else s_ch)
        del chunk, rot, p, s_rel, s_ch
    packed = torch.cat(packed_parts, dim=0).contiguous()
    s_rel = torch.cat(s_rel_parts, dim=0).contiguous()
    s_ch = torch.cat(s_ch_parts, dim=0).contiguous()
    return {"": packed, "_s_rel": s_rel, "_s_channel": s_ch, "_codebook": codebook.contiguous()}

def _quantize_row_chunk(reader: CheckpointReader, name: str, r0: int, r1: int,
                        group_size: int, convrot_groupsize: int,
                        codebook: torch.Tensor, device: torch.device,
                        compute_dtype: Optional[torch.dtype]) -> Dict[str, torch.Tensor]:
    chunk = reader.read_tensor(name)[r0:r1]
    if compute_dtype is not None and chunk.dtype not in FP8_DTYPES:
        chunk = chunk.to(compute_dtype)
    if device.type == "cuda":
        chunk = chunk.to(device)
    rotated = rotate_int8_convrot_weight(chunk, convrot_groupsize)
    packed, s_rel, s_ch, cb = _quantize_rotated_w4a8_with_codebook(
        rotated, group_size, codebook.to(device=rotated.device))
    out = {
        "": packed.cpu(),
        "_s_rel": s_rel.cpu(),
        "_s_channel": s_ch.cpu(),
        "_codebook": cb.cpu(),
    }
    del chunk, rotated, packed, s_rel, s_ch, cb
    return out

@dataclass
class _MetricAccumulator:
    name: str
    signal: float = 0.0
    error: float = 0.0
    dot: float = 0.0
    reconstructed: float = 0.0
    act_num_sq: Optional[torch.Tensor] = None
    act_den_sq: Optional[torch.Tensor] = None

    def update(self, original: torch.Tensor, dequant: torch.Tensor,
               activations: Optional[torch.Tensor]) -> None:
        original = original.float()
        dequant = dequant.float()
        delta = dequant - original
        self.signal += float(original.square().sum())
        self.error += float(delta.square().sum())
        self.dot += float((original * dequant).sum())
        self.reconstructed += float(dequant.square().sum())
        if activations is not None:
            x = activations.float()
            num = delta @ x.t()
            den = original @ x.t()
            part_num = num.square().sum(dim=0).cpu()
            part_den = den.square().sum(dim=0).cpu()
            self.act_num_sq = part_num if self.act_num_sq is None else self.act_num_sq + part_num
            self.act_den_sq = part_den if self.act_den_sq is None else self.act_den_sq + part_den

    def finish(self) -> TensorMetrics:
        if not all(math.isfinite(value) for value in (
                self.signal, self.error, self.dot, self.reconstructed)):
            return TensorMetrics(
                self.name, 1e30, -300.0, -1.0,
                act_rel_l2=1e30 if self.act_num_sq is not None else None)
        rel_l2 = math.sqrt(self.error / max(self.signal, 1e-12))
        snr_db = 300.0 if self.error <= 1e-30 else 10.0 * math.log10(
            max(self.signal, 1e-30) / self.error)
        if self.signal <= 1e-30 and self.reconstructed <= 1e-30:
            cosine = 1.0
        else:
            denom = math.sqrt(max(self.signal * self.reconstructed, 1e-30))
            cosine = max(-1.0, min(1.0, self.dot / denom))
        act_rel_l2 = None
        if self.act_num_sq is not None and self.act_den_sq is not None:
            act_rel_l2 = float(
                (self.act_num_sq.sqrt() / self.act_den_sq.clamp(min=1e-16).sqrt()).mean())
            if not math.isfinite(act_rel_l2):
                act_rel_l2 = 1e30
        return TensorMetrics(self.name, rel_l2, snr_db, cosine,
                             act_rel_l2=act_rel_l2)

def apply_sensitivity_prepass(info: CheckpointInfo,
                              decisions: List[TensorDecision],
                              analyzer: SensitivityAnalyzer,
                              max_mem: int, device: torch.device,
                              compute_dtype: Optional[torch.dtype]) -> None:
    """Freeze sensitivity decisions before the safetensors inventory is built."""
    with CheckpointReader(info) as reader:
        for decision in decisions:
            if decision.kind != DecisionKind.QUANTIZE:
                continue
            meta = info.by_name(decision.name)
            if meta is None:
                raise PolicyError(
                    f"sensitivity plan references missing tensor {decision.name!r}")
            activations = None
            if analyzer.calibration is not None:
                layer_stats = analyzer.calibration.layers.get(decision.name)
                if layer_stats is not None:
                    activations = layer_stats["samples"]
            if _quant_work_bytes(meta) <= max_mem:
                quantized = quantize_tensor_bounded(
                    reader, decision.name, FORMAT_W4A8, decision.group_size,
                    decision.convrot_groupsize, max_mem, device,
                    compute_dtype=compute_dtype)
                dequant = dequantize_weight_by_format(
                    quantized, FORMAT_W4A8, decision.group_size,
                    decision.convrot_groupsize, torch.float32)
                metrics = analyzer.evaluate(
                    decision.name, reader.read_tensor(decision.name), dequant)
                del quantized, dequant
            else:
                n, k = int(meta.shape[0]), int(meta.shape[1])
                chunk_rows = _chunk_rows_for_budget(k, n, max_mem)
                sample_size = _codebook_sample_size(max_mem, n * k)
                codebook = _gather_codebook_samples(
                    reader, decision.name, k, decision.group_size,
                    decision.convrot_groupsize, sample_size, chunk_rows,
                    compute_dtype=compute_dtype)
                accumulator = _MetricAccumulator(decision.name)
                for r0 in range(0, n, chunk_rows):
                    r1 = min(n, r0 + chunk_rows)
                    quantized = _quantize_row_chunk(
                        reader, decision.name, r0, r1, decision.group_size,
                        decision.convrot_groupsize, codebook, device, compute_dtype)
                    dequant = dequantize_weight_by_format(
                        quantized, FORMAT_W4A8, decision.group_size,
                        decision.convrot_groupsize, torch.float32)
                    accumulator.update(reader.read_tensor(decision.name)[r0:r1],
                                       dequant, activations)
                    del quantized, dequant
                metrics = accumulator.finish()
                analyzer.results[decision.name] = metrics
            keep, reason = analyzer.decide_keep(metrics)
            metrics.kept = keep
            metrics.reason = reason
            if keep:
                decision.kind = DecisionKind.KEEP_PRECISION
                decision.reason = f"sensitivity fallback: {reason}"
