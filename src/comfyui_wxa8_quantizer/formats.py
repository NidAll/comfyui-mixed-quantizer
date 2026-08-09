"""Quantization codecs: W4A8, W4A4, INT8 (quantize/dequantize, ConvRot, codebooks)."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import torch
from comfyui_wxa8_quantizer.constants import FORMAT_INT8, FORMAT_W4A4, FORMAT_W4A8, INT8_SCALE_MAX, W4A4_EMISSION_MAX, W4A4_QUANT_GROUP_SIZE
from comfyui_wxa8_quantizer.errors import PolicyError
from comfyui_wxa8_quantizer.io import _HADAMARD_CACHE
from comfyui_wxa8_quantizer.utils import FLOAT_DTYPES
def _is_power_of_four(value: int) -> bool:
    if value < 1:
        return False
    while value % 4 == 0:
        value //= 4
    return value == 1

def build_hadamard(size: int, device: Any = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Normalized REGULAR orthogonal Hadamard matrix (ConvRot), size = power of 4."""
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    key = (size, dev, dtype)
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached
    if size < 4 or not _is_power_of_four(size):
        raise PolicyError(f"Regular Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype, device=dev)
    h = h4
    current = 4
    while current < size:
        h = torch.kron(h, h4)
        current *= 4
    h_norm = h / (size ** 0.5)
    _HADAMARD_CACHE[key] = h_norm
    return h_norm

def rotate_weight(weight: torch.Tensor, h: torch.Tensor, group_size: int) -> torch.Tensor:
    """W_rot = W @ H_block^T (offline weight rotation)."""
    out_f, in_f = weight.shape
    if in_f % group_size != 0:
        raise PolicyError(f"in_features {in_f} not divisible by group_size {group_size}")
    n_groups = in_f // group_size
    grouped = weight.reshape(out_f, n_groups, group_size)
    h_t = h.T.to(dtype=weight.dtype, device=weight.device)
    return torch.matmul(grouped, h_t).reshape(out_f, in_f)

def rotate_activation(x: torch.Tensor, h: torch.Tensor, group_size: int) -> torch.Tensor:
    """x_rot = x @ H_block (online activation rotation)."""
    orig_shape = x.shape
    features = orig_shape[-1]
    if features % group_size != 0:
        raise PolicyError(f"features {features} not divisible by group_size {group_size}")
    n_groups = features // group_size
    grouped = x.reshape(-1, n_groups, group_size)
    hh = h.to(dtype=x.dtype, device=x.device)
    return torch.matmul(grouped, hh).reshape(orig_shape)

def rotate_int8_convrot_weight(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """Portable ConvRot weight rotation (identical to comfy-kitchen)."""
    h = build_hadamard(group_size, device=weight.device, dtype=weight.dtype)
    return rotate_weight(weight, h, group_size)

def validate_w4_shape(k: int, group_size: int, convrot_groupsize: int) -> None:
    if k % 16 != 0:
        raise PolicyError(f"K={k} must be divisible by 16 for 4-bit packing")
    if k % group_size != 0:
        raise PolicyError(f"K={k} must be divisible by group_size={group_size}")
    if k % convrot_groupsize != 0:
        raise PolicyError(f"K={k} must be divisible by convrot_groupsize={convrot_groupsize}")
    if group_size < 4:
        raise PolicyError(f"group_size must be >= 4, got {group_size}")
    if (16 % group_size != 0) and (group_size % 16 != 0):
        raise PolicyError(
            f"group_size must divide 16 or be a multiple of 16, got {group_size}")
    if convrot_groupsize < 4 or not _is_power_of_four(convrot_groupsize):
        raise PolicyError(
            f"convrot_groupsize must be a power of 4, got {convrot_groupsize}")

def w4_weight_is_quantizable(shape: Sequence[int], dtype: torch.dtype,
                             group_size: int, convrot_groupsize: int) -> Tuple[bool, str]:
    if len(shape) != 2:
        return False, f"not 2D (shape {tuple(shape)})"
    if dtype not in FLOAT_DTYPES:
        return False, f"not a float dtype ({dtype})"
    k = int(shape[1])
    try:
        validate_w4_shape(k, group_size, convrot_groupsize)
    except PolicyError as e:
        return False, str(e)
    return True, "ok"

def fit_codebook(normalized: torch.Tensor, levels: int = 16, iterations: int = 25,
                 sample_size: int = 300000) -> torch.Tensor:
    """Data-free Lloyd-Max codebook on normalized rotated weights.

    Deterministic: subsampling (when needed) uses a fixed-seed generator, exactly
    like the reference implementation.
    """
    samples = normalized.flatten()
    if samples.numel() > sample_size:
        generator = torch.Generator(device=samples.device).manual_seed(0)
        indices = torch.randint(0, samples.numel(), (sample_size,),
                                device=samples.device, generator=generator)
        samples = samples[indices]
    samples = samples.float()
    codebook = torch.quantile(samples, torch.linspace(0, 1, levels, device=samples.device))
    for _ in range(iterations):
        assignments = (samples.unsqueeze(-1) - codebook).abs().argmin(-1)
        updated = codebook.clone()
        for index in range(levels):
            selected = assignments == index
            if selected.any():
                updated[index] = samples[selected].mean()
        codebook = updated
    return codebook.contiguous()

def assign_codes(normalized: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Nearest codebook index per element (int32)."""
    best = (normalized - codebook[0]).abs()
    indices = torch.zeros_like(normalized, dtype=torch.int32)
    for index in range(1, codebook.numel()):
        distance = (normalized - codebook[index]).abs()
        closer = distance < best
        best = torch.where(closer, distance, best)
        indices = torch.where(closer, index, indices)
    return indices

def assign_grid(weight: torch.Tensor, levels: torch.Tensor, s_channel: torch.Tensor) -> torch.Tensor:
    """Nearest decoded int8 level per grouped weight (port of _assign_grid)."""
    target = weight / s_channel.view(-1, 1, 1)
    best = (target - levels[..., 0:1].expand_as(weight)).abs()
    indices = torch.zeros_like(weight, dtype=torch.int32)
    for index in range(1, 16):
        distance = (target - levels[..., index:index + 1].expand_as(weight)).abs()
        closer = distance < best
        best = torch.where(closer, distance, best)
        indices = torch.where(closer, index, indices)
    return indices

def quantize_w4a8_weight(weight: torch.Tensor, group_size: int = 16,
                         convrot_groupsize: int = 256, symmetric: bool = True,
                         scale_dtype: torch.dtype = torch.float8_e4m3fn,
                         codebook: bool = True,
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                    Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Quantize a floating 2D weight into W4A8 storage.

    Returns (packed, s_rel, s_channel, correction, codebook_tensor).
    Bit-exact with comfy-kitchen `quantize_w4a8_int8_weight` when run on the
    same device (see module docstring for the algorithm).
    """
    if scale_dtype not in (torch.float32, torch.float8_e4m3fn):
        raise PolicyError(f"scale_dtype must be float32 or float8_e4m3fn, got {scale_dtype}")
    validate_w4_shape(int(weight.shape[1]), group_size, convrot_groupsize)
    rotated = rotate_int8_convrot_weight(weight, convrot_groupsize)
    return _quantize_rotated_w4a8(rotated, group_size, symmetric, scale_dtype, codebook)

def _quantize_rotated_w4a8(weight: torch.Tensor, group_size: int, symmetric: bool,
                           scale_dtype: torch.dtype, codebook: bool,
                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                      Optional[torch.Tensor], Optional[torch.Tensor]]:
    original_dtype = weight.dtype
    n, k = weight.shape
    groups = k // group_size
    grouped_weight = weight.float().view(n, groups, group_size)

    codebook_tensor: Optional[torch.Tensor] = None
    correction: Optional[torch.Tensor] = None
    if symmetric and codebook:
        group_scale = grouped_weight.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        normalized = grouped_weight / group_scale
        codebook_tensor = fit_codebook(normalized, levels=16)
        quantized = assign_codes(normalized, codebook_tensor)
        for _ in range(3):
            qc = codebook_tensor[quantized]
            group_scale = (
                (grouped_weight * qc).sum(-1, keepdim=True)
                / (qc * qc).sum(-1, keepdim=True).clamp(min=1e-8)
            ).clamp(min=1e-8)
            quantized = assign_codes(grouped_weight / group_scale, codebook_tensor)
        unsigned = quantized.to(torch.int32).view(n, k)
        shifted_weight = codebook_tensor[quantized] * group_scale
    elif symmetric:
        group_scale = (grouped_weight.abs().amax(dim=-1, keepdim=True) / 7.0).clamp(min=1e-8)
        signed = (grouped_weight / group_scale).round().clamp(-8, 7).to(torch.int32)
        unsigned = (signed + 8).view(n, k)
        shifted_weight = signed * group_scale
    else:
        minimum = grouped_weight.amin(dim=-1, keepdim=True)
        group_scale = ((grouped_weight.amax(dim=-1, keepdim=True) - minimum) / 15.0).clamp(min=1e-8)
        unsigned = (
            ((grouped_weight - minimum) / group_scale).round().clamp(0, 15)
            .to(torch.int32).view(n, k)
        )
        shifted_weight = (unsigned.view(n, groups, group_size) - 8) * group_scale
        correction = (8.0 * group_scale + minimum).squeeze(-1).t().contiguous().to(original_dtype)

    s_channel = (shifted_weight.abs().amax(dim=(1, 2)) / 127.0).clamp(min=1e-8)
    s_rel = (group_scale.squeeze(-1) / s_channel.unsqueeze(1)).float().contiguous()
    if scale_dtype != torch.float32:
        s_rel = s_rel.to(scale_dtype).contiguous()
    if codebook_tensor is not None:
        levels = (
            (codebook_tensor.view(1, 1, 16) * s_rel.float().unsqueeze(-1))
            .round_().clamp_(-127, 127)
        )
        unsigned = assign_grid(grouped_weight, levels, s_channel).view(n, k)

    packed = (
        ((unsigned[:, 0::2] & 0xF) | ((unsigned[:, 1::2] & 0xF) << 4))
        .to(torch.int8).contiguous()
    )
    return packed, s_rel, s_channel.float().contiguous(), correction, codebook_tensor

def unpack_w4(packed: torch.Tensor) -> torch.Tensor:
    """int8 [N, K/2] packed int4 -> int32 codes [N, K] (even col = low nibble)."""
    n, k_half = packed.shape
    k = k_half * 2
    p = packed.to(torch.int32) & 0xFF
    out = torch.empty(n, k, dtype=torch.int32, device=packed.device)
    out[:, 0::2] = p & 0xF
    out[:, 1::2] = (p >> 4) & 0xF
    return out

def unpack_int4_signed(packed: torch.Tensor) -> torch.Tensor:
    """int8 [N, K/2] packed int4 -> int8 codes [N, K] with SIGNED nibble
    interpretation (even column = low nibble). Mirrors comfy-kitchen
    `_unpack_int4_row_major`; used by the W4A4 dequantization path."""
    n, k_half = packed.shape
    p = packed.to(torch.int32) & 0xFF
    lo = p & 0xF
    hi = (p >> 4) & 0xF
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    out = torch.empty(n, k_half * 2, dtype=torch.int8, device=packed.device)
    out[:, 0::2] = lo.to(torch.int8)
    out[:, 1::2] = hi.to(torch.int8)
    return out

def quantize_int8_tensorwise_weight(
        weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rowwise symmetric INT8 (int8_tensorwise, no ConvRot).

    Serialized contract (matches ComfyUI TensorWiseINT8Layout + the Comfy-Org
    int8_convrot checkpoint family): `weight` int8 [N, K], `weight_scale`
    fp32 [N]. Runtime applies dynamic rowwise int8 activation quantization.
    Works for any K (no divisibility requirement), which is why it is the
    universal fallback tier in mixed mode.
    """
    abs_max = weight.abs().amax(dim=-1, keepdim=True)
    scale = (abs_max.float() / float(INT8_SCALE_MAX)).clamp(min=1e-30)
    q = (weight / scale).round().clamp_(-128.0, 127.0).to(torch.int8)
    # scale is stored [N, 1] so both the eager dequant (q.float() * scale) and
    # the CUDA int8_linear paths broadcast correctly on every backend
    return q.contiguous(), scale.contiguous().to(torch.float32)

def dequantize_int8_tensorwise_weight(
        q: torch.Tensor, scale: torch.Tensor,
        output_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    s = scale.to(device=q.device, dtype=torch.float32)
    w = q.float() * s.reshape(-1, 1)
    return w.to(output_dtype)

def quantize_w4a4_weight(weight: torch.Tensor,
                         convrot_groupsize: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """ConvRot W4A4 quantization, byte-identical to comfy-kitchen's eager
    `quantize_convrot_w4a4_weight`: regular-Hadamard weight rotation, rowwise
    symmetric signed int4 with absmax scale = max/7, emission range [-7, 7],
    packed int4 low=even nibble. Returns (packed int8 [N, K/2], scale fp32 [N])."""
    h = build_hadamard(convrot_groupsize, device=weight.device, dtype=weight.dtype)
    rot = rotate_weight(weight, h, convrot_groupsize)
    abs_max = rot.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scale = abs_max / float(W4A4_EMISSION_MAX)
    q = (rot / scale).round().clamp_(-float(W4A4_EMISSION_MAX),
                                     float(W4A4_EMISSION_MAX)).to(torch.int8)
    packed = ((q[:, 0::2] & 0xF) | ((q[:, 1::2] & 0xF) << 4)).to(torch.int8).contiguous()
    return packed, scale.reshape(-1).contiguous().to(torch.float32)

def dequantize_w4a4_weight(packed: torch.Tensor, scale: torch.Tensor,
                           convrot_groupsize: int,
                           output_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Inverse of quantize_w4a4_weight (matches comfy-kitchen eager dequant)."""
    w_int = unpack_int4_signed(packed).float()
    w_rot = w_int * scale.to(device=packed.device, dtype=torch.float32).reshape(-1, 1)
    h = build_hadamard(convrot_groupsize, device=packed.device, dtype=torch.float32)
    return rotate_weight(w_rot.float(), h, convrot_groupsize).to(output_dtype)

def _pick_w4a4_convrot_group(k: int) -> int:
    """Largest power of 4 in {16, 64, 256} dividing K (kernel-supported set)."""
    for cgs in (256, 64, 16):
        if k % cgs == 0:
            return cgs
    raise PolicyError(f"K={k} has no ConvRot-W4A4 group in {{16, 64, 256}}")

def w4a4_weight_is_quantizable(shape: Sequence[int], dtype: torch.dtype,
                               convrot_groupsize: Optional[int] = None
                               ) -> Tuple[bool, str]:
    """W4A4 eligibility: 2D float, K % 64 == 0 (kernel contract), K % cgs == 0."""
    if len(shape) != 2:
        return False, "not 2D"
    if dtype not in FLOAT_DTYPES:
        return False, f"dtype {dtype} not float"
    k = int(shape[1])
    if k % W4A4_QUANT_GROUP_SIZE != 0:
        return False, f"K={k} not divisible by quant_group_size {W4A4_QUANT_GROUP_SIZE}"
    if convrot_groupsize is None:
        try:
            convrot_groupsize = _pick_w4a4_convrot_group(k)
        except PolicyError:
            return False, f"K={k} has no supported ConvRot group"
    if k % convrot_groupsize != 0:
        return False, f"K={k} not divisible by convrot_groupsize {convrot_groupsize}"
    return True, ""

def int8_weight_is_quantizable(shape: Sequence[int], dtype: torch.dtype
                               ) -> Tuple[bool, str]:
    if len(shape) != 2:
        return False, "not 2D"
    if dtype not in FLOAT_DTYPES:
        return False, f"dtype {dtype} not float"
    return True, ""

def quantized_format_bytes(n: int, k: int, fmt: str) -> int:
    """Exact serialized byte count of the per-layer format payloads."""
    if fmt == FORMAT_W4A8:
        return n * (k // 2) + n * (k // 16) + n * 4 + 16 * 4
    if fmt == FORMAT_W4A4:
        return n * (k // 2) + n * 4
    if fmt == FORMAT_INT8:
        return n * k + n * 4
    raise PolicyError(f"unknown format {fmt!r}")

def format_scale_suffixes(fmt: str) -> Tuple[str, ...]:
    """Output suffixes per format ('' = the packed weight slot)."""
    if fmt == FORMAT_W4A8:
        return ("", "_s_rel", "_s_channel", "_codebook")
    if fmt in (FORMAT_W4A4, FORMAT_INT8):
        return ("", "_scale")
    raise PolicyError(f"unknown format {fmt!r}")

def dequantize_w4a8_weight(packed: torch.Tensor, s_rel: torch.Tensor,
                           s_channel: torch.Tensor,
                           codebook: Optional[torch.Tensor] = None,
                           correction: Optional[torch.Tensor] = None,
                           group_size: int = 16, convrot_groupsize: int = 256,
                           output_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Decode W4A8 storage back to the physical [N, K] weight (reference decode).

    Bit-exact with comfy-kitchen `dequantize_w4a8_int8_weight` (eager/triton/cuda).
    """
    if packed.dim() != 2 or packed.dtype != torch.int8:
        raise PolicyError("packed weight must be a 2D int8 tensor")
    n, k_half = packed.shape
    k = k_half * 2
    validate_w4_shape(k, group_size, convrot_groupsize)
    groups = k // group_size
    if tuple(s_rel.shape) != (n, groups):
        raise PolicyError(f"s_rel must have shape {(n, groups)}, got {tuple(s_rel.shape)}")
    if tuple(s_channel.shape) != (n,):
        raise PolicyError(f"s_channel must have shape {(n,)}, got {tuple(s_channel.shape)}")
    if correction is not None and tuple(correction.shape) != (groups, n):
        raise PolicyError(f"correction must have shape {(groups, n)}, got {tuple(correction.shape)}")
    if codebook is not None and tuple(codebook.shape) != (16,):
        raise PolicyError(f"codebook must have shape (16,), got {tuple(codebook.shape)}")

    codes = unpack_w4(packed)
    if codebook is not None:
        values = codebook.to(device=packed.device, dtype=torch.float32)[codes]
    else:
        values = codes.float() - 8.0
    values = values.view(n, groups, group_size) * s_rel.float().unsqueeze(-1)
    int8_weight = values.view(n, k).round().clamp_(-127, 127).to(torch.int8)

    weight_rotated = int8_weight.float().view(n, groups, group_size)
    weight_rotated = weight_rotated * s_channel.float().view(n, 1, 1)
    if correction is not None:
        weight_rotated = weight_rotated + correction.t().unsqueeze(-1).float()
    weight_rotated = weight_rotated.view(n, k)
    return rotate_int8_convrot_weight(weight_rotated, convrot_groupsize).to(output_dtype)

def decode_w4a8_runtime_weight(packed: torch.Tensor, s_rel: torch.Tensor,
                                codebook: Optional[torch.Tensor],
                                group_size: int) -> torch.Tensor:
    """Decode W4A8 storage into the ROTATED runtime weight representation the
    W4A8 GEMM consumes: packed int4 codes -> codebook values -> per-group
    fp8 scales -> rounded int8 levels, WITHOUT the inverse ConvRot rotation.

    This is the weight side of the runtime operation
    (comfy-kitchen `_dequant_int4_grouped_to_int8`). The physical weight
    (`dequantize_w4a8_weight`) additionally rotates back to the ordinary
    basis; mixing the two coordinate systems in one simulation is wrong."""
    codes = unpack_w4(packed)
    if codebook is not None:
        values = codebook.to(device=packed.device, dtype=torch.float32)[codes]
    else:
        values = codes.float() - 8.0
    n, k = values.shape
    groups = k // group_size
    values = values.view(n, groups, group_size) * s_rel.float().unsqueeze(-1)
    return values.view(n, k).round().clamp_(-127, 127).to(torch.int8)

def quantize_weight_by_format(weight: torch.Tensor, fmt: str, group_size: int,
                              convrot_groupsize: int) -> Dict[str, torch.Tensor]:
    """Quantize with the W4A8 layout; returns the per-layer output tensors
    keyed by suffix ('' for the packed weight, '_s_rel', '_s_channel',
    '_codebook', optional '_correction')."""
    if fmt == FORMAT_W4A8:
        packed, s_rel, s_ch, corr, cb = quantize_w4a8_weight(
            weight, group_size=group_size, convrot_groupsize=convrot_groupsize,
            symmetric=True, scale_dtype=torch.float8_e4m3fn, codebook=True)
        out = {"": packed, "_s_rel": s_rel, "_s_channel": s_ch, "_codebook": cb}
        if corr is not None:
            out["_correction"] = corr
        return out
    if fmt == FORMAT_INT8:
        q, scale = quantize_int8_tensorwise_weight(weight)
        return {"": q, "_scale": scale}
    if fmt == FORMAT_W4A4:
        packed, scale = quantize_w4a4_weight(weight, convrot_groupsize)
        return {"": packed, "_scale": scale}
    raise PolicyError(f"unknown quantization format {fmt!r}")

def dequantize_weight_by_format(tensors: Dict[str, torch.Tensor], fmt: str,
                                group_size: int, convrot_groupsize: int,
                                output_dtype: torch.dtype) -> torch.Tensor:
    if fmt == FORMAT_W4A8:
        return dequantize_w4a8_weight(
            tensors[""], tensors["_s_rel"], tensors["_s_channel"],
            codebook=tensors.get("_codebook"), correction=tensors.get("_correction"),
            group_size=group_size, convrot_groupsize=convrot_groupsize,
            output_dtype=output_dtype)
    if fmt == FORMAT_INT8:
        return dequantize_int8_tensorwise_weight(tensors[""], tensors["_scale"],
                                                 output_dtype)
    if fmt == FORMAT_W4A4:
        return dequantize_w4a4_weight(tensors[""], tensors["_scale"],
                                      convrot_groupsize, output_dtype)
    raise PolicyError(f"unknown quantization format {fmt!r}")
