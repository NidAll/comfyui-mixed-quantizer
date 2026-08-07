#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""comfyui_wxa8_quantizer.py -- standalone W4A8 generative-model checkpoint converter.

Converts supported generative-model checkpoints (safetensors / sharded safetensors /
HF-style model directories / optionally torch pickles with --trust-pickle) into a
ComfyUI-compatible W4A8 ("asym_w4a8_int8") checkpoint.

This utility is fully standalone.  It does not import, require, or execute any
ComfyUI / comfy-kitchen / ComfyUI-custom-node code at runtime.  Every inspection,
detection, quantization, packing, metadata and validation component is reimplemented
here from the verified reference behaviour described below.

------------------------------------------------------------------------------
Verified format specification (research notes -- exact source revisions)
------------------------------------------------------------------------------

Reference implementation (authoritative, merged):
  * comfy-kitchen PR #90  "Add optimized w4a8 with int8 codebook" (MERGED)
        https://github.com/Comfy-Org/comfy-kitchen/pull/90
        merge commit : aa1ab2263dc06225d9de6702dfc087313d4bc971 (2026-08-06)
        head commit  : b812819a97ac11d01f4a3a16ba47dd38de3b2519
        files studied:
          comfy_kitchen/tensor/w4a8_int8.py            (layout + serialization contract)
          comfy_kitchen/backends/eager/w4a8_int8.py    (quantize / dequantize / linear)
          comfy_kitchen/backends/triton/w4a8_int8.py   (fused triton dequant, bit-exact)
          comfy_kitchen/backends/cuda/ops/w4a8_gemm.cu (CUDA dequant kernel expectations)
          comfy_kitchen/backends/cuda/__init__.py      (launcher / dtype codes)
          comfy_kitchen/tensor/int8_utils.py           (ConvRot Hadamard rotation)
          comfy_kitchen/tensor/base.py                 (QuantizedTensor state-dict layout)

  * comfy-kitchen PR #96 (compatible later producer update, MERGED)
        https://github.com/Comfy-Org/comfy-kitchen/pull/96
        merge commit : 3d16bed29c91a3d9d1abdc0574c87a5ba2b1ef33 (2026-08-06)
        Replaces per-weight Lloyd-Max fitting with a frozen LUT and adds row-chunked
        producer quantization.  It preserves the same serialized runtime layout.
        This converter deliberately retains the verified PR #90 numerical contract;
        its stored per-layer codebook remains consumable by later runtimes.

  * ComfyUI PR #15308  "Support asym w4a8_int" (MERGED 2026-08-07 as
        commit 344b43989e, shipped in ComfyUI v0.31.0; earlier head
        8c3a2b27c37bd34e87b58846baf962407c92843c was studied at research time,
        base bdcb886a4705a03cf40f4a7226de9fc7c059fc90)
        files studied:
          comfy/ops.py        (_load_quantized_module: pops weight_s_rel / weight_s_channel
                               / weight_codebook, reads layer_conf "group_size" and
                               "convrot_groupsize", registers "asym_w4a8_int8")
          comfy/quant_ops.py  (QUANT_ALGOS["asym_w4a8_int8"] = {"storage_t": torch.int8,
                               "parameters": {"weight_scale"},
                               "comfy_tensor_layout": "AsymW4A8Int8Layout",
                               "quantize_input": False})

  * Reference serialized example (produced by the PR author, used for testing):
        https://huggingface.co/Kijai/MiniMax-H3-experimental/blob/main/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors
        (12.5 GB; header inspected at research time)
        - quantized layers: {layer}.weight (I8, [N, K/2]), {layer}.weight_s_rel
          (F8_E4M3, [N, K/group_size]), {layer}.weight_s_channel (F32, [N]),
          {layer}.weight_codebook (F32, [16])
        - safetensors __metadata__["_quantization_metadata"] =
          {"layers": {layer: {"format": "asym_w4a8_int8", "group_size": 16,
                              "convrot": true, "convrot_groupsize": 256}}}
        - non-quantized layers stay at their original dtypes (F16/BF16/F32).

W4A8 numerical representation ("asym_w4a8_int8", verified against the eager backend):
  * Input  : 2D weight W [N, K];  K % 16 == 0, K % group_size == 0,
             K % convrot_groupsize == 0, group_size >= 4 and
             (16 % group_size == 0 or group_size % 16 == 0).
  * ConvRot: W_rot = W @ (I (x) H)^T, H = normalized regular Hadamard built from
             H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]/2 via Kronecker
             products (size must be a power of 4).  The same rotation is applied
             online to activations at runtime (x @ (I (x) H)), so the GEMM result
             is exactly x @ W.T.
  * Group scale (symmetric + codebook, the default path):
       group_scale   = amax(|W_rot|, dim=group).clamp(min=1e-8)
       normalized    = W_rot / group_scale                      (values in [-1, 1])
       codebook      = Lloyd-Max fit (16 levels, 25 iterations, deterministic
                       subsample of up to 300000 elements, torch.quantile init)
       codes         = argmin |normalized - codebook|           (per element)
       3 refinement rounds: group_scale = sum(w*q)/sum(q*q) (least squares),
                            then re-assign codes.
  * Scales:
       s_channel[r] = amax(|codebook[codes] * group_scale|, row r) / 127 (clamp 1e-8)
       s_rel[r,g]   = group_scale[r,g] / s_channel[r]  (fp32 -> fp8_e4m3fn)
       levels       = round(codebook * s_rel).clamp(-127, 127)   (int8 grid)
       final codes  = argmin over the 16 decoded int8 levels (activation of the
                      runtime decode: round(clamp(level(code) * s_rel, -127, 127)))
  * Packing: int4 codes (unsigned 0..15), even column in the LOW nibble,
             odd column in the HIGH nibble;  packed dtype int8, shape [N, K/2].
  * Runtime decode (CUDA / Triton / eager, all bit-identical):
       out8[n,k] = round(clamp(codebook[codes[n,k]] * s_rel[n, k//G], -127, 127))
       W_rot     = out8 * s_channel[:, None]      (per-channel scale, GEMM epilogue)
       W         = unrotate(W_rot)                (same Hadamard, H symmetric)
  * Runtime activations: apply ConvRot, then dynamically quantize each input row
       to int8 with fp32 scale amax(abs(row))/127 (clamp min 1e-30), nearest
       rounding and clamp [-128, 127]; GEMM accumulation is int32.
  * Serialization (per quantized layer):
       {layer}.weight             int8   [N, K/2]
       {layer}.weight_s_rel       fp8_e4m3fn [N, K/group_size]
       {layer}.weight_s_channel   fp32   [N]
       {layer}.weight_codebook    fp32   [16]   (present when codebook mode used)
       (an asymmetric variant with a {layer}.weight_correction tensor [K/gs, N]
        exists in comfy-kitchen but is NOT consumed by the current ComfyUI loader;
        this converter always uses the symmetric codebook mode.)
  * Global metadata (read by ComfyUI comfy/utils.py convert_old_quants, which turns
    it into per-layer "{layer}.comfy_quant" JSON blobs consumed by ops.py):
       __metadata__["_quantization_metadata"] = {"layers": {layer: conf}}
       conf = {"format": "asym_w4a8_int8", "group_size": 16,
               "convrot": true, "convrot_groupsize": 256}
  * Runtime prerequisites (state explicitly in metadata/reports):
       - comfy-kitchen >= merge commit aa1ab2263dc06225d9de6702dfc087313d4bc971
         (PR #90; AsymW4A8Int8Layout registered; eager/triton/CUDA backends)
       - ComfyUI >= v0.31.0 (native loader, PR #15308 merged as
         344b43989e); older builds need patches/comfyui_w4a8_loader.patch
       - CUDA backend requires PyTorch cu130+ and SM >= 8.0; Triton >= 3.7 for
         ROCm; pure-torch eager works on CPU/CUDA/ROCm for dequant + linear.
  * Group sizes accepted by the CUDA dequant kernel: G in {4, 8, 16} or a
    multiple of 16 (a 16-wide vector spans at most 4 groups).
  * ConvRot group: always 256. The comfy-kitchen 0.2.27 CUDA fused kernels
    (activation rotation + quantize, chunked codebook GEMM, weight rotation)
    implement ConvRot only for a 256-wide Hadamard group and throw
    "convrot fused kernel only supports group_size 256" otherwise. A layer is
    therefore W4A8-quantized only when K % 256 == 0; any other 2D linear
    passes through at original precision with the reason recorded.

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
    python comfyui_wxa8_quantizer.py ORIGINAL_MODEL --output OUT --format w4a8
    python comfyui_wxa8_quantizer.py --self-test
    python comfyui_wxa8_quantizer.py --list-architectures

Run `python comfyui_wxa8_quantizer.py --help` for the full option list.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import enum
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import math
import mmap
import os
import re
import shutil
import stat
import struct
import sys
import tempfile
import time
import traceback
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # Required public dependencies
    import numpy as np
    import torch
    import safetensors
    from safetensors import safe_open
    import safetensors.torch  # submodule required for save_file and dtype registry
except Exception as _exc:  # pragma: no cover - import guard
    raise SystemExit(
        "comfyui_wxa8_quantizer requires: torch (>=2.1), safetensors (>=0.4.3), numpy.\n"
        f"Import failed: {_exc!r}"
    ) from _exc

# ---------------------------------------------------------------------------
# Version / revision constants (research record; see module docstring)
# ---------------------------------------------------------------------------
CONVERTER_NAME = "comfyui_wxa8_quantizer"
CONVERTER_VERSION = "1.2.2"
FORMAT_W4A8 = "asym_w4a8_int8"

# comfy-kitchen 0.2.27 CUDA fused kernels implement ConvRot only for a 256-wide
# Hadamard group (int8_linear.cu: "convrot fused kernel only supports group_size
# 256"). W4A8 therefore requires K % 256 == 0; layers whose K is not divisible
# by 256 pass through at original precision instead of being quantized with a
# smaller ConvRot group.
W4A8_CONVROT_GROUPSIZE = 256
FORMAT_W4A8_REVISION = "asym-w4a8-int8-r1"
MAX_SAFETENSORS_HEADER_SIZE = 100_000_000
METADATA_KEY_QUANT = "_quantization_metadata"     # official key read by ComfyUI
METADATA_KEY_EXT = "comfy_wxa8"                   # namespaced extension key (never official)
LAYER_CONF_KEY = "comfy_quant"                    # per-layer blob key used by ComfyUI loader
COMFY_KITCHEN_REV = "aa1ab2263dc06225d9de6702dfc087313d4bc971"   # PR #90 merge commit
COMFYUI_PR = 15308
COMFYUI_PR_HEAD = "8c3a2b27c37bd34e87b58846baf962407c92843c"
COMFYUI_BASE = "bdcb886a4705a03cf40f4a7226de9fc7c059fc90"
W4A8_KERNEL_MIN_SM = (8, 0)
TRITON_MIN_VERSION = (3, 7)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class QuantizerError(Exception):
    """Base class for all converter errors."""

class UsageError(QuantizerError):
    """Bad CLI usage."""

class InputError(QuantizerError):
    """Unreadable / unsafe / malformed input."""

class PickleInputError(InputError):
    """Pickle-based input without explicit --trust-pickle."""

class UnknownArchitectureError(QuantizerError):
    """Architecture could not be identified unambiguously."""

class UnsupportedArchitectureError(QuantizerError):
    """Architecture is known but has no safe conversion policy."""

class PolicyError(QuantizerError):
    """Architecture policy violation."""

class CalibrationError(QuantizerError):
    """Calibration data problem."""

class ValidationError(QuantizerError):
    """Output failed standalone validation."""

class OutputError(QuantizerError):
    """Output path / serialization problem."""

class SelfTestFailure(QuantizerError):
    """Embedded self-test failure."""

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
class JsonLogHandler(logging.Handler):
    """Emit each record as one JSON line (optional --json-log)."""

    def __init__(self, path: str):
        super().__init__()
        self._path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        fd, _ = _open_regular_nofollow(
            path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
        self._fh = os.fdopen(fd, "a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "ts": time.time(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            if record.exc_info:
                entry["exc"] = "".join(traceback.format_exception(*record.exc_info))
            self._fh.write(json.dumps(entry) + "\n")
            self._fh.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:  # pragma: no cover
        try:
            self._fh.close()
        finally:
            super().close()


def setup_logging(level: str = "info", json_log: Optional[str] = None) -> None:
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        raise UsageError(f"invalid --log-level {level!r}")
    root = logging.getLogger("wxa8")
    root.setLevel(numeric)
    root.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if json_log:
        root.addHandler(JsonLogHandler(json_log))
    root.propagate = False


def log() -> logging.Logger:
    return logging.getLogger("wxa8")

# ---------------------------------------------------------------------------
# Environment inspection
# ---------------------------------------------------------------------------
@dataclass
class EnvironmentInfo:
    python: str
    torch_version: str
    torch_cuda: Optional[str]
    torch_hip: Optional[str]
    cuda_available: bool
    cuda_device: Optional[str]
    cuda_capability: Optional[Tuple[int, int]]
    rocm_arch: Optional[str]
    safetensors_version: str
    numpy_version: str
    platform: str
    cpu_count: int
    has_comfy_kitchen: bool = False
    comfy_kitchen_rev: Optional[str] = None
    comfy_kitchen_has_w4a8_layout: bool = False
    has_comfy_quant_ops: bool = False
    comfyui_quant_algos: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def inspect_environment() -> EnvironmentInfo:
    info = EnvironmentInfo(
        python=sys.version.split()[0],
        torch_version=torch.__version__,
        torch_cuda=getattr(torch.version, "cuda", None),
        torch_hip=getattr(torch.version, "hip", None),
        cuda_available=torch.cuda.is_available(),
        cuda_device=None,
        cuda_capability=None,
        rocm_arch=None,
        safetensors_version=safetensors.__version__,
        numpy_version=np.__version__,
        platform=sys.platform,
        cpu_count=os.cpu_count() or 1,
    )
    if info.cuda_available:
        try:
            info.cuda_device = torch.cuda.get_device_name(0)
            info.cuda_capability = tuple(int(x) for x in torch.cuda.get_device_capability(0))
            props = torch.cuda.get_device_properties(0)
            info.rocm_arch = getattr(props, "gcnArchName", None)
        except Exception as e:
            log().debug("CUDA environment probe failed: %s", e)
    # Optional runtime compatibility probing is deliberately static.  Importing
    # either project here would execute third-party package initializers and
    # would violate the converter's standalone/runtime-isolation guarantee.
    try:
        dist = importlib.metadata.distribution("comfy-kitchen")
        info.has_comfy_kitchen = True
        info.comfy_kitchen_rev = dist.version
        for rel in dist.files or ():
            rel_text = str(rel).replace("\\", "/")
            if not rel_text.endswith(".py") or "comfy_kitchen" not in rel_text:
                continue
            path = Path(dist.locate_file(rel))
            try:
                if path.stat().st_size <= 4 * 1024 * 1024 and \
                        "AsymW4A8Int8Layout" in path.read_text(encoding="utf-8", errors="ignore"):
                    info.comfy_kitchen_has_w4a8_layout = True
                    break
            except OSError:
                continue
    except importlib.metadata.PackageNotFoundError:
        pass
    except Exception as e:
        log().debug("comfy-kitchen static compatibility probe failed: %s", e)
    try:
        spec = importlib.util.find_spec("comfy")
        roots = list(spec.submodule_search_locations or ()) if spec is not None else []
        for root in roots:
            quant_ops = Path(root) / "quant_ops.py"
            if not quant_ops.is_file() or quant_ops.stat().st_size > 4 * 1024 * 1024:
                continue
            source = quant_ops.read_text(encoding="utf-8", errors="ignore")
            info.has_comfy_quant_ops = True
            # Only report formats proven present in the static source.  This is
            # not a runtime import or a claim that the whole installation works.
            if FORMAT_W4A8 in source:
                info.comfyui_quant_algos.append(FORMAT_W4A8)
            break
    except Exception as e:
        log().debug("ComfyUI static compatibility probe failed: %s", e)
    return info

# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def parse_size(text: str) -> int:
    """Parse a size like 2G, 512M, 1024K or a plain byte count."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGTP]?B?)?\s*", text, re.I)
    if not m:
        raise UsageError(f"invalid size {text!r}")
    value = float(m.group(1))
    unit = (m.group(2) or "").upper().rstrip("B")
    mult = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3,
            "T": 1024**4, "P": 1024**5}[unit]
    result = value * mult
    if not math.isfinite(result) or result > sys.maxsize:
        raise UsageError(f"size {text!r} exceeds this platform's supported range")
    return int(result)


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if abs(n) < 1024 or unit == "PiB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.2f} PiB"  # pragma: no cover


def sha256_file(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def sha256_safetensors_payload(path: str, chunk: int = 1 << 22) -> str:
    """Hash only the tensor-data section, which is stable across metadata rewrites."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) != 8:
            raise InputError(f"{path}: truncated safetensors header")
        header_len = struct.unpack("<Q", head)[0]
        if header_len > os.fstat(f.fileno()).st_size - 8:
            raise InputError(f"{path}: safetensors header exceeds file size")
        f.seek(8 + header_len)
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def _fsync_parent(path: str) -> None:
    """Best-effort directory fsync after atomic publication (POSIX)."""
    if os.name == "nt":
        return
    parent = os.path.dirname(os.path.abspath(path)) or "."
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(parent, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_json(path: str, payload: Any, *, indent: Optional[int] = None) -> None:
    """Write JSON through an unpredictable sibling temp and atomically replace."""
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{Path(path).name}.", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=indent, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_parent(path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _open_regular_nofollow(path: str, flags: int, mode: int = 0o600):
    """Open a regular file without following a final-component symlink."""
    flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, mode)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise OutputError(f"refusing non-regular file: {path}")
        return fd, st
    except Exception:
        os.close(fd)
        raise


def _remove_temp_path(path: str) -> None:
    """Remove only the named temp entry; never follow it if it is a symlink."""
    try:
        os.lstat(path)
    except FileNotFoundError:
        return
    os.unlink(path)


def _same_path(left: str, right: str) -> bool:
    if os.path.abspath(left) == os.path.abspath(right):
        return True
    if os.path.exists(left) and os.path.exists(right):
        try:
            return os.path.samefile(left, right)
        except OSError:
            pass
    return False


def _validate_destination_paths(info: CheckpointInfo, args: Any) -> None:
    destinations: List[Tuple[str, str]] = []
    if getattr(args, "output", None):
        output = os.path.abspath(args.output)
        destinations.extend((
            ("output", output),
            ("conversion temp", output + ".tmp"),
            ("staged output", output + ".staged"),
            ("validation output", output + ".validation"),
            ("resume state", output + ".state.json"),
        ))
        if getattr(args, "metadata_only", False):
            destinations.append(("metadata sidecar", output + ".metadata.json"))
    if getattr(args, "report", None):
        destinations.extend((("report", args.report),
                             ("JSON report", args.report + ".json")))
    if getattr(args, "json_log", None):
        destinations.append(("JSON log", args.json_log))
    if getattr(args, "calibration_cache", None):
        destinations.append(("calibration cache", args.calibration_cache))

    for label, path in destinations:
        for source in info.files:
            if _same_path(path, source):
                raise OutputError(f"{label} path must not alias input file {source}")
    for index, (left_label, left) in enumerate(destinations):
        for right_label, right in destinations[index + 1:]:
            if _same_path(left, right):
                raise OutputError(
                    f"{left_label} and {right_label} paths must be different: {left}")


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def torch_dtype_name(dt: torch.dtype) -> str:
    return str(dt).replace("torch.", "")


def flatten_regex(patterns: Sequence[str]) -> re.Pattern:
    try:
        return re.compile("|".join(f"(?:{p})" for p in patterns))
    except re.error as e:
        raise UsageError(f"invalid regular expression: {e}") from e


def _peak_rss_bytes() -> int:
    try:
        import resource
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS/BSD reports bytes.
        return peak if sys.platform == "darwin" else peak * 1024
    except Exception:
        return 0



# ---------------------------------------------------------------------------
# Safe checkpoint reading
# ---------------------------------------------------------------------------
# safetensors <-> torch dtype tables (stable, documented format)
SAFE_TO_TORCH: Dict[str, torch.dtype] = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
    "BOOL": torch.bool, "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
}
_OPTIONAL_SAFE_DTYPES = {
    "F8_E4M3FNUZ": "float8_e4m3fnuz",
    "F8_E5M2FNUZ": "float8_e5m2fnuz",
    "C64": "complex64",
    "U16": "uint16",
    "U32": "uint32",
    "U64": "uint64",
}
_installed_safe_types = getattr(safetensors.torch, "_TYPES", {})
for _safe_name, _torch_name in _OPTIONAL_SAFE_DTYPES.items():
    _dtype = getattr(torch, _torch_name, None)
    if _dtype is not None and _safe_name in _installed_safe_types:
        SAFE_TO_TORCH[_safe_name] = _dtype
TORCH_TO_SAFE: Dict[torch.dtype, str] = {v: k for k, v in SAFE_TO_TORCH.items()}
FP8_DTYPES = {dtype for name, dtype in SAFE_TO_TORCH.items()
              if name.startswith("F8_")}
FLOAT_DTYPES = {torch.float32, torch.float16, torch.bfloat16, torch.float64}


def tensor_nbytes(dtype: torch.dtype, shape: Sequence[int]) -> int:
    return math.prod(int(s) for s in shape) * dtype.itemsize


def torch_dtype_from_safe(name: str) -> torch.dtype:
    try:
        return SAFE_TO_TORCH[name]
    except KeyError as e:
        raise InputError(f"unsupported safetensors dtype {name!r}") from e


@dataclass
class TensorMeta:
    """Header-level information about one tensor (no data loaded)."""
    name: str
    dtype: torch.dtype
    shape: Tuple[int, ...]
    nbytes: int
    source: str          # file path
    offset: int          # byte offset within source file
    end: int             # exclusive byte end


@dataclass
class CheckpointInfo:
    """Everything known about an input checkpoint without loading data."""
    kind: str                     # "safetensors" | "sharded-safetensors" | "pickle"
    files: List[str]              # data files in logical order
    metadata: Dict[str, str]      # safetensors __metadata__ (or {})
    tensors: List[TensorMeta]     # logical tensor order
    config: Dict[str, Any] = field(default_factory=dict)   # merged HF-style config.json
    model_index: Dict[str, Any] = field(default_factory=dict)  # model_index.json
    shard_index: Dict[str, Any] = field(default_factory=dict)  # model.safetensors.index.json
    total_bytes: int = 0
    is_quantized_input: bool = False
    source_hashes: Dict[str, str] = field(default_factory=dict)
    _tensor_by_name: Dict[str, TensorMeta] = field(
        init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        for tensor in self.tensors:
            if tensor.name in self._tensor_by_name:
                raise InputError(f"duplicate tensor {tensor.name!r} in checkpoint inventory")
            self._tensor_by_name[tensor.name] = tensor

    def key_set(self) -> set:
        return set(self._tensor_by_name)

    def by_name(self, name: str) -> Optional[TensorMeta]:
        return self._tensor_by_name.get(name)


class RawSafetensorsFile:
    """Lazy mmap reader for one .safetensors file.

    Provides zero-copy raw byte access (for passthrough copy) and lazy torch
    tensor views (for quantization).  Everything is validated against the
    header before any data is touched.
    """

    HEADER_LEN = 8

    def __init__(self, path: str):
        self.path = str(path)
        self._fh = open(self.path, "rb")
        self._mm: Optional[mmap.mmap] = None
        self.entries: Dict[str, Tuple[torch.dtype, Tuple[int, ...], int, int]] = {}
        self.metadata: Dict[str, str] = {}
        self.header_bytes = b""
        self.file_size = os.fstat(self._fh.fileno()).st_size
        try:
            head = self._fh.read(self.HEADER_LEN)
            if len(head) < self.HEADER_LEN:
                raise InputError(f"{self.path}: not a safetensors file (too short)")
            header_len = struct.unpack("<Q", head)[0]
            if header_len == 0 or header_len > MAX_SAFETENSORS_HEADER_SIZE:
                raise InputError(f"{self.path}: absurd safetensors header size {header_len}")
            if header_len > self.file_size - self.HEADER_LEN:
                raise InputError(f"{self.path}: safetensors header exceeds file size")
            self.header_bytes = self._fh.read(header_len)
            if len(self.header_bytes) != header_len:
                raise InputError(f"{self.path}: truncated safetensors header")
            def _no_duplicates(pairs):
                obj = {}
                for key, value in pairs:
                    if key in obj:
                        raise InputError(f"{self.path}: duplicate JSON key {key!r}")
                    obj[key] = value
                return obj
            header = json.loads(self.header_bytes.decode("utf-8"),
                                object_pairs_hook=_no_duplicates)
            if not isinstance(header, dict):
                raise InputError(f"{self.path}: malformed safetensors header")
            raw_metadata = header.get("__metadata__") or {}
            if not isinstance(raw_metadata, dict) or not all(
                    isinstance(k, str) and isinstance(v, str)
                    for k, v in raw_metadata.items()):
                raise InputError(f"{self.path}: __metadata__ must map strings to strings")
            self.metadata = dict(raw_metadata)
            data_start = self.HEADER_LEN + header_len
            ranges: List[Tuple[int, int, str]] = []
            for name, spec in header.items():
                if name == "__metadata__":
                    continue
                if not isinstance(name, str) or not name or not isinstance(spec, dict):
                    raise InputError(f"{self.path}: malformed entry {name!r}")
                raw_dtype = spec.get("dtype")
                raw_shape = spec.get("shape")
                raw_offsets = spec.get("data_offsets")
                if not isinstance(raw_dtype, str):
                    raise InputError(f"{self.path}: non-string dtype for {name!r}")
                if not isinstance(raw_shape, list) or any(
                        type(dim) is not int for dim in raw_shape):
                    raise InputError(
                        f"{self.path}: shape for {name!r} must contain JSON integers")
                if not isinstance(raw_offsets, list) or len(raw_offsets) != 2 or any(
                        type(offset) is not int for offset in raw_offsets):
                    raise InputError(
                        f"{self.path}: data_offsets for {name!r} must be two JSON integers")
                try:
                    dtype = torch_dtype_from_safe(raw_dtype)
                    shape = tuple(raw_shape)
                    offs = tuple(raw_offsets)
                except Exception as e:
                    raise InputError(
                        f"{self.path}: malformed entry {name!r}: {e}") from e
                if any(dim < 0 for dim in shape):
                    raise InputError(f"{self.path}: negative shape dimension for {name!r}")
                if offs[0] < 0 or offs[1] < 0 or offs[0] > offs[1]:
                    raise InputError(f"{self.path}: bad data_offsets for {name!r}")
                nbytes = offs[1] - offs[0]
                expected = tensor_nbytes(dtype, shape)
                if nbytes != expected:
                    raise InputError(
                        f"{self.path}: {name!r} size mismatch (header {nbytes}B, "
                        f"shape {shape} {dtype} needs {expected}B)")
                if data_start + offs[1] > self.file_size:
                    raise InputError(f"{self.path}: {name!r} data range exceeds file size")
                self.entries[name] = (dtype, shape, data_start + offs[0], data_start + offs[1])
                ranges.append((offs[0], offs[1], name))
            cursor = 0
            for start, end, name in sorted(ranges, key=lambda item: (item[0], item[1], item[2])):
                if start != cursor:
                    kind = "overlap" if start < cursor else "hole"
                    raise InputError(
                        f"{self.path}: tensor {name!r} creates a data {kind} "
                        f"({start} != expected {cursor})")
                cursor = end
            if data_start + cursor != self.file_size:
                raise InputError(
                    f"{self.path}: unindexed trailing data "
                    f"({self.file_size - (data_start + cursor)} bytes)")
            # mmap the whole file for zero-copy access
            # ACCESS_COPY keeps the source immutable while exposing a writable
            # buffer view, avoiding PyTorch's non-writable-frombuffer warning.
            self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_COPY)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._mm is not None:
            with contextlib.suppress(Exception):
                self._mm.close()
            self._mm = None
        with contextlib.suppress(Exception):
            self._fh.close()

    def __enter__(self) -> "RawSafetensorsFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def get(self, name: str) -> Tuple[torch.dtype, Tuple[int, ...], int, int]:
        try:
            return self.entries[name]
        except KeyError as e:
            raise InputError(f"{self.path}: no tensor named {name!r}") from e

    def read_bytes(self, name: str) -> memoryview:
        """Zero-copy raw byte range for a tensor."""
        _, _, start, end = self.get(name)
        if self._mm is None:
            raise InputError(f"{self.path}: checkpoint reader is closed")
        return memoryview(self._mm)[start:end]

    def read_tensor(self, name: str) -> torch.Tensor:
        """Lazy torch view of a tensor (zero-copy; do not mutate)."""
        dtype, shape, start, end = self.get(name)
        if self._mm is None:
            raise InputError(f"{self.path}: checkpoint reader is closed")
        raw = memoryview(self._mm)[start:end]
        if dtype in FP8_DTYPES:
            t = torch.frombuffer(raw, dtype=torch.uint8).view(dtype)
        else:
            t = torch.frombuffer(raw, dtype=dtype)
        return t.view(shape)


def _load_json_object(path: str, label: str, *, nofollow: bool = False) -> Dict[str, Any]:
    def _no_duplicates(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise InputError(f"{path}: duplicate JSON key {key!r}")
            obj[key] = value
        return obj
    if nofollow:
        fd, file_stat = _open_regular_nofollow(path, os.O_RDONLY)
        file_obj = os.fdopen(fd, "r", encoding="utf-8")
        file_size = file_stat.st_size
    else:
        file_obj = open(path, "r", encoding="utf-8")
        file_size = os.fstat(file_obj.fileno()).st_size
    with file_obj as f:
        if file_size > 64 * 1024 * 1024:
            raise InputError(f"{path}: JSON file exceeds 64 MiB safety limit")
        data = json.load(f, object_pairs_hook=_no_duplicates)
    if not isinstance(data, dict):
        raise InputError(f"{path}: malformed {label}")
    return data


def _parse_index_json(path: str) -> Dict[str, Any]:
    return _load_json_object(path, "index file")


def _parse_config_json(path: str) -> Dict[str, Any]:
    return _load_json_object(path, "config.json")


def _looks_like_pickle(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(4)
    except OSError as e:
        raise InputError(f"cannot read {path}: {e}") from e
    return head[:2] == b"\x80" or head[:4] == b"PK\x03\x04" or head[:2] == b"\x93" or head[:4] == b"1L\x0b" or head[:4] == b"1M\x0b" or head[:2] == b"\x81"


def _is_safetensors(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(8)
    except OSError:
        return False
    if len(head) < 8:
        return False
    try:
        hlen = struct.unpack("<Q", head)[0]
    except Exception:
        return False
    return 0 < hlen < 512 * 1024 * 1024


def discover_checkpoint(input_path: str, trust_pickle: bool = False) -> CheckpointInfo:
    """Discover and header-inspect an input checkpoint.

    Accepts: single safetensors file, sharded safetensors (index + shards),
    model directory (HF-style), or pickle checkpoints with --trust-pickle.
    """
    p = Path(input_path)
    if not p.exists():
        raise InputError(f"input path does not exist: {input_path}")
    if p.is_dir():
        return _discover_directory(p, trust_pickle)
    if p.name.endswith(".safetensors.index.json"):
        return _discover_directory(p.parent, trust_pickle, index_path=p)

    # Single file
    if _is_safetensors(str(p)):
        with RawSafetensorsFile(str(p)) as rf:
            metas = []
            for name, (dtype, shape, start, end) in sorted(rf.entries.items()):
                metas.append(TensorMeta(name, dtype, shape, end - start, str(p), start, end))
        config_path = p.parent / "config.json"
        model_index_path = p.parent / "model_index.json"
        cp = CheckpointInfo(
            kind="safetensors", files=[str(p)], metadata=rf.metadata,
            tensors=metas,
            config=_parse_config_json(str(config_path)) if config_path.is_file() else {},
            model_index=_parse_config_json(str(model_index_path))
            if model_index_path.is_file() else {},
            total_bytes=sum(t.nbytes for t in metas),
        )
        _flag_quantized_input(cp)
        return cp

    if str(p).endswith((".pt", ".ckpt", ".pth", ".bin", ".safetensors.bak", ".pickle", ".pkl")) or _looks_like_pickle(str(p)):
        if not trust_pickle:
            raise PickleInputError(
                f"{input_path} looks like a pickle-based (torch) checkpoint. "
                "Deserializing pickle checkpoints executes arbitrary code embedded "
                "in the file; pass --trust-pickle only for files you trust.")
        return _read_pickle_checkpoint(str(p))

    raise InputError(
        f"unsupported input file {input_path}: not a safetensors file and not "
        "recognized as a pickle checkpoint")


def _resolve_under(base: Path, relative: str) -> Path:
    """Resolve an untrusted index path and require it to stay below base."""
    rel = Path(relative)
    if rel.is_absolute():
        raise InputError(f"absolute shard path is not allowed: {relative!r}")
    root = base.resolve()
    try:
        resolved = (root / rel).resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as e:
        raise InputError(
            f"shard path escapes model directory or is missing: {relative!r}") from e
    if not resolved.is_file():
        raise InputError(f"shard path is not a regular file: {relative!r}")
    return resolved


def _extract_tensor_state_dict(obj: Any, path: str) -> Dict[str, torch.Tensor]:
    """Unwrap common trusted PyTorch checkpoint containers."""
    if not isinstance(obj, dict):
        raise InputError(f"{path}: not a state-dict checkpoint")
    selected = obj
    for key in ("state_dict", "model", "module"):
        nested = obj.get(key)
        if isinstance(nested, dict) and any(isinstance(v, torch.Tensor) for v in nested.values()):
            selected = nested
            break
    tensors: Dict[str, torch.Tensor] = {}
    for name, value in selected.items():
        if isinstance(value, torch.Tensor):
            if not isinstance(name, str) or not name:
                raise InputError(f"{path}: tensor keys must be non-empty strings")
            tensors[name] = value.detach().cpu()
        else:
            log().debug("ignoring non-tensor pickle entry %r (%s)", name,
                        type(value).__name__)
    if not tensors:
        raise InputError(f"{path}: checkpoint contains no tensor state dict")
    return tensors


def _load_pickle_state_dict(path: str) -> Dict[str, torch.Tensor]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise InputError(f"failed to load pickle checkpoint {path}: {e}") from e
    return _extract_tensor_state_dict(obj, path)


def _discover_directory(d: Path, trust_pickle: bool,
                        index_path: Optional[Path] = None) -> CheckpointInfo:
    config: Dict[str, Any] = {}
    model_index: Dict[str, Any] = {}
    shard_index: Dict[str, Any] = {}
    index_candidates = [index_path] if index_path is not None else \
        sorted(d.glob("*.safetensors.index.json"))
    if len(index_candidates) > 1:
        preferred = d / "model.safetensors.index.json"
        if preferred in index_candidates:
            index_candidates = [preferred]
        else:
            raise InputError(
                f"multiple safetensors indexes found under {d}; pass the specific "
                "component directory instead")
    config_names = ["config.json", "model_index.json"]
    if index_candidates:
        config_names.append(index_candidates[0].name)
    for cfg in config_names:
        cand = d / cfg
        if cand.is_file():
            try:
                data = (_parse_index_json(str(cand))
                        if cfg.endswith(".safetensors.index.json")
                        else _parse_config_json(str(cand)))
            except Exception as e:
                raise InputError(f"cannot parse {cand}: {e}") from e
            if cfg == "config.json":
                config = data
            elif cfg == "model_index.json":
                model_index = data
            elif cfg.endswith(".safetensors.index.json"):
                shard_index = data

    # Respect the shard index when present (only its listed shards are read).
    if shard_index and "weight_map" in shard_index:
        weight_map = shard_index["weight_map"]
        if not isinstance(weight_map, dict) or not weight_map or not all(
                isinstance(k, str) and isinstance(v, str)
                for k, v in weight_map.items()):
            raise InputError("safetensors shard index has an invalid weight_map")
        safetensors_files = sorted({str(_resolve_under(d, rel))
                                    for rel in weight_map.values()})
    else:
        safetensors_files = sorted(str(f.resolve()) for f in d.rglob("*.safetensors")
                                   if f.is_file())
    if not safetensors_files:
        pickle_files = sorted(str(f) for f in d.rglob("*.bin"))
        pickle_files += sorted(str(f) for f in d.rglob("*.pt"))
        if pickle_files:
            if not trust_pickle:
                raise PickleInputError(
                    f"{d} contains pickle checkpoints ({pickle_files[0]}); pass "
                    "--trust-pickle for files you trust.")
            tensors: List[TensorMeta] = []
            metadata: Dict[str, str] = {}
            total = 0
            seen_pickle: set = set()
            for pf in pickle_files:
                sd = _load_pickle_state_dict(pf)
                for name, t in sd.items():
                    if name in seen_pickle:
                        raise InputError(f"duplicate tensor {name!r} across pickle shards")
                    seen_pickle.add(name)
                    meta = TensorMeta(name, t.dtype, tuple(t.shape), t.nbytes, pf, 0, t.nbytes)
                    tensors.append(meta)
                    total += t.nbytes
            cp = CheckpointInfo(kind="pickle", files=pickle_files, metadata=metadata,
                                tensors=tensors, config=config, model_index=model_index,
                                shard_index=shard_index, total_bytes=total)
            _flag_quantized_input(cp)
            return cp
        raise InputError(f"no safetensors or pickle weight files found under {d}")

    # Multi-file: build a logical tensor order from the shard index if available,
    # otherwise concatenate per-file entries (sorted) and reject duplicates.
    readers: Dict[str, RawSafetensorsFile] = {}
    try:
        for sf in safetensors_files:
            readers[str(Path(sf).resolve())] = RawSafetensorsFile(sf)
        physical_locations: Dict[str, List[str]] = {}
        for rf in readers.values():
            for tensor_name in rf.entries:
                physical_locations.setdefault(tensor_name, []).append(rf.path)
        duplicated = {
            name: paths for name, paths in physical_locations.items()
            if len(paths) > 1
        }
        if duplicated:
            duplicate_name = sorted(duplicated)[0]
            raise InputError(
                f"duplicate tensor {duplicate_name!r} across shards: "
                f"{duplicated[duplicate_name]}")
        metadata: Dict[str, str] = {}
        for rf in readers.values():
            for key, value in rf.metadata.items():
                if key in metadata and metadata[key] != value:
                    raise InputError(f"conflicting metadata key {key!r} across shards")
                metadata[key] = value
        tensors: List[TensorMeta] = []
        seen: set = set()
        if shard_index and "weight_map" in shard_index:
            for name, shard in shard_index["weight_map"].items():
                shard_path = str(_resolve_under(d, shard))
                rf = readers.get(shard_path)
                if rf is None:
                    raise InputError(f"shard index references missing file {shard!r}")
                if name in seen:
                    raise InputError(f"duplicate tensor {name!r} in shard index")
                dtype, shape, start, end = rf.get(name)
                tensors.append(TensorMeta(name, dtype, shape, end - start, rf.path, start, end))
                seen.add(name)
            for rf in readers.values():
                for name in sorted(rf.entries):
                    if name not in seen:
                        dtype, shape, start, end = rf.entries[name]
                        tensors.append(TensorMeta(name, dtype, shape, end - start,
                                                  rf.path, start, end))
                        seen.add(name)
        else:
            for rf in readers.values():
                for name in sorted(rf.entries):
                    if name in seen:
                        raise InputError(f"duplicate tensor {name!r} across shards")
                    seen.add(name)
                    dtype, shape, start, end = rf.entries[name]
                    tensors.append(TensorMeta(name, dtype, shape, end - start, rf.path, start, end))
        cp = CheckpointInfo(
            kind="sharded-safetensors" if len(readers) > 1 else "safetensors",
            files=safetensors_files, metadata=metadata, tensors=tensors,
            config=config, model_index=model_index, shard_index=shard_index,
            total_bytes=sum(t.nbytes for t in tensors),
        )
        _flag_quantized_input(cp)
        return cp
    finally:
        for rf in readers.values():
            rf.close()


def _read_pickle_checkpoint(path: str) -> CheckpointInfo:
    """Read a pickle checkpoint with explicit trust opt-in (full RAM load)."""
    sd = _load_pickle_state_dict(path)
    tensors: List[TensorMeta] = []
    total = 0
    for name, t in sd.items():
        tensors.append(TensorMeta(name, t.dtype, tuple(t.shape), t.nbytes, path, 0, t.nbytes))
        total += t.nbytes
    cp = CheckpointInfo(kind="pickle", files=[path], metadata={}, tensors=tensors,
                        total_bytes=total)
    _flag_quantized_input(cp)
    return cp


def _flag_quantized_input(cp: CheckpointInfo) -> None:
    keys = cp.key_set()
    if METADATA_KEY_QUANT in cp.metadata or METADATA_KEY_EXT in cp.metadata:
        cp.is_quantized_input = True
    if any(k.endswith(f".{LAYER_CONF_KEY}") for k in keys):
        cp.is_quantized_input = True


class CheckpointReader:
    """Lazy tensor accessor over a CheckpointInfo (keeps mmaps open)."""

    def __init__(self, info: CheckpointInfo):
        self.info = info
        self._files: Dict[str, RawSafetensorsFile] = {}
        self._pickle_sd: Optional[Dict[str, torch.Tensor]] = None
        if info.kind == "pickle":
            self._pickle_sd = {}
            for f in info.files:
                for name, tensor in _load_pickle_state_dict(f).items():
                    if name in self._pickle_sd:
                        raise InputError(f"duplicate tensor {name!r} across pickle shards")
                    self._pickle_sd[name] = tensor
        else:
            for f in info.files:
                self._files[f] = RawSafetensorsFile(f)

    def close(self) -> None:
        for rf in self._files.values():
            rf.close()
        self._files.clear()
        self._pickle_sd = None

    def __enter__(self) -> "CheckpointReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def read_tensor(self, name: str) -> torch.Tensor:
        meta = self.info.by_name(name)
        if meta is None:
            raise InputError(f"no tensor named {name!r}")
        if self._pickle_sd is not None:
            return self._pickle_sd[name]
        rf = self._files[meta.source]
        return rf.read_tensor(name)

    def read_bytes(self, name: str) -> memoryview:
        meta = self.info.by_name(name)
        if meta is None:
            raise InputError(f"no tensor named {name!r}")
        if self._pickle_sd is not None:
            return memoryview(tensor_to_bytes(self._pickle_sd[name]))
        rf = self._files[meta.source]
        return rf.read_bytes(name)



# ---------------------------------------------------------------------------
# ConvRot Hadamard rotation (port of comfy-kitchen tensor/int8_utils.py)
# ---------------------------------------------------------------------------
_HADAMARD_CACHE: Dict[Tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


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


# ---------------------------------------------------------------------------
# Shape validation (W4A8: port of validate_w4a8_weight_shape / operands)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Lloyd-Max codebook fitting (deterministic; port of comfy-kitchen _fit_codebook)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# W4A8 quantization (bit-exact port of comfy-kitchen eager backend)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
def quantize_weight_by_format(weight: torch.Tensor, fmt: str, group_size: int,
                              convrot_groupsize: int) -> Dict[str, torch.Tensor]:
    """Quantize with the W4A8 layout; returns the per-layer output tensors
    keyed by suffix ('' for the packed weight, '_s_rel', '_s_channel',
    '_codebook', optional '_correction')."""
    if fmt != FORMAT_W4A8:
        raise PolicyError(f"unknown quantization format {fmt!r}")
    packed, s_rel, s_ch, corr, cb = quantize_w4a8_weight(
        weight, group_size=group_size, convrot_groupsize=convrot_groupsize,
        symmetric=True, scale_dtype=torch.float8_e4m3fn, codebook=True)
    out = {"": packed, "_s_rel": s_rel, "_s_channel": s_ch, "_codebook": cb}
    if corr is not None:
        out["_correction"] = corr
    return out


def dequantize_weight_by_format(tensors: Dict[str, torch.Tensor], fmt: str,
                                group_size: int, convrot_groupsize: int,
                                output_dtype: torch.dtype) -> torch.Tensor:
    if fmt != FORMAT_W4A8:
        raise PolicyError(f"unknown quantization format {fmt!r}")
    return dequantize_w4a8_weight(
        tensors[""], tensors["_s_rel"], tensors["_s_channel"],
        codebook=tensors.get("_codebook"), correction=tensors.get("_correction"),
        group_size=group_size, convrot_groupsize=convrot_groupsize,
        output_dtype=output_dtype)



# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------
# Every architecture family supported by ComfyUI (research revision
# bdcb886a4705a03cf40f4a7226de9fc7c059fc90, 2026-08-06) is represented by its
# own named policy profile below.  Detection signatures are derived from
# comfy/model_detection.py detect_unet_config() and comfy/supported_models.py.
#
# Policy fields:
#   detect_primary : key substrings that identify the family (evidence)
#   detect_hints   : secondary evidence keys
#   quantize       : regex list matched against the full layer key (without the
#                    unet prefix and without the trailing ".weight"); a matching
#                    layer's weight is a quantization candidate
#   keep           : regex list for layers kept at original precision
#   exclude        : regex list for tensors never quantized (norms, embeddings,
#                    positionals, convs, buffers ...)
#   runtime_status : "verified" (reference-format runtime exists upstream),
#                    "experimental" (shared ComfyUI mixed-ops load path, but no
#                    upstream-produced W4A8 example), "unsupported" (no runtime
#                    quantization path in ComfyUI)
@dataclass(frozen=True)
class FamilyPolicy:
    family: str
    comfyui_classes: Tuple[str, ...]
    detect_primary: Tuple[str, ...]
    detect_hints: Tuple[str, ...] = ()
    quantize: Tuple[str, ...] = ()
    keep: Tuple[str, ...] = ()
    exclude: Tuple[str, ...] = ()
    group_size: int = 16
    min_weight_numel: int = 4096
    max_rel_l2: float = 0.25
    min_cosine: float = 0.95
    runtime_status: str = "experimental"
    notes: str = ""

    def quantize_re(self) -> re.Pattern:
        return flatten_regex(self.quantize)

    def keep_re(self) -> re.Pattern:
        return flatten_regex(self.keep)

    def exclude_re(self) -> re.Pattern:
        return flatten_regex(self.exclude)

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        return d


# Universal exclusions shared by every family (norms, embeddings, positionals,
# convs, biases, rope, buffers).  Weights matching these are never quantized.
UNIVERSAL_EXCLUDE = (
    r"(^|\.)(norm|norm1|norm2|norm3|ln\w*|layer_norm|rms_norm|final_norm|final_layer_norm|"
    r"q_norm|k_norm|query_norm|key_norm|pre_norm|post_norm|adaln_norm|norm_added_q|"
    r"prenorm|input_layernorm|self_attn_norm|post_attention_layernorm|"
    r"ln_final|ln1|ln2|ln_pre|ln_post|emb_norm|token_norm|txt_norm|ffn_norm1|ffn_norm2)"
    r"\.(weight|bias|scale|shift)$",
    r"(^|\.)(pos_embed|pos_embedding|positional_encoding|position_ids|"
    r"emb_pos|pos_emb|rotary_pos_emb|rope|inv_freq|freqs_cis|emb_tokens|"
    r"embed_positions|patch_embedding|patch_embed|adaln_t_table|cap_pad_token|"
    r"__x0__|__sequential__|memory_tokens|timestep_features)(\.|$)",
    r"(^|\.)(time_embed|t_embedder|timestep_embedder|time_embedder|time_embeddings|"
    r"ofs_embedding|fps_embedding|style_embedding|cond_embedding|guidance_in|"
    r"vector_in|txt_in|img_in|input_embedder|x_embedder|y_embedder|context_embedder|"
    r"patch_embedder|patchify_proj|video_patch_proj|audio_patch_proj|condition_proj|"
    r"cond_proj|cond_embed|input_proj|img_emb|text_proj|caption_proj|t5_yproj|"
    r"final_layer|output_layer|head|out_layer|final_linear|"
    r"audio_out|video_out|linear_fc2|to_gate_logits|genre_embedder|speaker_embedder|"
    r"lyric_proj|text_embedding|ref_image_patch_embedder|ofs_embedding_linear_1|"
    r"ofs_embedding_linear_2|time_embedding_linear_1|time_embedding_linear_2|"
    r"adaln_proj|adaln_modulation|adaLN_modulation|adaln_single|time_caption_embed|extra_embedder|"
    r"text_embedder|label_emb|clip_txt_mapper|clip_img_mapper|clip_mapper|"
    r"clip_txt_pooled_mapper|cond_type_embedding|distilled_guidance_layer|"
    r"control_adapter|ref_conv|latent_in|cond_in|cam_out_layer|repo_layers|"
    r"content_map|gate_map|final_map|input_layer|cam_enc|cam_dec|"
    r"visual_embeddings|time_embeddings|ofs_embedding_linear|patch_embedding_mask|"
    r"patch_embedding_pose|patch_embedding_global|emb|embed|embedding|"
    r"condition_embedder|adaln_curve|llm_cond_proj|input_embedder|pos_embed_proj|"
    r"txtfusion|visual_transformer_blocks|text_transformer_blocks|"
    r"encoder|decoder|lyric_encoder|ssl_|vocoder|first_stage|cond_stage)(\.|$)",
)

# Patterns for the classic SD1/SD2/SDXL/SVD UNet families.
UNET_ATTN_Q = (
    r"(input_blocks|output_blocks|middle_block)\.\d+(\.\d+)?\.transformer_blocks\.\d+\.attn[12]\.(to_q|to_k|to_v|to_out\.0)\.weight",
)
UNET_ATTN_K = (
    r"(input_blocks|output_blocks|middle_block)\.\d+(\.\d+)?\.transformer_blocks\.\d+\.attn[12]\.(to_q|to_k|to_v|to_out\.0)\.weight",
)
UNET_FF = (
    r"(input_blocks|output_blocks|middle_block)\.\d+(\.\d+)?\.transformer_blocks\.\d+\.ff\.net\.(0|2)\.(proj)?\.?weight$",
)
UNET_TIME_EMBED = (r"(^|\.)time_embed\.(0|2)\.weight$",)
UNET_LABEL_EMB = (r"(^|\.)label_emb\.0\.0\.weight$",)


def _sd_unet_policy(family: str, classes: Tuple[str, ...], notes: str = "") -> FamilyPolicy:
    return FamilyPolicy(
        family=family, comfyui_classes=classes,
        detect_primary=(),
        detect_hints=(),
        quantize=UNET_ATTN_Q + UNET_FF,
        keep=UNET_TIME_EMBED + UNET_LABEL_EMB + (
            r"(^|\.)(out\.\d+|output_blocks\.\d+\.\d+\.conv|input_blocks\.\d+\.\d+\.conv)\.weight$",
        ),
        exclude=UNIVERSAL_EXCLUDE,
        runtime_status="experimental",
        notes=notes,
    )


REGISTRY_ORDER: List[str] = []
REGISTRY: Dict[str, FamilyPolicy] = {}


def _register(policy: FamilyPolicy) -> None:
    REGISTRY[policy.family] = policy
    REGISTRY_ORDER.append(policy.family)


# --- mmdit (SD3 / SD3.5 family; joint_blocks) ---
_register(FamilyPolicy(
    family="mmdit_sd3",
    comfyui_classes=("SD3",),
    detect_primary=("joint_blocks.0.context_block.attn.qkv.weight",),
    detect_hints=("x_embedder.proj.weight", "final_layer.linear.weight", "y_embedder.mlp.0.weight"),
    quantize=(
        r"joint_blocks\.\d+\.(x_block|context_block)\.attn\.(qkv|proj)\.weight$",
        r"joint_blocks\.\d+\.(x_block|context_block)\.mlp\.(fc1|fc2)\.weight$",
    ),
    keep=(r"(^|\.)(x_embedder|y_embedder|context_embedder|final_layer|pos_embed)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
    notes="SD3 / SD3.5 MMDiT joint-block family (also covers sd3.5 medium/large).",
))

# --- stable cascade ---
_register(FamilyPolicy(
    family="stable_cascade",
    comfyui_classes=("Stable_Cascade_C", "Stable_Cascade_B"),
    detect_primary=("clf.1.weight",),
    detect_hints=("clip_txt_mapper.weight", "clip_mapper.weight", "clip_img_mapper.weight"),
    quantize=(
        r"blocks\.\d+\.attn\.(to_q|to_k|to_v|to_out\.0)\.weight$",
        r"blocks\.\d+\.ff\.net\.(0|2)\.(proj)?\.?weight$",
        r"blocks\.\d+\.ff\.(0|2)\.(proj)?\.?weight$",
        r"(^|\.)mapper\.weight$",
    ),
    keep=(r"(^|\.)(clip_txt_mapper|clip_img_mapper|clip_mapper|clip_txt_pooled_mapper)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
    notes="Würstchen stage B/C DiT.",
))

# --- stable audio ---
_register(FamilyPolicy(
    family="stable_audio",
    comfyui_classes=("StableAudio", "StableAudio3"),
    detect_primary=("transformer.rotary_pos_emb.inv_freq",),
    detect_hints=("to_global_embed.0.weight", "to_cond_embed.0.weight", "to_timestep_embed.0.weight"),
    quantize=(
        r"transformer\.layers\.\d+\.self_attn\.(to_qkv|to_q|to_k|to_v|to_out\.0)\.weight$",
        r"transformer\.layers\.\d+\.ff\.net\.(0|2)\.(proj)?\.?weight$",
        r"transformer\.layers\.\d+\.(to_local_embed|to_global_embed|to_cond_embed|to_timestep_embed)\.\d+\.weight$",
    ),
    keep=(r"(^|\.)(to_global_embed|to_cond_embed|to_timestep_embed|postprocess_conv|project_in|project_out|transformer\.project_in|transformer\.project_out|global_cond_embedder)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
    notes="Stable Audio 1/3 DiT (audio).",
))

# --- aura flow ---
_register(FamilyPolicy(
    family="aura_flow",
    comfyui_classes=("AuraFlow",),
    detect_primary=("double_layers.0.attn.w1q.weight",),
    detect_hints=("single_layers.0.attn.w1q.weight", "cond_seq_linear.weight", "positional_encoding"),
    quantize=(
        r"(double_layers|single_layers)\.\d+\.attn\.(w1q|w1k|w1v|w2|o_proj|w2q|w2k|w2v|w1o|w2o)\.weight$",
        r"(double_layers|single_layers)\.\d+\.mlp\.(c_fc1|c_fc2|c_proj)\.weight$",
    ),
    keep=(r"(^|\.)(cond_seq_linear|init_x_linear|final_linear|positional_encoding)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

# --- hunyuan dit ---
_register(FamilyPolicy(
    family="hydit",
    comfyui_classes=("HunyuanDiT", "HunyuanDiT1"),
    detect_primary=("mlp_t5.0.weight",),
    detect_hints=("blocks.0.attn.qkv.weight", "x_embedder.proj.weight", "extra_embedder.0.weight"),
    quantize=(
        r"blocks\.\d+\.attn\.(qkv|proj)\.weight$",
        r"blocks\.\d+\.mlp\.(fc1|fc2)\.weight$",
    ),
    keep=(r"(^|\.)(x_embedder|t_embedder|extra_embedder|text_embedder|final_layer|pos_embed|mlp_t5)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

# --- hunyuan video (and hunyuan image / video 1.5 variants) ---
_register(FamilyPolicy(
    family="hunyuan_video",
    comfyui_classes=("HunyuanVideo", "HunyuanVideoI2V", "HunyuanVideoSkyreelsI2V",
                     "HunyuanImage21", "HunyuanImage21Refiner", "HunyuanVideo15",
                     "HunyuanVideo15_SR_Distilled"),
    detect_primary=("txt_in.individual_token_refiner.blocks.0.norm1.weight",),
    detect_hints=("img_in.proj.weight", "final_layer.linear.weight",
                  "double_blocks.0.attn.qkv.weight", "txt_in.input_embedder.weight"),
    quantize=(
        r"double_blocks\.\d+\.attn\.(qkv|proj)\.weight$",
        r"single_blocks\.\d+\.attn\.(qkv|proj)\.weight$",
        r"double_blocks\.\d+\.mlp\.(fc1|fc2)\.weight$",
        r"single_blocks\.\d+\.mlp\.(fc1|fc2)\.weight$",
    ),
    keep=(r"(^|\.)(img_in|txt_in|vector_in|guidance_in|fps_embedding|style_embedding|"
          r"cond_embedding|txt_std|txt_emb|final_layer|individual_token_refiner|"
          r"byt5_in|time_r_in|vision_in|cond_type_embedding|time_embed|extra_embedder|"
          r"audio_embed|adaln_)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
    notes="HunyuanVideo 1.x + HunyuanImage 2.1 families share this DiT structure.",
))

# --- flux / chroma / flux2 / longcat / ovis ---
_register(FamilyPolicy(
    family="flux",
    comfyui_classes=("Flux", "FluxInpaint", "FluxSchnell", "LongCatImage"),
    detect_primary=("double_blocks.0.img_attn.norm.key_norm.weight",
                    "double_blocks.0.img_attn.norm.key_norm.scale"),
    detect_hints=("img_in.weight", "txt_in.weight", "single_blocks.0.linear1.weight",
                  "guidance_in.in_layer.weight", "vector_in.in_layer.weight"),
    quantize=(
        r"double_blocks\.\d+\.img_attn\.(qkv|proj)\.weight$",
        r"double_blocks\.\d+\.txt_attn\.(qkv|proj)\.weight$",
        r"double_blocks\.\d+\.img_mlp\.(w1|w2|gate_proj|up_proj|down_proj)\.weight$",
        r"double_blocks\.\d+\.txt_mlp\.(w1|w2|gate_proj|up_proj|down_proj)\.weight$",
        r"single_blocks\.\d+\.(linear1|linear2)\.weight$",
    ),
    keep=(r"(^|\.)(img_in|txt_in|vector_in|guidance_in|time_text_embed|txt_embed|"
          r"final_layer|distilled_guidance_layer|txt_norm)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="verified",
    notes="Flux family; runtime verified via the reference W4A8 pipeline (same "
          "double_blocks/single_blocks structure as MiniMax H3).",
))

_register(FamilyPolicy(
    family="flux2",
    comfyui_classes=("Flux2",),
    detect_primary=("double_stream_modulation_img.lin.weight",),
    detect_hints=("double_stream_layers.0.img_attn.qkv.weight", "img_in.weight"),
    quantize=(
        r"double_stream_layers\.\d+\.img_attn\.(qkv|proj)\.weight$",
        r"double_stream_layers\.\d+\.txt_attn\.(qkv|proj)\.weight$",
        r"double_stream_layers\.\d+\.img_mlp\.(w1|w2|gate_proj|up_proj|down_proj)\.weight$",
        r"double_stream_layers\.\d+\.txt_mlp\.(w1|w2|gate_proj|up_proj|down_proj)\.weight$",
        r"single_stream_layers\.\d+\.(linear1|linear2)\.weight$",
    ),
    keep=(r"(^|\.)(img_in|txt_in|vector_in|guidance_in|time_text_embed|final_layer)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

_register(FamilyPolicy(
    family="chroma",
    comfyui_classes=("Chroma", "ChromaRadiance"),
    detect_primary=("distilled_guidance_layer.norms.0.weight",),
    detect_hints=("distilled_guidance_layer.0.norms.0.weight", "nerf_blocks.0.norm.weight"),
    quantize=(
        r"(distilled_guidance_layer|double_blocks|single_blocks|nerf_blocks)\.\d+\.\w+\.(qkv|proj|w1|w2|linear1|linear2)\.weight$",
        r"(distilled_guidance_layer|double_blocks|single_blocks|nerf_blocks)\.\d+\.(img_attn|txt_attn|attn)\.(qkv|proj)\.weight$",
    ),
    keep=(r"(^|\.)(img_in|txt_in|vector_in|guidance_in|final_layer|nerf_final_layer|nerf_embedder|img_in_patch)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

# --- mochi ---
_register(FamilyPolicy(
    family="mochi",
    comfyui_classes=("GenmoMochi",),
    detect_primary=("t5_yproj.weight",),
    detect_hints=("time_blocks.0.attn.qkv_x.weight", "patch_embed.proj.weight"),
    quantize=(
        r"(time_blocks|t5_blocks)\.\d+\.attn\.(qkv_x|qkv_y|proj_x|proj_y)\.weight$",
        r"(time_blocks|t5_blocks)\.\d+\.mlp\.(w1|w2|fc1|fc2)\.weight$",
    ),
    keep=(r"(^|\.)(patch_embed|t5_yproj|final_layer|cond_embedder|timestep_embedder|pos_embed|mod)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

# --- minimax h3 (the reference W4A8 model family) ---
_register(FamilyPolicy(
    family="minimax_h3",
    comfyui_classes=("MiniMaxH3",),
    detect_primary=("video_patch_proj.weight", "audio_patch_proj.weight"),
    detect_hints=("blocks.0.attn.qkv_proj.weight", "final_layer.video_out.weight",
                  "adaln_t_table", "rope.inv_freq"),
    quantize=(
        r"blocks\.\d+\.attn\.(qkv_proj|out_proj)\.weight$",
        r"blocks\.\d+\.mlp\.(fc1|fc2)\.weight$",
    ),
    keep=(r"(^|\.)(video_patch_proj|audio_patch_proj|condition_proj|final_layer|"
          r"adaln_proj|adaln_t_table|token_refiner|time_embedder|rope)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="verified",
    notes="Reference family: the Kijai W4A8 test model quantizes exactly "
          "attn.qkv_proj / attn.out_proj / mlp.fc1 / mlp.fc2 per block.",
))

# --- ltxv / ltxav ---
_register(FamilyPolicy(
    family="ltxv",
    comfyui_classes=("LTXV", "LTXAV"),
    detect_primary=("adaln_single.emb.timestep_embedder.linear_1.bias",),
    detect_hints=("transformer_blocks.0.attn2.to_k.weight", "audio_adaln_single.linear.weight"),
    quantize=(
        r"transformer_blocks\.\d+\.attn[12]\.(to_q|to_k|to_v|to_out\.0)\.weight$",
        r"transformer_blocks\.\d+\.ff\.net\.(0|2)\.(proj)?\.?weight$",
        r"transformer_blocks\.\d+\.(attn1|attn2)\.(to_q|to_k|to_v|to_out\.0)\.weight$",
        r"transformer_blocks\.\d+\.ff\.(0|2)\.weight$",
    ),
    keep=(r"(^|\.)(patchify_proj|time_embed|cond_proj|caption_proj|proj_out|adaln_single|"
          r"audio_adaln_single|pos_embed|rope)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

# --- ace-step (audio) ---
_register(FamilyPolicy(
    family="ace_step",
    comfyui_classes=("ACEStep", "ACEStep15"),
    detect_primary=("genre_embedder.weight",),
    detect_hints=("encoder.lyric_encoder.layers.0.input_layernorm.weight",
                  "decoder.layers.0.self_attn.q_proj.weight"),
    quantize=(
        r"(encoder|decoder)\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$",
        r"(encoder|decoder)\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)\.weight$",
    ),
    keep=(r"(^|\.)(genre_embedder|speaker_embedder|lyric_proj|ssl_|enc|dec)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
    notes="ACE-Step music diffusion (audio).",
))

# --- pixart ---
_register(FamilyPolicy(
    family="pixart",
    comfyui_classes=("PixArtAlpha", "PixArtSigma"),
    detect_primary=("t_block.1.weight",),
    detect_hints=("blocks.0.attn.qkv.weight", "x_embedder.proj.weight",
                  "y_embedder.y_embedding", "ar_embedder.mlp.0.weight"),
    quantize=(
        r"blocks\.\d+\.attn[12]\.(qkv|to_q|to_k|to_v|proj|to_out\.0)\.weight$",
        r"blocks\.\d+\.ff\.net\.(0|2)\.(proj)?\.?weight$",
        r"blocks\.\d+\.mlp\.(fc1|fc2|w1|w2)\.weight$",
    ),
    keep=(r"(^|\.)(x_embedder|y_embedder|t_block|pos_embed|final_layer|csize_embedder|"
          r"ar_embedder|pe_interpolation)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

# --- cosmos ---
_register(FamilyPolicy(
    family="cosmos",
    comfyui_classes=("CosmosT2V", "CosmosI2V"),
    detect_primary=("blocks.block0.blocks.0.block.attn.to_q.0.weight",),
    detect_hints=("x_embedder.proj.1.weight", "adaln_lora"),
    quantize=(
        r"blocks\.block\d+\.blocks\.\d+\.block\.attn\.(to_q|to_k|to_v|to_out)\.\d+\.weight$",
        r"blocks\.block\d+\.blocks\.\d+\.block\.mlp\.(w1|w2|fc1|fc2)\.weight$",
        r"blocks\.block\d+\.blocks\.\d+\.block\.cross_attn\.(to_q|to_k|to_v|to_out)\.\d+\.weight$",
    ),
    keep=(r"(^|\.)(x_embedder|t_embedder|adaln|pos_emb|final_layer|patch_embed|cond_embed|"
          r"cross_attn_norm|norm)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

_register(FamilyPolicy(
    family="cosmos_predict2",
    comfyui_classes=("CosmosT2IPredict2", "CosmosI2VPredict2"),
    detect_primary=("blocks.0.mlp.layer1.weight",),
    detect_hints=("x_embedder.proj.1.weight",),
    quantize=(
        r"blocks\.\d+\.attn\.(q_proj|k_proj|v_proj|output_proj)\.weight$",
        r"blocks\.\d+\.mlp\.(layer1|layer2)\.weight$",
    ),
    keep=(r"(^|\.)(x_embedder|t_embedder|adaln|pos_emb|final_layer)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

# --- anima ---
_register(FamilyPolicy(
    family="anima",
    comfyui_classes=("Anima",),
    detect_primary=("__x0__",),
    detect_hints=("layers.0.attn.q_proj.weight",),
    quantize=(
        r"layers\.\d+\.attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$",
        r"layers\.\d+\.mlp\.(w1|w2|fc1|fc2)\.weight$",
    ),
    keep=(r"(^|\.)(x_embedder|t_embedder|cond_embed|final_layer|modulation)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

# --- lumina2 / zimage ---
_register(FamilyPolicy(
    family="lumina2",
    comfyui_classes=("Lumina2", "ZImage", "ZImagePixelSpace"),
    detect_primary=("cap_embedder.1.weight",),
    detect_hints=("noise_refiner.0.attention.k_norm.weight", "layers.0.attn.qkv.weight",
                  "layers.0.attention.qkv.weight", "dec_net.cond_embed.weight"),
    quantize=(
        r"layers\.\d+\.attn\.(qkv|o_proj|proj)\.weight$",
        r"layers\.\d+\.mlp\.(w1|w2|fc1|fc2)\.weight$",
        # real Lumina2 / Z-Image naming (comfy/ldm/lumina/model.py):
        # JointTransformerBlock.attention.{qkv,out}, FeedForward.{w1,w2,w3}
        r"(layers|context_refiner|noise_refiner)\.\d+\.attention\.(qkv|out)\.weight$",
        r"(layers|context_refiner|noise_refiner)\.\d+\.feed_forward\.(w1|w2|w3)\.weight$",
    ),
    keep=(r"(^|\.)(cap_embedder|clip_text_pooled_proj|siglip_embedder|x_embedder|t_embedder|"
          r"cond_embed|final_layer|dec_net|pos_embed|adaLN_modulation)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

# --- pixeldit / pid ---
_register(FamilyPolicy(
    family="pixeldit",
    comfyui_classes=("PixelDiTT2I", "PiD"),
    detect_primary=("core.pixel_embedder.proj.weight", "lq_proj.latent_proj.0.weight"),
    detect_hints=("cap_embedder.1.weight", "noise_refiner.0.attention.k_norm.weight",
                  "x_embedder.proj.weight"),
    quantize=(
        r"core\.(blocks|transformer_blocks)\.\d+\.(attn|attention)\.(qkv|to_q|to_k|to_v|proj|to_out\.0)\.weight$",
        r"core\.(blocks|transformer_blocks)\.\d+\.mlp\.(w1|w2|fc1|fc2)\.weight$",
    ),
    keep=(r"(^|\.)(core\.pixel_embedder|cap_embedder|noise_refiner|x_embedder|t_embedder|"
          r"final_layer|pos_embed|lq_proj)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

# --- wan ---
_register(FamilyPolicy(
    family="wan",
    comfyui_classes=("WAN21_T2V", "WAN21_CausalAR_T2V", "WAN21_I2V", "WAN21_FunControl2V",
                     "WAN21_Camera", "WAN22_Camera", "WAN21_Vace", "WAN21_HuMo",
                     "WAN22_S2V", "WAN22_Animate", "WAN22_T2V", "WAN21_FlowRVS",
                     "WAN21_SCAIL", "WAN21_SCAIL2", "WAN22_WanDancer"),
    detect_primary=("head.modulation",),
    detect_hints=("blocks.0.self_attn.q.weight", "blocks.0.cross_attn.k.weight",
                  "patch_embedding.weight", "txt_embedding.weight"),
    quantize=(
        r"blocks\.\d+\.(self_attn|cross_attn)\.(q|k|v|o)\.weight$",
        r"blocks\.\d+\.feed_forward\.(w1|w2)\.weight$",
        r"blocks\.\d+\.ffn\.(0|2)\.weight$",
        r"blocks\.\d+\.(self_attn|cross_attn)\.(q_img|k_img|v_img|o_img)\.weight$",
    ),
    keep=(r"(^|\.)(patch_embedding|text_embedding|time_embedding|time_projection|"
          r"final_linear|head|before_proj|after_proj|audio_proj|cond_embedding|"
          r"vace_patch_embedding|control_adapter|img_emb|face_adapter|latent_in|"
          r"patch_embedding_mask|patch_embedding_pose|patch_embedding_global)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

# --- hunyuan3d ---
_register(FamilyPolicy(
    family="hunyuan3d",
    comfyui_classes=("Hunyuan3Dv2", "Hunyuan3Dv2_1", "Hunyuan3Dv2mini"),
    detect_primary=("latent_in.weight",),
    detect_hints=("x_embedder.weight", "cond_in.weight", "blocks.0.attn.to_q.weight"),
    quantize=(
        r"blocks\.\d+\.attn\.(to_q|to_k|to_v|to_out\.0|qkv|proj)\.weight$",
        r"blocks\.\d+\.mlp\.(fc1|fc2|w1|w2)\.weight$",
    ),
    keep=(r"(^|\.)(latent_in|cond_in|x_embedder|t_embedder|final_layer|pos_embed|mod)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

# --- triposplat ---
_register(FamilyPolicy(
    family="triposplat",
    comfyui_classes=("TripoSplat",),
    detect_primary=("cam_out_layer.weight",),
    detect_hints=("repo_layers.0.final_map.weight", "cond_embedder.weight"),
    quantize=(
        r"(cam_enc|cam_dec|repo_layers)\.\d+\.(qkv|to_q|to_k|to_v|proj|fc1|fc2|w1|w2|final_map|content_map|gate_map)\.weight$",
    ),
    keep=(r"(^|\.)(cond_embedder|input_layer|out_layer|cam_out_layer|pos_embed)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

# --- hidream ---
_register(FamilyPolicy(
    family="hidream",
    comfyui_classes=("HiDream", "HiDreamO1"),
    detect_primary=("t_embedder1.mlp.0.weight",),
    detect_hints=("x_embedder.proj1.weight", "caption_projection.0.linear.weight"),
    quantize=(
        r"(visual|text)_transformer_blocks\.\d+\.(attn|attention)\.(qkv|to_q|to_k|to_v|proj|to_out\.0)\.weight$",
        r"(visual|text)_transformer_blocks\.\d+\.mlp\.(w1|w2|fc1|fc2)\.weight$",
        r"double_stream_blocks\.\d+\.(attn|mlp)\.\w+\.weight$",
        r"single_stream_blocks\.\d+\.(attn|mlp)\.\w+\.weight$",
    ),
    keep=(r"(^|\.)(x_embedder|t_embedder|caption_projection|time_embed|pos_embed|final_layer|cond_embed)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

# --- seedvr2 ---
_register(FamilyPolicy(
    family="seedvr2",
    comfyui_classes=("SeedVR2",),
    detect_primary=("cap_embedder.1.weight",),
    detect_hints=("noise_refiner.0.attention.k_norm.weight",
                  "x_embedder.proj.1.weight", "lq_proj.gate_modules.0.content_proj.weight"),
    quantize=(
        r"blocks\.\d+\.attn\.(qkv|proj)\.weight$",
        r"blocks\.\d+\.mlp\.(fc1|fc2|w1|w2)\.weight$",
        r"noise_refiner\.\d+\.attention\.(qkv|proj)\.weight$",
    ),
    keep=(r"(^|\.)(x_embedder|cap_embedder|noise_refiner|t_embedder|final_layer|pos_embed|lq_proj)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
    notes="SeedVR2 shares Lumina2-like signatures; disambiguated by hints.",
))

# --- omnigen2 ---
# Real OmniGen2 naming (BAAI/OmniGen2 state dict): layers.N.attn.to_q / to_k /
# to_v / to_out.0, layers.N.feed_forward.linear_1/2/3, and the same structure
# inside context_refiner / noise_refiner / ref_image_refiner.
_register(FamilyPolicy(
    family="omnigen2",
    comfyui_classes=("Omnigen2",),
    detect_primary=("time_caption_embed.timestep_embedder.linear_1.bias",
                    "layers.0.attn.to_q.weight"),
    detect_hints=("layers.0.feed_forward.linear_1.weight", "context_refiner.0.attn.to_q.weight",
                  "ref_image_patch_embedder.weight", "x_embedder.weight"),
    quantize=(
        r"layers\.\d+\.attn\.(to_q|to_k|to_v|to_out\.0)\.weight$",
        r"layers\.\d+\.feed_forward\.(linear_1|linear_2|linear_3)\.weight$",
        r"(context_refiner|noise_refiner|ref_image_refiner)\.\d+\.attn\.(to_q|to_k|to_v|to_out\.0)\.weight$",
        r"(context_refiner|noise_refiner|ref_image_refiner)\.\d+\.feed_forward\.(linear_1|linear_2|linear_3)\.weight$",
    ),
    keep=(r"(^|\.)(norm_out|image_index_embedding)\.", r"norm\d+\.linear\.(weight|bias)$"),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

# --- boogu ---
# Real Boogu-Image-0.1 naming (Base/Turbo/Edit, Comfy-Org repack verified):
# double_stream_layers.N.img_self_attn / img_instruct_attn(.processor) /
# img_feed_forward / instruct_feed_forward, single_stream_layers.N.attn /
# feed_forward, plus the OmniGen2-style refiners and embedders.
_register(FamilyPolicy(
    family="boogu",
    comfyui_classes=("Boogu",),
    detect_primary=("double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight",
                    "double_stream_layers.0.img_self_attn.to_q.weight",
                    "double_stream_layers.0.img_feed_forward.linear_1.weight"),
    detect_hints=("single_stream_layers.0.attn.to_q.weight",
                  "single_stream_layers.0.feed_forward.linear_1.weight",
                  "context_refiner.0.attn.to_q.weight",
                  "ref_image_patch_embedder.weight", "x_embedder.weight"),
    quantize=(
        r"double_stream_layers\.\d+\.(img_self_attn|img_instruct_attn)\.(to_q|to_k|to_v|to_out\.0)\.weight$",
        r"double_stream_layers\.\d+\.img_instruct_attn\.processor\.(img_to_q|img_to_k|img_to_v|img_out|instruct_to_q|instruct_to_k|instruct_to_v|instruct_out)\.weight$",
        r"double_stream_layers\.\d+\.(img_feed_forward|instruct_feed_forward)\.(linear_1|linear_2|linear_3)\.weight$",
        r"single_stream_layers\.\d+\.attn\.(to_q|to_k|to_v|to_out\.0)\.weight$",
        r"single_stream_layers\.\d+\.feed_forward\.(linear_1|linear_2|linear_3)\.weight$",
        r"(context_refiner|noise_refiner|ref_image_refiner)\.\d+\.attn\.(to_q|to_k|to_v|to_out\.0)\.weight$",
        r"(context_refiner|noise_refiner|ref_image_refiner)\.\d+\.feed_forward\.(linear_1|linear_2|linear_3)\.weight$",
    ),
    keep=(r"(^|\.)(norm_out|image_index_embedding|time_caption_embed)\.", r"norm\d+\.linear\.(weight|bias)$"),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
    notes="Real Boogu-Image-0.1 naming (double/single_stream_layers, img_instruct_attn.processor).",
))

# --- lens ---
_register(FamilyPolicy(
    family="lens",
    comfyui_classes=("Lens",),
    detect_primary=("transformer_blocks.0.attn.norm_added_q.weight",),
    detect_hints=("transformer_blocks.0.img_mlp.w1.weight", "img_in.weight"),
    quantize=(
        r"transformer_blocks\.\d+\.(img_attn|txt_attn)\.(qkv|proj)\.weight$",
        r"transformer_blocks\.\d+\.(img_mlp|txt_mlp)\.(w1|w2|gate_proj|up_proj|down_proj)\.weight$",
    ),
    keep=(r"(^|\.)(img_in|txt_in|proj_out|txt_norm|time_text_embed|final_layer)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

# --- mage flow ---
_register(FamilyPolicy(
    family="mage_flow",
    comfyui_classes=("MageFlow",),
    detect_primary=("txt_norm.weight",),
    detect_hints=("proj_out.weight", "transformer_blocks.0.img_attn.qkv.weight"),
    quantize=(
        r"transformer_blocks\.\d+\.(img_attn|txt_attn)\.(qkv|proj)\.weight$",
        r"transformer_blocks\.\d+\.(img_mlp|txt_mlp)\.(w1|w2|gate_proj|up_proj|down_proj)\.weight$",
    ),
    keep=(r"(^|\.)(img_in|txt_in|proj_out|txt_norm|time_text_embed|final_layer)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
    notes="Disambiguated from qwen_image by txt_norm dim 2560 / proj_out 128.",
))

# --- qwen image ---
_register(FamilyPolicy(
    family="qwen_image",
    comfyui_classes=("QwenImage",),
    detect_primary=("txt_norm.weight",),
    detect_hints=("proj_out.weight", "transformer_blocks.0.attn.to_q.weight",
                  "img_in.weight", "time_text_embed.addition_t_embedding.weight"),
    quantize=(
        r"transformer_blocks\.\d+\.attn\.(to_q|to_k|to_v|to_out\.0)\.weight$",
        r"transformer_blocks\.\d+\.(img_mlp|txt_mlp|mlp)\.(w1|w2|gate_proj|up_proj|down_proj)\.weight$",
    ),
    keep=(r"(^|\.)(img_in|txt_in|proj_out|txt_norm|time_text_embed|final_layer|"
          r"time_caption_embed|add_k_proj|add_q_proj|add_v_proj|to_add_out)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

# --- ideogram4 ---
_register(FamilyPolicy(
    family="ideogram4",
    comfyui_classes=("Ideogram4",),
    detect_primary=("embed_image_indicator.weight",),
    detect_hints=("input_proj.weight", "layers.0.attn.qkv.weight"),
    quantize=(
        r"layers\.\d+\.attn\.(qkv|o|proj)\.weight$",
        r"layers\.\d+\.mlp\.(mlp_in|mlp_out|w1|w2|fc1|fc2)\.weight$",
    ),
    keep=(r"(^|\.)(input_proj|embed_image_indicator|adaln_proj|adaln_modulation|"
          r"llm_cond_proj|time_embed|pos_embed|final_layer)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

# --- krea2 ---
_register(FamilyPolicy(
    family="krea2",
    comfyui_classes=("Krea2",),
    detect_primary=("txtfusion.projector.weight",),
    detect_hints=("txtfusion.layerwise_blocks.0.prenorm.scale",
                  "layers.0.attn.wq.weight"),
    quantize=(
        r"layers\.\d+\.attn\.(wq|wk|wv|wo)\.weight$",
        r"layers\.\d+\.mlp\.(w1|w2|gate_proj|up_proj|down_proj)\.weight$",
        r"txtfusion\.layerwise_blocks\.\d+\.(attn|mlp)\.\w+\.weight$",
    ),
    keep=(r"(^|\.)(first|projector|txtfusion\.projector|input_proj|time_embed|"
          r"pos_embed|final_layer|adaln)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

# --- kandinsky5 ---
_register(FamilyPolicy(
    family="kandinsky5",
    comfyui_classes=("Kandinsky5", "Kandinsky5Image"),
    detect_primary=("visual_transformer_blocks.0.cross_attention.key_norm.weight",),
    detect_hints=("visual_embeddings.in_layer.weight", "text_transformer_blocks.0.self_attention.q_norm.weight"),
    quantize=(
        r"(visual|text)_transformer_blocks\.\d+\.(self_attention|cross_attention)\.(to_query|to_key|to_value|to_out\.0)\.weight$",
        r"(visual|text)_transformer_blocks\.\d+\.ff\.(in_layer|out_layer)\.weight$",
    ),
    keep=(r"(^|\.)(visual_embeddings|time_embeddings|in_layer|out_layer|head|pos_embed)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

# --- cogvideox ---
_register(FamilyPolicy(
    family="cogvideox",
    comfyui_classes=("CogVideoX_T2V", "CogVideoX_I2V", "CogVideoX_Inpaint"),
    detect_primary=("blocks.0.norm1.linear.weight",),
    detect_hints=("patch_embed.proj.weight", "transformer_blocks.0.attn1.to_q.weight"),
    quantize=(
        r"transformer_blocks\.\d+\.attn[12]\.(to_q|to_k|to_v|to_out\.0)\.weight$",
        r"transformer_blocks\.\d+\.ff\.net\.(0|2)\.(proj)?\.?weight$",
    ),
    keep=(r"(^|\.)(patch_embed|time_embed|text_proj|proj_out|final_layer|ofs_embedding|"
          r"norm1\.linear|cond_embed)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

# --- ernie image ---
_register(FamilyPolicy(
    family="ernie_image",
    comfyui_classes=("ErnieImage",),
    detect_primary=("layers.0.mlp.linear_fc2.weight",),
    detect_hints=("text_proj.weight", "visual_transformer_blocks.0.feed_forward.in_layer.weight"),
    quantize=(
        r"layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$",
        r"layers\.\d+\.mlp\.(linear_fc1|linear_fc2)\.weight$",
        r"(visual|text)_transformer_blocks\.\d+\.(self_attention|cross_attention)\.(to_query|to_key|to_value|to_out\.0)\.weight$",
        r"(visual|text)_transformer_blocks\.\d+\.feed_forward\.(in_layer|out_layer)\.weight$",
    ),
    keep=(r"(^|\.)(text_proj|visual_embeddings|time_embeddings|patch_embed|final_linear|pos_embed)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

# --- classic SD UNets (shape-classified: sd15, sd20, sdxl, sdxl_refiner, svd, sd_x4) ---
_register(_sd_unet_policy(
    "sd15", ("SD15", "SD15_instructpix2pix", "Stable_Zero123"),
    "SD1.5 UNet (context_dim 768). Attention q/k/v/out are 1x1 convs in the "
    "non-linear-attention layout; only feed-forward linears plus linear-attention "
    "variants are quantized."))

_register(_sd_unet_policy(
    "sd20", ("SD20", "SD21UnclipL", "SD21UnclipH", "SD_X4Upscaler", "LotusD"),
    "SD2.x UNet (context_dim 1024, linear attention)."))

_register(_sd_unet_policy(
    "sdxl", ("SDXL", "SSD1B", "Segmind_Vega", "KOALA_700M", "KOALA_1B",
             "SDXL_instructpix2pix"),
    "SDXL / SSD-1B / Segmind-Vega / KOALA UNets (context_dim 2048)."))

_register(_sd_unet_policy(
    "sdxl_refiner", ("SDXLRefiner",),
    "SDXL refiner UNet (model_channels 384, context_dim 1280)."))

_register(_sd_unet_policy(
    "svd", ("SVD_img2vid", "SV3D_u", "SV3D_p"),
    "SVD / SV3D video UNets (temporal attention)."))

# --- joyimage ---
_register(FamilyPolicy(
    family="joyimage",
    comfyui_classes=("JoyImage",),
    detect_primary=("double_blocks.0.attn.img_attn_qkv.weight",
                    "double_blocks.0.attn.img_attn_q_norm.weight"),
    detect_hints=("condition_embedder.time_embedder.linear_1.weight", "img_in.weight"),
    quantize=(
        r"double_blocks\.\d+\.attn\.(img_attn_qkv|img_attn_proj|txt_attn_qkv|txt_attn_proj)\.weight$",
        r"double_blocks\.\d+\.mlp\.(fc1|fc2|w1|w2)\.weight$",
    ),
    keep=(r"(^|\.)(img_in|txt_in|condition_embedder|time_embedder|proj_out|"
          r"final_layer|pos_embed|adaln)\.",),
    exclude=UNIVERSAL_EXCLUDE,
    runtime_status="experimental",
))

# --- perception models with no ComfyUI quantized-loading path ---
for _fam, _cls in (("rt_detr_v4", ("RT_DETR_v4",)),
                   ("depth_anything3", ("DepthAnything3",)),
                   ("sam3", ("SAM3", "SAM31"))):
    _register(FamilyPolicy(
        family=_fam, comfyui_classes=_cls,
        detect_primary=(
            "encoder.pan_blocks.1.cv4.conv.weight",
            "backbone.embeddings.patch_embeddings.projection.weight",
            "backbone.encoder.layer.0.attention.q_norm.weight",
        ),
        detect_hints=(),
        quantize=(), keep=(), exclude=UNIVERSAL_EXCLUDE,
        runtime_status="unsupported",
        notes="Perception model: ComfyUI loads it through its own node, not through "
              "the mixed-precision (quantized) loader; W4A8 output would not "
              "be consumable. Conversion refused unless --architecture forces it.",
    ))


def family_names() -> List[str]:
    return list(REGISTRY_ORDER)


def get_family(name: str) -> FamilyPolicy:
    normalized = re.sub(r"[^a-z0-9]", "", name.lower())
    for family, policy in REGISTRY.items():
        aliases = (family,) + policy.comfyui_classes
        if any(re.sub(r"[^a-z0-9]", "", alias.lower()) == normalized
               for alias in aliases):
            return policy
    raise UnknownArchitectureError(
        f"unknown architecture {name!r}; use --list-architectures")



# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------
UNET_PREFIX_CANDIDATES = ("model.diffusion_model.", "model.model.", "net.")


def unet_prefix_from_keys(keys: Iterable[str]) -> str:
    counts = {c: 0 for c in UNET_PREFIX_CANDIDATES}
    for k in keys:
        for c in UNET_PREFIX_CANDIDATES:
            if k.startswith(c):
                counts[c] += 1
                break
    top = max(counts, key=counts.get)
    if counts[top] > 5:
        return top
    return "model."


@dataclass
class DetectionResult:
    architecture: str
    confidence: str                 # high | medium | low
    policy: FamilyPolicy
    unet_prefix: str
    evidence: List[str] = field(default_factory=list)
    hints: List[str] = field(default_factory=list)
    competing: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    classifier_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "architecture": self.architecture,
            "confidence": self.confidence,
            "unet_prefix": self.unet_prefix,
            "evidence": self.evidence,
            "hints": self.hints,
            "competing": self.competing,
            "warnings": self.warnings,
            "classifier_info": self.classifier_info,
        }


def _match_signatures(keys: Iterable[str], prefix: str,
                      signatures: Sequence[str]) -> List[str]:
    """Return the signature keys that appear as substrings of some state-dict key
    (with the unet prefix stripped)."""
    stripped = [k[len(prefix):] if k.startswith(prefix) else k for k in keys]
    found = []
    for sig in signatures:
        for k in stripped:
            if sig in k:
                found.append(sig)
                break
    return found


def detect_architecture(info: CheckpointInfo, override: Optional[str] = None,
                        shape_lookup: Optional[Callable[[str], Optional[Tuple[int, ...]]]] = None,
                        ) -> DetectionResult:
    """Detect the model architecture from checkpoint structure alone."""
    keys = list(info.key_set())
    prefix = unet_prefix_from_keys(keys)
    if override:
        policy = get_family(override)
        return DetectionResult(
            architecture=policy.family, confidence="high",
            policy=policy, unet_prefix=prefix,
            evidence=[f"user override --architecture {override}"],
            warnings=["architecture supplied by the user; detection was not used"])
    if shape_lookup is None:
        shape_lookup = lambda name: None  # noqa: E731

    # ---- structural families first (mirroring ComfyUI's branch order) ----
    stripped = [k[len(prefix):] if k.startswith(prefix) else k for k in keys]

    def has(*subs: str) -> bool:
        for s in subs:
            if not any(s in k for k in stripped):
                return False
        return True

    def any_of(*subs: str) -> bool:
        return any(s in k for k in stripped for s in subs)

    candidates: List[Tuple[str, List[str], List[str]]] = []  # (family, evidence, hints)

    # 1. mmdit (SD3 / SD3.5)
    if has("joint_blocks.0.context_block.attn.qkv.weight", "x_embedder.proj.weight"):
        candidates.append(("mmdit_sd3", ["joint_blocks.0.context_block.attn.qkv.weight",
                                         "x_embedder.proj.weight"],
                           [s for s in ("final_layer.linear.weight", "y_embedder.mlp.0.weight",
                                        "context_embedder.weight") if has(s)]))
    # 2. stable cascade
    if has("clf.1.weight"):
        candidates.append(("stable_cascade", ["clf.1.weight"],
                           [s for s in ("clip_txt_mapper.weight", "clip_mapper.weight",
                                        "clip_img_mapper.weight") if has(s)]))
    # 3. stable audio
    if has("transformer.rotary_pos_emb.inv_freq"):
        candidates.append(("stable_audio", ["transformer.rotary_pos_emb.inv_freq"],
                           [s for s in ("to_global_embed.0.weight", "to_timestep_embed.0.weight") if has(s)]))
    # 4. aura flow
    if has("double_layers.0.attn.w1q.weight"):
        candidates.append(("aura_flow", ["double_layers.0.attn.w1q.weight"],
                           [s for s in ("single_layers.0.attn.w1q.weight", "cond_seq_linear.weight") if has(s)]))
    # 5. hydit
    if has("mlp_t5.0.weight"):
        candidates.append(("hydit", ["mlp_t5.0.weight"],
                           [s for s in ("blocks.0.attn.qkv.weight", "x_embedder.proj.weight",
                                        "extra_embedder.0.weight") if has(s)]))
    # 6. hunyuan video / image
    if has("txt_in.individual_token_refiner.blocks.0.norm1.weight"):
        candidates.append(("hunyuan_video", ["txt_in.individual_token_refiner.blocks.0.norm1.weight"],
                           [s for s in ("img_in.proj.weight", "final_layer.linear.weight",
                                        "double_blocks.0.attn.qkv.weight") if has(s)]))
    # 7. flux / chroma / flux2
    if any_of("double_blocks.0.img_attn.norm.key_norm.weight",
              "double_blocks.0.img_attn.norm.key_norm.scale"):
        if has("double_stream_modulation_img.lin.weight"):
            candidates.append(("flux2", ["double_stream_modulation_img.lin.weight"],
                               [s for s in ("double_stream_layers.0.img_attn.qkv.weight",) if has(s)]))
        elif any_of("distilled_guidance_layer.norms.0.weight",
                    "distilled_guidance_layer.0.norms.0.weight"):
            candidates.append(("chroma", ["distilled_guidance_layer.norms.0.weight"],
                               [s for s in ("nerf_blocks.0.norm.weight", "img_in.weight") if has(s)]))
        else:
            candidates.append(("flux", ["double_blocks.0.img_attn.norm.key_norm.weight"],
                               [s for s in ("img_in.weight", "txt_in.weight",
                                            "single_blocks.0.linear1.weight") if has(s)]))
    # 8. mochi
    if has("t5_yproj.weight"):
        candidates.append(("mochi", ["t5_yproj.weight"],
                           [s for s in ("time_blocks.0.attn.qkv_x.weight",) if has(s)]))
    # 9. minimax h3 (checked before ltxv, like ComfyUI)
    if has("video_patch_proj.weight", "audio_patch_proj.weight"):
        candidates.append(("minimax_h3", ["video_patch_proj.weight", "audio_patch_proj.weight"],
                           [s for s in ("blocks.0.attn.qkv_proj.weight", "final_layer.video_out.weight",
                                        "adaln_t_table") if has(s)]))
    # 10. ltxv / ltxav
    if has("adaln_single.emb.timestep_embedder.linear_1.bias"):
        candidates.append(("ltxv", ["adaln_single.emb.timestep_embedder.linear_1.bias"],
                           [s for s in ("transformer_blocks.0.attn2.to_k.weight",
                                        "audio_adaln_single.linear.weight") if has(s)]))
    # 11. ace-step
    if has("genre_embedder.weight"):
        candidates.append(("ace_step", ["genre_embedder.weight"],
                           [s for s in ("encoder.lyric_encoder.layers.0.input_layernorm.weight",) if has(s)]))
    # 12. pixart
    if has("t_block.1.weight"):
        candidates.append(("pixart", ["t_block.1.weight"],
                           [s for s in ("blocks.0.attn.qkv.weight", "x_embedder.proj.weight",
                                        "y_embedder.y_embedding") if has(s)]))
    # 13. cosmos
    if has("blocks.block0.blocks.0.block.attn.to_q.0.weight"):
        candidates.append(("cosmos", ["blocks.block0.blocks.0.block.attn.to_q.0.weight"],
                           [s for s in ("x_embedder.proj.1.weight",) if has(s)]))
    # 14. PiD (checked before PixelDiT)
    if has("lq_proj.latent_proj.0.weight"):
        candidates.append(("pixeldit", ["lq_proj.latent_proj.0.weight"],
                           [s for s in ("lq_proj.gate_modules.0.content_proj.weight",
                                        "lq_proj.pit_head.weight") if has(s)]))
    # 15. PixelDiT T2I
    if has("core.pixel_embedder.proj.weight"):
        candidates.append(("pixeldit", ["core.pixel_embedder.proj.weight"],
                           [s for s in ("cap_embedder.1.weight", "noise_refiner.0.attention.k_norm.weight") if has(s)]))
    # 16. lumina2 / zimage
    if has("cap_embedder.1.weight") and has("noise_refiner.0.attention.k_norm.weight"):
        candidates.append(("lumina2", ["cap_embedder.1.weight"],
                           [s for s in ("layers.0.attn.qkv.weight", "cap_pad_token",
                                        "dec_net.cond_embed.weight") if has(s)]))
    # 17. cogvideox
    if has("blocks.0.norm1.linear.weight"):
        candidates.append(("cogvideox", ["blocks.0.norm1.linear.weight"],
                           [s for s in ("patch_embed.proj.weight", "transformer_blocks.0.attn1.to_q.weight") if has(s)]))
    # 18. wan
    if has("head.modulation"):
        candidates.append(("wan", ["head.modulation"],
                           [s for s in ("blocks.0.self_attn.q.weight", "blocks.0.cross_attn.k.weight",
                                        "patch_embedding.weight") if has(s)]))
    # 19. seedvr2 (must be checked before generic lumina2-ish catches; ComfyUI order)
    if any_of("blocks.35.mlp.vid.proj_out.weight", "blocks.35.mlp.all.proj_in_gate.weight",
              "blocks.31.mlp.all.proj_in_gate.weight"):
        candidates.append(("seedvr2", ["blocks.35.mlp.vid.proj_out.weight"
                                       if has("blocks.35.mlp.vid.proj_out.weight")
                                       else "blocks.35.mlp.all.proj_in_gate.weight"],
                           [s for s in ("x_embedder.proj.1.weight",) if has(s)]))
    # 20. cosmos predict2 / anima
    if has("blocks.0.mlp.layer1.weight"):
        if has("__x0__"):
            candidates.append(("anima", ["blocks.0.mlp.layer1.weight", "__x0__"],
                               [s for s in ("layers.0.attn.q_proj.weight",) if has(s)]))
        else:
            candidates.append(("cosmos_predict2", ["blocks.0.mlp.layer1.weight"],
                               [s for s in ("x_embedder.proj.1.weight",) if has(s)]))
    # 21. boogu (checked before omnigen2; both share the embedder skeleton)
    if has("double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight",
           "double_stream_layers.0.img_self_attn.to_q.weight",
           "double_stream_layers.0.img_feed_forward.linear_1.weight"):
        candidates.append(("boogu",
                           ["double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight",
                            "double_stream_layers.0.img_self_attn.to_q.weight",
                            "double_stream_layers.0.img_feed_forward.linear_1.weight"],
                           [s for s in ("single_stream_layers.0.attn.to_q.weight",
                                        "single_stream_layers.0.feed_forward.linear_1.weight",
                                        "context_refiner.0.attn.to_q.weight",
                                        "ref_image_patch_embedder.weight") if has(s)]))
    # 22. omnigen2
    if has("time_caption_embed.timestep_embedder.linear_1.bias", "layers.0.attn.to_q.weight"):
        candidates.append(("omnigen2",
                           ["time_caption_embed.timestep_embedder.linear_1.bias",
                            "layers.0.attn.to_q.weight"],
                           [s for s in ("layers.0.feed_forward.linear_1.weight",
                                        "context_refiner.0.attn.to_q.weight",
                                        "ref_image_patch_embedder.weight",
                                        "x_embedder.weight") if has(s)]))
    # 23. lens
    if has("transformer_blocks.0.attn.norm_added_q.weight",
           "transformer_blocks.0.img_mlp.w1.weight"):
        candidates.append(("lens", ["transformer_blocks.0.attn.norm_added_q.weight"],
                           [s for s in ("img_in.weight", "proj_out.weight") if has(s)]))
    # 24. mage flow (shape-disambiguated from qwen image)
    if has("txt_norm.weight") and has("proj_out.weight"):
        tn_shape = shape_lookup(prefix + "txt_norm.weight")
        po_shape = shape_lookup(prefix + "proj_out.weight")
        if (tn_shape is not None and len(tn_shape) == 1 and tn_shape[0] == 2560
                and po_shape is not None and len(po_shape) == 1 and po_shape[0] == 128):
            candidates.append(("mage_flow", ["txt_norm.weight", "proj_out.weight"],
                               [s for s in ("transformer_blocks.0.img_attn.qkv.weight",) if has(s)]))
    # 25. qwen image
    if has("txt_norm.weight") and has("proj_out.weight") and not any(c[0] == "mage_flow" for c in candidates):
        candidates.append(("qwen_image", ["txt_norm.weight", "proj_out.weight"],
                           [s for s in ("img_in.weight", "transformer_blocks.0.attn.to_q.weight",
                                        "time_text_embed.addition_t_embedding.weight") if has(s)]))
    # 26. ideogram4
    if has("embed_image_indicator.weight"):
        candidates.append(("ideogram4", ["embed_image_indicator.weight"],
                           [s for s in ("input_proj.weight", "layers.0.attn.qkv.weight") if has(s)]))
    # 27. krea2
    if has("txtfusion.projector.weight"):
        candidates.append(("krea2", ["txtfusion.projector.weight"],
                           [s for s in ("txtfusion.layerwise_blocks.0.prenorm.scale",
                                        "layers.0.attn.wq.weight") if has(s)]))
    # 28. kandinsky5
    if has("visual_transformer_blocks.0.cross_attention.key_norm.weight"):
        candidates.append(("kandinsky5", ["visual_transformer_blocks.0.cross_attention.key_norm.weight"],
                           [s for s in ("visual_embeddings.in_layer.weight",) if has(s)]))
    # 29. ace 1.5 (music)
    if has("encoder.lyric_encoder.layers.0.input_layernorm.weight") and not any(c[0] == "ace_step" for c in candidates):
        candidates.append(("ace_step", ["encoder.lyric_encoder.layers.0.input_layernorm.weight"],
                           [s for s in ("decoder.layers.0.self_attn.q_proj.weight",) if has(s)]))
    # 30. RT-DETR / DepthAnything / SAM3 (perception: unsupported)
    if has("encoder.pan_blocks.1.cv4.conv.weight"):
        candidates.append(("rt_detr_v4", ["encoder.pan_blocks.1.cv4.conv.weight"], []))
    if has("backbone.embeddings.patch_embeddings.projection.weight"):
        candidates.append(("depth_anything3", ["backbone.embeddings.patch_embeddings.projection.weight"],
                           [s for s in ("head.scratch.refinenet1.out_conv.weight",) if has(s)]))
    if has("backbone.encoder.layer.0.attention.q_norm.weight") or has("backbone.encoder.layer.0.attention.self.query.weight"):
        candidates.append(("sam3", ["backbone.encoder.layer.0.attention.q_norm.weight"],
                           [s for s in ("head.projects.0.weight",) if has(s)]))
    # 31. ernie image
    if has("layers.0.mlp.linear_fc2.weight"):
        candidates.append(("ernie_image", ["layers.0.mlp.linear_fc2.weight"],
                           [s for s in ("text_proj.weight",) if has(s)]))
    # 32. classic SD UNets
    sd_family = None
    if "input_blocks.0.0.weight" in stripped:
        mc_shape = shape_lookup(prefix + "input_blocks.0.0.weight")
        model_channels = int(mc_shape[0]) if mc_shape else None
        # context dim from the first transformer block's cross-attention to_k
        ctx_dim = None
        lin_attn = False
        for probe in ("input_blocks.1.0.transformer_blocks.0.attn2.to_k.weight",
                      "input_blocks.1.0.transformer_blocks.0.attn2.q.weight",
                      "input_blocks.2.0.transformer_blocks.0.attn2.to_k.weight"):
            shp = shape_lookup(prefix + probe)
            if shp is not None:
                ctx_dim = int(shp[1]) if len(shp) == 2 else None
                lin_attn = probe.endswith("to_k.weight")
                break
        in_ch = None
        in_shp = shape_lookup(prefix + "input_blocks.0.0.weight")
        if in_shp is not None:
            in_ch = int(in_shp[1])
        temporal = any(("time_stack" in k or "temporal_transformer" in k) for k in stripped)
        has_label_emb = has("label_emb.0.0.weight")
        # Local config is supporting evidence only.  Tensor shapes remain the
        # primary signal, but a standard HF field can resolve a missing probe.
        cfg_ctx = info.config.get("cross_attention_dim") or info.config.get("context_dim")
        if ctx_dim is None and isinstance(cfg_ctx, int):
            ctx_dim = int(cfg_ctx)
        if in_ch is None and isinstance(info.config.get("in_channels"), int):
            in_ch = int(info.config["in_channels"])
        classifier_info = {"model_channels": model_channels, "context_dim": ctx_dim,
                           "in_channels": in_ch, "linear_attention": lin_attn,
                           "temporal": temporal, "label_emb": has_label_emb,
                           "config_keys": sorted(info.config.keys())[:32]}
        if model_channels == 256:
            sd_family = "sd20"  # SD_X4Upscaler
        elif ctx_dim == 2048:
            sd_family = "sdxl"
        elif ctx_dim == 1280:
            sd_family = "sdxl_refiner"
        elif ctx_dim == 1024:
            sd_family = "svd" if (temporal or in_ch == 8) else "sd20"
        elif ctx_dim == 768:
            sd_family = "sd15"
        elif model_channels == 384:
            sd_family = "sdxl_refiner"
        if sd_family is not None:
            candidates.append((sd_family, ["input_blocks.0.0.weight"],
                               ["config.cross_attention_dim"]
                               if cfg_ctx is not None else []))

    # ---- score candidates ----
    best: Optional[Tuple[str, int, List[str], List[str]]] = None
    competing = []
    for fam, ev, hints in candidates:
        score = 2 * len(ev) + len(hints)
        if best is None or score > best[1]:
            best = (fam, score, ev, hints)
            competing = [fam]
        elif score == best[1] and fam != best[0]:
            competing.append(fam)

    if best is None:
        # try generic signature matching against the registry for anything missed
        for fam in REGISTRY_ORDER:
            pol = REGISTRY[fam]
            ev = _match_signatures(keys, prefix, pol.detect_primary)
            if ev:
                hints = _match_signatures(keys, prefix, pol.detect_hints)
                score = 2 * len(ev) + len(hints)
                if best is None or score > best[1]:
                    best = (fam, score, ev, hints)
                    competing = [fam]
                elif score == best[1] and fam != best[0]:
                    competing.append(fam)

    if best is None:
        raise UnknownArchitectureError(
            "could not identify the model architecture from checkpoint structure. "
            "Use --architecture NAME (see --list-architectures) to supply one "
            "explicitly, or --inspect to review the checkpoint contents.")

    fam, score, ev, hints = best
    competing = [c for c in competing if c != fam]
    if competing:
        choices = ", ".join([fam] + sorted(set(competing)))
        raise UnknownArchitectureError(
            f"architecture detection is ambiguous between: {choices}. "
            "Pass --architecture NAME after inspecting the checkpoint; refusing "
            "to guess.")
    policy = REGISTRY[fam]
    if len(ev) >= 2:
        confidence = "high"
    elif len(ev) == 1 and len(hints) >= 1:
        confidence = "medium"
    else:
        confidence = "low"
    warnings = []
    if confidence == "low":
        warnings.append("low detection confidence; consider --architecture override")
    if policy.runtime_status == "unsupported":
        warnings.append(
            f"architecture {fam!r} has no ComfyUI quantized-loading path; conversion "
            "would not be consumable by ComfyUI (refusing unless forced)")

    result = DetectionResult(
        architecture=fam, confidence=confidence, policy=policy,
        unet_prefix=prefix, evidence=ev, hints=hints,
        competing=competing, warnings=warnings,
        classifier_info=locals().get("classifier_info", {}))
    return result


# ---------------------------------------------------------------------------
# Tensor classification / conversion planning
# ---------------------------------------------------------------------------
class DecisionKind(enum.Enum):
    QUANTIZE = "quantize"
    KEEP = "keep"              # passthrough at original precision
    KEEP_PRECISION = "keep_precision"   # kept for sensitivity / policy reasons


@dataclass
class TensorDecision:
    name: str                      # input tensor name (state-dict key)
    kind: DecisionKind
    reason: str
    layer: Optional[str] = None    # layer name (key minus ".weight") when quantized
    group_size: int = 16
    convrot_groupsize: int = 256
    out_dtype: Optional[torch.dtype] = None   # passthrough cast target (if any)
    quantized: bool = False        # True once actually quantized


@dataclass
class ConversionPlan:
    fmt: str
    detection: DetectionResult
    decisions: List[TensorDecision]
    metadata_quant: Dict[str, Any]     # _quantization_metadata payload
    metadata_ext: Dict[str, Any]       # comfy_wxa8 extension payload
    output_entries: List[Dict[str, Any]]  # (name, dtype, shape) in write order
    total_in_bytes: int = 0
    total_out_bytes: int = 0
    n_quantized: int = 0
    n_kept: int = 0
    device: str = "cpu"
    chunked_layers: set = field(default_factory=set)   # layers quantized via the
                                                       # bounded-memory chunked path

    def quantized_layers(self) -> List[TensorDecision]:
        return [d for d in self.decisions if d.kind == DecisionKind.QUANTIZE]


def classify_tensors(info: CheckpointInfo, detection: DetectionResult,
                     fmt: str, group_size: Optional[int], include: Sequence[str],
                     exclude: Sequence[str], keep_precision: Sequence[str],
                     output_dtype: Optional[torch.dtype],
                     min_numel: Optional[int]) -> List[TensorDecision]:
    """Decide, for every input tensor, whether it is quantized or passed through."""
    policy = detection.policy
    prefix = detection.unet_prefix
    # effective prefix: if the detected prefix does not actually prefix any key
    # (e.g. prefix-less models like MiniMax H3 where ComfyUI falls back to
    # "model."), classify against the full keys
    if not any(k.startswith(prefix) for k in info.key_set()):
        prefix = ""
    if group_size is None:
        group_size = policy.group_size
    if min_numel is None:
        min_numel = policy.min_weight_numel

    include_re = flatten_regex(list(include)) if include else None
    exclude_re = flatten_regex(list(exclude)) if exclude else None
    keep_re_user = flatten_regex(list(keep_precision)) if keep_precision else None
    q_re = policy.quantize_re()
    k_re = policy.keep_re()
    x_re = policy.exclude_re()

    decisions: List[TensorDecision] = []
    for meta in info.tensors:
        name = meta.name
        if not name.startswith(prefix):
            decisions.append(TensorDecision(name, DecisionKind.KEEP,
                                            "outside the diffusion-model prefix; passthrough"))
            continue
        rel = name[len(prefix):] if name.startswith(prefix) else name
        if not rel.endswith(".weight"):
            decisions.append(TensorDecision(name, DecisionKind.KEEP,
                                            "not a weight tensor (bias/norm/buffer); passthrough"))
            continue
        # layer = FULL state-dict key minus ".weight" (includes the unet prefix),
        # matching the names ComfyUI uses in _quantization_metadata / comfy_quant
        layer = name[:-len(".weight")]

        # user-level filters first
        if exclude_re is not None and exclude_re.search(name):
            decisions.append(TensorDecision(name, DecisionKind.KEEP,
                                            "matched --exclude pattern; passthrough"))
            continue
        if keep_re_user is not None and keep_re_user.search(name):
            decisions.append(TensorDecision(name, DecisionKind.KEEP_PRECISION,
                                            "matched --keep-precision pattern; passthrough"))
            continue

        # policy exclusions (norms / embeddings / positionals / heads / embedders)
        if x_re.search(rel):
            decisions.append(TensorDecision(name, DecisionKind.KEEP,
                                            "policy exclude (universal); passthrough"))
            continue
        if k_re.search(rel):
            decisions.append(TensorDecision(name, DecisionKind.KEEP_PRECISION,
                                            f"policy keep for {policy.family}; passthrough"))
            continue

        # quantize patterns are written against full keys (including ".weight")
        candidate = bool(q_re.search(rel)) if policy.quantize else True
        if include_re is not None and include_re.search(name):
            candidate = True
        if not candidate:
            decisions.append(TensorDecision(name, DecisionKind.KEEP,
                                            f"not in {policy.family} quantize set; passthrough"))
            continue

        # shape / dtype gates. ConvRot is always 256-wide: the comfy-kitchen
        # 0.2.27 CUDA fused kernels throw unless convrot_groupsize == 256, so
        # layers whose K is not divisible by 256 pass through at original
        # precision (they cannot run on the CUDA runtime with a smaller group).
        ok, why = w4_weight_is_quantizable(meta.shape, meta.dtype, group_size,
                                           W4A8_CONVROT_GROUPSIZE)
        if not ok:
            decisions.append(TensorDecision(name, DecisionKind.KEEP, f"not quantizable: {why}; passthrough"))
            continue
        if meta.nbytes < min_numel * meta.dtype.itemsize:
            decisions.append(TensorDecision(name, DecisionKind.KEEP,
                                            f"small tensor (<{min_numel} elements); passthrough"))
            continue
        decisions.append(TensorDecision(
            name, DecisionKind.QUANTIZE, "policy candidate",
            layer=layer, group_size=group_size,
            convrot_groupsize=W4A8_CONVROT_GROUPSIZE))

    return decisions


def build_output_entries(info: CheckpointInfo, decisions: List[TensorDecision],
                         fmt: str, output_dtype: Optional[torch.dtype],
                         ) -> Tuple[List[Dict[str, Any]], int]:
    """Compute the exact output tensor inventory (name, dtype, shape, nbytes)
    in deterministic write order: quantized layers first (original weight slot),
    then passthrough tensors in input order."""
    if fmt != FORMAT_W4A8:
        raise PolicyError(f"unknown quantization format {fmt!r}")
    entries: List[Dict[str, Any]] = []
    total = 0
    seen = set()
    for d in decisions:
        if d.kind == DecisionKind.QUANTIZE:
            layer = d.layer
            meta = info.by_name(d.name)
            n, k = int(meta.shape[0]), int(meta.shape[1])
            q_shape = (n, k // 2)
            groups = k // d.group_size
            base = f"{layer}.weight"
            extras = [
                (f"{layer}.weight_s_rel", torch.float8_e4m3fn, (n, groups)),
                (f"{layer}.weight_s_channel", torch.float32, (n,)),
                (f"{layer}.weight_codebook", torch.float32, (16,)),
            ]
            for ename, edtype, eshape in [(base, torch.int8, q_shape)] + extras:
                if ename in seen:
                    raise PolicyError(f"duplicate output tensor {ename!r}")
                seen.add(ename)
                nb = tensor_nbytes(edtype, eshape)
                entries.append({"name": ename, "dtype": edtype, "shape": eshape, "nbytes": nb})
                total += nb
        else:
            meta = info.by_name(d.name)
            dt = meta.dtype
            if output_dtype is not None and dt in FLOAT_DTYPES:
                dt = output_dtype
            if d.name in seen:
                raise PolicyError(f"duplicate output tensor {d.name!r}")
            seen.add(d.name)
            nb = tensor_nbytes(dt, meta.shape)
            entries.append({"name": d.name, "dtype": dt, "shape": meta.shape, "nbytes": nb})
            total += nb
    return entries, total



# ---------------------------------------------------------------------------
# Calibration (standalone; framework-independent local data)
# ---------------------------------------------------------------------------
# Calibration is OPTIONAL.  The W4A8 reference format is calibration-free
# (per-group absmax scales); calibration data is used here only for
#  (a) activation-aware sensitivity analysis (keep-precision selection) and
#  (b) provenance reporting.
# Accepted calibration sources (never downloaded automatically):
#   * a .npz file with arrays named exactly like the linear layer keys
#     (full state-dict names, e.g. "model.diffusion_model.input_blocks.3.0..."
#     or "blocks.0.attn.qkv_proj.weight"); each array is [S, K] activations.
#   * a .pt file with a dict {layer_key: tensor}
#   * a directory containing any mix of the above
#   * a .json manifest: {"samples": N, "layers": {key: {"path": ..., "rows": N}}}
@dataclass
class CalibrationStats:
    source: str
    files: List[str]
    n_samples: int
    layers: Dict[str, Dict[str, Any]]      # key -> {samples: Tensor[S,K], rows:int}
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "files": self.files,
            "n_samples": self.n_samples,
            "layers": {k: {
                "rows": v["rows"],
                "features": int(v["samples"].shape[1]),
                "absmax_mean": float(v["samples"].abs().amax(dim=0).mean().item()),
            } for k, v in self.layers.items()},
            "provenance": self.provenance,
        }


def _load_npz_calibration(path: str, layer_keys: set,
                          max_rows: int) -> Dict[str, torch.Tensor]:
    out = {}
    with np.load(path, allow_pickle=False) as data:
        for k in data.files:
            arr = data[k]
            if arr.ndim != 2:
                continue
            if k in layer_keys or (k + ".weight") in layer_keys:
                out[k] = torch.from_numpy(
                    np.asarray(arr[:max_rows], dtype=np.float32))
    return out


def _load_pt_calibration(path: str, layer_keys: set,
                         max_rows: int) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict):
        raise CalibrationError(f"{path}: calibration .pt must contain a dict")
    out = {}
    for k, v in obj.items():
        if isinstance(v, torch.Tensor) and v.ndim == 2 and (k in layer_keys or (k + ".weight") in layer_keys):
            out[k] = v.detach()[:max_rows].float().cpu()
    return out


def _check_calibration_file_budget(path: str, max_memory: int) -> int:
    """Reject calibration files that can expand beyond the working budget."""
    file_size = os.path.getsize(path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) > 100_000:
                raise CalibrationError(
                    f"{path}: archive has too many members ({len(members)})")
            expanded = [member.file_size for member in members]
        # The loader retains selected activation rows from multiple members;
        # bounding the total expanded archive prevents a many-member zip bomb
        # even when each individual array would fit.
        required = sum(expanded)
    else:
        required = file_size
    if required > max_memory:
        raise CalibrationError(
            f"{path}: calibration data may expand to {human_bytes(required)}, "
            f"above --max-memory {human_bytes(max_memory)}")
    return required


def load_calibration(source: str, info: CheckpointInfo, max_samples: Optional[int],
                     cache_path: Optional[str],
                     max_memory: int = 2 * 1024**3) -> CalibrationStats:
    """Load local calibration activations for the checkpoint's linear layers."""
    effective_max_samples = 64 if max_samples is None else int(max_samples)
    if effective_max_samples <= 0:
        raise CalibrationError("--calibration-samples must be positive")
    layer_keys = set()
    for t in info.tensors:
        if t.name.endswith(".weight"):
            layer_keys.add(t.name)
    # fast path: load a precomputed statistics cache before touching the source
    if cache_path is not None and os.path.exists(cache_path):
        try:
            stats = {}
            prov: Dict[str, Any] = {}
            _check_calibration_file_budget(cache_path, max_memory)
            cache_hash_before = sha256_file(cache_path)
            cache_fd, _ = _open_regular_nofollow(cache_path, os.O_RDONLY)
            with os.fdopen(cache_fd, "rb") as cache_probe:
                is_npz = cache_probe.read(4) == b"PK\x03\x04"
                cache_probe.seek(0)
                if is_npz:
                    with np.load(cache_probe, allow_pickle=False) as payload:
                        if "__provenance__" in payload.files:
                            prov = json.loads(
                                np.asarray(payload["__provenance__"], dtype=np.uint8)
                                .tobytes().decode("utf-8"))
                        for key in payload.files:
                            if key == "__provenance__" or key not in layer_keys:
                                continue
                            array = payload[key]
                            tensor = torch.from_numpy(np.asarray(
                                array[:effective_max_samples], dtype=np.float32))
                            meta = info.by_name(key)
                            if tensor.ndim == 2 and tensor.shape[0] > 0 and meta is not None \
                                    and tensor.shape[1] == meta.shape[1]:
                                tensor = tensor.contiguous()
                                stats[key] = {"samples": tensor,
                                              "rows": int(tensor.shape[0])}
            if not is_npz:
                # Backward-compatible reader for early JSON caches.  v1.1
                # summary-only caches are ignored because they cannot reproduce
                # activation-aware output error.
                if os.path.getsize(cache_path) > min(64 * 1024 * 1024,
                                                     max(1, max_memory // 4)):
                    raise CalibrationError(
                        f"{cache_path}: JSON cache exceeds the working-memory limit")
                payload = _load_json_object(
                    cache_path, "calibration cache", nofollow=True)
                prov = payload.get("provenance", {})
                for key, value in payload.get("stats", {}).items():
                    samples = value.get("samples")
                    if samples is None or key not in layer_keys:
                        continue
                    tensor = torch.as_tensor(
                        samples[:effective_max_samples], dtype=torch.float32)
                    meta = info.by_name(key)
                    if tensor.ndim == 2 and tensor.shape[0] > 0 and meta is not None \
                            and tensor.shape[1] == meta.shape[1]:
                        tensor = tensor.contiguous()
                        stats[key] = {"samples": tensor,
                                      "rows": int(tensor.shape[0])}
            cache_hash_after = sha256_file(cache_path)
            if cache_hash_after != cache_hash_before:
                raise CalibrationError(
                    "calibration cache changed while it was being read")
            if not isinstance(prov, dict):
                raise CalibrationError("calibration cache provenance must be an object")
            prov = dict(prov)
            prov["cache_file_sha256"] = cache_hash_after
            if stats:
                log().info("calibration activation rows loaded from cache %s", cache_path)
                return CalibrationStats(source=source, files=[cache_path],
                                        n_samples=max(v["rows"] for v in stats.values()),
                                        layers=stats, provenance=prov)
        except Exception as e:
            log().warning("could not read calibration cache %s: %s", cache_path, e)

    src_path = Path(source)
    if not src_path.exists():
        raise CalibrationError(f"calibration source not found: {source}")

    files: List[str] = []
    tensors: Dict[str, torch.Tensor] = {}
    if src_path.is_dir():
        for f in sorted(src_path.iterdir()):
            if f.suffix in (".npz", ".pt", ".npy"):
                files.append(str(f))
    else:
        files = [str(src_path)]

    manifest: Optional[Dict[str, Any]] = None
    if src_path.is_file() and src_path.suffix == ".json":
        manifest = _load_json_object(str(src_path), "calibration manifest")
        for entry in manifest.get("layers", {}).values():
            raw_path = entry.get("path", "")
            if not isinstance(raw_path, str) or not raw_path:
                raise CalibrationError("calibration manifest contains an invalid path")
            if Path(raw_path).is_absolute():
                raise CalibrationError(
                    "absolute paths are not allowed inside calibration manifests")
            p = _resolve_under(src_path.parent, raw_path)
            if p.suffix in (".npz", ".pt", ".npy") and str(p) not in files:
                files.append(str(p))

    expanded_total = 0
    for f in files:
        expanded_total += _check_calibration_file_budget(f, max_memory)
        if expanded_total > max_memory:
            raise CalibrationError(
                "combined calibration data may expand to "
                f"{human_bytes(expanded_total)}, above --max-memory "
                f"{human_bytes(max_memory)}")
    calibration_hashes = {f: sha256_file(f) for f in files}
    for f in files:
        try:
            if f.endswith(".npz"):
                data = _load_npz_calibration(
                    f, layer_keys, effective_max_samples)
            elif f.endswith(".pt"):
                data = _load_pt_calibration(
                    f, layer_keys, effective_max_samples)
            elif f.endswith(".npy"):
                arr = np.load(f, allow_pickle=False, mmap_mode="r")
                if arr.ndim == 2:
                    # single-layer file: name derived from the file name
                    name = Path(f).stem
                    if name in layer_keys or (name + ".weight") in layer_keys:
                        data = {name: torch.from_numpy(np.asarray(
                            arr[:effective_max_samples], dtype=np.float32))}
                    else:
                        data = {}
                else:
                    data = {}
            else:
                data = {}
        except Exception as e:
            raise CalibrationError(
                f"failed to read calibration file {f}: {e}") from e
        for k, v in data.items():
            canonical = k if k in layer_keys else k + ".weight"
            meta = info.by_name(canonical)
            if meta is None or len(meta.shape) != 2:
                continue
            if int(v.shape[1]) != int(meta.shape[1]):
                raise CalibrationError(
                    f"{f}: activation width for {canonical} is {v.shape[1]}, "
                    f"expected {meta.shape[1]}")
            part = v.detach()[:effective_max_samples].float().cpu()
            if canonical in tensors:
                remaining = effective_max_samples - int(tensors[canonical].shape[0])
                if remaining > 0:
                    tensors[canonical] = torch.cat(
                        (tensors[canonical], part[:remaining]), dim=0)
            else:
                tensors[canonical] = part.contiguous()
    final_calibration_hashes = {f: sha256_file(f) for f in files}
    if final_calibration_hashes != calibration_hashes:
        raise CalibrationError(
            "one or more calibration files changed while they were being read")

    if not tensors:
        raise CalibrationError(
            f"no usable calibration activations found in {source} (expected arrays "
            "named exactly like the checkpoint's linear layer keys)")

    stats: Dict[str, Dict[str, Any]] = {}
    for k, v in tensors.items():
        stats[k] = {"samples": v.contiguous(), "rows": int(v.shape[0])}

    provenance = {
        "source": source,
        "files": files,
        "n_files": len(files),
        "n_layers_with_data": len(stats),
        "max_samples_per_layer": effective_max_samples,
        "method": "recorded local activation rows used directly for output-error analysis",
        "synthetic": False,
        "file_sha256": final_calibration_hashes,
    }

    if cache_path:
        try:
            parent = os.path.dirname(os.path.abspath(cache_path)) or "."
            os.makedirs(parent, exist_ok=True)
            fd, cache_tmp = tempfile.mkstemp(
                prefix=f".{Path(cache_path).name}.", suffix=".tmp", dir=parent)
            try:
                arrays = {key: value["samples"].numpy()
                          for key, value in stats.items()}
                arrays["__provenance__"] = np.frombuffer(
                    json_dumps(provenance).encode("utf-8"), dtype=np.uint8)
                with os.fdopen(fd, "wb") as cache_file:
                    np.savez_compressed(cache_file, **arrays)
                    cache_file.flush()
                    os.fsync(cache_file.fileno())
                os.replace(cache_tmp, cache_path)
                _fsync_parent(cache_path)
            except Exception:
                with contextlib.suppress(OSError):
                    os.close(fd)
                with contextlib.suppress(OSError):
                    os.unlink(cache_tmp)
                raise
            log().info("calibration activation rows cached to %s", cache_path)
        except Exception as e:
            log().warning("could not write calibration cache %s: %s", cache_path, e)

    return CalibrationStats(source=source, files=files,
                            n_samples=max(v["rows"] for v in stats.values()),
                            layers=stats, provenance=provenance)



# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------
@dataclass
class TensorMetrics:
    name: str
    rel_l2: float
    snr_db: float
    cosine: float
    act_rel_l2: Optional[float] = None      # activation-aware (when calibration exists)
    kept: bool = False
    reason: str = ""


def compute_weight_metrics(original: torch.Tensor, dequant: torch.Tensor) -> TensorMetrics:
    o = original.float()
    d = dequant.float()
    signal = float(o.square().sum())
    error = float((d - o).square().sum())
    reconstructed = float(d.square().sum())
    dot = float((o * d).sum())
    if not all(math.isfinite(value)
               for value in (signal, error, reconstructed, dot)):
        return TensorMetrics(name="", rel_l2=1e30, snr_db=-300.0, cosine=-1.0)
    rel_l2 = math.sqrt(error / max(signal, 1e-12))
    snr = 300.0 if error <= 1e-30 else 10.0 * math.log10(
        max(signal, 1e-30) / error)
    if signal <= 1e-30 and reconstructed <= 1e-30:
        cos = 1.0
    else:
        cosine_denom = math.sqrt(max(signal * reconstructed, 1e-30))
        cos = max(-1.0, min(1.0, dot / cosine_denom))
    return TensorMetrics(name="", rel_l2=rel_l2, snr_db=snr, cosine=cos)


def activation_aware_error(original: torch.Tensor, dequant: torch.Tensor,
                           activations: Optional[torch.Tensor]) -> Optional[float]:
    """mean over samples of ||(Q(W)-W) x|| / ||W x|| using recorded activations."""
    if activations is None:
        return None
    o = original.float()
    d = dequant.float()
    delta = d - o
    x = activations.float()
    num = (delta @ x.t()).norm(dim=0)
    den = (o @ x.t()).norm(dim=0).clamp(min=1e-8)
    value = float((num / den).mean())
    return value if math.isfinite(value) else 1e30


class SensitivityAnalyzer:
    """Decides keep-precision for candidate layers based on metrics + thresholds."""

    def __init__(self, threshold: Optional[float], error_threshold: float,
                 calibration: Optional[CalibrationStats]):
        self.threshold = threshold
        self.error_threshold = error_threshold
        self.calibration = calibration
        self.results: Dict[str, TensorMetrics] = {}

    def evaluate(self, name: str, original: torch.Tensor, dequant: torch.Tensor,
                 ) -> TensorMetrics:
        m = compute_weight_metrics(original, dequant)
        m.name = name
        if self.calibration is not None:
            acts = None
            for key in (name, name[:-len(".weight")]):
                if key in self.calibration.layers:
                    acts = self.calibration.layers[key]["samples"]
                    break
            m.act_rel_l2 = activation_aware_error(original, dequant, acts)
        self.results[name] = m
        return m

    def decide_keep(self, m: TensorMetrics) -> Tuple[bool, str]:
        reasons = []
        if m.rel_l2 > self.error_threshold:
            reasons.append(f"relL2 {m.rel_l2:.4f} > error-threshold {self.error_threshold}")
        if self.threshold is not None:
            score = m.act_rel_l2 if (m.act_rel_l2 is not None and self.calibration is not None) else m.rel_l2
            if score > self.threshold:
                reasons.append(f"sensitivity {score:.4f} > threshold {self.threshold}")
        return (bool(reasons), "; ".join(reasons))



# ---------------------------------------------------------------------------
# Streaming safetensors writer (bounded memory, atomic finalization)
# ---------------------------------------------------------------------------
class SafetensorsStreamWriter:
    """Writes a .safetensors file incrementally.

    The header is computed up front (all output tensor names / dtypes / shapes /
    byte sizes are known after planning), then tensor data is streamed in
    exactly the planned order.  Supports resume via an external state file.
    """

    def __init__(self, path: str, entries: List[Dict[str, Any]],
                 resume_offsets: Optional[Dict[str, int]] = None,
                 resume_mode: bool = False,
                 expected_identity: Optional[Tuple[int, int]] = None):
        self.path = path
        self.entries = list(entries)
        # deterministic write order
        self.order = [e["name"] for e in self.entries]
        self._by_name = {e["name"]: e for e in self.entries}
        self._resume_offsets = resume_offsets or {}
        self._resume_mode = resume_mode
        self._expected_identity = expected_identity
        self._offsets: Dict[str, int] = {}
        self._header = self._build_header()
        # convert relative (data-relative) offsets to absolute file offsets
        self._offsets = {n: len(self._header) + rel for n, rel in self._offsets.items()}
        self._fh: Optional[Any] = None
        self._pos = 0
        self.identity: Optional[Tuple[int, int]] = None

    def _build_header(self) -> bytes:
        header: Dict[str, Any] = {"__metadata__": {}}
        offset = 0
        # Safetensors data offsets are contiguous.  The JSON header itself is
        # padded to an 8-byte boundary; inserting holes between tensors would
        # make otherwise-valid odd-byte tensors unreadable by strict loaders.
        for e in self.entries:
            start = offset
            end = start + e["nbytes"]
            header[e["name"]] = {
                "dtype": TORCH_TO_SAFE[e["dtype"]],
                "shape": list(e["shape"]),
                "data_offsets": [start, end],
            }
            self._offsets[e["name"]] = start
            offset = end
        self._data_size = offset
        raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(raw) > MAX_SAFETENSORS_HEADER_SIZE:
            raise OutputError(
                "output header exceeds the safetensors 100 MB safety limit")
        raw += b" " * ((-len(raw)) % 8)
        return struct.pack("<Q", len(raw)) + raw

    def header_bytes(self) -> bytes:
        return self._header

    def data_size(self) -> int:
        return self._data_size

    def open(self) -> None:
        if self._fh is not None:
            return
        if os.path.exists(self.path):
            if not self._resume_mode:
                raise OutputError(f"output file already exists: {self.path} (use --overwrite)")
            fd, st = _open_regular_nofollow(self.path, os.O_RDWR)
            identity = (int(st.st_dev), int(st.st_ino))
            if self._expected_identity is not None and identity != self._expected_identity:
                os.close(fd)
                raise OutputError(
                    f"resume temp identity changed for {self.path}; refusing possible "
                    "symlink/hardlink replacement")
            self._fh = os.fdopen(fd, "r+b")
            self.identity = identity
            expected_size = len(self._header) + self._data_size
            if st.st_size != expected_size:
                self.close()
                raise OutputError(
                    f"resume temp size mismatch: {st.st_size} != {expected_size}")
            self._fh.seek(0)
            if self._fh.read(len(self._header)) != self._header:
                self.close()
                raise OutputError("resume temp header does not match the conversion plan")
            self._pos = expected_size
        else:
            if self._resume_mode:
                raise OutputError(f"resume temp file missing: {self.path}")
            fd, st = _open_regular_nofollow(
                self.path, os.O_RDWR | os.O_CREAT | os.O_EXCL)
            self._fh = os.fdopen(fd, "w+b")
            self.identity = (int(st.st_dev), int(st.st_ino))
            self._fh.write(self._header)
            self._fh.truncate(len(self._header) + self._data_size)
            self._pos = len(self._header) + self._data_size

    def write_tensor_bytes(self, name: str, data: bytes) -> int:
        """Write one complete tensor at its planned offset."""
        self.open()
        if self._fh is None:
            raise OutputError("output writer failed to open")
        e = self._by_name[name]
        expected_off = self._offsets[name]     # deterministic, header-derived
        if self._resume_offsets and name in self._resume_offsets:
            # already written by a previous run: the recorded offset must equal
            # the header-derived offset (a real interruption writes strictly in
            # plan order, so this also detects corrupted/out-of-order states)
            off = self._resume_offsets[name]
            if off != expected_off:
                raise OutputError(
                    f"resume state inconsistent for {name}: recorded offset {off} "
                    f"!= header offset {expected_off}")
            self._fh.seek(off + e["nbytes"])
            self._pos = off + e["nbytes"]
            return off
        if len(data) != e["nbytes"]:
            raise OutputError(f"tensor {name}: expected {e['nbytes']} bytes, got {len(data)}")
        self._fh.seek(expected_off)
        self._fh.write(data)
        self._pos = expected_off + e["nbytes"]
        return expected_off

    def write_tensor_slice(self, name: str, byte_offset: int, data: bytes) -> int:
        """Write a byte slice inside a tensor, enabling bounded row-chunk output."""
        self.open()
        if self._fh is None:
            raise OutputError("output writer failed to open")
        e = self._by_name[name]
        if byte_offset < 0 or byte_offset + len(data) > e["nbytes"]:
            raise OutputError(
                f"tensor slice for {name} exceeds planned range: "
                f"{byte_offset}+{len(data)} > {e['nbytes']}")
        absolute = self._offsets[name] + byte_offset
        self._fh.seek(absolute)
        self._fh.write(data)
        self._pos = absolute + len(data)
        return absolute

    def offset_for(self, name: str) -> int:
        return self._offsets[name]

    def invalidate_resume_tensor(self, name: str) -> None:
        """Allow a partially completed logical layer to rewrite this tensor."""
        self._resume_offsets.pop(name, None)

    def tensor_sha256(self, name: str, chunk: int = 1 << 20) -> str:
        self.open()
        if self._fh is None:
            raise OutputError("output writer failed to open")
        entry = self._by_name[name]
        self._fh.flush()
        self._fh.seek(self._offsets[name])
        remaining = entry["nbytes"]
        digest = hashlib.sha256()
        while remaining:
            data = self._fh.read(min(chunk, remaining))
            if not data:
                raise OutputError(f"resume temp is truncated inside tensor {name!r}")
            digest.update(data)
            remaining -= len(data)
        return digest.hexdigest()

    def current_pos(self) -> int:
        return self._pos

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def finalize(self, final_path: str) -> None:
        """Flush, fsync and atomically publish to final_path."""
        self.flush()
        self.close()
        # verify total size
        expected = len(self._header) + self._data_size
        actual = os.path.getsize(self.path)
        if actual != expected:
            raise OutputError(
                f"output size mismatch: {actual} != {expected} (incomplete write?)")
        os.replace(self.path, final_path)
        _fsync_parent(final_path)
        log().info("published output to %s (%s)", final_path, human_bytes(actual))


def tensor_to_bytes(t: torch.Tensor) -> bytes:
    # Viewing storage as uint8 works for BF16/FP8 as well as NumPy-supported
    # dtypes, and preserves the exact little-endian bytes used by safetensors.
    flat = t.detach().cpu().contiguous().reshape(-1)
    return flat.view(torch.uint8).numpy().tobytes()


# ---------------------------------------------------------------------------
# Chunked quantization (bounded memory for very large tensors)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Conversion engine (streaming, resumable, atomic)
# ---------------------------------------------------------------------------
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
        "format": plan.fmt,
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
        "source_sha256": source_hashes,
        "decisions": [{
            "name": d.name, "kind": d.kind.value, "layer": d.layer,
            "group_size": d.group_size,
            "convrot_groupsize": d.convrot_groupsize,
            "out_dtype": torch_dtype_name(d.out_dtype) if d.out_dtype else None,
        } for d in plan.decisions],
        "entries": [{
            "name": e["name"], "dtype": torch_dtype_name(e["dtype"]),
            "shape": list(e["shape"]), "nbytes": e["nbytes"],
        } for e in plan.output_entries],
    }
    h.update(json_dumps(options).encode("utf-8"))
    return h.hexdigest()


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

    @staticmethod
    def _decision_output_names(d: TensorDecision) -> List[str]:
        if d.kind == DecisionKind.QUANTIZE:
            if d.layer is None:
                raise PolicyError(f"quantized tensor {d.name!r} has no layer name")
            return [d.layer + suffix for suffix in
                    (".weight", ".weight_s_rel", ".weight_s_channel", ".weight_codebook")]
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
                        self.reader, d.name, self.plan.fmt, d.group_size,
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
        suffix_map = {"": ".weight", "_s_rel": ".weight_s_rel",
                      "_s_channel": ".weight_s_channel", "_codebook": ".weight_codebook",
                      "_correction": ".weight_correction"}
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



# ---------------------------------------------------------------------------
# Metadata creation
# ---------------------------------------------------------------------------
def build_quant_metadata(info: CheckpointInfo, plan: ConversionPlan) -> Dict[str, Any]:
    """Official `_quantization_metadata` payload: {"layers": {layer: conf}}."""
    layers: Dict[str, Any] = {}
    for d in plan.quantized_layers():
        if d.layer is None:
            raise PolicyError(f"quantized tensor {d.name!r} has no layer name")
        layers[d.layer] = {
            "format": plan.fmt,
            "group_size": d.group_size,
            "convrot": True,
            "convrot_groupsize": d.convrot_groupsize,
        }
    return {"layers": layers}


def build_extension_metadata(info: CheckpointInfo, plan: ConversionPlan,
                             env: EnvironmentInfo, args: Any,
                             calibration: Optional[CalibrationStats],
                             sensitivity: Optional[SensitivityAnalyzer],
                             input_hashes: Dict[str, str],
                             tensor_payload_sha256: str,
                             validation_summary: Dict[str, Any],
                             warnings: List[str]) -> Dict[str, Any]:
    """Namespaced extension metadata (never described as official ComfyUI keys)."""
    d = plan.detection
    quant_layers = plan.quantized_layers()
    conf = {
        "schema": "comfy_wxa8/v1",
        "converter": CONVERTER_NAME,
        "converter_version": CONVERTER_VERSION,
        "format": plan.fmt,
        "format_revision": FORMAT_W4A8_REVISION,
        "architecture": d.architecture,
        "detection_confidence": d.confidence,
        "unet_prefix": d.unet_prefix,
        "source": {
            "kind": info.kind,
            "files": _portable_file_labels(info.files),
            "total_bytes": info.total_bytes,
            "sha256": _portable_hash_manifest(info.files, input_hashes),
        },
        "quantization": {
            "weight_bits": 4,
            "activation_bits": 8,
            "weight_quantization": "per-group 16-entry symmetric Lloyd-Max codebook",
            "activation_quantization": "runtime dynamic symmetric int8 per input row after ConvRot",
            "activation_scale": "fp32 amax(row)/127, clamped to at least 1e-30",
            "activation_rounding": "nearest integer, clamped to [-128,127]",
            "group_size": quant_layers[0].group_size if quant_layers else None,
            "convrot": True,
            "convrot_groupsize": quant_layers[0].convrot_groupsize if quant_layers else None,
            "scale_dtype": "fp8_e4m3fn",
            "packing": "int4-nibble-lsb",
            "symmetric": True,
            "n_quantized_layers": len(quant_layers),
            "n_kept_tensors": plan.n_kept,
            "compute_dtype": getattr(args, "compute_dtype", "auto"),
            "effective_compute_dtype": torch_dtype_name(
                getattr(args, "_compute_dtype_tensor", torch.float32)),
            "passthrough_output_dtype": getattr(args, "output_dtype", "auto"),
            "chunked_layers": sorted(plan.chunked_layers),
        },
        "calibration": calibration.to_dict() if calibration is not None else {
            "source": None, "method": "calibration-free (reference format)",
            "synthetic": False},
        "sensitivity": {
            "enabled": sensitivity is not None,
            "threshold": getattr(args, "sensitivity_threshold", None),
            "error_threshold": getattr(args, "error_threshold", None),
            "layers_kept": sorted(m.name for m in (sensitivity.results.values() if sensitivity else []) if m.kept),
        },
        "reproducibility": {
            "seed": getattr(args, "seed", 0),
            "device": getattr(args, "device", "auto"),
            "effective_device": plan.device,
            "torch_version": env.torch_version,
            "deterministic_on_same_backend": True,
            "codebook_subsample_seed": 0,
        },
        "compatibility": {
            "comfy_kitchen": {
                "required_revision": COMFY_KITCHEN_REV,
                "pr": 90,
                "merged": True,
                "layout": "AsymW4A8Int8Layout",
            },
            "comfyui": {
                "required_pr": COMFYUI_PR,
                "required_head": COMFYUI_PR_HEAD,
                "merged": True,
                "merged_commit": "344b43989e",
                "min_version": "0.31.0",
                "note": "ComfyUI PR #15308 (asym_w4a8_int8 loader) merged 2026-08-07; "
                        "ComfyUI >= v0.31.0 loads W4A8 natively, older builds need "
                        "patches/comfyui_w4a8_loader.patch (base " + COMFYUI_BASE + ")",
            },
            "cuda_backend": {
                "requires": "PyTorch cu130+, SM >= 8.0",
                "min_sm": list(W4A8_KERNEL_MIN_SM),
            },
            "triton_backend": {"requires": f"triton >= {TRITON_MIN_VERSION[0]}.{TRITON_MIN_VERSION[1]} (ROCm)"},
        },
        "output": {
            "tensor_data_sha256": tensor_payload_sha256,
            "tensor_data_bytes": plan.total_out_bytes,
            "entries": len(plan.output_entries),
            "file_sha256": None,
            "file_sha256_note": "The full-file SHA256 is emitted in the report. "
                                "It cannot be embedded in the file it hashes.",
        },
        "policy_summary": {
            "decision_counts": {
                kind.value: sum(1 for item in plan.decisions if item.kind == kind)
                for kind in DecisionKind
            },
            "reason_counts": dict(sorted({
                reason: sum(1 for item in plan.decisions if item.reason == reason)
                for reason in {item.reason for item in plan.decisions}
            }.items())),
            "full_manifest": "conversion report",
        },
        "validation": validation_summary,
        "warnings": warnings,
    }
    return conf


def metadata_json_bytes(meta: Dict[str, str]) -> torch.Tensor:
    return torch.tensor(list(json.dumps(meta).encode("utf-8")), dtype=torch.uint8)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
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
                schema_ok = (
                    schema_ok
                    and ext_payload.get("schema") == "comfy_wxa8/v1"
                    and ext_payload.get("format") == self.plan.fmt
                    and ext_payload.get("format_revision") == FORMAT_W4A8_REVISION
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
                    and quant_block.get("weight_bits") == 4
                    and quant_block.get("activation_bits") == 8
                    and quant_block.get("packing") == "int4-nibble-lsb"
                    and quant_block.get("scale_dtype") == "fp8_e4m3fn"
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
                conf_ok = all(
                    isinstance(value, dict)
                    and value.get("format") == self.plan.fmt
                    and value.get("group_size") ==
                    decision_by_layer.get(layer).group_size
                    and value.get("convrot") is True
                    and value.get("convrot_groupsize") ==
                    decision_by_layer.get(layer).convrot_groupsize
                    for layer, value in layers.items()
                    if layer in decision_by_layer)
                conf_ok = conf_ok and not mism
                self.check("metadata-layer-conf", conf_ok,
                           "format/group/ConvRot fields valid" if conf_ok else
                           "invalid format/group/ConvRot field")
            except Exception as e:
                self.check("metadata-json", False, f"unparseable: {e}")

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
            for d in sample:
                if d.layer is None:
                    self.check(f"recon-{d.name}", False,
                               "quantized decision has no layer name")
                    continue
                try:
                    cb = st.get_tensor(d.layer + ".weight_codebook")
                    orig = self.info.by_name(d.name)
                    if orig is not None and reader is not None:
                        max_mem = getattr(args, "max_memory", 2 * 1024**3)
                        bounded = (
                            d.name in getattr(self.plan, "chunked_layers", set())
                            or _quant_work_bytes(orig) > max_mem
                        )
                        if bounded:
                            n, k = int(orig.shape[0]), int(orig.shape[1])
                            chunk_rows = _chunk_rows_for_budget(k, n, max_mem)
                            acc = _MetricAccumulator(d.name)
                            pack_ok = True
                            original_view = reader.read_tensor(d.name)
                            packed_slice = st.get_slice(d.layer + ".weight")
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
                            m = acc.finish()
                        else:
                            packed = st.get_tensor(d.layer + ".weight")
                            s_rel = st.get_tensor(d.layer + ".weight_s_rel")
                            s_ch = st.get_tensor(d.layer + ".weight_s_channel")
                            orig_t = reader.read_tensor(d.name).float()
                            dq = dequantize_w4a8_weight(
                                packed, s_rel, s_ch, codebook=cb,
                                group_size=d.group_size,
                                convrot_groupsize=d.convrot_groupsize,
                                output_dtype=torch.float32)
                            m = compute_weight_metrics(orig_t, dq)
                            rt = unpack_w4(packed)
                            repacked = (
                                (rt[:, 0::2] & 0xF)
                                | ((rt[:, 1::2] & 0xF) << 4)
                            ).to(torch.int8)
                            pack_ok = bool(torch.equal(repacked, packed))
                        worst[d.layer] = m.rel_l2
                        if m.rel_l2 > self.plan.detection.policy.max_rel_l2:
                            self.check(f"recon-{d.layer}", False,
                                       f"relL2 {m.rel_l2:.4f} > {self.plan.detection.policy.max_rel_l2}")
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
                self.check("reconstruction-error-bound", mx <= self.plan.detection.policy.max_rel_l2,
                           f"max {'full' if full_validation else 'sampled'} relL2 {mx:.4f} "
                           f"(policy bound {self.plan.detection.policy.max_rel_l2})")

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
                    reader, d0.name, self.plan.fmt, d0.group_size,
                    d0.convrot_groupsize, max_mem, torch.device("cpu"),
                    compute_dtype=compute_dtype)
                out2 = quantize_tensor_bounded(
                    reader, d0.name, self.plan.fmt, d0.group_size,
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
            if self.env.has_comfy_kitchen:
                compatible = self.env.comfy_kitchen_has_w4a8_layout
                self.check("compat-comfy-kitchen", True,
                           detail=f"installed comfy-kitchen: {self.env.comfy_kitchen_rev or 'version unknown'}, "
                                  "static package source contains AsymW4A8Int8Layout: "
                                  f"{compatible}",
                           warn=not compatible)
            else:
                self.check("compat-comfy-kitchen", True, "comfy-kitchen not installed (skipped)",
                           skipped=True, reason="optional runtime probe")
            if self.env.has_comfy_quant_ops:
                compatible = self.plan.fmt in self.env.comfyui_quant_algos
                self.check("compat-comfyui", True,
                           detail=f"static ComfyUI quant_ops formats: "
                                  f"{self.env.comfyui_quant_algos}",
                           warn=not compatible)
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


# ---------------------------------------------------------------------------
# Metadata embedding (republish final file with the real header metadata)
# ---------------------------------------------------------------------------
def republish_with_metadata(src_path: str, final_path: str, metadata: Dict[str, str],
                            entries: List[Dict[str, Any]]) -> str:
    """Rewrite src -> final with the final header (including __metadata__).

    Streaming copy; bounded memory; atomic publish.  The data payload is copied
    verbatim; only the header changes.  `src_path` may equal `final_path` (the
    published file is replaced atomically via a sibling temp file).
    """
    header: Dict[str, Any] = {"__metadata__": metadata}
    offset = 0
    for e in entries:
        start = offset
        end = start + e["nbytes"]
        header[e["name"]] = {
            "dtype": TORCH_TO_SAFE[e["dtype"]],
            "shape": list(e["shape"]),
            "data_offsets": [start, end],
        }
        offset = end
    # Canonical key ordering makes a metadata-only republish byte-identical
    # regardless of the mapping order returned by a safetensors implementation.
    raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(raw) > MAX_SAFETENSORS_HEADER_SIZE:
        raise OutputError("metadata header exceeds the safetensors 100 MB safety limit")
    raw += b" " * ((-len(raw)) % 8)
    parent = os.path.dirname(os.path.abspath(final_path)) or "."
    fd, final_tmp = tempfile.mkstemp(
        prefix=f".{Path(final_path).name}.metadata.", suffix=".tmp", dir=parent)
    try:
        with open(src_path, "rb") as src:
            head = src.read(8)
            if len(head) != 8:
                raise OutputError(f"{src_path}: truncated source header during republish")
            old_hlen = struct.unpack("<Q", head)[0]
            payload_start = 8 + old_hlen
            if payload_start > os.fstat(src.fileno()).st_size:
                raise OutputError(f"{src_path}: invalid source header during republish")
            payload_bytes = os.fstat(src.fileno()).st_size - payload_start
            if payload_bytes != offset:
                raise OutputError(
                    f"{src_path}: payload size {payload_bytes} != planned {offset}")
            src.seek(payload_start)
            with os.fdopen(fd, "wb") as dst:
                dst.write(struct.pack("<Q", len(raw)))
                dst.write(raw)
                shutil.copyfileobj(src, dst, length=1 << 20)
                dst.flush()
                os.fsync(dst.fileno())
        with RawSafetensorsFile(final_tmp) as candidate:
            if candidate.metadata != metadata:
                raise OutputError("republished metadata did not round-trip")
            if set(candidate.entries) != {entry["name"] for entry in entries}:
                raise OutputError("republished tensor inventory did not round-trip")
            for entry in entries:
                dtype, shape, _, _ = candidate.get(entry["name"])
                if dtype != entry["dtype"] or tuple(shape) != tuple(entry["shape"]):
                    raise OutputError(
                        f"republished shape/dtype mismatch for {entry['name']!r}")
        os.replace(final_tmp, final_path)
        _fsync_parent(final_path)
    except Exception:
        with contextlib.suppress(OSError):
            os.close(fd)
        with contextlib.suppress(OSError):
            os.unlink(final_tmp)
        raise
    return final_path


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
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


def build_report(info: CheckpointInfo, plan: ConversionPlan, env: EnvironmentInfo,
                 args: Any, result: DetectionResult, calibration: Optional[CalibrationStats],
                 metrics: Dict[str, TensorMetrics], validation: Dict[str, Any],
                 input_hashes: Dict[str, str], output_sha256: str,
                 elapsed: float, warnings: List[str],
                 quant_rows: List[str]) -> Dict[str, Any]:
    comp = [
        "comfy-kitchen >= %s (PR #90, merged) with AsymW4A8Int8Layout" % COMFY_KITCHEN_REV,
        "ComfyUI >= v0.31.0 (PR #%d merged as 344b43989e; older builds need "
        "patches/comfyui_w4a8_loader.patch)" % COMFYUI_PR,
        "CUDA: PyTorch cu130+, SM >= 8.0; ROCm: triton >= 3.7; eager works anywhere",
    ]
    return {
        "converter": CONVERTER_NAME, "converter_version": CONVERTER_VERSION,
        "format": plan.fmt, "format_revision": FORMAT_W4A8_REVISION,
        "architecture": result.architecture, "detection_confidence": result.confidence,
        "unet_prefix": result.unet_prefix,
        "detection_evidence": result.evidence, "detection_hints": result.hints,
        "competing": result.competing,
        "input_kind": info.kind, "input_files": info.files, "input_bytes": info.total_bytes,
        "output_path": args.output, "output_bytes": plan.total_out_bytes,
        "n_input_tensors": len(info.tensors), "n_quantized": plan.n_quantized,
        "n_kept": plan.n_kept,
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


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="comfyui_wxa8_quantizer.py",
        description=(
            "Standalone W4A8 checkpoint converter for ComfyUI-compatible generative "
            "models. See the module docstring for the verified format specification "
            "and exact source revisions."),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", nargs="?", metavar="ORIGINAL_MODEL",
                   help="path to the original checkpoint: a .safetensors file, a "
                        "sharded safetensors directory, an HF-style model directory, "
                        "or (with --trust-pickle) a torch pickle checkpoint")
    p.add_argument("--output", metavar="PATH", default=None,
                   help="output checkpoint path (required for conversion)")
    p.add_argument("--format", choices=["w4a8"], default="w4a8",
                   help="quantization format (only the ComfyUI reference w4a8 "
                        "format is supported)")
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
                   help="quantization group size (default: architecture policy, 16)")
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
        try:
            group_size = int(conf.get("group_size", 16))
            convrot_groupsize = int(conf.get("convrot_groupsize", 256))
        except (TypeError, ValueError) as e:
            raise ValidationError(
                f"{output_path}: invalid group metadata for {layer!r}") from e
        if conf.get("format") != fmt or group_size <= 0 or convrot_groupsize <= 0:
            raise ValidationError(
                f"{output_path}: incompatible format/group metadata for {layer!r}")
        if convrot_groupsize != W4A8_CONVROT_GROUPSIZE:
            raise ValidationError(
                f"{output_path}: layer {layer!r} uses convrot_groupsize "
                f"{convrot_groupsize}, but the comfy-kitchen 0.2.27 CUDA "
                f"runtime only supports {W4A8_CONVROT_GROUPSIZE} (K must be "
                "divisible by 256). Re-convert the checkpoint with converter "
                f">= 1.2.2; incompatible layers now pass through.")
        source_name = f"{layer}.weight"
        if info is not None:
            source_meta = info.by_name(source_name)
            if source_meta is None or len(source_meta.shape) != 2:
                raise ValidationError(
                    f"{output_path}: quantized layer {layer!r} has no matching "
                    "2D weight in the supplied original checkpoint")
            try:
                validate_w4_shape(
                    int(source_meta.shape[1]), group_size, convrot_groupsize)
            except PolicyError as e:
                raise ValidationError(
                    f"{output_path}: invalid W4A8 shape metadata for {layer!r}: {e}") from e
        decisions.append(TensorDecision(
            source_name, DecisionKind.QUANTIZE, "reconstructed from output metadata",
            layer=layer, group_size=group_size,
            convrot_groupsize=convrot_groupsize))
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


# ---------------------------------------------------------------------------
# Self-tests (engineering tests; NOT full-model quality validation)
# ---------------------------------------------------------------------------
def run_self_tests() -> int:
    log().info("running embedded self-tests ...")
    tests: List[Tuple[str, Callable[[], str]]] = [
        ("w4-pack-roundtrip", _test_w4_pack_roundtrip),
        ("odd-dims", _test_odd_dims),
        ("padding-removal", _test_padding_removal),
        ("scale-calculations", _test_scale_calculations),
        ("deterministic-conversion", _test_deterministic),
        ("compute-dtype-selection", _test_compute_dtype),
        ("real-activation-calibration", _test_activation_calibration),
        ("standalone-environment-probe", _test_standalone_environment),
        ("metadata-generation", _test_metadata),
        ("registry-behavior", _test_registry),
        ("architecture-detection-safety", _test_detection_safety),
        ("golden-vectors-vs-reference", _test_golden_vectors),
        ("malformed-checkpoints", _test_malformed),
        ("checkpoint-input-variants", _test_checkpoint_variants),
        ("unsupported-tensors", _test_unsupported),
        ("sensitivity-output-planning", _test_sensitivity_planning),
        ("resume-state-recovery", _test_resume),
        ("atomic-output", _test_atomic),
        ("end-to-end-mini-model-w4a8", _test_e2e_mini_model_w4a8),
    ]
    failed = 0
    try:
        for name, fn in tests:
            try:
                detail = fn()
                print(f"  [PASS] {name}" + (f" -- {detail}" if detail else ""))
            except Exception as e:
                failed += 1
                print(f"  [FAIL] {name} -- {e}")
                log().debug("self-test %s failed", name, exc_info=True)
    finally:
        for d in _TEST_DIRS:
            shutil.rmtree(d, ignore_errors=True)
        _TEST_DIRS.clear()
    print(f"self-tests: {len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


_TEST_DIRS: List[str] = []


def _tmpdir(prefix: str = "wxa8_selftest") -> str:
    d = tempfile.mkdtemp(prefix=prefix)
    _TEST_DIRS.append(d)
    return d


def _make_mini_checkpoint(path: str, seed: int = 0) -> None:
    """SDXL-shaped mini model: a few linears under model.diffusion_model."""
    torch.manual_seed(seed)
    sd = {
        "model.diffusion_model.input_blocks.0.0.weight": torch.randn(320, 4, 3, 3) * 0.1,
        "model.diffusion_model.input_blocks.0.0.bias": torch.randn(320) * 0.01,
        "model.diffusion_model.input_blocks.1.0.transformer_blocks.0.attn1.to_q.weight": torch.randn(1280, 1280) * 0.02,
        "model.diffusion_model.input_blocks.1.0.transformer_blocks.0.attn1.to_k.weight": torch.randn(1280, 1280) * 0.02,
        "model.diffusion_model.input_blocks.1.0.transformer_blocks.0.attn1.to_v.weight": torch.randn(1280, 1280) * 0.02,
        "model.diffusion_model.input_blocks.1.0.transformer_blocks.0.attn1.to_out.0.weight": torch.randn(1280, 1280) * 0.02,
        "model.diffusion_model.input_blocks.1.0.transformer_blocks.0.attn2.to_q.weight": torch.randn(1280, 2048) * 0.02,
        "model.diffusion_model.input_blocks.1.0.transformer_blocks.0.attn2.to_k.weight": torch.randn(1280, 2048) * 0.02,
        "model.diffusion_model.input_blocks.1.0.transformer_blocks.0.attn2.to_v.weight": torch.randn(1280, 2048) * 0.02,
        "model.diffusion_model.input_blocks.1.0.transformer_blocks.0.attn2.to_out.0.weight": torch.randn(1280, 1280) * 0.02,
        "model.diffusion_model.input_blocks.1.0.transformer_blocks.0.ff.net.0.proj.weight": torch.randn(5120, 1280) * 0.01,
        "model.diffusion_model.input_blocks.1.0.transformer_blocks.0.ff.net.2.weight": torch.randn(1280, 5120) * 0.01,
        "model.diffusion_model.input_blocks.1.0.transformer_blocks.0.norm1.weight": torch.randn(1280) * 0.1,
        "model.diffusion_model.time_embed.0.weight": torch.randn(320, 320) * 0.05,
        "model.diffusion_model.time_embed.0.bias": torch.randn(320) * 0.05,
        "model.diffusion_model.out.2.weight": torch.randn(4, 320) * 0.1,
        "model.diffusion_model.out.2.bias": torch.randn(4) * 0.1,
        "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight": torch.randn(49408, 768) * 0.02,
        "first_stage_model.encoder.conv_in.weight": torch.randn(128, 3, 3, 3) * 0.1,
    }
    safetensors.torch.save_file(sd, path, metadata={"_selftest": "1"})


def _test_w4_pack_roundtrip() -> str:
    torch.manual_seed(1)
    for k in (16, 32, 64, 128, 256):
        codes = torch.randint(0, 16, (7, k), dtype=torch.int32)
        packed = ((codes[:, 0::2] & 0xF) | ((codes[:, 1::2] & 0xF) << 4)).to(torch.int8)
        rt = unpack_w4(packed)
        assert torch.equal(rt, codes), f"K={k} mismatch"
    return "K=16..256 round trips"


def _test_odd_dims() -> str:
    torch.manual_seed(3)
    # odd N, K=48 (divisible by 16 but not by 32)
    w = torch.randn(17, 48)
    p, s_rel, s_ch, corr, cb = quantize_w4a8_weight(
        w, group_size=16, convrot_groupsize=16)
    assert corr is None and cb is not None
    assert p.shape == (17, 24)
    dq = dequantize_w4a8_weight(p, s_rel, s_ch, codebook=cb, group_size=16,
                                convrot_groupsize=16, output_dtype=torch.float32)
    assert dq.shape == (17, 48)
    return "N=17, K=48 (w4)"


def _test_padding_removal() -> str:
    d = _tmpdir()
    path = os.path.join(d, "pad.safetensors")
    tensors = {
        "a_bool": torch.tensor([True]),
        "b_u8": torch.tensor([1, 2, 3], dtype=torch.uint8),
        "c_float": torch.randn(5),
    }
    if getattr(torch, "uint16", None) in TORCH_TO_SAFE:
        tensors["d_u16"] = torch.tensor([1, 65535], dtype=torch.uint16)
    if torch.complex64 in TORCH_TO_SAFE:
        tensors["e_complex64"] = torch.tensor([1 + 2j], dtype=torch.complex64)
    safetensors.torch.save_file(tensors, path)
    entries = [
        {"name": name, "dtype": tensor.dtype, "shape": tuple(tensor.shape),
         "nbytes": tensor.numel() * tensor.element_size()}
        for name, tensor in tensors.items()
    ]
    w = SafetensorsStreamWriter(path + ".stream", entries)
    w.open()
    for name, tensor in tensors.items():
        w.write_tensor_bytes(name, tensor_to_bytes(tensor))
    w.finalize(path + ".final")
    with safe_open(path + ".final", framework="pt") as st:
        for name, tensor in tensors.items():
            assert torch.equal(st.get_tensor(name), tensor)
    return "contiguous odd-byte and optional safetensors dtypes reopen"


def _test_scale_calculations() -> str:
    torch.manual_seed(4)
    w = torch.randn(8, 64)
    p, s_rel, s_ch, corr, cb = quantize_w4a8_weight(w, group_size=16, convrot_groupsize=64)
    assert s_rel.shape == (8, 4)
    assert s_ch.shape == (8,)
    assert cb.shape == (16,)
    assert corr is None
    # the decoded weights must be within ~1 int8 LSB of the pre-grid reconstruction
    codes = unpack_w4(p)
    shifted = cb[codes].view(8, 4, 16) * (s_rel.float().unsqueeze(-1) * s_ch.view(8, 1, 1))
    vals = cb[codes].view(8, 4, 16) * s_rel.float().unsqueeze(-1)
    i8 = vals.view(8, 64).round().clamp(-127, 127)
    decoded = i8.float() * s_ch.view(-1, 1)
    assert (decoded - shifted.view(8, 64)).abs().max() <= s_ch.max() * 1.01 + 1e-6
    assert (cb.abs() <= 1.0).all()
    assert torch.isfinite(s_rel.float()).all() and (s_rel.float() > 0).all()
    assert torch.isfinite(s_ch).all() and (s_ch > 0).all()
    zero_metrics = compute_weight_metrics(torch.zeros(4, 4), torch.zeros(4, 4))
    assert zero_metrics.rel_l2 == 0.0 and zero_metrics.cosine == 1.0
    return "s_rel/s_channel/codebook shapes, positivity, 1-LSB grid bound"


def _test_deterministic() -> str:
    torch.manual_seed(5)
    w = torch.randn(64, 256)
    o1 = quantize_w4a8_weight(w)
    o2 = quantize_w4a8_weight(w)
    for a, b in zip(o1, o2, strict=True):
        if a is None or b is None:
            assert a is None and b is None
        else:
            assert torch.equal(a, b)
    return "two runs byte-identical (w4)"


def _test_compute_dtype() -> str:
    d = _tmpdir()
    path = os.path.join(d, "bf16.safetensors")
    torch.manual_seed(51)
    safetensors.torch.save_file({"w": torch.randn(65, 64).bfloat16()}, path)
    info = discover_checkpoint(path)
    with CheckpointReader(info) as reader:
        fp32 = quantize_tensor_bounded(
            reader, "w", FORMAT_W4A8, 16, 64, 256 * 1024**2,
            torch.device("cpu"), compute_dtype=torch.float32)
        bf16 = quantize_tensor_bounded(
            reader, "w", FORMAT_W4A8, 16, 64, 256 * 1024**2,
            torch.device("cpu"), compute_dtype=torch.bfloat16)
    assert any(not torch.equal(fp32[key], bf16[key]) for key in fp32)
    return "fp32 and bf16 compute paths produce distinct deterministic tensors"


def _test_activation_calibration() -> str:
    torch.manual_seed(52)
    original = torch.randn(8, 16)
    dequant = original + torch.randn_like(original) * 0.03
    activations = torch.randn(5, 16)
    d = _tmpdir()
    model_path = os.path.join(d, "calibration_model.safetensors")
    source_path = os.path.join(d, "activations.npz")
    cache_path = os.path.join(d, "activations.cache")
    safetensors.torch.save_file({"layer.weight": original}, model_path)
    np.savez(source_path, **{"layer.weight": activations.numpy()})
    info = discover_checkpoint(model_path)
    calibration = load_calibration(source_path, info, 5, cache_path)
    cached = load_calibration(source_path, info, 5, cache_path)
    assert torch.equal(cached.layers["layer.weight"]["samples"], activations)
    with open(cache_path, "rb") as f:
        assert f.read(4) == b"PK\x03\x04"
    analyzer = SensitivityAnalyzer(None, 1.0, calibration)
    metrics = analyzer.evaluate("layer.weight", original, dequant)
    expected = activation_aware_error(original, dequant, activations)
    fake = activations.abs().amax(dim=0).unsqueeze(0).expand(5, -1)
    fake_value = activation_aware_error(original, dequant, fake)
    assert expected is not None and abs(metrics.act_rel_l2 - expected) < 1e-8
    assert abs(metrics.act_rel_l2 - fake_value) > 1e-6
    return "real rows used directly; compressed safe cache round-trips"


def _test_standalone_environment() -> str:
    import builtins
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "comfy" or name.startswith("comfy.") or \
                name == "comfy_kitchen" or name.startswith("comfy_kitchen."):
            raise AssertionError(f"standalone probe imported {name}")
        return original_import(name, *args, **kwargs)

    builtins.__import__ = guarded_import
    try:
        inspected = inspect_environment()
    finally:
        builtins.__import__ = original_import
    assert inspected.python and inspected.torch_version
    return "compatibility inspection performs no ComfyUI/comfy-kitchen imports"


def _test_metadata() -> str:
    d = _tmpdir()
    path = os.path.join(d, "mini.safetensors")
    _make_mini_checkpoint(path)
    info = discover_checkpoint(path)
    det = detect_architecture(info, shape_lookup=lambda n: (info.by_name(n).shape if info.by_name(n) else None))
    dec = classify_tensors(info, det, FORMAT_W4A8, None, [], [], [], None, None)
    plan = ConversionPlan(fmt=FORMAT_W4A8, detection=det, decisions=dec,
                          metadata_quant={}, metadata_ext={}, output_entries=[])
    qm = build_quant_metadata(info, plan)
    assert "layers" in qm
    for d_ in plan.quantized_layers():
        assert d_.layer in qm["layers"]
        conf = qm["layers"][d_.layer]
        assert conf["format"] == FORMAT_W4A8 and conf["group_size"] >= 4
    plan.n_quantized = len(plan.quantized_layers())
    plan.n_kept = len(plan.decisions) - plan.n_quantized
    ext = build_extension_metadata(
        info, plan, inspect_environment(),
        _selftest_args(os.path.join(d, "out.safetensors"), FORMAT_W4A8),
        None, None, hash_checkpoint_files(info), "0" * 64, {}, [])
    assert ext["schema"] == "comfy_wxa8/v1"
    assert ext["source"]["files"] == ["mini.safetensors"]
    assert list(ext["source"]["sha256"]) == ["mini.safetensors"]
    assert "runtime dynamic symmetric int8" in \
        ext["quantization"]["activation_quantization"]
    return f"{len(qm['layers'])} layers recorded; extension schema and provenance valid"


def _test_registry() -> str:
    names = family_names()
    assert len(names) == len(set(names))
    for n in names:
        pol = get_family(n)
        assert pol.family == n and pol.comfyui_classes
        assert pol.runtime_status in ("verified", "experimental", "unsupported")
    # every ComfyUI supported-model class at the research revision is covered
    covered = set()
    for n in names:
        covered.update(get_family(n).comfyui_classes)
    comfyui_classes = {
        "SD15", "SD20", "SD21UnclipL", "SD21UnclipH", "SDXLRefiner", "SDXL", "SSD1B",
        "Segmind_Vega", "KOALA_700M", "KOALA_1B", "SVD_img2vid", "SV3D_u", "SV3D_p",
        "Stable_Zero123", "SD_X4Upscaler", "Stable_Cascade_C", "Stable_Cascade_B",
        "SD15_instructpix2pix", "SDXL_instructpix2pix", "LotusD", "SD3", "StableAudio",
        "StableAudio3", "AuraFlow", "PixArtAlpha", "PixArtSigma", "HunyuanDiT",
        "HunyuanDiT1", "Flux", "FluxInpaint", "FluxSchnell", "Flux2", "Lens",
        "GenmoMochi", "LTXV", "LTXAV", "MiniMaxH3", "HunyuanVideo", "HunyuanVideoI2V",
        "HunyuanVideoSkyreelsI2V", "CosmosT2V", "CosmosI2V", "CosmosT2IPredict2",
        "CosmosI2VPredict2", "Anima", "Lumina2", "ZImage", "ZImagePixelSpace",
        "PixelDiTT2I", "PiD", "WAN21_T2V", "WAN21_CausalAR_T2V", "WAN21_I2V",
        "WAN21_FunControl2V", "WAN21_Camera", "WAN22_Camera", "WAN21_Vace",
        "WAN21_HuMo", "WAN22_S2V", "WAN22_Animate", "WAN22_T2V", "WAN21_FlowRVS",
        "WAN21_SCAIL", "WAN21_SCAIL2", "WAN22_WanDancer", "Hunyuan3Dv2",
        "Hunyuan3Dv2_1", "Hunyuan3Dv2mini", "TripoSplat", "HiDream", "HiDreamO1",
        "Chroma", "SeedVR2", "ChromaRadiance", "ACEStep", "Omnigen2", "Boogu",
        "Ideogram4", "Krea2", "MageFlow", "QwenImage", "JoyImage", "HunyuanImage21",
        "HunyuanImage21Refiner", "HunyuanVideo15", "HunyuanVideo15_SR_Distilled",
        "Kandinsky5", "Kandinsky5Image", "ACEStep15", "LongCatImage", "RT_DETR_v4",
        "DepthAnything3", "ErnieImage", "SAM3", "SAM31", "CogVideoX_T2V",
        "CogVideoX_I2V", "CogVideoX_Inpaint",
    }
    missing = sorted(comfyui_classes - covered)
    assert not missing, f"registry missing ComfyUI classes: {missing}"
    return f"{len(names)} families, {len(covered)} ComfyUI classes covered"


def _ckpt(keys_shapes: Sequence[Tuple[str, Tuple[int, ...]]]) -> CheckpointInfo:
    tensors = [TensorMeta(name, torch.float32, shape, int(np.prod(shape)) * 4,
                          "", 0, int(np.prod(shape)) * 4)
               for name, shape in keys_shapes]
    return CheckpointInfo(kind="safetensors", files=[], metadata={}, tensors=tensors)


def _classify_real(keys_shapes: Sequence[Tuple[str, Tuple[int, ...]]]):
    info = _ckpt(keys_shapes)
    det = detect_architecture(
        info, shape_lookup=lambda n: (info.by_name(n).shape
                                      if info.by_name(n) else None))
    decisions = classify_tensors(info, det, FORMAT_W4A8, 16, None, None,
                                 None, None, None)
    return det, decisions


def _test_detection_safety() -> str:
    # Real Boogu-Image-0.1 key naming (Comfy-Org repack, verified against the
    # published checkpoint): double/single stream layers plus OmniGen2-style
    # refiners and embedders. Detection must land on the dedicated boogu
    # family, and the linear attention / FFN weights must actually quantize.
    # K must be divisible by 256 for W4A8 (CUDA ConvRot is 256-only); the
    # K=320 context_refiner to_k below is the passthrough case (it would have
    # crashed the comfy-kitchen 0.2.27 CUDA kernel as convrot_groupsize 64).
    boogu_keys: Sequence[Tuple[str, Tuple[int, ...]]] = [
        ("double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight", (256, 256)),
        ("double_stream_layers.0.img_instruct_attn.processor.img_out.weight", (256, 256)),
        ("double_stream_layers.0.img_self_attn.to_q.weight", (256, 256)),
        ("double_stream_layers.0.img_self_attn.to_k.weight", (64, 256)),
        ("double_stream_layers.0.img_self_attn.to_out.0.weight", (256, 256)),
        ("double_stream_layers.0.img_feed_forward.linear_1.weight", (256, 256)),
        ("double_stream_layers.0.img_feed_forward.linear_2.weight", (256, 256)),
        ("double_stream_layers.0.img_feed_forward.linear_3.weight", (256, 256)),
        ("double_stream_layers.0.instruct_feed_forward.linear_1.weight", (256, 256)),
        ("double_stream_layers.0.img_norm1.linear.weight", (256, 64)),
        ("double_stream_layers.0.img_norm1.linear.bias", (256,)),
        ("double_stream_layers.0.img_norm1.norm.weight", (64,)),
        ("single_stream_layers.0.attn.to_q.weight", (256, 256)),
        ("single_stream_layers.0.attn.to_k.weight", (64, 256)),
        ("single_stream_layers.0.attn.to_out.0.weight", (256, 256)),
        ("single_stream_layers.0.feed_forward.linear_1.weight", (256, 256)),
        ("single_stream_layers.0.feed_forward.linear_2.weight", (256, 256)),
        ("single_stream_layers.0.norm1.linear.weight", (256, 64)),
        ("context_refiner.0.attn.to_q.weight", (256, 256)),
        ("context_refiner.0.attn.to_k.weight", (64, 320)),
        ("context_refiner.0.feed_forward.linear_1.weight", (256, 256)),
        ("noise_refiner.0.attn.to_q.weight", (256, 256)),
        ("ref_image_refiner.0.feed_forward.linear_1.weight", (256, 256)),
        ("x_embedder.weight", (64, 64)),
        ("ref_image_patch_embedder.weight", (64, 64)),
        ("time_caption_embed.timestep_embedder.linear_1.weight", (64, 64)),
        ("norm_out.linear_1.weight", (64, 64)),
        ("norm_out.linear_2.weight", (64, 64)),
        ("image_index_embedding", (5, 64)),
    ]
    det, decisions = _classify_real(boogu_keys)
    assert det.architecture == "boogu", det.architecture
    assert det.confidence == "high", det.confidence
    assert get_family("Boogu").family == "boogu"
    quantized = {d.name for d in decisions if d.kind == DecisionKind.QUANTIZE}
    kept = {d.name for d in decisions if d.kind != DecisionKind.QUANTIZE}
    assert len(quantized) == 18, sorted(quantized)
    assert "double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight" in quantized
    assert "double_stream_layers.0.img_self_attn.to_q.weight" in quantized
    assert "single_stream_layers.0.feed_forward.linear_1.weight" in quantized
    assert "context_refiner.0.attn.to_q.weight" in quantized
    # K=320 is not divisible by 256: must pass through, not quantize with a
    # smaller ConvRot group (CUDA fused kernel is 256-only)
    cr_k = next(d for d in decisions
                if d.name == "context_refiner.0.attn.to_k.weight")
    assert cr_k.kind == DecisionKind.KEEP, cr_k
    assert "256" in cr_k.reason, cr_k.reason
    assert "double_stream_layers.0.img_norm1.linear.weight" in kept  # modulation
    assert "single_stream_layers.0.norm1.linear.weight" in kept      # modulation
    assert "norm_out.linear_1.weight" in kept                        # output head
    assert "x_embedder.weight" in kept                               # embedder
    assert "time_caption_embed.timestep_embedder.linear_1.weight" in kept

    # Real OmniGen2 key naming (BAAI/OmniGen2): layers.N + refiners. Detection
    # must land on omnigen2 and the linear weights must quantize too.
    og2_keys: Sequence[Tuple[str, Tuple[int, ...]]] = [
        ("layers.0.attn.to_q.weight", (256, 256)),
        ("layers.0.attn.to_k.weight", (64, 256)),
        ("layers.0.attn.to_v.weight", (64, 256)),
        ("layers.0.attn.to_out.0.weight", (256, 256)),
        ("layers.0.feed_forward.linear_1.weight", (256, 256)),
        ("layers.0.feed_forward.linear_2.weight", (256, 256)),
        ("layers.0.feed_forward.linear_3.weight", (256, 256)),
        ("layers.0.norm1.linear.weight", (256, 64)),
        ("layers.0.attn.norm_k.weight", (64,)),
        ("context_refiner.0.attn.to_q.weight", (256, 256)),
        ("context_refiner.0.feed_forward.linear_1.weight", (256, 256)),
        ("noise_refiner.0.attn.to_q.weight", (256, 256)),
        ("noise_refiner.0.feed_forward.linear_1.weight", (256, 256)),
        ("time_caption_embed.timestep_embedder.linear_1.bias", (64,)),
        ("x_embedder.weight", (64, 64)),
        ("ref_image_patch_embedder.weight", (64, 64)),
        ("norm_out.linear_1.weight", (64, 64)),
    ]
    det, decisions = _classify_real(og2_keys)
    assert det.architecture == "omnigen2", det.architecture
    assert det.confidence == "high", det.confidence
    quantized = {d.name for d in decisions if d.kind == DecisionKind.QUANTIZE}
    kept = {d.name for d in decisions if d.kind != DecisionKind.QUANTIZE}
    assert len(quantized) == 11, sorted(quantized)
    assert "layers.0.attn.to_q.weight" in quantized
    assert "layers.0.feed_forward.linear_2.weight" in quantized
    assert "context_refiner.0.attn.to_q.weight" in quantized
    assert "noise_refiner.0.feed_forward.linear_1.weight" in quantized
    assert "layers.0.norm1.linear.weight" in kept
    assert "norm_out.linear_1.weight" in kept

    keys = ("clf.1.weight", "head.modulation")
    ambiguous = CheckpointInfo(
        kind="safetensors", files=[], metadata={},
        tensors=[TensorMeta(name, torch.float32, (1,), 4, "", 0, 4)
                 for name in keys])
    try:
        detect_architecture(ambiguous)
        raise AssertionError("ambiguous checkpoint was guessed")
    except UnknownArchitectureError as exc:
        assert "ambiguous" in str(exc)
    return ("real Boogu/OmniGen2 naming quantizes; Boogu is its own family; "
        "equal-score architectures fail closed")


def _test_malformed() -> str:
    d = _tmpdir()
    # truncated header
    p1 = os.path.join(d, "trunc.safetensors")
    with open(p1, "wb") as f:
        f.write(b"\x00" * 8)
    try:
        discover_checkpoint(p1)
        raise AssertionError("expected InputError")
    except InputError:
        pass
    # negative and overlapping ranges must fail before any tensor is exposed
    for filename, spec, payload in (
        ("negative.safetensors",
         {"x": {"dtype": "U8", "shape": [1], "data_offsets": [-1, 0]}}, b""),
        ("coerced-shape.safetensors",
         {"x": {"dtype": "F32", "shape": ["1"], "data_offsets": [0, 4]}},
         b"\x00" * 4),
        ("overlap.safetensors", {
            "a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
            "b": {"dtype": "U8", "shape": [2], "data_offsets": [1, 3]},
        }, b"\x00" * 3),
    ):
        malformed = os.path.join(d, filename)
        header = json.dumps(spec).encode("utf-8")
        with open(malformed, "wb") as f:
            f.write(struct.pack("<Q", len(header)) + header + payload)
        try:
            discover_checkpoint(malformed)
            raise AssertionError(f"expected InputError for {filename}")
        except InputError:
            pass
    # bad data offsets
    p2 = os.path.join(d, "badoff.safetensors")
    hdr = json.dumps({"x": {"dtype": "F32", "shape": [4], "data_offsets": [0, 100]}}).encode()
    with open(p2, "wb") as f:
        f.write(struct.pack("<Q", len(hdr)) + hdr + b"\x00" * 100)
    try:
        discover_checkpoint(p2)
        raise AssertionError("expected InputError")
    except InputError:
        pass
    # pickle without trust
    p3 = os.path.join(d, "evil.ckpt")
    torch.save({"w": torch.zeros(4)}, p3)
    try:
        discover_checkpoint(p3)
        raise AssertionError("expected PickleInputError")
    except PickleInputError:
        pass
    # Explicit refresh must detect source mutation rather than returning the
    # cached pre-conversion identity.
    p4 = os.path.join(d, "mutable.safetensors")
    safetensors.torch.save_file({"x": torch.zeros(16)}, p4)
    mutable_info = discover_checkpoint(p4)
    before = hash_checkpoint_files(mutable_info)
    safetensors.torch.save_file({"x": torch.ones(16)}, p4)
    assert hash_checkpoint_files(mutable_info) == before
    assert hash_checkpoint_files(mutable_info, refresh=True) != before
    return "truncation, size/range/overlap validation, pickle guard, source rehash"


def _test_checkpoint_variants() -> str:
    d = Path(_tmpdir())
    shard = d / "model-00001-of-00001.safetensors"
    safetensors.torch.save_file({
        "mapped": torch.arange(4, dtype=torch.float32),
        "extra_bool": torch.tensor([True]),
    }, str(shard))
    with open(d / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump({"weight_map": {"mapped": shard.name}}, f)
    sharded = discover_checkpoint(str(d))
    assert sharded.key_set() == {"mapped", "extra_bool"}
    assert sharded.by_name("extra_bool").nbytes == 1
    shard2 = d / "model-00002-of-00002.safetensors"
    safetensors.torch.save_file({
        "mapped2": torch.arange(2, dtype=torch.float32),
        "extra_bool": torch.tensor([False]),
    }, str(shard2))
    with open(d / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump({"weight_map": {
            "mapped": shard.name, "mapped2": shard2.name,
        }}, f)
    try:
        discover_checkpoint(str(d))
        raise AssertionError("duplicate unindexed shard tensor was accepted")
    except InputError as exc:
        assert "duplicate tensor" in str(exc)

    pickle_path = d / "nested.pt"
    expected = torch.arange(16, dtype=torch.float32).reshape(4, 4).bfloat16()
    torch.save({"state_dict": {"nested.weight": expected}, "epoch": 3}, pickle_path)
    pickled = discover_checkpoint(str(pickle_path), trust_pickle=True)
    assert pickled.key_set() == {"nested.weight"}
    with CheckpointReader(pickled) as reader:
        assert bytes(reader.read_bytes("nested.weight")) == tensor_to_bytes(expected)
    return ("indexed extra tensor and nested BF16 pickle load; duplicate shard "
            "tensor rejected")


def _test_unsupported() -> str:
    d = _tmpdir()
    path = os.path.join(d, "mini.safetensors")
    _make_mini_checkpoint(path)
    info = discover_checkpoint(path)
    det = detect_architecture(info, shape_lookup=lambda n: (info.by_name(n).shape if info.by_name(n) else None))
    dec = classify_tensors(info, det, FORMAT_W4A8, None, [], [], [], None, None)
    by_name = {dd.name: dd for dd in dec}
    # 4D conv weights must not be quantized
    assert by_name["model.diffusion_model.input_blocks.0.0.weight"].kind == DecisionKind.KEEP
    # embeddings / norms kept
    assert by_name["cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"].kind == DecisionKind.KEEP
    assert by_name["model.diffusion_model.input_blocks.1.0.transformer_blocks.0.norm1.weight"].kind == DecisionKind.KEEP
    # linears quantized
    assert by_name["model.diffusion_model.input_blocks.1.0.transformer_blocks.0.attn1.to_q.weight"].kind == DecisionKind.QUANTIZE
    return "conv/embedding/norm kept; linear quantized"


def _test_sensitivity_planning() -> str:
    d = _tmpdir()
    path = os.path.join(d, "wan_small.safetensors")
    out = os.path.join(d, "wan_small_w4a8.safetensors")
    torch.manual_seed(53)
    q_name = "model.diffusion_model.blocks.0.self_attn.q.weight"
    k_name = "model.diffusion_model.blocks.0.cross_attn.k.weight"
    safetensors.torch.save_file({
        "model.diffusion_model.head.modulation": torch.zeros(1),
        q_name: torch.zeros(256, 256),
        k_name: torch.randn(256, 256),
    }, path)
    info = discover_checkpoint(path)
    det = detect_architecture(
        info, shape_lookup=lambda name: info.by_name(name).shape
        if info.by_name(name) else None)
    decisions = classify_tensors(
        info, det, FORMAT_W4A8, None, [], [], [], None, None)
    analyzer = SensitivityAnalyzer(0.01, 1.0, None)
    apply_sensitivity_prepass(
        info, decisions, analyzer, 256 * 1024**2,
        torch.device("cpu"), torch.float32)
    by_name = {item.name: item for item in decisions}
    assert by_name[q_name].kind == DecisionKind.QUANTIZE
    assert by_name[k_name].kind == DecisionKind.KEEP_PRECISION
    entries, total = build_output_entries(info, decisions, FORMAT_W4A8, None)
    plan = ConversionPlan(
        fmt=FORMAT_W4A8, detection=det, decisions=decisions,
        metadata_quant={}, metadata_ext={}, output_entries=entries,
        total_out_bytes=total)
    plan.n_quantized = len(plan.quantized_layers())
    plan.n_kept = len(decisions) - plan.n_quantized
    args = _selftest_args(
        out, FORMAT_W4A8,
        extra={"max_memory": 256 * 1024**2, "sensitivity_threshold": 0.01,
               "error_threshold": 1.0, "_compute_dtype_tensor": torch.float32})
    engine = ConversionEngine(info, plan, args, out + ".state.json", out + ".tmp", out)
    try:
        engine.run()
    finally:
        engine.close()
    with safe_open(out, framework="pt") as st:
        assert tuple(st.get_tensor(q_name).shape) == (256, 128)
        assert tuple(st.get_tensor(k_name).shape) == (256, 256)
        assert st.get_tensor(k_name).dtype == torch.float32
    return "sensitivity decisions frozen before packed/passthrough offsets"


def _test_resume() -> str:
    d = _tmpdir()
    path = os.path.join(d, "mini.safetensors")
    _make_mini_checkpoint(path)
    out = os.path.join(d, "out.safetensors")
    args = _selftest_args(out, FORMAT_W4A8)
    info = discover_checkpoint(path)
    det = detect_architecture(info, shape_lookup=lambda n: (info.by_name(n).shape if info.by_name(n) else None))
    dec = classify_tensors(info, det, FORMAT_W4A8, None, [], [], [], None, None)
    plan = ConversionPlan(fmt=FORMAT_W4A8, detection=det, decisions=dec,
                          metadata_quant={}, metadata_ext={}, output_entries=[])
    entries, total = build_output_entries(info, dec, FORMAT_W4A8, None)
    plan.output_entries = entries
    plan.total_out_bytes = total
    plan.n_quantized = len(plan.quantized_layers())
    plan.n_kept = len(dec) - plan.n_quantized
    state_path = out + ".state.json"
    tmp_path = out + ".tmp"
    # first run: simulate a crash after 6 tensors (strictly in plan order)
    eng = ConversionEngine(info, plan, args, state_path, tmp_path, out)
    eng._crash_after = 6
    try:
        eng.run()
        raise AssertionError("expected simulated crash")
    except _SimulatedCrash:
        eng.save_state()
    eng.close()
    assert os.path.exists(tmp_path)

    # A changed conversion option must invalidate the resume plan.
    args_drift = _selftest_args(
        out, FORMAT_W4A8, resume=True,
        extra={"max_memory": 1024 * 1024**2})
    drift = ConversionEngine(info, plan, args_drift, state_path, tmp_path, out)
    try:
        drift.run()
        raise AssertionError("resume accepted changed max-memory")
    except OutputError as exc:
        assert "parameters" in str(exc)
    finally:
        drift.close()

    # Completed tensor bytes are checksummed before resume.
    with open(state_path, "r", encoding="utf-8") as f:
        saved = json.load(f)
    first_record = next(iter(saved["entries"].values()))
    byte_pos = int(first_record["offset"])
    with open(tmp_path, "r+b") as f:
        f.seek(byte_pos)
        original_byte = f.read(1)
        f.seek(byte_pos)
        f.write(bytes([original_byte[0] ^ 0x01]))
    corrupt = ConversionEngine(
        info, plan, _selftest_args(out, FORMAT_W4A8, resume=True),
        state_path, tmp_path, out)
    try:
        corrupt.run()
        raise AssertionError("resume accepted corrupted completed tensor")
    except OutputError as exc:
        assert "checksum" in str(exc)
    finally:
        corrupt.close()
    with open(tmp_path, "r+b") as f:
        f.seek(byte_pos)
        f.write(original_byte)

    # second run: resume and finish
    args2 = _selftest_args(out, FORMAT_W4A8, resume=True)
    eng2 = ConversionEngine(info, plan, args2, state_path, tmp_path, out)
    eng2.run()
    eng2.close()
    # A crash after tensor finalization but before metadata publication must
    # resume from the checksummed staged file without requantizing.
    eng3 = ConversionEngine(
        info, plan, _selftest_args(out, FORMAT_W4A8, resume=True),
        state_path, tmp_path, out)
    eng3.run()
    eng3.close()
    with safe_open(out, framework="pt") as st:
        ql = plan.quantized_layers()[0]
        assert ql.layer is not None
        w = st.get_tensor(ql.layer + ".weight")
        assert w.shape == (info.by_name(ql.name).shape[0], info.by_name(ql.name).shape[1] // 2)
    return ("option drift/data corruption rejected; partial and post-conversion "
            "interruptions resumed")


def _test_atomic() -> str:
    d = _tmpdir()
    path = os.path.join(d, "mini.safetensors")
    _make_mini_checkpoint(path)
    out = os.path.join(d, "out.safetensors")
    args = _selftest_args(out, FORMAT_W4A8)
    info = discover_checkpoint(path)
    det = detect_architecture(info, shape_lookup=lambda n: (info.by_name(n).shape if info.by_name(n) else None))
    dec = classify_tensors(info, det, FORMAT_W4A8, None, [], [], [], None, None)
    plan = ConversionPlan(fmt=FORMAT_W4A8, detection=det, decisions=dec,
                          metadata_quant={}, metadata_ext={}, output_entries=[])
    entries, total = build_output_entries(info, dec, FORMAT_W4A8, None)
    plan.output_entries = entries
    plan.total_out_bytes = total
    plan.n_quantized = len(plan.quantized_layers())
    plan.n_kept = len(dec) - plan.n_quantized
    # Ensure the user-visible path never appears until metadata publication,
    # and that the source remains untouched.
    src_mtime = os.path.getmtime(path)
    staged = out + ".staged"
    eng = ConversionEngine(
        info, plan, args, out + ".state.json", out + ".tmp", staged)
    eng.run()
    eng.close()
    assert not os.path.exists(out)
    assert os.path.exists(staged)
    assert not os.path.exists(out + ".tmp")
    republish_with_metadata(staged, out, {}, entries)
    assert os.path.exists(out)
    assert os.path.getmtime(path) == src_mtime
    return "requested path withheld until atomic metadata publish; original untouched"


def _run_mini_convert(out: str, fmt: str, resume: bool = False, overwrite: bool = False,
                      extra_args: Optional[Dict[str, Any]] = None) -> Tuple[Any, ConversionPlan, CheckpointInfo, DetectionResult]:
    d = os.path.dirname(out)
    path = os.path.join(d, "mini.safetensors")
    if not os.path.exists(path):
        _make_mini_checkpoint(path)
    args = _selftest_args(out, fmt, resume=resume, overwrite=overwrite, extra=extra_args)
    info = discover_checkpoint(path)
    det = detect_architecture(info, shape_lookup=lambda n: (info.by_name(n).shape if info.by_name(n) else None))
    dec = classify_tensors(info, det, fmt, None, [], [], [], None, None)
    plan = ConversionPlan(fmt=fmt, detection=det, decisions=dec,
                          metadata_quant={}, metadata_ext={}, output_entries=[])
    entries, total = build_output_entries(info, dec, fmt, None)
    plan.output_entries = entries
    plan.total_out_bytes = total
    plan.n_quantized = len(plan.quantized_layers())
    plan.n_kept = len(dec) - plan.n_quantized
    qm = build_quant_metadata(info, plan)
    eng = ConversionEngine(info, plan, args, out + ".state.json", out + ".tmp", out)
    eng.run()
    eng.close()
    meta = dict(info.metadata)
    meta[METADATA_KEY_QUANT] = json_dumps(qm)
    meta[METADATA_KEY_EXT] = json_dumps(build_extension_metadata(
        info, plan, inspect_environment(), args, None, None,
        hash_checkpoint_files(info), sha256_safetensors_payload(out),
        {"status": "selftest"}, []))
    republish_with_metadata(out, out, meta, entries)
    return args, plan, info, det


def _test_e2e_mini_model_w4a8() -> str:
    d = _tmpdir()
    out = os.path.join(d, "out_w4a8.safetensors")
    _, plan, info, detection = _run_mini_convert(out, FORMAT_W4A8)
    with safe_open(out, framework="pt") as st:
        assert "_quantization_metadata" in st.metadata()
        assert st.metadata().get("_selftest") == "1"
        metadata = st.metadata()
        qm = json.loads(st.metadata()["_quantization_metadata"])
        assert len(qm["layers"]) == plan.n_quantized
        ext = json.loads(st.metadata()[METADATA_KEY_EXT])
        assert ext["schema"] == "comfy_wxa8/v1"
        assert ext["quantization"]["activation_bits"] == 8
        # spot-check one quantized layer
        layer = plan.quantized_layers()[0].layer
        w = st.get_tensor(layer + ".weight")
        assert w.dtype == torch.int8 and w.shape[1] == 640
    duplicate = os.path.join(d, "out_w4a8_duplicate.safetensors")
    republish_with_metadata(out, duplicate, metadata, plan.output_entries)
    assert sha256_file(duplicate) == sha256_file(out)
    validation_plan = plan_from_output(out, detection, FORMAT_W4A8, info)
    assert {entry["name"] for entry in validation_plan.output_entries} == {
        entry["name"] for entry in plan.output_entries}
    return (f"{plan.n_quantized} quantized layers; metadata preserved; "
            "validation inventory and deterministic serialization verified")


def _selftest_args(out: str, fmt: str, resume: bool = False, overwrite: bool = False,
                   extra: Optional[Dict[str, Any]] = None) -> argparse.Namespace:
    ns = argparse.Namespace(
        output=out, format="w4a8",
        architecture="auto", device="cpu", compute_dtype="auto", output_dtype="auto",
        group_size=None, calibration_source=None, calibration_samples=None,
        calibration_cache=None, seed=0, include=[], exclude=[], keep_precision=[], min_numel_override=None,
        sensitivity_threshold=None, error_threshold=0.35, max_memory=2 * 1024**3,
        streaming=True, resume=resume, overwrite=overwrite, dry_run=False,
        inspect=False, validate=False, validation_only=False, metadata_only=False,
        report=None, log_level="warning", json_log=None, trust_pickle=False,
        yes=True, self_test=False, model=None)
    if extra:
        for k, v in extra.items():
            setattr(ns, k, v)
    return ns


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    operation_modes = {
        "--inspect": args.inspect,
        "--validation-only": args.validation_only,
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

    if args.model is None:
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
        fmt = FORMAT_W4A8
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
                f"cgs={decision.convrot_groupsize}")
        report = build_report(info, plan, env, args, detection, None, {},
                              summary, validation_input_hashes,
                              summary.get("output_sha256", ""),
                              time.time() - t_start, warnings, quant_rows)
        _write_reports(args, report)
        return 0 if summary["n_failed"] == 0 else 2

    fmt = FORMAT_W4A8

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

    # Sensitivity decisions are a planning operation.  They must be complete
    # before output shapes and offsets are frozen.
    needs_sensitivity_prepass = (
        args.sensitivity_threshold is not None
        or args.error_threshold is not None
        or args.calibration_source is not None
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
    if needs_sensitivity_prepass:
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
            f"{d.layer}: {tuple(m.shape)} gs={d.group_size} cgs={d.convrot_groupsize}")

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
        print(f"[dry-run] would write {len(entries)} tensors / {human_bytes(total_out)} "
              f"to {out_path}; {plan.n_quantized} layers quantized")
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



# ---------------------------------------------------------------------------
# Golden vectors for the bit-exactness self-test
# ---------------------------------------------------------------------------
# Captured from the reference implementation (comfy-kitchen merge commit
# aa1ab2263dc06225d9de6702dfc087313d4bc971, `quantize_w4a8_int8_weight`,
# eager backend, CPU, torch 2.13.0) on fixed-seed weights.
_GOLDEN_BLOB = """
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
a1Aumr+8+t858OtT7/zrx6im0bE96sdNfJ7jh384OjXKP52E8vGXt79ejWx6nX337FHo/b88jKd2iv5eM1P817dPPmZOLpj/auvk
8Pt1+esl5B+HtWeviE4Mnbn5w8gX/mlH1A/Dc2+EKdRHh1NPIP/++iVa5+ST7o/HT83UnHou7//q5bSZuqJ1XifWEB7zM2fl+S1W
8cnVZ/2ePenE/MkN/vQf/ZrkUdsiDfsfAZviWmsp9d4em6NNE/+7qrSlXPiqXCeTnfj2SvaWyI/Tse6UbvPmChnrCb5lefzEJ89/
n35tqR7XuKv863n+wrKOZemglzdWSx68dfNflly0XbmmrV7xy21A8r+ssuPwzbDjPXwfLLOwO3nPFg3nmc/nasncD5P3T6ll/KNq
PAtkFiXvXcPgfNOmoSVL+XLsH51J+LnUDOsR8IO71nL44N9bVFi3IejKjW80m388B8+6PcvpONRam3hn0z6Walew3JLn88GDSMut
OzG25ablocGD+0BZoL6O5caMg81rY+FYSinCY7vpv4z3aDhZmsiO7or/HiK/64Nlaao1yXuCRlGe+JK79Z9hIem+adx11X/Heuxz
ejl03dYvM7s5F7ocv8siss1IZ5kiYaZBDkv07RN5w3rME8yg/lidLLUHfdfNBMPY6mlRLK0dWBTUyiXan7y1/mcwyaM+RZv8b5b+
X/+DIcj/g97LJvQGwZ6W/D0lGfb/oRcrHfH/lU0U+V/o7T/L/u//XNMiy+f/NvRygPszVPeYK2TLl/hX5EFL3nf1ODzQskRdrv2i
p7wiVYbFrUx+kMziG7zIScNMOITPldy0A86QO8taFNo9mnK2Jo6OhWstepfw4HALYKzr2h142JdgbAV4WddMdiZzotn56Jh5f0pJ
/M6nq4iV408CbUb+4jbWA7ZOx7y0tEBgD6oEsXEh9FIxx47xJGaGQTcI8k6Au8c3RVYCreN796OBwOaz8tNZIBAUd1jN10NQ3cOG
HrmYbhkAnhWWvbvrsyxOx/lpUIWfSlIGzIDf3KGCNwl7cewOsGr13DKcCJK8l83PbGlal9ObtlZs7Op3oCpE57bVIj2mgiu7aRWF
xT0emhAdRpwtYOTZVdoqQ/Bopqxjq4VCKl0D60pPTz0frAcrFKUrPx8B7g7MfYp4+sllFQTVLJX+1XoO1HbYMWgSkuYAmH2Wt939
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
8uRV
"""


def _load_golden() -> Dict[str, Dict[str, Any]]:
    import base64 as _b64
    import zlib as _zlib
    payload = json.loads(_zlib.decompress(_b64.b64decode(_GOLDEN_BLOB)).decode("utf-8"))
    out = {}
    np_dtypes = {torch.float16: np.float16, torch.float32: np.float32,
                 torch.uint8: np.uint8, torch.int8: np.int8}

    def _tensor(record, key, dtype, shape):
        return torch.from_numpy(
            np.frombuffer(_b64.b64decode(record[key]),
                          dtype=np_dtypes[dtype]).copy()).view(shape)

    for name, g in payload.items():
        shape = tuple(g["shape"])
        out[name] = {
            "weight": _tensor(g, "weight_b64", torch.float16, shape),
            "packed": _tensor(
                g, "packed_b64", torch.int8, (shape[0], shape[1] // 2)),
            "s_rel_u8": _tensor(
                g, "s_rel_u8_b64", torch.uint8,
                (shape[0], shape[1] // g["gs"])),
            "s_channel": _tensor(
                g, "s_channel_b64", torch.float32, (shape[0],)),
            "codebook": _tensor(g, "codebook_b64", torch.float32, (16,)),
            "gs": g["gs"], "cgs": g["cgs"],
        }
    return out


GOLDEN = _load_golden()
GOLDEN_W4 = GOLDEN["w4a8_default"]


def _test_golden_vectors() -> str:
    """Golden vectors captured from the reference comfy-kitchen implementation.

    Packed int8 codes and fp8 (e4m3) scales are compared byte-exactly. The fp32
    scale fields (s_channel, codebook) are compared with a tight relative
    tolerance, because torch reductions (quantile, amax, mean) can differ in the
    last ULPs between platforms (x86 vs ARM, Windows vs Linux) while the packed
    output stays identical.
    """
    detail = []
    d_ch = 0.0
    d_cb = 0.0
    for name, g in GOLDEN.items():
        p, s_rel, s_ch, corr, cb = quantize_w4a8_weight(
            g["weight"], group_size=g["gs"], convrot_groupsize=g["cgs"])
        assert torch.equal(p, g["packed"]), f"{name}: packed mismatch vs reference"
        assert torch.equal(s_rel.view(torch.uint8), g["s_rel_u8"]), f"{name}: s_rel mismatch vs reference"
        d_ch = max(d_ch, float((s_ch - g["s_channel"]).abs().max()))
        assert torch.allclose(s_ch, g["s_channel"], rtol=1e-4, atol=1e-8), \
            f"{name}: s_channel max abs diff {d_ch:.3e} vs reference"
        d_cb = max(d_cb, float((cb - g["codebook"]).abs().max()))
        assert torch.allclose(cb, g["codebook"], rtol=1e-4, atol=1e-6), \
            f"{name}: codebook max abs diff {d_cb:.3e} vs reference"
        assert corr is None
        detail.append(f"{name}(gs={g['gs']},cgs={g['cgs']})")
    return ("golden match: packed/fp8 byte-exact, fp32 max diffs "
            f"s_ch={d_ch:.1e} cb={d_cb:.1e}")



if __name__ == "__main__":
    try:
        sys.exit(main())
    except QuantizerError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        sys.exit(130)
