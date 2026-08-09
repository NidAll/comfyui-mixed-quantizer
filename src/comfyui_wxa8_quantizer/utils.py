"""Small shared utilities: sizes, hashing, atomic file helpers, dtype tables."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from pathlib import Path
import contextlib
import hashlib
import json
import math
import os
import re
import safetensors
import safetensors.torch
import stat
import struct
import sys
import tempfile
import torch
from comfyui_wxa8_quantizer.errors import InputError, OutputError, UsageError
def parse_size(text: str) -> int:
    """Parse a size like 2G, 512M, 1024K or a plain byte count."""
# SPDX-License-Identifier: Apache-2.0
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
