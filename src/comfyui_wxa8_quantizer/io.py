"""Checkpoint discovery/reading, safetensors streaming writer, hashing and atomic publication."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from pathlib import Path
import contextlib
from dataclasses import dataclass, field
import hashlib
import json
import math
import mmap
import os
import shutil
import struct
import tempfile
import torch
from comfyui_wxa8_quantizer.constants import LAYER_CONF_KEY, MAX_SAFETENSORS_HEADER_SIZE, METADATA_KEY_EXT, METADATA_KEY_QUANT
from comfyui_wxa8_quantizer.errors import InputError, OutputError, PickleInputError
from comfyui_wxa8_quantizer.logging_utils import log
from comfyui_wxa8_quantizer.utils import FP8_DTYPES, SAFE_TO_TORCH, TORCH_TO_SAFE, _fsync_parent, _open_regular_nofollow, _same_path, human_bytes
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
# SPDX-License-Identifier: Apache-2.0
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

_HADAMARD_CACHE: Dict[Tuple[int, torch.device, torch.dtype], torch.Tensor] = {}

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

KNOWN_GATE_DIMS: Dict[str, Tuple[int, ...]] = {
    "boogu": (3360,),
    "omnigen2": (2520,),        # fails K % 16 too
    "pixart": (1152,),
    "hydit": (1408,),
    "cogvideox": (1920,),       # only the 2b variant; 5b (3072) is clean
    "minimax_h3": (1152,),      # mlp.fc2 only
    "sdxl": (320, 640),
    "sd15": (320,),
    "svd": (320, 640),
    "stable_cascade": (320, 640),
    "ace_step": (1152,),
}
