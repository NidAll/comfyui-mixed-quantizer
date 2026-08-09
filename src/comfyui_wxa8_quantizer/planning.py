"""Tensor classification, conversion plans, calibration and quality metrics."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from pathlib import Path
import contextlib
from dataclasses import dataclass, field
import enum
import json
import math
import numpy as np
import os
import tempfile
import torch
import zipfile
from comfyui_wxa8_quantizer.constants import FORMAT_INT8, FORMAT_MIXED, FORMAT_W4A4, FORMAT_W4A8, INT8_SCALE_MAX, W4A4_EMISSION_MAX, W4A8_CONVROT_GROUPSIZE
from comfyui_wxa8_quantizer.errors import CalibrationError, PolicyError
from comfyui_wxa8_quantizer.formats import build_hadamard, decode_w4a8_runtime_weight, rotate_activation, unpack_int4_signed, w4_weight_is_quantizable
from comfyui_wxa8_quantizer.io import CheckpointInfo, _load_json_object, _resolve_under, tensor_nbytes
from comfyui_wxa8_quantizer.logging_utils import log
from comfyui_wxa8_quantizer.policies import DetectionResult
from comfyui_wxa8_quantizer.utils import FLOAT_DTYPES, _fsync_parent, _open_regular_nofollow, flatten_regex, human_bytes, json_dumps, sha256_file
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
    format: str = FORMAT_W4A8      # per-layer format (mixed mode)
    linear_dtype: Optional[str] = None   # W4A4 execution variant (per decision)
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
    mixed_plan: Optional[Dict[str, Any]] = None        # mixed-planner summary

    def quantized_layers(self) -> List[TensorDecision]:
        return [d for d in self.decisions if d.kind == DecisionKind.QUANTIZE]

def classify_tensors(info: CheckpointInfo, detection: DetectionResult,
                     fmt: str, group_size: Optional[int], include: Sequence[str],
                     exclude: Sequence[str], keep_precision: Sequence[str],
                     output_dtype: Optional[torch.dtype],
                     min_numel: Optional[int]) -> List[TensorDecision]:
    """Decide, for every input tensor, whether it is quantized or passed through."""
# SPDX-License-Identifier: Apache-2.0
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
        user_forced = include_re is not None and include_re.search(name)
        if user_forced:
            candidate = True
        if not candidate:
            decisions.append(TensorDecision(name, DecisionKind.KEEP,
                                            f"not in {policy.family} quantize set; passthrough"))
            continue

        # shape / dtype gates. W4A8 ConvRot is always 256-wide: the
        # comfy-kitchen 0.2.27 CUDA fused kernels throw unless
        # convrot_groupsize == 256, so in w4a8 mode layers whose K is not
        # divisible by 256 pass through. Mixed mode instead keeps every 2D
        # float linear as a candidate and lets the planner pick per-layer
        # formats (W4A4 when K % 64 == 0, W4A8 when K % 256 == 0, INT8 for
        # any K), so dimension-incompatible layers stay quantizable.
        if fmt == FORMAT_W4A8:
            ok, why = w4_weight_is_quantizable(meta.shape, meta.dtype, group_size,
                                               W4A8_CONVROT_GROUPSIZE)
            if not ok:
                decisions.append(TensorDecision(name, DecisionKind.KEEP,
                                                f"not quantizable: {why}; passthrough"))
                continue
        elif fmt == FORMAT_MIXED:
            if len(meta.shape) != 2 or meta.dtype not in FLOAT_DTYPES:
                decisions.append(TensorDecision(name, DecisionKind.KEEP,
                                                "not a 2D float weight; passthrough"))
                continue
        else:
            raise PolicyError(f"unknown quantization format {fmt!r}")
        if meta.nbytes < min_numel * meta.dtype.itemsize:
            decisions.append(TensorDecision(name, DecisionKind.KEEP,
                                            f"small tensor (<{min_numel} elements); passthrough"))
            continue
        decisions.append(TensorDecision(
            name, DecisionKind.QUANTIZE,
            "user-forced via --include" if user_forced else "policy candidate",
            layer=layer, group_size=group_size,
            convrot_groupsize=W4A8_CONVROT_GROUPSIZE,
            format=fmt))

    return decisions

def build_output_entries(info: CheckpointInfo, decisions: List[TensorDecision],
                         fmt: str, output_dtype: Optional[torch.dtype],
                         ) -> Tuple[List[Dict[str, Any]], int]:
    """Compute the exact output tensor inventory (name, dtype, shape, nbytes)
    in deterministic write order: quantized layers first (original weight slot),
    then passthrough tensors in input order."""
    if fmt not in (FORMAT_W4A8, FORMAT_MIXED):
        raise PolicyError(f"unknown quantization format {fmt!r}")
    entries: List[Dict[str, Any]] = []
    total = 0
    seen = set()
    for d in decisions:
        if d.kind == DecisionKind.QUANTIZE:
            layer = d.layer
            meta = info.by_name(d.name)
            n, k = int(meta.shape[0]), int(meta.shape[1])
            base = f"{layer}.weight"
            if d.format == FORMAT_W4A8:
                groups = k // d.group_size
                extras = [
                    (f"{layer}.weight_s_rel", torch.float8_e4m3fn, (n, groups)),
                    (f"{layer}.weight_s_channel", torch.float32, (n,)),
                    (f"{layer}.weight_codebook", torch.float32, (16,)),
                ]
                payload = [(base, torch.int8, (n, k // 2))] + extras
            elif d.format == FORMAT_W4A4:
                payload = [(base, torch.int8, (n, k // 2)),
                           (f"{layer}.weight_scale", torch.float32, (n,))]
            elif d.format == FORMAT_INT8:
                payload = [(base, torch.int8, (n, k)),
                           (f"{layer}.weight_scale", torch.float32, (n, 1))]
            else:
                raise PolicyError(
                    f"unknown per-layer format {d.format!r} for {d.name!r}")
            for ename, edtype, eshape in payload:
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

def _act_quant_int8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dynamic rowwise int8 activation quantization (runtime behavior)."""
    abs_max = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    scale = abs_max / float(INT8_SCALE_MAX)
    q = (x / scale).round().clamp_(-128.0, 127.0).to(torch.int8)
    return q, scale

def _act_quant_int4(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dynamic rowwise signed int4 activation quantization (eager W4A4 path)."""
    abs_max = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scale = abs_max / float(W4A4_EMISSION_MAX)
    q = (x / scale).round().clamp_(-float(W4A4_EMISSION_MAX),
                                   float(W4A4_EMISSION_MAX)).to(torch.int8)
    return q, scale

def _simulate_quantized_chunk(original_chunk: torch.Tensor,
                              tensors: Dict[str, torch.Tensor], fmt: str,
                              group_size: int, convrot_groupsize: int,
                              act_q: torch.Tensor, act_scale: torch.Tensor,
                              linear_dtype: str) -> torch.Tensor:
    """Exact eager-path emulation of the quantized linear for one weight chunk.

    Returns y_q [S, rows] for the chunk. The activation was already rotated
    and quantized once (act_q, act_scale). The emulations mirror the
    comfy-kitchen eager kernels:
      int8_tensorwise: (x_int8 @ w_int8.T) * x_scale * w_scale
      convrot_w4a4:    (x_int4/8 @ w_int4.T) * x_scale * w_scale
      asym_w4a8_int8:  (x_int8 * x_scale) @ dequant(W).T
    """
    if fmt == FORMAT_INT8:
        q, wscale = tensors[""], tensors["_scale"]
        return ((act_q.float() @ q.float().T)
                * act_scale * wscale.float().reshape(1, -1))
    if fmt == FORMAT_W4A4:
        w_int = unpack_int4_signed(tensors[""]).float()
        wscale = tensors["_scale"].float().reshape(1, -1)
        return ((act_q.float() @ w_int.T)
                * act_scale * wscale)
    if fmt == FORMAT_W4A8:
        # the runtime GEMM operates in the ConvRot basis: int8 activation
        # rows against the ROTATED int8 weight, scaled by the activation and
        # channel scales. Do NOT use the inverse-rotated physical weight here.
        w_int8 = decode_w4a8_runtime_weight(
            tensors[""], tensors["_s_rel"], tensors.get("_codebook"),
            group_size)
        s_channel = tensors["_s_channel"].float().reshape(1, -1)
        return (act_q.float() @ w_int8.float().T) * act_scale * s_channel
    raise PolicyError(f"unknown format {fmt!r} for runtime simulation")

def runtime_output_rel_l2(original: torch.Tensor,
                          tensors: Dict[str, torch.Tensor], fmt: str,
                          group_size: int, convrot_groupsize: int,
                          activations: torch.Tensor,
                          w4a4_activation_bits: int = 8) -> Optional[float]:
    """mean over samples of ||Y_quant - Y_bf16|| / ||Y_bf16|| for the actual
    runtime operation (activation rotation, activation quantization, quantized
    GEMM with scales), not a reconstructed-weight approximation.

    w4a4_activation_bits is the EFFECTIVE W4A4 activation precision on the
    target runtime (4 on eager regardless of linear_dtype, 4 or 8 on CUDA)."""
    x = activations.float()
    y_ref = x @ original.float().T
    if fmt in (FORMAT_W4A4, FORMAT_W4A8):
        h = build_hadamard(convrot_groupsize, device=x.device, dtype=torch.float32)
        x_rot = rotate_activation(x, h, convrot_groupsize)
    else:
        x_rot = x
    if fmt == FORMAT_W4A4 and w4a4_activation_bits == 4:
        act_q, act_scale = _act_quant_int4(x_rot)
    else:
        act_q, act_scale = _act_quant_int8(x_rot)
    y_q = _simulate_quantized_chunk(original, tensors, fmt, group_size,
                                    convrot_groupsize, act_q, act_scale,
                                    "int4" if w4a4_activation_bits == 4
                                    else "int8")
    num = (y_q - y_ref).norm(dim=1)
    den = y_ref.norm(dim=1).clamp(min=1e-8)
    value = float((num / den).mean())
    return value if math.isfinite(value) else 1e30

class _OutputErrorAccumulator:
    """Chunked accumulation of the per-sample runtime output error."""

    def __init__(self, n_samples: int):
        self.num_sq = torch.zeros(n_samples, dtype=torch.float64)
        self.den_sq = torch.zeros(n_samples, dtype=torch.float64)

    def update(self, y_q: torch.Tensor, y_ref: torch.Tensor) -> None:
        d = (y_q.double() - y_ref.double())
        self.num_sq += d.square().sum(dim=1)
        self.den_sq += y_ref.double().square().sum(dim=1)

    def finish(self) -> float:
        ratio = (self.num_sq.sqrt() / self.den_sq.clamp(min=1e-16).sqrt())
        value = float(ratio.mean())
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
