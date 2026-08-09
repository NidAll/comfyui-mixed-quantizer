"""Golden reference vectors for the W4A8/W4A4/INT8 self-tests.

The vectors were generated from comfy-kitchen 0.2.28 and are versioned test
data, stored at golden_data/v1.blob.txt (base64 + zlib JSON). They are loaded
LAZILY, only when a self-test calls get_golden(); normal imports never decode
them. The single-file artifact built by tools/build_single_file.py embeds the
blob directly so its --self-test stays self-contained.
"""
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from comfyui_wxa8_quantizer.formats import quantize_w4a8_weight

_GOLDEN_BLOB: Optional[str] = None  # embedded by tools/build_single_file.py in the single-file artifact
_GOLDEN_DATA_FILE = Path(__file__).resolve().parent / "golden_data" / "v1.blob.txt"
_GOLDEN: Optional[Dict[str, Dict[str, Any]]] = None


def _read_golden_blob() -> str:
    """Return the embedded blob, or read the versioned data file."""
    if _GOLDEN_BLOB is not None:
        return _GOLDEN_BLOB
    if not _GOLDEN_DATA_FILE.is_file():
        raise FileNotFoundError(
            f"golden data missing: {_GOLDEN_DATA_FILE} (reinstall the package "
            "or rebuild the single-file artifact)")
    return _GOLDEN_DATA_FILE.read_text(encoding="utf-8")


def get_golden() -> Dict[str, Dict[str, Any]]:
    """Lazily load and cache the golden reference vectors."""
    global _GOLDEN  # noqa: PLW0603
    if _GOLDEN is None:
        _GOLDEN = _load_golden()
    return _GOLDEN


def _load_golden() -> Dict[str, Dict[str, Any]]:
    import base64 as _b64
    import zlib as _zlib
    payload = json.loads(_zlib.decompress(_b64.b64decode(_read_golden_blob())).decode("utf-8"))
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


# Eager globals for the single-file artifact (set by the build script after it
# embeds the blob). The package keeps them None until get_golden() is called.
GOLDEN: Optional[Dict[str, Dict[str, Any]]] = None
GOLDEN_W4: Optional[Dict[str, Any]] = None


def _test_golden_vectors() -> str:
    """Golden-vector regression: packed/fp8 byte-exact, fp32 scales tolerant.

    The embedded reference weights were produced on x86_64 with comfy-kitchen
    0.2.28. Packed int8 codes and fp8 (e4m3) scales are compared byte-exactly.
    The fp32 scale fields (s_channel, codebook) are compared with a tight
    relative tolerance, because torch reductions (quantile, amax, mean) can
    differ in the last ULPs between platforms (x86 vs ARM, Windows vs Linux)
    while the packed output stays identical.
    """
    golden = get_golden()
    detail = []
    d_ch = 0.0
    d_cb = 0.0
    for name, g in golden.items():
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
