"""Version, format and revision constants (single source of truth)."""

from __future__ import annotations

CONVERTER_NAME = "comfyui_wxa8_quantizer"

_CONVERTER_VERSION = "1.4.0-experimental"


def get_converter_version() -> str:
    """Return the current converter version (observable at call time).

    Internal code must read the version through this function (or
    set_converter_version) so that plan-hash/resume checks observe changes;
    the self-tests temporarily mutate it to prove version drift is rejected.
    """
# SPDX-License-Identifier: Apache-2.0
    return _CONVERTER_VERSION


def set_converter_version(version: str) -> None:
    """Override the converter version (self-tests only; not for production use)."""
    global _CONVERTER_VERSION  # noqa: PLW0603
    _CONVERTER_VERSION = version


# Module-level attribute kept for external compatibility and for the
# generated single-file artifact; internal readers use the functions above.
CONVERTER_VERSION = _CONVERTER_VERSION

FORMAT_W4A8 = "asym_w4a8_int8"

FORMAT_MIXED = "mixed"

FORMAT_W4A4 = "convrot_w4a4"

FORMAT_INT8 = "int8_tensorwise"

MIXED_FORMATS = (FORMAT_W4A8, FORMAT_W4A4, FORMAT_INT8)

FORMAT_ORIGINAL = "original"   # planner-only candidate: keep at source precision

W4A4_QUANT_GROUP_SIZE = 64

W4A4_EMISSION_MAX = 7

DEFAULT_W4A4_LINEAR_DTYPE = "int8"  # default execution variant: "int4" or "int8"

INT8_SCALE_MAX = 127

W4A4_MAX_REL_L2 = 0.20

INT8_MAX_REL_L2 = 0.05

MIXED_PROFILES = ("auto", "balanced", "conservative", "size-first")

FORMAT_MIXED_REVISION = "mixed-r1"

W4A8_CONVROT_GROUPSIZE = 256

QUANT_ALGORITHM_REVISION = "lloydmax-codebook-r1"

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
