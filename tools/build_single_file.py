#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the single-file converter artifact from the modular package.

The canonical source of truth is the package under
``src/comfyui_wxa8_quantizer/``. This script bundles it into the standalone
``comfyui_wxa8_quantizer.py`` artifact (same CLI, same metadata writing, same
``--self-test`` suite), which is what users download and what the HF helpers
execute. The artifact embeds the golden reference vectors so its self-tests
stay self-contained.

Usage:
    python tools/build_single_file.py [--output comfyui_wxa8_quantizer.py]

The generated file must stay byte-stable for a given package revision:
run the artifact's ``--self-test`` (39/39) and a fixture conversion after
every change to the package.
"""
from __future__ import annotations

import argparse
import ast
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
PKG = REPO / "src" / "comfyui_wxa8_quantizer"
GOLDEN_DATA = PKG / "golden_data" / "v1.blob.txt"
DEFAULT_OUT = REPO / "comfyui_wxa8_quantizer.py"

# Bundle order: dependency order for module-level executed statements. Only
# policies.py executes calls at module level (_register), and they are
# self-contained, so this order is a readability/lint choice.
BUNDLE_ORDER = [
    "constants", "errors", "golden", "utils", "logging_utils", "io",
    "formats", "policies", "runtime", "planning", "quantize", "planner",
    "engine", "metadata", "reporting", "validation", "selftests", "cli",
]

HEADER = '''#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""comfyui_wxa8_quantizer.py -- standalone ComfyUI-native checkpoint quantizer.

GENERATED ARTIFACT. This file is built from the modular package under
src/comfyui_wxa8_quantizer/ by tools/build_single_file.py. Do not edit it
directly; edit the package and rebuild. The package and this artifact share
one implementation and produce identical outputs.

Converts supported generative-model checkpoints (safetensors / sharded
safetensors / HF-style model directories / optionally torch pickles with
--trust-pickle) into a ComfyUI-compatible quantized checkpoint. The default
mode produces W4A8 ("asym_w4a8_int8"). The experimental mixed mode
(--format mixed) selects per layer between convrot_w4a4 / asym_w4a8_int8 /
int8_tensorwise under quality gates; see AGENTS.md.

This utility is fully standalone. It does not import, require, or execute any
ComfyUI / comfy-kitchen / ComfyUI-custom-node code at runtime. Every
inspection, detection, quantization, packing, metadata and validation
component is reimplemented here from the verified reference behaviour
described in the package sources.

The verified W4A8/W4A4/INT8 format specifications (comfy-kitchen PR #90,
ComfyUI PR #15308, comfy-kitchen 0.2.28 eager kernels) are documented in the
README and in the original commit history of this repository.
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
        "comfyui_wxa8_quantizer requires: torch (>=2.1), safetensors (>=0.4.3), numpy.\\n"
        f"Import failed: {_exc!r}"
    ) from _exc

'''

ENTRY = '''
if __name__ == "__main__":
    sys.exit(main())
'''


def _module_body(module: str) -> str:
    """Return the module source without its docstring/import header."""
    text = (PKG / f"{module}.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    first = None
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str) and node.lineno <= 3:
            continue  # module docstring
        first = node
        break
    if first is None:
        return text
    start = first.lineno
    if isinstance(first, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) \
            and first.decorator_list:
        start = min(d.lineno for d in first.decorator_list)
    return "\n".join(text.splitlines()[start - 1:])


def build() -> str:
    parts = [HEADER.rstrip("\n")]
    for module in BUNDLE_ORDER:
        body = _module_body(module)
        if module == "golden":
            blob = GOLDEN_DATA.read_text(encoding="utf-8").strip()
            marker = "_GOLDEN_BLOB: Optional[str] = None  # embedded by tools/build_single_file.py in the single-file artifact"
            assert marker in body, "golden marker missing"
            body = body.replace(marker, f'_GOLDEN_BLOB = """{blob}"""')
            body += '\n\n# Eager globals for the standalone artifact (package loads lazily).\n' \
                    'GOLDEN = _load_golden()\nGOLDEN_W4 = GOLDEN["w4a8_default"]\n'
        parts.append(body.rstrip("\n"))
    parts.append(ENTRY.strip("\n"))
    return "\n\n".join(parts) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
