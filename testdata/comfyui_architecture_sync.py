#!/usr/bin/env python3
"""Architecture coverage synchronization check.

Compares the converter's embedded policy registry (the union of
`comfyui_classes` across all 43 families) against the ComfyUI
`comfy/supported_models.py` class set at a pinned revision.

Modes:
  --comfy-src DIR    compare against a local ComfyUI source tree
                     (default: ../research/ComfyUI relative to the repo root)
  --tarball URL      download and compare against a ComfyUI source tarball
                     (codeload URL; used by CI when no local checkout exists)
  --pinned REV       ComfyUI revision label for the report (defaults to the
                     converter's COMFYUI_BASE constant)

Exit codes: 0 = coverage matches or the registry is a superset, 1 = a ComfyUI
model class is not accounted for by any family policy, 2 = comparison could
not be run.

The registry may legally cover MORE classes than the checked revision (older
revisions keep coverage). It must never cover FEWER. Unknown architectures
fail closed at conversion time via --architecture, so a new class surfaces
here before it can ever be silently mis-quantized.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def registry_classes() -> set[str]:
    sys.path.insert(0, REPO)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "wxa8_registry", os.path.join(REPO, "comfyui_wxa8_quantizer.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wxa8_registry"] = mod
    spec.loader.exec_module(mod)
    covered: set[str] = set()
    for family in mod.family_names():
        covered.update(mod.get_family(family).comfyui_classes)
    return covered


def comfy_classes_from_source(source: str) -> set[str]:
    import ast
    tree = ast.parse(source)
    names = {node.name for node in tree.body
             if isinstance(node, ast.ClassDef)}
    return names


def comfy_classes_from_tree(root: str) -> set[str]:
    path = os.path.join(root, "comfy", "supported_models.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        return comfy_classes_from_source(f.read())


def comfy_classes_from_tarball(url: str) -> set[str]:
    with urllib.request.urlopen(url, timeout=300) as resp:  # noqa: S310 (CI pin)
        data = resp.read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        member = next((m for m in tf.getmembers()
                       if m.name.endswith("/comfy/supported_models.py")), None)
        if member is None:
            raise RuntimeError("supported_models.py not found in tarball")
        return comfy_classes_from_source(
            tf.extractfile(member).read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--comfy-src", default=None)
    ap.add_argument("--tarball", default=None)
    ap.add_argument("--pinned", default=None)
    ap.add_argument("--json", default=None, metavar="PATH")
    args = ap.parse_args()

    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location(
        "wxa8_pin", os.path.join(REPO, "comfyui_wxa8_quantizer.py"))
    mod = _ilu.module_from_spec(spec)
    sys.modules["wxa8_pin"] = mod
    spec.loader.exec_module(mod)
    pinned = args.pinned or mod.COMFYUI_BASE

    try:
        covered = registry_classes()
    except Exception as e:  # pragma: no cover
        print(f"registry import failed: {e}")
        return 2

    if args.tarball:
        try:
            comfy = comfy_classes_from_tarball(args.tarball)
        except Exception as e:
            print(f"tarball fetch failed: {e}")
            return 2
    else:
        src = args.comfy_src or os.path.join(REPO, "research", "ComfyUI")
        try:
            comfy = comfy_classes_from_tree(src)
        except FileNotFoundError as e:
            print(f"no ComfyUI source tree at {e.filename}; pass --comfy-src "
                  "or --tarball")
            return 2

    missing = sorted(comfy - covered)
    extra = sorted(covered - comfy)
    ok = not missing
    result = {
        "pinned_revision": pinned,
        "comfy_classes": len(comfy),
        "registry_classes": len(covered),
        "unaccounted": missing,
        "registry_only": extra,
        "ok": ok,
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    print(f"pinned ComfyUI revision : {pinned}")
    print(f"ComfyUI model classes   : {len(comfy)}")
    print(f"registry coverage       : {len(covered)}")
    if missing:
        print(f"UNACCOUNTED CLASSES ({len(missing)}):")
        for name in missing:
            print(f"  - {name}")
        print("add a policy family (or an explicit passthrough-only family) "
              "for each class before this check passes.")
        return 1
    if extra:
        print(f"registry-only classes   : {len(extra)} (older revision "
              "coverage; allowed)")
    print("OK: every ComfyUI model class at this revision is covered by a "
          "registered policy family.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
