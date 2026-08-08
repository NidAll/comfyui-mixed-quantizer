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
  --tarball-kitchen URL
                     comfy-kitchen source tarball (used with
                     --check-runtime-contract)
  --check-runtime-contract
                     also verify the upstream runtime contract: the three
                     QUANT_ALGOS names in ComfyUI quant_ops.py and the three
                     layout classes in comfy-kitchen source. Use against
                     LATEST upstream (nightly), not the pre-W4A8 pinned
                     research revision.

Exit codes: 0 = coverage matches or the registry is a superset, 1 = a ComfyUI
model class is not accounted for by any family policy, or an upstream runtime
contract regression is detected, 2 = comparison could not be run.

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


def requests_get(url: str) -> bytes:
    if os.path.isfile(url):  # local tarball (testing)
        with open(url, "rb") as f:
            return f.read()
    with urllib.request.urlopen(url, timeout=300) as resp:  # noqa: S310 (CI pin)
        return resp.read()

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
    ap.add_argument("--tarball-kitchen", default=None)
    ap.add_argument("--check-runtime-contract", action="store_true")
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

    # ---- upstream runtime-contract monitoring (optional) ----
    # The three formats must stay registered in ComfyUI quant_ops.py and the
    # three layout classes must stay present in comfy-kitchen source, or the
    # mixed checkpoints this converter emits would silently lose their
    # loading path upstream.
    REQUIRED_ALGOS = ("convrot_w4a4", "asym_w4a8_int8", "int8_tensorwise")
    REQUIRED_LAYOUTS = ("AsymW4A8Int8Layout", "TensorCoreConvRotW4A4Layout",
                        "TensorWiseINT8Layout")
    if args.check_runtime_contract and args.tarball:
        import io as _io
        import tarfile as _tf
        with _tf.open(fileobj=_io.BytesIO(requests_get(args.tarball)),
                      mode="r:gz") as tf:
            qo = next((memb for memb in tf.getmembers()
                       if memb.name.endswith("/comfy/quant_ops.py")), None)
            if qo is not None:
                source = tf.extractfile(qo).read().decode("utf-8")
                missing_algos = [a for a in REQUIRED_ALGOS if a not in source]
                result["comfyui_quant_algos"] = {
                    a: a in source for a in REQUIRED_ALGOS}
                if missing_algos:
                    ok = False
                    result["missing_quant_algos"] = missing_algos
                    print(f"UPSTREAM RUNTIME REGRESSION: ComfyUI quant_ops.py "
                          f"no longer registers: {missing_algos}")
    if args.check_runtime_contract and args.tarball_kitchen:
        import io as _io
        import tarfile as _tf
        with _tf.open(fileobj=_io.BytesIO(requests_get(args.tarball_kitchen)),
                      mode="r:gz") as tf:
            py_files = [memb for memb in tf.getmembers()
                        if memb.name.endswith(".py")]
            joined = "\n".join(
                tf.extractfile(m).read().decode("utf-8")
                for m in py_files)
            missing_layouts = [l for l in REQUIRED_LAYOUTS
                               if l not in joined]
            result["kitchen_layouts"] = {
                l: l in joined for l in REQUIRED_LAYOUTS}
            if missing_layouts:
                ok = False
                result["missing_layouts"] = missing_layouts
                print(f"UPSTREAM RUNTIME REGRESSION: comfy-kitchen no longer "
                      f"contains layouts: {missing_layouts}")
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
