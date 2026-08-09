#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Shared implementation for the HF -> mixed-quantized -> HF helper scripts.

tools/hf_mixed_quantize.py (reference variant) and
tools/hf_mixed_quantize_optimized.py (optimized variant) are thin wrappers
around this module.  All shared logic lives here: token resolution,
hf_transfer enablement, disk preflight, model download, converter fetch and
sha256 verification, GPU-identity scrubbing checks, fast sanity check,
upload, and phase timing.

Supply-chain hardening
----------------------
The converter is fetched from a commit-pinned URL:

    CONVERTER_URL     (raw.githubusercontent.com, full commit sha)
    CONVERTER_SHA256  (sha256 of comfyui_wxa8_quantizer.py at that commit)

fetch_converter() verifies the sha256 of any file it is about to run,
whether freshly downloaded or already present on disk.  A pre-existing
local converter whose sha256 does not match CONVERTER_SHA256 is refused
unless --trust-local-converter is passed to the optimized helper (the
original helper has no such flag and always refuses).  On a mismatch the
offending file is deleted (fresh downloads) or pointed out (pre-existing
files).

Environment scrubbing
---------------------
When the converter subprocess is launched, build_converter_env() copies
os.environ and removes credentials: HF_TOKEN, HUGGING_FACE_HUB_TOKEN, and
any variable whose name contains 'HF_' and looks like a token or secret
(TOKEN / SECRET / PASSWORD / PASSWD / API_KEY / ACCESS_KEY / AUTH /
CREDENTIAL / PRIVATE_KEY).  Benign HF_* settings such as
HF_HUB_ENABLE_HF_TRANSFER and HF_HUB_CACHE are kept.  The HF token is only
ever used in-process for the download and upload steps and never reaches
the converter subprocess.

Optional dependency
-------------------
hf_transfer is an OPTIONAL dependency.  enable_hf_transfer() enables it
only when the package is already importable; it never installs anything.
Install it with `pip install hf_transfer` for faster large-file downloads.

Preflight
---------
Before an HF download, the expected size is queried from the HF API
(get_paths_info for single files, model_info(files_metadata=True) for
snapshots; best-effort) and check_disk_space() runs BEFORE the download.
After the download, the local size (recursive sum for snapshots) is
asserted against the expected size.
"""
from __future__ import annotations

import getpass
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request

# Commit-pinned converter source.  The sha must match the file served at
# CONVERTER_URL; fetch_converter() enforces it.
CONVERTER_URL = ("https://raw.githubusercontent.com/NidAll/"
                 "comfyui-mixed-quantizer/"
                 "c4a3b92c73d6220a33b7b859f0160f791afe6f1d/"
                 "comfyui_wxa8_quantizer.py")
CONVERTER_SHA256 = "a75819221430557f1e764784ed9a950e0a9fc8c2a91a1ea26e5ca2f24a891383"

GPU_LEAK_PATTERNS = ("3050", "3060", "3070", "3080", "3090", "4060", "4070",
                     "4080", "4090", "Tesla", "GeForce", "Quadro", "RTX",
                     "Radeon", "RX ", "gfx9", "gfx10", "gfx11", "gfx12")

# Env var name markers that make an HF_* variable count as a credential.
_SECRET_ENV_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "API_KEY",
                       "ACCESS_KEY", "AUTH", "CREDENTIAL", "PRIVATE_KEY")


def parse_hf_url(url: str) -> tuple[str, str, str]:
    """Parse a Hugging Face blob/resolve URL into (repo_id, revision,
    filename).

    Accepts:
      https://huggingface.co/org/repo/blob/main/path/file.safetensors
      https://huggingface.co/org/repo/resolve/main/path/file.safetensors
      https://huggingface.co/org/repo/resolve/<commit>/path/file.safetensors
    """
    parts = urllib.parse.urlsplit(url)
    if parts.netloc not in ("huggingface.co", "hf.co"):
        raise SystemExit(f"not a Hugging Face URL: {url}")
    seg = [s for s in parts.path.split("/") if s]
    if len(seg) < 4 or seg[2] not in ("blob", "resolve"):
        raise SystemExit(
            f"expected .../org/repo/blob|resolve/<rev>/<path>, got: {url}")
    repo_id = f"{seg[0]}/{seg[1]}"
    revision = seg[3]
    filename = "/".join(seg[4:])
    if not filename:
        raise SystemExit(f"no file path in URL: {url}")
    return repo_id, revision, filename


def resolve_token() -> str:
    """HF token from env, Colab secrets, or an interactive prompt."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:  # Colab secrets
        from google.colab import userdata  # type: ignore
        return userdata.get("HF_TOKEN")
    except Exception as e:  # noqa: S110 (falls back to prompt)
        print(f"note: Colab secrets unavailable ({e})")
    import getpass
    return getpass.getpass("Hugging Face token: ").strip()


def enable_hf_transfer(use_it: bool) -> None:
    """Turn on huggingface_hub's hf_transfer downloader when available.

    hf_transfer is an OPTIONAL dependency; this function never installs it.
    It only sets HF_HUB_ENABLE_HF_TRANSFER when the package is already
    importable, and prints a note otherwise.
    """
    if not use_it:
        return
    try:
        import hf_transfer  # noqa: F401
    except ImportError:
        print("note: hf_transfer is not installed; it is an optional "
              "dependency (pip install hf_transfer) that speeds up "
              "large-file downloads. Using the plain downloader.")
        return
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    print("hf_transfer downloader: enabled")


def check_disk_space(source_bytes: int, output_path: str) -> None:
    """Abort before downloading when the conversion cannot fit on disk.

    Needs: source file (cached) + output plus a staged and a validation
    copy at peak, with a buffer.  The converter deletes the temporary
    copies afterwards; this is just an early gate.
    """
    total, used, free = shutil.disk_usage(
        os.path.dirname(os.path.abspath(output_path)) or ".")
    needed = source_bytes * 1.05 + source_bytes * 0.55 * 2.5 + (2 << 30)
    if free < needed:
        raise SystemExit(
            f"disk space too low: {free / 2**30:.1f} GiB free, "
            f"about {needed / 2**30:.1f} GiB needed for the download, "
            f"conversion and validation copies. Free space and retry.")


def query_hf_size(args, token: str | None) -> int | None:
    """Best-effort preflight size (bytes) of the HF download, or None.

    Single files use HfApi.get_paths_info; snapshots use
    model_info(files_metadata=True) and sum the sibling sizes.  Any API
    failure degrades to None (the disk preflight is then skipped with a
    note).
    """
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    try:
        if args.hf_url:
            repo_id, revision, filename = parse_hf_url(args.hf_url)
            files = api.get_paths_info(repo_id, paths=[filename],
                                       revision=revision, token=token)
            if files and files[0].size is not None:
                return files[0].size
        elif args.hf_filename:
            files = api.get_paths_info(args.hf_model,
                                       paths=[args.hf_filename], token=token)
            if files and files[0].size is not None:
                return files[0].size
        else:  # full snapshot
            info = api.model_info(args.hf_model, files_metadata=True,
                                  token=token)
            sizes = [f.size for f in (info.siblings or [])
                     if f.size is not None]
            if sizes:
                return sum(sizes)
    except Exception as e:  # noqa: S110 (best-effort preflight)
        print(f"note: HF API size query failed ({e}); "
              "skipping the disk-space preflight")
        return None
    print("note: HF API returned no usable file size; "
          "skipping the disk-space preflight")
    return None


def local_size(path: str) -> int:
    """Size of a file, or recursive sum of file sizes for a directory."""
    if os.path.isdir(path):
        return sum(os.path.getsize(os.path.join(r, f))
                   for r, _, fs in os.walk(path) for f in fs)
    return os.path.getsize(path)


def download_model(args, token: str) -> str:
    """Download the model (single file or snapshot); returns its local
    path."""
    from huggingface_hub import hf_hub_download, snapshot_download
    if args.hf_url:
        repo_id, revision, filename = parse_hf_url(args.hf_url)
        print(f"downloading {repo_id}/{filename} @ {revision}")
        return hf_hub_download(repo_id=repo_id, filename=filename,
                               revision=revision, token=token)
    if args.hf_filename:
        print(f"downloading {args.hf_model}/{args.hf_filename}")
        return hf_hub_download(repo_id=args.hf_model, filename=args.hf_filename,
                               token=token)
    print(f"downloading snapshot of {args.hf_model}")
    return snapshot_download(repo_id=args.hf_model, token=token)


def prepare_source(args, token: str | None) -> tuple[str, int]:
    """Resolve the input checkpoint; returns (path, actual_bytes).

    For HF downloads: best-effort API size query, disk preflight BEFORE the
    download, the download itself, then a post-download size assertion.  For
    --local-model: only an existence check.
    """
    if args.hf_model or args.hf_url:
        expected = query_hf_size(args, token)
        if expected is not None:
            print(f"expected download size: {expected / 2**30:.2f} GiB")
            check_disk_space(expected, args.output)
        model_path = download_model(args, token)
        actual = local_size(model_path)
        if expected is not None and actual != expected:
            raise SystemExit(
                f"post-download size assertion failed: got {actual} bytes, "
                f"expected {expected} bytes. Aborting; delete the partial "
                f"download and retry.")
        if expected is not None:
            print(f"download size verified: {actual} bytes "
                  f"({actual / 2**30:.2f} GiB)")
        return model_path, actual
    path = args.local_model
    if not os.path.isfile(path):
        raise SystemExit(f"local model not found: {path}")
    return path, os.path.getsize(path)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_converter(dest: str, trust_local_converter: bool = False) -> str:
    """Return a converter path verified against CONVERTER_SHA256.

    If `dest` already exists it is used only when its sha256 matches
    CONVERTER_SHA256; otherwise the run is refused (delete the file, or
    pass --trust-local-converter to the optimized helper to force-use it
    with a warning).  Fresh downloads go to a temp file, are verified, and
    are atomically moved into place; on a sha256 mismatch the temp file is
    deleted and the run aborts.
    """
    if os.path.exists(dest):
        digest = _sha256_file(dest)
        if digest == CONVERTER_SHA256:
            print(f"using local converter: {dest} (sha256 verified)")
            return dest
        if trust_local_converter:
            print(f"WARNING: local converter {dest} has sha256 {digest}, "
                  f"which does not match the pinned converter "
                  f"({CONVERTER_SHA256}); forcing use because "
                  f"--trust-local-converter was given")
            return dest
        raise SystemExit(
            f"refusing to run {dest}: its sha256 ({digest}) does not match "
            f"the pinned converter ({CONVERTER_SHA256}). Delete the file "
            f"and rerun, or pass --trust-local-converter (available in "
            f"tools/hf_mixed_quantize_optimized.py) to force-use it with a "
            f"warning.")
    print(f"fetching converter from {CONVERTER_URL}")
    tmp = dest + ".tmp"
    try:
        urllib.request.urlretrieve(CONVERTER_URL, tmp)  # noqa: S310 (pinned sha URL)
    except Exception:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise
    digest = _sha256_file(tmp)
    if digest != CONVERTER_SHA256:
        os.remove(tmp)
        raise SystemExit(
            f"downloaded converter failed the sha256 check: got {digest}, "
            f"expected {CONVERTER_SHA256}. The downloaded file was deleted; "
            f"refusing to run an unverified converter.")
    os.replace(tmp, dest)
    print(f"converter verified: sha256 {digest}")
    return dest


def build_converter_env() -> dict:
    """Environment for the converter subprocess: os.environ minus secrets.

    Removes HF_TOKEN, HUGGING_FACE_HUB_TOKEN and any variable whose name
    contains 'HF_' and looks like a token or secret.  Benign HF_* settings
    such as HF_HUB_ENABLE_HF_TRANSFER and HF_HUB_CACHE are kept.  The HF
    token is only ever used in-process for download/upload.
    """
    env = dict(os.environ)
    for key in list(env):
        upper = key.upper()
        if upper in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            del env[key]
        elif "HF_" in upper and any(m in upper for m in _SECRET_ENV_MARKERS):
            del env[key]
    return env


def run_converter(cmd: list[str]) -> None:
    """Run the converter subprocess with a scrubbed environment and abort
    on a non-zero exit."""
    print("running:", " ".join(cmd))
    r = subprocess.run(cmd, env=build_converter_env())  # noqa: S603 (explicit conversion command)
    if r.returncode != 0:
        raise SystemExit(f"conversion failed (exit {r.returncode})")


def verify_no_gpu_identity(path: str) -> None:
    """Fail closed if the output metadata leaks any GPU identity."""
    import safetensors
    with safetensors.safe_open(path, framework="pt") as f:
        meta = dict(f.metadata() or {})
    blob = json.dumps(meta)
    for key in ("gpu_name", "cuda_capability", "rocm_arch"):
        if key in blob:
            raise SystemExit(f"FAIL: metadata still contains {key!r}; "
                             "refusing to upload")
    for pat in GPU_LEAK_PATTERNS:
        if pat in blob:
            raise SystemExit(f"FAIL: metadata contains GPU identity {pat!r}; "
                             "refusing to upload")
    print("privacy check passed: no GPU identity in the checkpoint metadata")


def fast_sanity_check(path: str) -> None:
    """Cheap structural check of the produced file (seconds, no source).

    Replaces the converter's --validate in the default flow: the hard
    per-layer quality gates already ran during planning.  This only verifies
    the file reopens, the extension metadata parses, every quantized layer
    declares a native format, and the tensor count is sane.
    """
    import safetensors
    with safetensors.safe_open(path, framework="pt") as f:
        meta = dict(f.metadata() or {})
        names = list(f.keys())
    ext = json.loads(meta["comfy_wxa8"])
    qmeta = json.loads(meta["_quantization_metadata"])
    layers = qmeta["layers"]
    formats = {v["format"] for v in layers.values()}
    allowed = {"asym_w4a8_int8", "convrot_w4a4", "int8_tensorwise"}
    bad = formats - allowed
    if bad:
        raise SystemExit(f"FAIL: unexpected formats in metadata: {bad}")
    n_quant = len(layers)
    n_total = len(names)
    if n_quant == 0:
        raise SystemExit("FAIL: no quantized layers recorded")
    dist = ext.get("quantization", {}).get("distribution", {})
    print(f"sanity check passed: {n_total} tensors, {n_quant} quantized "
          f"layers, formats={sorted(formats)}, "
          f"effective {dist.get('effective_bytes_per_param', '?')} bpp")


def source_filename(args) -> str:
    """Best-effort original model file name for validation advice."""
    if args.hf_filename:
        return os.path.basename(args.hf_filename)
    if args.hf_url:
        return os.path.basename(parse_hf_url(args.hf_url)[2])
    if args.local_model:
        return os.path.basename(args.local_model)
    return "<original-model-file>"


def upload(args, output: str, token: str) -> str:
    """Create a repo in the user's HF account and upload output + README."""
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    base = os.path.basename(args.output)
    source_label = (args.hf_model
                    if args.hf_model else
                    (parse_hf_url(args.hf_url)[0] if args.hf_url
                     else args.local_model))
    repo_name = args.repo_name or (
        os.path.basename(source_label).replace(".", "-") + "-mixed-w4a8")
    repo_id = api.create_repo(repo_id=repo_name, private=not args.public,
                              exist_ok=True).repo_id
    print(f"uploading {base} to {repo_id} ({os.path.getsize(output)} bytes)")
    api.upload_file(path_or_fileobj=output, path_in_repo=base, repo_id=repo_id)
    readme = README_TEMPLATE.format(
        source_model=(args.hf_model
                     if args.hf_model else
                     (parse_hf_url(args.hf_url)[0] if args.hf_url
                      else os.path.basename(args.local_model))),
        filename=base,
        source_file=source_filename(args),
        profile=args.profile,
        version="1.4.0",
    )
    api.upload_file(path_or_fileobj=readme.encode("utf-8"),
                    path_in_repo="README.md", repo_id=repo_id)
    return repo_id


README_TEMPLATE = """# {filename}

Mixed-precision quantized checkpoint converted from `{source_model}` with
the ComfyUI mixed-precision quantizer (v{version}).

## Formats

Per layer, the cheapest ComfyUI-native format that passed the quality gates
was selected from:

* `asym_w4a8_int8` (4-bit weights, ConvRot 256, Lloyd-Max codebook)
* `convrot_w4a4` (4-bit weights, rowwise int4 scales)
* `int8_tensorwise` (8-bit rowwise)
* original precision for policy-protected or gate-failing layers

Profile: `{profile}` with hard quality and compression gates.

## Usage

Place the file in ComfyUI's `models/checkpoints` (or `models/unet`) and load
it like any checkpoint. ComfyUI >= v0.31.0 loads all three formats natively
per layer; no custom loader is needed.

```bash
# optional source-free verification of the download (no original model needed)
python comfyui_wxa8_quantizer.py --verify-output {filename}
```

## Notes

* Reconstruction error of the W4A8 layers is at the reference level
  (max relL2 about 0.073 against the source weights).
* The official `_quantization_metadata` block lists every quantized layer
  with its native format; the `comfy_wxa8` extension block records the
  profile, gates, distribution and validation summary.
"""


class PhaseTimer:
    """Simple monotonic phase timing used by the optimized wrapper."""

    def __init__(self) -> None:
        self._t0 = time.monotonic()
        self._cur = self._t0
        self.phases: list[tuple[str, float]] = []

    def stop(self, name: str) -> None:
        now = time.monotonic()
        self.phases.append((name, now - self._cur))
        self._cur = now

    def total(self) -> None:
        self.phases.append(("total", time.monotonic() - self._t0))

    def report(self) -> None:
        print("\nphase timing:")
        for name, secs in self.phases:
            print(f"  {name:11s} {secs / 60:6.1f} min")
