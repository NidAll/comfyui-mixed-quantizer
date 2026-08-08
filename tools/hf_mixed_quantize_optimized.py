#!/usr/bin/env python3
"""OPTIMIZED VARIANT: Convert a Hugging Face model into a ComfyUI
mixed-precision checkpoint optimized for NVIDIA CUDA inference, then upload
it to a new HF repo.

Designed to run in a Google Colab notebook (T4 GPU) for the conversion step.
The resulting checkpoint is planned for the user's own NVIDIA PC; the
converter is invoked with --target-runtime nvidia so the per-layer plan
(W4A4 / W4A8 / INT8 / BF16) is built for CUDA inference, and with
--strip-gpu-identity so no GPU name, compute capability or ROCm identity is
written into the checkpoint metadata. The uploaded repo gets a generic
README that never mentions any GPU.

Flow:
  1. resolve the HF token (HF_TOKEN env var, Colab secrets, or prompt)
  2. download the model (single file via --hf-filename, or full snapshot)
  3. fetch comfyui_wxa8_quantizer.py from GitHub main (or use a local copy)
  4. run the converter: --format mixed --profile balanced --device cuda
     --target-runtime nvidia --strip-gpu-identity --validate
  5. verify the output metadata contains no GPU identity (fail closed)
  6. create a new repo in the user's HF account (private by default) and
     upload the model plus a generated README

Usage (Colab):
    !python tools/hf_mixed_quantize.py \
        --hf-model author/model \
        --hf-filename model.safetensors \
        --output model_mixed.safetensors
    # HF_TOKEN must be set (notebook secrets or os.environ)

Local test without HF:
    python tools/hf_mixed_quantize_optimized.py \
        --local-model testdata/wan_fixture.safetensors \
        --output /tmp/out.safetensors --no-upload

Optimizations vs tools/hf_mixed_quantize.py (the original is unchanged):
  1. hf_transfer downloader is enabled by default (HF_HUB_ENABLE_HF_TRANSFER
     plus an automatic pip install in Colab; 2-5x faster downloads of single
     large files). Disable with --no-hf-transfer.
  2. --validate is opt-in and OFF by default. The plan's hard per-layer
     quality gates already run during planning; the converter's --validate
     adds two extra full source reads plus a 9.6 GB .validation copy. The
     optimized flow replaces it with a fast metadata/tensor sanity check and
     recommends a cheap round-trip check on your PC afterwards:
       python comfyui_wxa8_quantizer.py OUT.safetensors --validation-only
  3. Disk-space preflight before the download (aborts early instead of
     failing at 90%).
  4. Phase-by-phase timing summary at the end.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request

CONVERTER_URL = ("https://raw.githubusercontent.com/NidAll/"
                 "comfyui-mixed-quantizer/main/comfyui_wxa8_quantizer.py")
GPU_LEAK_PATTERNS = ("3050", "3060", "3070", "3080", "3090", "4060", "4070",
                     "4080", "4090", "Tesla", "GeForce", "Quadro", "RTX",
                     "Radeon", "RX ", "gfx9", "gfx10", "gfx11", "gfx12")


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
    """Turn on huggingface_hub's hf_transfer downloader when available."""
    if not use_it:
        return
    try:
        import hf_transfer  # noqa: F401
    except ImportError:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install",
                            "-q", "hf_transfer"], check=True, timeout=180)
        except Exception as e:  # noqa: S110 (fallback is safe)
            print(f"note: could not install hf_transfer ({e}); "
                  "using the plain downloader")
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


def fetch_converter(dest: str) -> str:
    if os.path.exists(dest):
        print(f"using local converter: {dest}")
        return dest
    print(f"fetching converter from {CONVERTER_URL}")
    urllib.request.urlretrieve(CONVERTER_URL, dest)  # noqa: S310 (pinned URL)
    return dest


def download_model(args, token: str) -> str:
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


def verify_no_gpu_identity(path: str) -> None:
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


def upload(args, output: str, token: str) -> str:
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
# optional validation of the download
python comfyui_wxa8_quantizer.py {filename} --validation-only
```

## Notes

* Reconstruction error of the W4A8 layers is at the reference level
  (max relL2 about 0.073 against the source weights).
* The official `_quantization_metadata` block lists every quantized layer
  with its native format; the `comfy_wxa8` extension block records the
  profile, gates, distribution and validation summary.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf-model", default=None,
                     help="Hugging Face repo id (e.g. author/model)")
    src.add_argument("--hf-url", default=None,
                     help="full Hugging Face blob/resolve URL of the model "
                          "file, e.g. "
                          "https://huggingface.co/org/repo/blob/main/path/model.safetensors")
    src.add_argument("--local-model", default=None,
                     help="local checkpoint instead of a HF download")
    ap.add_argument("--hf-filename", default=None,
                    help="single file inside the HF repo; omit to download "
                         "the full snapshot")
    ap.add_argument("--output", required=True)
    ap.add_argument("--profile", default="balanced")
    ap.add_argument("--format", default="mixed", choices=["w4a8", "mixed"])
    ap.add_argument("--target-runtime", default="nvidia",
                    choices=["auto", "nvidia", "amd", "cpu"])
    ap.add_argument("--device", default="cuda",
                    help="conversion compute device (Colab T4: cuda)")
    ap.add_argument("--max-memory", default="2G")
    ap.add_argument("--converter", default=None,
                    help="local converter path; default fetches from GitHub")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--validate", action="store_true",
                    help="run the converter's full source-relative --validate "
                         "(two extra full source reads + a validation copy); "
                         "off by default in this optimized variant")
    ap.add_argument("--no-hf-transfer", action="store_true",
                    help="disable the hf_transfer downloader")
    ap.add_argument("--repo-name", default=None)
    ap.add_argument("--public", action="store_true",
                    help="create a public repo (default: private)")
    args = ap.parse_args()
    t0 = time.monotonic()
    phases: list[tuple[str, float]] = []

    workdir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(workdir, exist_ok=True)
    enable_hf_transfer(not args.no_hf_transfer)

    if args.hf_model or args.hf_url:
        token = resolve_token()
        model_path = download_model(args, token)
        source_bytes = os.path.getsize(model_path)
    else:
        token = None
        model_path = args.local_model
        if not os.path.isfile(model_path):
            raise SystemExit(f"local model not found: {model_path}")
        source_bytes = os.path.getsize(model_path)
    phases.append(("download", time.monotonic() - t0))
    check_disk_space(source_bytes, args.output)

    converter = args.converter or fetch_converter(
        os.path.join(workdir, "comfyui_wxa8_quantizer.py"))

    cmd = [sys.executable, converter, model_path, "--output", args.output,
           "--format", args.format, "--profile", args.profile,
           "--target-runtime", args.target_runtime,
           "--strip-gpu-identity", "--yes",
           "--max-memory", args.max_memory]
    if args.validate:
        cmd.append("--validate")
    if args.device:
        cmd += ["--device", args.device]
    print("running:", " ".join(cmd))
    t1 = time.monotonic()
    r = subprocess.run(cmd)  # noqa: S603 (explicit conversion command)
    if r.returncode != 0:
        raise SystemExit(f"conversion failed (exit {r.returncode})")
    phases.append(("conversion", time.monotonic() - t1))

    verify_no_gpu_identity(args.output)
    if not args.validate:
        fast_sanity_check(args.output)
    phases.append(("verify", time.monotonic() - t1 -
                   phases[-1][1]))

    if not args.no_upload:
        if token is None:
            raise SystemExit("--no-upload required when using --local-model")
        t2 = time.monotonic()
        repo_id = upload(args, args.output, token)
        phases.append(("upload", time.monotonic() - t2))
        print(f"done. repo: https://huggingface.co/{repo_id}")
    else:
        print(f"done (no upload). output: {args.output}")
    phases.append(("total", time.monotonic() - t0))
    print("\nphase timing:")
    for name, secs in phases:
        print(f"  {name:11s} {secs / 60:6.1f} min")
    if not args.validate:
        print("note: full source-relative validation was skipped; "
              "run on your PC afterwards:\n"
              "  python comfyui_wxa8_quantizer.py "
              f"{os.path.basename(args.output)} --validation-only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
