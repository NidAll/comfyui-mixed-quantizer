#!/usr/bin/env python3
"""Convert a Hugging Face model into a ComfyUI mixed-precision checkpoint
optimized for NVIDIA CUDA inference, then upload it to a new HF repo.

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
    python tools/hf_mixed_quantize.py \
        --local-model testdata/wan_fixture.safetensors \
        --output /tmp/out.safetensors --no-upload
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
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
    ap.add_argument("--repo-name", default=None)
    ap.add_argument("--public", action="store_true",
                    help="create a public repo (default: private)")
    args = ap.parse_args()

    workdir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(workdir, exist_ok=True)

    if args.hf_model or args.hf_url:
        token = resolve_token()
        model_path = download_model(args, token)
    else:
        token = None
        model_path = args.local_model
        if not os.path.isfile(model_path):
            raise SystemExit(f"local model not found: {model_path}")

    converter = args.converter or fetch_converter(
        os.path.join(workdir, "comfyui_wxa8_quantizer.py"))

    cmd = [sys.executable, converter, model_path, "--output", args.output,
           "--format", args.format, "--profile", args.profile,
           "--target-runtime", args.target_runtime,
           "--strip-gpu-identity", "--validate", "--yes",
           "--max-memory", args.max_memory]
    if args.device:
        cmd += ["--device", args.device]
    print("running:", " ".join(cmd))
    r = subprocess.run(cmd)  # noqa: S603 (explicit conversion command)
    if r.returncode != 0:
        raise SystemExit(f"conversion failed (exit {r.returncode})")

    verify_no_gpu_identity(args.output)

    if not args.no_upload:
        if token is None:
            raise SystemExit("--no-upload required when using --local-model")
        repo_id = upload(args, args.output, token)
        print(f"done. repo: https://huggingface.co/{repo_id}")
    else:
        print(f"done (no upload). output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
