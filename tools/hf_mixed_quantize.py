#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert a Hugging Face model into a ComfyUI mixed-precision checkpoint
optimized for NVIDIA CUDA inference, then upload it to a new HF repo.

Designed to run in a Google Colab notebook (T4 GPU) for the conversion step.
The resulting checkpoint is planned for the user's own NVIDIA PC; the
converter is invoked with --target-runtime nvidia so the per-layer plan
(W4A4 / W4A8 / INT8 / BF16) is built for CUDA inference, and with
--strip-gpu-identity so no GPU name, compute capability or ROCm identity is
written into the checkpoint metadata. The uploaded repo gets a generic
README that never mentions any GPU.

All shared logic (token resolution, download with preflight, converter
fetch with sha256 verification, privacy check, upload) lives in
tools/hf_common.py; this reference variant keeps its original CLI and its
always-on --validate behavior, and is a thin wrapper around the shared code.

Flow:
  1. resolve the HF token (HF_TOKEN env var, Colab secrets, or prompt)
  2. disk preflight: query the expected size from the HF API and check free
     space BEFORE the download
  3. download the model (single file via --hf-filename, or full snapshot)
  4. fetch comfyui_wxa8_quantizer.py from GitHub at a commit-pinned URL and
     verify its sha256 (or use a verified local copy; a local copy whose
     sha256 does not match is refused)
  5. run the converter in a credential-scrubbed environment: --format mixed
     --profile balanced --device cuda --target-runtime nvidia
     --strip-gpu-identity --validate (always on in this variant)
  6. verify the output metadata contains no GPU identity (fail closed)
  7. create a new repo in the user's HF account (private by default) and
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
import os
import sys

from hf_common import (fetch_converter, prepare_source, resolve_token,
                       run_converter, upload, verify_no_gpu_identity)


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
    else:
        token = None
    model_path, _source_bytes = prepare_source(args, token)

    converter = fetch_converter(
        args.converter or os.path.join(workdir, "comfyui_wxa8_quantizer.py"))

    cmd = [sys.executable, converter, model_path, "--output", args.output,
           "--format", args.format, "--profile", args.profile,
           "--target-runtime", args.target_runtime,
           "--strip-gpu-identity", "--validate", "--yes",
           "--max-memory", args.max_memory]
    if args.device:
        cmd += ["--device", args.device]
    run_converter(cmd)

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
