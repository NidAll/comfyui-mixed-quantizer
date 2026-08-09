#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
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

All shared logic (token resolution, download with preflight, converter
fetch with sha256 verification, privacy and sanity checks, upload) lives in
tools/hf_common.py; this file is a thin CLI wrapper around it.

Flow:
  1. resolve the HF token (HF_TOKEN env var, Colab secrets, or prompt)
  2. disk preflight: query the expected size from the HF API and check free
     space BEFORE the download
  3. download the model (single file via --hf-filename, or full snapshot)
  4. fetch comfyui_wxa8_quantizer.py from GitHub at a commit-pinned URL and
     verify its sha256 (or use a verified local copy; a local copy whose
     sha256 does not match is refused unless --trust-local-converter is
     given)
  5. run the converter in a credential-scrubbed environment: --format mixed
     --profile balanced --device cuda --target-runtime nvidia
     --strip-gpu-identity --validate (opt-in)
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
    python tools/hf_mixed_quantize_optimized.py \
        --local-model testdata/wan_fixture.safetensors \
        --output /tmp/out.safetensors --no-upload

Optimizations vs tools/hf_mixed_quantize.py (the original is unchanged):
  1. hf_transfer downloader is enabled by default when the optional
     hf_transfer package is already installed (HF_HUB_ENABLE_HF_TRANSFER;
     2-5x faster downloads of single large files). It is never installed
     automatically; without it the plain downloader is used. Disable with
     --no-hf-transfer.
  2. --validate is opt-in and OFF by default. The plan's hard per-layer
     quality gates already run during planning; the converter's --validate
     adds two extra full source reads plus a 9.6 GB .validation copy. The
     optimized flow replaces it with a fast metadata/tensor sanity check and
     recommends a cheap source-free check on your PC afterwards:
       python comfyui_wxa8_quantizer.py --verify-output OUT.safetensors
  3. Disk-space preflight before the download (aborts early instead of
     failing at 90%).
  4. Phase-by-phase timing summary at the end.
"""
from __future__ import annotations

import argparse
import os
import sys

from hf_common import (PhaseTimer, check_disk_space, enable_hf_transfer,
                       fast_sanity_check, fetch_converter, prepare_source,
                       resolve_token, run_converter, source_filename, upload,
                       verify_no_gpu_identity)


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
    ap.add_argument("--trust-local-converter", action="store_true",
                    help="force-use a pre-existing local converter whose "
                         "sha256 does not match the pinned CONVERTER_SHA256 "
                         "(with a warning) instead of aborting")
    args = ap.parse_args()
    timer = PhaseTimer()

    workdir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(workdir, exist_ok=True)
    enable_hf_transfer(not args.no_hf_transfer)

    if args.hf_model or args.hf_url:
        token = resolve_token()
    else:
        token = None
    model_path, source_bytes = prepare_source(args, token)
    if not (args.hf_model or args.hf_url):
        check_disk_space(source_bytes, args.output)
    timer.stop("download")

    converter = fetch_converter(
        args.converter or os.path.join(workdir, "comfyui_wxa8_quantizer.py"),
        trust_local_converter=args.trust_local_converter)

    cmd = [sys.executable, converter, model_path, "--output", args.output,
           "--format", args.format, "--profile", args.profile,
           "--target-runtime", args.target_runtime,
           "--strip-gpu-identity", "--yes",
           "--max-memory", args.max_memory]
    if args.validate:
        cmd.append("--validate")
    if args.device:
        cmd += ["--device", args.device]
    run_converter(cmd)
    timer.stop("conversion")

    verify_no_gpu_identity(args.output)
    if not args.validate:
        fast_sanity_check(args.output)
    timer.stop("verify")

    if not args.no_upload:
        if token is None:
            raise SystemExit("--no-upload required when using --local-model")
        repo_id = upload(args, args.output, token)
        timer.stop("upload")
        print(f"done. repo: https://huggingface.co/{repo_id}")
    else:
        print(f"done (no upload). output: {args.output}")
    timer.total()
    timer.report()
    if not args.validate:
        print("note: full source-relative validation was skipped; "
              "run a source-free check on your PC afterwards:\n"
              "  python comfyui_wxa8_quantizer.py --verify-output "
              f"{os.path.basename(args.output)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
