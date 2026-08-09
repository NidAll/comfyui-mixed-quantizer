#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Certified conversion orchestrator (staging/publishing workflow).

Keeps the standalone converter free of ComfyUI/comfy-kitchen imports while
still offering the strongest publish guarantee:

    1. run tools/runtime_certify.py                 -> cert.json (when
       --certify is given and --cert-path is not; the certificate MUST exist
       before conversion so --require-runtime-certificate can be honored)
    2. convert with the standalone quantizer        -> MODEL.staged
       (--runtime-certificate cert --require-runtime-certificate)
    3. run testdata/comfyui_smoke.py                -> real load + forward
    4. run testdata/model_quality.py                -> BF16-relative model gate
    5. all pass?  -> atomically rename .staged -> final
       otherwise   -> keep the failure report, never publish

Only steps 1, 2 and 5 run unconditionally (step 1 produces the certificate
only when --certify asks for it). Steps 3-4 run when the flags are given
(they need the target machine, ComfyUI >= v0.31.0 and the original
checkpoint). Without them the tool behaves like the plain converter plus
staging.

The report (--output + .cert-report.json) is written before EVERY early
return, so a failed convert/certify/smoke/quality step still leaves the
failing step's ok/returncode/tail on disk.

Usage:
    python tools/certified_convert.py MODEL.safetensors \
        --output MODEL_certified.safetensors \
        --source MODEL_bf16.safetensors \
        --certify \
        --smoke \
        --quality \
        --quality-threshold 0.05 \
        --comfyui-src /path/to/ComfyUI
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONVERTER = os.path.join(REPO, "comfyui_wxa8_quantizer.py")
CERTIFY = os.path.join(REPO, "tools", "runtime_certify.py")
SMOKE = os.path.join(REPO, "testdata", "comfyui_smoke.py")
QUALITY = os.path.join(REPO, "testdata", "model_quality.py")


def run(cmd: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    return subprocess.run(  # noqa: S603 (explicit CLI invocation)
        cmd, capture_output=True, text=True, env=full_env, timeout=7200)


def write_report(report: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="original checkpoint")
    ap.add_argument("--output", required=True)
    ap.add_argument("--source", default=None,
                    help="original BF16 checkpoint (needed for --quality)")
    ap.add_argument("--format", default="mixed", choices=["w4a8", "mixed"])
    ap.add_argument("--profile", default="balanced")
    ap.add_argument("--target-runtime", default="auto",
                    choices=["auto", "nvidia", "amd", "cpu"])
    ap.add_argument("--certify", action="store_true",
                    help="run tools/runtime_certify.py and require the cert")
    ap.add_argument("--smoke", action="store_true",
                    help="run testdata/comfyui_smoke.py on the staged output")
    ap.add_argument("--quality", action="store_true",
                    help="run testdata/model_quality.py on the staged output")
    ap.add_argument("--quality-threshold", type=float, default=0.05)
    ap.add_argument("--comfyui-src", default=None,
                    help="ComfyUI checkout for the smoke/quality steps")
    ap.add_argument("--cert-path", default=None, metavar="PATH")
    ap.add_argument("--require-formats", action="append", default=[],
                    help="--require-format values for the smoke step")
    args = ap.parse_args()

    staged = args.output + ".staged"
    report_path = args.output + ".cert-report.json"
    report: dict = {"steps": {}}

    def fail(step: str, rc: int, tail: str, msg: str) -> int:
        report["steps"][step] = {
            "ok": False, "returncode": rc, "tail": tail[-800:]}
        write_report(report, report_path)
        print(msg)
        print(tail[-800:])
        print(f"certificate report: {report_path}")
        return 1

    # 1) runtime certificate FIRST (target machine). The converter is
    # invoked with --require-runtime-certificate below, so the certificate
    # must exist before conversion starts; generating it after the convert
    # step (the historical order) made --certify fail every time.
    cert_path = args.cert_path
    if args.certify and cert_path is None:
        cert_path = args.output + ".runtime-cert.json"
        r = run([sys.executable, CERTIFY, "--output", cert_path])
        report["steps"]["certify"] = {
            "ok": r.returncode == 0, "returncode": r.returncode,
            "path": cert_path, "tail": (r.stdout + r.stderr)[-600:]}
        if r.returncode != 0:
            return fail("certify", r.returncode,
                        r.stdout + r.stderr,
                        "runtime certification FAILED; nothing published")

    # 2) convert to the staged path, requiring the certificate
    cmd = [sys.executable, CONVERTER, args.model, "--output", staged,
           "--format", args.format, "--profile", args.profile,
           "--target-runtime", args.target_runtime]
    if cert_path:
        cmd += ["--runtime-certificate", cert_path,
                "--require-runtime-certificate"]
    r = run(cmd)
    report["steps"]["convert"] = {
        "ok": r.returncode == 0, "returncode": r.returncode,
        "tail": (r.stdout + r.stderr)[-800:]}
    if r.returncode != 0:
        return fail("convert", r.returncode, r.stdout + r.stderr,
                    "convert FAILED; nothing published")

    env = None
    if args.comfyui_src:
        env = {"PYTHONPATH": args.comfyui_src}

    # 3) ComfyUI smoke on the staged output
    if args.smoke:
        cmd = [sys.executable, SMOKE, "--model", staged]
        for fmt in args.require_formats:
            cmd += ["--require-format", fmt]
        r = run(cmd, env=env)
        report["steps"]["smoke"] = {
            "ok": r.returncode == 0, "returncode": r.returncode,
            "tail": (r.stdout + r.stderr)[-800:]}
        if r.returncode != 0:
            return fail("smoke", r.returncode, r.stdout + r.stderr,
                        "ComfyUI smoke FAILED; nothing published")

    # 4) model-level BF16-relative quality gate
    if args.quality:
        if not args.source:
            return fail(
                "quality", 2,
                "--quality needs --source (the original BF16 checkpoint)",
                "model quality gate FAILED; nothing published")
        cmd = [sys.executable, QUALITY, "--model", staged,
               "--source", args.source,
               "--threshold", str(args.quality_threshold)]
        r = run(cmd, env=env)
        report["steps"]["quality"] = {
            "ok": r.returncode == 0, "returncode": r.returncode,
            "tail": (r.stdout + r.stderr)[-800:]}
        if r.returncode != 0:
            return fail("quality", r.returncode, r.stdout + r.stderr,
                        "model quality gate FAILED; nothing published")

    # 5) publish atomically
    os.replace(staged, args.output)
    report["published"] = args.output
    write_report(report, report_path)
    print(f"certified conversion published: {args.output}")
    print(f"certificate report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
