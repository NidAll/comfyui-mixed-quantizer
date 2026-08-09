"""CLI-level tests: version, help, verify-output."""
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")


def run_cli(*argv: str, timeout: int = 600):
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    # cwd=SRC so the repo-root single-file artifact cannot shadow the package
    return subprocess.run(
        [sys.executable, "-c",
         "import sys; from comfyui_wxa8_quantizer.cli import main; sys.exit(main())",
         *argv],
        capture_output=True, text=True, timeout=timeout, env=env, cwd=SRC)


def test_version_flag():
    r = run_cli("--version")
    assert r.returncode == 0
    assert "comfyui_wxa8_quantizer.py" in r.stdout
    assert "1.4.0" in r.stdout


def test_help():
    r = run_cli("--help")
    assert r.returncode == 0
    for flag in ("--verify-output", "--validation-only", "--format", "--profile"):
        assert flag in r.stdout


def test_list_architectures():
    r = run_cli("--list-architectures")
    assert r.returncode == 0
    assert "wan" in r.stdout


def test_verify_output_missing_file():
    r = run_cli("--verify-output", "/nonexistent/out.safetensors")
    assert r.returncode != 0
    assert "not found" in (r.stdout + r.stderr)
