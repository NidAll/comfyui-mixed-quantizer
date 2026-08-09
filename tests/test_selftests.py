"""Embedded self-test suite exposed as pytest cases.

Each embedded self-test runs in its own pytest item so results are reported
explicitly as PASS (passed), FAIL (raised), or SKIP (pytest.skip inside the
test). This mirrors the converter's --self-test CLI.
"""
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from comfyui_wxa8_quantizer.selftests import SELF_TEST_CASES


@pytest.mark.parametrize("name,fn", SELF_TEST_CASES, ids=[n for n, _ in SELF_TEST_CASES])
def test_embedded_selftest(name: str, fn):
    detail = fn()
    assert detail is not None  # a passing test returns a summary string
