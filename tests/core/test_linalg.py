"""Tests for openavl.linalg."""

from __future__ import annotations

import numpy as np
import pytest

from openavl.linalg import baksub, ludcmp, lusolve
from tests.helpers import load_json_fixture, run_ref_binary

pytestmark = pytest.mark.core


def _to_col_major(row_major: list[float], n: int) -> np.ndarray:
    out = np.zeros((n, n), dtype=np.float64, order="F")
    for r in range(n):
        for c in range(n):
            out[r, c] = row_major[r * n + c]
    return out


@pytest.mark.fixture
def test_ludcmp_baksub_matches_js_fixture(fixtures_dir):
    """LUDCMP/BAKSUB match JS reference fixture."""
    data = load_json_fixture(fixtures_dir, "linalg_cases.json")
    for case, expected in zip(data["cases"], data["expected"], strict=True):
        n = case["n"]
        a = _to_col_major(case["A"], n).copy()
        b = np.array(case["B"], dtype=np.float64)
        indx = np.zeros(n, dtype=np.int32)
        work = np.zeros(n, dtype=np.float64)
        ludcmp(a, n, indx, work)
        baksub(a, n, indx, b)
        np.testing.assert_allclose(b, expected, rtol=0, atol=1e-4)


@pytest.mark.fixture
def test_lusolve_matches_fixture(fixtures_dir):
    """SciPy LU solve matches expected solutions."""
    data = load_json_fixture(fixtures_dir, "linalg_cases.json")
    for case, expected in zip(data["cases"], data["expected"], strict=True):
        n = case["n"]
        a = np.array(case["A"], dtype=np.float64).reshape(n, n)
        b = np.array(case["B"], dtype=np.float64)
        x = lusolve(a, b)
        np.testing.assert_allclose(x, expected, rtol=0, atol=1e-4)


@pytest.mark.reference
def test_ludcmp_baksub_matches_fortran_ref(ref_binary, fixtures_dir):
    """LUDCMP/BAKSUB match Fortran matrix_linpack_ref when available."""
    ref_path = ref_binary("matrix_linpack_ref")
    if ref_path is None:
        pytest.skip("Fortran reference binary not found")

    data = load_json_fixture(fixtures_dir, "linalg_cases.json")
    input_lines = [str(len(data["cases"]))]
    for case in data["cases"]:
        input_lines.append(str(case["n"]))
        input_lines.append(" ".join(str(v) for v in case["A"]))
        input_lines.append(" ".join(str(v) for v in case["B"]))
    ref_vals = run_ref_binary(ref_path, "\n".join(input_lines))

    offset = 0
    for case in data["cases"]:
        n = case["n"]
        expected = ref_vals[offset : offset + n]
        offset += n
        a = _to_col_major(case["A"], n).copy()
        b = np.array(case["B"], dtype=np.float64)
        indx = np.zeros(n, dtype=np.int32)
        work = np.zeros(n, dtype=np.float64)
        ludcmp(a, n, indx, work)
        baksub(a, n, indx, b)
        np.testing.assert_allclose(b, expected, rtol=0, atol=1e-4)
