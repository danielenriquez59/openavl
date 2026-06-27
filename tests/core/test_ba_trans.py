"""Tests for openavl.ba_trans."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openavl.ba_trans import ba2sa_mat, ba2wa_mat
from tests.helpers import REF_DIR

pytestmark = pytest.mark.core

CASES = [
    np.array([0.1, 0.2, 1.0], dtype=np.float64),
    np.array([0.0, -0.5, 2.0], dtype=np.float64),
    np.array([1.2, 0.1, 0.8], dtype=np.float64),
    np.array([-0.7, 0.3, 1.5], dtype=np.float64),
]
VALUES_PER_CASE = 45


def _run_fortran_ref(cases: list[np.ndarray]) -> list[np.ndarray]:
    from tests.helpers import run_ref_binary

    ref_path = REF_DIR / "ba_trans_ref"
    if not (ref_path.is_file() or (REF_DIR / "ba_trans_ref.exe").is_file()):
        pytest.skip("Fortran reference binary not found: ba_trans_ref")
    lines = [str(len(cases))]
    for vals in cases:
        lines.append(" ".join(str(float(v)) for v in vals))
    values = run_ref_binary(ref_path if ref_path.is_file() else REF_DIR / "ba_trans_ref.exe", "\n".join(lines))
    return [
        np.array(values[i * VALUES_PER_CASE : i * VALUES_PER_CASE + VALUES_PER_CASE], dtype=np.float64)
        for i in range(len(cases))
    ]


def assert_close_array(actual, expected, tol: float = 1e-5) -> None:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape
    diff = np.abs(actual - expected)
    assert np.all(diff <= tol), f"max diff {diff.max()} > {tol}"


@pytest.mark.reference
def test_ba_trans_matches_fortran_ref():
    ref = _run_fortran_ref(CASES)
    for vals, expected in zip(CASES, ref, strict=True):
        alfa, beta, binv = float(vals[0]), float(vals[1]), float(vals[2])
        p, p_a, p_b = ba2wa_mat(alfa, beta, binv)
        p_sa, p_sa_a = ba2sa_mat(alfa)
        assert_close_array(p.reshape(-1), expected[:9])
        assert_close_array(p_a.reshape(-1), expected[9:18])
        assert_close_array(p_b.reshape(-1), expected[18:27])
        assert_close_array(p_sa.reshape(-1), expected[27:36])
        assert_close_array(p_sa_a.reshape(-1), expected[36:45])
