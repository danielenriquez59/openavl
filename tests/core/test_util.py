"""Tests for openavl.util."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openavl.util import m3inv, rateki3, rotens3
from tests.helpers import REF_DIR

pytestmark = pytest.mark.core

M3INV_MATRICES = [
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
    np.array([[2, 0, 0], [0, 3, 0], [0, 0, 4]], dtype=np.float64),
    np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]], dtype=np.float64),
    np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64),
]

ANGLE_CASES = [
    np.array([0.1, 0.2, -0.3], dtype=np.float64),
    np.array([0.5, -0.4, 0.25], dtype=np.float64),
    np.array([-1.2, 0.9, 0.0], dtype=np.float64),
    np.array([0.0, 0.0, 0.0], dtype=np.float64),
]


def _run_fortran_ref(bin_name: str, cases: list[np.ndarray], values_per_case: int) -> list[np.ndarray]:
    from tests.helpers import run_ref_binary

    ref_path = REF_DIR / bin_name
    if not (ref_path.is_file() or (REF_DIR / f"{bin_name}.exe").is_file()):
        pytest.skip(f"Fortran reference binary not found: {bin_name}")
    lines = [str(len(cases))]
    for vals in cases:
        lines.append(" ".join(str(float(v)) for v in vals.reshape(-1)))
    values = run_ref_binary(ref_path if ref_path.is_file() else REF_DIR / f"{bin_name}.exe", "\n".join(lines))
    out: list[np.ndarray] = []
    for i in range(len(cases)):
        out.append(np.array(values[i * values_per_case : i * values_per_case + values_per_case], dtype=np.float64))
    return out


def assert_close_array(actual, expected, tol: float = 1e-5) -> None:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape
    diff = np.abs(actual - expected)
    assert np.all(diff <= tol), f"max diff {diff.max()} > {tol}"


def test_cross3_dot3():
    u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    v = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    from openavl.util import cross3, dot3

    w = cross3(u, v)
    assert_close_array(w, [0.0, 0.0, 1.0])
    assert float(dot3(u, v)) == pytest.approx(0.0)


@pytest.mark.reference
def test_m3inv_matches_fortran_ref():
    ref = _run_fortran_ref("m3inv_ref", M3INV_MATRICES, 9)
    for mat, expected in zip(M3INV_MATRICES, ref, strict=True):
        out = m3inv(mat)
        assert_close_array(out.reshape(-1), expected)


@pytest.mark.reference
def test_rateki3_matches_fortran_ref():
    ref = _run_fortran_ref("rateki3_ref", ANGLE_CASES, 36)
    for angles, expected in zip(ANGLE_CASES, ref, strict=True):
        r, r_a = rateki3(angles)
        assert_close_array(r.reshape(-1), expected[:9])
        assert_close_array(r_a.reshape(-1), expected[9:])


@pytest.mark.reference
def test_rotens3_matches_fortran_ref():
    ref = _run_fortran_ref("rotens3_ref", ANGLE_CASES, 36)
    for angles, expected in zip(ANGLE_CASES, ref, strict=True):
        t, t_a = rotens3(angles)
        assert_close_array(t.reshape(-1), expected[:9])
        assert_close_array(t_a.reshape(-1), expected[9:])
