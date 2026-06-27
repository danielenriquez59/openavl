"""Tests for openavl.spline."""

from __future__ import annotations

import numpy as np
import pytest

from openavl.spline import deval, seval, spline, splind
from tests.helpers import load_json_fixture, run_ref_binary

pytestmark = pytest.mark.core


@pytest.mark.fixture
def test_spline_matches_js_fixture(fixtures_dir):
    """SPLINE/SPLIND/SEVAL/DEVAL match JS reference fixture."""
    data = load_json_fixture(fixtures_dir, "spline_cases.json")
    n = data["n"]
    s = np.array(data["s"], dtype=np.float64)
    x = np.array(data["x"], dtype=np.float64)
    xs_spline = np.zeros(n, dtype=np.float64)
    xs_splind = np.zeros(n, dtype=np.float64)

    spline(x, xs_spline, s, n)
    splind(x, xs_splind, s, n, 999.0, -999.0)

    np.testing.assert_allclose(xs_spline, data["xs_spline"], rtol=0, atol=1e-5)
    np.testing.assert_allclose(xs_splind, data["xs_splind"], rtol=0, atol=1e-5)
    assert seval(data["ss"], x, xs_spline, s, n) == pytest.approx(data["seval"], abs=1e-5)
    assert deval(data["ss"], x, xs_spline, s, n) == pytest.approx(data["deval"], abs=1e-5)


@pytest.mark.reference
def test_spline_matches_fortran_ref(ref_binary, fixtures_dir):
    """Spline routines match Fortran spline_ref when available."""
    ref_path = ref_binary("spline_ref")
    if ref_path is None:
        pytest.skip("Fortran reference binary not found")

    data = load_json_fixture(fixtures_dir, "spline_cases.json")
    ref = run_ref_binary(ref_path)

    n = data["n"]
    s = np.array(data["s"], dtype=np.float64)
    x = np.array(data["x"], dtype=np.float64)
    xs_spline = np.zeros(n, dtype=np.float64)
    xs_splind = np.zeros(n, dtype=np.float64)
    spline(x, xs_spline, s, n)
    splind(x, xs_splind, s, n, 999.0, -999.0)

    out = [
        *xs_spline.tolist(),
        *xs_splind.tolist(),
        seval(data["ss"], x, xs_spline, s, n),
        deval(data["ss"], x, xs_spline, s, n),
    ]
    np.testing.assert_allclose(out, ref[: len(out)], rtol=0, atol=1e-5)
