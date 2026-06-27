"""Tests for openavl.airfoil."""

from __future__ import annotations

import numpy as np
import pytest

from openavl.airfoil import build_camber_slope, build_naca_slope, getcam
from tests.helpers import load_json_fixture, run_ref_binary

pytestmark = pytest.mark.core


@pytest.mark.fixture
def test_getcam_matches_js_fixture(fixtures_dir):
    """GETCAM matches JS/Fortran reference fixture."""
    data = load_json_fixture(fixtures_dir, "airfoil_cases.json")
    x = np.array(data["x"], dtype=np.float64)
    y = np.array(data["y"], dtype=np.float64)
    xc, yc, tc = getcam(x, y, data["nc"], normalize=data["normalize"])
    actual = [*xc.tolist(), *yc.tolist(), *tc.tolist()]
    np.testing.assert_allclose(actual, data["expected"], rtol=0, atol=1e-5)


@pytest.mark.reference
def test_getcam_matches_fortran_ref(ref_binary, fixtures_dir):
    """GETCAM matches Fortran airutil_ref when available."""
    ref_path = ref_binary("airutil_ref")
    if ref_path is None:
        pytest.skip("Fortran reference binary not found")

    data = load_json_fixture(fixtures_dir, "airfoil_cases.json")
    ref = run_ref_binary(ref_path)
    x = np.array(data["x"], dtype=np.float64)
    y = np.array(data["y"], dtype=np.float64)
    xc, yc, tc = getcam(x, y, data["nc"], normalize=data["normalize"])
    actual = [*xc.tolist(), *yc.tolist(), *tc.tolist()]
    np.testing.assert_allclose(actual, ref[: len(actual)], rtol=0, atol=1e-5)


def test_build_naca_flat_plate():
    """NACA 0000 produces zero camber slope and thickness."""
    cam = build_naca_slope("0000", samples=10)
    np.testing.assert_allclose(cam.s, 0.0, atol=1e-12)
    np.testing.assert_allclose(cam.c, 0.0, atol=1e-12)
    np.testing.assert_allclose(cam.t, 0.0, atol=1e-12)


def test_build_naca_0012_has_thickness():
    """NACA 0012 produces non-zero thickness for CPOML."""
    cam = build_naca_slope("0012", samples=20)
    assert float(np.max(cam.t)) > 0.05
    np.testing.assert_allclose(cam.c, 0.0, atol=1e-12)


@pytest.mark.fixture
def test_build_camber_slope_from_coords(fixtures_dir):
    """build_camber_slope returns normalized slope tables from coordinates."""
    data = load_json_fixture(fixtures_dir, "airfoil_cases.json")
    coords = [[data["x"][i], data["y"][i]] for i in range(len(data["x"]))]
    samples = 10
    cam = build_camber_slope(coords, samples=samples)
    assert cam is not None
    expected = min(samples, len(coords))
    assert cam.x.size == expected
    assert cam.s.size == expected
    assert cam.c.size == expected
    assert cam.t.size == expected
    assert cam.x[0] == pytest.approx(0.0, abs=1e-6)
    assert cam.x[-1] == pytest.approx(1.0, abs=1e-6)
