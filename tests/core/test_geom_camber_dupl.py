"""Regression tests for MAKESURF SLOPEV (GEO-02) and SDUPL AINC1/AINC2 (GEO-14)."""

from __future__ import annotations

import numpy as np
import pytest

from openavl.geom import build_geometry
from openavl.parser import parse_avl_file, prepare_model
from openavl.state import AVLState

pytestmark = pytest.mark.core

NACA_M = 0.02
NACA_P = 0.4

WING_AVL = """Test Wing
0.0
0 0 0.0
4.0 1.0 4.0
0.25 0.0 0.0
SURFACE
Wing
10 1.0 6 1.0
YDUPLICATE
0.0
SECTION
0.0 0.0 0.0 1.0 {ainc_root} 0 0
NACA
2412
SECTION
0.0 2.0 0.0 1.0 {ainc_tip} 0 0
NACA
2412
"""


def _build(tmp_path, ainc_root=0.0, ainc_tip=0.0):
    path = tmp_path / "wing.avl"
    path.write_text(WING_AVL.format(ainc_root=ainc_root, ainc_tip=ainc_tip), encoding="utf-8")
    model = prepare_model(parse_avl_file(path), base_dir=tmp_path)
    state = AVLState.from_model(model)
    build_geometry(state, model)
    return state


def _naca_slope(x):
    """Analytic NACA 4-digit mean-line slope dz/dx."""
    x = np.asarray(x, dtype=np.float64)
    fwd = 2.0 * NACA_M * (NACA_P - x) / NACA_P**2
    aft = 2.0 * NACA_M * (NACA_P - x) / (1.0 - NACA_P) ** 2
    return np.where(x < NACA_P, fwd, aft)


def _naca_curvature(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x < NACA_P, -2.0 * NACA_M / NACA_P**2, -2.0 * NACA_M / (1.0 - NACA_P) ** 2)


def test_slopev_is_camber_slope_at_vortex(tmp_path):
    """SLOPEV is the interpolated camber slope at xvr (amake.f), not dS/dx."""
    state = _build(tmp_path)
    nv = state.nvor
    assert nv > 0
    strip = np.repeat(np.arange(state.nstrip), state.nvstrp[: state.nstrip])[:nv]
    chord = state.chord[strip]
    xvr = (state.rv[0, :nv] - state.rle[0, strip]) / chord
    xcp = (state.rc[0, :nv] - state.rle[0, strip]) / chord

    slopev = state.slopev[:nv]
    slopec = state.slopec[:nv]
    # Table is 61 uniform points of a piecewise-linear slope; Akima is exact
    # away from the kink at x=p and within a few 1e-4 next to it.
    np.testing.assert_allclose(slopev, _naca_slope(xvr), rtol=0, atol=2e-3)
    np.testing.assert_allclose(slopec, _naca_slope(xcp), rtol=0, atol=2e-3)
    far = np.abs(xvr - NACA_P) > 0.05
    np.testing.assert_allclose(slopev[far], _naca_slope(xvr[far]), rtol=0, atol=1e-10)
    # Not the curvature (the pre-fix behaviour).
    assert np.max(np.abs(slopev - _naca_curvature(xvr))) > 0.05


def test_sdupl_swaps_ainc_edges_on_image(tmp_path):
    """Image strips take AINC1 <- AINC2 and AINC2 <- AINC1 (amake.f SDUPL)."""
    state = _build(tmp_path, ainc_root=0.0, ainc_tip=5.0)
    assert state.nsurf == 2
    nj = int(state.nj[0])
    assert int(state.nj[1]) == nj
    j0 = int(state.jfrst[0])
    j1 = int(state.jfrst[1])
    base1 = state.ainc1[j0 : j0 + nj]
    base2 = state.ainc2[j0 : j0 + nj]
    img1 = state.ainc1[j1 : j1 + nj]
    img2 = state.ainc2[j1 : j1 + nj]
    assert np.all(np.abs(base2 - base1) > 0.0)  # twist present
    np.testing.assert_array_equal(img1, base2)
    np.testing.assert_array_equal(img2, base1)
    np.testing.assert_array_equal(state.ainc[j1 : j1 + nj], state.ainc[j0 : j0 + nj])
    # Edge 1 of each image strip is the mirror of edge 2 of its base strip,
    # so the incidence carried with it must match.
    np.testing.assert_allclose(state.rle1[1, j1 : j1 + nj], -state.rle2[1, j0 : j0 + nj])
