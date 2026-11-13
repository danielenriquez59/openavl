"""Regression tests for YDUPLICATE mirroring in display helpers (GEO-16)."""

from __future__ import annotations

import numpy as np
import pytest

from openavl.geom import build_geometry
from openavl.geom.display import control_hinge_polylines
from openavl.fileio.parser import parse_avl, prepare_model
from openavl.plotting.aircraft3d import _mirror_point
from openavl.core.state import AVLState

pytestmark = pytest.mark.core

XHINGE = 0.7
CHORD = 1.0


def _avl_text(ydup: float) -> str:
    return f"""Mirror test
0.0
0 0 0.0
4.0 1.0 4.0
0.0 0.0 0.0
0.0
SURFACE
Wing
4 1.0 4 1.0
YDUPLICATE
{ydup}
SECTION
0.0 1.0 0.0 {CHORD} 0.0
CONTROL
flap 1.0 {XHINGE} 0.0 0.0 0.0 1.0
SECTION
0.0 3.0 0.0 {CHORD} 0.0
CONTROL
flap 1.0 {XHINGE} 0.0 0.0 0.0 1.0
"""


def _model(ydup: float):
    return prepare_model(parse_avl(_avl_text(ydup)))


@pytest.mark.parametrize("ydup", [0.0, 0.5])
def test_mirrored_hinge_polyline_reflects_about_ydup(ydup):
    polylines = control_hinge_polylines(_model(ydup))
    assert len(polylines) == 2
    (_, _, original), (_, _, mirrored) = polylines
    assert len(original) == len(mirrored)
    for (x, y, z), (xm, ym, zm) in zip(original, mirrored):
        assert xm == pytest.approx(x)
        assert ym == pytest.approx(2.0 * ydup - y)
        assert zm == pytest.approx(z)


@pytest.mark.parametrize("ydup", [0.0, 0.5])
def test_mirrored_hinge_polyline_matches_solver_phinge(ydup):
    model = _model(ydup)
    state = AVLState.from_model(model)
    build_geometry(state, model)

    _, _, mirrored = control_hinge_polylines(model)[1]
    ys = [pt[1] for pt in mirrored]
    ylo, yhi = min(ys), max(ys)
    x_hinge = mirrored[0][0]

    mirror_surf = 1
    strips = [j for j in range(int(state.nstrip)) if int(state.lssurf[j]) == mirror_surf]
    assert strips
    for j in strips:
        px, py, _pz = state.phinge[:, j, 0]
        assert px == pytest.approx(x_hinge)
        assert ylo - 1e-12 <= py <= yhi + 1e-12


@pytest.mark.parametrize("ydup", [0.0, 0.5])
def test_plot_mirror_point_reflects_about_ydup(ydup):
    point = np.array([0.3, 1.25, -0.2])
    mirrored = _mirror_point(point, ydup)
    np.testing.assert_allclose(mirrored, [0.3, 2.0 * ydup - 1.25, -0.2])
