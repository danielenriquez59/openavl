"""Tests for 3D viewer overlay exports in the web session."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from openavl.core.solver import AVLSolver
from openavl.web.session import _build_lift_distribution_3d, _build_trefftz_data, _build_wake_3d

from tests.helpers import GEOMETRIES_DIR

SUPRA_AVL = GEOMETRIES_DIR / "supra.avl"
SUPRA_MASS = GEOMETRIES_DIR.parent / "mass" / "supra.mass"

pytestmark = pytest.mark.ui


@pytest.mark.skipif(not SUPRA_AVL.is_file(), reason="supra.avl not found")
def test_build_lift_distribution_3d_after_solve():
    """Solved supra geometry exports strip loading points for the web viewer."""
    kwargs = {"xcg": 3.75, "base_dir": SUPRA_AVL.parent}
    if SUPRA_MASS.is_file():
        kwargs["mass_file"] = SUPRA_MASS

    solver = AVLSolver(str(SUPRA_AVL), **kwargs)
    solver.execute_run(max_iter=20)

    lift = _build_lift_distribution_3d(solver)
    assert lift["surfaces"], "expected at least one lifting surface"
    first = lift["surfaces"][0]
    assert first["strips"], "expected strip data on first surface"
    strip = first["strips"][0]
    assert "ensy" in strip and "ensz" in strip
    assert strip["points"], "expected vortex collocation points"
    point = strip["points"][0]
    assert all(key in point for key in ("x", "y", "z", "cl"))


@pytest.mark.skipif(not SUPRA_AVL.is_file(), reason="supra.avl not found")
def test_build_wake_3d_exports_downstream_filaments():
    """Wake export extends each active strip trailing edge by 1.5 spans."""
    solver = AVLSolver(str(SUPRA_AVL), base_dir=SUPRA_AVL.parent)

    wake = _build_wake_3d(solver)
    assert wake["surfaces"], "expected at least one waking surface"
    filament = wake["surfaces"][0]["filaments"][0]
    start = (filament["x0"], filament["y0"], filament["z0"])
    end = (filament["x1"], filament["y1"], filament["z1"])
    assert math.dist(start, end) == pytest.approx(1.5 * solver.state.bref)


@pytest.mark.skipif(not SUPRA_AVL.is_file(), reason="supra.avl not found")
def test_build_trefftz_data_includes_lift_3d_and_cg():
    """Trefftz payload includes 3D viewer overlay data and CG coordinates."""
    solver = AVLSolver(str(SUPRA_AVL), xcg=3.75, base_dir=SUPRA_AVL.parent)
    solver.execute_run(max_iter=20)

    payload = _build_trefftz_data(solver)
    assert "lift_3d" in payload
    assert payload["lift_3d"]["surfaces"]
    assert "wake_3d" in payload
    assert payload["wake_3d"]["surfaces"]
    assert "cg" in payload
    assert payload["cg"]["x"] == pytest.approx(3.75)
