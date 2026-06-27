"""Tests for 3D lift distribution export in the web session."""

from __future__ import annotations

from pathlib import Path

import pytest

from openavl.core.solver import AVLSolver
from openavl.web.session import _build_lift_distribution_3d, _build_trefftz_data

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
def test_build_trefftz_data_includes_lift_3d_and_cg():
    """Trefftz payload includes 3D lift overlay data and CG coordinates."""
    solver = AVLSolver(str(SUPRA_AVL), xcg=3.75, base_dir=SUPRA_AVL.parent)
    solver.execute_run(max_iter=20)

    payload = _build_trefftz_data(solver)
    assert "lift_3d" in payload
    assert payload["lift_3d"]["surfaces"]
    assert "cg" in payload
    assert payload["cg"]["x"] == pytest.approx(3.75)
