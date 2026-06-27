"""Tests for hinge-moment payload formatting in the web session."""

from __future__ import annotations

import pytest

from openavl import constants as C
from openavl.solver import AVLSolver
from openavl.web.session import (
    _build_hinge_moments,
    _hinge_moment_dimensional,
    _hinge_moment_physical,
)

from tests.helpers import GEOMETRIES_DIR

PLANE_AVL = GEOMETRIES_DIR / "plane.avl"
SUPRA_AVL = GEOMETRIES_DIR / "supra.avl"
SUPRA_MASS = GEOMETRIES_DIR.parent / "mass" / "supra.mass"

pytestmark = pytest.mark.ui


@pytest.mark.skipif(not PLANE_AVL.is_file(), reason="plane.avl not found")
def test_hinge_moment_nondimensional_case_omits_physical_moment():
    """AVL default rho=1, V=1 is treated as coefficient-only (no moment column)."""
    solver = AVLSolver(PLANE_AVL)
    solver.set_parameter("density", 1.0)
    solver.set_parameter("velocity", 1.0)
    solver.execute_run()

    assert _hinge_moment_dimensional(solver.state) is False
    payload = _build_hinge_moments(solver)
    assert payload["dimensional"] is False
    assert payload["moment_units"] is None
    assert payload["controls"]
    assert all("moment" not in row for row in payload["controls"])


@pytest.mark.skipif(not PLANE_AVL.is_file(), reason="plane.avl not found")
def test_hinge_moment_dimensional_conversion():
    """Physical hinge moment follows M = Chinge * q * Sref_d * Cref_d."""
    solver = AVLSolver(PLANE_AVL)
    solver.set_parameter("density", 1.2)
    solver.set_parameter("velocity", 10.0)
    solver.execute_run()

    state = solver.state
    assert _hinge_moment_dimensional(state) is True

    payload = _build_hinge_moments(solver)
    assert payload["dimensional"] is True
    assert payload["moment_units"] == "force*length"

    q = 0.5 * 1.2 * 10.0**2
    unitl = float(state.unitl)
    sref_d = float(state.sref) * unitl * unitl
    cref_d = float(state.cref) * unitl
    for row in payload["controls"]:
        expected = _hinge_moment_physical(row["Chinge"], state)
        assert row["moment"] == pytest.approx(expected)
        assert row["moment"] == pytest.approx(row["Chinge"] * q * sref_d * cref_d)


@pytest.mark.skipif(
    not SUPRA_AVL.is_file() or not SUPRA_MASS.is_file(),
    reason="supra example files not found",
)
def test_hinge_moment_scales_geometry_units_from_mass_file():
    """Inch-based Supra geometry with Lunit=0.0254 m yields SI-order moments."""
    solver = AVLSolver(SUPRA_AVL, mass_file=SUPRA_MASS)
    solver.execute_run(max_iter=20)

    state = solver.state
    assert state.unitl == pytest.approx(0.0254)

    payload = _build_hinge_moments(solver)
    flap = next(row for row in payload["controls"] if row["name"] == "flap")
    assert abs(flap["moment"]) < 1.0
    assert flap["moment"] == pytest.approx(_hinge_moment_physical(flap["Chinge"], state))
