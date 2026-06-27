"""End-to-end integration test for square.avl (analytical square wing)."""

from __future__ import annotations

import pytest

from openavl.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR

SQUARE_AVL = GEOMETRIES_DIR / "square.avl"
TOL_CL = 5e-3
TOL_ALPHA = 0.05
TOL_SLOPE_REL = 2e-2

REF_CL_AT_5_73 = 0.145103
REF_CL_ALPHA_SLOPE = 1.460227


def build_square_solver(alpha_deg: float) -> AVLSolver:
    """Build square solver with alpha constrained directly to alpha_deg."""
    if not SQUARE_AVL.is_file():
        pytest.skip(f"square.avl not found: {SQUARE_AVL}")

    solver = AVLSolver(SQUARE_AVL)
    solver.set_constraint("alpha", "alpha", alpha_deg)
    return solver


pytestmark = pytest.mark.integration


@pytest.mark.reference
def test_square_alpha_direct():
    """Alpha fixed at 5.73 deg; CL matches AVL square.run reference."""
    solver = build_square_solver(5.72958)
    solver.execute_run(max_iter=20)
    results = solver.get_results()

    assert results["converged"]
    assert results["alpha_deg"] == pytest.approx(5.72958, abs=TOL_ALPHA)
    assert results["CL"] == pytest.approx(REF_CL_AT_5_73, abs=TOL_CL)


@pytest.mark.reference
def test_square_cl_alpha_slope():
    """CL/alpha slope matches analytical VLM value within 2%."""
    alpha_deg = 1.0
    solver = build_square_solver(alpha_deg)
    solver.execute_run(max_iter=20)
    results = solver.get_results()

    assert results["converged"]
    alpha_rad = alpha_deg * solver.state.dtr
    slope = results["CL"] / alpha_rad
    assert slope == pytest.approx(REF_CL_ALPHA_SLOPE, rel=TOL_SLOPE_REL)
