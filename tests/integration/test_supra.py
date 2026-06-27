"""End-to-end integration test for supra.avl (CL sweep sailplane)."""

from __future__ import annotations

import pytest

from openavl.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR

SUPRA_AVL = GEOMETRIES_DIR / "supra.avl"
TOL_CL = 5e-3
TOL_ALPHA = 0.08

# Fortran AVL supra.run / supra.runv solved alphas (alpha constrained to CL).
REF_ALPHA_BY_CL = {
    0.7: 3.695,
    0.5: 1.717,
    0.35: 0.233,
    0.2: -1.246,
    0.1: -2.230,
}


def build_supra_solver() -> AVLSolver:
    """Build supra solver with run-case dimensional parameters."""
    if not SUPRA_AVL.is_file():
        pytest.skip(f"supra.avl not found: {SUPRA_AVL}")

    solver = AVLSolver(
        SUPRA_AVL,
        cd0=0.015,
        rho=1.225,
        gravity=9.81,
        xcg=3.75,
    )
    # supra.run: alpha->CL, beta->0, aileron/elevator/rudder moment trims
    solver.set_constraint("beta", "beta", 0.0)
    solver.set_constraint("aileron", "cll", 0.0)
    solver.set_constraint("elevator", "cm", 0.0)
    solver.set_constraint("rudder", "cn", 0.0)
    return solver


pytestmark = pytest.mark.integration

@pytest.mark.reference
def test_supra_alpha_direct():
    """Trim alpha to CL=0.7 with supra.run moment constraints."""
    solver = build_supra_solver()
    solver.set_constraint("alpha", "cl", 0.7)
    solver.execute_run(max_iter=20)
    results = solver.get_results()

    assert results["converged"]
    assert results["CL"] == pytest.approx(0.7, abs=TOL_CL)
    assert results["alpha_deg"] == pytest.approx(REF_ALPHA_BY_CL[0.7], abs=TOL_ALPHA)


@pytest.mark.reference
@pytest.mark.parametrize("cl", [0.5, 0.35, 0.2, 0.1], ids=[f"cl_{c}" for c in [0.5, 0.35, 0.2, 0.1]])
def test_supra_cl_sweep(cl: float):
    """CL sweep with alpha trimmed to target CL and supra.run moment constraints."""
    solver = build_supra_solver()
    solver.set_constraint("alpha", "cl", cl)
    solver.execute_run(max_iter=20)
    results = solver.get_results()

    assert results["converged"]
    assert results["CL"] == pytest.approx(cl, abs=TOL_CL)
    assert results["alpha_deg"] == pytest.approx(REF_ALPHA_BY_CL[cl], abs=TOL_ALPHA)
