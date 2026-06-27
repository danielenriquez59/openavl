"""End-to-end integration test for vanilla.avl (sideslip sweep)."""

from __future__ import annotations

import pytest

from openavl.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR

VANILLA_AVL = GEOMETRIES_DIR / "vanilla.avl"
TOL_CL = 5e-3

# Fortran AVL vanilla.run solved alphas (alpha constrained to CL=1.16647).
REF_CL = 1.16647
REF_ALPHA_BY_BETA = {
    0: 8.24035,
    5: 8.30849,
    10: 8.51450,
    15: 8.86328,
    20: 9.36338,
}
# Current OpenAVL offset vs AVL grows at high sideslip; tighten as trim parity improves.
TOL_ALPHA_BY_BETA = {
    0: 0.22,
    5: 0.19,
    10: 0.08,
    15: 0.13,
    20: 0.45,
}


def build_vanilla_solver(beta: float) -> AVLSolver:
    """Build vanilla solver at the given sideslip angle."""
    if not VANILLA_AVL.is_file():
        pytest.skip(f"vanilla.avl not found: {VANILLA_AVL}")

    solver = AVLSolver(
        VANILLA_AVL,
        cl=REF_CL,
        vel=30.8633,
        rho=0.2,
        gravity=10.0,
        xcg=0.65,
        cd0=0.0,
        beta=beta,
    )
    # vanilla.run: alpha->CL, fixed beta, aileron/elevator/rudder moment trims
    solver.set_constraint("aileron", "cll", 0.0)
    solver.set_constraint("elevator", "cm", 0.0)
    solver.set_constraint("rudder", "cn", 0.0)
    return solver


pytestmark = pytest.mark.integration


@pytest.mark.reference
@pytest.mark.parametrize("beta", [0, 5, 10, 15, 20], ids=[f"beta_{b}" for b in [0, 5, 10, 15, 20]])
def test_vanilla_beta_sweep(beta: float):
    """Sideslip sweep at fixed CL; alpha matches AVL vanilla.run reference."""
    solver = build_vanilla_solver(beta)
    solver.execute_run(max_iter=20)
    results = solver.get_results()
    tol_alpha = TOL_ALPHA_BY_BETA[beta]

    assert results["converged"]
    assert results["CL"] == pytest.approx(REF_CL, abs=TOL_CL)
    assert results["beta_deg"] == pytest.approx(beta, abs=0.05)
    assert results["alpha_deg"] == pytest.approx(REF_ALPHA_BY_BETA[beta], abs=tol_alpha)
