"""End-to-end integration test for bd.avl (Bubble Dancer banked turns)."""

from __future__ import annotations

import pytest

from openavl.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR

BD_AVL = GEOMETRIES_DIR / "bd.avl"
TOL_CL = 5e-3
# OpenAVL alpha is ~0.13 deg high vs AVL bd.run with fuseBD body loads.
TOL_ALPHA = 0.13

REF_CL = 0.7
REF_ALPHA = 2.688


def build_bd_solver(bank: float) -> AVLSolver:
    """Build Bubble Dancer solver with non-default constraint wiring."""
    if not BD_AVL.is_file():
        pytest.skip(f"bd.avl not found: {BD_AVL}")

    solver = AVLSolver(
        BD_AVL,
        cl=REF_CL,
        cd0=0.017,
        rho=1.225,
        gravity=9.81,
        xcg=3.4,
        zcg=0.5,
        bank=bank,
    )
    # bd.run: alpha->CL, beta->Cl, elevator->Cm, rudder->Cn
    solver.set_constraint("beta", "cll", 0.0)
    solver.set_constraint("elevator", "cm", 0.0)
    solver.set_constraint("rudder", "cn", 0.0)
    return solver


pytestmark = pytest.mark.integration


@pytest.mark.reference
def test_bd_level_flight():
    """Level flight at bank=0; CL matches AVL bd.run case 1 reference."""
    solver = build_bd_solver(0.0)
    solver.execute_run(max_iter=20)
    results = solver.get_results()

    assert results["converged"]
    assert results["CL"] == pytest.approx(REF_CL, abs=TOL_CL)
    assert results["alpha_deg"] == pytest.approx(REF_ALPHA, abs=TOL_ALPHA)


@pytest.mark.reference
@pytest.mark.parametrize("bank", [10, 20, 30, 40, 50, 60], ids=[f"bank_{b}" for b in [10, 20, 30, 40, 50, 60]])
def test_bd_bank_sweep(bank: float):
    """Banked turn sweep; CL and alpha match AVL bd.run reference."""
    solver = build_bd_solver(bank)
    solver.execute_run(max_iter=20)
    results = solver.get_results()

    assert results["converged"]
    assert results["CL"] == pytest.approx(REF_CL, abs=TOL_CL)
    assert results["alpha_deg"] == pytest.approx(REF_ALPHA, abs=TOL_ALPHA)
