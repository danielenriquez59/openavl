"""End-to-end integration test for w.avl (Hershey bar wing-only geometry)."""

from __future__ import annotations

import pytest

from openavl.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR

W_AVL = GEOMETRIES_DIR / "w.avl"
TOL_CL = 5e-3
TOL_ALPHA = 0.05

REF_ALPHA_AT_CL_0_8 = 4.228


def build_w_solver() -> AVLSolver:
    """Build Hershey bar solver trimmed to CL=0.8."""
    if not W_AVL.is_file():
        pytest.skip(f"w.avl not found: {W_AVL}")

    solver = AVLSolver(W_AVL, cl=0.8)
    return solver


pytestmark = pytest.mark.integration


@pytest.mark.reference
def test_w_cl_trim():
    """Alpha trimmed to CL=0.8; alpha matches AVL w.run reference."""
    solver = build_w_solver()
    solver.execute_run(max_iter=20)
    results = solver.get_results()

    assert results["converged"]
    assert results["CL"] == pytest.approx(0.8, abs=TOL_CL)
    assert results["alpha_deg"] == pytest.approx(REF_ALPHA_AT_CL_0_8, abs=TOL_ALPHA)
