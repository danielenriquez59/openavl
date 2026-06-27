"""End-to-end integration test for greff.avl (multiple flap controls)."""

from __future__ import annotations

import pytest

from openavl.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR

GREFF_AVL = GEOMETRIES_DIR / "greff.avl"
TOL_CL = 5e-3
TOL_ALPHA = 0.05

REF_ALPHA_AT_CL_1_0 = 8.75
def build_greff_solver() -> AVLSolver:
    """Build greff solver trimmed to CL=1.0 with flaps at zero."""
    if not GREFF_AVL.is_file():
        pytest.skip(f"greff.avl not found: {GREFF_AVL}")

    return AVLSolver(GREFF_AVL, cl=1.0)


pytestmark = pytest.mark.integration


@pytest.mark.reference
def test_greff_cl_trim():
    """Alpha trimmed to CL=1.0; alpha matches AVL greff.run reference."""
    solver = build_greff_solver()
    solver.execute_run(max_iter=20)
    results = solver.get_results()

    assert results["converged"]
    assert results["CL"] == pytest.approx(1.0, abs=TOL_CL)
    assert results["alpha_deg"] == pytest.approx(REF_ALPHA_AT_CL_1_0, abs=TOL_ALPHA)
