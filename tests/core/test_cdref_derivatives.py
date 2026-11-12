"""Regression test: reference drag (CDp / cd0) contribution to CY_b (AERO-03).

``aero()`` adds ``CDREF*VINF(2)*VMAG`` to CY. Fortran AVL omits the matching
CYTOT_U term, so its CY_b is short by CDREF; OpenAVL deliberately includes it.
"""

from __future__ import annotations

import math

import pytest

from openavl.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR

PLANE_AVL = GEOMETRIES_DIR / "plane.avl"
ALPHA_DEG = 3.0
BETA_DEG = 5.0
STEP_DEG = 0.01

pytestmark = pytest.mark.core


def _solve(cd0: float, beta_deg: float) -> AVLSolver:
    solver = AVLSolver(PLANE_AVL, cd0=cd0)
    solver.set_constraint("alpha", "alpha", ALPHA_DEG)
    solver.set_constraint("beta", "beta", beta_deg)
    solver.execute_run(max_iter=20)
    return solver


@pytest.mark.skipif(not PLANE_AVL.is_file(), reason="plane.avl not found")
@pytest.mark.parametrize("cd0", [0.0, 0.02])
def test_cy_beta_matches_finite_difference_with_cd0(cd0):
    """Reported CY_b equals a central difference of CY in beta at fixed alpha."""
    cy_b = _solve(cd0, BETA_DEG).get_stability_derivatives().CY_b
    cy_plus = float(_solve(cd0, BETA_DEG + STEP_DEG).state.cytot)
    cy_minus = float(_solve(cd0, BETA_DEG - STEP_DEG).state.cytot)
    cy_b_fd = (cy_plus - cy_minus) / (2.0 * math.radians(STEP_DEG))

    assert cy_b == pytest.approx(cy_b_fd, abs=1e-6)
