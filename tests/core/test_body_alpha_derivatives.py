"""Alpha derivatives of models with bodies against finite differences.

Lift and drag are measured in stability axes, which turn with alpha, so the
body force tilts between them. Fortran AVL's BDFORC leaves that term out
(its CL_a misses -CD_body and CD_a misses +CL_body); OpenAVL includes it.
"""

from __future__ import annotations

import math

import pytest

from openavl.core.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR

ALPHA = 4.0
H = 0.01  # deg


def _solve(path, alpha: float) -> AVLSolver:
    solver = AVLSolver(path)
    solver.set_constraint("alpha", "alpha", alpha)
    solver.execute_run(max_iter=20)
    assert solver.state.lsol
    return solver


@pytest.mark.parametrize("name", ["supraf.avl", "supra.avl"])  # bodies only; wing + body
def test_alpha_derivatives_match_finite_difference(name: str) -> None:
    path = GEOMETRIES_DIR / name
    derivs = _solve(path, ALPHA).get_stability_derivatives()
    plus, minus = _solve(path, ALPHA + H).state, _solve(path, ALPHA - H).state
    assert plus.nbody > 0
    step = math.radians(2 * H)
    assert derivs.CL_a == pytest.approx((plus.cltot - minus.cltot) / step, rel=2e-8, abs=2e-8)
    assert derivs.CD_a == pytest.approx((plus.cdtot - minus.cdtot) / step, rel=2e-8, abs=2e-8)
