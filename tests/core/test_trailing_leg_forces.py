"""Trailing-leg forces (``ltrforce``) include control-deflection sensitivities.

AVL 3.52 ``aero.f`` adds ``FGAM_D = 2 GAM_D (Veff x G)`` for the parts of the
trailing legs lying on the wing. Without it the control derivatives, and the
trim Jacobian built from them, ignored the leg forces entirely: switching
``ltrforce`` on changed them by exactly zero (Fortran AVL: -2.05e-4 per degree
in CL for the plane.avl aileron at alpha = beta = 5 deg). The leg forces only
appear with sideslip or rotation, so the case below has both.
"""

from __future__ import annotations

import numpy as np
import pytest

from openavl.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR

PLANE_AVL = GEOMETRIES_DIR / "plane.avl"
DEFLECTION, STEP = 3.0, 0.01  # degrees

pytestmark = pytest.mark.core


def _solve(deflection: float) -> AVLSolver:
    solver = AVLSolver(PLANE_AVL)
    solver.state.ltrforce = True
    for variable, value in {"alpha": 5.0, "beta": 5.0, "pb/2V": 0.05, "rb/2V": 0.02}.items():
        solver.set_constraint(variable, variable, value)
    first, *others = solver.state.control_names
    solver.set_constraint(first, first, deflection)
    for name in others:
        solver.set_constraint(name, name, 0.0)
    solver.execute_run()
    return solver


def _totals(solver: AVLSolver) -> np.ndarray:
    return np.concatenate([solver.state.cftot[:3], solver.state.cmtot[:3]])


@pytest.mark.skipif(not PLANE_AVL.is_file(), reason="plane.avl not found")
def test_control_derivatives_match_finite_differences_with_trailing_leg_forces():
    solver = _solve(DEFLECTION)
    assert solver.state.ncontrol > 0
    analytic = np.concatenate([solver.state.cftot_d[:3, 0], solver.state.cmtot_d[:3, 0]])  # per degree
    finite_diff = (_totals(_solve(DEFLECTION + STEP)) - _totals(_solve(DEFLECTION - STEP))) / (2 * STEP)
    # Before the fix the error was 1.0e-2 of the largest derivative.
    np.testing.assert_allclose(analytic, finite_diff, rtol=0, atol=1e-9 * np.abs(finite_diff).max())


# Fortran AVL 3.52 (optvl 2.5.0, LTRFORCE on), plane.avl, alpha = beta = 5 deg,
# aileron 3 deg, other controls 0. Stability axes (CL, CY, Cl', Cm, Cn'), per degree.
FORTRAN_CONTROL_DERIVS = {
    "aileron": (-0.00032143309262324613, -0.0052366617826027535, -0.019906802977253456,
                0.0004166979143747548, 0.0009019027254587707),
    "elevator": (0.010432946251985533, -9.30392194119552e-06, -1.3979599762338373e-05,
                 -0.06077014052513749, 7.885253421692155e-06),
    "rudder": (2.9457510236892025e-06, -0.001530230429864769, -2.3374425803519685e-05,
               -4.3280256024698084e-05, 0.0006186337289323413),
}


@pytest.mark.skipif(not PLANE_AVL.is_file(), reason="plane.avl not found")
def test_control_derivatives_match_fortran_avl_with_trailing_leg_forces():
    solver = AVLSolver(PLANE_AVL)
    solver.state.ltrforce = True
    solver.set_constraint("alpha", "alpha", 5.0)
    solver.set_constraint("beta", "beta", 5.0)
    first, *others = solver.state.control_names
    solver.set_constraint(first, first, DEFLECTION)
    for name in others:
        solver.set_constraint(name, name, 0.0)
    solver.execute_run()

    derivs = solver.get_control_derivatives()  # per radian
    cols = [derivs.cols.index(c) for c in ("CL", "CY", "Cl", "Cm", "Cn")]
    for name, row in zip(derivs.rows, derivs.values):
        per_degree = np.array([row[c] for c in cols]) * np.pi / 180.0
        expected = np.array(FORTRAN_CONTROL_DERIVS[name])
        # Matches to ~6e-9 of the largest; without the leg _d terms it fails.
        np.testing.assert_allclose(per_degree, expected, rtol=0, atol=1e-7 * np.abs(expected).max(), err_msg=name)
