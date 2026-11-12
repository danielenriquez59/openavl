"""Rotation rates in turns follow AVL's conventions, and alpha derivatives are axis-independent.

* Rate constraints are stability-axis rates by default, as in AVL 3.52
  (``LSA_RATES = .TRUE.`` in avl.f). With body-axis rates, plane.avl at
  alpha = 8 deg with all three rates set differed from Fortran AVL by 57% in
  CY; the level-turn trim, which computes its rates for stability axes, gave
  an 18% low roll moment on Supra at 30 deg bank.
* The alpha derivative of the body rates (held in stability axes) is a pure
  rotation and must not depend on the output-axis flag ``lnasa_sa``. AVL's
  formula flips its sign in NASA axes, which made CL_a and Cm_a in a turn
  0.8% / 1% off the finite-difference values; OpenAVL departs from AVL here.
"""

from __future__ import annotations

import math

import pytest

from openavl import constants as C
from openavl.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR

PLANE_AVL = GEOMETRIES_DIR / "plane.avl"
SUPRA_AVL = GEOMETRIES_DIR / "supra.avl"
SUPRA_MASS = GEOMETRIES_DIR.parent / "mass" / "supra.mass"

pytestmark = pytest.mark.core

# Fortran AVL 3.52 (optvl 2.5.0, default options): plane.avl, alpha = 8 deg,
# pb/2V = 0.05, qc/2V = 0.02, rb/2V = 0.03, controls 0.
FORTRAN_TOTALS = {
    "CL": 1.4927318488283419, "CD": -0.006818617501966012, "CY": -0.002680325873204837,
    "Cl": -0.02348214954136178, "Cm": -1.2535411975516, "Cn": -0.011016801890286535,
}


@pytest.mark.skipif(not PLANE_AVL.is_file(), reason="plane.avl not found")
def test_default_rate_constraints_match_fortran_avl():
    solver = AVLSolver(PLANE_AVL)
    solver.set_constraint("alpha", "alpha", 8.0)
    for name, value in (("pb/2V", 0.05), ("qc/2V", 0.02), ("rb/2V", 0.03)):
        solver.set_constraint(name, name, value)
    solver.execute_run()
    results = solver.get_results()
    for key, expected in FORTRAN_TOTALS.items():
        assert results[key] == pytest.approx(expected, rel=1e-7, abs=1e-9), key


needs_supra = pytest.mark.skipif(
    not SUPRA_AVL.is_file() or not SUPRA_MASS.is_file(), reason="supra files not found"
)


def _turn(lnasa_sa: bool, alpha: float | None = None, rates: dict | None = None) -> AVLSolver:
    solver = AVLSolver(SUPRA_AVL, mass_file=SUPRA_MASS)
    solver.state.lnasa_sa = lnasa_sa
    solver.set_parameter("bank", 30.0)
    solver.setup_trim(1)
    if alpha is not None:  # hold the turn's rates, step alpha
        solver.set_constraint("alpha", "alpha", alpha)
        for name, value in rates.items():
            solver.set_constraint(name, name, value)
    solver.execute_run()
    return solver


@needs_supra
@pytest.mark.parametrize("lnasa_sa", [True, False], ids=["NASA axes", "geometric axes"])
def test_alpha_derivatives_in_a_turn_match_finite_differences(lnasa_sa):
    trimmed = _turn(lnasa_sa)
    s = trimmed.state
    rates = {
        "pb/2V": float(s.conval[C.ICROTX, 0]),
        "qc/2V": float(s.conval[C.ICROTY, 0]),
        "rb/2V": float(s.conval[C.ICROTZ, 0]),
    }
    assert abs(rates["rb/2V"]) > 0.05, "case needs a real turn rate"
    alpha0, h = math.degrees(s.alfa), 0.01
    up = _turn(lnasa_sa, alpha0 + h, rates).get_results()
    dn = _turn(lnasa_sa, alpha0 - h, rates).get_results()
    derivs = trimmed.get_stability_derivatives()
    for key, analytic in (("CL", derivs.CL_a), ("Cm", derivs.Cm_a)):
        finite_diff = (up[key] - dn[key]) / math.radians(2 * h)
        # Before the fix, NASA axes were 0.8% (CL_a) and 1% (Cm_a) off.
        assert analytic == pytest.approx(finite_diff, rel=1e-5), key
