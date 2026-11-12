"""Moving the CG and setting body rates directly give the physically right flow.

* Body source/doublet data depend on the moment reference point, so a solver
  reused after a CG change must match a fresh solver at the new CG (AVL clears
  LSRD whenever Xref changes).
* ``set_variable`` on pb/2V, qc/2V, rb/2V must mean the same thing as a rate
  constraint of that value, rather than a raw entry in the internal ``wrot``.
"""

from __future__ import annotations

import pytest

from openavl import constants as C
from openavl.core.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR

BD_AVL = GEOMETRIES_DIR / "bd.avl"
PLANE_AVL = GEOMETRIES_DIR / "plane.avl"

pytestmark = pytest.mark.core

RATES = {"pb/2V": 0.05, "qc/2V": 0.02, "rb/2V": 0.03}
OUTPUTS = ("CL", "CD", "CY", "Cl", "Cm", "Cn")


def _results(solver):
    r = solver.get_results()
    return {k: float(r[k]) for k in OUTPUTS}


# ---- CG change must rebuild body source/doublet data ------------------------


def _bd_run(solver, xcg):
    solver.set_parameter("xcg", xcg)
    solver.set_constraint("alpha", "alpha", 3.0)
    for name, value in RATES.items():
        solver.set_constraint(name, name, value)
    solver.execute_run()
    return _results(solver)


def test_reused_solver_after_cg_change_matches_fresh_solver():
    assert AVLSolver(BD_AVL).state.nbody > 0, "test needs a model with bodies"

    reused = AVLSolver(BD_AVL)
    _bd_run(reused, 3.4)
    moved = _bd_run(reused, 3.0)

    fresh = _bd_run(AVLSolver(BD_AVL), 3.0)
    for key in OUTPUTS:
        assert moved[key] == pytest.approx(fresh[key], rel=1e-10, abs=1e-12), key


def test_body_data_rebuilt_when_reference_point_moves_without_a_setter():
    """xyzref can be moved by other code paths; setup must still notice."""
    solver = AVLSolver(BD_AVL)
    _bd_run(solver, 3.4)
    src_before = solver.state.src_u.copy()
    solver.state.parval[C.IPXCG, 0] = 3.0
    solver.execute_run()
    assert not (solver.state.src_u == src_before).all()
    assert solver.state.xyzref_srd[0] == pytest.approx(3.0)


# ---- set_variable on rates -----------------------------------------------


def _constrained(alpha, lsa_rates=False):
    solver = AVLSolver(PLANE_AVL)
    solver.state.lsa_rates = lsa_rates
    solver.set_constraint("alpha", "alpha", alpha)
    for name, value in RATES.items():
        solver.set_constraint(name, name, value)
    solver.execute_run()
    return solver


def _direct(alpha, rates_first=False, lsa_rates=False):
    solver = AVLSolver(PLANE_AVL)
    solver.state.lsa_rates = lsa_rates
    if not rates_first:
        solver.set_variable("alpha", alpha)
    for name, value in RATES.items():
        solver.set_variable(name, value)
    if rates_first:
        solver.set_variable("alpha", alpha)
    solver.execute_run(max_iter=0)
    return solver


@pytest.mark.parametrize("lsa_rates", [False, True], ids=["body-axis rates", "stability-axis rates"])
@pytest.mark.parametrize("alpha", [0.0, 6.0])
def test_set_variable_rates_match_rate_constraints(alpha, lsa_rates):
    direct = _direct(alpha, lsa_rates=lsa_rates)
    constrained = _constrained(alpha, lsa_rates=lsa_rates)
    assert constrained.get_results()["converged"]
    assert direct.state.wrot == pytest.approx(constrained.state.wrot, rel=1e-9, abs=1e-12)
    for key, value in _results(constrained).items():
        assert _results(direct)[key] == pytest.approx(value, rel=1e-9, abs=1e-12), key


def test_set_variable_rates_are_scaled_not_raw():
    solver = AVLSolver(PLANE_AVL)
    solver.set_variable("qc/2V", 0.02)
    # q = (qc/2V) * 2 / cref, not 0.02
    assert solver.state.wrot[1] == pytest.approx(0.02 * 2.0 / solver.state.cref)
    assert solver.state.cref != pytest.approx(2.0)


def test_rates_and_alpha_can_be_set_in_any_order():
    """Matters with stability-axis rates, where the body rates depend on alpha."""
    a = _direct(6.0, rates_first=False, lsa_rates=True).state.wrot
    b = _direct(6.0, rates_first=True, lsa_rates=True).state.wrot
    assert a == pytest.approx(b, rel=1e-12, abs=1e-15)
