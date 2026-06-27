"""Tests for stability derivative extraction."""

from __future__ import annotations

import json

import pytest

from openavl import constants as C
from openavl.solver import AVLSolver

from tests.helpers import FIXTURES_DIR, GEOMETRIES_DIR

PLANE_AVL = GEOMETRIES_DIR / "plane.avl"
FIXTURE_PATH = FIXTURES_DIR / "plane_fortran_derivatives.json"
TOL = 1e-4

pytestmark = pytest.mark.core


def _configure_derivative_run(solver: AVLSolver) -> None:
    """Match the Fortran derivative reference run-case setup."""
    solver.set_parameter("mass", 1.0)
    solver.set_parameter("velocity", 1.0)
    solver.set_parameter("density", 1.0)
    solver.set_parameter("gravity", 1.0)
    solver.set_parameter("mach", 0.0)
    solver.set_parameter("bank", 0.0)
    solver.set_parameter("cd0", 0.0)
    solver.set_variable("alpha", 1.0)
    solver.set_variable("beta", 0.0)

    s = solver.state
    ir = 0
    s.icon[C.IVALFA, ir] = C.ICALFA
    s.icon[C.IVBETA, ir] = C.ICBETA
    s.icon[C.IVROTX, ir] = C.ICROTX
    s.icon[C.IVROTY, ir] = C.ICROTY
    s.icon[C.IVROTZ, ir] = C.ICROTZ
    s.conval[C.ICALFA, ir] = 1.0
    s.conval[C.ICBETA, ir] = 0.0
    s.conval[C.ICROTX, ir] = 0.0
    s.conval[C.ICROTY, ir] = 0.0
    s.conval[C.ICROTZ, ir] = 0.0
    for n in range(s.ncontrol):
        iv = C.IVTOT + n
        ic = C.ICTOT + n
        s.icon[iv, ir] = ic
        s.conval[ic, ir] = 0.0


@pytest.mark.skipif(not PLANE_AVL.is_file(), reason="plane.avl not found")
@pytest.mark.skipif(not FIXTURE_PATH.is_file(), reason="derivative fixture not found")
@pytest.mark.reference
def test_stability_derivatives_match_fixture():
    """Compare stability derivatives to plane_fortran_derivatives.json."""
    refs = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    expected = refs["stability"]

    solver = AVLSolver(PLANE_AVL)
    _configure_derivative_run(solver)
    solver.execute_run(max_iter=20)
    derivs = solver.get_stability_derivatives()

    assert derivs.CL_a == pytest.approx(expected["CLa"], abs=TOL)
    assert derivs.Cm_q == pytest.approx(expected["Cmq"], abs=TOL)
    assert derivs.Cl_p == pytest.approx(expected["Clp"], abs=TOL)
    assert derivs.Cn_r == pytest.approx(expected["Cnr"], abs=TOL)
    assert derivs.CL_b == pytest.approx(expected["CLb"], abs=TOL)
    assert derivs.CY_b == pytest.approx(expected["CYb"], abs=TOL)
    assert derivs.Cm_a == pytest.approx(expected["Cma"], abs=TOL)

    ctrl_name = solver.state.control_names[0]
    assert derivs.CY_d[ctrl_name] == pytest.approx(expected["CYd1"], abs=TOL)
    assert derivs.Cl_d[ctrl_name] == pytest.approx(expected["Cld1"], abs=TOL)
    assert derivs.Cn_d[ctrl_name] == pytest.approx(expected["Cnd1"], abs=TOL)


@pytest.mark.skipif(not PLANE_AVL.is_file(), reason="plane.avl not found")
def test_control_derivatives_are_dicts():
    """Control derivatives are keyed by control name."""
    solver = AVLSolver(PLANE_AVL)
    _configure_derivative_run(solver)
    solver.execute_run(max_iter=20)
    derivs = solver.get_stability_derivatives()

    assert len(derivs.CL_d) == solver.state.ncontrol
    for name in solver.state.control_names:
        assert name in derivs.CL_d
