"""CORE-06: Mach numbers outside the subsonic range give a clear error."""

from __future__ import annotations

import math

import pytest

from openavl import constants as C
from openavl.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR

PLANE_AVL = GEOMETRIES_DIR / "plane.avl"

pytestmark = pytest.mark.core

BAD_MACH = [1.0, 1.5, -0.1, math.nan, math.inf]


@pytest.mark.parametrize("mach", BAD_MACH)
def test_set_parameter_rejects_out_of_range_mach(mach):
    solver = AVLSolver(str(PLANE_AVL))
    before = solver.state.parval[C.IPMACH, 0]
    with pytest.raises(ValueError, match="Mach number must satisfy"):
        solver.set_parameter("mach", mach)
    # A rejected value leaves the run case untouched.
    assert solver.state.parval[C.IPMACH, 0] == before


@pytest.mark.parametrize("mach", BAD_MACH)
def test_execute_run_rejects_out_of_range_mach_in_run_case(mach):
    """Values written straight into the run case are caught before the AIC build."""
    solver = AVLSolver(str(PLANE_AVL))
    solver.state.parval[C.IPMACH, 0] = mach
    with pytest.raises(ValueError, match="Mach number must satisfy"):
        solver.execute_run()


@pytest.mark.parametrize("mach", [0.0, 0.5, 0.95])
def test_subsonic_mach_still_solves(mach):
    solver = AVLSolver(str(PLANE_AVL))
    solver.set_parameter("mach", mach)
    solver.set_variable("alpha", 2.0)
    solver.execute_run()
    assert math.isfinite(solver.get_results()["CL"])
