"""Re-trimming after a parameter change keeps lift equal to n·W, as AVL does.

After ``setup_trim``, AVL's trim menu zeroes the dependent value whenever a
parameter is edited (atrim.f): in level/banked flight CL is held and V is
recomputed, except that editing V recomputes CL. A re-trim used to keep the
old V and CL, so a 45° bank still gave L/W = 1.
"""

from __future__ import annotations

import math
import warnings

import pytest

from openavl import constants as C
from openavl.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR

SUPRA_AVL = GEOMETRIES_DIR / "supra.avl"
SUPRA_MASS = GEOMETRIES_DIR.parent / "mass" / "supra.mass"

pytestmark = pytest.mark.core


def _trimmed(**params):
    solver = AVLSolver(SUPRA_AVL, mass_file=SUPRA_MASS)
    solver.set_parameter("cl", 0.7)
    for name, value in params.items():
        solver.set_parameter(name, value)
    solver.setup_trim(mode=1)
    solver.set_constraint("elevator", "cm", 0.0)
    return solver


def _solve(solver):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver.execute_run()
    return solver


def _lift_over_weight(solver):
    s = solver.state
    p = s.parval[:, 0]
    sref = s.sref * s.unitl**2
    lift = 0.5 * p[C.IPRHO] * p[C.IPVEE] ** 2 * sref * s.cltot
    return lift / (p[C.IPMASS] * p[C.IPGEE])


def _v(solver):
    return float(solver.state.parval[C.IPVEE, 0])


@pytest.mark.parametrize("bank", [30.0, 45.0, -60.0])
def test_bank_change_after_trim_gives_load_factor(bank):
    solver = _solve(_trimmed())
    assert _lift_over_weight(solver) == pytest.approx(1.0, rel=1e-6)

    solver.set_parameter("bank", bank)
    _solve(solver)
    assert _lift_over_weight(solver) == pytest.approx(1.0 / math.cos(math.radians(bank)), rel=1e-6)
    # Same state as trimming a fresh solver at that bank angle.
    fresh = _solve(_trimmed(bank=bank))
    assert _v(solver) == pytest.approx(_v(fresh), rel=1e-12)
    assert solver.state.wrot == pytest.approx(fresh.state.wrot, rel=1e-9, abs=1e-12)


def test_calling_setup_trim_again_after_an_edit_is_harmless():
    solver = _solve(_trimmed())
    solver.set_parameter("bank", 45.0)
    solver.setup_trim(mode=1)
    _solve(solver)
    assert _lift_over_weight(solver) == pytest.approx(math.sqrt(2.0), rel=1e-6)


@pytest.mark.parametrize(("name", "factor"), [("mass", 1.5), ("density", 0.8), ("gravity", 1.2)])
def test_mass_density_gravity_changes_recompute_speed_at_fixed_cl(name, factor):
    solver = _solve(_trimmed())
    v0 = _v(solver)
    key = {"mass": C.IPMASS, "density": C.IPRHO, "gravity": C.IPGEE}[name]
    solver.set_parameter(name, float(solver.state.parval[key, 0]) * factor)
    _solve(solver)
    assert solver.state.cltot == pytest.approx(0.7, rel=1e-6)
    assert _lift_over_weight(solver) == pytest.approx(1.0, rel=1e-6)
    expected = v0 * math.sqrt(factor if name != "density" else 1.0 / factor)
    assert _v(solver) == pytest.approx(expected, rel=1e-9)


def test_velocity_change_recomputes_cl():
    solver = _solve(_trimmed())
    v0 = _v(solver)
    solver.set_parameter("velocity", 1.2 * v0)
    _solve(solver)
    assert _v(solver) == pytest.approx(1.2 * v0)
    assert solver.state.cltot == pytest.approx(0.7 / 1.44, rel=1e-6)
    assert _lift_over_weight(solver) == pytest.approx(1.0, rel=1e-6)


def test_cl_change_recomputes_speed():
    solver = _solve(_trimmed())
    v0 = _v(solver)
    solver.set_parameter("cl", 1.0)
    _solve(solver)
    assert solver.state.cltot == pytest.approx(1.0, rel=1e-6)
    assert _v(solver) == pytest.approx(v0 * math.sqrt(0.7 / 1.0), rel=1e-9)


def test_pull_up_velocity_change_recomputes_load_factor():
    solver = AVLSolver(SUPRA_AVL, mass_file=SUPRA_MASS)
    solver.set_parameter("cl", 0.7)
    solver.set_parameter("velocity", 10.0)
    solver.setup_trim(mode=2)
    s = solver.state
    n0 = float(s.parval[C.IPFAC, 0])
    solver.set_parameter("velocity", 12.0)
    assert float(s.parval[C.IPFAC, 0]) == pytest.approx(n0 * 1.44, rel=1e-12)


def test_manual_alpha_constraint_ends_automatic_retrim():
    solver = _solve(_trimmed())
    v0 = _v(solver)
    solver.set_constraint("alpha", "alpha", 4.0)
    solver.set_parameter("bank", 30.0)
    assert _v(solver) == v0  # no re-trim once the user drives alpha directly


def test_parameters_without_trim_are_stored_as_given():
    solver = AVLSolver(SUPRA_AVL, mass_file=SUPRA_MASS)
    solver.set_parameter("velocity", 9.0)
    solver.set_parameter("bank", 30.0)
    assert _v(solver) == 9.0


def test_bank_of_90_degrees_is_rejected_in_level_trim():
    solver = _solve(_trimmed())
    before = solver.state.parval[:, 0].copy()
    with pytest.raises(ValueError, match="-90 < bank < 90"):
        solver.set_parameter("bank", 90.0)
    # A rejected value leaves the run case as it was (bank and V untouched).
    assert (solver.state.parval[:, 0] == before).all()
