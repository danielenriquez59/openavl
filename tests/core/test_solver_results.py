"""Tests for AVLSolver result payloads."""

from __future__ import annotations

import pytest

from openavl import constants as C
from openavl.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR

SUPRA_AVL = GEOMETRIES_DIR / "supra.avl"
SUPRA_MASS = GEOMETRIES_DIR.parent / "mass" / "supra.mass"

pytestmark = pytest.mark.core


@pytest.mark.skipif(not SUPRA_AVL.is_file(), reason="supra.avl not found")
def test_get_results_control_deflections():
    """Control deflections are exposed by name in get_results."""
    solver = AVLSolver(SUPRA_AVL)
    solver.set_variable("aileron", 5.0)
    solver.set_variable("elevator", -2.0)
    solver.set_variable("rudder", 1.5)
    solver.execute_run(max_iter=0)

    results = solver.get_results()
    controls = results["control_deflections"]
    assert controls["aileron"] == pytest.approx(5.0)
    assert controls["elevator"] == pytest.approx(-2.0)
    assert controls["rudder"] == pytest.approx(1.5)


@pytest.mark.skipif(not SUPRA_AVL.is_file(), reason="supra.avl not found")
def test_get_results_moment_scalars():
    """Force and moment vector members are exposed as scalar keys."""
    solver = AVLSolver(SUPRA_AVL, alpha=5.0, beta=2.0)
    solver.execute_run(max_iter=0)

    results = solver.get_results()
    assert {"CM", "CM_sa", "CF"}.isdisjoint(results)
    for key in ("Cl", "Cm", "Cn", "Cl_sa", "Cm_sa", "Cn_sa", "Cx", "Cy", "Cz"):
        assert key in results


@pytest.mark.skipif(not SUPRA_AVL.is_file(), reason="supra.avl not found")
def test_replace_constraints_clears_previous_trim_assignment():
    """A complete run case fixes variables omitted from its constraint list."""
    solver = AVLSolver(SUPRA_AVL, alpha=4.0)
    solver.set_variable("elevator", -3.0)
    solver.set_constraint("elevator", "cm", 0.0)

    solver.replace_constraints([("alpha", "cl", 0.8)])

    elevator = solver.model.control_map["elevator"]
    elevator_variable = C.IVTOT + elevator
    elevator_constraint = C.ICTOT + elevator
    assert solver.state.icon[C.IVALFA, 0] == C.ICCL
    assert solver.state.conval[C.ICCL, 0] == pytest.approx(0.8)
    assert solver.state.icon[elevator_variable, 0] == elevator_constraint
    assert solver.state.conval[elevator_constraint, 0] == pytest.approx(0.0)
    assert solver.state.delcon[elevator] == pytest.approx(0.0)


@pytest.mark.skipif(
    not SUPRA_AVL.is_file() or not SUPRA_MASS.is_file(),
    reason="supra geometry or mass file not found",
)
def test_get_aero_accel_from_integrated_loads():
    """Aerodynamic accelerations are consistent with dimensional body loads."""
    solver = AVLSolver(SUPRA_AVL, mass_file=SUPRA_MASS, alpha=5.0, rho=1.225, velocity=12.0)
    solver.execute_run(max_iter=0)

    acceleration = solver.get_aero_accel()

    assert set(acceleration) == {
        "dynamic_pressure",
        "force_body",
        "moment_body",
        "linear_acceleration_body",
        "rotational_acceleration_body",
        "mass",
        "inertia",
    }
    assert acceleration["force_body"].shape == (3,)
    assert acceleration["moment_body"].shape == (3,)
    assert acceleration["linear_acceleration_body"].shape == (3,)
    assert acceleration["rotational_acceleration_body"].shape == (3,)
    assert acceleration["inertia"].shape == (3, 3)
    assert acceleration["mass"] == pytest.approx(solver.state.parval[C.IPMASS, 0])
    assert acceleration["linear_acceleration_body"] == pytest.approx(
        acceleration["force_body"] / acceleration["mass"]
    )


@pytest.mark.skipif(
    not SUPRA_AVL.is_file() or not SUPRA_MASS.is_file(),
    reason="supra geometry or mass file not found",
)
def test_get_aero_accel_requires_positive_mass():
    """Aerodynamic acceleration requires a positive run-case mass."""
    solver = AVLSolver(SUPRA_AVL, mass_file=SUPRA_MASS)
    solver.state.parval[C.IPMASS, 0] = 0.0

    with pytest.raises(ValueError, match="mass must be positive"):
        solver.get_aero_accel()
