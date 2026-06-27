"""Tests for trim setup."""

from __future__ import annotations

import math

import pytest

from openavl import constants as C
from openavl.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR

PLANE_AVL = GEOMETRIES_DIR / "plane.avl"

pytestmark = pytest.mark.core


@pytest.mark.skipif(not PLANE_AVL.is_file(), reason="plane.avl not found")
def test_level_flight_trim():
    """CL is inferred from weight balance when lift coefficient is unspecified."""
    solver = AVLSolver(PLANE_AVL)
    solver.set_parameter("mass", 2.0)
    solver.set_parameter("density", 1.0)
    solver.set_parameter("gravity", 10.0)
    solver.set_parameter("velocity", 5.0)
    solver.set_parameter("cl", 0.0)
    solver.set_parameter("bank", 0.0)
    solver.state.conval[C.ICCL, 0] = 0.0

    solver.setup_trim(mode=1)

    s = solver.state
    ir = 0
    sref_d = s.sref * s.unitl * s.unitl
    expected_cl = 2.0 * s.parval[C.IPMASS, ir] * s.parval[C.IPGEE, ir] / (
        s.parval[C.IPRHO, ir] * sref_d * s.parval[C.IPVEE, ir] ** 2
    )
    assert s.parval[C.IPCL, ir] == pytest.approx(expected_cl, rel=1e-6)
    assert s.icon[C.IVALFA, ir] == C.ICCL
    assert s.conval[C.ICROTX, ir] == pytest.approx(0.0)
    assert s.conval[C.ICROTY, ir] == pytest.approx(0.0)
    assert s.conval[C.ICROTZ, ir] == pytest.approx(0.0)


@pytest.mark.skipif(not PLANE_AVL.is_file(), reason="plane.avl not found")
def test_banked_turn_trim():
    """Turn-rate constraints are set for a coordinated bank."""
    solver = AVLSolver(PLANE_AVL)
    solver.set_parameter("mass", 1.0)
    solver.set_parameter("density", 1.0)
    solver.set_parameter("gravity", 1.0)
    solver.set_parameter("velocity", 10.0)
    solver.set_parameter("cl", 0.8)
    solver.set_parameter("bank", 30.0)

    solver.setup_trim(mode=1)

    s = solver.state
    ir = 0
    phi = s.parval[C.IPPHI, ir] * s.dtr
    vee = s.parval[C.IPVEE, ir]
    gee = s.parval[C.IPGEE, ir]
    expected_rad = vee * vee * math.cos(phi) / (gee * math.sin(phi))
    cref_d = s.cref * s.unitl
    bref_d = s.bref * s.unitl

    assert s.parval[C.IPRAD, ir] == pytest.approx(expected_rad, rel=1e-6)
    assert s.conval[C.ICROTY, ir] == pytest.approx(math.sin(phi) * cref_d / (2.0 * expected_rad), rel=1e-6)
    assert s.conval[C.ICROTZ, ir] == pytest.approx(math.cos(phi) * bref_d / (2.0 * expected_rad), rel=1e-6)


@pytest.mark.skipif(not PLANE_AVL.is_file(), reason="plane.avl not found")
def test_pullup_trim():
    """Pull-up trim sets load factor through turn radius."""
    solver = AVLSolver(PLANE_AVL)
    solver.set_parameter("mass", 1.0)
    solver.set_parameter("density", 1.0)
    solver.set_parameter("cl", 0.6)
    solver.set_parameter("velocity", 20.0)
    solver.set_parameter("gravity", 9.81)

    solver.setup_trim(mode=2)

    s = solver.state
    ir = 0
    sref_d = s.sref * s.unitl * s.unitl
    expected_rad = s.parval[C.IPMASS, ir] / (0.5 * s.parval[C.IPRHO, ir] * sref_d * s.parval[C.IPCL, ir])
    assert s.parval[C.IPRAD, ir] == pytest.approx(expected_rad, rel=1e-6)
    assert s.conval[C.ICROTY, ir] == pytest.approx(s.cref * s.unitl / (2.0 * expected_rad), rel=1e-6)
