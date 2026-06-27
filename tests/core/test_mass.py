"""Tests for openavl.mass."""

from __future__ import annotations

import numpy as np
import pytest

from openavl import constants as C
from openavl.mass import (
    MassProperties,
    load_mass,
    masget,
    masini,
    masput,
    parse_mass_file,
    parse_mass_text,
)
from openavl.parser import parse_avl_file, prepare_model
from openavl.solver import AVLSolver
from openavl.state import AVLState
from tests.helpers import FIXTURES_DIR, GEOMETRIES_DIR, TESTS_DIR, load_json_fixture

MASS_DIR = TESTS_DIR / "data" / "avl" / "mass"
AMASS_TEST = MASS_DIR / "amass_test.mass"
SUPRA_MASS = MASS_DIR / "supra.mass"
PLANE_AVL = GEOMETRIES_DIR / "plane.avl"

pytestmark = pytest.mark.core


def test_masini_defaults():
    """MASINI sets identity inertia defaults and clears apparent mass."""
    state = AVLState()
    masini(state)
    assert state.rmass0 == pytest.approx(1.0)
    assert state.lmass is False
    np.testing.assert_array_equal(state.xyzmass0, np.zeros(3))
    np.testing.assert_array_equal(np.diag(state.riner0), np.ones(3))
    assert np.all(state.amass == 0.0)
    assert np.all(state.ainer == 0.0)


@pytest.mark.fixture
def test_parse_amass_test_fixture(fixtures_dir):
    """Parse amass_test.mass and match JSON fixture aggregates."""
    expected = load_json_fixture(fixtures_dir, "mass_amass_test_expected.json")
    props = parse_mass_file(AMASS_TEST)

    assert props.loaded is expected["lmass"]
    assert props.mass == pytest.approx(expected["rmass0"])
    np.testing.assert_allclose(props.cg, expected["xyzmass0"], rtol=1e-12)
    np.testing.assert_allclose(
        props.inertia.flatten("C"),
        expected["riner0"],
        rtol=1e-12,
    )
    assert props.gee == pytest.approx(expected["gee0"])
    assert props.rho == pytest.approx(expected["rho0"])
    assert props.unitl == pytest.approx(expected["unitl"])
    assert props.unitm == pytest.approx(expected["unitm"])
    assert props.unitt == pytest.approx(expected["unitt"])


@pytest.mark.fixture
def test_masget_populates_state(fixtures_dir):
    """MASGET writes mass properties and unit scales onto AVLState."""
    expected = load_json_fixture(fixtures_dir, "mass_amass_test_expected.json")
    state = AVLState()
    assert masget(state, AMASS_TEST) is True

    assert state.lmass is True
    assert state.rmass0 == pytest.approx(expected["rmass0"])
    np.testing.assert_allclose(state.xyzmass0, expected["xyzmass0"], rtol=1e-12)
    np.testing.assert_allclose(
        state.riner0.flatten("C"),
        expected["riner0"],
        rtol=1e-12,
    )
    assert state.gee0 == pytest.approx(expected["gee0"])
    assert state.rho0 == pytest.approx(expected["rho0"])
    assert state.unitl == pytest.approx(expected["unitl"])
    assert state.unitm == pytest.approx(expected["unitm"])
    assert state.unitt == pytest.approx(expected["unitt"])


def test_masput_writes_parval():
    """MASPUT stores mass, inertia, CG, g, and rho in PARVAL."""
    state = AVLState()
    state.rmass0 = 5.0
    state.riner0[:] = [
        [1.1, -0.1, -0.3],
        [-0.1, 2.2, -0.2],
        [-0.3, -0.2, 3.3],
    ]
    state.gee0 = 9.81
    state.rho0 = 1.225
    state.xyzmass0[:] = [0.4, -0.5, 0.6]
    state.unitl = 2.0

    masput(state, 0, 0)

    assert state.parval[C.IPMASS, 0] == pytest.approx(5.0)
    assert state.parval[C.IPIXX, 0] == pytest.approx(1.1)
    assert state.parval[C.IPIYY, 0] == pytest.approx(2.2)
    assert state.parval[C.IPIZZ, 0] == pytest.approx(3.3)
    assert state.parval[C.IPIXY, 0] == pytest.approx(-0.1)
    assert state.parval[C.IPIYZ, 0] == pytest.approx(-0.2)
    assert state.parval[C.IPIZX, 0] == pytest.approx(-0.3)
    assert state.parval[C.IPGEE, 0] == pytest.approx(9.81)
    assert state.parval[C.IPRHO, 0] == pytest.approx(1.225)
    assert state.parval[C.IPXCG, 0] == pytest.approx(0.2)
    assert state.parval[C.IPYCG, 0] == pytest.approx(-0.25)
    assert state.parval[C.IPZCG, 0] == pytest.approx(0.3)


def test_load_mass_end_to_end():
    """load_mass combines MASGET and MASPUT."""
    state = AVLState()
    props = load_mass(state, AMASS_TEST)
    assert isinstance(props, MassProperties)
    assert props.loaded is True
    assert state.lmass is True
    assert state.parval[C.IPMASS, 0] == pytest.approx(props.mass)


def test_masget_missing_file():
    """MASGET returns False for missing paths."""
    state = AVLState()
    assert masget(state, None) is False
    assert masget(state, "nonexistent.mass") is False
    assert state.lmass is False


def test_parse_mass_text_multiplier_and_adder():
    """* and + scaling lines apply to subsequent data rows."""
    text = "\n".join(
        [
            "* 10 1 1 1 1 1 1 1 1 1",
            "+ 0 0 0 0 0 0 0 0 0 0",
            "1 2 3 4",
        ]
    )
    props = parse_mass_text(text)
    assert props.loaded is True
    assert props.mass == pytest.approx(10.0)


@pytest.mark.skipif(not SUPRA_MASS.is_file(), reason="supra.mass not available")
def test_supra_mass_file():
    """Real supra.mass loads positive mass with inch/kg unit scaling."""
    props = parse_mass_file(SUPRA_MASS)
    assert props.loaded is True
    assert props.mass > 0.0
    assert props.unitl == pytest.approx(0.0254)
    assert props.unitm == pytest.approx(0.001)
    assert props.gee == pytest.approx(9.81)
    assert props.rho == pytest.approx(1.225)
    assert np.all(np.isfinite(props.cg))
    assert np.all(np.isfinite(props.inertia))


@pytest.mark.skipif(
    not PLANE_AVL.is_file() or not SUPRA_MASS.is_file(),
    reason="plane.avl or supra.mass not available",
)
def test_solver_loads_mass_file():
    """AVLSolver wires mass_file into state PARVAL."""
    solver = AVLSolver(PLANE_AVL, mass_file=SUPRA_MASS)
    assert solver.state.lmass is True
    assert solver.model.mass is not None
    assert solver.model.mass.loaded is True
    assert solver.state.parval[C.IPMASS, 0] == pytest.approx(solver.model.mass.mass)
    assert solver.state.parval[C.IPRHO, 0] == pytest.approx(solver.model.mass.rho)
    assert solver.state.parval[C.IPGEE, 0] == pytest.approx(solver.model.mass.gee)
