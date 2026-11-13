"""Tests for openavl.core.state."""

from __future__ import annotations

import pytest

from openavl import constants as C
from openavl.fileio.parser import parse_avl, parse_avl_file, prepare_model
from openavl.core.state import AVLState
from tests.helpers import GEOMETRIES_DIR, load_json_fixture

PLANE_AVL = GEOMETRIES_DIR / "plane.avl"
SUPRA_AVL = GEOMETRIES_DIR / "supra.avl"

pytestmark = pytest.mark.core


@pytest.mark.fixture
def test_plane_state_dimensions_match_js(fixtures_dir):
    """AVLState.from_model dimensions match JS buildExecState for plane.avl."""
    if not PLANE_AVL.is_file():
        pytest.skip(f"plane.avl not found: {PLANE_AVL}")

    expected = load_json_fixture(fixtures_dir, "state_plane_expected.json")
    model = prepare_model(parse_avl_file(PLANE_AVL), base_dir=PLANE_AVL.parent)
    state = AVLState.from_model(model)

    assert state.nvmax == expected["nvmax"]
    assert state.nstrmax == expected["nstrmax"]
    assert state.nsurfmax == expected["nsurfmax"]
    assert state.ncontrol == expected["ncontrol"]
    assert state.nbody == expected["nbody"]
    assert state.nlnode == expected["nlnode"]
    assert state.nlmax == expected["nlmax"]
    assert state.mach == pytest.approx(expected["mach"])
    assert state.sref == pytest.approx(expected["sref"])
    assert state.cref == pytest.approx(expected["cref"])
    assert state.bref == pytest.approx(expected["bref"])


@pytest.mark.fixture
def test_plane_state_run_case_defaults(fixtures_dir):
    """Default PARVAL/CONVAL/ICON wiring matches JS buildExecState."""
    expected = load_json_fixture(fixtures_dir, "state_plane_expected.json")
    model = prepare_model(parse_avl_file(PLANE_AVL), base_dir=PLANE_AVL.parent)
    state = AVLState.from_model(model)

    assert state.alfa / state.dtr == pytest.approx(expected["default_alpha_deg"])
    assert state.parval[C.IPCL, 0] == pytest.approx(expected["default_cl"])
    assert state.icon[C.IVALFA, 0] == expected["icon_alpha"]
    assert state.conval[C.ICCL, 0] == pytest.approx(expected["conval_cl"])
    assert len(state.control_names) == expected["ncontrol"]


@pytest.mark.fixture
def test_supra_state_body_allocation(fixtures_dir):
    """supra.avl allocates body node arrays before geometry build."""
    if not SUPRA_AVL.is_file():
        pytest.skip(f"supra.avl not found: {SUPRA_AVL}")

    expected = load_json_fixture(fixtures_dir, "geometry_expected.json")["supra"]
    model = prepare_model(parse_avl_file(SUPRA_AVL), base_dir=SUPRA_AVL.parent)
    state = AVLState.from_model(model)

    assert state.nbody == expected["nbody"]
    assert state.nlnode == expected["nlnode"]
    assert state.nlmax == expected["nlnode"]
    assert state.rl.shape[1] >= expected["nlnode"]
    assert state.radl.size >= expected["nlnode"]


def test_from_model_rejects_lattice_above_nvmax():
    """More than C.NVMAX vortices raises before any lattice arrays are allocated."""
    text = "Big\n0.0\n0 0 0\n1.0 1.0 1.0\n0 0 0\nSURFACE\nWing\n100 1.0 60 1.0\nSECTION\n0 0 0 1 0\nSECTION\n0 1 0 1 0\n"
    with pytest.raises(ValueError, match=rf"6000 vortices.*limit of {C.NVMAX}"):
        AVLState.from_model(parse_avl(text))


def test_from_model_rejects_body_nodes_above_nlmax():
    """More than C.NLMAX body nodes raises before any body arrays are allocated."""
    text = "Big\n0.0\n0 0 0\n1.0 1.0 1.0\n0 0 0\nBODY\nFuse\n300 1.0\nYDUPLICATE\n0.0\n"
    model = parse_avl(text)
    body = model.bodies[0]
    # Stand in for a BFIL thread so the body is counted.
    body.body_thread_x, body.body_thread_y, body.body_thread_t = [0.0, 1.0], [0.0, 0.0], [0.1, 0.1]
    with pytest.raises(ValueError, match=rf"600 body nodes.*limit of {C.NLMAX}.*Nbody"):
        AVLState.from_model(model)
