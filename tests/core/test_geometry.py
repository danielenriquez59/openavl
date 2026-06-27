"""Tests for openavl.geom geometry construction (build_geometry)."""

from __future__ import annotations

import numpy as np
import pytest

from openavl.geom import build_geometry
from openavl.parser import parse_avl_file, prepare_model
from openavl.state import AVLState
from tests.helpers import GEOMETRIES_DIR, load_json_fixture

PLANE_AVL = GEOMETRIES_DIR / "plane.avl"
SUPRA_AVL = GEOMETRIES_DIR / "supra.avl"

pytestmark = pytest.mark.core


def _build_state(geo_path):
    if not geo_path.is_file():
        pytest.skip(f"geometry not found: {geo_path}")
    model = prepare_model(parse_avl_file(geo_path), base_dir=geo_path.parent)
    state = AVLState.from_model(model)
    build_geometry(state, model)
    return state, model


@pytest.mark.fixture
def test_plane_geometry_dimensions(fixtures_dir):
    """plane.avl lattice dimensions match JS buildGeometry checkpoint."""
    expected = load_json_fixture(fixtures_dir, "geometry_expected.json")["plane"]
    state, _ = _build_state(PLANE_AVL)
    assert state.nvor == expected["nvor"]
    assert state.nstrip == expected["nstrip"]
    assert state.nsurf == expected["nsurf"]
    assert state.nbody == expected["nbody"]
    assert state.nlnode == expected["nlnode"]


def test_plane_panel_coordinates_finite():
    """plane.avl vortex and control points are finite after build."""
    state, _ = _build_state(PLANE_AVL)
    assert state.nvor > 0
    assert np.all(np.isfinite(state.rv1[:, : state.nvor]))
    assert np.all(np.isfinite(state.rv2[:, : state.nvor]))
    assert np.all(np.isfinite(state.rc[:, : state.nvor]))
    assert np.all(np.isfinite(state.enc[:, : state.nvor]))


@pytest.mark.fixture
def test_supra_body_geometry(fixtures_dir):
    """supra.avl populates body line node arrays (MAKEBODY)."""
    expected = load_json_fixture(fixtures_dir, "geometry_expected.json")["supra"]
    state, _ = _build_state(SUPRA_AVL)
    assert state.nvor == expected["nvor"]
    assert state.nstrip == expected["nstrip"]
    assert state.nsurf == expected["nsurf"]
    assert state.nbody >= expected["nbody"]
    assert state.nlnode >= expected["nlnode"]
    assert state.nl[0] == expected["body_nl"]
    l0 = int(state.lfrst[0])
    assert np.all(np.isfinite(state.rl[:, l0 : l0 + state.nl[0]]))
    assert np.all(np.isfinite(state.radl[l0 : l0 + state.nl[0]]))
