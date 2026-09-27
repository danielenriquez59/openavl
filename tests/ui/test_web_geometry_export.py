"""Tests for aerodynamic panel-mesh data exported to the web viewer."""

from __future__ import annotations

import pytest

from openavl.core.solver import AVLSolver
from openavl.web.geometry_export import model_to_geometry

from tests.helpers import GEOMETRIES_DIR

SUPRA_AVL = GEOMETRIES_DIR / "supra.avl"

pytestmark = pytest.mark.ui


@pytest.mark.skipif(not SUPRA_AVL.is_file(), reason="supra.avl not found")
def test_surface_export_includes_explicit_panel_edges():
    """Each aerodynamic surface exports chordwise and spanwise panel loops."""
    solver = AVLSolver(str(SUPRA_AVL), base_dir=SUPRA_AVL.parent)
    geometry = model_to_geometry(solver.model, solver.state)

    assert geometry["surfaces"]
    for surface in geometry["surfaces"]:
        panel_lines = surface["panel_lines"]
        assert panel_lines
        assert len(panel_lines) % 24 == 0


@pytest.mark.skipif(not SUPRA_AVL.is_file(), reason="supra.avl not found")
def test_geometry_export_includes_section_airfoil_labels():
    """AFIL section labels use filenames at scaled/translated leading edges."""
    solver = AVLSolver(str(SUPRA_AVL), base_dir=SUPRA_AVL.parent)
    geometry = model_to_geometry(solver.model, solver.state)

    labels = geometry["section_labels"]
    assert labels
    names = {entry["name"] for entry in labels}
    assert "ag40d.dat" in names
    assert "ag41d.dat" in names
    for entry in labels:
        assert "/" not in entry["name"] and "\\" not in entry["name"]
        assert all(isinstance(entry[key], float) for key in ("x", "y", "z"))


@pytest.mark.skipif(not SUPRA_AVL.is_file(), reason="supra.avl not found")
def test_geometry_export_includes_control_hinges():
    """Control hinge polylines are exported with at least two XYZ points each."""
    solver = AVLSolver(str(SUPRA_AVL), base_dir=SUPRA_AVL.parent)
    geometry = model_to_geometry(solver.model, solver.state)

    hinges = geometry["hinges"]
    assert hinges
    names = {entry["name"] for entry in hinges}
    assert "flap" in names
    assert "aileron" in names
    assert "elevator" in names
    for entry in hinges:
        assert len(entry["positions"]) >= 6
        assert len(entry["positions"]) % 3 == 0
