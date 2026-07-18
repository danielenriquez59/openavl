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
