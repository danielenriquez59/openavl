"""Tests for aero-mesh CAD export (ASCII STL)."""

from __future__ import annotations

import pytest

from openavl.core.solver import AVLSolver
from openavl.fileio.cad_export import sanitize_stl_name

from tests.helpers import GEOMETRIES_DIR

SUPRA_AVL = GEOMETRIES_DIR / "supra.avl"


@pytest.mark.skipif(not SUPRA_AVL.is_file(), reason="supra.avl not found")
def test_export_stl_writes_surfaces_and_bodies(tmp_path):
    """Supra export includes named surface and body solids with triangle facets."""
    solver = AVLSolver(str(SUPRA_AVL), base_dir=SUPRA_AVL.parent)
    out = tmp_path / "supra_aero_mesh.stl"

    result = solver.export_stl(out)

    assert result == out
    assert out.is_file()
    text = out.read_text(encoding="utf-8")

    assert "solid " in text
    assert "endsolid " in text
    assert "facet normal" in text
    assert "vertex " in text

    # Lifting surfaces and fuselage body are present as named solids.
    assert sanitize_stl_name("Inner Wing") in text
    assert sanitize_stl_name("Fuse pod") in text

    n_facets = text.count("endfacet")
    n_vertices = text.count("vertex ")
    assert n_facets > 100
    assert n_vertices == 3 * n_facets


@pytest.mark.skipif(not SUPRA_AVL.is_file(), reason="supra.avl not found")
def test_export_stl_can_omit_bodies(tmp_path):
    """include_bodies=False omits fuselage solids from the STL."""
    solver = AVLSolver(str(SUPRA_AVL), base_dir=SUPRA_AVL.parent)
    out = tmp_path / "supra_surfaces_only.stl"

    solver.export_stl(out, include_bodies=False)
    text = out.read_text(encoding="utf-8")

    assert sanitize_stl_name("Fuse pod") not in text
    assert sanitize_stl_name("Inner Wing") in text


def test_sanitize_stl_name():
    """STL solid names replace spaces and drop odd characters."""
    assert sanitize_stl_name("Fuse pod") == "Fuse_pod"
    assert sanitize_stl_name("  Wing #1  ") == "Wing_1"
    assert sanitize_stl_name("!!!") == "solid"
