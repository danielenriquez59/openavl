"""Tests for AFIL dependency handling in the web session."""

from __future__ import annotations

from pathlib import Path

import pytest

from openavl.web.session import (
    SessionState,
    add_airfoil_dependency,
    build_afil_dependencies,
    list_airfoil_dependencies,
    upload_airfoil,
)

from tests.helpers import GEOMETRIES_DIR

SUPRA_AVL = GEOMETRIES_DIR / "supra.avl"
AG40D = GEOMETRIES_DIR / "ag40d.dat"

pytestmark = pytest.mark.ui


@pytest.mark.skipif(not SUPRA_AVL.is_file(), reason="supra.avl not found")
def test_list_airfoil_dependencies_from_supra():
    """Supra AVL references external AFIL coordinate files."""
    text = SUPRA_AVL.read_text(encoding="utf-8", errors="replace")
    deps = list_airfoil_dependencies(text)
    assert "ag40d.dat" in deps
    assert "ag41d.dat" in deps


@pytest.mark.skipif(not SUPRA_AVL.is_file(), reason="supra.avl not found")
def test_build_afil_dependencies_ready_for_example_base_dir():
    """Example geometry directory resolves bundled airfoil files."""
    text = SUPRA_AVL.read_text(encoding="utf-8", errors="replace")
    session = SessionState(session_id="test")
    session.pending_avl_text = text
    session.airfoil_base_dir = SUPRA_AVL.parent
    rows = build_afil_dependencies(session)
    assert rows
    assert all(row["status"] == "ready" for row in rows)


@pytest.mark.skipif(not SUPRA_AVL.is_file() or not AG40D.is_file(), reason="supra assets not found")
def test_upload_airfoil_rebuilds_solver():
    """Uploaded airfoil text satisfies AFIL dependencies without a base directory."""
    text = SUPRA_AVL.read_text(encoding="utf-8", errors="replace")
    session = SessionState(session_id="test")
    session.pending_avl_text = text
    session.airfoil_base_dir = None

    response = upload_airfoil(session, "ag40d.dat", AG40D.read_text(encoding="utf-8", errors="replace"))
    assert response["type"] == "model_loaded"
    assert session.solver is not None
    rows = build_afil_dependencies(session)
    ag40 = next(row for row in rows if row["path"] == "ag40d.dat")
    assert ag40["status"] == "ready"


@pytest.mark.skipif(not SUPRA_AVL.is_file(), reason="supra.avl not found")
def test_add_manual_airfoil_dependency():
    """Users can register AFIL paths that are not present in the AVL text."""
    text = SUPRA_AVL.read_text(encoding="utf-8", errors="replace")
    session = SessionState(session_id="test")
    session.pending_avl_text = text
    session.airfoil_base_dir = None

    response = add_airfoil_dependency(session, "custom.dat")
    assert response["type"] == "model_loaded"
    rows = build_afil_dependencies(session)
    manual = next(row for row in rows if row["path"] == "custom.dat")
    assert manual["manual"] is True
    assert manual["status"] == "missing"


def test_upload_airfoil_invalid_text_marks_invalid():
    """Invalid coordinate text is rejected and tracked as invalid."""
    session = SessionState(session_id="test")
    session.pending_avl_text = "Test\n0 0 0\n"
    with pytest.raises(ValueError, match="Could not parse"):
        upload_airfoil(session, "bad.dat", "not coordinates\n")
    assert "bad.dat" in session.invalid_airfoil_paths
