"""Shared fixtures for JAX tests."""

from __future__ import annotations

import pytest

from openavl.core.setup import setup
from openavl.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR

PLANE_AVL = GEOMETRIES_DIR / "plane.avl"


@pytest.fixture
def plane_state():
    """Built and set-up AVLState for plane.avl."""
    if not PLANE_AVL.is_file():
        pytest.skip(f"plane.avl not found: {PLANE_AVL}")
    solver = AVLSolver(PLANE_AVL)
    setup(solver.state)
    return solver.state
