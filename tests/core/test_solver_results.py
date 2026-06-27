"""Tests for AVLSolver result payloads."""

from __future__ import annotations

import pytest

from openavl.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR

SUPRA_AVL = GEOMETRIES_DIR / "supra.avl"

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
