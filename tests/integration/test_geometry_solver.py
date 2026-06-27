"""Solver parity: Geometry API vs file-based AVLSolver."""

from __future__ import annotations

import pytest

from openavl import AVLSolver

from tests.helpers import GEOMETRIES_DIR
from tests.core.test_geometry_convert import build_plane_aircraft

PLANE_AVL = GEOMETRIES_DIR / "plane.avl"

pytestmark = pytest.mark.integration


def _run_solver(geo, **kwargs):
    solver = AVLSolver(geo, **kwargs)
    solver.execute_run(max_iter=20)
    return solver.get_results()


@pytest.mark.parametrize("alpha", [0.0, 5.0, -2.5])
@pytest.mark.reference
def test_plane_solver_parity(alpha: float):
    assert PLANE_AVL.is_file(), f"missing test geometry: {PLANE_AVL}"
    options = dict(alpha=alpha, beta=0.0, mach=0.0)
    file_results = _run_solver(PLANE_AVL, **options)
    api_results = _run_solver(build_plane_aircraft(), **options)

    assert file_results["geometry"]["NVOR"] == api_results["geometry"]["NVOR"]
    assert file_results["geometry"]["NSTRIP"] == api_results["geometry"]["NSTRIP"]
    assert file_results["CL"] == pytest.approx(api_results["CL"], abs=1e-10)
    assert file_results["CD"] == pytest.approx(api_results["CD"], abs=1e-10)
    assert file_results["CY"] == pytest.approx(api_results["CY"], abs=1e-10)
    assert file_results["Cl"] == pytest.approx(api_results["Cl"], abs=1e-10)
    assert file_results["Cm"] == pytest.approx(api_results["Cm"], abs=1e-10)
    assert file_results["Cn"] == pytest.approx(api_results["Cn"], abs=1e-10)
