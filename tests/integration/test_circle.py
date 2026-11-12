"""End-to-end integration test for circle.avl (circular planform wing).

Reference values are Fortran AVL 3.52 through ``optvl`` 2.5.0 at alpha =
5.72958 deg (0.1 rad), from the geometries folder::

    from optvl import OVLSolver
    o = OVLSolver(geo_file="circle.avl")
    o.set_constraint("alpha", "alpha", 5.72958); o.execute_run()
    o.get_total_forces()  # CL, CD (near-field), CDff (Trefftz), e

OpenAVL agrees to <= 2e-9. CL is linear in alpha here, so one point also pins
the lift slope.
"""

from __future__ import annotations

import pytest

from openavl.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR

CIRCLE_AVL = GEOMETRIES_DIR / "circle.avl"
TOL = 1e-7
REF = {"CL": 0.18542532, "CD": 0.00759339806, "CDFF": 0.00823106896, "SPANEF": 1.05289099}

pytestmark = pytest.mark.integration


@pytest.mark.reference
def test_circle_alpha_direct():
    """Alpha fixed at 0.1 rad; lift, near-field and Trefftz drag match AVL."""
    solver = AVLSolver(CIRCLE_AVL)
    solver.set_constraint("alpha", "alpha", 5.72958)
    solver.execute_run(max_iter=20)
    results = solver.get_results()

    assert results["converged"]
    for key, ref in REF.items():
        assert results[key] == pytest.approx(ref, abs=TOL), key
