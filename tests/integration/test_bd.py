"""End-to-end integration test for bd.avl (Bubble Dancer, with a fuselage body).

Reference values are Fortran AVL 3.52 through ``optvl`` 2.5.0, from the
geometries folder::

    from optvl import OVLSolver
    o = OVLSolver(geo_file="bd.avl")
    o.set_parameter("X cg", 3.4); o.set_parameter("Z cg", 0.5)
    o.set_parameter("CD0", 0.017)
    o.set_constraint("alpha", "CL", 0.7)
    o.set_constraint("beta", "Cl", 0.0)
    o.set_constraint("elevator", "Cm", 0.0)
    o.set_constraint("rudder", "Cn", 0.0)
    o.execute_run()

OpenAVL agrees to <= 5e-8. Turns are tested on Supra (test_supra.py), which
has a mass file; bd has none, and setting ``bank`` without a trim does not
change a plain run.
"""

from __future__ import annotations

import pytest

from openavl.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR

BD_AVL = GEOMETRIES_DIR / "bd.avl"
TOL = 1e-6
REF_ALPHA = 2.77598502
REF_ELEVATOR = -0.537245652
REF_CD = 0.0283123141

pytestmark = pytest.mark.integration


@pytest.mark.reference
def test_bd_level_flight():
    """Beta trimmed for zero roll moment, elevator and rudder trims; matches AVL."""
    solver = AVLSolver(BD_AVL, cl=0.7, cd0=0.017, rho=1.225, gravity=9.81, xcg=3.4, zcg=0.5)
    solver.set_constraint("beta", "cll", 0.0)
    solver.set_constraint("elevator", "cm", 0.0)
    solver.set_constraint("rudder", "cn", 0.0)
    solver.execute_run(max_iter=20)
    results = solver.get_results()

    assert results["converged"]
    assert results["alpha_deg"] == pytest.approx(REF_ALPHA, abs=TOL)
    assert results["beta_deg"] == pytest.approx(0.0, abs=TOL)
    assert results["control_deflections"]["elevator"] == pytest.approx(REF_ELEVATOR, abs=TOL)
    assert results["control_deflections"]["rudder"] == pytest.approx(0.0, abs=TOL)
    assert results["CD"] == pytest.approx(REF_CD, abs=TOL)
