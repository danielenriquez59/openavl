"""End-to-end integration tests for supra.avl (sailplane with a fuselage body).

Reference values are Fortran AVL 3.52 through ``optvl`` 2.5.0. To regenerate
(optvl is deliberately not a test dependency), from the geometries folder::

    from optvl import OVLSolver
    o = OVLSolver(geo_file="supra.avl")
    o.set_parameter("X cg", 3.75); o.set_parameter("CD0", 0.015)
    o.set_constraint("beta", "beta", 0.0)
    o.set_constraint("aileron", "Cl", 0.0)
    o.set_constraint("elevator", "Cm", 0.0)
    o.set_constraint("rudder", "Cn", 0.0)
    o.set_constraint("alpha", "CL", cl); o.execute_run()
    o.get_variable("alpha"), o.get_control_deflections(), o.get_total_forces()["CD"]

and for the level turn (AVL's C1 trim)::

    o = OVLSolver(geo_file="supra.avl", mass_file="../mass/supra.mass")
    o.set_trim_condition("bankAng", 30.0); o.set_trim_condition("CL", 0.6)
    o.execute_run()
    o.get_parameter("velocity"), o.get_variable("alpha"), o.get_total_forces()

OpenAVL agrees to <= 4e-8 in every value below. Do not use the solved values
in ``avl/runs/supra.run``: they come from an older AVL and are up to 0.06 deg
off in alpha.
"""

from __future__ import annotations

import pytest

from openavl import constants as C
from openavl.core.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR

SUPRA_AVL = GEOMETRIES_DIR / "supra.avl"
SUPRA_MASS = GEOMETRIES_DIR.parent / "mass" / "supra.mass"
TOL = 1e-6

# CL -> (alpha deg, aileron deg, elevator deg, rudder deg, CD)
REF_BY_CL = {
    0.7: (3.6929770, 0.0, -0.7408409, 0.0, 0.023965902),
    0.35: (0.2327085, 0.0, 0.2346892, 0.0, 0.017240659),
    0.1: (-2.2284572, 0.0, 0.8965646, 0.0, 0.015247751),
}

# 30 deg level turn at CL = 0.6, AVL sign convention for the moments.
REF_TURN = {
    "V": 7.92086166,
    "alpha_deg": 2.29958883,
    "CD": 0.0201709973,
    "CY": 0.02298269,
    "Cl": 0.019050749,
    "Cm": -0.0853177683,
    "Cn": -0.00473543883,
}

pytestmark = pytest.mark.integration


@pytest.mark.reference
@pytest.mark.parametrize("cl", list(REF_BY_CL), ids=[f"cl_{c}" for c in REF_BY_CL])
def test_supra_cl_sweep(cl: float):
    """Alpha trimmed to CL with moment trims; alpha, deflections and CD match AVL."""
    solver = AVLSolver(SUPRA_AVL, cd0=0.015, rho=1.225, gravity=9.81, xcg=3.75)
    solver.set_constraint("beta", "beta", 0.0)
    solver.set_constraint("aileron", "cll", 0.0)
    solver.set_constraint("elevator", "cm", 0.0)
    solver.set_constraint("rudder", "cn", 0.0)
    solver.set_constraint("alpha", "cl", cl)
    solver.execute_run(max_iter=20)
    results = solver.get_results()

    assert results["converged"]
    alpha, aileron, elevator, rudder, cd = REF_BY_CL[cl]
    assert results["alpha_deg"] == pytest.approx(alpha, abs=TOL)
    deflections = results["control_deflections"]
    assert deflections["aileron"] == pytest.approx(aileron, abs=TOL)
    assert deflections["elevator"] == pytest.approx(elevator, abs=TOL)
    assert deflections["rudder"] == pytest.approx(rudder, abs=TOL)
    assert results["CD"] == pytest.approx(cd, abs=TOL)


@pytest.mark.reference
def test_supra_level_turn_matches_avl():
    """30 deg level-turn trim: speed, alpha, rates and lateral loads match AVL C1."""
    solver = AVLSolver(SUPRA_AVL, SUPRA_MASS)
    solver.set_parameter("bank", 30.0)
    solver.setup_trim(1)
    solver.execute_run(max_iter=20)
    results = solver.get_results()

    assert results["converged"]
    assert results["CL"] == pytest.approx(0.6, abs=TOL)
    assert float(solver.state.parval[C.IPVEE, 0]) == pytest.approx(REF_TURN["V"], abs=TOL)
    for key in ("alpha_deg", "CD", "CY", "Cl", "Cm", "Cn"):
        assert results[key] == pytest.approx(REF_TURN[key], abs=TOL), key
