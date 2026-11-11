"""End-to-end integration test for vanilla.avl (sideslip sweep).

Reference values
----------------
The trimmed alpha and control deflections below come from Fortran AVL 3.52,
run through the ``optvl`` 2.5.0 Python package with default flags (trailing-leg
forces off) on the same ``vanilla.avl`` geometry (byte-identical), with
xcg = 0.65, CD0 = 0, and constraints alpha -> CL = 1.16647, beta fixed,
aileron -> Cl = 0, elevator -> Cm = 0, rudder -> Cn = 0.

To regenerate (optvl is deliberately not a test dependency)::

    from optvl import OVLSolver
    o = OVLSolver(geo_file="vanilla.avl")
    o.set_parameter("X cg", 0.65); o.set_parameter("CD0", 0.0)
    o.set_constraint("alpha", "CL", 1.16647)
    o.set_constraint("beta", "beta", beta)
    o.set_constraint("aileron", "Cl", 0.0)
    o.set_constraint("elevator", "Cm", 0.0)
    o.set_constraint("rudder", "Cn", 0.0)
    o.execute_run()
    o.get_variable("alpha"), o.get_control_deflections()

Do NOT use the solved values in ``avl/runs/vanilla.run``: they were produced
by an AVL older than 3.22, which applied beta before alpha in the freestream
vector (see the 3.22 entry in ``avl/version_notes.txt``) and had trailing-leg
forces on, so they differ from current AVL by up to ~0.7 deg in alpha at
high sideslip.
"""

from __future__ import annotations

import pytest

from openavl.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR

VANILLA_AVL = GEOMETRIES_DIR / "vanilla.avl"

REF_CL = 1.16647

# Fortran AVL 3.52 (optvl 2.5.0) trimmed state, degrees, keyed by beta.
REF_ALPHA_BY_BETA = {
    0: 8.253582,
    5: 8.356199,
    10: 8.670873,
    15: 9.219117,
    20: 10.040611,
}
REF_DEFLECTIONS_BY_BETA = {
    0: {"aileron": 0.000000, "elevator": 8.498724, "rudder": 0.000000},
    5: {"aileron": 1.840805, "elevator": 8.469440, "rudder": -0.232146},
    10: {"aileron": 3.720156, "elevator": 8.375706, "rudder": -0.221715},
    15: {"aileron": 5.680590, "elevator": 8.198203, "rudder": 0.316921},
    20: {"aileron": 7.773691, "elevator": 7.898401, "rudder": 1.769213},
}

# References are stored to 6 decimals (rounding error <= 5e-7 deg); OpenAVL
# agrees with unrounded AVL 3.52 to <= 1.5e-7 deg for alpha and all
# deflections. 1e-5 deg leaves roughly an order of magnitude of margin.
TOL_DEG = 1e-5


def build_vanilla_solver(beta: float) -> AVLSolver:
    """Build vanilla solver at the given sideslip angle."""
    if not VANILLA_AVL.is_file():
        pytest.skip(f"vanilla.avl not found: {VANILLA_AVL}")

    solver = AVLSolver(
        VANILLA_AVL,
        cl=REF_CL,
        vel=30.8633,
        rho=0.2,
        gravity=10.0,
        xcg=0.65,
        cd0=0.0,
        beta=beta,
    )
    # alpha->CL, fixed beta, aileron/elevator/rudder moment trims
    solver.set_constraint("aileron", "cll", 0.0)
    solver.set_constraint("elevator", "cm", 0.0)
    solver.set_constraint("rudder", "cn", 0.0)
    return solver


pytestmark = pytest.mark.integration


@pytest.mark.reference
@pytest.mark.parametrize("beta", [0, 5, 10, 15, 20], ids=[f"beta_{b}" for b in [0, 5, 10, 15, 20]])
def test_vanilla_beta_sweep(beta: float):
    """Sideslip sweep at fixed CL; trimmed alpha and deflections match AVL 3.52."""
    solver = build_vanilla_solver(beta)
    solver.execute_run(max_iter=20)
    results = solver.get_results()

    # CL is the trim target, so convergence is the meaningful check here.
    assert results["converged"]
    assert results["beta_deg"] == pytest.approx(beta, abs=1e-12)
    assert results["alpha_deg"] == pytest.approx(REF_ALPHA_BY_BETA[beta], abs=TOL_DEG)

    deflections = results["control_deflections"]
    for name, ref in REF_DEFLECTIONS_BY_BETA[beta].items():
        assert deflections[name] == pytest.approx(ref, abs=TOL_DEG), name
