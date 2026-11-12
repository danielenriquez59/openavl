"""JAX viscous drag against Fortran AVL, and body-only models against NumPy."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.jax_backend.require_jax import require_jax

require_jax()

from openavl.jax.solver import JaxAVLSolver
from openavl.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR

pytestmark = pytest.mark.integration

CDCL_AVL = GEOMETRIES_DIR / "testcdcl_surface.avl"
BODY_ONLY_AVL = GEOMETRIES_DIR / "supraf.avl"

# Fortran AVL 3.52 (optvl 2.5.0) on testcdcl_surface.avl (CDCL polar, flat
# wing): alpha 8, beta 6, stability-axis pb/2V 0.05, rb/2V -0.03.
CASE = {"alpha": 8.0, "beta": 6.0, "pb/2V": 0.05, "rb/2V": -0.03}
FORTRAN_CD = 0.024710477320


def _solve(solver: AVLSolver, case: dict[str, float]) -> None:
    for name, value in case.items():
        solver.set_constraint(name, name, value)
    solver.execute_run(max_iter=20)
    assert solver.state.lsol


@pytest.mark.reference
def test_jax_solver_viscous_drag_matches_fortran() -> None:
    """Viscous drag is on for a CDCL model and uses the strip lift direction.

    The jitted path only; the eager branch of JaxAVLSolver.run runs the same
    analysis and is covered by the body-only test below.
    """
    solver = JaxAVLSolver(CDCL_AVL)
    _solve(solver.numpy_solver, CASE)
    assert float(solver.run().CD) == pytest.approx(FORTRAN_CD, abs=1e-9)


@pytest.mark.reference
def test_openmdao_components_viscous_drag_match_fortran() -> None:
    """Both OpenMDAO entry points include the CDCL drag (stability-axis rates)."""
    om = pytest.importorskip("openmdao.api")
    from openavl.jax.openmdao import JaxAVLComp
    from openavl.jax.openmdao_group import OpenAVLGroup

    flat = om.Problem()
    flat.model.add_subsystem("avl", JaxAVLComp(geo_file=str(CDCL_AVL)))
    flat.setup()
    flat.set_val("avl.alpha", math.radians(CASE["alpha"]))
    flat.set_val("avl.beta", math.radians(CASE["beta"]))
    flat.set_val("avl.pb2v", CASE["pb/2V"])
    flat.set_val("avl.rb2v", CASE["rb/2V"])
    flat.run_model()

    group = om.Problem()
    group.model = OpenAVLGroup(geo_file=str(CDCL_AVL))
    group.setup()
    group.set_val("alpha", CASE["alpha"])
    group.set_val("beta", CASE["beta"])
    group.set_val("pb2v", CASE["pb/2V"])
    group.set_val("rb2v", CASE["rb/2V"])
    group.run_model()

    for cd in (flat.get_val("avl.CD"), group.get_val("CD")):
        assert float(np.asarray(cd).item()) == pytest.approx(FORTRAN_CD, abs=1e-9)


@pytest.mark.reference
@pytest.mark.parametrize("use_jit", [True, False])
def test_jax_solver_body_only_model_matches_numpy(use_jit: bool) -> None:
    """A model with bodies and no surfaces runs, and its gradient is right.

    Fortran AVL cannot solve a model without vortices, so NumPy is the reference:
    its totals and a central finite difference of its CL.
    """
    solver = JaxAVLSolver(BODY_ONLY_AVL, use_jit=use_jit)
    state = solver.numpy_solver.state
    assert state.nsurf == 0 and state.nbody > 0
    _solve(solver.numpy_solver, {"alpha": 4.0})

    result = solver.run()
    assert float(result.CL) == pytest.approx(state.cltot, abs=1e-12)
    assert float(result.CD) == pytest.approx(state.cdtot, abs=1e-12)
    np.testing.assert_allclose(np.asarray(result.CM), state.cmtot, atol=1e-12)

    h = 0.01
    cl = []
    for alpha in (4.0 - h, 4.0 + h):
        numpy_solver = AVLSolver(BODY_ONLY_AVL)
        _solve(numpy_solver, {"alpha": alpha})
        cl.append(numpy_solver.state.cltot)
    cl_alpha = (cl[1] - cl[0]) / math.radians(2 * h)
    assert float(solver.grad("CL").alfa) == pytest.approx(cl_alpha, rel=1e-6)
