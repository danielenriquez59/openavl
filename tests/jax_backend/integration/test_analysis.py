"""Primal force comparison: JAX run_analysis vs NumPy solver (Phase 4A)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.jax_backend.require_jax import require_jax

require_jax()

from openavl.jax.analysis import make_run_analysis_jit, run_analysis
from openavl.jax.solver import JaxAVLSolver
from openavl.jax.snapshot import snapshot_analysis_geometry, snapshot_flow, snapshot_refs
from openavl.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR

PLANE_AVL = GEOMETRIES_DIR / "plane.avl"
B737_AVL = GEOMETRIES_DIR / "b737.avl"
SUPRA_AVL = GEOMETRIES_DIR / "supra.avl"
PRIMAL_TOL = 1e-10

GEOMETRY_CASES = [
    pytest.param(PLANE_AVL, {"alpha": 5.0, "beta": 0.0}, id="plane"),
    pytest.param(B737_AVL, {"alpha": 3.0, "beta": 0.0}, id="b737"),
    pytest.param(SUPRA_AVL, {"alpha": 2.0, "beta": 0.0}, id="supra"),
]


pytestmark = pytest.mark.integration


def _run_numpy_solver(avl_path: Path, **kwargs) -> AVLSolver:
    """Build, configure, and execute the NumPy reference solver."""
    if not avl_path.is_file():
        pytest.skip(f"{avl_path.name} not found: {avl_path}")
    solver = AVLSolver(avl_path)
    if "alpha" in kwargs:
        solver.set_variable("alpha", kwargs["alpha"])
    if "beta" in kwargs:
        solver.set_variable("beta", kwargs["beta"])
    solver.execute_run(max_iter=1)
    return solver


def _assert_primal_match(solver: AVLSolver) -> None:
    """Compare JAX and NumPy total force coefficients."""
    state = solver.state
    geom = snapshot_analysis_geometry(state)
    flow = snapshot_flow(state)
    refs = snapshot_refs(state)

    result = run_analysis(flow, geom, refs)
    result_jit = make_run_analysis_jit(geom, refs)(flow)

    assert float(result.CL) == pytest.approx(state.cltot, abs=PRIMAL_TOL, rel=PRIMAL_TOL)
    assert float(result.CD) == pytest.approx(state.cdtot, abs=PRIMAL_TOL, rel=PRIMAL_TOL)
    assert float(result.CY) == pytest.approx(state.cytot, abs=PRIMAL_TOL, rel=PRIMAL_TOL)
    np.testing.assert_allclose(
        np.asarray(result.CM),
        np.asarray(state.cmtot),
        atol=PRIMAL_TOL,
        rtol=PRIMAL_TOL,
    )

    assert float(result_jit.CL) == pytest.approx(float(result.CL), abs=PRIMAL_TOL, rel=PRIMAL_TOL)
    assert float(result_jit.CD) == pytest.approx(float(result.CD), abs=PRIMAL_TOL, rel=PRIMAL_TOL)
    assert float(result_jit.CY) == pytest.approx(float(result.CY), abs=PRIMAL_TOL, rel=PRIMAL_TOL)
    np.testing.assert_allclose(
        np.asarray(result_jit.CM),
        np.asarray(result.CM),
        atol=PRIMAL_TOL,
        rtol=PRIMAL_TOL,
    )


@pytest.mark.parametrize("avl_path, kwargs", GEOMETRY_CASES)
@pytest.mark.reference
def test_run_analysis_matches_numpy(avl_path: Path, kwargs: dict) -> None:
    """JAX primal forces match NumPy cltot/cdtot/cmtot."""
    solver = _run_numpy_solver(avl_path, **kwargs)
    _assert_primal_match(solver)


@pytest.mark.smoke
def test_jax_solver_run_dispatches_jit_and_eager() -> None:
    """High-level JAX solver accepts both bound JIT and eager runners."""
    if not PLANE_AVL.is_file():
        pytest.skip(f"{PLANE_AVL.name} not found: {PLANE_AVL}")

    jit_solver = JaxAVLSolver(PLANE_AVL, use_jit=True)
    eager_solver = JaxAVLSolver(PLANE_AVL, use_jit=False)

    jit_result = jit_solver.run()
    eager_result = eager_solver.run()

    assert float(jit_result.CL) == pytest.approx(float(eager_result.CL), abs=PRIMAL_TOL)
    assert float(jit_result.CD) == pytest.approx(float(eager_result.CD), abs=PRIMAL_TOL)
    assert float(jit_result.CY) == pytest.approx(float(eager_result.CY), abs=PRIMAL_TOL)
    np.testing.assert_allclose(np.asarray(jit_result.CM), np.asarray(eager_result.CM), atol=PRIMAL_TOL)
