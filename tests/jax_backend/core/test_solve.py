"""Tests for JAX circulation solve pipeline (Phases 3A-3C)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.jax_backend.require_jax import require_jax

jax = require_jax()
import jax.numpy as jnp

from openavl.jax.setup import compute_circulation, compute_velocities
from openavl.jax.snapshot import snapshot_flow, snapshot_circulation_geometry, snapshot_refs
from openavl.jax.solve import solve_circulation
from openavl.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR

PLANE_AVL = GEOMETRIES_DIR / "plane.avl"
GAMMA_TOL = 1e-10
VJP_TOL = 1e-9
FD_STEP = 1e-7

pytestmark = pytest.mark.core


def _finite_difference_grad(
    fun,
    x: jnp.ndarray,
    eps: float = FD_STEP,
) -> jnp.ndarray:
    """Central finite-difference gradient for a scalar function of a vector."""
    x_np = np.asarray(x, dtype=np.float64)
    grad = np.zeros_like(x_np)
    for i in range(x_np.size):
        xp = x_np.copy()
        xm = x_np.copy()
        xp[i] += eps
        xm[i] -= eps
        grad[i] = (float(fun(jnp.asarray(xp))) - float(fun(jnp.asarray(xm)))) / (2.0 * eps)
    return jnp.asarray(grad)


@pytest.fixture(scope="module")
def plane_state():
    """Built and solved ``plane.avl`` state for circulation validation."""
    if not PLANE_AVL.is_file():
        pytest.skip(f"plane.avl not found: {PLANE_AVL}")
    solver = AVLSolver(PLANE_AVL, alpha=5.0, beta=0.0)
    solver.execute_run(max_iter=1)
    return solver.state


@pytest.mark.reference
def test_solve_circulation_custom_vjp_matches_finite_differences():
    """Custom VJP backward pass agrees with central finite differences."""
    rng = np.random.default_rng(42)
    n = 8
    a = rng.standard_normal((n, n))
    aicn = jnp.asarray(a @ a.T + n * np.eye(n), dtype=jnp.float64)
    rhs = jnp.asarray(rng.standard_normal(n), dtype=jnp.float64)

    def objective(r):
        return jnp.sum(solve_circulation(aicn, r))

    grad_jax = jax.grad(objective)(rhs)
    grad_fd = _finite_difference_grad(objective, rhs)
    assert np.allclose(np.asarray(grad_jax), np.asarray(grad_fd), atol=VJP_TOL, rtol=VJP_TOL)


@pytest.mark.reference
def test_compute_circulation_matches_numpy_solver(plane_state):
    """End-to-end JAX circulation matches NumPy ``state.gam``."""
    geom = snapshot_circulation_geometry(plane_state)
    flow = snapshot_flow(plane_state)
    refs = snapshot_refs(plane_state)

    gamma_jax = compute_circulation(geom, flow, refs)
    gamma_np = plane_state.gam[: plane_state.nvor]

    assert np.allclose(
        np.asarray(gamma_jax),
        np.asarray(gamma_np),
        atol=GAMMA_TOL,
        rtol=GAMMA_TOL,
    )


@pytest.mark.reference
def test_compute_velocities_matches_numpy_velsum(plane_state):
    """JAX induced velocities match NumPy ``velsum`` output."""
    geom = snapshot_circulation_geometry(plane_state)
    flow = snapshot_flow(plane_state)
    refs = snapshot_refs(plane_state)

    gamma = compute_circulation(geom, flow, refs)
    vc_jax, vv_jax = compute_velocities(geom, gamma)

    vc_np = plane_state.vc[:, : plane_state.nvor]
    vv_np = plane_state.vv[:, : plane_state.nvor]

    assert np.allclose(np.asarray(vc_jax), np.asarray(vc_np), atol=GAMMA_TOL, rtol=GAMMA_TOL)
    assert np.allclose(np.asarray(vv_jax), np.asarray(vv_np), atol=GAMMA_TOL, rtol=GAMMA_TOL)
