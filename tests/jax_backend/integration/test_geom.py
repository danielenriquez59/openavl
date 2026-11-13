"""Tests for differentiable geometry update (geom_jax)."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pytest

from tests.jax_backend.require_jax import require_jax

jax = require_jax()

from openavl.jax.geom_jax import (
    design_params_from_state,
    snapshot_topology,
    update_geometry,
)
from openavl.jax.setup import compute_circulation
from openavl.jax.snapshot import snapshot_analysis_geometry, snapshot_flow, snapshot_refs
from openavl.core.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR

PLANE_AVL = GEOMETRIES_DIR / "plane.avl"
B737_AVL = GEOMETRIES_DIR / "b737.avl"
SUPRA_AVL = GEOMETRIES_DIR / "supra.avl"

PRIMAL_TOL = 1e-10

GEOMETRY_CASES = [
    pytest.param(PLANE_AVL, id="plane"),
    pytest.param(B737_AVL, id="b737"),
    pytest.param(SUPRA_AVL, id="supra"),
]


pytestmark = pytest.mark.integration


@cache
def _snapshot(avl_path: Path):
    """Solved model, its JAX snapshot, and a no-op geometry update (read-only)."""
    if not avl_path.is_file():
        pytest.skip(f"{avl_path.name} not found: {avl_path}")
    solver = AVLSolver(avl_path)
    solver.set_variable("alpha", 5.0)
    solver.set_variable("beta", 0.0)
    solver.execute_run(max_iter=1)
    state = solver.state
    baseline = snapshot_analysis_geometry(state)
    topo = snapshot_topology(state, solver.model)
    params = design_params_from_state(state, solver.model)
    return state, baseline, topo, update_geometry(topo, params, baseline)


@pytest.mark.parametrize("avl_path", GEOMETRY_CASES)
@pytest.mark.reference
def test_update_geometry_baseline_match(avl_path: Path) -> None:
    """Baseline design params reproduce snapshotted geometry arrays."""
    _, baseline, _, updated = _snapshot(avl_path)

    np.testing.assert_allclose(
        np.asarray(updated.circulation.enc),
        np.asarray(baseline.circulation.enc),
        atol=PRIMAL_TOL,
        rtol=PRIMAL_TOL,
    )
    np.testing.assert_allclose(
        np.asarray(updated.force.rv),
        np.asarray(baseline.force.rv),
        atol=1e-9,
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(updated.force.rc),
        np.asarray(baseline.force.rc),
        atol=1e-9,
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(updated.circulation.aicn),
        np.asarray(baseline.circulation.aicn),
        atol=1e-8,
        rtol=1e-8,
    )


@pytest.mark.reference
def test_update_geometry_nowake_kutta_rhs_matches_numpy() -> None:
    """A7: lvnc must stay False on Kutta/strip-off rows after a geometry update.

    ``b737.avl`` has a NOWAKE (``lfwake=False``) fuselage-fin surface, so its
    solved NumPy ``gam`` enforces Sigma-gamma=0 on the Kutta row. Running the
    same (no-op) geometry update through the JAX pipeline must reproduce that
    circulation exactly -- if ``lvnc`` is recomputed from strip degeneracy
    alone, the Kutta row's RHS is nonzero and gamma is wrong even though the
    AIC matrix rows are correctly rebuilt.
    """
    state, _, topo, updated = _snapshot(B737_AVL)
    assert topo.kutta_iv.shape[0] > 0, "expected b737.avl to exercise a NOWAKE Kutta row"

    gamma_jax = compute_circulation(updated.circulation, snapshot_flow(state), snapshot_refs(state))

    np.testing.assert_allclose(
        np.asarray(gamma_jax),
        np.asarray(state.gam[: state.nvor]),
        atol=1e-8,
        rtol=1e-8,
    )
