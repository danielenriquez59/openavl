"""Tests for differentiable geometry update (geom_jax)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.jax_backend.require_jax import require_jax

jax = require_jax()
import jax.numpy as jnp

from openavl.jax.geom_jax import (
    design_params_from_state,
    run_analysis_with_geometry,
    snapshot_topology,
    update_geometry,
)
from openavl.jax.setup import compute_circulation
from openavl.jax.snapshot import snapshot_analysis_geometry, snapshot_flow, snapshot_refs
from openavl.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR

PLANE_AVL = GEOMETRIES_DIR / "plane.avl"
B737_AVL = GEOMETRIES_DIR / "b737.avl"
SUPRA_AVL = GEOMETRIES_DIR / "supra.avl"

PRIMAL_TOL = 1e-10
FD_TOL = 1e-4
FD_STEP = 1e-6

GEOMETRY_CASES = [
    pytest.param(PLANE_AVL, id="plane"),
    pytest.param(B737_AVL, id="b737"),
    pytest.param(SUPRA_AVL, id="supra"),
]


pytestmark = pytest.mark.integration


def _build_solver(avl_path: Path) -> AVLSolver:
    if not avl_path.is_file():
        pytest.skip(f"{avl_path.name} not found: {avl_path}")
    solver = AVLSolver(avl_path)
    solver.set_variable("alpha", 5.0)
    solver.set_variable("beta", 0.0)
    solver.execute_run(max_iter=1)
    return solver


@pytest.mark.parametrize("avl_path", GEOMETRY_CASES)
@pytest.mark.reference
def test_update_geometry_baseline_match(avl_path: Path) -> None:
    """Baseline design params reproduce snapshotted geometry arrays."""
    solver = _build_solver(avl_path)
    state = solver.state
    baseline = snapshot_analysis_geometry(state)
    topo = snapshot_topology(state, solver.model)
    params = design_params_from_state(state, solver.model)

    updated = update_geometry(topo, params, baseline)

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
    solver = _build_solver(B737_AVL)
    state = solver.state
    baseline = snapshot_analysis_geometry(state)
    topo = snapshot_topology(state, solver.model)
    params = design_params_from_state(state, solver.model)
    flow = snapshot_flow(state)
    refs = snapshot_refs(state)

    assert topo.kutta_iv.shape[0] > 0, "expected b737.avl to exercise a NOWAKE Kutta row"

    updated = update_geometry(topo, params, baseline)
    gamma_jax = compute_circulation(updated.circulation, flow, refs)

    np.testing.assert_allclose(
        np.asarray(gamma_jax),
        np.asarray(state.gam[: state.nvor]),
        atol=1e-8,
        rtol=1e-8,
    )


def _central_diff_params(
    flow,
    params,
    topo,
    baseline,
    refs,
    field: str,
    index: int,
    eps: float = FD_STEP,
) -> float:
    """Central FD of CL w.r.t. one design parameter component."""

    def cl_from_params(p):
        result = run_analysis_with_geometry(flow, p, topo, baseline, refs)
        return float(result.CL)

    def bump(delta: float):
        if field == "aincs":
            return params._replace(aincs=params.aincs.at[index].add(delta))
        if field == "chords":
            return params._replace(chords=params.chords.at[index].add(delta))
        if field == "xles":
            return params._replace(xles=params.xles.at[index].add(delta))
        raise KeyError(field)

    return (cl_from_params(bump(eps)) - cl_from_params(bump(-eps))) / (2.0 * eps)


@pytest.mark.parametrize("avl_path", [PLANE_AVL, SUPRA_AVL])
@pytest.mark.reference
def test_geometry_derivatives_fd(avl_path: Path) -> None:
    """jax.jacrev of run_analysis_with_geometry matches central FD for geometry DVs."""
    solver = _build_solver(avl_path)
    state = solver.state
    baseline = snapshot_analysis_geometry(state)
    topo = snapshot_topology(state, solver.model)
    params = design_params_from_state(state, solver.model)
    flow = snapshot_flow(state)
    refs = snapshot_refs(state)

    jac = jax.jacrev(lambda p: run_analysis_with_geometry(flow, p, topo, baseline, refs).CL)(params)

    # Full FD check on plane; spot-check aincs only on larger models.
    dv_fields = ("aincs", "chords", "xles") if avl_path.stem == "plane" else ("aincs",)
    for field in dv_fields:
        ad_val = float(getattr(jac, field)[0])
        fd_val = _central_diff_params(flow, params, topo, baseline, refs, field, 0)
        assert ad_val == pytest.approx(fd_val, abs=FD_TOL, rel=FD_TOL), field
