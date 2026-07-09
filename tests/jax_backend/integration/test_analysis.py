"""Primal force comparison: JAX run_analysis vs NumPy solver (Phase 4A)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.jax_backend.require_jax import require_jax

require_jax()

from openavl.jax.analysis import _make_forces_checkpoint, make_run_analysis_jit, run_analysis
from openavl.jax.snapshot import snapshot_analysis_geometry, snapshot_flow, snapshot_refs
from openavl.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR

PLANE_AVL = GEOMETRIES_DIR / "plane.avl"
B737_AVL = GEOMETRIES_DIR / "b737.avl"
SUPRA_AVL = GEOMETRIES_DIR / "supra.avl"
CDCL_SURFACE_AVL = GEOMETRIES_DIR / "testcdcl_surface.avl"
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


@pytest.mark.reference
def test_run_analysis_checkpoint_honors_lvisc() -> None:
    """A4 regression: the default checkpoint must not silently drop ``lvisc``.

    ``testcdcl_surface.avl`` has a real ``CDCL`` polar, so ``lvisc=True``
    changes CD relative to ``lvisc=False``. The checkpointed path (used by
    ``run_analysis_with_geometry``) must match the eager path once the
    caller's flag is threaded through instead of being hard-bound to False.
    """
    if not CDCL_SURFACE_AVL.is_file():
        pytest.skip(f"{CDCL_SURFACE_AVL.name} not found: {CDCL_SURFACE_AVL}")
    solver = AVLSolver(CDCL_SURFACE_AVL, alpha=4.0, beta=0.0)
    solver.execute_run(max_iter=20)
    assert solver.state.lsol
    assert solver.state.lviscstrp.any()

    state = solver.state
    geom = snapshot_analysis_geometry(state)
    flow = snapshot_flow(state)
    refs = snapshot_refs(state)

    eager_inviscid = run_analysis(flow, geom, refs, lvisc=False, use_checkpoint=False)
    eager_viscous = run_analysis(flow, geom, refs, lvisc=True, use_checkpoint=False)
    assert float(eager_viscous.CD) != pytest.approx(float(eager_inviscid.CD))

    checkpointed_viscous = run_analysis(flow, geom, refs, lvisc=True, use_checkpoint=True)
    assert float(checkpointed_viscous.CD) == pytest.approx(
        float(eager_viscous.CD), abs=PRIMAL_TOL, rel=PRIMAL_TOL
    )

    checkpointed_inviscid = run_analysis(flow, geom, refs, lvisc=False, use_checkpoint=True)
    assert float(checkpointed_inviscid.CD) == pytest.approx(
        float(eager_inviscid.CD), abs=PRIMAL_TOL, rel=PRIMAL_TOL
    )


def test_integrate_forces_asserts_on_checkpoint_flag_mismatch() -> None:
    """A4 safety net: a checkpoint built with different flags must trip an assert."""
    if not PLANE_AVL.is_file():
        pytest.skip(f"{PLANE_AVL.name} not found: {PLANE_AVL}")
    solver = _run_numpy_solver(PLANE_AVL, alpha=5.0, beta=0.0)
    state = solver.state
    geom = snapshot_analysis_geometry(state)
    flow = snapshot_flow(state)
    refs = snapshot_refs(state)

    mismatched_checkpoint = _make_forces_checkpoint(
        iysym=int(refs.iysym),
        include_body=geom.body.nl.shape[0] > 0,
        lvisc=False,
        lnfld_wv=False,
    )
    with pytest.raises(AssertionError):
        run_analysis(
            flow, geom, refs,
            lvisc=True, use_checkpoint=True,
            forces_checkpoint=mismatched_checkpoint,
        )


@pytest.mark.reference
def test_run_analysis_body_source_updates_with_flow() -> None:
    """A6 regression: body source strength must be reformed from live flow.

    ``bdforc_jax`` used to reuse the body source strength frozen at the
    snapshot flow condition for every flow evaluated through the frozen
    geometry, which kills the flow-dependence of the slender-body force.
    Snapshot geometry once at ``alpha=2`` and evaluate at ``alpha=6``; the
    result must match a fresh NumPy solve at ``alpha=6`` (same geometry and
    Mach), not one contaminated by the ``alpha=2`` source strength.
    """
    if not SUPRA_AVL.is_file():
        pytest.skip(f"supra.avl not found: {SUPRA_AVL}")

    snapshot_solver = AVLSolver(SUPRA_AVL, alpha=2.0, beta=0.0)
    snapshot_solver.execute_run(max_iter=1)
    geom = snapshot_analysis_geometry(snapshot_solver.state)
    refs = snapshot_refs(snapshot_solver.state)
    assert geom.body.nl.shape[0] > 0, "supra.avl is expected to carry a body"

    eval_solver = AVLSolver(SUPRA_AVL, alpha=6.0, beta=0.0)
    eval_solver.execute_run(max_iter=1)
    flow_eval = snapshot_flow(eval_solver.state)

    result = run_analysis(flow_eval, geom, refs)

    assert float(result.CL) == pytest.approx(eval_solver.state.cltot, abs=1e-6, rel=1e-6)
    assert float(result.CD) == pytest.approx(eval_solver.state.cdtot, abs=1e-6, rel=1e-6)
