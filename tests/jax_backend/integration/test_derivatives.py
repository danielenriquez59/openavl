"""Derivative validation: hand-coded, JAX AD, and finite differences (Phase 4A)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.jax_backend.require_jax import require_jax

jax = require_jax()
import jax.numpy as jnp

from openavl.analysis.deriv import StabilityDerivatives, compute_stability_derivatives
from openavl.jax.analysis import run_analysis
from openavl.jax.snapshot import snapshot_analysis_geometry, snapshot_flow, snapshot_refs
from openavl.jax.types import FlowCondition
from openavl.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR

PLANE_AVL = GEOMETRIES_DIR / "plane.avl"
B737_AVL = GEOMETRIES_DIR / "b737.avl"
SUPRA_AVL = GEOMETRIES_DIR / "supra.avl"

JAX_TOL = 1e-6
JAX_TOL_BODY = 5e-5
FD_TOL = 1e-5
FD_STEP = 1e-7

GEOMETRY_CASES = [
    pytest.param(PLANE_AVL, {"alpha": 5.0, "beta": 0.0}, id="plane"),
    pytest.param(B737_AVL, {"alpha": 3.0, "beta": 0.0}, id="b737"),
    pytest.param(SUPRA_AVL, {"alpha": 2.0, "beta": 0.0}, id="supra"),
]


pytestmark = pytest.mark.integration


def _build_solver(avl_path: Path, alpha: float, beta: float) -> AVLSolver:
    if not avl_path.is_file():
        pytest.skip(f"{avl_path.name} not found: {avl_path}")
    solver = AVLSolver(avl_path)
    solver.set_variable("alpha", alpha)
    solver.set_variable("beta", beta)
    solver.execute_run(max_iter=0)
    return solver


def _stability_moments(flow: FlowCondition, result_cm: jnp.ndarray, lnasa_sa: bool) -> jnp.ndarray:
    """Transform body-axis CM to stability-axis roll/pitch/yaw moments."""
    dir_ = -1.0 if lnasa_sa else 1.0
    ca = jnp.cos(flow.alfa)
    sa = jnp.sin(flow.alfa)
    cl_roll = dir_ * (result_cm[0] * ca + result_cm[2] * sa)
    cm_pitch = result_cm[1]
    cn_yaw = dir_ * (result_cm[2] * ca - result_cm[0] * sa)
    return jnp.array([cl_roll, cm_pitch, cn_yaw])


def _perturb_flow(flow: FlowCondition, field: str, index: int | None, delta: float) -> FlowCondition:
    """Return a copy of ``flow`` with one component perturbed."""
    value = getattr(flow, field)
    value = value + delta if index is None else value.at[index].add(delta)
    return flow._replace(**{field: value})


OUTPUTS = ("CL", "CD", "CY", "Cl", "Cm", "Cn")  # moments in stability axes
VARIABLES = {"a": ("alfa", None), "b": ("beta", None), "p": ("wrot", 0), "q": ("wrot", 1), "r": ("wrot", 2)}


def _derivative_tools(state):
    """AD Jacobian of all six outputs, and a cached central difference per input.

    One ``jacrev`` gives every entry, and each finite-difference input is
    perturbed once (through a jit-compiled run) for all six outputs.
    """
    geom, refs, flow = snapshot_analysis_geometry(state), snapshot_refs(state), snapshot_flow(state)

    def outputs(f: FlowCondition) -> jnp.ndarray:
        result = run_analysis(f, geom, refs)
        return jnp.concatenate([
            jnp.stack([result.CL, result.CD, result.CY]),
            _stability_moments(f, result.CM, state.lnasa_sa),
        ])

    jac = jax.jacrev(outputs)(flow)
    run = jax.jit(outputs)
    cache: dict = {}

    def central_difference(field: str, index: int | None = None) -> np.ndarray:
        if (field, index) not in cache:
            plus = run(_perturb_flow(flow, field, index, FD_STEP))
            minus = run(_perturb_flow(flow, field, index, -FD_STEP))
            cache[field, index] = np.asarray(plus - minus) / (2.0 * FD_STEP)
        return cache[field, index]

    return jac, central_difference


def _compare_derivative_set(solver: AVLSolver, *, label: str) -> None:
    """Three-way compare for stability-axis force/moment derivatives."""
    state = solver.state
    derivs = compute_stability_derivatives(state)
    jac, central_difference = _derivative_tools(state)

    # Full FD triple-check only on the smallest model (plane). For larger models,
    # validate JAX vs hand-coded only. wrot-p/r has no hand reference, so we
    # keep a two-case spot-check for CL_p and CL_r on all geometries.
    full_fd = label == "plane"
    tol = JAX_TOL_BODY if label == "supra" else JAX_TOL
    fd_tol = JAX_TOL_BODY if label == "supra" else FD_TOL
    q_scale = 2.0 / float(state.cref)  # AVL's q is qc/2V

    for i, out in enumerate(OUTPUTS):
        for var, (field, index) in VARIABLES.items():
            name = f"{out}_{var}"
            scale = q_scale if var == "q" else 1.0
            column = np.asarray(getattr(jac, field))
            jax_val = float(column[i] if index is None else column[i, index]) * scale
            if var in ("p", "r"):
                if full_fd or out == "CL":
                    fd_val = float(central_difference(field, index)[i])
                    assert jax_val == pytest.approx(fd_val, abs=FD_TOL, rel=FD_TOL), (
                        f"{label} {name}: JAX {jax_val} vs FD {fd_val}"
                    )
                continue
            hand = float(getattr(derivs, name))
            assert jax_val == pytest.approx(hand, abs=tol, rel=tol), (
                f"{label} {name}: JAX {jax_val} vs hand {hand}"
            )
            if full_fd:
                fd_val = float(central_difference(field, index)[i]) * scale
                assert fd_val == pytest.approx(hand, abs=fd_tol, rel=fd_tol), (
                    f"{label} {name}: FD {fd_val} vs hand {hand}"
                )

    rtd = 180.0 / np.pi
    for n, name in enumerate(state.control_names[: state.ncontrol]):
        for i, out in enumerate(OUTPUTS):
            hand = float(getattr(derivs, out + "_d")[name])
            jax_val = float(jac.delcon[i, n]) * rtd
            assert jax_val == pytest.approx(hand, abs=JAX_TOL, rel=JAX_TOL)
            if full_fd or n == 0:
                fd_val = float(central_difference("delcon", n)[i]) * rtd
                assert fd_val == pytest.approx(hand, abs=FD_TOL, rel=FD_TOL)


@pytest.mark.parametrize("avl_path, kwargs", GEOMETRY_CASES)
@pytest.mark.reference
def test_derivatives_three_way(avl_path: Path, kwargs: dict) -> None:
    """Hand-coded, JAX, and FD derivatives agree on multiple geometries."""
    solver = _build_solver(avl_path, kwargs["alpha"], kwargs["beta"])
    _compare_derivative_set(solver, label=avl_path.stem)


@pytest.mark.reference
def test_mach_derivative_matches_finite_difference() -> None:
    """d(CL, CD, CY, CM)/d(mach) from ``jax.jacrev(run_analysis)`` matches FD (A2).

    Regression test: the lattice AIC/influence matrices used to be baked in
    at a fixed Mach, so ``jac.*.mach`` was silently ~0 regardless of the true
    Mach sensitivity. Uses a nonzero Mach (0.3) because ``d(betm)/d(mach) = 0``
    at ``mach=0``, which would pass trivially even with a dead partial.
    """
    if not PLANE_AVL.is_file():
        pytest.skip(f"{PLANE_AVL.name} not found: {PLANE_AVL}")
    solver = AVLSolver(PLANE_AVL)
    solver.set_variable("alpha", 5.0)
    solver.set_variable("beta", 2.0)
    solver.set_parameter("mach", 0.3)
    solver.execute_run(max_iter=0)
    state = solver.state

    geom = snapshot_analysis_geometry(state)
    flow = snapshot_flow(state)
    refs = snapshot_refs(state)

    def outputs(f: FlowCondition) -> jnp.ndarray:  # CL, CD, CY, body-axis CM
        result = run_analysis(f, geom, refs)
        return jnp.concatenate([jnp.stack([result.CL, result.CD, result.CY]), result.CM])

    jac_mach = np.asarray(jax.jacrev(outputs)(flow).mach)
    run = jax.jit(outputs)
    fd = np.asarray(
        run(_perturb_flow(flow, "mach", None, FD_STEP)) - run(_perturb_flow(flow, "mach", None, -FD_STEP))
    ) / (2.0 * FD_STEP)

    for i, output in enumerate(("CL", "CD", "CY", "CMx", "CMy", "CMz")):
        if output in ("CL", "CD"):
            assert abs(fd[i]) > 1e-8, f"finite-difference d({output})/d(mach) unexpectedly ~0"
        # CY can be Mach-insensitive on symmetric wings; still verify JAX matches FD.
        assert jac_mach[i] == pytest.approx(fd[i], abs=FD_TOL, rel=FD_TOL), (
            f"d({output})/d(mach): JAX {jac_mach[i]} vs FD {fd[i]}"
        )
