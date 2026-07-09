"""Tests for JAX force integration (Phases 3D-3G)."""

from __future__ import annotations

import numpy as np
import pytest

from tests.jax_backend.require_jax import require_jax

jax = require_jax()

from openavl.aero.trefftz import tpforc
from openavl.jax.backend import jnp
from openavl.jax.forces import (
    compute_forces,
    flow_from_state,
    force_geometry_from_state,
    refs_from_state,
    sfforc_jax,
    trefftz_geometry_from_state,
    velocities_from_state,
)
from openavl.jax.trefftz import tpforc_jax
from openavl.solver import AVLSolver

from tests.helpers import GEOMETRIES_DIR

PLANE_AVL = GEOMETRIES_DIR / "plane.avl"
SQUARE_AVL = GEOMETRIES_DIR / "square.avl"
TOL = 1e-9

pytestmark = pytest.mark.core


def _run_plane_solver():
    if not PLANE_AVL.is_file():
        pytest.skip(f"plane.avl not found: {PLANE_AVL}")
    solver = AVLSolver(
        PLANE_AVL,
        alpha=-0.1455,
        beta=0.0,
        cl=0.390510,
        vel=64.5396,
        rho=0.0005846,
        gee=32.18,
        cd0=0.00835,
        xcg=0.02463,
        ycg=0.0,
        zcg=0.2239,
    )
    solver.execute_run(max_iter=20)
    assert solver.state.lsol
    return solver


def _run_symmetric_solver(cd0: float = 0.0):
    if not SQUARE_AVL.is_file():
        pytest.skip(f"square.avl not found: {SQUARE_AVL}")
    solver = AVLSolver(SQUARE_AVL, alpha=2.0, cd0=cd0)
    solver.execute_run(max_iter=20)
    assert solver.state.lsol
    assert solver.state.iysym == 1
    return solver


@pytest.mark.reference
def test_force_geometry_from_state_rejects_clmax_surf():
    """A13 guard: ``clmax_surf`` sectional-lift clipping has no JAX port yet."""
    solver = _run_plane_solver()
    state = solver.state
    state.clmax_surf[0] = 0.9
    with pytest.raises(NotImplementedError):
        force_geometry_from_state(state)


@pytest.mark.reference
def test_sfforc_jax_strip_forces_plane():
    """Compare JAX strip forces against NumPy state after full solve."""
    solver = _run_plane_solver()
    state = solver.state

    geom = force_geometry_from_state(state)
    flow = flow_from_state(state)
    refs = refs_from_state(state)
    velocities = velocities_from_state(state)
    gamma = jnp.asarray(state.gam.reshape(-1))

    result = sfforc_jax(
        geom,
        gamma,
        velocities,
        flow,
        refs,
        lnfld_wv=bool(state.lnfld_wv),
        lvisc=bool(state.lvisc),
        ltrforce=bool(state.ltrforce),
    )

    np.testing.assert_allclose(
        np.asarray(result.strips.clstrp),
        state.clstrp,
        atol=TOL,
        err_msg="clstrp mismatch",
    )
    np.testing.assert_allclose(
        np.asarray(result.strips.cdstrp),
        state.cdstrp,
        atol=TOL,
        err_msg="cdstrp mismatch",
    )
    np.testing.assert_allclose(
        np.asarray(result.strips.cystrp),
        state.cystrp,
        atol=TOL,
        err_msg="cystrp mismatch",
    )


@pytest.mark.fixture
def test_tpforc_jax_matches_numpy_fixture():
    """Compare Trefftz coefficients against NumPy TPFORC on a small fixture."""
    class _State:
        pi = np.pi
        amach = np.float64(0.3)
        ysym = np.float64(0.2)
        zsym = np.float64(-0.1)
        iysym = 0
        izsym = 0
        vrcorec = np.float64(0.01)
        vrcorew = np.float64(0.02)
        nstrip = 2
        nvor = 2
        nsurf = 1
        ncontrol = 0
        ndesign = 0
        numax = NUMAX
        sref = np.float64(1.5)
        bref = np.float64(2.0)
        mach = np.float64(0.3)
        cref = np.float64(1.0)
        xyzref = np.zeros(3)
        cdref = np.float64(0.0)

        def __init__(self) -> None:
            self.ijfrst = np.array([0, 1], dtype=np.int32)
            self.nvstrp = np.array([1, 1], dtype=np.int32)
            self.jfrst = np.array([0], dtype=np.int32)
            self.nj = np.array([2], dtype=np.int32)
            self.gam = np.array([0.0, 0.5, 0.3], dtype=np.float64)
            self.gam_u = np.zeros((3, NUMAX), dtype=np.float64)
            for i in range(1, 3):
                for n in range(NUMAX):
                    self.gam_u[i, n] = np.float64(0.05 * (i + n + 1))
            self.gam_d = np.zeros((3, 1), dtype=np.float64)
            self.gam_g = np.zeros((3, 1), dtype=np.float64)
            self.rv1 = np.zeros((3, 3), dtype=np.float64)
            self.rv2 = np.zeros((3, 3), dtype=np.float64)
            self.rc = np.zeros((3, 3), dtype=np.float64)
            self.rv1[:, 0] = [0.0, 0.0, 0.0]
            self.rv2[:, 0] = [1.0, 0.5, 0.2]
            self.rv1[:, 1] = [0.2, -0.3, 0.1]
            self.rv2[:, 1] = [1.2, 0.2, -0.1]
            self.rc[:, 0] = [0.5, 0.1, 0.05]
            self.rc[:, 1] = [0.8, -0.1, 0.2]
            self.chord = np.array([0.0, 1.0, 0.8], dtype=np.float64)
            self.lssurf = np.array([0, 0], dtype=np.int32)
            self.lncomp = np.array([0, 1], dtype=np.int32)
            self.lfload = np.array([True], dtype=bool)
            self.lstripoff = np.zeros(2, dtype=bool)
            self.lviscstrp = np.zeros(2, dtype=bool)

    state = _State()
    tpforc(state)

    tgeom = trefftz_geometry_from_state(state)
    refs = refs_from_state(state)
    gamma = jnp.asarray(state.gam[: state.nvor])

    result = tpforc_jax(gamma, tgeom, refs)
    np.testing.assert_allclose(float(result.CL), state.clff, atol=1e-9)
    np.testing.assert_allclose(float(result.CY), state.cyff, atol=1e-9)
    np.testing.assert_allclose(float(result.CDi), state.cdff, atol=1e-9)
    np.testing.assert_allclose(float(result.spanef), state.spanef, atol=1e-9)
    np.testing.assert_allclose(np.asarray(result.dwwake), state.dwwake, atol=1e-9)


@pytest.mark.reference
def test_compute_forces_trefftz_not_double_symmetrized():
    """Far-field Trefftz totals should match TPFORC for symmetric cases."""
    solver = _run_symmetric_solver()
    state = solver.state

    tpforc(state)

    geom = force_geometry_from_state(state)
    flow = flow_from_state(state)
    refs = refs_from_state(state)
    velocities = velocities_from_state(state)
    tgeom = trefftz_geometry_from_state(state)
    gamma = jnp.asarray(state.gam.reshape(-1))

    result = compute_forces(
        geom,
        gamma,
        velocities,
        flow,
        refs,
        tgeom=tgeom,
        lnfld_wv=bool(state.lnfld_wv),
        lvisc=bool(state.lvisc),
    )

    assert float(result.CL) == pytest.approx(state.cltot, abs=TOL)
    assert float(result.CD) == pytest.approx(state.cdtot, abs=TOL)
    assert float(result.CY) == pytest.approx(state.cytot, abs=TOL)
    np.testing.assert_allclose(np.asarray(result.CM), state.cmtot, atol=TOL)
    np.testing.assert_allclose(np.asarray(result.CF), state.cftot, atol=TOL)
    assert float(result.CLFF) == pytest.approx(state.clff, abs=TOL)
    assert float(result.CYFF) == pytest.approx(state.cyff, abs=TOL)
    assert float(result.CDFF) == pytest.approx(state.cdff, abs=TOL)


@pytest.mark.reference
def test_sfforc_jax_grad_finite_with_disabled_strip():
    """A9 gradient regression: a deliberately disabled (zero-width,
    turned-off) strip must not poison reverse-mode gradients through the
    divide-then-mask paths in ``_vortex_forces``.
    """
    solver = _run_plane_solver()
    state = solver.state

    geom = force_geometry_from_state(state)
    flow = flow_from_state(state)
    refs = refs_from_state(state)
    velocities = velocities_from_state(state)
    gamma = jnp.asarray(state.gam.reshape(-1))

    geom = geom._replace(
        wstrip=geom.wstrip.at[0].set(0.0),
        strip_map=geom.strip_map._replace(
            lstripoff=geom.strip_map.lstripoff.at[0].set(True)
        ),
    )

    def total(gamma_: jnp.ndarray, alfa_: jnp.ndarray) -> jnp.ndarray:
        f = flow._replace(alfa=alfa_)
        result = sfforc_jax(
            geom, gamma_, velocities, f, refs,
            lnfld_wv=bool(state.lnfld_wv),
            lvisc=bool(state.lvisc),
            ltrforce=bool(state.ltrforce),
        )
        return result.CL + result.CD

    grad_gamma, grad_alfa = jax.grad(total, argnums=(0, 1))(gamma, flow.alfa)
    assert bool(jnp.all(jnp.isfinite(grad_gamma))), "gamma gradient is not finite with a disabled strip"
    assert bool(jnp.isfinite(grad_alfa)), "alfa gradient is not finite with a disabled strip"


@pytest.mark.reference
def test_apply_viscous_strips_grad_finite_mixed_polar():
    """A10 gradient regression: strips without a viscous polar (the common
    tails/fins case) must not poison gradients when vmapped alongside a
    sibling strip that does have a valid polar.
    """
    solver = _run_plane_solver()
    state = solver.state

    geom = force_geometry_from_state(state)
    flow = flow_from_state(state)
    refs = refs_from_state(state)
    velocities = velocities_from_state(state)
    gamma = jnp.asarray(state.gam.reshape(-1))

    # plane.avl has no CDCL command, so every strip's clcd row is already
    # all-zero (no polar). Give exactly one strip a real polar so the model
    # has genuinely mixed polar coverage while every other strip keeps its
    # degenerate (clmax<=cl0<=clmin) row.
    valid_polar = jnp.array([-1.0, 0.08, 0.0, 0.02, 1.0, 0.09])
    geom = geom._replace(
        clcd=geom.clcd.at[0].set(valid_polar),
        strip_map=geom.strip_map._replace(
            lviscstrp=geom.strip_map.lviscstrp.at[0].set(True)
        ),
    )

    def total_cd(gamma_: jnp.ndarray) -> jnp.ndarray:
        result = sfforc_jax(
            geom, gamma_, velocities, flow, refs,
            lnfld_wv=bool(state.lnfld_wv),
            lvisc=True,
        )
        return result.CD

    grad = jax.grad(total_cd)(gamma)
    assert bool(jnp.all(jnp.isfinite(grad))), "viscous drag gradient is not finite with mixed polar coverage"


@pytest.mark.reference
def test_compute_forces_cdref_after_symmetry_iysym1():
    """A12 parity: cdref must be added after XZ-symmetry doubling.

    NumPy's ``aero()`` doubles the inviscid/body/Trefftz totals for
    ``iysym=1`` and only then adds the ``cdref`` baseline-drag term, so CD
    gets ``cdref*V^2`` (not ``2*cdref*V^2``) and the CY/CF sideforce term
    from ``cdref`` survives the CY zeroing. JAX must match this order.
    """
    solver = _run_symmetric_solver(cd0=0.02)
    state = solver.state
    assert state.cdref > 0.0

    geom = force_geometry_from_state(state)
    flow = flow_from_state(state)
    refs = refs_from_state(state)
    velocities = velocities_from_state(state)
    tgeom = trefftz_geometry_from_state(state)
    gamma = jnp.asarray(state.gam.reshape(-1))

    result = compute_forces(
        geom,
        gamma,
        velocities,
        flow,
        refs,
        tgeom=tgeom,
        lnfld_wv=bool(state.lnfld_wv),
        lvisc=bool(state.lvisc),
    )

    assert float(result.CD) == pytest.approx(state.cdtot, abs=TOL)
    assert float(result.CY) == pytest.approx(state.cytot, abs=TOL)
    np.testing.assert_allclose(np.asarray(result.CF), state.cftot, atol=TOL)
