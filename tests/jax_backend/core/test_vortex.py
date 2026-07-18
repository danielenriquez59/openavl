"""Tests for JAX vortex and AIC kernels."""

from __future__ import annotations

import numpy as np
import pytest

from tests.jax_backend.require_jax import require_jax

jax = require_jax()

from openavl.aero.aic import vvor
from openavl.jax.aic import vvor_jax
from openavl.jax.backend import jnp
from openavl.jax.vortex import vorvelc_jax

pytestmark = pytest.mark.core


@pytest.mark.reference
def test_vvor_jax_matches_numpy_plane(plane_state):
    """JAX vvor_jax matches NumPy vvor on plane.avl geometry."""
    state = plane_state
    nvor = state.nvor

    wc_numpy = vvor(
        state.betm,
        state.iysym,
        state.ysym,
        state.izsym,
        state.zsym,
        state.vrcorec,
        state.vrcorew,
        nvor,
        state.rv1,
        state.rv2,
        state.lvcomp,
        state.chordv,
        nvor,
        state.rc,
        state.lvcomp,
        False,
        None,
        nvor,
    )

    wc_jax = vvor_jax(
        state.betm,
        state.iysym,
        state.ysym,
        state.izsym,
        state.zsym,
        state.vrcorec,
        state.vrcorew,
        jnp.array(state.rv1[:, :nvor]),
        jnp.array(state.rv2[:, :nvor]),
        jnp.array(state.lvcomp[:nvor]),
        jnp.array(state.chordv[:nvor]),
        jnp.array(state.rc[:, :nvor]),
        jnp.array(state.lvcomp[:nvor]),
        False,
    )

    np.testing.assert_allclose(
        np.asarray(wc_jax),
        wc_numpy[:, :nvor, :nvor],
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.reference
def test_vorvelc_jax_grad_finite_at_leg_endpoint():
    """A9 gradient regression: a control point exactly at a trailing-leg
    endpoint (amag or bmag == 0, zero core radius) must not poison the
    reverse-mode gradient via the divide-then-mask trap.
    """

    def total(x0: jnp.ndarray) -> jnp.ndarray:
        u, v, w = vorvelc_jax(
            x0, jnp.array(0.0), jnp.array(0.0), True,
            jnp.array(0.0), jnp.array(0.0), jnp.array(0.0),
            jnp.array(1.0), jnp.array(0.5), jnp.array(0.2),
            jnp.array(0.98), jnp.array(0.0),
        )
        return u + v + w

    grad = jax.grad(total)(jnp.array(0.0))
    assert bool(jnp.isfinite(grad)), f"grad is not finite: {grad}"


@pytest.mark.reference
def test_vvor_jax_grad_finite_zero_width_vortex():
    """A9 gradient regression: a zero-width (degenerate) filament assembled
    into the AIC matrix must not poison reverse-mode gradients.
    """
    rv1 = jnp.array([[0.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    rv2 = jnp.array([[1.0, 1.0], [0.5, 1.0], [0.2, 0.0]])  # vortex 1 is zero-width
    rc = jnp.array([[0.5, 1.5], [0.3, 1.5], [0.1, 0.05]])
    ncomp = jnp.array([0, 0], dtype=jnp.int32)
    chordv = jnp.array([1.0, 0.0])

    def total(rv2_: jnp.ndarray) -> jnp.ndarray:
        wc = vvor_jax(
            0.98, 0, 0.0, 0, 0.0, 0.01, 0.02,
            rv1, rv2_, ncomp, chordv, rc, ncomp, False,
        )
        return jnp.sum(wc)

    grad = jax.grad(total)(rv2)
    assert bool(jnp.all(jnp.isfinite(grad))), f"grad is not finite: {grad}"
