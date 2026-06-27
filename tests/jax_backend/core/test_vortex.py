"""Tests for JAX vortex and AIC kernels."""

from __future__ import annotations

import numpy as np
import pytest

from tests.jax_backend.require_jax import require_jax

require_jax()

from openavl.aero.aic import vvor
from openavl.jax.aic import vvor_jax
from openavl.jax.backend import jnp

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
