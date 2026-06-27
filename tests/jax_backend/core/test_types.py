"""Tests for openavl.jax.types and snapshot helpers."""

from __future__ import annotations

import numpy as np
import pytest

from tests.jax_backend.require_jax import require_jax

require_jax()

from openavl.jax.snapshot import snapshot_flow, snapshot_geometry, snapshot_refs

pytestmark = pytest.mark.core


def test_snapshot_geometry_shapes(plane_state):
    """Snapshot geometry arrays match AVLState dimensions."""
    state = plane_state
    geom = snapshot_geometry(state)

    nvor = state.nvor
    nstrip = state.nstrip

    assert geom.nvor == nvor
    assert geom.nstrip == nstrip
    assert geom.nsurf == state.nsurf
    assert geom.rv1.shape == (3, nvor)
    assert geom.rv2.shape == (3, nvor)
    assert geom.rv.shape == (3, nvor)
    assert geom.rc.shape == (3, nvor)
    assert geom.enc.shape == (3, nvor)
    assert geom.env.shape == (3, nvor)
    assert geom.chordv.shape == (nvor,)
    assert geom.lvcomp.shape == (nvor,)
    assert geom.wc_gam.shape == (3, nvor, nvor)
    assert geom.wv_gam.shape == (3, nvor, nvor)
    assert geom.aicn.shape == (nvor, nvor)

    sm = geom.strip_map
    assert sm.vortex_to_strip.shape == (nvor,)
    assert sm.strip_to_surface.shape == (nstrip,)
    assert sm.ijfrst.shape == (nstrip,)
    assert sm.nvstrp.shape == (nstrip,)
    assert sm.chord.shape == (nstrip,)
    assert sm.ainc.shape == (nstrip,)
    assert sm.lstripoff.shape == (nstrip,)
    assert sm.ess.shape == (3, nstrip)


def test_snapshot_round_trip_values(plane_state):
    """Snapshot copies NumPy state without changing values."""
    state = plane_state
    geom = snapshot_geometry(state)
    flow = snapshot_flow(state)
    refs = snapshot_refs(state)

    nvor = state.nvor
    nstrip = state.nstrip

    np.testing.assert_allclose(np.asarray(geom.rv1), state.rv1[:, :nvor], rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(geom.rc), state.rc[:, :nvor], rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(geom.aicn), state.aicn[:nvor, :nvor], rtol=0, atol=0)
    np.testing.assert_allclose(
        np.asarray(geom.strip_map.ijfrst), state.ijfrst[:nstrip], rtol=0, atol=0
    )
    np.testing.assert_allclose(np.asarray(flow.alfa), state.alfa, rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(flow.beta), state.beta, rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(flow.wrot), state.wrot, rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(refs.xyzref), state.xyzref, rtol=0, atol=0)
    assert float(refs.sref) == state.sref
    assert float(refs.cref) == state.cref
    assert float(refs.bref) == state.bref


def test_vortex_to_strip_mapping(plane_state):
    """Each vortex maps to the strip that owns it."""
    state = plane_state
    geom = snapshot_geometry(state)
    v2s = np.asarray(geom.strip_map.vortex_to_strip)

    for j in range(state.nstrip):
        i1 = int(state.ijfrst[j])
        for k in range(int(state.nvstrp[j])):
            assert v2s[i1 + k] == j
