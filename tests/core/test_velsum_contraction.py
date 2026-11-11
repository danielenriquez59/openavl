"""velsum outputs match an independent einsum contraction of wc_gam/wv_gam."""

from __future__ import annotations

import numpy as np
import pytest

from openavl import constants as C
from openavl.core.setup import velsum
from openavl.core.state import AVLState
from openavl.solver import AVLSolver
from tests.helpers import GEOMETRIES_DIR

RTOL = 1e-12


def _reference(state: AVLState) -> dict[str, np.ndarray]:
    """Explicit einsum contraction, independent of velsum's implementation."""
    n = state.nvor
    nu = state.numax
    nc = state.ncontrol
    u = np.concatenate((state.vinf, state.wrot))[:nu]
    wc = state.wc_gam[:, :n, :n]
    wv = state.wv_gam[:, :n, :n]
    ref = {
        "vc": np.einsum("kij,j->ki", wc, state.gam[:n]),
        "vv": np.einsum("kij,j->ki", wv, state.gam[:n]),
        "vc_u": np.einsum("kij,jn->kin", wc, state.gam_u[:n, :nu]),
        "vv_u": np.einsum("kij,jn->kin", wv, state.gam_u[:n, :nu]),
        "vc_d": np.einsum("kij,jn->kin", wc, state.gam_d[:n, :nc]),
        "vv_d": np.einsum("kij,jn->kin", wv, state.gam_d[:n, :nc]),
        "wcsrd": np.einsum("kin,n->ki", state.wcsrd_u[:, :n, :nu], u),
        "wvsrd": np.einsum("kin,n->ki", state.wvsrd_u[:, :n, :nu], u),
    }
    ref["wc"] = ref["vc"] + ref["wcsrd"]
    ref["wv"] = ref["vv"] + ref["wvsrd"]
    ref["wc_u"] = ref["vc_u"] + state.wcsrd_u[:, :n, :nu]
    ref["wv_u"] = ref["vv_u"] + state.wvsrd_u[:, :n, :nu]
    ref["wc_d"] = ref["vc_d"]
    ref["wv_d"] = ref["vv_d"]
    return ref


def _assert_matches(state: AVLState) -> None:
    ref = _reference(state)
    n = state.nvor
    trailing = {"u": state.numax, "d": state.ncontrol}
    for name, expected in ref.items():
        arr = getattr(state, name)
        suffix = name.rsplit("_", 1)[-1] if "_" in name else ""
        got = arr[:, :n, : trailing[suffix]] if suffix in trailing else arr[:, :n]
        assert got.shape == expected.shape, name
        scale = max(np.abs(expected).max(initial=0.0), 1e-300)
        err = np.abs(got - expected).max(initial=0.0) / scale
        assert err <= RTOL, f"{name}: rel err {err:.3e}"


def test_velsum_random_state_with_controls_and_padding():
    """Random tensors and controls, nvmax > nvor (strided views)."""
    rng = np.random.default_rng(7)
    nvor, nvmax, ncontrol = 37, 45, 2
    state = AVLState(
        nvor=nvor,
        nvmax=nvmax,
        ncontrol=ncontrol,
        numax=C.NUMAX,
        ndmax=C.NDMAX,
    )
    state._allocate_arrays(C.NDMAX, 1, nvmax, 1)
    state.vinf[:] = [0.9, -0.1, 0.2]
    state.wrot[:] = [0.05, -0.02, 0.03]
    for name in ("wc_gam", "wv_gam", "wcsrd_u", "wvsrd_u", "gam", "gam_u", "gam_d"):
        arr = getattr(state, name)
        arr[...] = rng.standard_normal(arr.shape)

    velsum(state)
    _assert_matches(state)


@pytest.mark.parametrize("geometry", ["bd.avl", "vanilla.avl"])
def test_velsum_solved_geometry(geometry: str):
    """Real lattices: bd.avl has a body and controls, vanilla.avl has four controls."""
    path = GEOMETRIES_DIR / geometry
    if not path.is_file():
        pytest.skip(f"{geometry} not found")
    solver = AVLSolver(path, alpha=4.0, beta=3.0)
    solver.execute_run(max_iter=0)
    state = solver.state
    assert state.ncontrol > 0
    if geometry == "bd.avl":
        assert state.nlnode > 0
    assert np.abs(state.gam_d[: state.nvor, : state.ncontrol]).max() > 0.0

    velsum(state)
    _assert_matches(state)
