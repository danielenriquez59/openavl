"""Tests for openavl.aic."""

from __future__ import annotations

import numpy as np
import pytest

from openavl.aic import srdset, srdvelc, vorvelc, vsrd, vvor
from tests.helpers import load_json_fixture

pytestmark = pytest.mark.core


def _build_aic_inputs():
    betm = 0.9
    iysym = 0
    ysym = 0.2
    izsym = 0
    zsym = -0.1
    vrcorec = 0.01
    vrcorew = 0.02
    srcore = 0.1
    nv = 2
    nc = 2

    rv1 = np.zeros((3, nv), dtype=np.float64)
    rv2 = np.zeros((3, nv), dtype=np.float64)
    rc = np.zeros((3, nc), dtype=np.float64)
    rv1[:, 0] = [0.0, 0.0, 0.0]
    rv2[:, 0] = [1.0, 0.5, 0.2]
    rv1[:, 1] = [0.2, -0.3, 0.1]
    rv2[:, 1] = [1.2, 0.2, -0.1]
    rc[:, 0] = [0.5, 0.1, 0.05]
    rc[:, 1] = [0.8, -0.1, 0.2]

    chordv = np.array([1.0, 0.8], dtype=np.float64)
    ncompv = np.array([1, 2], dtype=np.int32)
    ncompc = np.array([1, 2], dtype=np.int32)

    nbody = 1
    nldim = 3
    lfrst = np.array([0], dtype=np.int32)
    nl = np.array([3], dtype=np.int32)
    rl = np.zeros((3, nldim), dtype=np.float64)
    rl[:, 0] = [0.0, 0.0, 0.0]
    rl[:, 1] = [1.0, 0.1, 0.0]
    rl[:, 2] = [2.0, 0.1, 0.1]
    radl = np.array([0.2, 0.25, 0.3], dtype=np.float64)
    xyzref = np.zeros(3, dtype=np.float64)

    return {
        "betm": betm,
        "iysym": iysym,
        "ysym": ysym,
        "izsym": izsym,
        "zsym": zsym,
        "vrcorec": vrcorec,
        "vrcorew": vrcorew,
        "srcore": srcore,
        "nv": nv,
        "nc": nc,
        "rv1": rv1,
        "rv2": rv2,
        "rc": rc,
        "chordv": chordv,
        "ncompv": ncompv,
        "ncompc": ncompc,
        "nbody": nbody,
        "nldim": nldim,
        "lfrst": lfrst,
        "nl": nl,
        "rl": rl,
        "radl": radl,
        "xyzref": xyzref,
    }


@pytest.mark.fixture
def test_aic_matches_js_fixture(fixtures_dir):
    """AIC routines match JS reference fixture."""
    data = load_json_fixture(fixtures_dir, "aic_expected.json")
    inp = _build_aic_inputs()

    u, v, w = vorvelc(
        0.55, 0.15, -0.02, True,
        0.0, 0.0, 0.0, 1.0, 0.5, 0.2,
        inp["betm"], 0.03,
    )
    np.testing.assert_allclose([u, v, w], data["vorvelc"], rtol=0, atol=1e-5)

    uvws, uvwd = srdvelc(
        0.55, 0.15, -0.02,
        0.0, 0.0, 0.0, 1.0, 0.5, 0.2,
        inp["betm"], 0.05,
    )
    np.testing.assert_allclose(uvws, data["srdvelc_v"], rtol=0, atol=1e-5)
    np.testing.assert_allclose(uvwd.ravel(order="F"), data["srdvelc_m"], rtol=0, atol=1e-5)

    src_u, dbl_u = srdset(
        inp["betm"], inp["xyzref"], inp["iysym"],
        inp["nbody"], inp["lfrst"], inp["nldim"],
        inp["nl"], inp["rl"], inp["radl"],
    )
    src_out = []
    for l in range(2):
        for iu in range(6):
            src_out.append(src_u[l, iu])
    dbl_out = []
    for k in range(3):
        for l in range(2):
            for iu in range(6):
                dbl_out.append(dbl_u[k, l, iu])
    np.testing.assert_allclose(src_out, data["srdset_src"], rtol=0, atol=1e-5)
    np.testing.assert_allclose(dbl_out, data["srdset_dbl"], rtol=0, atol=1e-5)

    wc_u = vsrd(
        inp["betm"], inp["iysym"], inp["ysym"], inp["izsym"], inp["zsym"], inp["srcore"],
        inp["nbody"], inp["lfrst"], inp["nldim"],
        inp["nl"], inp["rl"], inp["radl"],
        6, src_u, dbl_u,
        inp["nc"], inp["rc"],
    )
    wc_out = []
    for i in range(inp["nc"]):
        for iu in range(6):
            for k in range(3):
                wc_out.append(wc_u[k, i, iu])
    np.testing.assert_allclose(wc_out, data["vsrd"], rtol=0, atol=1e-5)

    wc_gam = vvor(
        inp["betm"], inp["iysym"], inp["ysym"], inp["izsym"], inp["zsym"],
        inp["vrcorec"], inp["vrcorew"],
        inp["nv"], inp["rv1"], inp["rv2"], inp["ncompv"], inp["chordv"],
        inp["nc"], inp["rc"], inp["ncompc"], True,
    )
    vvor_out = []
    for i in range(inp["nc"]):
        for j in range(inp["nv"]):
            for k in range(3):
                vvor_out.append(wc_gam[k, i, j])
    np.testing.assert_allclose(vvor_out, data["vvor"], rtol=0, atol=1e-5)
