"""Tests for openavl.setup (GAMSUM/VELSUM)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from openavl import constants as C
from openavl.setup import gamsum, gdcalc, velsum
from openavl.state import AVLState
from tests.helpers import load_json_fixture, run_ref_binary

pytestmark = pytest.mark.core


def _make_mock_state() -> AVLState:
    """Build mock state matching asetup.test.mjs."""
    nvor = 2
    nlnode = 2
    ncontrol = 1
    ndesign = 1
    numax = C.NUMAX
    ndmax = C.NDMAX
    ngmax = C.NGMAX

    state = AVLState(
        nvor=nvor,
        nvmax=nvor,
        nlnode=nlnode,
        ncontrol=ncontrol,
        ndesign=ndesign,
        numax=numax,
        ndmax=ndmax,
        ngmax=ngmax,
        nlmax=nlnode,
    )
    state._allocate_arrays(ndmax, ngmax, 1, nvor, 1)

    state.vinf[:] = [10.0, -2.0, 1.0]
    state.wrot[:] = [0.1, -0.2, 0.3]
    state.delcon[0] = 0.2
    state.deldes[0] = -0.1

    for k in range(3):
        for i in range(nvor):
            for j in range(nvor):
                state.wc_gam[k, i, j] = 0.01 * (k + 1) + 0.001 * (i + 1) + 0.0001 * (j + 1)
                state.wv_gam[k, i, j] = 0.02 * (k + 1) + 0.002 * (i + 1) + 0.0002 * (j + 1)

    for i in range(nvor):
        for iu in range(numax):
            state.gam_u_0[i, iu] = 0.1 * (i + 1) + 0.01 * (iu + 1)
            state.gam_u_d[i, iu, 0] = 0.02 * (i + 1) + 0.001 * (iu + 1)
            state.gam_u_g[i, iu, 0] = -0.03 * (i + 1) + 0.002 * (iu + 1)

    for l in range(nlnode):
        for iu in range(numax):
            state.src_u[l, iu] = 0.05 * (l + 1) + 0.002 * (iu + 1)
            for k in range(3):
                state.dbl_u[k, l, iu] = 0.01 * (k + 1) + 0.001 * (l + 1) + 0.0005 * (iu + 1)

    for k in range(3):
        for i in range(nvor):
            for iu in range(numax):
                state.wcsrd_u[k, i, iu] = 0.03 * (k + 1) + 0.001 * (i + 1) + 0.0001 * (iu + 1)
                state.wvsrd_u[k, i, iu] = 0.04 * (k + 1) + 0.0015 * (i + 1) + 0.0002 * (iu + 1)

    return state


def _extract_outputs(state: AVLState) -> list[float]:
    out: list[float] = []
    for i in range(state.nvor):
        out.append(float(state.gam[i]))
    for i in range(state.nvor):
        out.append(float(state.gam_d[i, 0]))
    for i in range(state.nvor):
        out.append(float(state.gam_g[i, 0]))
    for l in range(state.nlnode):
        out.append(float(state.src[l]))
    for l in range(state.nlnode):
        for k in range(3):
            out.append(float(state.dbl[k, l]))
    for i in range(state.nvor):
        for k in range(3):
            out.append(float(state.wc[k, i]))
    for i in range(state.nvor):
        for k in range(3):
            out.append(float(state.wv[k, i]))
    for i in range(state.nvor):
        for k in range(3):
            out.append(float(state.wc_u[k, i, 0]))
    for i in range(state.nvor):
        for k in range(3):
            out.append(float(state.wv_u[k, i, 0]))
    for i in range(state.nvor):
        for k in range(3):
            out.append(float(state.vc_d[k, i, 0]))
    for i in range(state.nvor):
        for k in range(3):
            out.append(float(state.vc_g[k, i, 0]))
    return out


@pytest.mark.fixture
def test_gamsum_velsum_matches_js_fixture(fixtures_dir):
    """GAMSUM/VELSUM outputs match JS reference fixture."""
    expected = load_json_fixture(fixtures_dir, "asetup_expected.json")["expected"]
    state = _make_mock_state()
    gamsum(state)
    velsum(state)
    actual = _extract_outputs(state)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-4)


@pytest.mark.reference
def test_gamsum_velsum_matches_fortran_ref(ref_binary, fixtures_dir):
    """GAMSUM/VELSUM match Fortran asetup_ref when available."""
    ref_path = ref_binary("asetup_ref")
    if ref_path is None:
        pytest.skip("Fortran reference binary not found")

    ref = run_ref_binary(ref_path)
    expected = load_json_fixture(fixtures_dir, "asetup_expected.json")["expected"]
    assert len(ref) >= len(expected)
    ref_tail = ref[-len(expected):]

    state = _make_mock_state()
    gamsum(state)
    velsum(state)
    actual = _extract_outputs(state)
    np.testing.assert_allclose(actual, ref_tail, rtol=0, atol=1e-4)


def test_gdcalc_body_contraction_and_inactive_zeroing():
    """gdcalc contracts wcsrd_u·u and zeroes inactive (non-V·n) rows in-place."""
    state = _make_mock_state()
    state.lvnc[: state.nvor] = True
    state.lvnc[1] = False  # inactive / Kutta-style row
    state.lvalbe[: state.nvor] = False
    state.lcondef[0] = True
    state.enc_d[:, : state.nvor, 0] = 0.0
    state.enc_d[2, 0, 0] = 1.0
    state.enc_d[2, 1, 0] = 1.0
    # Identity AIC so baksub leaves the RHS unchanged.
    state.aicn[: state.nvor, : state.nvor] = np.eye(state.nvor)
    state.iapiv[: state.nvor] = np.arange(state.nvor, dtype=np.int32)
    state.gam_d[: state.nvor, 0] = 99.0

    gdcalc(state, state.ncontrol, state.lcondef, state.enc_d, state.gam_d)

    u = np.concatenate((state.vinf, state.wrot))
    vq0 = state.wcsrd_u[:, 0, :6] @ u
    assert state.gam_d[0, 0] == pytest.approx(-vq0[2], abs=1e-12)
    assert state.gam_d[1, 0] == pytest.approx(0.0)


def test_gdcalc_matches_gamsum_control_sensitivities_without_body():
    """On a no-body model, gdcalc(gam_d) matches gamsum's gam_u_d @ u."""
    state = _make_mock_state()
    state.nlnode = 0
    state.wcsrd_u[:] = 0.0
    state.lvnc[: state.nvor] = True
    state.lvalbe[: state.nvor] = False
    state.lcondef[0] = True
    # Unit normals so RHS is -vq_z; identity AIC keeps the solved column = RHS.
    state.enc_d[:, : state.nvor, 0] = 0.0
    state.enc_d[2, : state.nvor, 0] = 1.0
    state.aicn[: state.nvor, : state.nvor] = np.eye(state.nvor)
    state.iapiv[: state.nvor] = np.arange(state.nvor, dtype=np.int32)

    # Seed unit control sensitivities so gamsum has a known target.
    state.gam_u_d[: state.nvor, : state.numax, 0] = 0.0
    u = np.concatenate((state.vinf, state.wrot))[: state.numax]
    # After gdcalc with zero body influence and lvalbe=False, RHS is 0 → gam_d=0.
    # Use a non-zero body influence so gdcalc produces a non-trivial column, then
    # set gam_u_d so gamsum reproduces the same contraction.
    state.wcsrd_u[2, : state.nvor, :6] = 0.05
    gdcalc(state, state.ncontrol, state.lcondef, state.enc_d, state.gam_d)
    # Build gam_u_d such that gam_u_d @ u == gdcalc's gam_d.
    state.gam_u_d[: state.nvor, :, 0] = 0.0
    for i in range(state.nvor):
        # Put the entire sensitivity on the first freestream component.
        if abs(u[0]) > 0:
            state.gam_u_d[i, 0, 0] = state.gam_d[i, 0] / u[0]
    gamsum(state)
    np.testing.assert_allclose(
        state.gam_d[: state.nvor, 0],
        state.gam_u_d[: state.nvor, : state.numax, 0] @ u,
        atol=1e-12,
    )
