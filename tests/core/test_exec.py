"""Tests for openavl.exec."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openavl import constants as C
from openavl.exec import exec_solve
from openavl.state import AVLState
from tests.helpers import REF_DIR

pytestmark = pytest.mark.core


def _make_exec_state() -> AVLState:
    nvor = 1
    ndmax = 1
    ngmax = 1
    state = AVLState(
        nvor=nvor,
        nvmax=nvor,
        nlnode=0,
        ncontrol=0,
        ndesign=0,
        numax=C.NUMAX,
        ndmax=ndmax,
        ngmax=ngmax,
        nlmax=1,
        nstrip=0,
        nsurf=0,
        laic=True,
        lsrd=True,
        lvel=True,
        lnasa_sa=False,
        mach=0.2,
        amach=0.2,
        alfa=0.1,
        beta=0.05,
        wrot=np.array([0.01, -0.02, 0.03], dtype=np.float64),
        sref=1.0,
        cref=1.0,
        bref=1.0,
    )
    state._allocate_arrays(ndmax, ngmax, 1, nvor, 1)
    ir = 0
    state.parval[C.IPMACH, ir] = 0.2
    state.parval[C.IPCD0, ir] = 0.01
    state.parval[C.IPXCG, ir] = 0.1
    state.parval[C.IPYCG, ir] = 0.2
    state.parval[C.IPZCG, ir] = 0.3
    state.aicn[0, 0] = 1.0
    state.iapiv[0] = 0
    state.lvnc[0] = True
    state.lvalbe[0] = False
    state.enc[2, 0] = 1.0
    state.wc_gam[:, 0, 0] = [0.1, 0.2, 0.3]
    state.wv_gam[:, 0, 0] = [0.2, 0.4, 0.6]
    return state


def _extract_outputs(state: AVLState) -> list[float]:
    return [
        float(state.vinf[0]),
        float(state.vinf[1]),
        float(state.vinf[2]),
        float(state.wrot[0]),
        float(state.wrot[1]),
        float(state.wrot[2]),
        float(state.gam[0]),
        float(state.wc[0, 0]),
        float(state.wc[1, 0]),
        float(state.wc[2, 0]),
        float(state.wv[0, 0]),
        float(state.wv[1, 0]),
        float(state.wv[2, 0]),
        float(state.parval[C.IPALFA, 0]),
        float(state.parval[C.IPBETA, 0]),
        float(state.parval[C.IPCL, 0]),
    ]


@pytest.mark.reference
def test_exec_solve_matches_fortran_ref():
    ref_path = REF_DIR / "exec_ref"
    if not (ref_path.is_file() or (REF_DIR / "exec_ref.exe").is_file()):
        pytest.skip("Fortran reference binary not found: exec_ref")
    from tests.helpers import run_ref_binary

    ref = run_ref_binary(ref_path if ref_path.is_file() else REF_DIR / "exec_ref.exe")
    state = _make_exec_state()
    exec_solve(state, niter=0)
    actual = _extract_outputs(state)
    np.testing.assert_allclose(actual, ref, atol=1e-4)


def test_exec_solve_failed_trim_skips_parval_lsen_update():
    """Exhausted Newton iterations must not set lsen or overwrite parval."""
    import warnings

    from openavl import AVLSolver
    from openavl.geometry import Aircraft

    aircraft = Aircraft(name="TrimFail", sref=10.0, cref=1.0, bref=10.0)
    wing = aircraft.add_wing("W", n_chord=4, n_span=6)
    wing.add_section(xyzle=[0.0, 0.0, 0.0], chord=1.0).set_airfoil_naca("0012")
    wing.add_section(xyzle=[0.0, 5.0, 0.0], chord=1.0).set_airfoil_naca("0012")

    # Impossible CL so the Newton loop cannot converge.
    solver = AVLSolver(aircraft, cl=50.0, beta=0.0, mach=0.0)
    ir = 0
    state = solver.state
    state.parval[C.IPALFA, ir] = -99.0
    state.parval[C.IPBETA, ir] = -88.0
    state.parval[C.IPCL, ir] = -77.0
    state.lsen = False

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver.execute_run(max_iter=3)

    msgs = [str(w.message) for w in caught]
    assert any(
        ("Trim convergence failed" in m) or ("Trim aborted" in m) for m in msgs
    ), msgs
    assert state.lsol is False
    assert state.lsen is False
    assert state.parval[C.IPALFA, ir] == pytest.approx(-99.0)
    assert state.parval[C.IPBETA, ir] == pytest.approx(-88.0)
    assert state.parval[C.IPCL, ir] == pytest.approx(-77.0)
