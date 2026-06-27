"""Tests for openavl.trefftz."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openavl import constants as C
from openavl.trefftz import tpforc
from tests.helpers import REF_DIR


class _TpforcState:
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
    numax = C.NUMAX
    sref = np.float64(1.5)
    bref = np.float64(2.0)

    def __init__(self) -> None:
        self.ijfrst = np.array([0, 1], dtype=np.int32)
        self.nvstrp = np.array([1, 1], dtype=np.int32)
        self.gam = np.array([0.0, 0.5, 0.3], dtype=np.float64)
        self.gam_u = np.zeros((3, C.NUMAX), dtype=np.float64)
        for i in range(1, 3):
            for n in range(C.NUMAX):
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


pytestmark = pytest.mark.core


@pytest.mark.reference
def test_tpforc_matches_fortran_ref():
    ref_path = REF_DIR / "atpforc_ref"
    if not (ref_path.is_file() or (REF_DIR / "atpforc_ref.exe").is_file()):
        pytest.skip("Fortran reference binary not found: atpforc_ref")
    from tests.helpers import run_ref_binary

    ref = run_ref_binary(ref_path if ref_path.is_file() else REF_DIR / "atpforc_ref.exe")
    offset = 0
    ref_scalars = ref[offset : offset + 4]
    offset += 4
    ref_dwwake = ref[offset : offset + 2]
    offset += 2
    ref_clff_u = ref[offset : offset + 6]
    offset += 6
    ref_cyff_u = ref[offset : offset + 6]
    offset += 6
    ref_cdff_u = ref[offset : offset + 6]
    offset += 6
    ref_span_u = ref[offset : offset + 6]

    state = _TpforcState()
    tpforc(state)
    scalars = [state.clff, state.cyff, state.cdff, state.spanef]

    np.testing.assert_allclose(scalars, ref_scalars, atol=1e-5)
    np.testing.assert_allclose(state.dwwake, ref_dwwake, atol=1e-5)
    np.testing.assert_allclose(state.clff_u, ref_clff_u, atol=1e-5)
    np.testing.assert_allclose(state.cyff_u, ref_cyff_u, atol=1e-5)
    np.testing.assert_allclose(state.cdff_u, ref_cdff_u, atol=1e-5)
    np.testing.assert_allclose(state.spanef_u, ref_span_u, atol=1e-4)
