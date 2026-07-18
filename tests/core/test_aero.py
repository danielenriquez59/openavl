"""Tests for openavl.aero."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from openavl.aero import aero, vinfab
from openavl.constants import NUMAX

from tests.helpers import FIXTURES_DIR, REF_DIR

FIXTURE_PATH = FIXTURES_DIR / "aero_js_expected.json"

pytestmark = pytest.mark.core


def _load_expected() -> dict:
    with FIXTURE_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def assert_close_array(actual, expected, tol: float = 1e-4) -> None:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape
    diff = np.abs(actual - expected)
    assert np.all(diff <= tol), f"max diff {diff.max()} > {tol}"


@dataclass
class AeroTestState:
    """Minimal 0-based state matching the JS/Fortran aero_ref fixture."""

    pi: float = np.pi
    alfa: float = np.float64(0.1)
    beta: float = np.float64(0.05)
    mach: float = np.float64(0.3)
    amach: float = np.float64(0.3)
    iysym: int = 0
    izsym: int = 0
    ysym: float = 0.0
    zsym: float = 0.0
    vrcorec: float = 0.01
    vrcorew: float = 0.02
    sref: float = 1.5
    cref: float = 1.0
    bref: float = 2.0
    cdref: float = 0.02
    xyzref: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    vinf: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    vinf_a: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    vinf_b: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    wrot: np.ndarray = field(default_factory=lambda: np.array([0.01, -0.02, 0.03], dtype=np.float64))

    nstrip: int = 1
    nvor: int = 1
    nsurf: int = 1
    nbody: int = 0
    ncontrol: int = 0
    ndesign: int = 0
    numax: int = NUMAX

    ltrforce: bool = False
    lnfld_wv: bool = False
    lvisc: bool = False

    def __post_init__(self) -> None:
        i0 = 0

        self.lviscstrp = np.zeros(self.nstrip, dtype=bool)
        self.lstripoff = np.zeros(self.nstrip, dtype=bool)

        self.ijfrst = np.array([0], dtype=np.int32)
        self.nvstrp = np.array([1], dtype=np.int32)
        self.jfrst = np.array([0], dtype=np.int32)
        self.nj = np.array([1], dtype=np.int32)
        self.lssurf = np.array([0], dtype=np.int32)
        self.imags = np.array([1], dtype=np.int32)
        self.lfload = np.array([True], dtype=bool)
        self.lncomp = np.array([1], dtype=np.int32)

        self.chord = np.array([1.0], dtype=np.float64)
        self.wstrip = np.array([0.5], dtype=np.float64)
        self.chord1 = np.array([1.0], dtype=np.float64)
        self.chord2 = np.array([1.0], dtype=np.float64)
        self.rle1 = np.zeros((3, self.nstrip), dtype=np.float64)
        self.rle2 = np.zeros((3, self.nstrip), dtype=np.float64)
        self.rle = np.zeros((3, self.nstrip), dtype=np.float64)
        self.ensy = np.zeros(self.nstrip, dtype=np.float64)
        self.ensz = np.array([1.0], dtype=np.float64)
        self.ess = np.zeros((3, self.nstrip), dtype=np.float64)
        self.ainc = np.zeros(self.nstrip, dtype=np.float64)
        self.xsref = np.array([0.25], dtype=np.float64)
        self.ysref = np.zeros(self.nstrip, dtype=np.float64)
        self.zsref = np.zeros(self.nstrip, dtype=np.float64)
        self.ssurf = np.array([0.5], dtype=np.float64)
        self.cavesurf = np.array([1.0], dtype=np.float64)
        self.clmax_surf = np.zeros(self.nsurf, dtype=np.float64)

        self.rv1 = np.zeros((3, self.nvor), dtype=np.float64)
        self.rv2 = np.zeros((3, self.nvor), dtype=np.float64)
        self.rv = np.zeros((3, self.nvor), dtype=np.float64)
        self.rc = np.zeros((3, self.nvor), dtype=np.float64)
        self.dxv = np.array([1.0], dtype=np.float64)
        self.env = np.zeros((3, self.nvor), dtype=np.float64)

        self.vv = np.zeros((3, self.nvor), dtype=np.float64)
        self.vv_u = np.zeros((3, self.nvor, NUMAX), dtype=np.float64)
        self.wv = np.zeros((3, self.nvor), dtype=np.float64)
        self.wv_u = np.zeros((3, self.nvor, NUMAX), dtype=np.float64)

        self.gam = np.array([0.4], dtype=np.float64)
        self.gam_u = np.zeros((self.nvor, NUMAX), dtype=np.float64)
        for n in range(NUMAX):
            self.gam_u[i0, n] = 0.01 * (n + 1)

        self.dcp = np.zeros(self.nvor, dtype=np.float64)
        self.dcp_u = np.zeros((self.nvor, NUMAX), dtype=np.float64)
        self.cnc = np.zeros(self.nstrip, dtype=np.float64)
        self.cnc_u = np.zeros((self.nstrip, NUMAX), dtype=np.float64)

        self.cf_lstrp = np.zeros((3, self.nstrip), dtype=np.float64)
        self.cm_lstrp = np.zeros((3, self.nstrip), dtype=np.float64)
        self.cfstrp = np.zeros((3, self.nstrip), dtype=np.float64)
        self.cmstrp = np.zeros((3, self.nstrip), dtype=np.float64)
        self.cdstrp = np.zeros(self.nstrip, dtype=np.float64)
        self.cystrp = np.zeros(self.nstrip, dtype=np.float64)
        self.clstrp = np.zeros(self.nstrip, dtype=np.float64)
        self.cdst_a = np.zeros(self.nstrip, dtype=np.float64)
        self.cyst_a = np.zeros(self.nstrip, dtype=np.float64)
        self.clst_a = np.zeros(self.nstrip, dtype=np.float64)
        self.cdst_u = np.zeros((self.nstrip, NUMAX), dtype=np.float64)
        self.cyst_u = np.zeros((self.nstrip, NUMAX), dtype=np.float64)
        self.clst_u = np.zeros((self.nstrip, NUMAX), dtype=np.float64)
        self.cfst_u = np.zeros((3, self.nstrip, NUMAX), dtype=np.float64)
        self.cmst_u = np.zeros((3, self.nstrip, NUMAX), dtype=np.float64)

        self.cl_lstrp = np.zeros(self.nstrip, dtype=np.float64)
        self.cd_lstrp = np.zeros(self.nstrip, dtype=np.float64)
        self.cmc4_lstrp = np.zeros(self.nstrip, dtype=np.float64)
        self.ca_lstrp = np.zeros(self.nstrip, dtype=np.float64)
        self.cn_lstrp = np.zeros(self.nstrip, dtype=np.float64)
        self.clt_lstrp = np.zeros(self.nstrip, dtype=np.float64)
        self.cla_lstrp = np.zeros(self.nstrip, dtype=np.float64)
        self.cmle_lstrp = np.zeros(self.nstrip, dtype=np.float64)
        self.cdv_lstrp = np.zeros(self.nstrip, dtype=np.float64)

        self.cf_lsrf = np.zeros((3, self.nsurf), dtype=np.float64)
        self.cm_lsrf = np.zeros((3, self.nsurf), dtype=np.float64)
        self.cdsurf = np.zeros(self.nsurf, dtype=np.float64)
        self.cysurf = np.zeros(self.nsurf, dtype=np.float64)
        self.clsurf = np.zeros(self.nsurf, dtype=np.float64)
        self.cfsurf = np.zeros((3, self.nsurf), dtype=np.float64)
        self.cmsurf = np.zeros((3, self.nsurf), dtype=np.float64)
        self.cdvsurf = np.zeros(self.nsurf, dtype=np.float64)
        self.cds_a = np.zeros(self.nsurf, dtype=np.float64)
        self.cys_a = np.zeros(self.nsurf, dtype=np.float64)
        self.cls_a = np.zeros(self.nsurf, dtype=np.float64)
        self.cds_u = np.zeros((self.nsurf, NUMAX), dtype=np.float64)
        self.cys_u = np.zeros((self.nsurf, NUMAX), dtype=np.float64)
        self.cls_u = np.zeros((self.nsurf, NUMAX), dtype=np.float64)
        self.cfs_u = np.zeros((3, self.nsurf, NUMAX), dtype=np.float64)
        self.cms_u = np.zeros((3, self.nsurf, NUMAX), dtype=np.float64)
        self.cl_lsrf = np.zeros(self.nsurf, dtype=np.float64)
        self.cd_lsrf = np.zeros(self.nsurf, dtype=np.float64)
        self.clcd = np.zeros((self.nstrip, 6), dtype=np.float64)

        self.chinge = np.zeros(0, dtype=np.float64)
        self.cdtot = 0.0
        self.cytot = 0.0
        self.cltot = 0.0
        self.cftot = np.zeros(3, dtype=np.float64)
        self.cmtot = np.zeros(3, dtype=np.float64)
        self.cdvtot = 0.0
        self.cdtot_a = 0.0
        self.cltot_a = 0.0
        self.cdtot_u = np.zeros(NUMAX, dtype=np.float64)
        self.cytot_u = np.zeros(NUMAX, dtype=np.float64)
        self.cltot_u = np.zeros(NUMAX, dtype=np.float64)
        self.cftot_u = np.zeros((3, NUMAX), dtype=np.float64)
        self.cmtot_u = np.zeros((3, NUMAX), dtype=np.float64)

        self.clff = 0.0
        self.cyff = 0.0
        self.cdff = 0.0
        self.spanef = 0.0
        self.clff_u = np.zeros(NUMAX, dtype=np.float64)
        self.cyff_u = np.zeros(NUMAX, dtype=np.float64)
        self.cdff_u = np.zeros(NUMAX, dtype=np.float64)
        self.spanef_u = np.zeros(NUMAX, dtype=np.float64)
        self.dwwake = np.zeros(self.nstrip, dtype=np.float64)

        self.rle2[1, 0] = 1.0
        self.ess[1, 0] = 1.0
        self.rv1[1, i0] = -0.5
        self.rv2[0, i0] = 1.0
        self.rv2[1, i0] = 0.5
        self.rv[0, i0] = 0.5
        self.rc[0, i0] = 0.25
        self.env[2, i0] = 1.0
        self.vv[:, i0] = [0.01, 0.02, 0.03]


@pytest.mark.fixture
def test_vinfab_matches_js_fixture():
    expected = _load_expected()
    state = AeroTestState()
    vinfab(state)
    assert_close_array(state.vinf, expected["vinf"])
    assert_close_array(state.vinf_a, expected["vinf_a"])
    assert_close_array(state.vinf_b, expected["vinf_b"])


@pytest.mark.fixture
def test_aero_matches_js_fixture():
    expected = _load_expected()
    state = AeroTestState()
    vinfab(state)
    aero(state)

    assert_close_array([state.cdtot, state.cytot, state.cltot], [expected["cdtot"], expected["cytot"], expected["cltot"]])
    assert_close_array(state.cftot, expected["cftot"])
    assert_close_array(state.cmtot, expected["cmtot"])
    assert_close_array(
        [state.cdvtot, state.clff, state.cyff, state.cdff, state.spanef],
        [expected["cdvtot"], expected["clff"], expected["cyff"], expected["cdff"], expected["spanef"]],
    )
    assert_close_array([state.dcp[0], state.cnc[0]], [expected["dcp0"], expected["cnc0"]])
    assert_close_array(state.cfstrp[:, 0], expected["cfstrp"])
    assert_close_array(state.cmstrp[:, 0], expected["cmstrp"])
    assert_close_array(
        [state.cdstrp[0], state.cystrp[0], state.clstrp[0]],
        [expected["cdstrp0"], expected["cystrp0"], expected["clstrp0"]],
    )
    assert_close_array(
        [state.cdvsurf[0], state.cdsurf[0], state.cysurf[0], state.clsurf[0]],
        [expected["cdvsurf0"], expected["cdsurf0"], expected["cysurf0"], expected["clsurf0"]],
    )


def _run_fortran_ref(bin_name: str) -> list[float]:
    bin_path = REF_DIR / bin_name
    if not bin_path.exists():
        exe_path = REF_DIR / f"{bin_name}.exe"
        if exe_path.exists():
            bin_path = exe_path
        else:
            pytest.skip(f"Fortran reference binary not found under {REF_DIR}")
    try:
        proc = subprocess.run([str(bin_path)], capture_output=True, text=True, check=False)
    except OSError as exc:
        pytest.skip(f"Cannot execute Fortran reference binary on this platform: {exc}")
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or f"ref exited with {proc.returncode}")
    return [float(x) for x in proc.stdout.strip().split()]


@pytest.mark.reference
def test_aero_matches_fortran_reference():
    """Compare against tests/data/avl/ref/aero_ref when built."""
    ref = _run_fortran_ref("aero_ref")
    offset = 0
    ref_scalars = ref[offset : offset + 3]
    offset += 3
    ref_cftot = ref[offset : offset + 3]
    offset += 3
    ref_cmtot = ref[offset : offset + 3]
    offset += 3
    ref_more = ref[offset : offset + 5]
    offset += 5
    ref_dcp = ref[offset : offset + 2]
    offset += 2
    ref_cfstrp = ref[offset : offset + 3]
    offset += 3
    ref_cmstrp = ref[offset : offset + 3]
    offset += 3
    ref_strip = ref[offset : offset + 3]
    offset += 3
    ref_surf = ref[offset : offset + 4]

    state = AeroTestState()
    vinfab(state)
    aero(state)

    assert_close_array([state.cdtot, state.cytot, state.cltot], ref_scalars)
    assert_close_array(state.cftot, ref_cftot)
    assert_close_array(state.cmtot, ref_cmtot)
    assert_close_array([state.cdvtot, state.clff, state.cyff, state.cdff, state.spanef], ref_more)
    assert_close_array([state.dcp[0], state.cnc[0]], ref_dcp)
    assert_close_array(state.cfstrp[:, 0], ref_cfstrp)
    assert_close_array(state.cmstrp[:, 0], ref_cmstrp)
    assert_close_array([state.cdstrp[0], state.cystrp[0], state.clstrp[0]], ref_strip)
    assert_close_array([state.cdvsurf[0], state.cdsurf[0], state.cysurf[0], state.clsurf[0]], ref_surf)


class _EnvGVortexState:
    """Minimal state for ``_accumulate_all_vortex_forces`` (B7 env_g check).

    A single bound vortex (span along y, at the origin) with a freestream
    purely in z, so the vortex force ``f = veff x g_vec`` points along -x --
    orthogonal to the baseline normal ``env=[0,0,1]``, which makes the
    (currently always-zero) ``env_g`` contribution to ``dcp_g`` the
    *dominant* term rather than a small correction, so the missing-term bug
    is impossible to miss.
    """

    def __init__(self, gam_val: float, env_vec: np.ndarray) -> None:
        self.nvor = 1
        self.nstrip = 1
        self.vortex_to_strip = np.array([0], dtype=np.intp)
        self.rle = np.zeros((3, 1), dtype=np.float64)
        self.chord = np.array([1.0], dtype=np.float64)
        self.wstrip = np.array([1.0], dtype=np.float64)
        self.ensy = np.array([0.0], dtype=np.float64)
        self.ensz = np.array([1.0], dtype=np.float64)
        self.lstripoff = np.zeros(1, dtype=bool)
        self.rv1 = np.array([[0.0], [-0.5], [0.0]], dtype=np.float64)
        self.rv2 = np.array([[0.0], [0.5], [0.0]], dtype=np.float64)
        self.rv = np.zeros((3, 1), dtype=np.float64)
        self.xyzref = np.zeros(3, dtype=np.float64)
        self.wrot = np.zeros(3, dtype=np.float64)
        self.vinf = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.env = np.asarray(env_vec, dtype=np.float64).reshape(3, 1)
        self.dxv = np.array([1.0], dtype=np.float64)
        self.dcp = np.zeros(1, dtype=np.float64)
        self.dcp_u = np.zeros((1, NUMAX), dtype=np.float64)
        self.dcp_d = np.zeros((1, 1), dtype=np.float64)
        self.dcp_g = np.zeros((1, 1), dtype=np.float64)
        self._gam = np.array([gam_val], dtype=np.float64)

    def dcp_at(self, gam_val: float, env_vec: np.ndarray, *, gam_g: float = 0.0, env_g_true: np.ndarray | None = None):
        """Run ``_accumulate_all_vortex_forces`` and return ``(dcp, dcp_g)``."""
        from openavl.aero.forces import _accumulate_all_vortex_forces

        self.env = np.asarray(env_vec, dtype=np.float64).reshape(3, 1)
        gam = np.array([gam_val], dtype=np.float64)
        gam_u = np.zeros((1, NUMAX), dtype=np.float64)
        gam_d = np.zeros((1, 1), dtype=np.float64)
        gam_g_arr = np.array([[gam_g]], dtype=np.float64)
        vv = np.zeros((3, 1), dtype=np.float64)
        vv_u = np.zeros((3, 1, NUMAX), dtype=np.float64)
        vv_d = np.zeros((3, 1, 1), dtype=np.float64)
        vv_g = np.zeros((3, 1, 1), dtype=np.float64)
        wv = np.zeros((3, 1), dtype=np.float64)
        wv_u = np.zeros((3, 1, NUMAX), dtype=np.float64)
        wv_d = np.zeros((3, 1, 1), dtype=np.float64)
        wv_g = np.zeros((3, 1, 1), dtype=np.float64)
        env_d = np.zeros((3, 1, 1), dtype=np.float64)
        env_g = np.zeros((3, 1, 1), dtype=np.float64)
        if env_g_true is not None:
            env_g[:, 0, 0] = env_g_true
        _accumulate_all_vortex_forces(
            self, NUMAX, 0, 1,
            gam, gam_u, gam_d, gam_g_arr,
            vv, vv_u, vv_d, vv_g,
            wv, wv_u, wv_d, wv_g,
            env_d, env_g, False,
        )
        return float(self.dcp[0]), float(self.dcp_g[0, 0])


@pytest.mark.xfail(
    reason=(
        "B7: env_g (bound-vortex normal-vector sensitivity to design "
        "variables) is always zero -- see the 'term2 uses env_g' comment in "
        "aero/forces.py -- so dcp_g misses the normal-rotation term for any "
        "twist-type design variable. Remove this xfail once env_g is "
        "populated in geometry.py."
    ),
    strict=True,
)
def test_dcp_g_matches_fd_for_twist_design_variable():
    """FD dcp w.r.t. a twist design variable must match the analytic dcp_g.

    Models a design variable that rotates the bound-vortex normal about the
    span axis (``env(g) = [sin(g), 0, cos(g)]``, a "twist") while also
    linearly perturbing the circulation (``gam(g) = gam0 + gam_g0 * g``, the
    part already captured via ``gam_g``/``enc_g``). The true d(dcp)/dg at
    g=0 includes both the ``env * fgam_g`` term (already implemented) and
    the ``env_g * fgam`` term (currently always zero); central finite
    differences of the actual primal ``dcp`` expose the second term.
    """
    gam0 = 0.5
    gam_g0 = 0.2
    env0 = np.array([0.0, 0.0, 1.0])

    probe = _EnvGVortexState(gam0, env0)

    eps = 1.0e-6

    def gam_of(g: float) -> float:
        return gam0 + gam_g0 * g

    def env_of(g: float) -> np.ndarray:
        return np.array([math.sin(g), 0.0, math.cos(g)])

    dcp_plus, _ = probe.dcp_at(gam_of(eps), env_of(eps))
    dcp_minus, _ = probe.dcp_at(gam_of(-eps), env_of(-eps))
    dcp_g_fd = (dcp_plus - dcp_minus) / (2.0 * eps)

    _, dcp_g_today = probe.dcp_at(gam0, env0, gam_g=gam_g0, env_g_true=None)
    np.testing.assert_allclose(dcp_g_today, dcp_g_fd, rtol=1e-6, atol=1e-9)


def test_dcp_g_matches_fd_once_env_g_is_supplied():
    """Sanity check: supplying the true env_g fixes the B7 gap.

    Same setup as ``test_dcp_g_matches_fd_for_twist_design_variable``, but
    passing the correct (hand-derived) ``env_g`` into
    ``_accumulate_all_vortex_forces`` shows the formula itself is right --
    only the missing upstream population of ``env_g`` (in geometry.py) needs
    to be fixed.
    """
    gam0 = 0.5
    gam_g0 = 0.2
    env0 = np.array([0.0, 0.0, 1.0])
    env_g0_true = np.array([1.0, 0.0, 0.0])

    probe = _EnvGVortexState(gam0, env0)

    eps = 1.0e-6

    def gam_of(g: float) -> float:
        return gam0 + gam_g0 * g

    def env_of(g: float) -> np.ndarray:
        return np.array([math.sin(g), 0.0, math.cos(g)])

    dcp_plus, _ = probe.dcp_at(gam_of(eps), env_of(eps))
    dcp_minus, _ = probe.dcp_at(gam_of(-eps), env_of(-eps))
    dcp_g_fd = (dcp_plus - dcp_minus) / (2.0 * eps)

    _, dcp_g_fixed = probe.dcp_at(gam0, env0, gam_g=gam_g0, env_g_true=env_g0_true)
    np.testing.assert_allclose(dcp_g_fixed, dcp_g_fd, rtol=1e-6, atol=1e-9)
