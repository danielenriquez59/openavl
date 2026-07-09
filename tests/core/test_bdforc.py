"""Finite-difference checks for body-force (BDFORC) sensitivities."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from openavl.aero.forces import bdforc
from openavl.constants import NUMAX

pytestmark = pytest.mark.core


@dataclass
class _BodyOnlyState:
    """Minimal state for exercising ``bdforc`` at Mach > 0.

    The body segment is angled in y so freestream-u perturbations produce a
    nonzero normal-velocity sensitivity (a pure-x segment cancels ``un_u``).
    """

    mach: float = 0.5
    alfa: float = 0.1
    sref: float = 10.0
    cref: float = 1.0
    bref: float = 5.0
    nbody: int = 1
    numax: int = NUMAX
    xyzref: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float64)
    )
    vinf: np.ndarray = field(
        default_factory=lambda: np.array([0.995004165, 0.0, 0.0998334166], dtype=np.float64)
    )
    wrot: np.ndarray = field(
        default_factory=lambda: np.array([0.02, -0.03, 0.04], dtype=np.float64)
    )

    def __post_init__(self) -> None:
        self.nl = np.array([2], dtype=np.int32)
        self.lfrst = np.array([0], dtype=np.int32)
        self.rl = np.zeros((3, 2), dtype=np.float64)
        self.rl[:, 0] = [0.0, 0.0, 0.0]
        self.rl[:, 1] = [2.0, 0.5, 0.25]
        self.radl = np.array([0.2, 0.15], dtype=np.float64)
        self.src = np.array([0.5, 0.0], dtype=np.float64)
        # Hold source fixed w.r.t. u so the FD isolates the veff_u / betm path.
        self.src_u = np.zeros((2, 6), dtype=np.float64)
        self.dcpb = np.zeros((3, 2), dtype=np.float64)
        self.cdbdy = np.zeros(1, dtype=np.float64)
        self.cybdy = np.zeros(1, dtype=np.float64)
        self.clbdy = np.zeros(1, dtype=np.float64)
        self.cfbdy = np.zeros((3, 1), dtype=np.float64)
        self.cmbdy = np.zeros((3, 1), dtype=np.float64)
        self._reset_totals()

    def _reset_totals(self) -> None:
        """Zero force totals that ``bdforc`` accumulates into."""
        self.cdtot = 0.0
        self.cytot = 0.0
        self.cltot = 0.0
        self.cftot = np.zeros(3, dtype=np.float64)
        self.cmtot = np.zeros(3, dtype=np.float64)
        self.cdtot_u = np.zeros(6, dtype=np.float64)
        self.cytot_u = np.zeros(6, dtype=np.float64)
        self.cltot_u = np.zeros(6, dtype=np.float64)
        self.cftot_u = np.zeros((3, 6), dtype=np.float64)
        self.cmtot_u = np.zeros((3, 6), dtype=np.float64)


def _cdbdy(state: _BodyOnlyState) -> float:
    """Return body CD after a fresh ``bdforc`` evaluation."""
    state._reset_totals()
    bdforc(state)
    return float(state.cdbdy[0])


def test_bdforc_veff_u_matches_fd_at_mach():
    """All six VEFF_x sensitivities include 1/betm consistently (A3)."""
    state = _BodyOnlyState()
    assert state.mach > 0.0
    betm = np.sqrt(1.0 - state.mach * state.mach)
    assert betm < 1.0

    state._reset_totals()
    bdforc(state)
    analytic = state.cdtot_u.copy()
    assert np.any(np.abs(analytic) > 1e-8), "test setup must produce nonzero CD_u"

    eps = 1e-6
    fd = np.zeros(6, dtype=np.float64)
    for k in range(3):
        v0 = state.vinf[k]
        state.vinf[k] = v0 + eps
        cd_p = _cdbdy(state)
        state.vinf[k] = v0 - eps
        cd_m = _cdbdy(state)
        state.vinf[k] = v0
        fd[k] = (cd_p - cd_m) / (2.0 * eps)
    for k in range(3):
        w0 = state.wrot[k]
        state.wrot[k] = w0 + eps
        cd_p = _cdbdy(state)
        state.wrot[k] = w0 - eps
        cd_m = _cdbdy(state)
        state.wrot[k] = w0
        fd[k + 3] = (cd_p - cd_m) / (2.0 * eps)

    np.testing.assert_allclose(analytic, fd, rtol=1e-4, atol=1e-6)
