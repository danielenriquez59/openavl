"""Trim setup (port of AVL atrim.f TRMSET core)."""

from __future__ import annotations

import math

from openavl import constants as C
from openavl.core.state import AVLState


def setup_trim(state: AVLState, mode: int, ir: int = 0) -> None:
    """Configure trim constraints for a run case.

    Parameters
    ----------
    state:
        Solver state to update in place.
    mode:
        ``1`` for level flight or banked turn, ``2`` for pull-up maneuver.
    ir:
        Run-case index (default single run case ``0``).
    """
    ktrim = int(mode)
    if ktrim not in (1, 2):
        raise ValueError(f"Unsupported trim mode: {mode}")

    if ktrim in (1, 2) and state.parval[C.IPCL, ir] == 0.0:
        for iv in range(state.nvtot):
            if state.icon[iv, ir] == C.ICCL:
                state.parval[C.IPCL, ir] = state.conval[C.ICCL, ir]
                break

    if state.parval[C.IPRHO, ir] <= 0.0:
        state.parval[C.IPRHO, ir] = state.rho0
    if state.parval[C.IPGEE, ir] <= 0.0:
        state.parval[C.IPGEE, ir] = state.gee0
    if state.parval[C.IPMASS, ir] <= 0.0:
        state.parval[C.IPMASS, ir] = state.rmass0

    cref_d = state.cref * state.unitl
    bref_d = state.bref * state.unitl
    sref_d = state.sref * state.unitl * state.unitl

    phi = state.parval[C.IPPHI, ir]
    cl = state.parval[C.IPCL, ir]
    vee = state.parval[C.IPVEE, ir]
    rad = state.parval[C.IPRAD, ir]
    rho = state.parval[C.IPRHO, ir]
    gee = state.parval[C.IPGEE, ir]
    fac = state.parval[C.IPFAC, ir]
    rmass = state.parval[C.IPMASS, ir]

    sinp = math.sin(phi * state.dtr)
    cosp = math.cos(phi * state.dtr)

    if ktrim == 1:
        if vee <= 0.0 and cl > 0.0:
            vee = math.sqrt(2.0 * rmass * gee / (rho * sref_d * cl * cosp))
            state.parval[C.IPVEE, ir] = vee
        if cl <= 0.0 and vee > 0.0:
            cl = 2.0 * rmass * gee / (rho * sref_d * vee * vee * cosp)
            state.parval[C.IPCL, ir] = cl

        if sinp == 0.0:
            rad = 0.0
        else:
            rad = vee * vee * cosp / (gee * sinp)
        state.parval[C.IPRAD, ir] = rad

        fac = 1.0 / cosp if cosp != 0.0 else fac
        state.parval[C.IPFAC, ir] = fac
        state.parval[C.IPTHE, ir] = 0.0

        whx = 0.0
        why = 0.0
        whz = 0.0
        if rad > 0.0:
            why = sinp * cref_d / (2.0 * rad)
            whz = cosp * bref_d / (2.0 * rad)

        state.conval[C.ICCL, ir] = cl
        state.conval[C.ICROTX, ir] = whx
        state.conval[C.ICROTY, ir] = why
        state.conval[C.ICROTZ, ir] = whz

        state.icon[C.IVALFA, ir] = C.ICCL
        state.icon[C.IVROTX, ir] = C.ICROTX
        state.icon[C.IVROTY, ir] = C.ICROTY
        state.icon[C.IVROTZ, ir] = C.ICROTZ

    elif ktrim == 2:
        if rad == 0.0 and cl > 0.0:
            rad = rmass / (0.5 * rho * sref_d * cl)
            state.parval[C.IPRAD, ir] = rad
        if rad > 0.0 and cl == 0.0:
            cl = rmass / (0.5 * rho * sref_d * rad)
            state.parval[C.IPCL, ir] = cl
        if fac == 0.0 and cl > 0.0 and vee > 0.0 and gee > 0.0:
            fac = 0.5 * rho * vee * vee * sref_d * cl / (rmass * gee)
            state.parval[C.IPFAC, ir] = fac
        if fac > 0.0 and cl > 0.0 and vee == 0.0 and gee > 0.0:
            vee = math.sqrt(fac * rmass * gee / (0.5 * rho * sref_d * cl))
            state.parval[C.IPVEE, ir] = vee

        state.parval[C.IPTHE, ir] = 0.0

        whx = 0.0
        why = 0.0
        whz = 0.0
        if rad > 0.0:
            why = cref_d / (2.0 * rad)

        state.conval[C.ICCL, ir] = cl
        state.conval[C.ICROTX, ir] = whx
        state.conval[C.ICROTY, ir] = why
        state.conval[C.ICROTZ, ir] = whz

        state.icon[C.IVALFA, ir] = C.ICCL
        state.icon[C.IVROTX, ir] = C.ICROTX
        state.icon[C.IVROTY, ir] = C.ICROTY
        state.icon[C.IVROTZ, ir] = C.ICROTZ
