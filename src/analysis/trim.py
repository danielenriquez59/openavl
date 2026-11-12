"""Trim setup (port of AVL atrim.f TRMSET core)."""

from __future__ import annotations

import math

from openavl import constants as C
from openavl.core.state import AVLState


# Run-case values AVL's trim menu zeroes when one parameter is edited, so the
# next TRMSET pass recomputes them (atrim.f, C1/C2 menu). Mode 1 holds CL:
# changing bank, mass, density or gravity recomputes V; changing V recomputes CL.
TRIM_DEPENDENTS: dict[int, dict[int, tuple[int, ...]]] = {
    1: {
        C.IPPHI: (C.IPVEE,),
        C.IPCL: (C.IPVEE, C.IPRAD),
        C.IPVEE: (C.IPCL,),
        C.IPMASS: (C.IPVEE,),
        C.IPRHO: (C.IPVEE,),
        C.IPGEE: (C.IPVEE,),
    },
    2: {
        C.IPCL: (C.IPRAD, C.IPFAC),
        C.IPVEE: (C.IPFAC,),
        C.IPMASS: (C.IPRAD, C.IPFAC),
        C.IPRHO: (C.IPRAD, C.IPFAC),
        C.IPGEE: (C.IPFAC,),
    },
}


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

    # Entering the trim menu (atrim.f label 100): take CL from the CL
    # constraint if the run case has none, and fill in missing defaults.
    if state.parval[C.IPCL, ir] == 0.0:
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

    apply_trim(state, ktrim, ir)


def edit_trim_parameter(state: AVLState, mode: int, ipar: int, value: float, ir: int = 0) -> None:
    """Store ``value``, zero its dependents (:data:`TRIM_DEPENDENTS`) and re-apply the trim."""
    if ipar == C.IPPHI:
        _check_bank(mode, value)  # before anything is written, so a rejected value changes nothing
    state.parval[ipar, ir] = value
    for dependent in TRIM_DEPENDENTS[mode].get(ipar, ()):
        state.parval[dependent, ir] = 0.0
    apply_trim(state, mode, ir)


def _check_bank(ktrim: int, phi: float) -> None:
    """Same limit as AVL's trim menu: cos(phi) <= 0 has no level-turn solution."""
    if ktrim == 1 and not -90.0 < phi < 90.0:
        raise ValueError(f"Bank angle must satisfy -90 < bank < 90 for trim; got {phi}.")


def apply_trim(state: AVLState, ktrim: int, ir: int = 0) -> None:
    """Recompute dependent trim values and set constraints (atrim.f label 5)."""

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

    _check_bank(ktrim, phi)
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
