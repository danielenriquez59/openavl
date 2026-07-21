"""Newton constraint solver loop (port of aoper.f / exec_ref.f EXEC)."""

from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np

from openavl import constants as C
from openavl.aero.forces import aero, vinfab
from openavl.core.setup import gamsum, gucalc, setup, velsum
from openavl.math.linalg import baksub, ludcmp

_EPS = (2.0e-5)
_DMAX = (1.5708)


class SolveCancelledError(RuntimeError):
    """Raised when a caller cooperatively cancels an active solve."""


def _raise_if_cancelled(cancel_check: Callable[[], bool] | None) -> None:
    """Abort the solve when the caller's cancellation predicate is true."""
    if cancel_check is not None and cancel_check():
        raise SolveCancelledError("Solve stopped by user.")


def exec_solve(
    state: Any,
    niter: int = 0,
    info: int = 0,
    ir: int = 0,
    cancel_check: Callable[[], bool] | None = None,
) -> Any:
    """Run the AVL solve loop: setup, unit solutions, forces, optional Newton trim.

    On trim non-convergence or abort, leaves ``parval`` / ``lsen`` unchanged
    (matching AVL) and emits a warning. ``gamsum`` rebuilds control/design
    circulation from unit solutions, so the redundant ``gdcalc`` pre-pass
    used by older AVL builds is omitted.
    """
    _ = info
    dir_ = (-1.0 if state.lnasa_sa else 1.0)
    _raise_if_cancelled(cancel_check)

    state.lsol = False
    state._clmax_clip_warned: set[int] = set()
    state.xyzref[:] = [
        state.parval[C.IPXCG, ir],
        state.parval[C.IPYCG, ir],
        state.parval[C.IPZCG, ir],
    ]
    state.cdref = (state.parval[C.IPCD0, ir])
    state.mach = (state.parval[C.IPMACH, ir])

    if state.mach != state.amach:
        state.laic = False
        state.lsrd = False
        state.lvel = False
        state.lsol = False
        state.lsen = False

    setup(state)
    _raise_if_cancelled(cancel_check)

    if niter > 0:
        if int(state.icon[C.IVALFA, ir]) == C.ICALFA:
            state.alfa = (state.conval[C.ICALFA, ir] * state.dtr)
        if int(state.icon[C.IVBETA, ir]) == C.ICBETA:
            state.beta = (state.conval[C.ICBETA, ir] * state.dtr)
        if int(state.icon[C.IVROTX, ir]) == C.ICROTX:
            state.wrot[0] = (state.conval[C.ICROTX, ir] * 2.0 / state.bref)
        if int(state.icon[C.IVROTY, ir]) == C.ICROTY:
            state.wrot[1] = (state.conval[C.ICROTY, ir] * 2.0 / state.cref)
        if int(state.icon[C.IVROTZ, ir]) == C.ICROTZ:
            state.wrot[2] = (state.conval[C.ICROTZ, ir] * 2.0 / state.bref)

    gucalc(state)
    _raise_if_cancelled(cancel_check)
    vinfab(state)
    # gamsum rebuilds gam_d/gam_g from gam_u_d/gam_u_g; skip redundant gdcalc.
    gamsum(state)
    velsum(state)
    aero(state)

    nvtot = state.nvtot
    ivmax = C.IVMAX
    ndmax = state.ndmax

    if niter > 0:
        vsys = np.zeros((ivmax, ivmax), dtype=np.float64)
        vres = np.zeros(ivmax, dtype=np.float64)
        ddc = np.zeros(ndmax, dtype=np.float64)
        work = np.zeros(ivmax, dtype=np.float64)
        ivsys = np.zeros(ivmax, dtype=np.int32)

        for _iter in range(niter):
            _raise_if_cancelled(cancel_check)
            if state.lsa_rates:
                ca = (np.cos(state.alfa))
                sa = (np.sin(state.alfa))
                ca_a = (-sa)
                sa_a = (ca)
            else:
                ca = (1.0)
                sa = (0.0)
                ca_a = (0.0)
                sa_a = (0.0)

            vsys[:, :] = 0.0

            for iv in range(nvtot):
                ic = int(state.icon[iv, ir])
                if ic == C.ICALFA:
                    vres[iv] = (state.alfa - (state.conval[C.ICALFA, ir] * state.dtr))
                    vsys[iv, C.IVALFA] = 1.0
                elif ic == C.ICBETA:
                    vres[iv] = (state.beta - (state.conval[C.ICBETA, ir] * state.dtr))
                    vsys[iv, C.IVBETA] = 1.0
                elif ic == C.ICROTX:
                    vres[iv] = (
                        (state.wrot[0] * ca + state.wrot[2] * sa) * dir_
                        - (state.conval[C.ICROTX, ir] * 2.0 / state.bref)
                    )
                    vsys[iv, C.IVROTX] = (ca * dir_)
                    vsys[iv, C.IVROTZ] = (sa * dir_)
                    vsys[iv, C.IVALFA] = ((state.wrot[0] * ca_a + state.wrot[2] * sa_a) * dir_)
                elif ic == C.ICROTY:
                    vres[iv] = (state.wrot[1] - (state.conval[C.ICROTY, ir] * 2.0 / state.cref))
                    vsys[iv, C.IVROTY] = 1.0
                elif ic == C.ICROTZ:
                    vres[iv] = (
                        (state.wrot[2] * ca - state.wrot[0] * sa) * dir_
                        - (state.conval[C.ICROTZ, ir] * 2.0 / state.bref)
                    )
                    vsys[iv, C.IVROTX] = (-sa * dir_)
                    vsys[iv, C.IVROTZ] = (ca * dir_)
                    vsys[iv, C.IVALFA] = ((state.wrot[2] * ca_a - state.wrot[0] * sa_a) * dir_)
                elif ic == C.ICCL:
                    vres[iv] = (state.cltot - state.conval[C.ICCL, ir])
                    vsys[iv, C.IVALFA] = (
                        state.cltot_u[0] * state.vinf_a[0]
                        + state.cltot_u[1] * state.vinf_a[1]
                        + state.cltot_u[2] * state.vinf_a[2]
                        + state.cltot_a
                    )
                    vsys[iv, C.IVBETA] = (
                        state.cltot_u[0] * state.vinf_b[0]
                        + state.cltot_u[1] * state.vinf_b[1]
                        + state.cltot_u[2] * state.vinf_b[2]
                    )
                    vsys[iv, C.IVROTX] = (state.cltot_u[3])
                    vsys[iv, C.IVROTY] = (state.cltot_u[4])
                    vsys[iv, C.IVROTZ] = (state.cltot_u[5])
                    for n in range(state.ncontrol):
                        vsys[iv, C.IVTOT + n] = (state.cltot_d[n])
                elif ic == C.ICCY:
                    vres[iv] = (state.cytot - state.conval[C.ICCY, ir])
                    vsys[iv, C.IVALFA] = (
                        state.cytot_u[0] * state.vinf_a[0]
                        + state.cytot_u[1] * state.vinf_a[1]
                        + state.cytot_u[2] * state.vinf_a[2]
                    )
                    vsys[iv, C.IVBETA] = (
                        state.cytot_u[0] * state.vinf_b[0]
                        + state.cytot_u[1] * state.vinf_b[1]
                        + state.cytot_u[2] * state.vinf_b[2]
                    )
                    vsys[iv, C.IVROTX] = (state.cytot_u[3])
                    vsys[iv, C.IVROTY] = (state.cytot_u[4])
                    vsys[iv, C.IVROTZ] = (state.cytot_u[5])
                    for n in range(state.ncontrol):
                        vsys[iv, C.IVTOT + n] = (state.cytot_d[n])
                elif ic == C.ICMOMX:
                    vres[iv] = ((state.cmtot[0] * ca + state.cmtot[2] * sa) * dir_ - state.conval[C.ICMOMX, ir])
                    vsys[iv, C.IVALFA] = (
                        (
                            state.cmtot_u[0, 0] * state.vinf_a[0]
                            + state.cmtot_u[0, 1] * state.vinf_a[1]
                            + state.cmtot_u[0, 2] * state.vinf_a[2]
                        )
                        * ca
                        * dir_
                        + (
                            state.cmtot_u[2, 0] * state.vinf_a[0]
                            + state.cmtot_u[2, 1] * state.vinf_a[1]
                            + state.cmtot_u[2, 2] * state.vinf_a[2]
                        )
                        * sa
                        * dir_
                        + (state.cmtot[0] * ca_a + state.cmtot[2] * sa_a) * dir_
                    )
                    vsys[iv, C.IVBETA] = (
                        (
                            state.cmtot_u[0, 0] * state.vinf_b[0]
                            + state.cmtot_u[0, 1] * state.vinf_b[1]
                            + state.cmtot_u[0, 2] * state.vinf_b[2]
                        )
                        * ca
                        * dir_
                        + (
                            state.cmtot_u[2, 0] * state.vinf_b[0]
                            + state.cmtot_u[2, 1] * state.vinf_b[1]
                            + state.cmtot_u[2, 2] * state.vinf_b[2]
                        )
                        * sa
                        * dir_
                    )
                    vsys[iv, C.IVROTX] = ((state.cmtot_u[0, 3] * ca + state.cmtot_u[2, 3] * sa) * dir_)
                    vsys[iv, C.IVROTY] = ((state.cmtot_u[0, 4] * ca + state.cmtot_u[2, 4] * sa) * dir_)
                    vsys[iv, C.IVROTZ] = ((state.cmtot_u[0, 5] * ca + state.cmtot_u[2, 5] * sa) * dir_)
                    for n in range(state.ncontrol):
                        vsys[iv, C.IVTOT + n] = (
                            (state.cmtot_d[0, n] * ca + state.cmtot_d[2, n] * sa) * dir_
                        )
                elif ic == C.ICMOMY:
                    vres[iv] = (state.cmtot[1] - state.conval[C.ICMOMY, ir])
                    vsys[iv, C.IVALFA] = (
                        state.cmtot_u[1, 0] * state.vinf_a[0]
                        + state.cmtot_u[1, 1] * state.vinf_a[1]
                        + state.cmtot_u[1, 2] * state.vinf_a[2]
                    )
                    vsys[iv, C.IVBETA] = (
                        state.cmtot_u[1, 0] * state.vinf_b[0]
                        + state.cmtot_u[1, 1] * state.vinf_b[1]
                        + state.cmtot_u[1, 2] * state.vinf_b[2]
                    )
                    vsys[iv, C.IVROTX] = (state.cmtot_u[1, 3])
                    vsys[iv, C.IVROTY] = (state.cmtot_u[1, 4])
                    vsys[iv, C.IVROTZ] = (state.cmtot_u[1, 5])
                    for n in range(state.ncontrol):
                        vsys[iv, C.IVTOT + n] = (state.cmtot_d[1, n])
                elif ic == C.ICMOMZ:
                    vres[iv] = ((state.cmtot[2] * ca - state.cmtot[0] * sa) * dir_ - state.conval[C.ICMOMZ, ir])
                    vsys[iv, C.IVALFA] = (
                        (
                            state.cmtot_u[2, 0] * state.vinf_a[0]
                            + state.cmtot_u[2, 1] * state.vinf_a[1]
                            + state.cmtot_u[2, 2] * state.vinf_a[2]
                        )
                        * ca
                        * dir_
                        - (
                            state.cmtot_u[0, 0] * state.vinf_a[0]
                            + state.cmtot_u[0, 1] * state.vinf_a[1]
                            + state.cmtot_u[0, 2] * state.vinf_a[2]
                        )
                        * sa
                        * dir_
                        + (state.cmtot[2] * ca_a - state.cmtot[0] * sa_a) * dir_
                    )
                    vsys[iv, C.IVBETA] = (
                        (
                            state.cmtot_u[2, 0] * state.vinf_b[0]
                            + state.cmtot_u[2, 1] * state.vinf_b[1]
                            + state.cmtot_u[2, 2] * state.vinf_b[2]
                        )
                        * ca
                        * dir_
                        - (
                            state.cmtot_u[0, 0] * state.vinf_b[0]
                            + state.cmtot_u[0, 1] * state.vinf_b[1]
                            + state.cmtot_u[0, 2] * state.vinf_b[2]
                        )
                        * sa
                        * dir_
                    )
                    vsys[iv, C.IVROTX] = ((state.cmtot_u[2, 3] * ca - state.cmtot_u[0, 3] * sa) * dir_)
                    vsys[iv, C.IVROTY] = ((state.cmtot_u[2, 4] * ca - state.cmtot_u[0, 4] * sa) * dir_)
                    vsys[iv, C.IVROTZ] = ((state.cmtot_u[2, 5] * ca - state.cmtot_u[0, 5] * sa) * dir_)
                    for n in range(state.ncontrol):
                        vsys[iv, C.IVTOT + n] = (
                            (state.cmtot_d[2, n] * ca - state.cmtot_d[0, n] * sa) * dir_
                        )
                else:
                    matched = False
                    for n in range(state.ncontrol):
                        iccon = C.ICTOT + n
                        ivcon = C.IVTOT + n
                        if ic == iccon:
                            vres[iv] = (state.delcon[n] - state.conval[iccon, ir])
                            vsys[iv, ivcon] = 1.0
                            matched = True
                            break
                    if not matched:
                        raise ValueError(f"Illegal constraint index: {ic}")

            sub = vsys[:nvtot, :nvtot].copy()
            rhs = vres[:nvtot].copy()
            indx = np.zeros(nvtot, dtype=np.int32)
            wsub = np.zeros(nvtot, dtype=np.float64)
            ludcmp(sub, nvtot, indx, wsub)
            baksub(sub, nvtot, indx, rhs)

            if not np.all(np.isfinite(rhs)):
                warnings.warn(
                    "Trim aborted: non-finite Newton step",
                    stacklevel=2,
                )
                return state

            dal = (-rhs[C.IVALFA])
            dbe = (-rhs[C.IVBETA])
            dwx = (-rhs[C.IVROTX])
            dwy = (-rhs[C.IVROTY])
            dwz = (-rhs[C.IVROTZ])
            ddc[:] = 0.0
            for n in range(state.ncontrol):
                ddc[n] = (-rhs[C.IVTOT + n])

            dmaxa = _DMAX
            dmaxr = (5.0 * _DMAX / state.bref)
            if abs(state.alfa + dal) > dmaxa or abs(state.beta + dbe) > dmaxa:
                warnings.warn(
                    "Trim aborted: alpha/beta step exceeds limit",
                    stacklevel=2,
                )
                return state
            if abs(state.wrot[0] + dwx) > dmaxr or abs(state.wrot[1] + dwy) > dmaxr or abs(state.wrot[2] + dwz) > dmaxr:
                warnings.warn(
                    "Trim aborted: rotation-rate step exceeds limit",
                    stacklevel=2,
                )
                return state

            state.alfa = (state.alfa + dal)
            state.beta = (state.beta + dbe)
            state.wrot[0] = (state.wrot[0] + dwx)
            state.wrot[1] = (state.wrot[1] + dwy)
            state.wrot[2] = (state.wrot[2] + dwz)
            for n in range(state.ncontrol):
                state.delcon[n] = (state.delcon[n] + ddc[n])

            vinfab(state)
            gamsum(state)
            velsum(state)
            aero(state)
            _raise_if_cancelled(cancel_check)

            delmax = max(
                (
                    abs(dal),
                    abs(dbe),
                    abs(dwx * state.bref / 2.0),
                    abs(dwy * state.cref / 2.0),
                    abs(dwz * state.bref / 2.0),
                    *(abs(ddc[n]) for n in range(state.ncontrol)),
                ),
                default=0.0,
            )
            if delmax < _EPS:
                state.lsol = True
                break

    # Only commit run-case parameters / sensitivity flag after a successful
    # solve (or a pure force evaluation with niter==0). Failed trim keeps
    # the pre-trim parval and leaves lsen unchanged, matching AVL.
    if niter == 0 or state.lsol:
        state.parval[C.IPALFA, ir] = (state.alfa / state.dtr)
        state.parval[C.IPBETA, ir] = (state.beta / state.dtr)
        state.parval[C.IPROTX, ir] = (state.wrot[0] * 0.5 * state.bref)
        state.parval[C.IPROTY, ir] = (state.wrot[1] * 0.5 * state.cref)
        state.parval[C.IPROTZ, ir] = (state.wrot[2] * 0.5 * state.bref)
        state.parval[C.IPCL, ir] = (state.cltot)
        state.amach = (state.mach)
        state.lsen = True
    else:
        warnings.warn(
            "Trim convergence failed",
            stacklevel=2,
        )
    return state
