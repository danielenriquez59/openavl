"""Eigenvalue analysis (port of AVL amode.f SYSMAT/APPMAT/EIGSOL)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eig

from openavl import constants as C
from openavl.core.state import AVLState
from openavl.math.util import m3inv, rateki3, rotens3

ICRS = (1, 2, 0)
JCRS = (2, 0, 1)
NSYS = C.JETOT
_LN2 = np.log(2.0)


@dataclass(frozen=True)
class EigenmodeMetrics:
    """Scalars derived from one eigenvalue at a trimmed flight condition."""

    sigma: float
    omega: float
    frequency_hz: float
    damping_ratio: float
    time_constant: float
    period_s: float
    time_to_half_s: float


def compute_eigenmode_metrics(
    eigenvalue: complex,
    *,
    vee: float,
    bref: float,
    unitl: float,
) -> EigenmodeMetrics:
    """Compute frequency, damping, decay time, period, and half-life from λ = σ + jω.

    ``build_sysmat`` assembles the state matrix in fully dimensional units
    (row/column scalings by ``vee``/``rot`` convert every nondimensional AVL
    "unit" perturbation back to actual m/s and rad/s before it is written
    into the matrix — see e.g. the direct ``d(theta)/dt = q`` identity row),
    so the eigenvalues ``λ = σ + jω`` returned by :func:`solve_eigenvalues`
    are already dimensional, in rad/s. Frequency is therefore simply
    ``f = ω / (2π)``, with no additional ``V``/``bref`` scaling — matching
    the convention already used by ``time_constant``/``time_to_half_s``
    below (which treat ``σ`` as 1/s).

    Parameters
    ----------
    eigenvalue:
        Complex eigenvalue λ = σ + jω from the state matrix (dimensional,
        in rad/s).
    vee, bref, unitl:
        Unused by the frequency/damping/time-constant calculations (λ is
        already dimensional); kept for API stability with existing callers.

    Returns
    -------
    EigenmodeMetrics
        Real/imag parts of λ plus derived frequency, damping, time constant,
        oscillation period, and amplitude half-life.
    """
    del vee, bref, unitl
    sigma = float(np.real(eigenvalue))
    omega = abs(float(np.imag(eigenvalue)))
    freq_hz = omega / (2.0 * np.pi)

    if omega > 1e-12:
        damping = -sigma / np.hypot(sigma, omega)
    else:
        damping = 1.0 if sigma < 0.0 else -1.0

    tau = -1.0 / sigma if sigma < -1e-12 else float("inf")
    period_s = 1.0 / freq_hz if freq_hz > 1e-12 else float("inf")

    if np.isfinite(tau) and tau > 0.0:
        time_to_half_s = _LN2 * tau
    elif sigma < -1e-12:
        time_to_half_s = _LN2 / abs(sigma)
    else:
        time_to_half_s = float("inf")

    return EigenmodeMetrics(
        sigma=sigma,
        omega=omega,
        frequency_hz=freq_hz,
        damping_ratio=damping,
        time_constant=tau,
        period_s=period_s,
        time_to_half_s=time_to_half_s,
    )


@dataclass
class FlightMode:
    """Identified flight dynamic mode."""

    name: str
    eigenvalue: complex
    eigenvector: np.ndarray
    frequency_hz: float = 0.0
    damping_ratio: float = 0.0
    time_constant: float = 0.0
    period_s: float = 0.0
    time_to_half_s: float = 0.0


@dataclass
class EigenAnalysisResult:
    """Result of a linearized flight dynamics eigenanalysis."""

    state_matrix: np.ndarray
    control_matrix: np.ndarray
    eigenvalues: list[complex]
    eigenvectors: np.ndarray
    modes: list[FlightMode] = field(default_factory=list)
    residual_vector: np.ndarray | None = None


def runchk(state: AVLState, ir: int = 0) -> bool:
    """Return True if run case ``ir`` has no redundant constraints."""
    for iv in range(state.nvtot):
        for jv in range(state.nvtot):
            if iv != jv and state.icon[iv, ir] == state.icon[jv, ir]:
                return False
    return True


def _build_mass_matrices(
    state: AVLState,
    ir: int,
) -> tuple[float, float, float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Assemble mass/inertia matrices and dimensional scalars.

    B4 sign-convention note (verified against AVL 3.52 ``amode.f``): the
    off-diagonal terms are placed into ``riner`` directly from ``parval``,
    with **no** extra negation. This is correct because ``parval[IPIXY/
    IPIYZ/IPIZX]`` already stores the true inertia-tensor components (minus
    sign folded in), not raw products of inertia. That negation happens
    once, upstream, in ``openavl.fileio.mass.parse_mass_text`` /
    ``masput`` (mirroring AVL's ``amass.f`` ``MASGET``/``MASPUT``:
    ``RINER0(1,2) = -Ixy*UNITM*UNITL**2`` then
    ``PARVAL(IPIXY,IR) = RINER0(1,2)``), and at the ``AVLSolver.set_parameter``
    API boundary, which negates a user-supplied raw product of inertia
    before writing it to ``parval`` (see ``core/solver.py``). AVL's own
    ``amode.f`` ``SYSMAT``/``APPMAT`` place ``PARVAL(IPIXY,IR)`` into
    ``RINER(1,2)`` unchanged, exactly like this function.
    """
    gee = float(state.parval[C.IPGEE, ir])
    rho = float(state.parval[C.IPRHO, ir])
    vee = float(state.parval[C.IPVEE, ir])
    rmass = float(state.parval[C.IPMASS, ir])

    riner = np.array(
        [
            [state.parval[C.IPIXX, ir], state.parval[C.IPIXY, ir], state.parval[C.IPIZX, ir]],
            [state.parval[C.IPIXY, ir], state.parval[C.IPIYY, ir], state.parval[C.IPIYZ, ir]],
            [state.parval[C.IPIZX, ir], state.parval[C.IPIYZ, ir], state.parval[C.IPIZZ, ir]],
        ],
        dtype=np.float64,
    )

    if vee <= 0.0 or rmass <= 0.0 or riner[0, 0] <= 0.0 or riner[1, 1] <= 0.0 or riner[2, 2] <= 0.0:
        return None

    sref_d = state.sref * state.unitl * state.unitl
    bref_d = state.bref * state.unitl
    cref_d = state.cref * state.unitl
    qs = 0.5 * rho * vee * vee * sref_d
    rot = vee / state.unitl

    mamat = state.amass * rho
    for k in range(3):
        mamat[k, k] += rmass

    rimat = riner + state.ainer * rho
    mainv = m3inv(mamat.copy())
    riinv = m3inv(rimat.copy())
    return gee, vee, rot, qs, sref_d, bref_d, cref_d, mainv, riinv


def build_sysmat(state: AVLState, ir: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the 12x12 state-space system matrix and control matrix."""
    built = _build_mass_matrices(state, ir)
    if built is None:
        empty_a = np.zeros((NSYS, NSYS), dtype=np.float64)
        empty_b = np.zeros((NSYS, max(1, state.ncontrol)), dtype=np.float64)
        return empty_a, empty_b, np.zeros(NSYS, dtype=np.float64)

    gee, vee, rot, qs, _, bref_d, cref_d, mainv, riinv = built

    phi = float(state.parval[C.IPPHI, ir])
    the = float(state.parval[C.IPTHE, ir])
    psi = float(state.parval[C.IPPSI, ir])
    xcg = float(state.parval[C.IPXCG, ir])
    ycg = float(state.parval[C.IPYCG, ir])
    zcg = float(state.parval[C.IPZCG, ir])
    dcl_u = float(state.parval[C.IPCLU, ir])
    dcm_u = float(state.parval[C.IPCMU, ir])
    dcl_a = float(state.parval[C.IPCLA, ir])
    dcm_a = float(state.parval[C.IPCMA, ir])

    state.xyzref[:] = (xcg, ycg, zcg)

    rho = float(state.parval[C.IPRHO, ir])
    mamat = state.amass * rho
    for k in range(3):
        mamat[k, k] += state.parval[C.IPMASS, ir]
    # Off-diagonal placed unchanged from parval (already the tensor sign
    # convention) -- see the B4 note in _build_mass_matrices above.
    rimat = np.array(
        [
            [state.parval[C.IPIXX, ir], state.parval[C.IPIXY, ir], state.parval[C.IPIZX, ir]],
            [state.parval[C.IPIXY, ir], state.parval[C.IPIYY, ir], state.parval[C.IPIYZ, ir]],
            [state.parval[C.IPIZX, ir], state.parval[C.IPIYZ, ir], state.parval[C.IPIZZ, ir]],
        ],
        dtype=np.float64,
    ) + state.ainer * rho

    p = np.zeros(3, dtype=np.float64)
    h = np.zeros(3, dtype=np.float64)
    p_u = np.zeros((3, 6), dtype=np.float64)
    h_u = np.zeros((3, 6), dtype=np.float64)

    for k in range(3):
        p[k] = -(mamat[k, 0] * state.vinf[0] + mamat[k, 1] * state.vinf[1] + mamat[k, 2] * state.vinf[2]) * vee
        p_u[k, 0] = -mamat[k, 0] * vee
        p_u[k, 1] = -mamat[k, 1] * vee
        p_u[k, 2] = -mamat[k, 2] * vee
        h[k] = (
            rimat[k, 0] * state.wrot[0]
            + rimat[k, 1] * state.wrot[1]
            + rimat[k, 2] * state.wrot[2]
        ) * rot
        h_u[k, 3] = rimat[k, 0] * rot
        h_u[k, 4] = rimat[k, 1] * rot
        h_u[k, 5] = rimat[k, 2] * rot

    wxp = np.zeros(3, dtype=np.float64)
    wxh = np.zeros(3, dtype=np.float64)
    wxp_u = np.zeros((3, 6), dtype=np.float64)
    wxh_u = np.zeros((3, 6), dtype=np.float64)

    for k in range(3):
        i = ICRS[k]
        j = JCRS[k]
        wxp[k] = (state.wrot[i] * p[j] - state.wrot[j] * p[i]) * rot
        wxh[k] = (state.wrot[i] * h[j] - state.wrot[j] * h[i]) * rot
        for iu in range(6):
            wxp_u[k, iu] = (state.wrot[i] * p_u[j, iu] - state.wrot[j] * p_u[i, iu]) * rot
            wxh_u[k, iu] = (state.wrot[i] * h_u[j, iu] - state.wrot[j] * h_u[i, iu]) * rot
        wxp_u[k, i + 3] += p[j] * rot
        wxp_u[k, j + 3] -= p[i] * rot
        wxh_u[k, i + 3] += h[j] * rot
        wxh_u[k, j + 3] -= h[i] * rot

    mif = np.zeros(3, dtype=np.float64)
    rim = np.zeros(3, dtype=np.float64)
    prf = np.zeros(3, dtype=np.float64)
    prm = np.zeros(3, dtype=np.float64)
    mif_u = np.zeros((3, 6), dtype=np.float64)
    rim_u = np.zeros((3, 6), dtype=np.float64)
    prf_u = np.zeros((3, 6), dtype=np.float64)
    prm_u = np.zeros((3, 6), dtype=np.float64)
    mif_d = np.zeros((3, state.ncontrol), dtype=np.float64)
    rim_d = np.zeros((3, state.ncontrol), dtype=np.float64)

    for k in range(3):
        mif[k] = mainv[k, 0] * state.cftot[0] * qs + mainv[k, 1] * state.cftot[1] * qs + mainv[k, 2] * state.cftot[2] * qs
        rim[k] = (
            riinv[k, 0] * state.cmtot[0] * qs * bref_d
            + riinv[k, 1] * state.cmtot[1] * qs * cref_d
            + riinv[k, 2] * state.cmtot[2] * qs * bref_d
        )
        prf[k] = mainv[k, 0] * wxp[0] + mainv[k, 1] * wxp[1] + mainv[k, 2] * wxp[2]
        prm[k] = riinv[k, 0] * wxh[0] + riinv[k, 1] * wxh[1] + riinv[k, 2] * wxh[2]

        for iu in range(6):
            mif_u[k, iu] = (
                mainv[k, 0] * state.cftot_u[0, iu] * qs
                + mainv[k, 1] * state.cftot_u[1, iu] * qs
                + mainv[k, 2] * state.cftot_u[2, iu] * qs
            )
            rim_u[k, iu] = (
                riinv[k, 0] * state.cmtot_u[0, iu] * qs * bref_d
                + riinv[k, 1] * state.cmtot_u[1, iu] * qs * cref_d
                + riinv[k, 2] * state.cmtot_u[2, iu] * qs * bref_d
            )
            prf_u[k, iu] = mainv[k, 0] * wxp_u[0, iu] + mainv[k, 1] * wxp_u[1, iu] + mainv[k, 2] * wxp_u[2, iu]
            prm_u[k, iu] = riinv[k, 0] * wxh_u[0, iu] + riinv[k, 1] * wxh_u[1, iu] + riinv[k, 2] * wxh_u[2, iu]

        for n in range(state.ncontrol):
            mif_d[k, n] = (
                mainv[k, 0] * state.cftot_d[0, n] * qs
                + mainv[k, 1] * state.cftot_d[1, n] * qs
                + mainv[k, 2] * state.cftot_d[2, n] * qs
            )
            rim_d[k, n] = (
                riinv[k, 0] * state.cmtot_d[0, n] * qs * bref_d
                + riinv[k, 1] * state.cmtot_d[1, n] * qs * cref_d
                + riinv[k, 2] * state.cmtot_d[2, n] * qs * bref_d
            )

        mif_u[k, 0] -= mainv[k, 2] * dcl_u * qs
        rim_u[k, 0] -= riinv[k, 1] * dcm_u * qs * cref_d
        mif_u[k, 2] += mainv[k, 2] * dcl_a * qs
        rim_u[k, 2] += riinv[k, 1] * dcm_a * qs * cref_d

    ang = np.array([phi * state.dtr, the * state.dtr, psi * state.dtr], dtype=np.float64)
    tt, tt_ang = rotens3(ang)
    rt, rt_ang = rateki3(ang)

    asys = np.zeros((NSYS, NSYS), dtype=np.float64)
    bsys = np.zeros((NSYS, max(1, state.ncontrol)), dtype=np.float64)
    rsys = np.zeros(NSYS, dtype=np.float64)

    def set_force_row(ieq: int, k: int) -> None:
        rsys[ieq] = mif[k] - prf[k] - gee * tt[2, k]
        asys[ieq, C.JEU] = -(mif_u[k, 0] - prf_u[k, 0]) / vee
        asys[ieq, C.JEV] = -(mif_u[k, 1] - prf_u[k, 1]) / vee
        asys[ieq, C.JEW] = -(mif_u[k, 2] - prf_u[k, 2]) / vee
        asys[ieq, C.JEP] = (mif_u[k, 3] - prf_u[k, 3]) / rot
        asys[ieq, C.JEQ] = (mif_u[k, 4] - prf_u[k, 4]) / rot
        asys[ieq, C.JER] = (mif_u[k, 5] - prf_u[k, 5]) / rot
        asys[ieq, C.JEPH] = -gee * tt_ang[2, k, 0]
        asys[ieq, C.JETH] = -gee * tt_ang[2, k, 1]
        asys[ieq, C.JEPS] = -gee * tt_ang[2, k, 2]
        for n in range(state.ncontrol):
            bsys[ieq, n] = mif_d[k, n]

    def set_moment_row(ieq: int, k: int) -> None:
        rsys[ieq] = rim[k] - prm[k]
        asys[ieq, C.JEU] = -(rim_u[k, 0] - prm_u[k, 0]) / vee
        asys[ieq, C.JEV] = -(rim_u[k, 1] - prm_u[k, 1]) / vee
        asys[ieq, C.JEW] = -(rim_u[k, 2] - prm_u[k, 2]) / vee
        asys[ieq, C.JEP] = (rim_u[k, 3] - prm_u[k, 3]) / rot
        asys[ieq, C.JEQ] = (rim_u[k, 4] - prm_u[k, 4]) / rot
        asys[ieq, C.JER] = (rim_u[k, 5] - prm_u[k, 5]) / rot
        for n in range(state.ncontrol):
            bsys[ieq, n] = rim_d[k, n]

    def set_angle_row(ieq: int, k: int) -> None:
        rsys[ieq] = rot * (rt[k, 0] * state.wrot[0] + rt[k, 1] * state.wrot[1] + rt[k, 2] * state.wrot[2])
        asys[ieq, C.JEP] = rt[k, 0]
        asys[ieq, C.JEQ] = rt[k, 1]
        asys[ieq, C.JER] = rt[k, 2]
        asys[ieq, C.JEPH] = rot * (
            rt_ang[k, 0, 0] * state.wrot[0] + rt_ang[k, 1, 0] * state.wrot[1] + rt_ang[k, 2, 0] * state.wrot[2]
        )
        asys[ieq, C.JETH] = rot * (
            rt_ang[k, 0, 1] * state.wrot[0] + rt_ang[k, 1, 1] * state.wrot[1] + rt_ang[k, 2, 1] * state.wrot[2]
        )
        asys[ieq, C.JEPS] = rot * (
            rt_ang[k, 0, 2] * state.wrot[0] + rt_ang[k, 1, 2] * state.wrot[1] + rt_ang[k, 2, 2] * state.wrot[2]
        )

    def set_position_row(ieq: int, k: int) -> None:
        rsys[ieq] = -(tt[k, 0] * state.vinf[0] + tt[k, 1] * state.vinf[1] + tt[k, 2] * state.vinf[2]) * vee
        asys[ieq, C.JEU] = tt[k, 0]
        asys[ieq, C.JEV] = tt[k, 1]
        asys[ieq, C.JEW] = tt[k, 2]
        asys[ieq, C.JEPH] = -(
            tt_ang[k, 0, 0] * state.vinf[0] + tt_ang[k, 1, 0] * state.vinf[1] + tt_ang[k, 2, 0] * state.vinf[2]
        ) * vee
        asys[ieq, C.JETH] = -(
            tt_ang[k, 0, 1] * state.vinf[0] + tt_ang[k, 1, 1] * state.vinf[1] + tt_ang[k, 2, 1] * state.vinf[2]
        ) * vee
        asys[ieq, C.JEPS] = -(
            tt_ang[k, 0, 2] * state.vinf[0] + tt_ang[k, 1, 2] * state.vinf[1] + tt_ang[k, 2, 2] * state.vinf[2]
        ) * vee

    set_force_row(C.JEU, 0)
    set_force_row(C.JEV, 1)
    set_force_row(C.JEW, 2)
    set_moment_row(C.JEP, 0)
    set_moment_row(C.JEQ, 1)
    set_moment_row(C.JER, 2)
    set_angle_row(C.JEPH, 0)
    set_angle_row(C.JETH, 1)
    set_angle_row(C.JEPS, 2)
    set_position_row(C.JEX, 0)
    set_position_row(C.JEY, 1)
    set_position_row(C.JEZ, 2)

    return asys, bsys, rsys


def build_appmat(state: AVLState, ir: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the 12x12 short-period approximation system matrix."""
    built = _build_mass_matrices(state, ir)
    if built is None:
        empty_a = np.zeros((NSYS, NSYS), dtype=np.float64)
        empty_b = np.zeros((NSYS, max(1, state.ncontrol)), dtype=np.float64)
        return empty_a, empty_b, np.zeros(NSYS, dtype=np.float64)

    gee, vee, rot, qs, _, bref_d, cref_d, _, _ = built
    rmass = float(state.parval[C.IPMASS, ir])
    rinxx = float(state.parval[C.IPIXX, ir])
    rinyy = float(state.parval[C.IPIYY, ir])
    rinzz = float(state.parval[C.IPIZZ, ir])

    phi = float(state.parval[C.IPPHI, ir])
    the = float(state.parval[C.IPTHE, ir])
    psi = float(state.parval[C.IPPSI, ir])
    xcg = float(state.parval[C.IPXCG, ir])
    ycg = float(state.parval[C.IPYCG, ir])
    zcg = float(state.parval[C.IPZCG, ir])

    state.xyzref[:] = (xcg, ycg, zcg)

    qsc = qs * cref_d
    qsb = qs * bref_d

    ang = np.array([phi * state.dtr, the * state.dtr, psi * state.dtr], dtype=np.float64)
    tt, tt_ang = rotens3(ang)
    rt, rt_ang = rateki3(ang)

    asys = np.zeros((NSYS, NSYS), dtype=np.float64)
    bsys = np.zeros((NSYS, max(1, state.ncontrol)), dtype=np.float64)
    rsys = np.zeros(NSYS, dtype=np.float64)

    asys[C.JEU, C.JEU] = -state.cftot_u[0, 0] * qs / rmass / vee
    asys[C.JEU, C.JEW] = -state.cftot_u[0, 2] * qs / rmass / vee
    asys[C.JEU, C.JEQ] = state.cftot_u[0, 4] * qs / rmass / rot + state.vinf[2] * vee
    asys[C.JEU, C.JETH] = gee
    for n in range(state.ncontrol):
        bsys[C.JEU, n] = state.cftot_d[0, n] * qs / rmass

    asys[C.JEW, C.JEU] = -state.cftot_u[2, 0] * qs / rmass / vee
    asys[C.JEW, C.JEW] = -state.cftot_u[2, 2] * qs / rmass / vee
    asys[C.JEW, C.JEQ] = state.cftot_u[2, 4] * qs / rmass / rot - state.vinf[0] * vee
    for n in range(state.ncontrol):
        bsys[C.JEW, n] = state.cftot_d[2, n] * qs / rmass

    asys[C.JEQ, C.JEU] = -state.cmtot_u[1, 0] * qsc / rinyy / vee
    asys[C.JEQ, C.JEW] = -state.cmtot_u[1, 2] * qsc / rinyy / vee
    asys[C.JEQ, C.JEQ] = state.cmtot_u[1, 4] * qsc / rinyy / rot
    for n in range(state.ncontrol):
        bsys[C.JEQ, n] = state.cmtot_d[1, n] * qsc / rinyy

    asys[C.JETH, C.JEQ] = 1.0

    asys[C.JEV, C.JEV] = -state.cftot_u[1, 1] * qs / rmass / vee
    asys[C.JEV, C.JEP] = state.cftot_u[1, 3] * qs / rmass / rot - state.vinf[2] * vee
    asys[C.JEV, C.JER] = state.cftot_u[1, 5] * qs / rmass / rot + state.vinf[0] * vee
    asys[C.JEV, C.JEPH] = gee
    for n in range(state.ncontrol):
        bsys[C.JEV, n] = state.cftot_d[1, n] * qs / rmass

    asys[C.JEP, C.JEV] = -state.cmtot_u[0, 1] * qsb / rinxx / vee
    asys[C.JEP, C.JEP] = state.cmtot_u[0, 3] * qsb / rinxx / rot
    asys[C.JEP, C.JER] = state.cmtot_u[0, 5] * qsb / rinxx / rot
    for n in range(state.ncontrol):
        bsys[C.JEP, n] = state.cmtot_d[0, n] * qsb / rinxx

    asys[C.JER, C.JEV] = -state.cmtot_u[2, 1] * qsb / rinzz / vee
    asys[C.JER, C.JEP] = state.cmtot_u[2, 3] * qsb / rinzz / rot
    asys[C.JER, C.JER] = state.cmtot_u[2, 5] * qsb / rinzz / rot
    for n in range(state.ncontrol):
        bsys[C.JER, n] = state.cmtot_d[2, n] * qsb / rinzz

    asys[C.JEPH, C.JEP] = -1.0

    k = 2
    rsys[C.JEPS] = rot * (rt[k, 0] * state.wrot[0] + rt[k, 1] * state.wrot[1] + rt[k, 2] * state.wrot[2])
    asys[C.JEPS, C.JEP] = rt[k, 0]
    asys[C.JEPS, C.JEQ] = rt[k, 1]
    asys[C.JEPS, C.JER] = rt[k, 2]
    asys[C.JEPS, C.JEPH] = rot * (
        rt_ang[k, 0, 0] * state.wrot[0] + rt_ang[k, 1, 0] * state.wrot[1] + rt_ang[k, 2, 0] * state.wrot[2]
    )
    asys[C.JEPS, C.JETH] = rot * (
        rt_ang[k, 0, 1] * state.wrot[0] + rt_ang[k, 1, 1] * state.wrot[1] + rt_ang[k, 2, 1] * state.wrot[2]
    )
    asys[C.JEPS, C.JEPS] = rot * (
        rt_ang[k, 0, 2] * state.wrot[0] + rt_ang[k, 1, 2] * state.wrot[1] + rt_ang[k, 2, 2] * state.wrot[2]
    )

    for ieq, k in ((C.JEX, 0), (C.JEY, 1), (C.JEZ, 2)):
        rsys[ieq] = -(tt[k, 0] * state.vinf[0] + tt[k, 1] * state.vinf[1] + tt[k, 2] * state.vinf[2]) * vee
        asys[ieq, C.JEU] = tt[k, 0]
        asys[ieq, C.JEV] = tt[k, 1]
        asys[ieq, C.JEW] = tt[k, 2]
        asys[ieq, C.JEPH] = -(
            tt_ang[k, 0, 0] * state.vinf[0] + tt_ang[k, 1, 0] * state.vinf[1] + tt_ang[k, 2, 0] * state.vinf[2]
        ) * vee
        asys[ieq, C.JETH] = -(
            tt_ang[k, 0, 1] * state.vinf[0] + tt_ang[k, 1, 1] * state.vinf[1] + tt_ang[k, 2, 1] * state.vinf[2]
        ) * vee
        asys[ieq, C.JEPS] = -(
            tt_ang[k, 0, 2] * state.vinf[0] + tt_ang[k, 1, 2] * state.vinf[1] + tt_ang[k, 2, 2] * state.vinf[2]
        ) * vee

    return asys, bsys, rsys


def identify_modes(
    eigenvalues: list[complex],
    eigenvectors: np.ndarray,
    vee: float,
    bref: float,
    unitl: float,
) -> list[FlightMode]:
    """Identify classical flight modes from eigenvalues and eigenvectors."""
    lon_idx = [C.JEU, C.JEW, C.JEQ, C.JETH]
    lat_idx = [C.JEV, C.JEP, C.JER, C.JEPH]
    pos_idx = [C.JEX, C.JEY, C.JEZ, C.JEPS]

    modes: list[FlightMode] = []
    used_names: set[str] = set()

    for idx, value in enumerate(eigenvalues):
        vec = eigenvectors[:, idx]
        mag = np.abs(vec)
        total = float(np.sum(mag * mag)) or 1.0
        lon_part = float(np.sum(mag[lon_idx] ** 2)) / total
        lat_part = float(np.sum(mag[lat_idx] ** 2)) / total
        pos_part = float(np.sum(mag[pos_idx] ** 2)) / total

        sigma = value.real
        omega = abs(value.imag)
        metrics = compute_eigenmode_metrics(value, vee=vee, bref=bref, unitl=unitl)
        freq_hz = metrics.frequency_hz
        damping = metrics.damping_ratio
        tau = metrics.time_constant

        if pos_part > 0.5:
            name = "position/heading"
        elif lon_part >= lat_part:
            if omega > 0.05 * abs(sigma) and freq_hz > 0.2:
                name = "short period"
            elif omega > 1e-8:
                name = "phugoid"
            else:
                name = "longitudinal subsidence"
        else:
            if omega > 0.05 * abs(sigma) and omega > 1e-8:
                name = "Dutch roll"
            elif abs(sigma) > 1e-8:
                if tau > 5.0:
                    name = "spiral"
                else:
                    name = "roll subsidence"
            else:
                name = "lateral mode"

        base = name
        suffix = 2
        while name in used_names:
            name = f"{base} {suffix}"
            suffix += 1
        used_names.add(name)

        modes.append(
            FlightMode(
                name=name,
                eigenvalue=value,
                eigenvector=vec,
                frequency_hz=freq_hz,
                damping_ratio=damping,
                time_constant=tau,
                period_s=metrics.period_s,
                time_to_half_s=metrics.time_to_half_s,
            )
        )

    return modes


_BODY_AXIS_SIGN_INDICES = (C.JEU, C.JEW, C.JEP, C.JER, C.JEX, C.JEZ)


def apply_body_axis_signs(
    asys: np.ndarray,
    bsys: np.ndarray,
    rsys: np.ndarray,
    ncontrol: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply AVL ``SYSSHO`` row/column sign changes for body-axis display.

    AVL multiplies rows and columns associated with ``u``, ``w``, ``p``, ``r``,
    ``x``, and ``z`` by ``-1`` before printing the modal system matrix. The
    eigenvectors returned by :func:`solve_eigenvalues` correspond to the
    unmodified matrix; pass ``in_body_axis=True`` to :meth:`AVLSolver.get_system_matrix`
    to obtain the signed form.
    """
    nsys = asys.shape[0]
    usgn = np.ones(nsys, dtype=np.float64)
    for idx in _BODY_AXIS_SIGN_INDICES:
        if 0 <= idx < nsys:
            usgn[idx] = -1.0

    asys_out = asys * usgn[:, np.newaxis] * usgn[np.newaxis, :]
    ncol = bsys.shape[1] if ncontrol is None else min(ncontrol, bsys.shape[1])
    bsys_out = bsys[:, :ncol] * usgn[:, np.newaxis]
    rsys_out = rsys * usgn
    return asys_out, bsys_out, rsys_out


def solve_eigenvalues(
    state: AVLState,
    ir: int = 0,
    use_approx: bool = False,
    etol: float = 1e-4,
) -> EigenAnalysisResult:
    """Compute eigenvalues and identify flight modes for a solved run case."""
    if use_approx:
        asys, bsys, rsys = build_appmat(state, ir)
    else:
        asys, bsys, rsys = build_sysmat(state, ir)

    if not np.any(asys):
        return EigenAnalysisResult(
            state_matrix=asys,
            control_matrix=bsys,
            eigenvalues=[],
            eigenvectors=np.zeros((NSYS, 0), dtype=np.complex128),
            modes=[],
            residual_vector=rsys,
        )

    evals, evecs = eig(asys)

    vee = float(state.parval[C.IPVEE, ir])
    bref_d = state.bref * state.unitl
    etolsq = (etol * vee / max(1e-12, bref_d)) ** 2

    eigenvalues: list[complex] = []
    columns: list[np.ndarray] = []
    for idx in range(len(evals)):
        value = complex(evals[idx])
        if value.real * value.real + value.imag * value.imag < etolsq:
            continue
        if value.imag < 0.0:
            continue
        eigenvalues.append(value)
        columns.append(evecs[:, idx])

    if columns:
        eigenvectors = np.column_stack(columns)
    else:
        eigenvectors = np.zeros((NSYS, 0), dtype=np.complex128)

    modes = identify_modes(eigenvalues, eigenvectors, vee, state.bref, state.unitl)
    return EigenAnalysisResult(
        state_matrix=asys,
        control_matrix=bsys,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        modes=modes,
        residual_vector=rsys,
    )
