"""Result extraction and aerodynamic load post-processing."""

from __future__ import annotations

from typing import Any

import numpy as np

from openavl import constants as C
from openavl.core.reporting import reported_totals


def get_results(self) -> dict[str, Any]:
    """Return aerodynamic coefficients and metadata from the latest solve.

    The returned mapping summarizes the converged (or last-iterated) run
    case after :meth:`execute_run`. Lift, drag, and side force are
    stability-axis coefficients. Reported body-axis force and moment
    coefficients follow AVL's ``OUTTOT`` / ``get_total_forces`` sign
    convention (NASA standard when ``state.lnasa_sa`` is ``True``).

    Returns
    -------
    dict[str, Any]
        Dictionary with the following keys:

        ``CL``, ``CD``, ``CY``
            Total lift, drag, and side-force coefficients.
        ``Cl``, ``Cm``, ``Cn``
            Body-axis rolling, pitching, and yawing moment coefficients as
            in AVL ``Cltot`` / ``Cmtot`` / ``Cntot``.
        ``Cl_sa``, ``Cm_sa``, ``Cn_sa``
            Stability-axis roll, pitch, and yaw moment coefficients as in
            AVL ``Cl'tot``, ``Cmtot``, and ``Cn'tot``.
        ``Cx``, ``Cy``, ``Cz``
            Body-axis force coefficients (``DIR`` applied to ``Cx`` and
            ``Cz`` when using NASA-standard axes).
        ``CDV``, ``CLFF``, ``CDFF``, ``CYFF``, ``SPANEF``
            Viscous and Trefftz-plane drag bookkeeping terms.
        ``alpha_deg``, ``beta_deg``
            Solved angle of attack and sideslip in degrees.
        ``control_deflections``
            Control surface deflections in degrees, keyed by control name
            (e.g. ``results["control_deflections"]["aileron"]``).
        ``mach``
            Mach number for the run case.
        ``converged``
            ``True`` if the Newton iteration met the convergence tolerance.
        ``geometry``
            Lattice size summary with keys ``NVOR``, ``NSTRIP``,
            ``NSURF``, ``NBODY``, and ``NLNODE``.

    Notes
    -----
    Raw integration values remain in ``state.cmtot`` and ``state.cftot``.
    If no solve has been run, values reflect the initialized state and
    ``converged`` will be ``False``.
    """
    s = self.state
    reported = reported_totals(s)
    cl, cm, cn = reported["CM"]
    cl_sa, cm_sa, cn_sa = reported["CM_sa"]
    cx, cy, cz = reported["CF"]
    control_deflections = {
        name: float(s.delcon[idx])
        for idx, name in enumerate(s.control_names[: s.ncontrol])
    }
    return {
        "CL": s.cltot,
        "CD": s.cdtot,
        "CY": s.cytot,
        "Cl": cl,
        "Cm": cm,
        "Cn": cn,
        "Cl_sa": cl_sa,
        "Cm_sa": cm_sa,
        "Cn_sa": cn_sa,
        "Cx": cx,
        "Cy": cy,
        "Cz": cz,
        "CDV": s.cdvtot,
        "CLFF": s.clff,
        "CDFF": s.cdff,
        "CYFF": s.cyff,
        "SPANEF": s.spanef,
        "converged": s.lsol,
        "geometry": {
            "NVOR": s.nvor,
            "NSTRIP": s.nstrip,
            "NSURF": s.nsurf,
            "NBODY": s.nbody,
            "NLNODE": s.nlnode,
        },
        "alpha_deg": s.alfa / s.dtr,
        "beta_deg": s.beta / s.dtr,
        "control_deflections": control_deflections,
        "mach": s.mach,
    }

def get_aero_accel(self) -> dict[str, Any]:
    """Return body-axis accelerations derived from integrated aero loads.

    Dimensional force and moment vectors are computed from the latest
    solved body-axis coefficients, then converted to translational and
    rotational accelerations with Newton-Euler rigid-body dynamics. These
    values are post-processed results, not native solver state variables.

    Returns
    -------
    dict[str, Any]
        Dynamic pressure, dimensional body-axis force and moment vectors,
        linear and rotational body-axis accelerations, mass, and inertia.
        With SI geometry and mass-file units, linear acceleration is in
        m/s² and rotational acceleration is in rad/s².

    Raises
    ------
    ValueError
        If the current run-case mass is not positive.
    numpy.linalg.LinAlgError
        If the inertia matrix is singular.
    """
    s = self.state
    ir = 0
    rho = float(s.parval[C.IPRHO, ir])
    velocity = float(s.parval[C.IPVEE, ir])
    mass = float(s.parval[C.IPMASS, ir])
    if mass <= 0.0:
        raise ValueError("mass must be positive to calculate acceleration")

    inertia = np.array(
        [
            [s.parval[C.IPIXX, ir], s.parval[C.IPIXY, ir], s.parval[C.IPIZX, ir]],
            [s.parval[C.IPIXY, ir], s.parval[C.IPIYY, ir], s.parval[C.IPIYZ, ir]],
            [s.parval[C.IPIZX, ir], s.parval[C.IPIYZ, ir], s.parval[C.IPIZZ, ir]],
        ],
        dtype=np.float64,
    )
    reported = reported_totals(s)
    cl, cm, cn = reported["CM"]

    dynamic_pressure = 0.5 * rho * velocity**2
    sref = float(s.sref) * s.unitl * s.unitl
    bref = float(s.bref) * s.unitl
    cref = float(s.cref) * s.unitl
    force_body = dynamic_pressure * sref * np.asarray(reported["CF"], dtype=np.float64)
    moment_body = dynamic_pressure * sref * np.array(
        [bref * cl, cref * cm, bref * cn],
        dtype=np.float64,
    )

    linear_acceleration = force_body / mass
    omega_body = s.wrot * velocity / s.unitl
    angular_momentum = inertia @ omega_body
    rotational_acceleration = np.linalg.solve(
        inertia,
        moment_body - np.cross(omega_body, angular_momentum),
    )

    return {
        "dynamic_pressure": dynamic_pressure,
        "force_body": force_body,
        "moment_body": moment_body,
        "linear_acceleration_body": linear_acceleration,
        "rotational_acceleration_body": rotational_acceleration,
        "mass": mass,
        "inertia": inertia,
    }
