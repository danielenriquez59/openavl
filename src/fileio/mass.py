"""Mass file parsing (port of amass.f)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from openavl import constants as C

_FLOAT_RE = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?",
)


@dataclass
class MassProperties:
    """Aggregated mass, CG, and inertia from a .mass file."""

    mass: float = 1.0
    cg: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    inertia: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float64),
    )
    unitl: float = 1.0
    unitm: float = 1.0
    unitt: float = 1.0
    gee: float = 1.0
    rho: float = 1.0
    lunit_name: str = "Lunit"
    munit_name: str = "Munit"
    tunit_name: str = "Tunit"
    loaded: bool = False
    components: list[dict[str, float]] = field(default_factory=list)


def _strip(text: str) -> str:
    """Return trimmed text."""
    return str(text or "").strip()


def _get_floats(line: str, max_count: int = 0) -> list[float]:
    """Extract floating-point values from a line (Fortran GETFLT-style)."""
    clean = line.split("!")[0]
    matches = _FLOAT_RE.findall(clean)
    vals = [float(v.replace("d", "e").replace("D", "e")) for v in matches]
    if max_count > 0:
        return vals[:max_count]
    return vals


def _apply_mass_properties(state: Any, props: MassProperties) -> None:
    """Copy parsed mass properties onto solver state arrays."""
    state.rmass0 = float(props.mass)
    state.xyzmass0[:] = props.cg
    state.riner0[:, :] = props.inertia
    state.unitl = float(props.unitl)
    state.unitm = float(props.unitm)
    state.unitt = float(props.unitt)
    state.gee0 = float(props.gee)
    state.rho0 = float(props.rho)
    state.lmass = props.loaded


def parse_mass_text(text: str) -> MassProperties:
    """Parse .mass file contents and return aggregated mass properties."""
    props = MassProperties(loaded=False)

    fac = np.ones(10, dtype=np.float64)
    add = np.zeros(10, dtype=np.float64)

    sum_m = 0.0
    sum_mx = 0.0
    sum_my = 0.0
    sum_mz = 0.0
    sum_mxx = 0.0
    sum_myy = 0.0
    sum_mzz = 0.0
    sum_mxy = 0.0
    sum_mxz = 0.0
    sum_myz = 0.0
    sum_ixx = 0.0
    sum_iyy = 0.0
    sum_izz = 0.0
    sum_ixy = 0.0
    sum_ixz = 0.0
    sum_iyz = 0.0
    components: list[dict[str, float]] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r\n")
        if not line:
            continue

        first = line[0]
        if first in "#!":
            continue

        if first == "*":
            vals = _get_floats(line[1:], 10)
            for k, val in enumerate(vals):
                fac[k] = val
            continue

        if first == "+":
            vals = _get_floats(line[1:], 10)
            for k, val in enumerate(vals):
                add[k] = val
            continue

        eq = line.find("=")
        if eq > 0:
            key = line[:eq]
            rest = _strip(line[eq + 1 :])
            parts = rest.split(None, 1)
            if not parts:
                continue
            value = float(parts[0])
            unit = parts[1].strip() if len(parts) > 1 else ""

            if "Lunit" in key:
                props.unitl = value
                props.lunit_name = unit or props.lunit_name
                continue
            if "Munit" in key:
                props.unitm = value
                props.munit_name = unit or props.munit_name
                continue
            if "Tunit" in key:
                props.unitt = value
                props.tunit_name = unit or props.tunit_name
                continue
            if "g" in key:
                props.gee = value
                continue
            if "rho" in key:
                props.rho = value
                continue

        rinp = _get_floats(line, 10)
        if not rinp:
            continue
        while len(rinp) < 10:
            rinp.append(0.0)

        mi = fac[0] * rinp[0] + add[0]
        xi = fac[1] * rinp[1] + add[1]
        yi = fac[2] * rinp[2] + add[2]
        zi = fac[3] * rinp[3] + add[3]
        ixxi = fac[4] * rinp[4] + add[4]
        iyyi = fac[5] * rinp[5] + add[5]
        izzi = fac[6] * rinp[6] + add[6]
        ixyi = fac[7] * rinp[7] + add[7]
        ixzi = fac[8] * rinp[8] + add[8]
        iyzi = fac[9] * rinp[9] + add[9]

        components.append({"mass": mi, "x": xi, "y": yi, "z": zi})

        sum_m += mi
        sum_mx += mi * xi
        sum_my += mi * yi
        sum_mz += mi * zi
        sum_mxx += mi * xi * xi
        sum_myy += mi * yi * yi
        sum_mzz += mi * zi * zi
        sum_mxy += mi * xi * yi
        sum_mxz += mi * xi * zi
        sum_myz += mi * yi * zi

        sum_ixx += ixxi
        sum_iyy += iyyi
        sum_izz += izzi
        sum_ixy += ixyi
        sum_ixz += ixzi
        sum_iyz += iyzi

    if sum_m == 0.0:
        return props

    xcg = sum_mx / sum_m
    ycg = sum_my / sum_m
    zcg = sum_mz / sum_m

    ixx = sum_ixx + (sum_myy + sum_mzz) - sum_m * (ycg * ycg + zcg * zcg)
    iyy = sum_iyy + (sum_mzz + sum_mxx) - sum_m * (zcg * zcg + xcg * xcg)
    izz = sum_izz + (sum_mxx + sum_myy) - sum_m * (xcg * xcg + ycg * ycg)
    ixy = sum_ixy + sum_mxy - sum_m * xcg * ycg
    ixz = sum_ixz + sum_mxz - sum_m * xcg * zcg
    iyz = sum_iyz + sum_myz - sum_m * ycg * zcg

    inertia_scale = props.unitm * props.unitl * props.unitl
    props.mass = sum_m * props.unitm
    props.cg = np.array(
        [xcg * props.unitl, ycg * props.unitl, zcg * props.unitl],
        dtype=np.float64,
    )
    props.inertia = np.array(
        [
            [ixx, -ixy, -ixz],
            [-ixy, iyy, -iyz],
            [-ixz, -iyz, izz],
        ],
        dtype=np.float64,
    ) * inertia_scale
    props.components = [
        {
            "mass": comp["mass"] * props.unitm,
            "x": comp["x"] * props.unitl,
            "y": comp["y"] * props.unitl,
            "z": comp["z"] * props.unitl,
        }
        for comp in components
    ]
    props.loaded = True
    return props


def parse_mass_file(path: str | Path) -> MassProperties:
    """Read and parse a .mass file from disk."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return parse_mass_text(text)


def masini(state: Any) -> Any:
    """Initialize default mass and inertia tensors (MASINI)."""
    state.rmass0 = 1.0
    state.xyzmass0[:] = 0.0
    state.riner0[:] = 0.0
    np.fill_diagonal(state.riner0, 1.0)

    state.amass[:] = 0.0
    state.ainer[:] = 0.0

    state.lmass = False
    return state


def unitset(state: Any) -> Any:
    """Derive force/area/velocity/inertia unit scales (UNITSET)."""
    state.unitf = state.unitm * state.unitl / (state.unitt * state.unitt)
    state.units = state.unitl * state.unitl
    state.unitv = state.unitl / state.unitt
    state.unita = state.unitl / (state.unitt * state.unitt)
    state.uniti = state.unitm * state.unitl * state.unitl
    state.unitd = state.unitm / (state.unitl * state.unitl * state.unitl)
    return state


def _read_mass_properties(mass_file: str | Path | None) -> MassProperties | None:
    """Read a mass file and return parsed properties, or ``None``."""
    if mass_file is None or str(mass_file).strip() == "":
        return None

    path = Path(mass_file)
    if not path.is_file():
        return None

    props = parse_mass_file(path)
    return props if props.loaded else None


def masget(state: Any, mass_file: str | Path | None) -> bool:
    """Read a mass file and populate default mass/inertia on state (MASGET)."""
    masini(state)
    props = _read_mass_properties(mass_file)
    if props is None:
        return False

    _apply_mass_properties(state, props)
    unitset(state)
    return True


def masput(state: Any, ir1: int = 0, ir2: int = 0) -> Any:
    """Store default mass and inertias in run-case parameter array (MASPUT)."""
    for ir in range(ir1, ir2 + 1):
        state.parval[C.IPMASS, ir] = state.rmass0
        state.parval[C.IPIXX, ir] = state.riner0[0, 0]
        state.parval[C.IPIYY, ir] = state.riner0[1, 1]
        state.parval[C.IPIZZ, ir] = state.riner0[2, 2]
        state.parval[C.IPIXY, ir] = state.riner0[0, 1]
        state.parval[C.IPIYZ, ir] = state.riner0[1, 2]
        state.parval[C.IPIZX, ir] = state.riner0[2, 0]

        state.parval[C.IPGEE, ir] = state.gee0
        state.parval[C.IPRHO, ir] = state.rho0

        if state.unitl != 0.0:
            state.parval[C.IPXCG, ir] = state.xyzmass0[0] / state.unitl
            state.parval[C.IPYCG, ir] = state.xyzmass0[1] / state.unitl
            state.parval[C.IPZCG, ir] = state.xyzmass0[2] / state.unitl
    return state


def appget(state: Any) -> Any:
    """Calculate apparent mass and inertia for unit air density (APPGET)."""
    state.amass[:, :] = 0.0
    state.ainer[:, :] = 0.0

    unitl3 = state.unitl**3
    unitl5 = state.unitl**5
    for j in range(state.nstrip):
        cr = float(state.chord[j])
        sr = cr * float(state.wstrip[j])

        un = np.array([0.0, state.ensy[j], state.ensz[j]], dtype=np.float64)
        us = np.array(
            [
                state.rle2[0, j] - state.rle1[0, j] + 0.5 * (state.chord2[j] - state.chord1[j]),
                state.rle2[1, j] - state.rle1[1, j],
                state.rle2[2, j] - state.rle1[2, j],
            ],
            dtype=np.float64,
        )
        umag = float(np.linalg.norm(us))
        if umag > 0.0:
            us /= umag

        rm = np.array([state.rle[0, j] + 0.5 * cr, state.rle[1, j], state.rle[2, j]], dtype=np.float64)
        rxun = np.cross(rm, un)

        cperp = cr * (us[1] * un[2] - us[2] * un[1])
        appm = sr * 0.25 * np.pi * cperp
        appi = sr * 0.25 * np.pi * cperp**3 / 64.0

        state.amass += appm * np.outer(un, un) * unitl3
        state.ainer += (appm * np.outer(rxun, rxun) + appi * np.outer(us, us)) * unitl5

    return state


def load_mass(state: Any, mass_file: str | Path) -> MassProperties | None:
    """Parse a .mass file and write results into state and PARVAL."""
    masini(state)
    props = _read_mass_properties(mass_file)
    if props is None:
        return None
    _apply_mass_properties(state, props)
    unitset(state)
    appget(state)
    masput(state, 0, 0)
    return props
