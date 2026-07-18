"""Geometry and mass settings inspection for the high-level solver API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openavl.fileio.parser import BodyDef, SectionDef, SurfaceDef

_SETTINGS_SECTIONS = ("aircraft", "surfaces", "bodies", "mass")


def get_settings(self, section: str | None = None) -> dict[str, Any]:
    """Return a summary of current geometry and mass settings.

    The payload is intentionally compact: reference geometry, surface/body
    outlines (including airfoil labels), lattice counts, and mass properties.
    Run-case parameters and trim constraints are not included.

    Parameters
    ----------
    section:
        Optional filter. One of ``aircraft``, ``surfaces``, ``bodies``, or
        ``mass``. When ``None``, all sections are returned.

    Returns
    -------
    dict[str, Any]
        Nested settings dictionary. With ``section`` set, the returned
        mapping contains only that key.

    Raises
    ------
    KeyError
        If ``section`` is not a recognized settings section.
    """
    data = {
        "aircraft": _aircraft_summary(self),
        "surfaces": [_surface_summary(surf) for surf in self.model.surfaces],
        "bodies": [_body_summary(body) for body in self.model.bodies],
        "mass": _mass_summary(self),
    }
    if section is None:
        return data
    key = section.strip().lower()
    if key not in data:
        raise KeyError(
            f"Unknown settings section: {section!r}. "
            f"Choose from {_SETTINGS_SECTIONS}."
        )
    return {key: data[key]}


def print_settings(self, section: str | None = None) -> None:
    """Pretty-print current geometry and mass settings to the console.

    Parameters
    ----------
    section:
        Optional filter matching :meth:`get_settings`. When ``None``, prints
        the full geometry and mass summary.
    """
    print(format_settings(get_settings(self, section=section)))


def format_settings(settings: dict[str, Any]) -> str:
    """Format a settings dictionary as a human-readable multi-line string."""
    lines: list[str] = []
    if "aircraft" in settings:
        lines.extend(_format_aircraft(settings["aircraft"]))
    if "surfaces" in settings:
        lines.extend(_format_surfaces(settings["surfaces"]))
    if "bodies" in settings:
        lines.extend(_format_bodies(settings["bodies"]))
    if "mass" in settings:
        lines.extend(_format_mass(settings["mass"]))
    return "\n".join(lines)


def _aircraft_summary(solver: Any) -> dict[str, Any]:
    """Build the top-level aircraft / lattice summary dictionary."""
    header = solver.model.header
    state = solver.state
    geo_file = getattr(solver, "geo_file", None)
    mass_file = getattr(solver, "mass_file", None)
    return {
        "title": header.title,
        "geo_file": str(geo_file) if geo_file is not None else None,
        "mass_file": str(mass_file) if mass_file is not None else None,
        "sref": float(header.sref),
        "cref": float(header.cref),
        "bref": float(header.bref),
        "xref": float(header.xref),
        "yref": float(header.yref),
        "zref": float(header.zref),
        "mach": float(header.mach),
        "iysym": int(header.iysym),
        "izsym": int(header.izsym),
        "zsym": float(header.zsym),
        "nsurf": int(state.nsurf),
        "nvor": int(state.nvor),
        "nstrip": int(state.nstrip),
        "nbody": int(state.nbody),
        "controls": list(state.control_names),
    }


def _surface_summary(surf: SurfaceDef) -> dict[str, Any]:
    """Build a compact summary for one lifting surface."""
    airfoils: list[str] = []
    controls: list[str] = []
    for sec in surf.sections:
        label = _airfoil_label(sec)
        if label not in airfoils:
            airfoils.append(label)
        for ctrl in sec.controls:
            if ctrl.name not in controls:
                controls.append(ctrl.name)
    return {
        "name": surf.name,
        "n_chord": int(surf.n_chord),
        "c_space": float(surf.c_space),
        "n_span": int(surf.n_span),
        "s_space": float(surf.s_space),
        "angle_deg": float(surf.angle_deg),
        "yduplicate": None if surf.yduplicate is None else float(surf.yduplicate),
        "n_sections": len(surf.sections),
        "airfoils": airfoils,
        "controls": controls,
        "nowake": bool(surf.nowake),
        "noload": bool(surf.noload),
    }


def _body_summary(body: BodyDef) -> dict[str, Any]:
    """Build a compact summary for one body."""
    body_file = body.body_file
    return {
        "name": body.name,
        "n_body": int(body.n_body),
        "b_space": float(body.b_space),
        "body_file": Path(body_file).name if body_file else None,
        "yduplicate": None if body.yduplicate is None else float(body.yduplicate),
    }


def _mass_summary(solver: Any) -> dict[str, Any]:
    """Build the mass / inertia summary from the loaded model or state defaults."""
    props = getattr(solver.model, "mass", None)
    if props is not None and props.loaded:
        inertia = props.inertia
        return {
            "loaded": True,
            "mass": float(props.mass),
            "cg": [float(props.cg[0]), float(props.cg[1]), float(props.cg[2])],
            "inertia": {
                "Ixx": float(inertia[0, 0]),
                "Iyy": float(inertia[1, 1]),
                "Izz": float(inertia[2, 2]),
                "Ixy": float(inertia[0, 1]),
                "Iyz": float(inertia[1, 2]),
                "Izx": float(inertia[2, 0]),
            },
            "units": {
                "length": props.lunit_name,
                "mass": props.munit_name,
                "time": props.tunit_name,
                "unitl": float(props.unitl),
                "unitm": float(props.unitm),
                "unitt": float(props.unitt),
            },
            "gee": float(props.gee),
            "rho": float(props.rho),
            "n_components": len(props.components),
        }

    state = solver.state
    inertia = state.riner0
    return {
        "loaded": False,
        "mass": float(state.rmass0),
        "cg": [
            float(state.xyzmass0[0]),
            float(state.xyzmass0[1]),
            float(state.xyzmass0[2]),
        ],
        "inertia": {
            "Ixx": float(inertia[0, 0]),
            "Iyy": float(inertia[1, 1]),
            "Izz": float(inertia[2, 2]),
            "Ixy": float(inertia[0, 1]),
            "Iyz": float(inertia[1, 2]),
            "Izx": float(inertia[2, 0]),
        },
        "units": None,
        "gee": float(state.gee0),
        "rho": float(state.rho0),
        "n_components": 0,
    }


def _airfoil_label(sec: SectionDef) -> str:
    """Return a short airfoil identifier for summary listings."""
    if sec.naca:
        return f"NACA {sec.naca}"
    if sec.airfoil_file:
        return Path(sec.airfoil_file).name
    if sec.airfoil_coords:
        return "inline"
    return "flat"


def _format_aircraft(aircraft: dict[str, Any]) -> list[str]:
    """Format the aircraft section for console output."""
    controls = ", ".join(aircraft["controls"]) if aircraft["controls"] else "(none)"
    return [
        "=== Aircraft ===",
        f"  title    : {aircraft['title']}",
        f"  geo_file : {aircraft['geo_file'] or '(Geometry API)'}",
        f"  mass_file: {aircraft['mass_file'] or '(none)'}",
        (
            f"  Sref/Cref/Bref : {aircraft['sref']:.6g} / "
            f"{aircraft['cref']:.6g} / {aircraft['bref']:.6g}"
        ),
        (
            f"  XYZRef  : ({aircraft['xref']:.6g}, "
            f"{aircraft['yref']:.6g}, {aircraft['zref']:.6g})"
        ),
        (
            f"  lattice : {aircraft['nsurf']} surfaces, "
            f"{aircraft['nbody']} bodies, "
            f"{aircraft['nstrip']} strips, "
            f"{aircraft['nvor']} vortices"
        ),
        f"  controls: {controls}",
        "",
    ]


def _format_surfaces(surfaces: list[dict[str, Any]]) -> list[str]:
    """Format the surfaces section for console output."""
    lines = ["=== Surfaces ==="]
    if not surfaces:
        lines.append("  (none)")
        lines.append("")
        return lines
    for idx, surf in enumerate(surfaces):
        yduple = "none" if surf["yduplicate"] is None else f"{surf['yduplicate']:.6g}"
        airfoils = ", ".join(surf["airfoils"]) if surf["airfoils"] else "(none)"
        controls = ", ".join(surf["controls"]) if surf["controls"] else "(none)"
        flags: list[str] = []
        if surf["nowake"]:
            flags.append("nowake")
        if surf["noload"]:
            flags.append("noload")
        flag_text = f"  [{' '.join(flags)}]" if flags else ""
        lines.append(
            f"  [{idx}] {surf['name']}  "
            f"nchord={surf['n_chord']}  nspan={surf['n_span']}  "
            f"angle={surf['angle_deg']:.4g} deg  yduple={yduple}"
            f"{flag_text}"
        )
        lines.append(
            f"      sections={surf['n_sections']}  "
            f"airfoils=[{airfoils}]  controls=[{controls}]"
        )
    lines.append("")
    return lines


def _format_bodies(bodies: list[dict[str, Any]]) -> list[str]:
    """Format the bodies section for console output."""
    lines = ["=== Bodies ==="]
    if not bodies:
        lines.append("  (none)")
        lines.append("")
        return lines
    for idx, body in enumerate(bodies):
        body_file = body["body_file"] or "(inline/none)"
        yduple = "none" if body["yduplicate"] is None else f"{body['yduplicate']:.6g}"
        lines.append(
            f"  [{idx}] {body['name']}  "
            f"nbody={body['n_body']}  file={body_file}  yduple={yduple}"
        )
    lines.append("")
    return lines


def _format_mass(mass: dict[str, Any]) -> list[str]:
    """Format the mass section for console output."""
    inertia = mass["inertia"]
    source = "mass file" if mass["loaded"] else "defaults (no mass file)"
    lines = [
        "=== Mass ===",
        f"  source : {source}",
        f"  mass   : {mass['mass']:.6g}",
        (
            f"  CG     : ({mass['cg'][0]:.6g}, "
            f"{mass['cg'][1]:.6g}, {mass['cg'][2]:.6g})"
        ),
        (
            f"  Ixx/Iyy/Izz : {inertia['Ixx']:.6g} / "
            f"{inertia['Iyy']:.6g} / {inertia['Izz']:.6g}"
        ),
        (
            f"  Ixy/Iyz/Izx : {inertia['Ixy']:.6g} / "
            f"{inertia['Iyz']:.6g} / {inertia['Izx']:.6g}"
        ),
    ]
    units = mass.get("units")
    if units:
        lines.append(
            f"  units  : {units['length']}={units['unitl']:g}, "
            f"{units['mass']}={units['unitm']:g}, "
            f"{units['time']}={units['unitt']:g}"
        )
        lines.append(f"  gee/rho: {mass['gee']:.6g} / {mass['rho']:.6g}")
        lines.append(f"  components: {mass['n_components']}")
    lines.append("")
    return lines
