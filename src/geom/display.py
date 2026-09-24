"""Shared display labels and colors for geometry consumers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openavl.fileio.parser import AVLModel, ControlDef, SectionDef, SurfaceDef

# Distinct colors for named surface roles; unknown surfaces fall back to tab10.
SURFACE_COLORS: dict[str, str] = {
    "inner wing": "#1f77b4",
    "outer wing": "#2ca02c",
    "stab": "#ff7f0e",
    "fin": "#9467bd",
    "fuse pod": "#8c564b",
    "fuselage": "#8c564b",
    "body": "#8c564b",
}

COMPONENT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

CONTROL_COLOR = "#3f3f46"
BODY_COLOR = "#6c757d"
REF_POINT_COLOR = "#212529"


def _control_at_section(section: SectionDef, name: str) -> ControlDef | None:
    """Return the named control declaration on one section."""
    return next((control for control in section.controls if control.name == name), None)


def _section_le_point(section: SectionDef, surface: SurfaceDef) -> tuple[float, float, float]:
    """Transform a section leading-edge point into solver coordinates."""
    scale = surface.scale or [1.0, 1.0, 1.0]
    translate = surface.translate or [0.0, 0.0, 0.0]
    return (
        scale[0] * section.xle + translate[0],
        scale[1] * section.yle + translate[1],
        scale[2] * section.zle + translate[2],
    )


def _hinge_point(section: SectionDef, surface: SurfaceDef, xhinge: float) -> tuple[float, float, float]:
    """Transform one section hinge-fraction point into solver coordinates."""
    scale = surface.scale or [1.0, 1.0, 1.0]
    translate = surface.translate or [0.0, 0.0, 0.0]
    return (
        scale[0] * (section.xle + section.chord * abs(xhinge)) + translate[0],
        scale[1] * section.yle + translate[1],
        scale[2] * section.zle + translate[2],
    )


def section_airfoil_labels(
    model: AVLModel,
) -> list[tuple[str, tuple[float, float, float]]]:
    """Return AFIL basenames at each section leading-edge location.

    Only sections with an ``AFIL`` path are included. Positions use the same
    surface scale/translate as control hinges.
    """
    labels: list[tuple[str, tuple[float, float, float]]] = []
    for surface in model.surfaces:
        for section in surface.sections:
            if not section.airfoil_file:
                continue
            name = Path(section.airfoil_file).name
            if not name:
                continue
            labels.append((name, _section_le_point(section, surface)))
    return labels


def control_hinge_polylines(
    model: AVLModel,
) -> list[tuple[str, str, list[tuple[float, float, float]]]]:
    """Return contiguous control-hinge runs at their declared section bounds."""
    polylines: list[tuple[str, str, list[tuple[float, float, float]]]] = []

    for surface in model.surfaces:
        control_names = list(
            dict.fromkeys(control.name for section in surface.sections for control in section.controls)
        )
        surface_runs: list[tuple[str, str, list[tuple[float, float, float]]]] = []

        for name in control_names:
            current: list[tuple[float, float, float]] | None = None
            for left, right in zip(surface.sections, surface.sections[1:]):
                left_control = _control_at_section(left, name)
                right_control = _control_at_section(right, name)
                if left_control is not None and right_control is not None:
                    left_point = _hinge_point(left, surface, left_control.xhinge)
                    right_point = _hinge_point(right, surface, right_control.xhinge)
                    if current is None:
                        current = [left_point, right_point]
                    else:
                        current.append(right_point)
                elif current is not None:
                    surface_runs.append((surface.name, name, current))
                    current = None

            if current is not None:
                surface_runs.append((surface.name, name, current))

        polylines.extend(surface_runs)
        if surface.yduplicate is not None:
            for surface_name, control_name, points in surface_runs:
                mirrored = [
                    (x, -y + surface.yduplicate, z)
                    for x, y, z in points
                ]
                polylines.append((surface_name, control_name, mirrored))

    return polylines


def surface_color(name: str, component: int, index: int) -> str:
    """Pick a stable hex color for a solver surface."""
    key = name.strip().lower()
    if key in SURFACE_COLORS:
        return SURFACE_COLORS[key]
    if "mirror" in key:
        base = key.replace(" (mirror)", "").strip()
        if base in SURFACE_COLORS:
            return SURFACE_COLORS[base]
    if component > 0:
        return COMPONENT_COLORS[(component - 1) % len(COMPONENT_COLORS)]
    return COMPONENT_COLORS[index % len(COMPONENT_COLORS)]
