"""Shared display labels and colors for geometry consumers."""

from __future__ import annotations

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

CONTROL_COLOR = "#e63946"
BODY_COLOR = "#6c757d"
REF_POINT_COLOR = "#212529"


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
