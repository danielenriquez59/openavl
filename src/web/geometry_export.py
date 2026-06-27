"""Export AVL lattice geometry as JSON triangle meshes for Three.js."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from openavl.geom.geometry import solver_surface_name
from openavl.geom.display import BODY_COLOR, surface_color

if TYPE_CHECKING:
    from openavl.core.state import AVLState
    from openavl.fileio.parser import AVLModel

_BODY_COLOR = BODY_COLOR


def _hex_to_rgba(hex_color: str, alpha: float = 0.85) -> list[float]:
    """Convert ``#RRGGBB`` to normalized RGBA floats."""
    clean = hex_color.lstrip("#")
    if len(clean) != 6:
        return [0.5, 0.5, 0.5, alpha]
    r = int(clean[0:2], 16) / 255.0
    g = int(clean[2:4], 16) / 255.0
    b = int(clean[4:6], 16) / 255.0
    return [r, g, b, alpha]


def _surface_labels(model: AVLModel, nsurf: int) -> list[str]:
    """Map lattice surface indices to human-readable labels."""
    return [solver_surface_name(model, isurf) for isurf in range(nsurf)]


def _surface_color(name: str, component: int, index: int) -> list[float]:
    """Pick a stable RGBA color for a lifting surface."""
    return _hex_to_rgba(surface_color(name, component, index))


def _build_surface_mesh(
    state: AVLState,
    isurf: int,
    *,
    include_dcp: bool,
) -> tuple[list[float], list[int], list[float] | None]:
    """Build flat position/index arrays for one lattice surface."""
    positions: list[float] = []
    indices: list[int] = []
    dcp_values: list[float] | None = [] if include_dcp else None
    vertex_base = 0

    nvor = int(state.nvor)
    dcp_arr = state.dcp[:nvor] if include_dcp and nvor > 0 else None

    for j in range(int(state.nstrip)):
        if int(state.lssurf[j]) != isurf:
            continue
        i0 = int(state.ijfrst[j])
        nvc = int(state.nvstrp[j])
        if nvc <= 1:
            continue

        for ivc in range(nvc - 1):
            i = i0 + ivc
            i_next = i0 + ivc + 1
            corners = (
                (state.rv1[0, i], state.rv1[1, i], state.rv1[2, i]),
                (state.rv2[0, i], state.rv2[1, i], state.rv2[2, i]),
                (state.rv2[0, i_next], state.rv2[1, i_next], state.rv2[2, i_next]),
                (state.rv1[0, i_next], state.rv1[1, i_next], state.rv1[2, i_next]),
            )
            corner_dcp = (
                float(dcp_arr[i]) if dcp_arr is not None else 0.0,
                float(dcp_arr[i]) if dcp_arr is not None else 0.0,
                float(dcp_arr[i_next]) if dcp_arr is not None else 0.0,
                float(dcp_arr[i_next]) if dcp_arr is not None else 0.0,
            )

            for corner, dcp_val in zip(corners, corner_dcp):
                positions.extend(corner)
                if dcp_values is not None:
                    dcp_values.append(dcp_val)

            indices.extend(
                [
                    vertex_base,
                    vertex_base + 1,
                    vertex_base + 2,
                    vertex_base,
                    vertex_base + 2,
                    vertex_base + 3,
                ]
            )
            vertex_base += 4

    return positions, indices, dcp_values


def _build_body_mesh(state: AVLState, ibody: int) -> tuple[list[float], list[int]]:
    """Build a coarse tube mesh for one fuselage body from solver nodes."""
    positions: list[float] = []
    indices: list[int] = []
    i0 = int(state.lfrst[ibody])
    n_nodes = int(state.nl[ibody])
    if n_nodes < 2:
        return positions, indices

    xs = state.rl[0, i0 : i0 + n_nodes]
    ys = state.rl[1, i0 : i0 + n_nodes]
    zs = state.rl[2, i0 : i0 + n_nodes]
    radii = state.radl[i0 : i0 + n_nodes]
    n_ring = 16
    ring_verts: list[list[tuple[float, float, float]]] = []

    for k in range(n_nodes):
        radius = float(radii[k])
        if radius <= 0.0:
            ring_verts.append([])
            continue
        theta = np.linspace(0.0, 2.0 * math.pi, n_ring, endpoint=False)
        ring = [
            (float(xs[k]), float(ys[k] + radius * math.cos(t)), float(zs[k] + radius * math.sin(t)))
            for t in theta
        ]
        ring_verts.append(ring)

    vertex_base = 0
    prev_ring_start: int | None = None
    prev_ring_count = 0
    for ring in ring_verts:
        if not ring:
            continue
        for x, y, z in ring:
            positions.extend([x, y, z])
        ring_start = vertex_base
        ring_count = len(ring)
        if prev_ring_start is not None and prev_ring_count > 0:
            for t in range(ring_count):
                t_next = (t + 1) % ring_count
                a = prev_ring_start + t
                b = prev_ring_start + t_next
                c = ring_start + t_next
                d = ring_start + t
                indices.extend([a, b, c, a, c, d])
        prev_ring_start = ring_start
        prev_ring_count = ring_count
        vertex_base += ring_count

    return positions, indices


def model_to_geometry(
    model: AVLModel,
    state: AVLState | None = None,
    *,
    include_dcp: bool = False,
) -> dict[str, Any]:
    """Convert an AVL model and optional solver state to Three.js-ready meshes.

    Parameters
    ----------
    model:
        Parsed AVL geometry.
    state:
        Built lattice state. When omitted, geometry is allocated from the model
        without running a flow solution.
    include_dcp:
        When ``True`` and ``state`` contains solved ``dcp`` values, attach a
        per-vertex pressure coefficient array on each surface mesh.

    Returns
    -------
    dict
        ``{"surfaces": [...], "bodies": [...]}`` where each entry has ``name``,
        ``color`` (RGBA floats), ``positions``, ``indices``, and optional ``dcp``.
    """
    from openavl.core.state import AVLState
    from openavl.geom.geometry import build_geometry

    if state is None:
        state = AVLState.from_model(model)
        build_geometry(state, model)

    labels = _surface_labels(model, int(state.nsurf))
    surfaces: list[dict[str, Any]] = []

    for isurf in range(int(state.nsurf)):
        comp = int(state.lncomp[isurf]) if isurf < state.lncomp.size else isurf + 1
        positions, indices, dcp_values = _build_surface_mesh(
            state,
            isurf,
            include_dcp=include_dcp and int(state.nvor) > 0,
        )
        if not positions:
            continue
        entry: dict[str, Any] = {
            "name": labels[isurf],
            "color": _surface_color(labels[isurf], comp, isurf),
            "positions": positions,
            "indices": indices,
        }
        if dcp_values is not None:
            entry["dcp"] = dcp_values
        surfaces.append(entry)

    bodies: list[dict[str, Any]] = []
    body_color = _hex_to_rgba(_BODY_COLOR)
    if int(state.nbody) > 0:
        for ibody in range(int(state.nbody)):
            positions, indices = _build_body_mesh(state, ibody)
            if not positions:
                continue
            name = model.bodies[ibody].name if ibody < len(model.bodies) else f"Body {ibody + 1}"
            bodies.append(
                {
                    "name": name,
                    "color": body_color,
                    "positions": positions,
                    "indices": indices,
                }
            )

    return {"surfaces": surfaces, "bodies": bodies}
