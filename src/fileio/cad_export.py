"""Export aero lattice geometry to CAD-readable formats."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from openavl.geom.geometry import solver_surface_name

if TYPE_CHECKING:
    from openavl.core.state import AVLState
    from openavl.fileio.parser import AVLModel

_STL_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def build_surface_mesh(
    state: AVLState,
    isurf: int,
    *,
    include_dcp: bool = False,
) -> tuple[list[float], list[int], list[float] | None]:
    """Build flat position/index arrays for one lattice surface.

    Parameters
    ----------
    state:
        Built solver state containing vortex-lattice arrays.
    isurf:
        Zero-based surface index.
    include_dcp:
        When ``True``, also return per-vertex ``dcp`` samples aligned with
        ``positions``.

    Returns
    -------
    positions, indices, dcp_values
        Flat XYZ positions, triangle indices (two tris per panel quad), and
        optional ``dcp`` values (``None`` when ``include_dcp`` is ``False``).
    """
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


def build_body_mesh(state: AVLState, ibody: int) -> tuple[list[float], list[int]]:
    """Build a coarse tube mesh for one fuselage body from solver nodes.

    Parameters
    ----------
    state:
        Built solver state containing body centerline and radius arrays.
    ibody:
        Zero-based body index.

    Returns
    -------
    positions, indices
        Flat XYZ positions and triangle indices for a 16-sided tube.
    """
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


def collect_aero_meshes(
    model: AVLModel,
    state: AVLState,
    *,
    include_bodies: bool = True,
) -> list[dict[str, Any]]:
    """Collect named triangle meshes for lifting surfaces and bodies.

    Parameters
    ----------
    model:
        Parsed AVL geometry (used for surface/body names).
    state:
        Built lattice state.
    include_bodies:
        When ``True``, include fuselage body tube meshes.

    Returns
    -------
    list[dict]
        Each entry has ``name``, ``positions`` (flat XYZ), and ``indices``.
    """
    solids: list[dict[str, Any]] = []

    for isurf in range(int(state.nsurf)):
        positions, indices, _ = build_surface_mesh(state, isurf, include_dcp=False)
        if not positions:
            continue
        solids.append(
            {
                "name": solver_surface_name(model, isurf),
                "positions": positions,
                "indices": indices,
            }
        )

    if include_bodies and int(state.nbody) > 0:
        for ibody in range(int(state.nbody)):
            positions, indices = build_body_mesh(state, ibody)
            if not positions:
                continue
            name = model.bodies[ibody].name if ibody < len(model.bodies) else f"Body {ibody + 1}"
            solids.append(
                {
                    "name": name,
                    "positions": positions,
                    "indices": indices,
                }
            )

    return solids


def sanitize_stl_name(name: str) -> str:
    """Sanitize a component name for use in an ASCII STL ``solid`` header."""
    cleaned = _STL_NAME_RE.sub("_", name.strip().replace(" ", "_"))
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-")
    return cleaned or "solid"


def _facet_normal(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Return a unit normal for triangle ``(v0, v1, v2)``, or zeros if degenerate."""
    normal = np.cross(v1 - v0, v2 - v0)
    length = float(np.linalg.norm(normal))
    if length <= 0.0:
        return np.zeros(3, dtype=float)
    return normal / length


def write_ascii_stl(path: str | Path, solids: list[dict[str, Any]]) -> Path:
    """Write named triangle meshes as a multi-solid ASCII STL file.

    Parameters
    ----------
    path:
        Destination file path.
    solids:
        Mesh entries with ``name``, ``positions``, and ``indices``.

    Returns
    -------
    pathlib.Path
        Path to the written STL file.
    """
    out = Path(path)
    lines: list[str] = []

    for solid in solids:
        name = sanitize_stl_name(str(solid["name"]))
        positions = np.asarray(solid["positions"], dtype=float).reshape(-1, 3)
        indices = np.asarray(solid["indices"], dtype=int)

        lines.append(f"solid {name}")
        for i0, i1, i2 in indices.reshape(-1, 3):
            v0 = positions[i0]
            v1 = positions[i1]
            v2 = positions[i2]
            normal = _facet_normal(v0, v1, v2)
            if float(np.linalg.norm(normal)) <= 0.0:
                continue
            lines.append(
                f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}"
            )
            lines.append("    outer loop")
            lines.append(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}")
            lines.append(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}")
            lines.append(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}")
            lines.append("    endloop")
            lines.append("  endfacet")
        lines.append(f"endsolid {name}")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def export_stl(
    model: AVLModel,
    state: AVLState,
    path: str | Path,
    *,
    include_bodies: bool = True,
) -> Path:
    """Export the aero lattice mesh (surfaces + bodies) to an ASCII STL file.

    The mesh matches the vortex-lattice panels and body tubes used by the
    solver; it is not a watertight CAD outer-mold-line solid.

    Parameters
    ----------
    model:
        Parsed AVL geometry.
    state:
        Built lattice state.
    path:
        Destination ``.stl`` path.
    include_bodies:
        When ``True``, include fuselage body tube meshes.

    Returns
    -------
    pathlib.Path
        Path to the written STL file.
    """
    solids = collect_aero_meshes(model, state, include_bodies=include_bodies)
    return write_ascii_stl(path, solids)
