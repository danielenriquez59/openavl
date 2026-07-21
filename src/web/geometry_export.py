"""Export AVL lattice geometry as JSON triangle meshes for Three.js."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openavl.fileio.cad_export import build_body_mesh, build_surface_mesh
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


def _build_surface_panel_lines(state: AVLState, isurf: int) -> list[float]:
    """Build explicit chordwise and spanwise aerodynamic panel-edge segments."""
    positions: list[float] = []

    for j in range(int(state.nstrip)):
        if int(state.lssurf[j]) != isurf:
            continue

        i0 = int(state.ijfrst[j])
        nvc = int(state.nvstrp[j])
        if nvc <= 0:
            continue

        leading_edge = (
            (
                float(state.rle1[0, j]),
                float(state.rle1[1, j]),
                float(state.rle1[2, j]),
            ),
            (
                float(state.rle2[0, j]),
                float(state.rle2[1, j]),
                float(state.rle2[2, j]),
            ),
        )

        left_le, right_le = leading_edge
        for ivc in range(nvc):
            i = i0 + ivc
            left_te = (
                float(state.xyn1[0, i]),
                float(state.xyn1[1, i]),
                float(0.5 * (state.zlon1[i] + state.zupn1[i])),
            )
            right_te = (
                float(state.xyn2[0, i]),
                float(state.xyn2[1, i]),
                float(0.5 * (state.zlon2[i] + state.zupn2[i])),
            )

            # Every panel is emitted as a closed loop. Shared edges are
            # intentionally repeated so adjacent panels remain explicit.
            positions.extend((*left_le, *right_le))
            positions.extend((*right_le, *right_te))
            positions.extend((*right_te, *left_te))
            positions.extend((*left_te, *left_le))
            left_le, right_le = left_te, right_te

    return positions


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
        positions, indices, dcp_values = build_surface_mesh(
            state,
            isurf,
            include_dcp=include_dcp and int(state.nvor) > 0,
        )
        panel_lines = _build_surface_panel_lines(state, isurf)
        if not positions:
            continue
        entry: dict[str, Any] = {
            "name": labels[isurf],
            "color": _surface_color(labels[isurf], comp, isurf),
            "positions": positions,
            "indices": indices,
            "panel_lines": panel_lines,
        }
        if dcp_values is not None:
            entry["dcp"] = dcp_values
        surfaces.append(entry)

    bodies: list[dict[str, Any]] = []
    body_color = _hex_to_rgba(_BODY_COLOR)
    if int(state.nbody) > 0:
        for ibody in range(int(state.nbody)):
            positions, indices = build_body_mesh(state, ibody)
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
