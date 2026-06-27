"""Three-dimensional matplotlib visualization of AVL aircraft geometry."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from openavl.core.solver import AVLSolver
    from openavl.core.state import AVLState
    from openavl.fileio.parser import AVLModel, BodyDef, SectionDef, SurfaceDef
    from openavl.geometry.aircraft import Aircraft

from openavl.geom.display import BODY_COLOR, CONTROL_COLOR, REF_POINT_COLOR, surface_color

_CONTROL_COLOR = CONTROL_COLOR
_BODY_COLOR = BODY_COLOR
_REF_POINT_COLOR = REF_POINT_COLOR


def _import_matplotlib():
    """Import matplotlib lazily so core solver imports stay lightweight."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "plot_aircraft_3d requires matplotlib, which should be installed with openavl."
        ) from exc
    return plt


def _resolve_geometry(
    source: str | Path | Aircraft | AVLModel | AVLSolver,
    base_dir: str | Path | None,
) -> tuple[AVLModel, AVLState | None]:
    """Normalize plotting input to an AVLModel and optional built solver state."""
    from openavl.core.solver import AVLSolver
    from openavl.fileio.parser import AVLModel, parse_avl_file, prepare_model
    from openavl.geometry.aircraft import Aircraft

    if isinstance(source, AVLSolver):
        return source.model, source.state

    if isinstance(source, Aircraft):
        resolved_base = Path(base_dir) if base_dir is not None else None
        return source.to_avl_model(base_dir=resolved_base), None

    if isinstance(source, AVLModel):
        return source, None

    geo_path = Path(source)
    model = prepare_model(parse_avl_file(geo_path), base_dir=geo_path.parent)
    return model, None


def _build_state(model: AVLModel) -> AVLState:
    """Allocate solver state and construct panel/body geometry for plotting."""
    from openavl.core.state import AVLState
    from openavl.geom.geometry import build_geometry

    state = AVLState.from_model(model)
    build_geometry(state, model)
    return state


def _surface_labels(model: AVLModel, nsurf: int) -> list[str]:
    """Map lattice surface indices to human-readable labels."""
    from openavl.geom.geometry import solver_surface_name

    return [solver_surface_name(model, isurf) for isurf in range(nsurf)]


def _surface_color(name: str, component: int, index: int) -> str:
    """Pick a stable color for a lifting surface."""
    return surface_color(name, component, index)


def _transform_section_point(
    section: SectionDef,
    surf: SurfaceDef,
    *,
    x_fraction: float = 0.0,
) -> np.ndarray:
    """Return a 3D point along the local chord at ``x_fraction`` from the LE."""
    scale = np.asarray(surf.scale if surf.scale else [1.0, 1.0, 1.0], dtype=np.float64)
    translate = np.asarray(
        surf.translate if surf.translate else [0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    ainc = math.radians(section.ainc_deg + (surf.angle_deg or 0.0))
    chord = scale[0] * section.chord
    le = np.array(
        [
            scale[0] * section.xle + translate[0],
            scale[1] * section.yle + translate[1],
            scale[2] * section.zle + translate[2],
        ],
        dtype=np.float64,
    )
    return le + x_fraction * np.array(
        [chord * math.cos(ainc), 0.0, chord * math.sin(ainc)],
        dtype=np.float64,
    )


def _mirror_point(point: np.ndarray, yduplicate: float) -> np.ndarray:
    """Reflect a point about the AVL ``YDUPLICATE`` plane."""
    mirrored = point.copy()
    mirrored[1] = -mirrored[1] + yduplicate
    return mirrored


def _collect_control_polylines(model: AVLModel) -> list[tuple[str, str, np.ndarray]]:
    """Collect hinge-line polylines for each control on each surface."""
    polylines: list[tuple[str, str, np.ndarray]] = []
    for surf in model.surfaces:
        controls: dict[str, list[np.ndarray]] = {}
        for section in surf.sections:
            for ctrl in section.controls:
                hinge = _transform_section_point(section, surf, x_fraction=ctrl.xhinge)
                controls.setdefault(ctrl.name, []).append(hinge)
                if surf.yduplicate is not None:
                    controls.setdefault(ctrl.name, []).append(
                        _mirror_point(hinge, surf.yduplicate)
                    )
        for name, points in controls.items():
            if len(points) >= 2:
                order = np.argsort([pt[1] for pt in points])
                ordered = np.asarray([points[i] for i in order], dtype=np.float64)
            else:
                ordered = np.asarray(points, dtype=np.float64)
            polylines.append((surf.name, name, ordered))
    return polylines


def _plot_lattice_surface(
    ax: Axes,
    state: AVLState,
    isurf: int,
    color: str,
    label: str,
) -> None:
    """Draw vortex-lattice panel edges for one surface index."""
    drawn_label = False
    for j in range(int(state.nstrip)):
        if int(state.lssurf[j]) != isurf:
            continue
        i0 = int(state.ijfrst[j])
        nvc = int(state.nvstrp[j])
        if nvc <= 0:
            continue

        for ivc in range(nvc - 1):
            i = i0 + ivc
            i_next = i0 + ivc + 1
            xs = [state.rv1[0, i], state.rv2[0, i], state.rv2[0, i_next], state.rv1[0, i_next], state.rv1[0, i]]
            ys = [state.rv1[1, i], state.rv2[1, i], state.rv2[1, i_next], state.rv1[1, i_next], state.rv1[1, i]]
            zs = [state.rv1[2, i], state.rv2[2, i], state.rv2[2, i_next], state.rv1[2, i_next], state.rv1[2, i]]
            ax.plot(
                xs,
                ys,
                zs,
                color=color,
                linewidth=0.6,
                alpha=0.85,
                label=label if not drawn_label else None,
            )
            drawn_label = True


def _plot_wireframe_surface(
    ax: Axes,
    surf: SurfaceDef,
    color: str,
    label: str,
) -> None:
    """Draw section leading-edge and chord lines when lattice state is unavailable."""
    if len(surf.sections) < 2:
        return

    le_points = [_transform_section_point(sec, surf, x_fraction=0.0) for sec in surf.sections]
    te_points = [_transform_section_point(sec, surf, x_fraction=1.0) for sec in surf.sections]

    def _plot_polyline(points: list[np.ndarray], line_label: str | None) -> None:
        arr = np.asarray(points, dtype=np.float64)
        ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], color=color, linewidth=1.2, label=line_label)
        if surf.yduplicate is not None:
            mirrored = np.array([_mirror_point(pt, surf.yduplicate) for pt in points])
            ax.plot(mirrored[:, 0], mirrored[:, 1], mirrored[:, 2], color=color, linewidth=1.2)

    _plot_polyline(le_points, label)
    _plot_polyline(te_points, None)
    for le, te in zip(le_points, te_points):
        ax.plot([le[0], te[0]], [le[1], te[1]], [le[2], te[2]], color=color, linewidth=0.8, alpha=0.7)
        if surf.yduplicate is not None:
            le_m = _mirror_point(le, surf.yduplicate)
            te_m = _mirror_point(te, surf.yduplicate)
            ax.plot(
                [le_m[0], te_m[0]],
                [le_m[1], te_m[1]],
                [le_m[2], te_m[2]],
                color=color,
                linewidth=0.8,
                alpha=0.7,
            )


def _plot_bodies(ax: Axes, model: AVLModel, state: AVLState | None) -> None:
    """Draw fuselage centerlines and cross-section circles."""
    if state is not None and state.nbody > 0:
        for ibody in range(int(state.nbody)):
            i0 = int(state.lfrst[ibody])
            n_nodes = int(state.nl[ibody])
            if n_nodes < 2:
                continue
            xs = state.rl[0, i0 : i0 + n_nodes]
            ys = state.rl[1, i0 : i0 + n_nodes]
            zs = state.rl[2, i0 : i0 + n_nodes]
            ax.plot(xs, ys, zs, color=_BODY_COLOR, linewidth=1.5, label="Body" if ibody == 0 else None)
            for k in range(n_nodes):
                radius = float(state.radl[i0 + k])
                if radius <= 0.0:
                    continue
                theta = np.linspace(0.0, 2.0 * math.pi, 24)
                ax.plot(
                    np.full_like(theta, xs[k]),
                    ys[k] + radius * np.cos(theta),
                    zs[k] + radius * np.sin(theta),
                    color=_BODY_COLOR,
                    linewidth=0.5,
                    alpha=0.6,
                )
        return

    for ibody, body in enumerate(model.bodies):
        _plot_body_fallback(ax, body, label="Body" if ibody == 0 else None)


def _plot_body_fallback(ax: Axes, body: BodyDef, label: str | None) -> None:
    """Plot a body from thread coordinates when solver nodes are unavailable."""
    xs_raw: list[float] | None = None
    ys_raw: list[float] | None = None
    if body.body_thread_x and body.body_thread_y:
        xs_raw = list(body.body_thread_x)
        ys_raw = list(body.body_thread_y)
    elif body.body_coords:
        xs_raw = [pt[0] for pt in body.body_coords]
        ys_raw = [pt[1] for pt in body.body_coords]

    if not xs_raw or not ys_raw:
        return

    scale = np.asarray(body.scale if body.scale else [1.0, 1.0, 1.0], dtype=np.float64)
    translate = np.asarray(body.translate if body.translate else [0.0, 0.0, 0.0], dtype=np.float64)
    xs = scale[0] * np.asarray(xs_raw, dtype=np.float64) + translate[0]
    ys = np.full_like(xs, translate[1])
    zs = scale[2] * np.asarray(ys_raw, dtype=np.float64) + translate[2]
    ax.plot(xs, ys, zs, color=_BODY_COLOR, linewidth=1.5, label=label)


def _plot_control_surfaces(ax: Axes, model: AVLModel) -> None:
    """Highlight control-surface hinge lines and annotate control names."""
    labeled_controls: set[str] = set()
    for surf_name, ctrl_name, points in _collect_control_polylines(model):
        if points.ndim == 1:
            points = points.reshape(1, 3)
        ax.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=_CONTROL_COLOR,
            linewidth=2.5,
            linestyle="--",
            label="Control hinge" if "Control hinge" not in labeled_controls else None,
        )
        labeled_controls.add("Control hinge")
        mid = points[len(points) // 2]
        ax.scatter(
            [mid[0]],
            [mid[1]],
            [mid[2]],
            color=_CONTROL_COLOR,
            s=36,
            depthshade=False,
            zorder=5,
        )
        if ctrl_name not in labeled_controls:
            ax.text(
                mid[0],
                mid[1],
                mid[2],
                f" {ctrl_name}",
                color=_CONTROL_COLOR,
                fontsize=8,
                fontweight="bold",
            )
            labeled_controls.add(ctrl_name)


def _set_equal_aspect(ax: Axes, points: np.ndarray) -> None:
    """Set approximate equal scaling on a 3D axes."""
    if points.size == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    if radius <= 0.0:
        radius = 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def plot_aircraft_3d(
    source: str | Path | Aircraft | AVLModel | AVLSolver,
    *,
    ax: Axes | None = None,
    show: bool = True,
    block: bool | None = None,
    title: str | None = None,
    base_dir: str | Path | None = None,
    figsize: tuple[float, float] = (10.0, 8.0),
) -> tuple[Figure, Axes]:
    """Plot an AVL aircraft in 3D with per-surface colors and control callouts.

    Parameters
    ----------
    source:
        Geometry to visualize: path to an ``.avl`` file, an :class:`~openavl.geometry.Aircraft`,
        an :class:`~openavl.fileio.parser.AVLModel`, or an :class:`~openavl.core.solver.AVLSolver`.
    ax:
        Optional existing 3D matplotlib axes. When omitted, a new figure is created.
    show:
        Call ``matplotlib.pyplot.show()`` before returning.
    block:
        Forwarded to ``show(block=...)``. Defaults to ``show``.
    title:
        Figure title. Defaults to the aircraft name from the model header.
    base_dir:
        Directory for resolving relative airfoil/body paths when ``source`` is an
        :class:`~openavl.geometry.Aircraft`.
    figsize:
        Figure size when creating a new figure.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib figure and 3D axes containing the geometry plot.
    """
    plt = _import_matplotlib()

    model, state = _resolve_geometry(source, base_dir)
    if state is None:
        state = _build_state(model)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    labels = _surface_labels(model, int(state.nsurf))
    for isurf in range(int(state.nsurf)):
        comp = int(state.lncomp[isurf]) if isurf < state.lncomp.size else isurf + 1
        color = _surface_color(labels[isurf], comp, isurf)
        _plot_lattice_surface(ax, state, isurf, color, labels[isurf])

    if int(state.nvor) == 0:
        for index, surf in enumerate(model.surfaces):
            color = _surface_color(surf.name, surf.component or (index + 1), index)
            _plot_wireframe_surface(ax, surf, color, surf.name)

    _plot_bodies(ax, model, state)
    _plot_control_surfaces(ax, model)

    header = model.header
    ref = np.array([header.xref, header.yref, header.zref], dtype=np.float64)
    ax.scatter(
        [ref[0]],
        [ref[1]],
        [ref[2]],
        color=_REF_POINT_COLOR,
        s=40,
        marker="x",
        linewidths=2.0,
        label="Moment ref.",
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title or model.header.title or "AVL aircraft")
    ax.view_init(elev=20.0, azim=-60.0)
    ax.legend(loc="upper left", fontsize=8)

    sample_points = np.column_stack(
        (
            state.rv1[0, : int(state.nvor)],
            state.rv1[1, : int(state.nvor)],
            state.rv1[2, : int(state.nvor)],
        )
    )
    if sample_points.size:
        _set_equal_aspect(ax, sample_points)

    fig.tight_layout()
    if show:
        plt.show(block=block if block is not None else show)
    return fig, ax
