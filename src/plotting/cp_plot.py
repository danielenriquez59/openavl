"""Pressure-coefficient visualization on solved vortex-lattice surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from openavl.plotting.aircraft3d import (
    _import_matplotlib,
    _resolve_geometry,
    _set_equal_aspect,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from openavl.core.solver import AVLSolver
    from openavl.core.state import AVLState
    from openavl.fileio.parser import AVLModel
    from openavl.geometry.aircraft import Aircraft

from openavl.geom.geometry import solver_surface_name

from openavl.aero.cpoml import collect_cpoml_surfaces


@dataclass(frozen=True)
class CpSurfaceData:
    """Structured mesh and Cp samples for one solver surface."""

    label: str
    isurf: int
    xyz: np.ndarray
    cp: np.ndarray


def _strip_edge_boundaries(
    state: AVLState,
    i0: int,
    nvc: int,
    rv_side: np.ndarray,
    chord_strip: float,
) -> np.ndarray:
    """Return panel-edge positions along one spanwise strip edge.

    Computes ``(nvc + 1, 3)`` boundary vertices from bound-vortex leg positions
    on ``rv_side``. Panel leading edges sit one quarter panel-width upstream of
    each vortex; the trailing edge of the last panel sits three quarters
    panel-width downstream of the last vortex.
    """
    ivs = np.arange(i0, i0 + nvc, dtype=np.intp)
    out = np.zeros((nvc + 1, 3), dtype=np.float64)
    dxoc_scaled = state.dxv[ivs] * chord_strip / state.chordv[ivs]
    out[:nvc, 0] = rv_side[0, ivs] - 0.25 * dxoc_scaled
    out[nvc, 0] = rv_side[0, ivs[-1]] + 0.75 * dxoc_scaled[-1]
    out[:, 1] = rv_side[1, i0]
    out[:, 2] = rv_side[2, i0]
    return out


def collect_cp_surfaces(
    state: AVLState,
    model: AVLModel,
    *,
    component: int | None = None,
    load_only: bool = True,
    mode: str = "surface",
) -> list[CpSurfaceData]:
    """Collect structured surface meshes and Cp values from a solved state.

    Parameters
    ----------
    state:
        Solver state after force integration (``execute_run`` or equivalent).
    model:
        Aircraft model paired with ``state``.
    component:
        If set, include only surfaces with this AVL component index.
    load_only:
        Skip surfaces flagged with ``noload``.
    mode:
        ``"surface"`` returns absolute upper/lower surface Cp via CPOML.
        ``"delta"`` returns raw vortex-lattice ``Delta Cp`` on the camber mesh.

    Returns
    -------
    list[CpSurfaceData]
        One entry per included surface. For ``mode="surface"``, ``xyz`` has
        shape ``(n_span + 1, 2*n_chord + 1, 3)`` and ``cp`` has shape
        ``(n_span, 2*n_chord)``. For ``mode="delta"``, shapes are
        ``(n_span + 1, n_chord + 1, 3)`` and ``(n_span, n_chord)``.
    """
    if mode == "surface":
        oml = collect_cpoml_surfaces(
            state,
            model,
            component=component,
            load_only=load_only,
        )
        return [
            CpSurfaceData(label=item.label, isurf=item.isurf, xyz=item.xyz, cp=item.cp)
            for item in oml
        ]

    if mode != "delta":
        raise ValueError(f"Unknown Cp mode '{mode}'; expected 'surface' or 'delta'.")
    if not bool(getattr(state, "lsol", False)):
        raise ValueError(
            "Cp plotting requires a solved state; run execute_run() first."
        )

    surfaces: list[CpSurfaceData] = []
    for isurf in range(int(state.nsurf)):
        if load_only and not bool(state.lfload[isurf]):
            continue
        if component is not None and int(state.lncomp[isurf]) != component:
            continue

        j0 = int(state.jfrst[isurf])
        nj = int(state.nj[isurf])
        if nj <= 0:
            continue

        nvc = int(state.nvstrp[j0])
        if nvc <= 0:
            continue

        for jj in range(1, nj):
            if int(state.nvstrp[j0 + jj]) != nvc:
                nvc = 0
                break
        if nvc <= 0:
            continue

        imags = int(state.imags[isurf])
        xyz = np.zeros((nj + 1, nvc + 1, 3), dtype=np.float64)
        cp = np.zeros((nj, nvc), dtype=np.float64)

        if imags > 0:
            for jj in range(nj):
                j = j0 + jj
                i0 = int(state.ijfrst[j])
                xyz[jj] = _strip_edge_boundaries(
                    state,
                    i0,
                    nvc,
                    state.rv1,
                    float(state.chord1[j]),
                )
            j = j0 + nj - 1
            i0 = int(state.ijfrst[j])
            xyz[nj] = _strip_edge_boundaries(
                state,
                i0,
                nvc,
                state.rv2,
                float(state.chord2[j]),
            )
        else:
            j = j0
            i0 = int(state.ijfrst[j])
            xyz[0] = _strip_edge_boundaries(
                state,
                i0,
                nvc,
                state.rv2,
                float(state.chord2[j]),
            )
            for jj in range(1, nj + 1):
                j = j0 + jj - 1
                i0 = int(state.ijfrst[j])
                xyz[jj] = _strip_edge_boundaries(
                    state,
                    i0,
                    nvc,
                    state.rv1,
                    float(state.chord1[j]),
                )

        for jj in range(nj):
            j = j0 + jj
            i0 = int(state.ijfrst[j])
            cp[jj, :] = state.dcp[i0 : i0 + nvc]

        surfaces.append(
            CpSurfaceData(
                label=solver_surface_name(model, isurf),
                isurf=isurf,
                xyz=xyz,
                cp=cp,
            )
        )
    return surfaces


def plot_cp(
    source: str | Path | Aircraft | AVLModel | AVLSolver,
    *,
    component: int | None = None,
    load_only: bool = True,
    mode: str = "surface",
    ax: Axes | None = None,
    show: bool = True,
    block: bool | None = None,
    title: str | None = None,
    base_dir: str | Path | None = None,
    figsize: tuple[float, float] = (10.0, 8.0),
) -> tuple[Figure, Axes]:
    """Plot solved pressure coefficient on lifting surfaces in 3D.

    Parameters
    ----------
    source:
        An :class:`~openavl.core.solver.AVLSolver` after ``execute_run``, or
        geometry-only input (which cannot produce Cp data).
    component:
        Restrict to one AVL component index.
    load_only:
        Omit surfaces with ``noload`` set.
    mode:
        ``"surface"`` plots absolute Cp on upper/lower airfoil surfaces via
        CPOML. ``"delta"`` plots raw vortex-lattice ``Delta Cp`` loading.
    ax:
        Optional existing 3D matplotlib axes. When omitted, a new figure is created.
    show:
        Call ``matplotlib.pyplot.show()`` before returning.
    block:
        Forwarded to ``show(block=...)``. Defaults to ``show``.
    title:
        Figure title. Defaults to the aircraft name from the model header.
    base_dir:
        Directory for resolving relative paths when ``source`` is an
        :class:`~openavl.geometry.Aircraft`.
    figsize:
        Figure size when creating a new figure.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib figure and 3D axes containing the Cp plot.
    """
    from openavl.core.solver import AVLSolver

    plt = _import_matplotlib()
    from matplotlib import cm

    model, state = _resolve_geometry(source, base_dir)
    if not isinstance(source, AVLSolver) or state is None:
        raise ValueError(
            "plot_cp requires an AVLSolver with a completed run."
        )

    surfaces = collect_cp_surfaces(
        state,
        model,
        component=component,
        load_only=load_only,
        mode=mode,
    )
    if not surfaces:
        raise ValueError("No load-bearing surfaces matched the plot filters.")

    cp_min = min(float(np.min(item.cp)) for item in surfaces)
    cp_max = max(float(np.max(item.cp)) for item in surfaces)
    cp_amax = max(abs(cp_min), abs(cp_max), 1.0e-12)
    norm = plt.Normalize(vmin=-cp_amax, vmax=cp_amax)
    cmap = cm.bwr
    mappable = cm.ScalarMappable(cmap=cmap, norm=norm)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    all_points: list[np.ndarray] = []
    for item in surfaces:
        face_rgba = cmap(norm(item.cp))
        ax.plot_surface(
            item.xyz[:, :, 0],
            item.xyz[:, :, 1],
            item.xyz[:, :, 2],
            facecolors=face_rgba,
            rstride=1,
            cstride=1,
            shade=False,
            antialiased=False,
        )
        all_points.append(item.xyz.reshape(-1, 3))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(
        title
        or f"{model.header.title or 'AVL aircraft'} — {'Cp' if mode == 'surface' else 'Delta Cp'}"
    )
    ax.view_init(elev=20.0, azim=-60.0)
    ax.axis("off")
    ax.grid(False)

    if all_points:
        _set_equal_aspect(ax, np.vstack(all_points))

    fig.subplots_adjust(left=0.025, right=0.925, top=0.925, bottom=0.025)
    colorbar = fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.02)
    colorbar.set_label("Cp" if mode == "surface" else "Delta Cp", rotation=0, labelpad=20)

    if show:
        plt.show(block=block if block is not None else show)
    return fig, ax
