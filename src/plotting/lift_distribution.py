"""Spanwise lift distribution plots from a solved AVL state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from openavl.plotting.aircraft3d import (
    _import_matplotlib,
    _resolve_geometry,
    _surface_color,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from openavl.core.solver import AVLSolver
    from openavl.core.state import AVLState
    from openavl.fileio.parser import AVLModel
    from openavl.geometry.aircraft import Aircraft

from openavl.geom.geometry import solver_surface_index, solver_surface_name

Quantity = Literal["cl", "cnc"]


@dataclass(frozen=True)
class LiftDistributionSeries:
    """Spanwise samples for one solver surface."""

    label: str
    isurf: int
    y: np.ndarray
    cl: np.ndarray
    cnc: np.ndarray
    clmax: float


def collect_lift_distribution(
    state: AVLState,
    model: AVLModel,
    *,
    component: int | None = None,
    load_only: bool = True,
) -> list[LiftDistributionSeries]:
    """Collect stripwise lift data from a solved solver state.

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

    Returns
    -------
    list[LiftDistributionSeries]
        One entry per included solver surface, strips sorted by spanwise ``y``.
    """
    if not bool(getattr(state, "lsol", False)):
        raise ValueError(
            "Lift distribution requires a solved state; run execute_run() first."
        )

    series: list[LiftDistributionSeries] = []
    for isurf in range(int(state.nsurf)):
        if load_only and not bool(state.lfload[isurf]):
            continue
        if component is not None and int(state.lncomp[isurf]) != component:
            continue

        j0 = int(state.jfrst[isurf])
        nj = int(state.nj[isurf])
        if nj <= 0:
            continue

        js = np.arange(j0, j0 + nj, dtype=np.intp)
        y = state.rle[1, js].astype(np.float64, copy=True)
        order = np.argsort(y)
        series.append(
            LiftDistributionSeries(
                label=solver_surface_name(model, isurf),
                isurf=isurf,
                y=y[order],
                cl=state.cl_lstrp[js][order],
                cnc=state.cnc[js][order],
                clmax=float(state.clmax_surf[isurf]),
            )
        )
    return series


def plot_lift_distribution(
    source: str | Path | Aircraft | AVLModel | AVLSolver,
    *,
    quantity: Quantity = "cl",
    component: int | None = None,
    load_only: bool = True,
    ax: Axes | None = None,
    show: bool = True,
    block: bool | None = None,
    title: str | None = None,
    base_dir: str | Path | None = None,
    figsize: tuple[float, float] = (10.0, 5.0),
) -> tuple[Figure, Axes]:
    """Plot spanwise lift distribution from a solved case.

    Parameters
    ----------
    source:
        An :class:`~openavl.core.solver.AVLSolver` after ``execute_run``, or
        geometry-only input (which cannot produce lift data).
    quantity:
        ``"cl"`` plots local lift coefficient ``cl_lstrp``; ``"cnc"`` plots
        section normal-force loading ``cnc``.
    component:
        Restrict to one AVL component index (for example ``1`` for the Supra wing).
    load_only:
        Omit surfaces with ``noload`` set.
    ax:
        Optional existing matplotlib axes.
    show:
        Call ``matplotlib.pyplot.show()`` before returning.
    block:
        Forwarded to ``show(block=...)``. Defaults to ``show``.
    title:
        Plot title. Defaults to the aircraft name from the model header.
    base_dir:
        Directory for resolving relative paths when ``source`` is an
        :class:`~openavl.geometry.Aircraft`.
    figsize:
        Figure size when creating a new figure.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib figure and axes containing the lift distribution.
    """
    from openavl.core.solver import AVLSolver

    plt = _import_matplotlib()
    model, state = _resolve_geometry(source, base_dir)
    if not isinstance(source, AVLSolver):
        raise ValueError(
            "plot_lift_distribution requires an AVLSolver with a completed run."
        )
    if state is None:
        raise ValueError(
            "plot_lift_distribution requires an AVLSolver with a completed run."
        )

    series = collect_lift_distribution(
        state,
        model,
        component=component,
        load_only=load_only,
    )
    if not series:
        raise ValueError("No load-bearing surfaces matched the plot filters.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    y_label = "Local Cl" if quantity == "cl" else "cnc"
    clmax_drawn: set[tuple[str, float]] = set()

    for item in series:
        comp = int(state.lncomp[item.isurf]) if item.isurf < state.lncomp.size else item.isurf + 1
        color = _surface_color(item.label, comp, item.isurf)
        values = item.cl if quantity == "cl" else item.cnc
        ax.plot(item.y, values, marker="o", markersize=3.5, linewidth=1.4, color=color, label=item.label)

        if quantity == "cl" and item.clmax > 0.0:
            model_idx = solver_surface_index(model, item.isurf)
            base_name = (
                model.surfaces[model_idx].name
                if 0 <= model_idx < len(model.surfaces)
                else item.label.replace(" (mirror)", "")
            )
            key = (base_name, item.clmax)
            if key not in clmax_drawn:
                ax.axhline(
                    item.clmax,
                    color=color,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.7,
                    label=f"{base_name} Clmax",
                )
                clmax_drawn.add(key)

    ax.set_xlabel("Spanwise position Y")
    ax.set_ylabel(y_label)
    ax.set_title(title or f"{model.header.title or 'AVL aircraft'} — lift distribution")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    if show:
        plt.show(block=block if block is not None else show)
    return fig, ax
