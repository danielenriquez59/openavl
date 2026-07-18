"""Visualization and pressure-coefficient data adapters."""

from __future__ import annotations

from typing import Any


def plot_aircraft(self, **kwargs: Any) -> Any:
    """Plot the aircraft geometry in 3D.

    Delegates to :func:`openavl.plotting.plot_aircraft_3d`. Does not
    require a prior solve; uses the built lattice when available.

    Parameters
    ----------
    **kwargs
        Forwarded to :func:`openavl.plotting.plot_aircraft_3d` (for example
        ``show=False`` or ``title="My aircraft"``).

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and 3D axes.
    """
    from openavl.plotting.aircraft3d import plot_aircraft_3d

    return plot_aircraft_3d(self, **kwargs)

def plot_geom(self, **kwargs: Any) -> Any:
    """Alias for :meth:`plot_aircraft`."""
    return self.plot_aircraft(**kwargs)

def plot_lift_distribution(self, **kwargs: Any) -> Any:
    """Plot spanwise lift distribution from the latest solve.

    Delegates to :func:`openavl.plotting.plot_lift_distribution`.
    Requires a completed :meth:`execute_run`.

    Parameters
    ----------
    **kwargs
        Forwarded to :func:`openavl.plotting.plot_lift_distribution`
        (for example ``quantity="cnc"``, ``component=1``, or ``show=False``).

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and 2D axes.
    """
    from openavl.plotting.lift_distribution import plot_lift_distribution

    return plot_lift_distribution(self, **kwargs)

def get_cp_data(
    self,
    *,
    component: int | None = None,
    load_only: bool = True,
    mode: str = "surface",
) -> list[dict[str, object]]:
    """Return structured surface meshes and Cp samples from the latest solve.

    Parameters
    ----------
    component:
        Restrict to one AVL component index.
    load_only:
        Omit surfaces flagged with ``noload``.
    mode:
        ``"surface"`` for absolute CPOML Cp, ``"delta"`` for raw loading.

    Returns
    -------
    list[dict]
        Each entry contains ``label``, ``isurf``, ``xyz``, and ``cp`` arrays.
    """
    from openavl.plotting.cp_plot import collect_cp_surfaces

    surfaces = collect_cp_surfaces(
        self.state,
        self.model,
        component=component,
        load_only=load_only,
        mode=mode,
    )
    return [
        {
            "label": item.label,
            "isurf": item.isurf,
            "xyz": item.xyz,
            "cp": item.cp,
        }
        for item in surfaces
    ]

def plot_cp(self, **kwargs: Any) -> Any:
    """Plot the solved pressure-coefficient distribution on lifting surfaces.

    Delegates to :func:`openavl.plotting.plot_cp`. Requires a completed
    :meth:`execute_run`. By default plots absolute surface Cp via CPOML;
    pass ``mode="delta"`` for raw vortex-lattice loading.

    Parameters
    ----------
    **kwargs
        Forwarded to :func:`openavl.plotting.plot_cp` (for example
        ``show=False``, ``component=1``, ``mode="delta"``, or
        ``load_only=False``).

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and 3D axes when ``show=False``; otherwise
        returns after displaying the plot window.
    """
    from openavl.plotting.cp_plot import plot_cp

    return plot_cp(self, **kwargs)
