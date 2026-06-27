"""Control surface definitions for the Geometry API."""

from __future__ import annotations


class ControlSurface:
    """Hinge-line control surface attached to a :class:`Section`.

    Normally created via :meth:`Section.add_control`. The ``name`` becomes a
    trim variable in :class:`~openavl.core.solver.AVLSolver`.
    """

    def __init__(
        self,
        name: str,
        gain: float = 1.0,
        xhinge: float = 0.75,
        vhinge: list[float] | None = None,
        sgn_dup: float = 1.0,
    ) -> None:
        self.name = name
        self.gain = float(gain)
        self.xhinge = float(xhinge)
        self.vhinge = list(vhinge) if vhinge is not None else [0.0, 0.0, 0.0]
        self.sgn_dup = float(sgn_dup)
