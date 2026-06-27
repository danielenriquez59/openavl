"""Fuselage/nacelle body definitions for the Geometry API."""

from __future__ import annotations


class Body:
    """Slender body (fuselage or nacelle) attached to an :class:`Aircraft`.

    Normally created via :meth:`Aircraft.add_body`. Provide geometry through
    ``body_file`` or inline ``body_coords``.
    """

    def __init__(
        self,
        name: str,
        n_body: int = 0,
        b_space: float = 1.0,
        *,
        body_file: str | None = None,
        body_coords: list[list[float]] | None = None,
        translate: list[float] | None = None,
        scale: list[float] | None = None,
        yduplicate: float | None = None,
    ) -> None:
        self.name = name
        self.n_body = int(n_body)
        self.b_space = float(b_space)
        self.body_file = body_file
        self.body_coords = [list(pt) for pt in body_coords] if body_coords is not None else None
        self.translate = list(translate) if translate is not None else [0.0, 0.0, 0.0]
        self.scale = list(scale) if scale is not None else [1.0, 1.0, 1.0]
        self.yduplicate = float(yduplicate) if yduplicate is not None else None
