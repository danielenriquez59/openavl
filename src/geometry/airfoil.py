"""Airfoil definitions for the Geometry API."""

from __future__ import annotations

from enum import Enum


class AirfoilType(Enum):
    """How an airfoil section is specified."""

    NACA = "naca"
    FILE = "file"
    COORDS = "coords"


class Airfoil:
    """Airfoil specification for a wing section."""

    def __init__(
        self,
        af_type: AirfoilType,
        *,
        naca: str | None = None,
        file_path: str | None = None,
        coords: list[list[float]] | None = None,
    ) -> None:
        self.af_type = af_type
        self.naca = naca
        self.file_path = file_path
        self.coords = coords
