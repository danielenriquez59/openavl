"""Wing section definitions for the Geometry API."""

from __future__ import annotations

from openavl.geometry.airfoil import Airfoil, AirfoilType
from openavl.geometry.cdcl_polar import CdclPolar
from openavl.geometry.control import ControlSurface


class Section:
    """Cross-section station along a :class:`Wing`.

    Holds leading-edge geometry via :attr:`xyzle`, optional :class:`Airfoil`
    shape, and attached :class:`ControlSurface` definitions. Created via
    :meth:`Wing.add_section`.
    """

    def __init__(
        self,
        *,
        xyzle: list[float] | None = None,
        chord: float = 1.0,
        ainc: float = 0.0,
        n_span: int = 0,
        s_space: float = 1.0,
        claf: float = 1.0,
    ) -> None:
        if xyzle is None:
            xyzle = [0.0, 0.0, 0.0]
        self.xyzle = [float(v) for v in xyzle[:3]]
        while len(self.xyzle) < 3:
            self.xyzle.append(0.0)
        self.chord = float(chord)
        self.ainc = float(ainc)
        self.n_span = int(n_span)
        self.s_space = float(s_space)
        self.claf = float(claf)
        self.airfoil: Airfoil | None = None
        self.controls: list[ControlSurface] = []
        self.cdcl: CdclPolar | None = None

    @property
    def xle(self) -> float:
        """Leading-edge x coordinate (geometry length units)."""
        return self.xyzle[0]

    @property
    def yle(self) -> float:
        """Leading-edge y coordinate (geometry length units)."""
        return self.xyzle[1]

    @property
    def zle(self) -> float:
        """Leading-edge z coordinate (geometry length units)."""
        return self.xyzle[2]

    def set_airfoil_naca(self, code: str) -> Section:
        """Assign a NACA 4-digit airfoil by code.

        Parameters
        ----------
        code:
            NACA 4-digit designation, e.g. ``"2412"`` or ``"0012"``.

        Returns
        -------
        Section
            ``self``, for method chaining.
        """
        self.airfoil = Airfoil(AirfoilType.NACA, naca=str(code).strip())
        return self

    def set_airfoil_file(self, path: str) -> Section:
        """Assign an airfoil from a coordinate file.

        Parameters
        ----------
        path:
            Path to an AVL-format airfoil ``.dat`` file. Relative paths are
            resolved against ``base_dir`` when converting with
            :meth:`~openavl.geometry.Aircraft.to_avl_model` or loading via
            :class:`~openavl.core.solver.AVLSolver`.

        Returns
        -------
        Section
            ``self``, for method chaining.
        """
        self.airfoil = Airfoil(AirfoilType.FILE, file_path=str(path).strip())
        return self

    def set_airfoil_coords(self, coords: list[list[float]]) -> Section:
        """Assign an airfoil from inline coordinates.

        Parameters
        ----------
        coords:
            Airfoil profile points ``[[x, y], ...]`` in AVL convention
            (chord fraction ``x``, thickness ``y``).

        Returns
        -------
        Section
            ``self``, for method chaining.
        """
        self.airfoil = Airfoil(AirfoilType.COORDS, coords=[list(pt) for pt in coords])
        return self

    def set_cdcl_polar(self, polar: CdclPolar) -> Section:
        """Assign a viscous drag CD(CL) polar to this section.

        Parameters
        ----------
        polar:
            Three-point drag polar; see :class:`~openavl.geometry.CdclPolar`.

        Returns
        -------
        Section
            ``self``, for method chaining.
        """
        self.cdcl = polar
        return self

    def add_control(
        self,
        name: str,
        gain: float = 1.0,
        xhinge: float = 0.75,
        vhinge: list[float] | None = None,
        sgn_dup: float = 1.0,
    ) -> ControlSurface:
        """Attach a hinge-line control surface to this section.

        Parameters
        ----------
        name:
            Control name used by the solver (e.g. ``"aileron"``, ``"elevator"``,
            ``"rudder"``). Must be unique across the aircraft.
        gain:
            Control effectiveness multiplier (AVL ``gain``).
        xhinge:
            Hinge location as a fraction of local chord measured from the
            leading edge (``0.75`` = 75% chord).
        vhinge:
            Hinge axis direction vector ``[x, y, z]`` in section coordinates.
            Defaults to ``[0, 0, 0]`` (spanwise hinge for typical ailerons).
        sgn_dup:
            Sign multiplier applied on the mirrored (Y-duplicated) half of a
            symmetric surface. Use ``-1.0`` for ailerons so deflections oppose.

        Returns
        -------
        ControlSurface
            The newly created control; also appended to :attr:`controls`.
        """
        ctrl = ControlSurface(name, gain=gain, xhinge=xhinge, vhinge=vhinge, sgn_dup=sgn_dup)
        self.controls.append(ctrl)
        return ctrl
