"""Lifting surface definitions for the Geometry API."""

from __future__ import annotations

from openavl.geometry.cdcl_polar import CdclPolar
from openavl.geometry.section import Section


class Wing:
    """Lifting surface attached to an :class:`Aircraft`.

    Normally created via :meth:`Aircraft.add_wing`, not instantiated directly.
    Define the spanwise geometry by adding at least two :class:`Section`
    stations in order from root to tip.

    Set :attr:`clmax` to a positive value to cap sectional lift coefficients
    during force integration and approximate stall onset (OpenAVL extension).
    The default ``0.0`` leaves sectional lift uncapped. This option is available
    only through the Geometry API, not legacy ``.avl`` file input.
    """

    def __init__(
        self,
        name: str,
        n_chord: int = 0,
        c_space: float = 1.0,
        n_span: int = 0,
        s_space: float = 1.0,
        *,
        symmetric: bool = False,
        yduplicate: float | None = None,
        angle: float = 0.0,
        translate: list[float] | None = None,
        scale: list[float] | None = None,
        nowake: bool = False,
        noload: bool = False,
        component: int = 0,
        clmax: float = 0.0,
    ) -> None:
        """Configure a lifting surface.

        Parameters
        ----------
        name:
            Surface label (appears in AVL output).
        n_chord:
            Number of chordwise vortex panels per section strip.
        c_space:
            Chordwise panel spacing parameter (``1.0`` = uniform).
        n_span:
            Default number of spanwise panels between section stations.
        s_space:
            Spanwise panel spacing parameter. Values ``< 0`` use cosine
            clustering toward the ends; ``1.0`` is uniform.
        symmetric:
            If ``True``, mirror the surface about ``y = 0`` (equivalent to
            ``yduplicate=0.0``).
        yduplicate:
            Explicit Y-duplicate coordinate for mirroring. Ignored when
            ``symmetric=True``. ``None`` means no mirroring.
        angle:
            Surface incidence bias in degrees, applied uniformly to all
            sections (AVL ``ANGLE`` keyword).
        translate:
            ``[x, y, z]`` translation applied before section coordinates.
        scale:
            ``[x, y, z]`` scale factors applied to the surface.
        nowake:
            Exclude this surface from the wake model.
        noload:
            Exclude this surface from force integration.
        component:
            Component index for multi-body configurations (AVL ``INDEX``).
        clmax:
            Maximum local lift coefficient for this surface. When ``> 0``,
            strip forces whose local ``cl_lstrp`` exceeds this value are scaled
            down before surface integration, preserving force direction.
            ``0.0`` disables capping (default).
        """
        self.name = name
        self.n_chord = int(n_chord)
        self.c_space = float(c_space)
        self.n_span = int(n_span)
        self.s_space = float(s_space)
        if symmetric:
            self.yduplicate: float | None = 0.0
        elif yduplicate is not None:
            self.yduplicate = float(yduplicate)
        else:
            self.yduplicate = None
        self.angle = float(angle)
        self.translate = list(translate) if translate is not None else [0.0, 0.0, 0.0]
        self.scale = list(scale) if scale is not None else [1.0, 1.0, 1.0]
        self.nowake = bool(nowake)
        self.noload = bool(noload)
        self.component = int(component)
        self.clmax = float(clmax)
        self.sections: list[Section] = []
        self.cdcl: CdclPolar | None = None

    def set_cdcl_polar(self, polar: CdclPolar) -> Wing:
        """Assign a default viscous drag CD(CL) polar for all sections on this wing.

        Individual sections may override the wing default via
        :meth:`Section.set_cdcl_polar`.

        Parameters
        ----------
        polar:
            Three-point drag polar; see :class:`~openavl.geometry.CdclPolar`.

        Returns
        -------
        Wing
            ``self``, for method chaining.
        """
        self.cdcl = polar
        return self

    def add_section(
        self,
        *,
        xyzle: list[float] | None = None,
        chord: float = 1.0,
        ainc: float = 0.0,
        n_span: int = 0,
        s_space: float = 1.0,
        claf: float = 1.0,
    ) -> Section:
        """Add a spanwise section station and return it.

        Sections are ordered root-to-tip (or inboard-to-outboard). Each section
        defines the leading-edge position, chord, and incidence at a span
        station; panels are interpolated to the next section.

        Parameters
        ----------
        xyzle:
            Leading-edge coordinates ``[x, y, z]`` in geometry length units,
            relative to the surface :attr:`translate` (not yet applied at build
            time). Defaults to ``[0, 0, 0]``.
        chord:
            Section chord length (must be ``> 0``).
        ainc:
            Section incidence angle in degrees (twist relative to the surface
            :attr:`angle` bias).
        n_span:
            Number of spanwise panels from this section to the next. Overrides
            the wing-level default for this span segment only.
        s_space:
            Spanwise panel spacing from this section to the next. Overrides the
            wing-level default for this span segment only.
        claf:
            Fraction of chord over which the airfoil shape is applied (AVL
            ``AFIL`` first parameter; ``1.0`` = full chord).

        Returns
        -------
        Section
            The new section; chain airfoil and control setup on it::

                section.set_airfoil_naca("2412").add_control("aileron", ...)

        See Also
        --------
        Section.set_airfoil_naca, Section.set_airfoil_file,
        Section.set_cdcl_polar, Section.add_control
        """
        section = Section(
            xyzle=xyzle,
            chord=chord,
            ainc=ainc,
            n_span=n_span,
            s_space=s_space,
            claf=claf,
        )
        self.sections.append(section)
        return section
