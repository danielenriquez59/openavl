"""Top-level aircraft container for the Geometry API."""

from __future__ import annotations

from openavl.fileio.parser import AVLModel
from openavl.geometry.body import Body
from openavl.geometry.wing import Wing


class Aircraft:
    """Programmatic aircraft geometry definition.

    Build an aircraft by adding wings (lifting surfaces) and bodies (fuselages),
    then pass the instance directly to :class:`~openavl.core.solver.AVLSolver` or
    convert it with :meth:`to_avl_model`.

    Example
    -------
    >>> wing = aircraft.add_wing("Wing", n_chord=8, n_span=20, symmetric=True)
    >>> root = wing.add_section(xyzle=[0.0, 0.0, 0.0], chord=1.0)
    >>> root.set_airfoil_naca("2412").add_control("aileron", xhinge=0.75, sgn_dup=-1.0)
    """

    def __init__(
        self,
        name: str = "",
        sref: float = 1.0,
        cref: float = 1.0,
        bref: float = 1.0,
        xref: float = 0.0,
        yref: float = 0.0,
        zref: float = 0.0,
        mach: float = 0.0,
        iysym: int = 0,
        izsym: int = 0,
        zsym: float = 0.0,
    ) -> None:
        """Create an aircraft with reference dimensions and symmetry flags.

        Parameters
        ----------
        name:
            Aircraft title (first line of an equivalent ``.avl`` file).
        sref, cref, bref:
            Reference area, chord, and span for force/moment coefficients.
        xref, yref, zref:
            Moment-reference point in geometry length units.
        mach:
            Freestream Mach number (typically overridden at run time).
        iysym, izsym, zsym:
            Symmetry flags matching AVL header conventions (``0`` = no symmetry).
        """
        self.name = name
        self.sref = float(sref)
        self.cref = float(cref)
        self.bref = float(bref)
        self.xref = float(xref)
        self.yref = float(yref)
        self.zref = float(zref)
        self.mach = float(mach)
        self.iysym = int(iysym)
        self.izsym = int(izsym)
        self.zsym = float(zsym)
        self.wings: list[Wing] = []
        self.bodies: list[Body] = []

    def add_wing(
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
    ) -> Wing:
        """Add a lifting surface (wing, horizontal tail, vertical fin, etc.).

        Creates a :class:`Wing`, appends it to :attr:`wings`, and returns it so
        you can immediately call :meth:`Wing.add_section`.

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
            ``yduplicate=0.0``). Use for most wings and horizontal tails.
        yduplicate:
            Explicit Y-duplicate coordinate for mirroring. Ignored when
            ``symmetric=True``. Leave ``None`` for surfaces that are not
            mirrored (e.g. a single vertical fin).
        angle:
            Surface incidence bias in degrees, applied uniformly to all
            sections (AVL ``ANGLE`` keyword).
        translate:
            ``[x, y, z]`` translation applied before section coordinates
            (geometry length units).
        scale:
            ``[x, y, z]`` scale factors applied to the surface. The z-scale
            can encode dihedral when combined with section ``zle`` values.
        nowake:
            Exclude this surface from the wake model.
        noload:
            Exclude this surface from force integration.
        component:
            Component index for multi-body configurations (AVL ``INDEX``).

        Returns
        -------
        Wing
            The newly created surface; call ``add_section`` on it next.

        See Also
        --------
        add_body : Add a fuselage or nacelle body.
        """
        wing = Wing(
            name,
            n_chord=n_chord,
            c_space=c_space,
            n_span=n_span,
            s_space=s_space,
            symmetric=symmetric,
            yduplicate=yduplicate,
            angle=angle,
            translate=translate,
            scale=scale,
            nowake=nowake,
            noload=noload,
            component=component,
        )
        self.wings.append(wing)
        return wing

    def add_body(
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
    ) -> Body:
        """Add a slender body (fuselage or nacelle).

        Creates a :class:`Body`, appends it to :attr:`bodies`, and returns it.
        Specify geometry via ``body_file`` (path to a ``.dat`` coordinate file)
        or inline ``body_coords``.

        Parameters
        ----------
        name:
            Body label (appears in AVL output).
        n_body:
            Number of body circumferential panels.
        b_space:
            Body panel spacing parameter (``1.0`` = uniform along the axis).
        body_file:
            Path to a body coordinate file, relative to ``base_dir`` when
            converting with :meth:`to_avl_model` or passing to
            :class:`~openavl.core.solver.AVLSolver`.
        body_coords:
            Inline body profile coordinates ``[[x, y], ...]`` as an alternative
            to ``body_file``.
        translate:
            ``[x, y, z]`` translation applied to the body (geometry length units).
        scale:
            ``[x, y, z]`` scale factors applied to the body.
        yduplicate:
            Y-coordinate for mirroring the body about the x-z plane. ``None``
            means no mirroring.

        Returns
        -------
        Body
            The newly created body.

        See Also
        --------
        add_wing : Add a lifting surface.
        """
        body = Body(
            name,
            n_body=n_body,
            b_space=b_space,
            body_file=body_file,
            body_coords=body_coords,
            translate=translate,
            scale=scale,
            yduplicate=yduplicate,
        )
        self.bodies.append(body)
        return body

    def to_avl_model(self, base_dir: str | None = None) -> AVLModel:
        """Convert this aircraft to a solver-ready :class:`~openavl.fileio.parser.AVLModel`.

        Builds the internal dataclass representation, validates geometry, and
        runs :func:`~openavl.fileio.parser.prepare_model` to resolve airfoil
        camber slopes and body threads.

        Parameters
        ----------
        base_dir:
            Directory used to resolve relative ``body_file`` and airfoil file
            paths set on sections. Required when those paths are not absolute.
            Pass the same value as ``base_dir`` to
            :class:`~openavl.core.solver.AVLSolver`.

        Returns
        -------
        AVLModel
            Prepared model ready for the vortex-lattice solver.

        Raises
        ------
        ValueError
            If reference dimensions are non-positive, a wing has fewer than two
            sections, or any section chord is non-positive.
        """
        from openavl.geometry._convert import to_avl_model

        return to_avl_model(self, base_dir=base_dir)
