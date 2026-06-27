"""High-level AVL solver API."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from openavl import constants as C

if TYPE_CHECKING:
    from openavl.analysis.amode import EigenAnalysisResult
    from openavl.analysis.deriv import StabilityDerivatives
from openavl.core.reporting import reported_totals
from openavl.core.state import AVLState
from openavl.fileio.mass import load_mass, masini
from openavl.fileio.parser import AVLModel, parse_avl_file, prepare_model
from openavl.geom.geometry import build_geometry
from openavl.geometry.aircraft import Aircraft


class AVLSolver:
    """Python interface to the OpenAVL vortex-lattice aerodynamics solver.

    ``AVLSolver`` loads a standard AVL geometry file (``.avl``), constructs the
    discrete vortex lattice, and exposes AVL's run-case workflow for force
    integration, trimmed flight, stability-derivative extraction, and
    linearized flight-dynamics eigenanalysis. It is the primary entry point
    for scripting and application use.

    Typical workflow
    ----------------
    1. Construct a solver from a geometry (and optionally a mass) file.
    2. Set flight parameters with :meth:`set_parameter` and/or constructor
       keyword arguments.
    3. Define trim targets with :meth:`set_constraint`, optionally calling
       :meth:`setup_trim` for level-flight presets.
    4. Run the Newton iteration with :meth:`execute_run`.
    5. Retrieve coefficients (:meth:`get_results`), derivatives
       (:meth:`get_stability_derivatives`), and/or dynamic modes
       (:meth:`eigenvalues`). Visualize with :meth:`plot_aircraft`,
       :meth:`plot_lift_distribution`, and :meth:`plot_cp`.

    Attributes
    ----------
    geo_file : pathlib.Path
        Path to the loaded ``.avl`` geometry file.
    mass_file : pathlib.Path or None
        Path to the loaded ``.mass`` file, if one was provided.
    model : AVLModel
        Parsed geometry, surface, and control-surface definitions.
    state : AVLState
        Low-level solver state arrays and run-case data. Advanced users may
        inspect lattice dimensions via ``state.nvor``, control names via
        ``state.control_names``, and similar fields.
    debug : bool
        When ``True``, enables verbose solver diagnostics.

    See Also
    --------
    openavl.analysis.deriv.StabilityDerivatives
        Container returned by :meth:`get_stability_derivatives`.
    openavl.analysis.amode.EigenAnalysisResult
        Container returned by :meth:`eigenvalues`.
    """

    _VARIABLE_MAP = {
        "alpha": C.IVALFA,
        "beta": C.IVBETA,
        "pb/2v": C.IVROTX,
        "pb/2V": C.IVROTX,
        "qc/2v": C.IVROTY,
        "qc/2V": C.IVROTY,
        "rb/2v": C.IVROTZ,
        "rb/2V": C.IVROTZ,
    }

    _PARAMETER_MAP = {
        "mach": C.IPMACH,
        "velocity": C.IPVEE,
        "vee": C.IPVEE,
        "density": C.IPRHO,
        "rho": C.IPRHO,
        "gravity": C.IPGEE,
        "cl": C.IPCL,
        "cd0": C.IPCD0,
        "bank": C.IPPHI,
        "phi": C.IPPHI,
        "theta": C.IPTHE,
        "psi": C.IPPSI,
        "xcg": C.IPXCG,
        "ycg": C.IPYCG,
        "zcg": C.IPZCG,
        "mass": C.IPMASS,
        "ixx": C.IPIXX,
        "iyy": C.IPIYY,
        "izz": C.IPIZZ,
        "ixy": C.IPIXY,
        "iyz": C.IPIYZ,
        "izx": C.IPIZX,
        "ixz": C.IPIZX,
    }

    _CONSTRAINT_MAP = {
        "alpha": C.ICALFA,
        "beta": C.ICBETA,
        "pb/2v": C.ICROTX,
        "pb/2V": C.ICROTX,
        "qc/2v": C.ICROTY,
        "qc/2V": C.ICROTY,
        "rb/2v": C.ICROTZ,
        "rb/2V": C.ICROTZ,
        "cl": C.ICCL,
        "cy": C.ICCY,
        "cmx": C.ICMOMX,
        "cll": C.ICMOMX,
        "cm": C.ICMOMY,
        "cmy": C.ICMOMY,
        "cn": C.ICMOMZ,
        "cmz": C.ICMOMZ,
    }

    def __init__(
        self,
        geo: str | Path | Aircraft,
        mass_file: str | Path | None = None,
        debug: bool = False,
        **state_options: Any,
    ) -> None:
        """Load an aircraft model and prepare the vortex-lattice solution.

        Parses ``geo`` (a path or :class:`~openavl.geometry.Aircraft`), resolves
        relative airfoil and include paths from the geometry directory when
        applicable, allocates solver arrays, and builds the panel geometry.
        When ``mass_file`` is supplied, mass and inertia properties are read
        from the AVL mass file and merged with any run-case parameters passed
        as keyword arguments.

        Parameters
        ----------
        geo:
            Path to an AVL geometry file (``.avl``) or a programmatic
            :class:`~openavl.geometry.Aircraft` instance.
        mass_file:
            Optional path to an AVL mass/inertia file (``.mass``). Required
            for trim setups that infer airspeed from weight and target ``cl``.
        debug:
            Enable verbose diagnostic output from the solver core.
        **state_options:
            Initial run-case values forwarded to :class:`~openavl.core.state.AVLState`.
            Common keywords include ``alpha`` and ``beta`` (degrees),
            ``vel`` (airspeed), ``rho`` (density), ``gravity``, ``cl``,
            ``cd0``, ``bank``, and ``xcg``/``ycg``/``zcg``. Any name
            recognized by :meth:`set_parameter` (for example ``mach``) is
            reapplied after a mass file is loaded.

        Raises
        ------
        FileNotFoundError
            If ``mass_file`` is given but cannot be read or is empty.
        """
        self.debug = debug
        self.mass_file = Path(mass_file) if mass_file else None
        if isinstance(geo, Aircraft):
            base_dir = state_options.pop("base_dir", None)
            self.geo_file = None
            self.model: AVLModel = geo.to_avl_model(base_dir=base_dir)
        else:
            self.geo_file = Path(geo)
            self.model = prepare_model(parse_avl_file(self.geo_file), base_dir=self.geo_file.parent)
        self.state: AVLState = AVLState.from_model(self.model, debug=debug, **state_options)
        build_geometry(self.state, self.model)
        self.state.lgeo = True
        self.state.lenc = True

        masini(self.state)
        self._apply_default_mass_parameters()
        if self.mass_file is not None:
            props = load_mass(self.state, self.mass_file)
            if props is None:
                raise FileNotFoundError(f"Mass file not found or empty: {self.mass_file}")
            self.model.mass = props
            self._apply_parameter_options(state_options)

    def _apply_default_mass_parameters(self) -> None:
        """Seed run-case mass and inertia without overwriting CG or flow parameters."""
        s = self.state
        ir = 0
        s.parval[C.IPMASS, ir] = s.rmass0
        s.parval[C.IPIXX, ir] = s.riner0[0, 0]
        s.parval[C.IPIYY, ir] = s.riner0[1, 1]
        s.parval[C.IPIZZ, ir] = s.riner0[2, 2]
        s.parval[C.IPIXY, ir] = s.riner0[0, 1]
        s.parval[C.IPIYZ, ir] = s.riner0[1, 2]
        s.parval[C.IPIZX, ir] = s.riner0[2, 0]

    def _apply_parameter_options(self, options: dict[str, Any]) -> None:
        """Reapply constructor run-case parameters after mass-file defaults."""
        for name, value in options.items():
            if name.strip().lower() in self._PARAMETER_MAP:
                self.set_parameter(name, float(value))

    def _resolve_control_index(self, name: str) -> int | None:
        """Return control index for a name (case-insensitive), or None."""
        key = name.strip()
        if key in self.model.control_map:
            return self.model.control_map[key]
        lower = key.lower()
        for ctrl_name, idx in self.model.control_map.items():
            if ctrl_name.lower() == lower:
                return idx
        return None

    def _resolve_variable_index(self, name: str) -> int:
        """Map a variable or control name to the solver variable index."""
        key = name.strip()
        lower = key.lower()
        if lower in self._VARIABLE_MAP:
            return self._VARIABLE_MAP[lower]
        ctrl_idx = self._resolve_control_index(key)
        if ctrl_idx is not None:
            return C.IVTOT + ctrl_idx
        raise KeyError(f"Unknown variable: {name}")

    def set_variable(self, name: str, value: float) -> None:
        """Set a run-case variable (alpha/beta in degrees, controls in degrees)."""
        key = name.strip()
        lower = key.lower()
        if lower in self._VARIABLE_MAP:
            idx = self._VARIABLE_MAP[lower]
            if idx == C.IVALFA:
                self.state.alfa = float(value) * self.state.dtr
            elif idx == C.IVBETA:
                self.state.beta = float(value) * self.state.dtr
            else:
                self.state.wrot[idx - C.IVROTX] = float(value)
            return

        ctrl_idx = self._resolve_control_index(key)
        if ctrl_idx is not None:
            self.state.delcon[ctrl_idx] = float(value)
            return

        raise KeyError(f"Unknown variable: {name}")

    def set_parameter(self, name: str, value: float) -> None:
        """Set a run-case parameter for the current analysis.

        Parameters are fixed inputs to the AVL run case (freestream conditions,
        reference coefficients, attitude, and mass properties). Unlike
        :meth:`set_constraint`, they are not adjusted during the Newton trim
        iteration.

        Parameters
        ----------
        name:
            Parameter name (case-insensitive). Accepted names include
            ``mach``, ``velocity``/``vee``, ``density``/``rho``, ``gravity``,
            ``cl``, ``cd0``, ``bank``/``phi``, ``theta``, ``psi``,
            ``xcg``/``ycg``/``zcg``, ``mass``, and the inertia terms
            ``ixx``/``iyy``/``izz``/``ixy``/``ixz``/``iyz``. Products of
            inertia use AVL mass-file sign convention at the API boundary and
            are negated for the internal inertia matrix.
        value:
            Parameter value in the units expected by AVL (for example m/s for
            velocity, kg/m³ for density, degrees for bank angle).

        Notes
        -----
        When a mass file is loaded, setting ``cl`` before :meth:`setup_trim`
        lets the solver compute the airspeed required for that lift coefficient
        in level flight. The same keywords may also be passed to the
        :class:`AVLSolver` constructor.
        """
        key = name.strip().lower()
        if key not in self._PARAMETER_MAP:
            raise KeyError(f"Unknown parameter: {name}")
        idx = self._PARAMETER_MAP[key]
        stored_value = -float(value) if idx in {C.IPIXY, C.IPIYZ, C.IPIZX} else float(value)
        self.state.parval[idx, 0] = stored_value
        if idx == C.IPMACH:
            self.state.mach = float(value)

    def set_constraint(self, variable: str, constraint: str, value: float) -> None:
        """Couple a trim variable to a flight-state constraint for the Newton solve.

        Each call declares that ``variable`` should be adjusted until
        ``constraint`` equals ``value``. This is the programmatic equivalent of
        AVL run-case variable/constraint assignments used for trimmed flight.

        Parameters
        ----------
        variable:
            Name of the quantity to be solved. Built-in flight variables are
            ``alpha``, ``beta``, ``pb/2v``, ``qc/2v``, and ``rb/2v``. Any
            control surface name defined in the geometry (for example
            ``elevator`` or ``aileron``) may also be used.
        constraint:
            Target quantity to satisfy. Accepted names are ``alpha``, ``beta``,
            ``pb/2v``, ``qc/2v``, ``rb/2v``, ``cl``, ``cy``, ``cm`` (or
            ``cmy``), ``cll`` (or ``cmx``), and ``cn`` (or ``cmz``). Passing a
            control surface name with the same control as ``variable`` fixes
            that control deflection directly in degrees.
        value:
            Desired constraint value. Coefficient targets (``cl``, ``cy``,
            ``cm``, ``cll``, ``cn``) are dimensionless. Angle targets
            (``alpha``, ``beta``) are in degrees. Body-rate targets
            (``pb/2v``, ``qc/2v``, ``rb/2v``) are non-dimensional, as in AVL.

        Notes
        -----
        Each variable may be tied to at most one constraint, and each
        constraint may govern at most one variable. Conflicting assignments
        produce an ill-posed trim problem.

        Typical level-flight trim pairs ``alpha`` with ``cl``, lateral
        controls with rolling/yawing moment coefficients, and the elevator
        with ``cm``. Call :meth:`execute_run` after all constraints are set.

        Raises
        ------
        KeyError
            If ``variable`` or ``constraint`` is not recognized.
        """
        iv = self._resolve_variable_index(variable)
        con_key = constraint.strip().lower()
        ctrl_idx = self._resolve_control_index(constraint)
        if ctrl_idx is not None and iv == C.IVTOT + ctrl_idx:
            ic = C.ICTOT + ctrl_idx
            self.state.icon[iv, 0] = ic
            self.state.conval[ic, 0] = float(value)
            self.state.delcon[ctrl_idx] = float(value)
            return

        if con_key not in self._CONSTRAINT_MAP:
            raise KeyError(f"Unknown constraint: {constraint}")

        ic = self._CONSTRAINT_MAP[con_key]
        self.state.icon[iv, 0] = ic
        self.state.conval[ic, 0] = float(value)

    def execute_run(self, max_iter: int = 20) -> None:
        """Run the Newton trim iteration and update solver state.

        Builds the vortex lattice, evaluates aerodynamic forces and moments, and
        (when ``max_iter > 0``) iterates on the trim variables set by
        :meth:`set_constraint` until constraint residuals fall below the
        convergence tolerance (2e-5) or ``max_iter`` is reached.

        Parameters
        ----------
        max_iter:
            Maximum number of Newton iterations. Pass ``0`` to evaluate forces
            at the current variable values without trimming.

        Notes
        -----
        Converged coefficients and flight variables are stored in
        :attr:`state` and can be retrieved with :meth:`get_results`. Check
        ``get_results()['converged']`` if trim did not meet tolerance.
        """
        from openavl.core.exec import exec_solve

        exec_solve(self.state, niter=max_iter)

    def get_results(self) -> dict[str, Any]:
        """Return aerodynamic coefficients and metadata from the latest solve.

        The returned mapping summarizes the converged (or last-iterated) run
        case after :meth:`execute_run`. Lift, drag, and side force are
        stability-axis coefficients. Reported body-axis force and moment
        coefficients follow AVL's ``OUTTOT`` / OptVL ``get_total_forces`` sign
        convention (NASA standard when ``state.lnasa_sa`` is ``True``).

        Returns
        -------
        dict[str, Any]
            Dictionary with the following keys:

            ``CL``, ``CD``, ``CY``
                Total lift, drag, and side-force coefficients.
            ``Cl``, ``Cm``, ``Cn``
                Body-axis rolling, pitching, and yawing moment coefficients as
                in AVL ``Cltot`` / ``Cmtot`` / ``Cntot``.
            ``Cl_sa``, ``Cm_sa``, ``Cn_sa``
                Stability-axis roll, pitch, and yaw moment coefficients as in
                AVL ``Cl'tot``, ``Cmtot``, and ``Cn'tot``.
            ``Cx``, ``Cy``, ``Cz``
                Body-axis force coefficients (``DIR`` applied to ``Cx`` and
                ``Cz`` when using NASA-standard axes).
            ``CDV``, ``CLFF``, ``CDFF``, ``CYFF``, ``SPANEF``
                Viscous and Trefftz-plane drag bookkeeping terms.
            ``alpha_deg``, ``beta_deg``
                Solved angle of attack and sideslip in degrees.
            ``control_deflections``
                Control surface deflections in degrees, keyed by control name
                (e.g. ``results["control_deflections"]["aileron"]``).
            ``mach``
                Mach number for the run case.
            ``converged``
                ``True`` if the Newton iteration met the convergence tolerance.
            ``geometry``
                Lattice size summary with keys ``NVOR``, ``NSTRIP``,
                ``NSURF``, ``NBODY``, and ``NLNODE``.

        Notes
        -----
        Raw integration values remain in ``state.cmtot`` and ``state.cftot``.
        If no solve has been run, values reflect the initialized state and
        ``converged`` will be ``False``.
        """
        s = self.state
        reported = reported_totals(s)
        cl, cm, cn = reported["CM"]
        cl_sa, cm_sa, cn_sa = reported["CM_sa"]
        cx, cy, cz = reported["CF"]
        control_deflections = {
            name: float(s.delcon[idx])
            for idx, name in enumerate(s.control_names[: s.ncontrol])
        }
        return {
            "CL": s.cltot,
            "CD": s.cdtot,
            "CY": s.cytot,
            "Cl": cl,
            "Cm": cm,
            "Cn": cn,
            "Cl_sa": cl_sa,
            "Cm_sa": cm_sa,
            "Cn_sa": cn_sa,
            "Cx": cx,
            "Cy": cy,
            "Cz": cz,
            "CDV": s.cdvtot,
            "CLFF": s.clff,
            "CDFF": s.cdff,
            "CYFF": s.cyff,
            "SPANEF": s.spanef,
            "converged": s.lsol,
            "geometry": {
                "NVOR": s.nvor,
                "NSTRIP": s.nstrip,
                "NSURF": s.nsurf,
                "NBODY": s.nbody,
                "NLNODE": s.nlnode,
            },
            "alpha_deg": s.alfa / s.dtr,
            "beta_deg": s.beta / s.dtr,
            "control_deflections": control_deflections,
            "mach": s.mach,
        }

    def get_stability_derivatives(self) -> StabilityDerivatives:
        """Extract stability-axis force, moment, and control derivatives.

        Computes classical partial derivatives of the total aerodynamic
        coefficients with respect to small perturbations in angle of attack,
        sideslip, non-dimensional body rates, and control-surface deflections.
        The underlying sensitivity arrays are populated by the most recent call
        to :meth:`execute_run`.

        Returns
        -------
        StabilityDerivatives
            Dataclass of stability-axis derivatives. Perturbation derivatives
            with respect to ``alpha``, ``beta``, body rates, and control
            deflections use per-radian units (for example ``CL_a`` is dCL/dα
            and ``Cm_d["elevator"]`` is dCm/dδ_elevator). Rate derivatives
            ``*_p``, ``*_q``, and ``*_r`` are non-dimensional.

        Notes
        -----
        Run a converged trim or perturbation solve before calling this method;
        otherwise the returned values correspond to the uninitialized or
        partially converged state. Control effectiveness entries are omitted
        for surfaces that are not active in the current run case.
        """
        from openavl.analysis.deriv import compute_stability_derivatives

        return compute_stability_derivatives(self.state)

    def setup_trim(self, mode: int = 1) -> None:
        """Initialize longitudinal trim constraints and flight parameters.

        Applies AVL's trim preset logic to derive consistent airspeed, load
        factor, and body-rate constraints for a coordinated maneuver. This is
        a convenience helper; it does not run the aerodynamic solve.

        Parameters
        ----------
        mode:
            ``1`` for level flight or a steady banked turn (load factor from
            bank angle ``phi``). ``2`` for a pull-up maneuver with prescribed
            turn radius ``rad``.

        Notes
        -----
        The method reads and may update run-case parameters already stored in
        the solver state—principally ``cl``, ``velocity``, ``density``,
        ``gravity``, ``mass``, ``bank`` (``phi``), and ``rad``. For mode
        ``1``, if ``velocity`` is unset but ``cl`` and ``mass`` are available,
        airspeed is computed from the level-flight lift equation; conversely,
        ``cl`` is computed when ``velocity`` is given.

        Longitudinal trim is preset by constraining ``alpha`` to ``cl`` and
        fixing pitch/yaw rate constraints for the requested maneuver. Lateral
        and directional constraints (for example ``beta``, ``aileron``/``cll``,
        ``rudder``/``cn``) are **not** preserved—reapply them with
        :meth:`set_constraint` before calling :meth:`execute_run`.

        Raises
        ------
        ValueError
            If ``mode`` is not ``1`` or ``2``.
        """
        from openavl.analysis.trim import setup_trim

        setup_trim(self.state, mode=mode)

    def eigenvalues(self, use_approx: bool = False) -> EigenAnalysisResult:
        """Perform linearized flight-dynamics eigenanalysis at the current state.

        Assembles the perturbation state matrix from the solved aerodynamic
        derivatives and aircraft mass properties, then identifies classical
        rigid-body modes (short period, phugoid, Dutch roll, spiral, and
        related subsidence modes).

        Parameters
        ----------
        use_approx:
            If ``False`` (default), build the full state matrix from stability
            derivatives. If ``True``, use AVL's approximate matrix
            (``APPMAT``) for faster but less accurate mode estimates.

        Returns
        -------
        EigenAnalysisResult
            Object containing the state matrix, eigenvalues, eigenvectors, and
            a list of :class:`~openavl.analysis.amode.FlightMode` records.
            Each mode exposes ``name``, ``eigenvalue``, ``frequency_hz``,
            ``damping_ratio``, and ``time_constant`` when applicable.

        Notes
        -----
        Requires a previously converged solve with positive mass, inertia, and
        airspeed. Eigenanalysis at an untrimmed or unconverged point may
        return empty mode lists or non-physical roots.
        """
        from openavl.analysis.amode import solve_eigenvalues

        return solve_eigenvalues(self.state, use_approx=use_approx)

    def get_system_matrices(
        self,
        in_body_axis: bool = False,
        use_approx: bool = False,
        ir: int = 0,
    ) -> tuple[Any, Any, Any]:
        """Return the modal state-space matrices used for eigenanalysis.

        Builds the same ``A``, ``B``, and residual vectors as
        :meth:`eigenvalues` before eigen decomposition. By default the matrices
        match the eigenvectors returned by :meth:`eigenvalues`. Set
        ``in_body_axis=True`` to apply AVL's ``SYSSHO`` sign changes for
        body-axis display.

        Parameters
        ----------
        in_body_axis:
            If ``True``, flip signs on rows/columns for ``u``, ``w``, ``p``,
            ``r``, ``x``, and ``z`` (AVL geometry-axis to body-axis convention).
        use_approx:
            If ``True``, use the approximate ``APPMAT`` system instead of the
            full ``SYSMAT`` assembly.
        ir:
            Run-case index (default ``0``).

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            State matrix ``A`` (12×12), control matrix ``B`` (12×n_control),
            and residual vector ``R`` (length 12).
        """
        from openavl.analysis.amode import apply_body_axis_signs, build_appmat, build_sysmat

        if use_approx:
            asys, bsys, rsys = build_appmat(self.state, ir)
        else:
            asys, bsys, rsys = build_sysmat(self.state, ir)

        ncontrol = self.state.ncontrol
        if ncontrol > 0:
            bsys = bsys[:, :ncontrol]
        else:
            bsys = bsys[:, :0]

        if in_body_axis:
            asys, bsys, rsys = apply_body_axis_signs(asys, bsys, rsys, ncontrol=ncontrol)

        return asys, bsys, rsys

    def get_system_matrix(
        self,
        in_body_axis: bool = False,
        use_approx: bool = False,
        ir: int = 0,
    ) -> Any:
        """Return the modal state matrix ``A`` used for eigenanalysis.

        Convenience wrapper around :meth:`get_system_matrices` that returns only
        the ``A`` matrix. See that method for parameter details.
        """
        asys, _, _ = self.get_system_matrices(
            in_body_axis=in_body_axis,
            use_approx=use_approx,
            ir=ir,
        )
        return asys

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
