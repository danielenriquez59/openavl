"""High-level AVL solver API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openavl import constants as C

from openavl.fileio.mass import load_mass
from openavl.fileio.parser import AVLModel, parse_avl_file, prepare_model
from openavl.geometry.aircraft import Aircraft

from . import analysis as _analysis
from . import initialization as _initialization
from . import results as _results
from . import visualization as _visualization


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
    5. Retrieve coefficients (:meth:`get_results`), aerodynamic accelerations
       (:meth:`get_aero_accel`), derivatives (:meth:`get_stability_derivatives`),
       and/or dynamic modes (:meth:`eigenvalues`). Visualize with
       :meth:`plot_aircraft`, :meth:`plot_lift_distribution`, and
       :meth:`plot_cp`.

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
    openavl.analysis.deriv.BodyAxisDerivatives
        Container returned by :meth:`get_body_axis_derivatives`.
    openavl.analysis.deriv.ControlDerivatives
        Container returned by :meth:`get_control_derivatives`.
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
        resolved_mass_file = Path(mass_file) if mass_file else None
        if isinstance(geo, Aircraft):
            base_dir = state_options.pop("base_dir", None)
            geo_file = None
            model = geo.to_avl_model(base_dir=base_dir)
        else:
            geo_file = Path(geo)
            model = prepare_model(parse_avl_file(geo_file), base_dir=geo_file.parent)
        _initialization.initialize_solver(
            self,
            model,
            debug=debug,
            state_options=state_options,
            geo_file=geo_file,
            mass_file=resolved_mass_file,
        )
        if self.mass_file is not None:
            props = load_mass(self.state, self.mass_file)
            if props is None:
                raise FileNotFoundError(f"Mass file not found or empty: {self.mass_file}")
            self.model.mass = props
            self._apply_parameter_options(state_options)

    @classmethod
    def _from_model(
        cls,
        model: AVLModel,
        *,
        debug: bool = False,
        **state_options: Any,
    ) -> AVLSolver:
        """Construct a solver from a parsed model without loading geometry."""
        solver = object.__new__(cls)
        _initialization.initialize_solver(
            solver,
            model,
            debug=debug,
            state_options=state_options,
        )
        return solver

    def _apply_default_mass_parameters(self) -> None:
        """Seed run-case mass and inertia without overwriting CG or flow parameters."""
        _initialization.apply_default_mass_parameters(self.state)

    def _apply_parameter_options(self, options: dict[str, Any]) -> None:
        """Reapply constructor run-case parameters after mass-file defaults."""
        _initialization.apply_parameter_options(self, options)

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

    def replace_constraints(self, constraints: list[tuple[str, str, float]]) -> None:
        """Replace every run-case constraint assignment for the current analysis.

        Variables omitted from ``constraints`` are fixed at zero. This prevents
        assignments and solved control deflections from a previously applied run
        case from leaking into a new one. Explicit fixed values can be applied
        afterward with :meth:`set_variable`.

        Parameters
        ----------
        constraints:
            Complete ``(variable, constraint, value)`` assignments for the run
            case. Names and values follow :meth:`set_constraint`.
        """
        fixed_variables = [
            "alpha",
            "beta",
            "pb/2V",
            "qc/2V",
            "rb/2V",
            *self.state.control_names,
        ]
        for variable in fixed_variables:
            self.set_constraint(variable, variable, 0.0)
        for variable, constraint, value in constraints:
            self.set_constraint(variable, constraint, value)

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


    get_results = _results.get_results
    get_aero_accel = _results.get_aero_accel

    get_stability_derivatives = _analysis.get_stability_derivatives
    get_control_derivatives = _analysis.get_control_derivatives
    get_body_axis_derivatives = _analysis.get_body_axis_derivatives
    setup_trim = _analysis.setup_trim
    eigenvalues = _analysis.eigenvalues
    get_system_matrices = _analysis.get_system_matrices
    get_system_matrix = _analysis.get_system_matrix

    plot_aircraft = _visualization.plot_aircraft
    plot_geom = _visualization.plot_geom
    plot_lift_distribution = _visualization.plot_lift_distribution
    get_cp_data = _visualization.get_cp_data
    plot_cp = _visualization.plot_cp
