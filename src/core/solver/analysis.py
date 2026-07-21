"""Trim, derivative, and flight-dynamics analysis adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openavl.analysis.amode import EigenAnalysisResult
    from openavl.analysis.deriv import (
        BodyAxisDerivatives,
        ControlAxis,
        ControlDerivatives,
        StabilityDerivatives,
    )


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
        ``*_p``, ``*_q``, and ``*_r`` are non-dimensional. Also includes
        ``xnp`` (neutral point, geometry length units) and ``sm`` (static
        margin as a fraction of ``cref``).

    Notes
    -----
    Run a converged trim or perturbation solve before calling this method;
    otherwise the returned values correspond to the uninitialized or
    partially converged state. Control effectiveness entries are omitted
    for surfaces that are not active in the current run case.
    """
    from openavl.analysis.deriv import compute_stability_derivatives

    return compute_stability_derivatives(self.state)

def get_control_derivatives(
    self,
    axis: ControlAxis = "stability",
) -> ControlDerivatives:
    """Extract control-surface force and moment derivatives.

    Returns a control-only matrix (one row per surface) in either
    stability or body axes. Values are per radian of deflection.

    Parameters
    ----------
    axis:
        ``"stability"`` (default) for columns ``CL, CD, CY, Cl, Cm, Cn``,
        or ``"body"`` for ``CX, CY, CZ, Cl, Cm, Cn``.

    Returns
    -------
    ControlDerivatives
        ``rows`` are control names, ``cols`` are coefficient labels for
        the chosen axis, and ``values[i][j]`` is
        d(col_j) / d(δ_row_i) in 1/rad.

    Notes
    -----
    Requires a prior :meth:`execute_run` so the ``*_d`` sensitivity arrays
    are populated. Stability-axis moments use the same transform as
    :meth:`get_stability_derivatives`; body-axis rows match the control
    block of :func:`openavl.analysis.deriv.compute_body_axis_derivatives`.
    """
    from openavl.analysis.deriv import compute_control_derivatives

    return compute_control_derivatives(self.state, axis=axis)

def get_body_axis_derivatives(self) -> BodyAxisDerivatives:
    """Extract the body-axis force and moment derivative matrix.

    Returns AVL ``DERMATB``-style derivatives with respect to normalized
    velocity and rate perturbations (``u``–``r``) and control deflections.
    Control rows are per radian; velocity and rate rows follow
    :func:`openavl.analysis.deriv.compute_body_axis_derivatives`.

    Returns
    -------
    BodyAxisDerivatives
        Matrix with columns ``CX, CY, CZ, Cl, Cm, Cn``. Control rows are
        labeled ``d1``, ``d2``, … in index order; use
        :meth:`get_control_derivatives` with ``axis="body"`` for a
        control-only matrix keyed by control name.

    Notes
    -----
    Requires a prior :meth:`execute_run` so the ``*_u`` and ``*_d``
    sensitivity arrays are populated.
    """
    from openavl.analysis.deriv import compute_body_axis_derivatives

    return compute_body_axis_derivatives(self.state)

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
