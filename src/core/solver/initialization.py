"""Shared construction helpers for the high-level solver facade."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openavl import constants as C
from openavl.core.state import AVLState
from openavl.fileio.mass import masini
from openavl.fileio.parser import AVLModel
from openavl.geom.geometry import build_geometry


def initialize_solver(
    solver: Any,
    model: AVLModel,
    *,
    debug: bool,
    state_options: dict[str, Any],
    geo_file: Path | None = None,
    mass_file: Path | None = None,
) -> None:
    """Initialize a solver facade from an already parsed AVL model."""
    solver.debug = debug
    solver.geo_file = geo_file
    solver.mass_file = mass_file
    solver.model = model
    solver.state = AVLState.from_model(model, debug=debug, **state_options)
    build_geometry(solver.state, model)
    solver.state.lgeo = True
    solver.state.lenc = True
    masini(solver.state)
    apply_default_mass_parameters(solver.state)


def apply_default_mass_parameters(state: AVLState) -> None:
    """Seed run-case mass and inertia without replacing flow parameters."""
    ir = 0
    state.parval[C.IPMASS, ir] = state.rmass0
    state.parval[C.IPIXX, ir] = state.riner0[0, 0]
    state.parval[C.IPIYY, ir] = state.riner0[1, 1]
    state.parval[C.IPIZZ, ir] = state.riner0[2, 2]
    state.parval[C.IPIXY, ir] = state.riner0[0, 1]
    state.parval[C.IPIYZ, ir] = state.riner0[1, 2]
    state.parval[C.IPIZX, ir] = state.riner0[2, 0]


def apply_parameter_options(solver: Any, options: dict[str, Any]) -> None:
    """Apply constructor options recognized by the solver parameter map."""
    for name, value in options.items():
        if name.strip().lower() in solver._PARAMETER_MAP:
            solver.set_parameter(name, float(value))
