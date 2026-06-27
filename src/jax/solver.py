"""High-level JAX AVL solver API (Phase 4B)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openavl.core.solver import AVLSolver
from openavl.jax.analysis import make_run_analysis_jit, run_analysis
from openavl.jax.backend import jax
from openavl.jax.snapshot import snapshot_analysis_geometry, snapshot_flow, snapshot_refs
from openavl.jax.types import AnalysisResult, FlowCondition


class JaxAVLSolver:
    """JAX-enabled AVL solver with automatic differentiation on flight conditions."""

    def __init__(
        self,
        geo_file: str | Path,
        mass_file: str | Path | None = None,
        *,
        use_jit: bool = True,
        **state_options: Any,
    ) -> None:
        """Build geometry with the NumPy solver and snapshot JAX arrays.

        Parameters
        ----------
        geo_file:
            Path to an AVL geometry file.
        mass_file:
            Optional mass/inertia file.
        use_jit:
            When True (default), :meth:`run` uses the JIT-compiled analysis.
        **state_options:
            Forwarded to :class:`openavl.core.solver.AVLSolver`.
        """
        self._numpy_solver = AVLSolver(geo_file, mass_file, **state_options)
        state = self._numpy_solver.state
        self._geom = snapshot_analysis_geometry(state)
        self._refs = snapshot_refs(state)
        self._use_jit = use_jit
        self._jit_fn = make_run_analysis_jit(self._geom, self._refs)

    @property
    def numpy_solver(self) -> AVLSolver:
        """Underlying NumPy AVL solver (geometry builder and reference runner)."""
        return self._numpy_solver

    @property
    def state(self):
        """Latest NumPy ``AVLState`` from the underlying solver."""
        return self._numpy_solver.state

    def run(self, flow: FlowCondition | None = None) -> AnalysisResult:
        """Evaluate forces for a flow condition (defaults to current state)."""
        if flow is None:
            flow = snapshot_flow(self._numpy_solver.state)
        if self._use_jit:
            return self._jit_fn(flow)
        return run_analysis(flow, self._geom, self._refs)

    def grad(self, output: str, flow: FlowCondition | None = None):
        """Gradient of a scalar output coefficient w.r.t. ``flow``."""
        if flow is None:
            flow = snapshot_flow(self._numpy_solver.state)

        def objective(f: FlowCondition):
            return getattr(self.run(f), output)

        return jax.grad(objective)(flow)

    def jacobian(self, flow: FlowCondition | None = None):
        """Jacobian of :func:`run_analysis` outputs w.r.t. ``flow``."""
        if flow is None:
            flow = snapshot_flow(self._numpy_solver.state)
        return jax.jacrev(run_analysis)(flow, self._geom, self._refs)
