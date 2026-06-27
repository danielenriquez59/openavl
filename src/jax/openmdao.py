"""OpenMDAO explicit component wrapping JAX AVL analysis (Phase 4C)."""

from __future__ import annotations

from typing import Any

import numpy as np

from openavl.jax.analysis import run_analysis
from openavl.jax.backend import jax
from openavl.jax.snapshot import snapshot_analysis_geometry, snapshot_refs
from openavl.jax.types import FlowCondition

try:
    import openmdao.api as om
except ImportError:  # pragma: no cover - optional dependency
    om = None  # type: ignore[assignment]


if om is not None:

    class JaxAVLComp(om.ExplicitComponent):
        """OpenMDAO component exposing JAX AVL force coefficients and exact partials."""

        def initialize(self) -> None:
            self.options.declare("geo_file", types=str)
            self.options.declare("mass_file", default=None, types=(str, type(None)))

        def setup(self) -> None:
            from openavl.core.solver import AVLSolver

            solver = AVLSolver(self.options["geo_file"], self.options["mass_file"])
            solver.execute_run(max_iter=1)
            state = solver.state
            self._geom = snapshot_analysis_geometry(state)
            self._refs = snapshot_refs(state)
            self._ncontrol = int(state.ncontrol)

            self.add_input("alpha", val=0.0)
            self.add_input("beta", val=0.0)
            self.add_input("pb2v", val=0.0)
            self.add_input("qc2v", val=0.0)
            self.add_input("rb2v", val=0.0)
            self.add_input("mach", val=0.0)
            for n in range(self._ncontrol):
                self.add_input(f"delcon_{n}", val=0.0)

            self.add_output("CL", val=0.0)
            self.add_output("CD", val=0.0)
            self.add_output("CY", val=0.0)
            self.add_output("CMx", val=0.0)
            self.add_output("CMy", val=0.0)
            self.add_output("CMz", val=0.0)

            self.declare_partials(of="*", wrt="*")

        @staticmethod
        def _scalar(value: Any) -> np.float64:
            """Extract a JAX-compatible scalar from an OpenMDAO input array."""
            return np.float64(np.asarray(value, dtype=np.float64).item())

        def _flow_from_inputs(self, inputs: Any) -> FlowCondition:
            """Build a :class:`FlowCondition` with 0-D JAX leaves from OpenMDAO inputs."""
            delcon = np.array(
                [self._scalar(inputs[f"delcon_{n}"]) for n in range(self._ncontrol)],
                dtype=np.float64,
            )
            return FlowCondition(
                alfa=self._scalar(inputs["alpha"]),
                beta=self._scalar(inputs["beta"]),
                wrot=np.asarray(
                    [
                        self._scalar(inputs["pb2v"]),
                        self._scalar(inputs["qc2v"]),
                        self._scalar(inputs["rb2v"]),
                    ],
                    dtype=np.float64,
                ),
                mach=self._scalar(inputs["mach"]),
                delcon=delcon,
            )

        def compute(self, inputs: Any, outputs: Any) -> None:
            flow = self._flow_from_inputs(inputs)
            result = run_analysis(flow, self._geom, self._refs)
            outputs["CL"] = float(result.CL)
            outputs["CD"] = float(result.CD)
            outputs["CY"] = float(result.CY)
            outputs["CMx"] = float(result.CM[0])
            outputs["CMy"] = float(result.CM[1])
            outputs["CMz"] = float(result.CM[2])

        def compute_partials(self, inputs: Any, partials: Any) -> None:
            flow = self._flow_from_inputs(inputs)
            jac = jax.jacrev(run_analysis)(flow, self._geom, self._refs)

            def _set(out: str, wrt: str, value: float) -> None:
                partials[out, wrt] = value

            _set("CL", "alpha", float(jac.CL.alfa))
            _set("CL", "beta", float(jac.CL.beta))
            _set("CL", "pb2v", float(jac.CL.wrot[0]))
            _set("CL", "qc2v", float(jac.CL.wrot[1]))
            _set("CL", "rb2v", float(jac.CL.wrot[2]))
            _set("CL", "mach", float(jac.CL.mach))
            for n in range(self._ncontrol):
                _set("CL", f"delcon_{n}", float(jac.CL.delcon[n]))

            _set("CD", "alpha", float(jac.CD.alfa))
            _set("CD", "beta", float(jac.CD.beta))
            _set("CD", "pb2v", float(jac.CD.wrot[0]))
            _set("CD", "qc2v", float(jac.CD.wrot[1]))
            _set("CD", "rb2v", float(jac.CD.wrot[2]))
            _set("CD", "mach", float(jac.CD.mach))
            for n in range(self._ncontrol):
                _set("CD", f"delcon_{n}", float(jac.CD.delcon[n]))

            _set("CY", "alpha", float(jac.CY.alfa))
            _set("CY", "beta", float(jac.CY.beta))
            _set("CY", "pb2v", float(jac.CY.wrot[0]))
            _set("CY", "qc2v", float(jac.CY.wrot[1]))
            _set("CY", "rb2v", float(jac.CY.wrot[2]))
            _set("CY", "mach", float(jac.CY.mach))
            for n in range(self._ncontrol):
                _set("CY", f"delcon_{n}", float(jac.CY.delcon[n]))

            for i, out in enumerate(("CMx", "CMy", "CMz")):
                _set(out, "alpha", float(jac.CM.alfa[i]))
                _set(out, "beta", float(jac.CM.beta[i]))
                _set(out, "pb2v", float(jac.CM.wrot[0, i]))
                _set(out, "qc2v", float(jac.CM.wrot[1, i]))
                _set(out, "rb2v", float(jac.CM.wrot[2, i]))
                _set(out, "mach", float(jac.CM.mach[i]))
                for n in range(self._ncontrol):
                    _set(out, f"delcon_{n}", float(jac.CM.delcon[n, i]))

else:
    # OpenMDAO is optional; import openavl.jax.openmdao only when the dependency is installed.
    JaxAVLComp = None  # type: ignore[misc, assignment]

__all__ = ["JaxAVLComp"]
