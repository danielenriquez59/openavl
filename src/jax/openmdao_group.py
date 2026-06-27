"""OpenMDAO group wrapper with geometry design variables for JAX AVL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from openavl.jax.backend import jax, jnp
from openavl.jax.geom_jax import (
    run_analysis_with_geometry,
    snapshot_topology,
)
from openavl.jax.snapshot import snapshot_analysis_geometry, snapshot_refs
from openavl.jax.types import FlowCondition, GeometryDesignParams

try:
    import openmdao.api as om
except ImportError:  # pragma: no cover
    om = None  # type: ignore[assignment]


@dataclass(frozen=True)
class SurfInfo:
    """Section-level geometry metadata for one model surface."""

    name: str
    nsec: int
    sec_offset: int
    baseline_aincs_deg: np.ndarray
    baseline_chords: np.ndarray
    baseline_xles: np.ndarray
    baseline_yles: np.ndarray
    baseline_zles: np.ndarray


def _build_surface_map(model: Any, state: Any) -> dict[str, SurfInfo]:
    """Map AVL surface names to section counts and baseline parameter arrays."""
    surfaces: dict[str, SurfInfo] = {}
    sec_offset = 0
    for surf in model.surfaces:
        nsec = len(surf.sections)
        surfaces[surf.name] = SurfInfo(
            name=surf.name,
            nsec=nsec,
            sec_offset=sec_offset,
            baseline_aincs_deg=np.asarray(
                [s.ainc_deg for s in surf.sections], dtype=np.float64
            ),
            baseline_chords=np.asarray([s.chord for s in surf.sections], dtype=np.float64),
            baseline_xles=np.asarray([s.xle for s in surf.sections], dtype=np.float64),
            baseline_yles=np.asarray([s.yle for s in surf.sections], dtype=np.float64),
            baseline_zles=np.asarray([s.zle for s in surf.sections], dtype=np.float64),
        )
        sec_offset += nsec
    return surfaces


def _deg_scalar(value: Any) -> np.float64:
    """Extract a scalar in degrees from an OpenMDAO input."""
    return np.float64(np.deg2rad(np.asarray(value, dtype=np.float64).item()))


def _scalar(value: Any) -> np.float64:
    """Extract a scalar from an OpenMDAO input."""
    return np.float64(np.asarray(value, dtype=np.float64).item())


def _inputs_to_flow(comp: Any, inputs: Any) -> FlowCondition:
    """Convert OpenMDAO inputs to a :class:`FlowCondition` (angles in radians)."""
    delcon = np.array(
        [_deg_scalar(inputs[name]) for name in comp._control_names],
        dtype=np.float64,
    )
    return FlowCondition(
        alfa=_deg_scalar(inputs["alpha"]),
        beta=_deg_scalar(inputs["beta"]),
        wrot=np.asarray(
            [
                _scalar(inputs["pb2v"]),
                _scalar(inputs["qc2v"]),
                _scalar(inputs["rb2v"]),
            ],
            dtype=np.float64,
        ),
        mach=_scalar(inputs["mach"]),
        delcon=delcon,
    )


def _inputs_to_design_params(comp: Any, inputs: Any) -> GeometryDesignParams:
    """Convert ``SurfName:param`` OpenMDAO inputs to concatenated design arrays."""
    aincs: list[float] = []
    chords: list[float] = []
    xles: list[float] = []
    yles: list[float] = []
    zles: list[float] = []
    for info in comp._surface_infos:
        aincs.extend(np.deg2rad(np.asarray(inputs[f"{info.name}:aincs"], dtype=np.float64)))
        chords.extend(np.asarray(inputs[f"{info.name}:chords"], dtype=np.float64))
        xles.extend(np.asarray(inputs[f"{info.name}:xles"], dtype=np.float64))
        yles.extend(np.asarray(inputs[f"{info.name}:yles"], dtype=np.float64))
        zles.extend(np.asarray(inputs[f"{info.name}:zles"], dtype=np.float64))
    return GeometryDesignParams(
        aincs=np.asarray(aincs, dtype=np.float64),
        chords=np.asarray(chords, dtype=np.float64),
        xles=np.asarray(xles, dtype=np.float64),
        yles=np.asarray(yles, dtype=np.float64),
        zles=np.asarray(zles, dtype=np.float64),
    )


def _stability_outputs_jax(result: Any, flow: FlowCondition, lnasa_sa: bool) -> jnp.ndarray:
    """Convert body-axis analysis result to stability-axis outputs (JAX)."""
    dir_ = jnp.where(lnasa_sa, -1.0, 1.0)
    ca = jnp.cos(flow.alfa)
    sa = jnp.sin(flow.alfa)
    cmx = result.CM[0]
    cmy = result.CM[1]
    cmz = result.CM[2]
    cl = dir_ * (cmx * ca + cmz * sa)
    cm = cmy
    cn = dir_ * (cmz * ca - cmx * sa)
    return jnp.array([result.CL, result.CD, result.CY, cl, cm, cn], dtype=jnp.float64)


def _stability_outputs(result: Any, flow: FlowCondition, lnasa_sa: bool) -> tuple[float, float, float, float, float, float]:
    """Convert body-axis analysis result to stability-axis OpenMDAO outputs."""
    dir_ = -1.0 if lnasa_sa else 1.0
    ca = float(np.cos(flow.alfa))
    sa = float(np.sin(flow.alfa))
    cmx = float(result.CM[0])
    cmy = float(result.CM[1])
    cmz = float(result.CM[2])
    cl = dir_ * (cmx * ca + cmz * sa)
    cm = cmy
    cn = dir_ * (cmz * ca - cmx * sa)
    return float(result.CL), float(result.CD), float(result.CY), cl, cm, cn


def _scatter_cotangents(
    comp: Any,
    flow_bar: Any,
    params_bar: Any,
    d_inputs: Any,
) -> None:
    """Scatter reverse-mode cotangents from VJP into OpenMDAO ``d_inputs``."""
    if "alpha" in d_inputs:
        d_inputs["alpha"] += float(flow_bar.alfa) * (np.pi / 180.0)
    if "beta" in d_inputs:
        d_inputs["beta"] += float(flow_bar.beta) * (np.pi / 180.0)
    if "pb2v" in d_inputs:
        d_inputs["pb2v"] += float(flow_bar.wrot[0])
    if "qc2v" in d_inputs:
        d_inputs["qc2v"] += float(flow_bar.wrot[1])
    if "rb2v" in d_inputs:
        d_inputs["rb2v"] += float(flow_bar.wrot[2])
    if "mach" in d_inputs:
        d_inputs["mach"] += float(flow_bar.mach)

    for n, name in enumerate(comp._control_names):
        if name in d_inputs:
            d_inputs[name] += float(flow_bar.delcon[n]) * (np.pi / 180.0)

    offset = 0
    for info in comp._surface_infos:
        nsec = info.nsec
        sl = slice(offset, offset + nsec)
        if f"{info.name}:aincs" in d_inputs:
            d_inputs[f"{info.name}:aincs"] += np.asarray(params_bar.aincs[sl]) * (np.pi / 180.0)
        if f"{info.name}:chords" in d_inputs:
            d_inputs[f"{info.name}:chords"] += np.asarray(params_bar.chords[sl])
        if f"{info.name}:xles" in d_inputs:
            d_inputs[f"{info.name}:xles"] += np.asarray(params_bar.xles[sl])
        if f"{info.name}:yles" in d_inputs:
            d_inputs[f"{info.name}:yles"] += np.asarray(params_bar.yles[sl])
        if f"{info.name}:zles" in d_inputs:
            d_inputs[f"{info.name}:zles"] += np.asarray(params_bar.zles[sl])
        offset += nsec


if om is not None:

    class OpenAVLComp(om.ExplicitComponent):
        """OpenMDAO component with geometry design variables and matrix-free AD."""

        def initialize(self) -> None:
            self.options.declare("geo_file", types=str)
            self.options.declare("mass_file", default=None, types=(str, type(None)))

        def setup(self) -> None:
            from openavl.core.solver import AVLSolver

            solver = AVLSolver(self.options["geo_file"], self.options["mass_file"])
            solver.execute_run(max_iter=1)
            state = solver.state
            model = solver.model

            self._baseline = snapshot_analysis_geometry(state)
            self._topo = snapshot_topology(state, model)
            self._refs = snapshot_refs(state)
            self._lnasa_sa = bool(state.lnasa_sa)
            self._control_names = list(state.control_names)
            self._surface_infos = list(_build_surface_map(model, state).values())

            self.add_input("alpha", val=0.0, units="deg")
            self.add_input("beta", val=0.0, units="deg")
            self.add_input("pb2v", val=0.0)
            self.add_input("qc2v", val=0.0)
            self.add_input("rb2v", val=0.0)
            self.add_input("mach", val=0.0)

            for name in self._control_names:
                self.add_input(name, val=0.0, units="deg")

            for info in self._surface_infos:
                self.add_input(f"{info.name}:aincs", val=info.baseline_aincs_deg, units="deg")
                self.add_input(f"{info.name}:chords", val=info.baseline_chords)
                self.add_input(f"{info.name}:xles", val=info.baseline_xles)
                self.add_input(f"{info.name}:yles", val=info.baseline_yles)
                self.add_input(f"{info.name}:zles", val=info.baseline_zles)

            self.add_output("CL", val=0.0)
            self.add_output("CD", val=0.0)
            self.add_output("CY", val=0.0)
            self.add_output("Cl", val=0.0)
            self.add_output("Cm", val=0.0)
            self.add_output("Cn", val=0.0)

            self.declare_partials(of="*", wrt="*")

        def compute(self, inputs: Any, outputs: Any) -> None:
            flow = _inputs_to_flow(self, inputs)
            params = _inputs_to_design_params(self, inputs)
            result = run_analysis_with_geometry(
                flow, params, self._topo, self._baseline, self._refs
            )
            cl, cd, cy, roll, pitch, yaw = _stability_outputs(result, flow, self._lnasa_sa)
            outputs["CL"] = cl
            outputs["CD"] = cd
            outputs["CY"] = cy
            outputs["Cl"] = roll
            outputs["Cm"] = pitch
            outputs["Cn"] = yaw

        def compute_jacvec_product(self, inputs: Any, d_inputs: Any, d_outputs: Any, mode: str) -> None:
            flow = _inputs_to_flow(self, inputs)
            params = _inputs_to_design_params(self, inputs)

            def _analysis(flow_in: FlowCondition, params_in: GeometryDesignParams) -> jnp.ndarray:
                result = run_analysis_with_geometry(
                    flow_in, params_in, self._topo, self._baseline, self._refs
                )
                return _stability_outputs_jax(result, flow_in, self._lnasa_sa)

            if mode == "rev":
                out_bar = jnp.array(
                    [
                        float(np.asarray(d_outputs[name]).ravel()[0])
                        for name in ("CL", "CD", "CY", "Cl", "Cm", "Cn")
                    ],
                    dtype=jnp.float64,
                )
                _, vjp_fn = jax.vjp(_analysis, flow, params)
                flow_bar, params_bar = vjp_fn(out_bar)
                _scatter_cotangents(self, flow_bar, params_bar, d_inputs)
            elif mode == "fwd":
                flow_tan = _inputs_to_flow_tangent(self, d_inputs)
                params_tan = _inputs_to_design_params_tangent(self, d_inputs)
                # ``jax.jvp`` is unavailable through the custom_vjp linear solve; use
                # grad-dot-tangent (equivalent to one row of J @ v per output).
                for idx, out_name in enumerate(("CL", "CD", "CY", "Cl", "Cm", "Cn")):
                    def _output_i(
                        flow_in: FlowCondition,
                        params_in: GeometryDesignParams,
                        i: int = idx,
                    ) -> jnp.ndarray:
                        return _analysis(flow_in, params_in)[i]

                    grad_flow, grad_params = jax.grad(_output_i, argnums=(0, 1))(flow, params)
                    delta = _pytree_dot(grad_flow, flow_tan) + _pytree_dot(grad_params, params_tan)
                    d_outputs[out_name] += float(delta)

    class OpenAVLGroup(om.Group):
        """OpenMDAO group exposing geometry design variables for JAX AVL analysis."""

        def initialize(self) -> None:
            self.options.declare("geo_file", types=str)
            self.options.declare("mass_file", default=None, types=(str, type(None)))

        def setup(self) -> None:
            self.add_subsystem(
                "avl",
                OpenAVLComp(
                    geo_file=self.options["geo_file"],
                    mass_file=self.options["mass_file"],
                ),
                promotes=["*"],
            )

else:
    OpenAVLComp = None  # type: ignore[misc, assignment]
    OpenAVLGroup = None  # type: ignore[misc, assignment]


def _pytree_dot(a: Any, b: Any) -> float:
    """Inner product of two matching JAX pytrees."""
    return float(
        sum(
            jnp.sum(x * y)
            for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b))
        )
    )


def _dval(d_inputs: Any, name: str) -> float:
    """Read one scalar tangent from an OpenMDAO input vector."""
    if name not in d_inputs:
        return 0.0
    return float(np.asarray(d_inputs[name], dtype=np.float64).ravel()[0])


def _inputs_to_flow_tangent(comp: Any, d_inputs: Any) -> FlowCondition:
    """Build a tangent :class:`FlowCondition` from OpenMDAO ``d_inputs``."""
    delcon = np.array(
        [
            _dval(d_inputs, name) * (np.pi / 180.0)
            for name in comp._control_names
        ],
        dtype=np.float64,
    )
    return FlowCondition(
        alfa=_dval(d_inputs, "alpha") * (np.pi / 180.0),
        beta=_dval(d_inputs, "beta") * (np.pi / 180.0),
        wrot=np.asarray(
            [
                _dval(d_inputs, "pb2v"),
                _dval(d_inputs, "qc2v"),
                _dval(d_inputs, "rb2v"),
            ],
            dtype=np.float64,
        ),
        mach=_dval(d_inputs, "mach"),
        delcon=delcon,
    )


def _inputs_to_design_params_tangent(comp: Any, d_inputs: Any) -> GeometryDesignParams:
    """Build a tangent :class:`GeometryDesignParams` from OpenMDAO ``d_inputs``."""
    aincs: list[float] = []
    chords: list[float] = []
    xles: list[float] = []
    yles: list[float] = []
    zles: list[float] = []
    for info in comp._surface_infos:
        key = f"{info.name}:aincs"
        aincs.extend(
            np.asarray(d_inputs[key], dtype=np.float64) * (np.pi / 180.0)
            if key in d_inputs
            else np.zeros(info.nsec)
        )
        for key, dest in (
            (f"{info.name}:chords", chords),
            (f"{info.name}:xles", xles),
            (f"{info.name}:yles", yles),
            (f"{info.name}:zles", zles),
        ):
            dest.extend(
                np.asarray(d_inputs[key], dtype=np.float64)
                if key in d_inputs
                else np.zeros(info.nsec)
            )
    return GeometryDesignParams(
        aincs=np.asarray(aincs, dtype=np.float64),
        chords=np.asarray(chords, dtype=np.float64),
        xles=np.asarray(xles, dtype=np.float64),
        yles=np.asarray(yles, dtype=np.float64),
        zles=np.asarray(zles, dtype=np.float64),
    )


__all__ = ["OpenAVLComp", "OpenAVLGroup", "SurfInfo"]
