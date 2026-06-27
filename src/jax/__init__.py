"""JAX automatic differentiation backend for OpenAVL."""

from openavl.jax.aic import srdset_jax, vsrd_jax, vvor_jax
from openavl.jax.analysis import make_run_analysis_jit, run_analysis, run_analysis_jit
from openavl.jax.forces import compute_forces
from openavl.jax.freestream import vinfab
from openavl.jax.setup import build_rhs, compute_circulation, compute_velocities
from openavl.jax.snapshot import (
    snapshot_analysis_geometry,
    snapshot_circulation_geometry,
    snapshot_flow,
    snapshot_geometry,
    snapshot_refs,
)
from openavl.jax.solve import solve_circulation
from openavl.jax.solver import JaxAVLSolver
from openavl.jax.types import (
    AnalysisGeometry,
    AnalysisResult,
    CirculationGeometry,
    FlowCondition,
    GeometryArrays,
    GeometryStripMap,
    GeometryDesignParams,
    GeometryTopology,
    ReferenceQuantities,
    StripMap,
)
from openavl.jax.vortex import vorvelc_jax

__all__ = [
    "AnalysisGeometry",
    "AnalysisResult",
    "CirculationGeometry",
    "FlowCondition",
    "GeometryArrays",
    "GeometryStripMap",
    "JaxAVLSolver",
    "ReferenceQuantities",
    "StripMap",
    "build_rhs",
    "compute_circulation",
    "compute_forces",
    "compute_velocities",
    "run_analysis",
    "make_run_analysis_jit",
    "run_analysis_jit",
    "snapshot_analysis_geometry",
    "snapshot_circulation_geometry",
    "snapshot_flow",
    "snapshot_geometry",
    "snapshot_refs",
    "solve_circulation",
    "srdset_jax",
    "vorvelc_jax",
    "vinfab",
    "vvor_jax",
    "vsrd_jax",
]

try:
    from openavl.jax.openmdao import JaxAVLComp
except ImportError:  # pragma: no cover
    JaxAVLComp = None
else:
    __all__.append("JaxAVLComp")

try:
    from openavl.jax.openmdao_group import OpenAVLGroup
except ImportError:  # pragma: no cover
    OpenAVLGroup = None
else:
    __all__.append("OpenAVLGroup")

try:
    from openavl.jax.geom_jax import (
        design_params_from_state,
        run_analysis_with_geometry,
        snapshot_topology,
        update_geometry,
    )
except ImportError:  # pragma: no cover
    pass
else:
    __all__.extend(
        [
            "GeometryDesignParams",
            "GeometryTopology",
            "design_params_from_state",
            "run_analysis_with_geometry",
            "snapshot_topology",
            "update_geometry",
        ]
    )

