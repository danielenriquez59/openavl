"""OpenAVL — numerically faithful Python port of AVL."""

from openavl.core.solver import AVLSolver
from openavl.core.state import AVLState
from openavl.fileio.parser import AVLHeader, AVLModel, parse_avl, parse_avl_file
from openavl.geometry import Aircraft

__all__ = [
    "AVLSolver",
    "Aircraft",
    "AVLModel",
    "AVLHeader",
    "AVLState",
    "StabilityDerivatives",
    "EigenAnalysisResult",
    "FlightMode",
    "parse_avl",
    "parse_avl_file",
    "JaxAVLSolver",
]

def __getattr__(name: str):
    """Lazily import optional exports so core solver imports stay lightweight."""
    if name in {"EigenAnalysisResult", "FlightMode"}:
        from openavl.analysis.amode import EigenAnalysisResult, FlightMode

        return {"EigenAnalysisResult": EigenAnalysisResult, "FlightMode": FlightMode}[name]
    if name == "StabilityDerivatives":
        from openavl.analysis.deriv import StabilityDerivatives

        return StabilityDerivatives
    if name == "JaxAVLSolver":
        from openavl.jax.solver import JaxAVLSolver

        return JaxAVLSolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
