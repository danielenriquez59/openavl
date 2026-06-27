"""User-facing programmatic aircraft geometry API.

Build aircraft without ``.avl`` files using builder methods on :class:`Aircraft`,
:class:`Wing`, and :class:`Section`, then pass the result to
:class:`~openavl.core.solver.AVLSolver`::

    from openavl import AVLSolver, Aircraft

    ac = Aircraft(name="Demo", sref=10.0, cref=1.0, bref=8.0)
    wing = ac.add_wing("Wing", n_chord=8, n_span=16, symmetric=True)
    wing.clmax = 1.2  # optional: cap sectional lift (0 = disabled)
    wing.add_section(xyzle=[0, 0, 0], chord=1.0).set_airfoil_naca("2412")
    wing.add_section(xyzle=[0.2, 4, 0], chord=0.6)
    solver = AVLSolver(ac, base_dir="/path/to/airfoils")
"""

from openavl.geometry.aircraft import Aircraft
from openavl.geometry.airfoil import Airfoil, AirfoilType
from openavl.geometry.body import Body
from openavl.geometry.cdcl_polar import CdclPolar
from openavl.geometry.control import ControlSurface
from openavl.geometry.section import Section
from openavl.geometry.wing import Wing

__all__ = [
    "Aircraft",
    "Wing",
    "Section",
    "ControlSurface",
    "Airfoil",
    "AirfoilType",
    "CdclPolar",
    "Body",
]
