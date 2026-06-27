"""AVL-compatible reported force and moment coefficients.

Internal force integration stores geometry-axis ``cmtot`` and ``cftot``. AVL's
``AERO`` routine applies the NASA stability-axis sign convention (``GETSA`` /
``DIR``) before printing FT totals or exposing ``CRBAX`` / ``CRSAX``.
"""

from __future__ import annotations

import math
from typing import Any

from openavl.core.state import AVLState


def nasa_dir(state: AVLState) -> float:
    """Return AVL ``DIR`` for the active stability-axis orientation flag."""
    return -1.0 if state.lnasa_sa else 1.0


def reported_totals(state: AVLState) -> dict[str, Any]:
    """Map internal totals to AVL FT / reported coefficients.

    Body-axis moments ``CM`` match AVL ``Cltot``, ``Cmtot``, ``Cntot`` and
    ``Cl``, ``Cm``, ``Cn``. Stability-axis moments ``CM_sa`` match AVL
    ``Cl'tot``, ``Cmtot``, ``Cn'tot`` and  ``Cl'``, ``Cm``, ``Cn'``.
    """
    dir_ = nasa_dir(state)
    ca = math.cos(float(state.alfa))
    sa = math.sin(float(state.alfa))

    cr = float(state.cmtot[0])
    cm = float(state.cmtot[1])
    cn = float(state.cmtot[2])
    cx = float(state.cftot[0])
    cy = float(state.cftot[1])
    cz = float(state.cftot[2])

    return {
        "CF": [dir_ * cx, cy, dir_ * cz],
        "CM": [dir_ * cr, cm, dir_ * cn],
        "CM_sa": [
            dir_ * (cr * ca + cn * sa),
            cm,
            dir_ * (cn * ca - cr * sa),
        ],
    }
