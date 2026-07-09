"""Viscous drag polar lookup (port of cdcl.f).

Returns CD as a function of CL (drag polar). The polar is defined in four
pieces between and outside three CL,CD pairs:

1. Negative stall region (quadratic below CLneg)
2. Parabolic CD(CL) between negative stall and drag minimum
3. Parabolic CD(CL) between drag minimum and positive stall
4. Positive stall region (quadratic above CLpos)

The three CD,CL pairs are specified in ``cdclpol``:

- ``cdclpol[0]`` — CL (clneg) before negative stall drag rise
- ``cdclpol[1]`` — CD (cdneg) at clneg
- ``cdclpol[2]`` — CL (clcdmin) at minimum drag
- ``cdclpol[3]`` — CD (cdmin) at clcdmin
- ``cdclpol[4]`` — CL (clpos) before positive stall drag rise
- ``cdclpol[5]`` — CD (cdpos) at clpos
"""

from __future__ import annotations

import warnings

import numpy as np

F64 = np.float64

# Polar keys already warned (once per distinct breakpoint tuple), matching AVL's
# single "input CL data out of order" message rather than spamming every strip.
_warned_out_of_order: set[tuple[float, float, float, float, float, float]] = set()


def cdcl(cdclpol: np.ndarray, cl: float) -> tuple[np.float64, np.float64]:
    """Return CD and dCD/dCL for a piecewise-quadratic drag polar (CDCL).

    Degenerate polars (``clmax <= cl0`` or ``cl0 <= clmin``) return ``(0, 0)``
    with a one-time warning, matching AVL's graceful fallback instead of NaN.
    """
    clmin = F64(cdclpol[0])
    cdmin = F64(cdclpol[1])
    cl0 = F64(cdclpol[2])
    cd0 = F64(cdclpol[3])
    clmax = F64(cdclpol[4])
    cdmax = F64(cdclpol[5])

    if clmax <= cl0 or cl0 <= clmin:
        key = (
            float(clmin),
            float(cdmin),
            float(cl0),
            float(cd0),
            float(clmax),
            float(cdmax),
        )
        if key not in _warned_out_of_order:
            _warned_out_of_order.add(key)
            warnings.warn(
                "CDCL input CL data out of order; returning zero profile drag",
                stacklevel=2,
            )
        return F64(0.0), F64(0.0)

    # clinc and cdinc control drag rise rate in stall regions;
    # clinc=0.2 forces drag to increase by cdinc over deltacl=0.2 after stall
    clinc = F64(0.2)
    cdinc = F64(0.05)
    # Matching parameters, ported verbatim from AVL's cdcl.f (matches AVL 3.52
    # bit-for-bit, including cd_cl). Despite the "smooth at the stall joins"
    # comment inherited from AVL's own source, dCD/dCL is only continuous at
    # the CLMIN/CL0 and CLMAX/CL0 joins when |CLMIN-CL0| = 1 (respectively
    # |CLMAX-CL0| = 1): the stall-region cd_cl below matches the inner
    # parabola's slope cdx1 only in that special case. CD itself is always
    # continuous; only its slope can kink at typical (non-unit) CL spacing.
    # A slope-matched reformulation (replacing the quadratic stall-rise
    # coefficient so its join slope equals the parabola's) would restore true
    # C1 continuity, but is not applied here to keep bit-for-bit AVL parity.
    cdx1 = F64(2.0 * (cdmin - cd0) * (clmin - cl0) / ((clmin - cl0) * (clmin - cl0)))
    cdx2 = F64(2.0 * (cdmax - cd0) * (clmax - cl0) / ((clmax - cl0) * (clmax - cl0)))
    clfac = F64(1.0 / clinc)

    clv = F64(cl)
    if clv < clmin:
        # negative stall region; slope matches lower side, quadratic drag rise
        dcl = F64(clv - clmin)
        cd = F64(
            cdmin
            + F64(cdinc * F64(clfac * clfac)) * F64(dcl * dcl)
            + F64(cdx1 * F64(1.0 - F64((clv - cl0) / (clmin - cl0))))
        )
        cd_cl = F64(
            F64(cdinc * F64(clfac * clfac)) * F64(dcl * 2.0)
            - F64(cdx1 / (clmin - cl0))
        )
    elif clv < cl0:
        # lower quadratic; zero slope at min drag point
        dcl = F64(clv - cl0)
        cd = F64(cd0 + F64((cdmin - cd0) * F64(dcl * dcl) / F64((clmin - cl0) * (clmin - cl0))))
        cd_cl = F64((cdmin - cd0) * F64(dcl * 2.0) / F64((clmin - cl0) * (clmin - cl0)))
    elif clv < clmax:
        # upper quadratic; zero slope at min drag point
        dcl = F64(clv - cl0)
        cd = F64(cd0 + F64((cdmax - cd0) * F64(dcl * dcl) / F64((clmax - cl0) * (clmax - cl0))))
        cd_cl = F64((cdmax - cd0) * F64(dcl * 2.0) / F64((clmax - cl0) * (clmax - cl0)))
    else:
        # positive stall region; slope matches upper side, quadratic drag rise
        dcl = F64(clv - clmax)
        cd = F64(
            cdmax
            + F64(F64(cdinc * F64(clfac * clfac)) * F64(dcl * dcl))
            - F64(cdx2 * F64(1.0 - F64((clv - cl0) / (clmax - cl0))))
        )
        cd_cl = F64(
            F64(cdinc * F64(clfac * clfac)) * F64(dcl * 2.0)
            + F64(cdx2 / (clmax - cl0))
        )
    return cd, cd_cl
