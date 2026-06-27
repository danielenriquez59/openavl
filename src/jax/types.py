"""Immutable JAX PyTree containers for the differentiable AVL pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from jax import Array
else:
    Array = object  # noqa: UP018


class CirculationGeometry(NamedTuple):
    """Pre-computed lattice geometry for circulation solve (Phases 3A-3C)."""

    rc: Array  # [3, nvor]
    enc: Array  # [3, nvor]
    enc_d: Array  # [3, nvor, ncontrol]
    aicn: Array  # [nvor, nvor]
    wc_gam: Array  # [3, nvor, nvor]
    wv_gam: Array  # [3, nvor, nvor]
    wcsrd_u: Array  # [3, nvor, numax]
    lvnc: Array  # [nvor] bool
    lvalbe: Array  # [nvor] bool
    numax: int = 6
    ncontrol: int = 0


class GeometryStripMap(NamedTuple):
    """Strip index arrays for full geometry snapshots (Phase 1)."""

    vortex_to_strip: Array
    strip_to_surface: Array
    ijfrst: Array
    nvstrp: Array
    chord: Array
    ainc: Array
    lstripoff: Array
    lssurf: Array
    ess: Array
    ensy: Array
    ensz: Array


class GeometryArrays(NamedTuple):
    """Pre-computed geometry arrays (not differentiated in Phases 1-4)."""

    nvor: int
    nstrip: int
    nsurf: int
    nbody: int
    iysym: int
    izsym: int
    ysym: float
    zsym: float
    vrcorec: float
    vrcorew: float
    srcore: float
    betm: float
    rv1: Array
    rv2: Array
    rv: Array
    rc: Array
    enc: Array
    env: Array
    chord: Array
    chordv: Array
    lvcomp: Array
    wc_gam: Array
    wv_gam: Array
    aicn: Array
    strip_map: GeometryStripMap
    lfrst: Array
    nl: Array
    rl: Array
    radl: Array
    src_u: Array
    dbl_u: Array
    wcsrd_u: Array
    wvsrd_u: Array


class StripMap(NamedTuple):
    """Index arrays mapping vortices to strips and surfaces."""

    vortex_to_strip: Array
    ijfrst: Array
    nvstrp: Array
    strip_to_surface: Array
    jfrst: Array
    nj: Array
    lstripoff: Array
    lviscstrp: Array
    lssurf: Array
    lncomp: Array


class ForceGeometry(NamedTuple):
    """Strip/surface geometry for force integration."""

    rv1: Array
    rv2: Array
    rv: Array
    rc: Array
    env: Array
    dxv: Array
    rle: Array
    rle1: Array
    rle2: Array
    chord: Array
    chord1: Array
    chord2: Array
    wstrip: Array
    ensy: Array
    ensz: Array
    ess: Array
    ainc: Array
    xsref: Array
    ysref: Array
    zsref: Array
    ssurf: Array
    cavesurf: Array
    imags: Array
    lfload: Array
    clcd: Array
    strip_map: StripMap
    nbody: int


class BodyGeometry(NamedTuple):
    """Slender-body line geometry for source forces."""

    nl: Array
    lfrst: Array
    rl: Array
    radl: Array
    src: Array
    seg_i1: Array
    seg_i2: Array


class TrefftzGeometry(NamedTuple):
    """Trefftz-plane geometry for far-field drag integration."""

    rv1: Array
    rv2: Array
    rc: Array
    chord: Array
    strip_map: StripMap
    iysym: int
    izsym: int
    ysym: Array
    zsym: Array
    vrcorec: Array
    vrcorew: Array
    lfload: Array
    amach: Array


class GeometryDesignParams(NamedTuple):
    """Section-level design parameters differentiated in geometry optimization."""

    aincs: Array  # [total_sections] incidence angles (rad)
    chords: Array  # [total_sections] chord lengths
    xles: Array  # [total_sections] leading-edge x positions
    yles: Array  # [total_sections] leading-edge y positions
    zles: Array  # [total_sections] leading-edge z positions


class GeometryTopology(NamedTuple):
    """Fixed topology and interpolation data captured once at setup (not differentiated)."""

    nstrip: int
    nvor: int
    nsurf: int
    n_sections: int
    saxfr: float
    betm: float
    iysym: int
    izsym: int
    ysym: float
    zsym: float
    vrcorec: float
    vrcorew: float
    sec_left: Array  # [nstrip] global section index (left edge)
    sec_right: Array  # [nstrip] global section index (right edge)
    fc: Array  # [nstrip] chord-center interpolation fraction
    f1: Array  # [nstrip] left-edge interpolation fraction
    f2: Array  # [nstrip] right-edge interpolation fraction
    width: Array  # [nstrip] spanwise strip width
    tanle_slope: Array  # [nstrip] LE sweep slope (x/span)
    tante_slope: Array  # [nstrip] TE sweep slope (x/span)
    is_mirror: Array  # [nstrip] bool — Y-duplicated image strip
    ydup: Array  # [nstrip] YDUPLICATE plane offset
    imags_neg: Array  # [nstrip] bool — reversed edge ordering (IMAGS < 0)
    model_surf_idx: Array  # [nstrip] index into model.surfaces
    xyzscal_x: Array  # [nstrip] chordwise scale factor
    surf_sec_offset: Array  # [n_model_surf] offset into concatenated section arrays
    surf_nsec: Array  # [n_model_surf] section count per model surface
    surf_xyzscal: Array  # [n_model_surf, 3]
    surf_xyztran: Array  # [n_model_surf, 3]
    surf_addinc: Array  # [n_model_surf] surface incidence add-on (rad)
    vortex_to_strip: Array  # [nvor]
    xvr: Array  # [nvor] chordwise vortex fraction
    xcp: Array  # [nvor] chordwise control-point fraction
    slopec: Array  # [nvor] frozen camber slope at control point
    slopev: Array  # [nvor] frozen camber slope at vortex midpoint
    lvcomp: Array  # [nvor] component index per vortex
    lstripoff: Array  # [nstrip] bool
    lfwake: Array  # [nsurf] bool per solver surface
    jfrst: Array  # [nsurf]
    nj: Array  # [nsurf]
    ijfrst: Array  # [nstrip]
    nvstrp: Array  # [nstrip]
    kutta_iv: Array  # [n_kutta] trailing vortex index per Kutta row
    kutta_j1: Array  # [n_kutta] first bound vortex on strip
    kutta_j2: Array  # [n_kutta] last bound vortex on strip (Kutta row)
    stripoff_iv: Array  # [n_stripoff] identity-row vortex indices


class AnalysisGeometry(NamedTuple):
    """Bundled geometry for the full primal analysis pipeline."""

    circulation: CirculationGeometry
    force: ForceGeometry
    body: BodyGeometry
    trefftz: TrefftzGeometry


class FlowCondition(NamedTuple):
    """Differentiable flow inputs."""

    alfa: Array
    beta: Array
    wrot: Array
    delcon: Array
    mach: Array


class ReferenceQuantities(NamedTuple):
    """Reference lengths, moment center, and baseline drag."""

    sref: Array
    cref: Array
    bref: Array
    xyzref: Array
    cdref: Array = 0.0
    iysym: int = 0


class Velocities(NamedTuple):
    """Induced velocities at vortex and wake points."""

    vv: Array
    wv: Array


class StripForces(NamedTuple):
    """Per-strip force coefficients."""

    cdstrp: Array
    cystrp: Array
    clstrp: Array
    cfstrp: Array
    cmstrp: Array
    cdv_lstrp: Array
    cnc: Array
    dcp: Array


class SurfaceForces(NamedTuple):
    """Per-surface force coefficients."""

    cdsurf: Array
    cysurf: Array
    clsurf: Array
    cfsurf: Array
    cmsurf: Array
    cdvsurf: Array


class InviscidForces(NamedTuple):
    """Inviscid wing force integration result."""

    CL: Array
    CD: Array
    CY: Array
    CM: Array
    CF: Array
    CDV: Array
    strips: StripForces
    surfaces: SurfaceForces


class BodyForces(NamedTuple):
    """Slender-body force contributions."""

    CL: Array
    CD: Array
    CY: Array
    CM: Array
    CF: Array


class TrefftzForces(NamedTuple):
    """Trefftz-plane far-field force contributions."""

    CL: Array
    CY: Array
    CDi: Array
    spanef: Array
    dwwake: Array


class ForceResult(NamedTuple):
    """Total force coefficients including body, Trefftz, and viscous drag."""

    CL: Array
    CD: Array
    CY: Array
    CM: Array
    CF: Array
    CDV: Array
    CLFF: Array
    CYFF: Array
    CDFF: Array
    SPANEF: Array
    inviscid: InviscidForces | None = None
    body: BodyForces | None = None
    trefftz: TrefftzForces | None = None


class AnalysisResult(NamedTuple):
    """Aggregated force and moment coefficients."""

    CL: Array
    CD: Array
    CY: Array
    CM: Array
