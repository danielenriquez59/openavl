"""AVL solver state (replaces Fortran COMMON blocks)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from openavl import constants as C
from openavl.fileio.parser import AVLModel


@dataclass
class AVLState:
    """All solver arrays sized dynamically to the parsed model."""

    model: AVLModel | None = None

    # Dimensions
    nvor: int = 0
    nvmax: int = 0
    nstrip: int = 0
    nstrmax: int = 0
    nsurf: int = 0
    nsurfmax: int = 0
    ncontrol: int = 0
    ndesign: int = 0
    nbody: int = 0
    nlnode: int = 0
    nlmax: int = 0
    numax: int = C.NUMAX
    ndmax: int = C.NDMAX
    ngmax: int = C.NGMAX
    nvtot: int = C.IVTOT
    nctot: int = C.ICTOT

    # Physical constants / flags
    pi: float = np.pi
    dtr: float = np.pi / 180.0
    unitl: float = 1.0
    unitm: float = 1.0
    unitt: float = 1.0
    unitf: float = 1.0
    units: float = 1.0
    unitv: float = 1.0
    unita: float = 1.0
    uniti: float = 1.0
    unitd: float = 1.0
    gee0: float = 1.0
    rho0: float = 1.0
    rmass0: float = 1.0
    xyzmass0: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    riner0: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    iysym: int = 0
    izsym: int = 0
    ysym: float = 0.0
    zsym: float = 0.0
    vrcorec: float = 0.0
    vrcorew: float = 2.0
    srcore: float = 1.0
    saxfr: float = 0.25

    lgeo: bool = False
    lenc: bool = False
    laic: bool = False
    lsrd: bool = False
    lvel: bool = False
    lsol: bool = False
    lsen: bool = False
    lvisc: bool = False
    lbfforce: bool = False
    lmast: bool = False
    lmass: bool = False
    ltrforce: bool = False
    lnfld_wv: bool = False
    lnasa_sa: bool = True
    lsa_rates: bool = False

    alfa: float = 0.0
    beta: float = 0.0
    mach: float = 0.0
    amach: float = 0.0
    betm: float = 1.0
    sref: float = 1.0
    cref: float = 1.0
    bref: float = 1.0
    cdref: float = 0.0
    xyzref: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    vinf: np.ndarray   = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    vinf_a: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    vinf_b: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    wrot: np.ndarray   = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    parval: np.ndarray = field(default_factory=lambda: np.zeros((C.IPTOT + 1, 1), dtype=np.float64))
    conval: np.ndarray = field(default_factory=lambda: np.zeros((C.ICMAX + 1, 1), dtype=np.float64))
    icon: np.ndarray   = field(default_factory=lambda: np.zeros((C.IVMAX + 1, 1), dtype=np.int32))
    delcon: np.ndarray = field(default_factory=lambda: np.zeros(C.NDMAX + 1, dtype=np.float64))
    deldes: np.ndarray = field(default_factory=lambda: np.zeros(C.NGMAX + 1, dtype=np.float64))

    # Surface indexing
    imags: np.ndarray    = field(default_factory=lambda: np.zeros(1, dtype=np.int32))
    ifrst: np.ndarray    = field(default_factory=lambda: np.zeros(1, dtype=np.int32))
    jfrst: np.ndarray    = field(default_factory=lambda: np.zeros(1, dtype=np.int32))
    nk: np.ndarray       = field(default_factory=lambda: np.zeros(1, dtype=np.int32))
    nj: np.ndarray       = field(default_factory=lambda: np.zeros(1, dtype=np.int32))
    lfwake: np.ndarray   = field(default_factory=lambda: np.ones(1, dtype=np.int32))
    lfload: np.ndarray   = field(default_factory=lambda: np.ones(1, dtype=np.int32))
    lncomp: np.ndarray   = field(default_factory=lambda: np.zeros(1, dtype=np.int32))
    ssurf: np.ndarray    = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    cavesurf: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    clmax_surf: np.ndarray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64)
    )  # per-surface CLmax from Wing.clmax; 0 disables sectional capping

    # Strip geometry
    ijfrst: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int32))
    nvstrp: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int32))
    vortex_to_strip: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int32))
    lssurf: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int32))
    rle1: np.ndarray   = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    rle2: np.ndarray   = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    rle: np.ndarray    = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    chord1: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    chord2: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    chord: np.ndarray  = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    wstrip: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    tanle: np.ndarray  = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    tante: np.ndarray  = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    ainc: np.ndarray   = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    ainc_g: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float64))
    clcd: np.ndarray   = field(default_factory=lambda: np.zeros((1, 6), dtype=np.float64))
    lviscstrp: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=bool))
    lstripoff: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=bool))
    ess: np.ndarray   = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    ensy: np.ndarray  = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    ensz: np.ndarray  = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    xsref: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    ysref: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    zsref: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))

    # Vortex lattice
    rv1: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    rv2: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    rv: np.ndarray  = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    rc: np.ndarray  = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    rs: np.ndarray  = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    dxv: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    chordv: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    slopec: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    slopev: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    lvcomp: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int32))
    lvnc: np.ndarray   = field(default_factory=lambda: np.zeros(1, dtype=bool))
    lvalbe: np.ndarray = field(default_factory=lambda: np.ones(1, dtype=bool))

    # CPOML / upper-lower surface geometry (AVL VRTX_S common block)
    xyn1: np.ndarray = field(default_factory=lambda: np.zeros((2, 1), dtype=np.float64))
    xyn2: np.ndarray = field(default_factory=lambda: np.zeros((2, 1), dtype=np.float64))
    zlon1: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    zlon2: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    zupn1: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    zupn2: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    cpt: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    ainc1: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    ainc2: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    lrange: np.ndarray = field(default_factory=lambda: np.ones(1, dtype=bool))

    dcontrol: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float64))
    vhinge: np.ndarray   = field(default_factory=lambda: np.zeros((3, 1, 1), dtype=np.float64))
    vrefl: np.ndarray    = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float64))
    phinge: np.ndarray   = field(default_factory=lambda: np.zeros((3, 1, 1), dtype=np.float64))

    enc: np.ndarray   = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    env: np.ndarray   = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    enc_d: np.ndarray = field(default_factory=lambda: np.zeros((3, 1, 1), dtype=np.float64))
    env_d: np.ndarray = field(default_factory=lambda: np.zeros((3, 1, 1), dtype=np.float64))
    enc_g: np.ndarray = field(default_factory=lambda: np.zeros((3, 1, 1), dtype=np.float64))

    # Body geometry
    lfrst: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int32))
    nl: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int32))
    rl: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    radl: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))

    # Linear system
    aicn: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float64))
    iapiv: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int32))
    work: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    gam: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    gam_u_0: np.ndarray = field(default_factory=lambda: np.zeros((1, C.NUMAX), dtype=np.float64))
    gam_u_d: np.ndarray = field(default_factory=lambda: np.zeros((1, C.NUMAX, 1), dtype=np.float64))
    gam_u_g: np.ndarray = field(default_factory=lambda: np.zeros((1, C.NUMAX, 1), dtype=np.float64))
    gam_u: np.ndarray = field(default_factory=lambda: np.zeros((1, C.NUMAX), dtype=np.float64))
    gam_d: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float64))
    gam_g: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float64))
    wc_gam: np.ndarray = field(default_factory=lambda: np.zeros((3, 1, 1), dtype=np.float64))
    wv_gam: np.ndarray = field(default_factory=lambda: np.zeros((3, 1, 1), dtype=np.float64))
    wcsrd_u: np.ndarray = field(default_factory=lambda: np.zeros((3, 1, C.NUMAX), dtype=np.float64))
    wvsrd_u: np.ndarray = field(default_factory=lambda: np.zeros((3, 1, C.NUMAX), dtype=np.float64))
    wcsrd: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    wvsrd: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    src_u: np.ndarray = field(default_factory=lambda: np.zeros((1, C.NUMAX), dtype=np.float64))
    dbl_u: np.ndarray = field(default_factory=lambda: np.zeros((3, 1, C.NUMAX), dtype=np.float64))
    src: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    dbl: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    vc: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    vv: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    wc: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    wv: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    vc_u: np.ndarray = field(default_factory=lambda: np.zeros((3, 1, C.NUMAX), dtype=np.float64))
    vv_u: np.ndarray = field(default_factory=lambda: np.zeros((3, 1, C.NUMAX), dtype=np.float64))
    wc_u: np.ndarray = field(default_factory=lambda: np.zeros((3, 1, C.NUMAX), dtype=np.float64))
    wv_u: np.ndarray = field(default_factory=lambda: np.zeros((3, 1, C.NUMAX), dtype=np.float64))
    vc_d: np.ndarray = field(default_factory=lambda: np.zeros((3, 1, 1), dtype=np.float64))
    vv_d: np.ndarray = field(default_factory=lambda: np.zeros((3, 1, 1), dtype=np.float64))
    wc_d: np.ndarray = field(default_factory=lambda: np.zeros((3, 1, 1), dtype=np.float64))
    wv_d: np.ndarray = field(default_factory=lambda: np.zeros((3, 1, 1), dtype=np.float64))
    vc_g: np.ndarray = field(default_factory=lambda: np.zeros((3, 1, 1), dtype=np.float64))
    vv_g: np.ndarray = field(default_factory=lambda: np.zeros((3, 1, 1), dtype=np.float64))
    wc_g: np.ndarray = field(default_factory=lambda: np.zeros((3, 1, 1), dtype=np.float64))
    wv_g: np.ndarray = field(default_factory=lambda: np.zeros((3, 1, 1), dtype=np.float64))

    # Force results
    cltot: float = 0.0
    cdtot: float = 0.0
    cytot: float = 0.0
    cdvtot: float = 0.0
    cltot_a: float = 0.0
    cdtot_a: float = 0.0
    cytot_a: float = 0.0
    cmtot: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    cftot: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    cltot_u: np.ndarray = field(default_factory=lambda: np.zeros(C.NUMAX, dtype=np.float64))
    cdtot_u: np.ndarray = field(default_factory=lambda: np.zeros(C.NUMAX, dtype=np.float64))
    cytot_u: np.ndarray = field(default_factory=lambda: np.zeros(C.NUMAX, dtype=np.float64))
    cftot_u: np.ndarray = field(default_factory=lambda: np.zeros((3, C.NUMAX), dtype=np.float64))
    cmtot_u: np.ndarray = field(default_factory=lambda: np.zeros((3, C.NUMAX), dtype=np.float64))
    cltot_d: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    cdtot_d: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    cytot_d: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    cftot_d: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    cmtot_d: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    clstrp: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    cdstrp: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    cystrp: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    chinge: np.ndarray = field(default_factory=lambda: np.zeros(C.NDMAX + 1, dtype=np.float64))
    chinge_u: np.ndarray = field(default_factory=lambda: np.zeros((1, C.NUMAX), dtype=np.float64))
    chinge_d: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float64))
    chinge_g: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float64))
    dcp: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    dcp_u: np.ndarray = field(default_factory=lambda: np.zeros((1, C.NUMAX), dtype=np.float64))
    dcp_d: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float64))
    dcp_g: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float64))
    dcpb: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    cnc: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    cnc_u: np.ndarray = field(default_factory=lambda: np.zeros((1, C.NUMAX), dtype=np.float64))
    cnc_d: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float64))
    cnc_g: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float64))
    cf_lstrp: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    cm_lstrp: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    cfstrp: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    cmstrp: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    cdst_a: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    cyst_a: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    clst_a: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    cdst_u: np.ndarray = field(default_factory=lambda: np.zeros((1, C.NUMAX), dtype=np.float64))
    cyst_u: np.ndarray = field(default_factory=lambda: np.zeros((1, C.NUMAX), dtype=np.float64))
    clst_u: np.ndarray = field(default_factory=lambda: np.zeros((1, C.NUMAX), dtype=np.float64))
    cfst_u: np.ndarray = field(default_factory=lambda: np.zeros((3, 1, C.NUMAX), dtype=np.float64))
    cmst_u: np.ndarray = field(default_factory=lambda: np.zeros((3, 1, C.NUMAX), dtype=np.float64))
    cdst_d: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float64))
    cyst_d: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float64))
    clst_d: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float64))
    cfst_d: np.ndarray = field(default_factory=lambda: np.zeros((3, 1, 1), dtype=np.float64))
    cmst_d: np.ndarray = field(default_factory=lambda: np.zeros((3, 1, 1), dtype=np.float64))
    cl_lstrp: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    cd_lstrp: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    cmc4_lstrp: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    ca_lstrp: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    cn_lstrp: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    clt_lstrp: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    cla_lstrp: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    cmle_lstrp: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    cdv_lstrp: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    cf_lsrf: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    cm_lsrf: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    cl_lsrf: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    cd_lsrf: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    cdsurf: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    cysurf: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    clsurf: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    cfsurf: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    cmsurf: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    cdvsurf: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    cds_a: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    cys_a: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    cls_a: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    cds_u: np.ndarray = field(default_factory=lambda: np.zeros((1, C.NUMAX), dtype=np.float64))
    cys_u: np.ndarray = field(default_factory=lambda: np.zeros((1, C.NUMAX), dtype=np.float64))
    cls_u: np.ndarray = field(default_factory=lambda: np.zeros((1, C.NUMAX), dtype=np.float64))
    cfs_u: np.ndarray = field(default_factory=lambda: np.zeros((3, 1, C.NUMAX), dtype=np.float64))
    cms_u: np.ndarray = field(default_factory=lambda: np.zeros((3, 1, C.NUMAX), dtype=np.float64))
    cds_d: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float64))
    cys_d: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float64))
    cls_d: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float64))
    cfs_d: np.ndarray = field(default_factory=lambda: np.zeros((3, 1, 1), dtype=np.float64))
    cms_d: np.ndarray = field(default_factory=lambda: np.zeros((3, 1, 1), dtype=np.float64))
    cdbdy: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    cybdy: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    clbdy: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    cfbdy: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    cmbdy: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    clff: float = 0.0
    cyff: float = 0.0
    cdff: float = 0.0
    spanef: float = 0.0
    spanef_a: float = 0.0
    clff_u: np.ndarray = field(default_factory=lambda: np.zeros(C.NUMAX, dtype=np.float64))
    cyff_u: np.ndarray = field(default_factory=lambda: np.zeros(C.NUMAX, dtype=np.float64))
    cdff_u: np.ndarray = field(default_factory=lambda: np.zeros(C.NUMAX, dtype=np.float64))
    spanef_u: np.ndarray = field(default_factory=lambda: np.zeros(C.NUMAX, dtype=np.float64))
    clff_d: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    cyff_d: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    cdff_d: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    spanef_d: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    clff_g: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    cyff_g: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    cdff_g: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    spanef_g: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    dwwake: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    env_d: np.ndarray = field(default_factory=lambda: np.zeros((3, 1, 1), dtype=np.float64))
    # env_g is allocated but never populated — the derivative of bound-vortex midpoint normals
    # w.r.t. design variables is not yet computed. This means dcp_g is missing the
    # DOT(ENV_G, FGAM) contribution (see aero.f L507). To fix, compute env_g in geometry.py
    # alongside env_d. Note: the AVL Fortran reference (amake.f L1143) contains a copy-paste
    # bug where ENC (control-point normal) is used instead of ENV (vortex-midpoint normal) in
    # the EMAG_G dot product — the correct implementation should use ENV there.
    env_g: np.ndarray = field(default_factory=lambda: np.zeros((3, 1, 1), dtype=np.float64))
    vv_d: np.ndarray = field(default_factory=lambda: np.zeros((3, 1, 1), dtype=np.float64))
    vv_g: np.ndarray = field(default_factory=lambda: np.zeros((3, 1, 1), dtype=np.float64))
    wv_d: np.ndarray = field(default_factory=lambda: np.zeros((3, 1, 1), dtype=np.float64))
    wv_g: np.ndarray = field(default_factory=lambda: np.zeros((3, 1, 1), dtype=np.float64))
    amass: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    ainer: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=np.float64))

    lcondef: np.ndarray = field(default_factory=lambda: np.zeros(C.NDMAX + 1, dtype=bool))
    ldesdef: np.ndarray = field(default_factory=lambda: np.zeros(C.NGMAX + 1, dtype=bool))
    # Pre-allocated SFFORC work buffers (zero-filled each call).
    sfforc_spn: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    sfforc_udrag: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    sfforc_udrag_u: np.ndarray = field(default_factory=lambda: np.zeros((3, C.NUMAX), dtype=np.float64))
    sfforc_ulift: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    sfforc_ulift_u: np.ndarray = field(default_factory=lambda: np.zeros((3, C.NUMAX), dtype=np.float64))
    sfforc_ulift_d: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    sfforc_ulift_g: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    sfforc_ulmag_u: np.ndarray = field(default_factory=lambda: np.zeros(C.NUMAX, dtype=np.float64))
    sfforc_rc4: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    sfforc_cfx_u: np.ndarray = field(default_factory=lambda: np.zeros(C.NUMAX, dtype=np.float64))
    sfforc_cfy_u: np.ndarray = field(default_factory=lambda: np.zeros(C.NUMAX, dtype=np.float64))
    sfforc_cfz_u: np.ndarray = field(default_factory=lambda: np.zeros(C.NUMAX, dtype=np.float64))
    sfforc_cmx_u: np.ndarray = field(default_factory=lambda: np.zeros(C.NUMAX, dtype=np.float64))
    sfforc_cmy_u: np.ndarray = field(default_factory=lambda: np.zeros(C.NUMAX, dtype=np.float64))
    sfforc_cmz_u: np.ndarray = field(default_factory=lambda: np.zeros(C.NUMAX, dtype=np.float64))
    sfforc_cfx_d: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    sfforc_cfy_d: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    sfforc_cfz_d: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    sfforc_cmx_d: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    sfforc_cmy_d: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    sfforc_cmz_d: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    sfforc_cfx_g: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    sfforc_cfy_g: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    sfforc_cfz_g: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    sfforc_cmx_g: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    sfforc_cmy_g: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    sfforc_cmz_g: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    sfforc_veff: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    sfforc_veff_u: np.ndarray = field(default_factory=lambda: np.zeros((3, C.NUMAX), dtype=np.float64))
    sfforc_veffmag_u: np.ndarray = field(default_factory=lambda: np.zeros(C.NUMAX, dtype=np.float64))
    control_names: list[str] = field(default_factory=list)
    var_names: list[str] = field(default_factory=list)
    con_names: list[str] = field(default_factory=list)

    @classmethod
    def from_model(cls, model: AVLModel, **options: Any) -> AVLState:
        """Allocate state arrays and set run-case defaults."""
        has_cdcl = any(
            sec.cdcl and len(sec.cdcl) >= 6 and sec.cdcl[3] != 0.0
            for surf in model.surfaces
            for sec in surf.sections
        )

        dup_count = sum(1 for surf in model.surfaces if surf.yduplicate is not None)
        nsurfmax = len(model.surfaces) + dup_count
        ncontrol = len(model.control_map)
        ndesign = 0
        ndmax = max(1, ncontrol)
        ngmax = max(1, ndesign)

        nstrip = 0
        nvor = 0
        for surf in model.surfaces:
            nvc = surf.n_chord or 1
            nvs = surf.n_span or 0
            if nvs == 0:
                nvs = sum(sec.n_span or 0 for sec in surf.sections[:-1])
            copies = 2 if surf.yduplicate is not None else 1
            nstrip += nvs * copies
            nvor += nvs * nvc * copies

        nvmax = max(1, nvor)
        nstrmax = max(1, nstrip)

        nlnode = 0
        nbody = 0
        for body in model.bodies:
            n_nodes = int(round(body.n_body or 0))
            if n_nodes < 2:
                continue
            if (
                not body.body_thread_x
                or len(body.body_thread_x) < 2
                or not body.body_thread_y
                or len(body.body_thread_y) < 2
                or not body.body_thread_t
                or len(body.body_thread_t) < 2
            ):
                continue
            nbody += 1
            nlnode += n_nodes
        nlmax = max(1, nlnode)

        opts = {
            "vel": 0.0,
            "rho": 1.225,
            "gravity": 9.81,
            "cl": 0.6,
            "bank": 0.0,
            "alpha": 2.0,
            "beta": 0.0,
            "cmx": 0.0,
            "cmy": 0.0,
            "cmz": 0.0,
            "cd0": 0.0,
            "xcg": None,
            "ycg": None,
            "zcg": None,
        }
        opts.update(options)

        unitl = float(options.get("unitl", 1.0))
        if unitl <= 0:
            unitl = 1.0

        state = cls(
            model=model,
            nvor=0,
            nvmax=nvmax,
            nstrip=0,
            nstrmax=nstrmax,
            nsurf=0,
            nsurfmax=nsurfmax,
            ncontrol=ncontrol,
            ndesign=ndesign,
            nbody=nbody,
            nlnode=nlnode,
            nlmax=nlmax,
            nvtot=C.IVTOT + ncontrol,
            unitl=unitl,
            iysym=model.header.iysym,
            izsym=model.header.izsym,
            zsym=model.header.zsym,
            lvisc=has_cdcl,
            mach=float(model.header.mach),
            sref=float(model.header.sref),
            cref=float(model.header.cref),
            bref=float(model.header.bref),
        )

        state._allocate_arrays(ndmax, ngmax, nstrmax, nvmax, nsurfmax)

        state.xyzref[:] = [
            model.header.xref,
            model.header.yref,
            model.header.zref,
        ]

        ir = 0
        state.parval[C.IPMACH, ir] = state.mach
        state.parval[C.IPVEE, ir]  = float(opts["vel"])
        state.parval[C.IPRHO, ir]  = float(opts["rho"])
        state.parval[C.IPGEE, ir]  = float(opts["gravity"])
        state.parval[C.IPCL, ir]   = float(opts["cl"])
        state.parval[C.IPPHI, ir]  = float(opts["bank"])
        xcg = opts["xcg"] if opts["xcg"] is not None else model.header.xref
        ycg = opts["ycg"] if opts["ycg"] is not None else model.header.yref
        zcg = opts["zcg"] if opts["zcg"] is not None else model.header.zref
        state.parval[C.IPXCG, ir] = float(xcg)
        state.parval[C.IPYCG, ir] = float(ycg)
        state.parval[C.IPZCG, ir] = float(zcg)
        state.parval[C.IPCD0, ir] = float(opts["cd0"])
        state.alfa = float(opts["alpha"]) * state.dtr
        state.beta = float(opts["beta"]) * state.dtr

        state.conval[C.ICCL, ir]   = float(opts["cl"])
        state.conval[C.ICMOMX, ir] = float(opts["cmx"])
        state.conval[C.ICMOMY, ir] = float(opts["cmy"])
        state.conval[C.ICMOMZ, ir] = float(opts["cmz"])
        state.conval[C.ICBETA, ir] = float(opts["beta"])
        state.conval[C.ICROTX, ir] = 0.0
        state.conval[C.ICROTY, ir] = 0.0
        state.conval[C.ICROTZ, ir] = 0.0

        state.icon[C.IVALFA, ir] = C.ICCL
        state.icon[C.IVBETA, ir] = C.ICBETA
        state.icon[C.IVROTX, ir] = C.ICROTX
        state.icon[C.IVROTY, ir] = C.ICROTY
        state.icon[C.IVROTZ, ir] = C.ICROTZ

        icmax = C.ICTOT + ndmax
        ivmax = C.IVTOT + ndmax
        for n in range(ndmax):
            iv = C.IVTOT + n
            ic = C.ICTOT + n
            state.icon[iv, ir] = ic
            state.conval[ic, ir] = 0.0

        for n in range(ncontrol):
            state.lcondef[n] = True
        for n in range(ndesign):
            state.ldesdef[n] = True

        inv_control = {idx: name for name, idx in model.control_map.items()}
        state.control_names = [inv_control[i] for i in range(ncontrol)]
        state.var_names = ["alpha", "beta", "pb/2V", "qc/2V", "rb/2V"] + state.control_names
        state.con_names = (
            ["alpha", "beta", "pb/2V", "qc/2V", "rb/2V", "CL", "CY", "Cl", "Cm", "Cn"]
            + state.control_names
        )

        return state

    def _allocate_arrays(
        self,
        ndmax: int,
        ngmax: int,
        nstrmax: int,
        nvmax: int,
        nsurfmax: int,
    ) -> None:
        self.imags = np.zeros(nsurfmax, dtype=np.int32)
        self.ifrst = np.zeros(nsurfmax, dtype=np.int32)
        self.jfrst = np.zeros(nsurfmax, dtype=np.int32)
        self.nk = np.zeros(nsurfmax, dtype=np.int32)
        self.nj = np.zeros(nsurfmax, dtype=np.int32)
        self.lfwake = np.ones(nsurfmax, dtype=np.int32)
        self.lfload = np.ones(nsurfmax, dtype=np.int32)
        self.lncomp = np.zeros(nsurfmax, dtype=np.int32)
        self.ssurf = np.zeros(nsurfmax, dtype=np.float64)
        self.cavesurf = np.zeros(nsurfmax, dtype=np.float64)
        self.clmax_surf = np.zeros(nsurfmax, dtype=np.float64)

        self.ijfrst = np.zeros(nstrmax, dtype=np.int32)
        self.nvstrp = np.zeros(nstrmax, dtype=np.int32)
        self.lssurf = np.zeros(nstrmax, dtype=np.int32)
        self.rle1 = np.zeros((3, nstrmax), dtype=np.float64)
        self.rle2 = np.zeros((3, nstrmax), dtype=np.float64)
        self.rle  = np.zeros((3, nstrmax), dtype=np.float64)
        self.chord1 = np.zeros(nstrmax, dtype=np.float64)
        self.chord2 = np.zeros(nstrmax, dtype=np.float64)
        self.chord = np.zeros(nstrmax, dtype=np.float64)
        self.wstrip = np.zeros(nstrmax, dtype=np.float64)
        self.tanle = np.zeros(nstrmax, dtype=np.float64)
        self.tante = np.zeros(nstrmax, dtype=np.float64)
        self.ainc = np.zeros(nstrmax, dtype=np.float64)
        self.ainc_g = np.zeros((nstrmax, ngmax), dtype=np.float64)
        self.clcd = np.zeros((nstrmax, 6), dtype=np.float64)
        self.lviscstrp = np.zeros(nstrmax, dtype=bool)
        self.lstripoff = np.zeros(nstrmax, dtype=bool)
        self.ess = np.zeros((3, nstrmax), dtype=np.float64)
        self.ensy = np.zeros(nstrmax, dtype=np.float64)
        self.ensz = np.zeros(nstrmax, dtype=np.float64)
        self.xsref = np.zeros(nstrmax, dtype=np.float64)
        self.ysref = np.zeros(nstrmax, dtype=np.float64)
        self.zsref = np.zeros(nstrmax, dtype=np.float64)

        self.rv1 = np.zeros((3, nvmax), dtype=np.float64)
        self.rv2 = np.zeros((3, nvmax), dtype=np.float64)
        self.rv = np.zeros((3, nvmax), dtype=np.float64)
        self.rc = np.zeros((3, nvmax), dtype=np.float64)
        self.rs = np.zeros((3, nvmax), dtype=np.float64)
        self.dxv = np.zeros(nvmax, dtype=np.float64)
        self.chordv = np.zeros(nvmax, dtype=np.float64)
        self.slopec = np.zeros(nvmax, dtype=np.float64)
        self.slopev = np.zeros(nvmax, dtype=np.float64)
        self.lvcomp = np.zeros(nvmax, dtype=np.int32)
        self.lvnc = np.zeros(nvmax, dtype=bool)
        self.lvalbe = np.ones(nvmax, dtype=bool)

        self.xyn1 = np.zeros((2, nvmax), dtype=np.float64)
        self.xyn2 = np.zeros((2, nvmax), dtype=np.float64)
        self.zlon1 = np.zeros(nvmax, dtype=np.float64)
        self.zlon2 = np.zeros(nvmax, dtype=np.float64)
        self.zupn1 = np.zeros(nvmax, dtype=np.float64)
        self.zupn2 = np.zeros(nvmax, dtype=np.float64)
        self.cpt = np.zeros(nvmax, dtype=np.float64)
        self.ainc1 = np.zeros(nstrmax, dtype=np.float64)
        self.ainc2 = np.zeros(nstrmax, dtype=np.float64)
        self.lrange = np.ones(nsurfmax, dtype=bool)

        self.dcontrol = np.zeros((nvmax, ndmax), dtype=np.float64)
        self.vhinge = np.zeros((3, nstrmax, ndmax), dtype=np.float64)
        self.vrefl = np.zeros((nstrmax, ndmax), dtype=np.float64)
        self.phinge = np.zeros((3, nstrmax, ndmax), dtype=np.float64)

        self.enc = np.zeros((3, nvmax), dtype=np.float64)
        self.env = np.zeros((3, nvmax), dtype=np.float64)
        self.enc_d = np.zeros((3, nvmax, ndmax), dtype=np.float64)
        self.env_d = np.zeros((3, nvmax, ndmax), dtype=np.float64)
        self.enc_g = np.zeros((3, nvmax, ngmax), dtype=np.float64)

        nlmax = max(1, self.nlmax)
        nbmax = max(1, self.nbody)
        self.lfrst = np.zeros(nbmax, dtype=np.int32)
        self.nl = np.zeros(nbmax, dtype=np.int32)
        self.rl = np.zeros((3, nlmax), dtype=np.float64)
        self.radl = np.zeros(nlmax, dtype=np.float64)

        self.numax = C.NUMAX
        self.ndmax = ndmax
        self.ngmax = ngmax

        self.aicn = np.zeros((nvmax, nvmax), dtype=np.float64)
        self.iapiv = np.zeros(nvmax, dtype=np.int32)
        self.work = np.zeros(nvmax, dtype=np.float64)
        self.gam = np.zeros(self.nvor if self.nvor > 0 else nvmax, dtype=np.float64)
        self.gam_u_0 = np.zeros((nvmax, C.NUMAX), dtype=np.float64)
        self.gam_u_d = np.zeros((nvmax, C.NUMAX, ndmax), dtype=np.float64)
        self.gam_u_g = np.zeros((nvmax, C.NUMAX, ngmax), dtype=np.float64)
        self.gam_u = np.zeros((nvmax, C.NUMAX), dtype=np.float64)
        self.gam_d = np.zeros((nvmax, ndmax), dtype=np.float64)
        self.gam_g = np.zeros((nvmax, ngmax), dtype=np.float64)
        self.wc_gam = np.zeros((3, nvmax, nvmax), dtype=np.float64)
        self.wv_gam = np.zeros((3, nvmax, nvmax), dtype=np.float64)
        self.wcsrd_u = np.zeros((3, nvmax, C.NUMAX), dtype=np.float64)
        self.wvsrd_u = np.zeros((3, nvmax, C.NUMAX), dtype=np.float64)
        self.wcsrd = np.zeros((3, nvmax), dtype=np.float64)
        self.wvsrd = np.zeros((3, nvmax), dtype=np.float64)
        self.src_u = np.zeros((nlmax, C.NUMAX), dtype=np.float64)
        self.dbl_u = np.zeros((3, nlmax, C.NUMAX), dtype=np.float64)
        self.src = np.zeros(nlmax, dtype=np.float64)
        self.dbl = np.zeros((3, nlmax), dtype=np.float64)
        self.vc = np.zeros((3, nvmax), dtype=np.float64)
        self.vv = np.zeros((3, nvmax), dtype=np.float64)
        self.wc = np.zeros((3, nvmax), dtype=np.float64)
        self.wv = np.zeros((3, nvmax), dtype=np.float64)
        self.vc_u = np.zeros((3, nvmax, C.NUMAX), dtype=np.float64)
        self.vv_u = np.zeros((3, nvmax, C.NUMAX), dtype=np.float64)
        self.wc_u = np.zeros((3, nvmax, C.NUMAX), dtype=np.float64)
        self.wv_u = np.zeros((3, nvmax, C.NUMAX), dtype=np.float64)
        self.vc_d = np.zeros((3, nvmax, ndmax), dtype=np.float64)
        self.vv_d = np.zeros((3, nvmax, ndmax), dtype=np.float64)
        self.wc_d = np.zeros((3, nvmax, ndmax), dtype=np.float64)
        self.wv_d = np.zeros((3, nvmax, ndmax), dtype=np.float64)
        self.vc_g = np.zeros((3, nvmax, ngmax), dtype=np.float64)
        self.vv_g = np.zeros((3, nvmax, ngmax), dtype=np.float64)
        self.wc_g = np.zeros((3, nvmax, ngmax), dtype=np.float64)
        self.wv_g = np.zeros((3, nvmax, ngmax), dtype=np.float64)

        numax = C.NUMAX
        self.clstrp = np.zeros(nstrmax, dtype=np.float64)
        self.cdstrp = np.zeros(nstrmax, dtype=np.float64)
        self.cystrp = np.zeros(nstrmax, dtype=np.float64)
        self.chinge = np.zeros(ndmax, dtype=np.float64)
        self.chinge_u = np.zeros((ndmax, numax), dtype=np.float64)
        self.chinge_d = np.zeros((ndmax, ndmax), dtype=np.float64)
        self.chinge_g = np.zeros((ndmax, ngmax), dtype=np.float64)
        self.delcon = np.zeros(ndmax, dtype=np.float64)
        self.dcp = np.zeros(nvmax, dtype=np.float64)
        self.dcp_u = np.zeros((nvmax, numax), dtype=np.float64)
        self.dcp_d = np.zeros((nvmax, ndmax), dtype=np.float64)
        self.dcp_g = np.zeros((nvmax, ngmax), dtype=np.float64)
        self.dcpb = np.zeros((3, nlmax), dtype=np.float64)
        self.cnc = np.zeros(nstrmax, dtype=np.float64)
        self.cnc_u = np.zeros((nstrmax, numax), dtype=np.float64)
        self.cnc_d = np.zeros((nstrmax, ndmax), dtype=np.float64)
        self.cnc_g = np.zeros((nstrmax, ngmax), dtype=np.float64)
        self.cf_lstrp = np.zeros((3, nstrmax), dtype=np.float64)
        self.cm_lstrp = np.zeros((3, nstrmax), dtype=np.float64)
        self.cfstrp = np.zeros((3, nstrmax), dtype=np.float64)
        self.cmstrp = np.zeros((3, nstrmax), dtype=np.float64)
        self.cdst_a = np.zeros(nstrmax, dtype=np.float64)
        self.cyst_a = np.zeros(nstrmax, dtype=np.float64)
        self.clst_a = np.zeros(nstrmax, dtype=np.float64)
        self.cdst_u = np.zeros((nstrmax, numax), dtype=np.float64)
        self.cyst_u = np.zeros((nstrmax, numax), dtype=np.float64)
        self.clst_u = np.zeros((nstrmax, numax), dtype=np.float64)
        self.cfst_u = np.zeros((3, nstrmax, numax), dtype=np.float64)
        self.cmst_u = np.zeros((3, nstrmax, numax), dtype=np.float64)
        self.cdst_d = np.zeros((nstrmax, ndmax), dtype=np.float64)
        self.cyst_d = np.zeros((nstrmax, ndmax), dtype=np.float64)
        self.clst_d = np.zeros((nstrmax, ndmax), dtype=np.float64)
        self.cfst_d = np.zeros((3, nstrmax, ndmax), dtype=np.float64)
        self.cmst_d = np.zeros((3, nstrmax, ndmax), dtype=np.float64)
        self.cl_lstrp = np.zeros(nstrmax, dtype=np.float64)
        self.cd_lstrp = np.zeros(nstrmax, dtype=np.float64)
        self.cmc4_lstrp = np.zeros(nstrmax, dtype=np.float64)
        self.ca_lstrp = np.zeros(nstrmax, dtype=np.float64)
        self.cn_lstrp = np.zeros(nstrmax, dtype=np.float64)
        self.clt_lstrp = np.zeros(nstrmax, dtype=np.float64)
        self.cla_lstrp = np.zeros(nstrmax, dtype=np.float64)
        self.cmle_lstrp = np.zeros(nstrmax, dtype=np.float64)
        self.cdv_lstrp = np.zeros(nstrmax, dtype=np.float64)
        self.cf_lsrf = np.zeros((3, nsurfmax), dtype=np.float64)
        self.cm_lsrf = np.zeros((3, nsurfmax), dtype=np.float64)
        self.cl_lsrf = np.zeros(nsurfmax, dtype=np.float64)
        self.cd_lsrf = np.zeros(nsurfmax, dtype=np.float64)
        self.cdsurf = np.zeros(nsurfmax, dtype=np.float64)
        self.cysurf = np.zeros(nsurfmax, dtype=np.float64)
        self.clsurf = np.zeros(nsurfmax, dtype=np.float64)
        self.cfsurf = np.zeros((3, nsurfmax), dtype=np.float64)
        self.cmsurf = np.zeros((3, nsurfmax), dtype=np.float64)
        self.cdvsurf = np.zeros(nsurfmax, dtype=np.float64)
        self.cds_a = np.zeros(nsurfmax, dtype=np.float64)
        self.cys_a = np.zeros(nsurfmax, dtype=np.float64)
        self.cls_a = np.zeros(nsurfmax, dtype=np.float64)
        self.cds_u = np.zeros((nsurfmax, numax), dtype=np.float64)
        self.cys_u = np.zeros((nsurfmax, numax), dtype=np.float64)
        self.cls_u = np.zeros((nsurfmax, numax), dtype=np.float64)
        self.cfs_u = np.zeros((3, nsurfmax, numax), dtype=np.float64)
        self.cms_u = np.zeros((3, nsurfmax, numax), dtype=np.float64)
        self.cds_d = np.zeros((nsurfmax, ndmax), dtype=np.float64)
        self.cys_d = np.zeros((nsurfmax, ndmax), dtype=np.float64)
        self.cls_d = np.zeros((nsurfmax, ndmax), dtype=np.float64)
        self.cfs_d = np.zeros((3, nsurfmax, ndmax), dtype=np.float64)
        self.cms_d = np.zeros((3, nsurfmax, ndmax), dtype=np.float64)
        self.cdbdy = np.zeros(nbmax, dtype=np.float64)
        self.cybdy = np.zeros(nbmax, dtype=np.float64)
        self.clbdy = np.zeros(nbmax, dtype=np.float64)
        self.cfbdy = np.zeros((3, nbmax), dtype=np.float64)
        self.cmbdy = np.zeros((3, nbmax), dtype=np.float64)
        self.cltot_u = np.zeros(numax, dtype=np.float64)
        self.cdtot_u = np.zeros(numax, dtype=np.float64)
        self.cytot_u = np.zeros(numax, dtype=np.float64)
        self.cftot_u = np.zeros((3, numax), dtype=np.float64)
        self.cmtot_u = np.zeros((3, numax), dtype=np.float64)
        self.cltot_d = np.zeros(ndmax, dtype=np.float64)
        self.cdtot_d = np.zeros(ndmax, dtype=np.float64)
        self.cytot_d = np.zeros(ndmax, dtype=np.float64)
        self.cftot_d = np.zeros((3, ndmax), dtype=np.float64)
        self.cmtot_d = np.zeros((3, ndmax), dtype=np.float64)
        self.clff_u = np.zeros(numax, dtype=np.float64)
        self.cyff_u = np.zeros(numax, dtype=np.float64)
        self.cdff_u = np.zeros(numax, dtype=np.float64)
        self.spanef_u = np.zeros(numax, dtype=np.float64)
        self.clff_d = np.zeros(ndmax, dtype=np.float64)
        self.cyff_d = np.zeros(ndmax, dtype=np.float64)
        self.cdff_d = np.zeros(ndmax, dtype=np.float64)
        self.spanef_d = np.zeros(ndmax, dtype=np.float64)
        self.clff_g = np.zeros(ngmax, dtype=np.float64)
        self.cyff_g = np.zeros(ngmax, dtype=np.float64)
        self.cdff_g = np.zeros(ngmax, dtype=np.float64)
        self.spanef_g = np.zeros(ngmax, dtype=np.float64)
        self.dwwake = np.zeros(nstrmax, dtype=np.float64)
        self.env_d = np.zeros((3, nvmax, ndmax), dtype=np.float64)
        self.env_g = np.zeros((3, nvmax, ngmax), dtype=np.float64)
        self.vv_d = np.zeros((3, nvmax, ndmax), dtype=np.float64)
        self.vv_g = np.zeros((3, nvmax, ngmax), dtype=np.float64)
        self.wv_d = np.zeros((3, nvmax, ndmax), dtype=np.float64)
        self.wv_g = np.zeros((3, nvmax, ngmax), dtype=np.float64)
        self.ldesdef = np.zeros(ngmax, dtype=bool)

        icmax = C.ICTOT + ndmax
        ivmax = C.IVTOT + ndmax
        self.parval = np.zeros((C.IPTOT + 1, 1), dtype=np.float64)
        self.conval = np.zeros((icmax + 1, 1), dtype=np.float64)
        self.icon = np.zeros((ivmax + 1, 1), dtype=np.int32)
        self.lcondef = np.zeros(ndmax, dtype=bool)

        self.vortex_to_strip = np.zeros(nvmax, dtype=np.int32)
        self.sfforc_udrag_u = np.zeros((3, numax), dtype=np.float64)
        self.sfforc_ulift_u = np.zeros((3, numax), dtype=np.float64)
        self.sfforc_ulift_d = np.zeros((3, ndmax), dtype=np.float64)
        self.sfforc_ulift_g = np.zeros((3, ngmax), dtype=np.float64)
        self.sfforc_ulmag_u = np.zeros(numax, dtype=np.float64)
        self.sfforc_cfx_u = np.zeros(numax, dtype=np.float64)
        self.sfforc_cfy_u = np.zeros(numax, dtype=np.float64)
        self.sfforc_cfz_u = np.zeros(numax, dtype=np.float64)
        self.sfforc_cmx_u = np.zeros(numax, dtype=np.float64)
        self.sfforc_cmy_u = np.zeros(numax, dtype=np.float64)
        self.sfforc_cmz_u = np.zeros(numax, dtype=np.float64)
        self.sfforc_cfx_d = np.zeros(ndmax, dtype=np.float64)
        self.sfforc_cfy_d = np.zeros(ndmax, dtype=np.float64)
        self.sfforc_cfz_d = np.zeros(ndmax, dtype=np.float64)
        self.sfforc_cmx_d = np.zeros(ndmax, dtype=np.float64)
        self.sfforc_cmy_d = np.zeros(ndmax, dtype=np.float64)
        self.sfforc_cmz_d = np.zeros(ndmax, dtype=np.float64)
        self.sfforc_cfx_g = np.zeros(ngmax, dtype=np.float64)
        self.sfforc_cfy_g = np.zeros(ngmax, dtype=np.float64)
        self.sfforc_cfz_g = np.zeros(ngmax, dtype=np.float64)
        self.sfforc_cmx_g = np.zeros(ngmax, dtype=np.float64)
        self.sfforc_cmy_g = np.zeros(ngmax, dtype=np.float64)
        self.sfforc_cmz_g = np.zeros(ngmax, dtype=np.float64)
        self.sfforc_veff_u = np.zeros((3, numax), dtype=np.float64)
        self.sfforc_veffmag_u = np.zeros(numax, dtype=np.float64)


def build_vortex_to_strip(state: AVLState) -> np.ndarray:
    """Map each vortex index to its parent strip index (one-time geometry setup)."""
    nvor = state.nvor
    v2s = np.zeros(max(1, nvor), dtype=np.int32)
    for j in range(state.nstrip):
        i1 = int(state.ijfrst[j])
        nvc = int(state.nvstrp[j])
        v2s[i1 : i1 + nvc] = j
    state.vortex_to_strip = v2s[:nvor] if nvor > 0 else v2s[:0]
    return state.vortex_to_strip
