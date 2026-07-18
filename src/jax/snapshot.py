"""Extract JAX PyTrees from a built NumPy ``AVLState``."""

from __future__ import annotations

from typing import Any

import numpy as np

from openavl.jax.backend import jnp
from openavl.jax.forces import (
    body_geometry_from_state,
    force_geometry_from_state,
    trefftz_geometry_from_state,
)
from openavl.jax.types import (
    AnalysisGeometry,
    CirculationGeometry,
    FlowCondition,
    GeometryArrays,
    GeometryStripMap,
    ReferenceQuantities,
)


def _build_vortex_to_strip(state: Any) -> np.ndarray:
    """Map each vortex index to its strip index."""
    vortex_to_strip = np.zeros(state.nvor, dtype=np.int32)
    for j in range(state.nstrip):
        i1 = int(state.ijfrst[j])
        for k in range(int(state.nvstrp[j])):
            vortex_to_strip[i1 + k] = j
    return vortex_to_strip


def _rebuild_aicn(state: Any) -> np.ndarray:
    """Reconstruct the unfactored AIC matrix (mirrors ``setup`` before ``ludcmp``)."""
    nvor = state.nvor
    aicn = np.zeros((nvor, nvor), dtype=np.float64)

    for i in range(nvor):
        for j in range(nvor):
            aicn[i, j] = (
                state.wc_gam[0, i, j] * state.enc[0, i]
                + state.wc_gam[1, i, j] * state.enc[1, i]
                + state.wc_gam[2, i, j] * state.enc[2, i]
            )

    for n in range(state.nsurf):
        if state.lfwake[n]:
            continue
        j1 = int(state.jfrst[n])
        jn = j1 + int(state.nj[n]) - 1
        for j in range(j1, jn + 1):
            i1 = int(state.ijfrst[j])
            iv = int(state.ijfrst[j] + state.nvstrp[j] - 1)
            aicn[iv, :nvor] = 0.0
            for jv in range(i1, iv + 1):
                aicn[iv, jv] = 1.0

    for j in range(state.nstrip):
        if not state.lstripoff[j]:
            continue
        i1 = int(state.ijfrst[j])
        for k in range(state.nvstrp[j]):
            ii = i1 + k
            aicn[ii, :nvor] = 0.0
            aicn[ii, ii] = 1.0

    return aicn


def snapshot_geometry(state: Any) -> GeometryArrays:
    """Extract full geometry arrays from ``AVLState`` for JAX pipelines."""
    nvor = state.nvor
    nstrip = state.nstrip
    nsurf = state.nsurf
    nbody = state.nbody
    nlmax = max(1, state.nlmax)

    strip_map = GeometryStripMap(
        vortex_to_strip=jnp.array(_build_vortex_to_strip(state)),
        strip_to_surface=jnp.array(state.lssurf[:nstrip]),
        ijfrst=jnp.array(state.ijfrst[:nstrip]),
        nvstrp=jnp.array(state.nvstrp[:nstrip]),
        chord=jnp.array(state.chord[:nstrip]),
        ainc=jnp.array(state.ainc[:nstrip]),
        lstripoff=jnp.array(state.lstripoff[:nstrip]),
        lssurf=jnp.array(state.lssurf[:nstrip]),
        ess=jnp.array(state.ess[:, :nstrip]),
        ensy=jnp.array(state.ensy[:nstrip]),
        ensz=jnp.array(state.ensz[:nstrip]),
    )

    betm = state.betm
    if betm == 0.0:
        betm = float(np.sqrt(1.0 - state.mach * state.mach))

    return GeometryArrays(
        nvor=nvor,
        nstrip=nstrip,
        nsurf=nsurf,
        nbody=nbody,
        iysym=int(state.iysym),
        izsym=int(state.izsym),
        ysym=float(state.ysym),
        zsym=float(state.zsym),
        vrcorec=float(state.vrcorec),
        vrcorew=float(state.vrcorew),
        srcore=float(state.srcore),
        betm=float(betm),
        rv1=jnp.array(state.rv1[:, :nvor]),
        rv2=jnp.array(state.rv2[:, :nvor]),
        rv=jnp.array(state.rv[:, :nvor]),
        rc=jnp.array(state.rc[:, :nvor]),
        enc=jnp.array(state.enc[:, :nvor]),
        env=jnp.array(state.env[:, :nvor]),
        chord=jnp.array(state.chord[:nstrip]),
        chordv=jnp.array(state.chordv[:nvor]),
        lvcomp=jnp.array(state.lvcomp[:nvor]),
        wc_gam=jnp.array(state.wc_gam[:, :nvor, :nvor]),
        wv_gam=jnp.array(state.wv_gam[:, :nvor, :nvor]),
        aicn=jnp.array(state.aicn[:nvor, :nvor]),
        strip_map=strip_map,
        lfrst=jnp.array(state.lfrst[: max(1, nbody)]),
        nl=jnp.array(state.nl[: max(1, nbody)]),
        rl=jnp.array(state.rl[:, :nlmax]),
        radl=jnp.array(state.radl[:nlmax]),
        src_u=jnp.array(state.src_u[:nlmax, :6]),
        dbl_u=jnp.array(state.dbl_u[:, :nlmax, :6]),
        wcsrd_u=jnp.array(state.wcsrd_u[:, :nvor, :6]),
        wvsrd_u=jnp.array(state.wvsrd_u[:, :nvor, :6]),
    )


def _kutta_stripoff_row_indices(state: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Row indices for the AIC's Kutta-condition and strip-off overrides.

    Mirrors the row selection in ``_rebuild_aicn`` above (and
    ``geom_jax._kutta_stripoff_indices``, kept separate here to avoid a
    circular import between ``snapshot`` and ``geom_jax``): each lifting
    surface's trailing-edge row enforces circulation continuity across the
    strip ``[i1, iv]``, and each strip-off row is an identity row.
    """
    kutta_iv: list[int] = []
    kutta_j1: list[int] = []
    kutta_j2: list[int] = []
    stripoff_iv: list[int] = []

    for n in range(state.nsurf):
        if state.lfwake[n]:
            continue
        j1 = int(state.jfrst[n])
        jn = j1 + int(state.nj[n]) - 1
        for j in range(j1, jn + 1):
            i1 = int(state.ijfrst[j])
            iv = int(state.ijfrst[j] + state.nvstrp[j] - 1)
            kutta_iv.append(iv)
            kutta_j1.append(i1)
            kutta_j2.append(iv)

    for j in range(state.nstrip):
        if not state.lstripoff[j]:
            continue
        i1 = int(state.ijfrst[j])
        for k in range(int(state.nvstrp[j])):
            stripoff_iv.append(i1 + k)

    return (
        np.asarray(kutta_iv, dtype=np.int32),
        np.asarray(kutta_j1, dtype=np.int32),
        np.asarray(kutta_j2, dtype=np.int32),
        np.asarray(stripoff_iv, dtype=np.int32),
    )


def snapshot_circulation_geometry(state: Any) -> CirculationGeometry:
    """Extract circulation-solve geometry arrays from ``AVLState``.

    In addition to the baked-in ``aicn``/``wc_gam``/``wv_gam`` (captured at
    ``state.mach``), this also captures the raw lattice geometry (``rv1``,
    ``rv2``, ``rv``, ``chordv``, ``lvcomp``, symmetry/core-radius params, and
    Kutta/strip-off row indices) so ``openavl.jax.setup.rebuild_circulation_geometry``
    can recompute those matrices from a live, differentiable Mach (A2).
    """
    nvor = state.nvor
    ncontrol = state.ncontrol
    enc_d = (
        np.asarray(state.enc_d[:, :nvor, :ncontrol], dtype=np.float64)
        if ncontrol
        else np.zeros((3, nvor, 0), dtype=np.float64)
    )
    kutta_iv, kutta_j1, kutta_j2, stripoff_iv = _kutta_stripoff_row_indices(state)
    return CirculationGeometry(
        rc=jnp.asarray(state.rc[:, :nvor], dtype=jnp.float64),
        enc=jnp.asarray(state.enc[:, :nvor], dtype=jnp.float64),
        enc_d=jnp.asarray(enc_d, dtype=jnp.float64),
        aicn=jnp.asarray(_rebuild_aicn(state), dtype=jnp.float64),
        wc_gam=jnp.asarray(state.wc_gam[:, :nvor, :nvor], dtype=jnp.float64),
        wv_gam=jnp.asarray(state.wv_gam[:, :nvor, :nvor], dtype=jnp.float64),
        wcsrd_u=jnp.asarray(state.wcsrd_u[:, :nvor, : state.numax], dtype=jnp.float64),
        wvsrd_u=jnp.asarray(state.wvsrd_u[:, :nvor, : state.numax], dtype=jnp.float64),
        lvnc=jnp.asarray(state.lvnc[:nvor], dtype=bool),
        lvalbe=jnp.asarray(state.lvalbe[:nvor], dtype=bool),
        numax=int(state.numax),
        ncontrol=int(ncontrol),
        rv1=jnp.asarray(state.rv1[:, :nvor], dtype=jnp.float64),
        rv2=jnp.asarray(state.rv2[:, :nvor], dtype=jnp.float64),
        rv=jnp.asarray(state.rv[:, :nvor], dtype=jnp.float64),
        chordv=jnp.asarray(state.chordv[:nvor], dtype=jnp.float64),
        lvcomp=jnp.asarray(state.lvcomp[:nvor], dtype=jnp.int32),
        iysym=int(state.iysym),
        ysym=float(state.ysym),
        izsym=int(state.izsym),
        zsym=float(state.zsym),
        vrcorec=float(state.vrcorec),
        vrcorew=float(state.vrcorew),
        kutta_iv=jnp.asarray(kutta_iv, dtype=jnp.int32),
        kutta_j1=jnp.asarray(kutta_j1, dtype=jnp.int32),
        kutta_j2=jnp.asarray(kutta_j2, dtype=jnp.int32),
        stripoff_iv=jnp.asarray(stripoff_iv, dtype=jnp.int32),
        snapshot_mach=jnp.asarray(state.mach, dtype=jnp.float64),
    )


def snapshot_analysis_geometry(state: Any) -> AnalysisGeometry:
    """Extract bundled geometry for the full JAX analysis pipeline."""
    return AnalysisGeometry(
        circulation=snapshot_circulation_geometry(state),
        force=force_geometry_from_state(state),
        body=body_geometry_from_state(state),
        trefftz=trefftz_geometry_from_state(state),
    )


def snapshot_flow(state: Any) -> FlowCondition:
    """Extract differentiable flow inputs from ``AVLState``."""
    ncontrol = state.ncontrol
    delcon = (
        np.asarray(state.delcon[:ncontrol], dtype=np.float64)
        if ncontrol
        else np.zeros(0, dtype=np.float64)
    )
    return FlowCondition(
        alfa=jnp.asarray(state.alfa, dtype=jnp.float64),
        beta=jnp.asarray(state.beta, dtype=jnp.float64),
        wrot=jnp.asarray(state.wrot, dtype=jnp.float64),
        mach=jnp.asarray(state.mach, dtype=jnp.float64),
        delcon=jnp.asarray(delcon, dtype=jnp.float64),
    )


def snapshot_refs(state: Any) -> ReferenceQuantities:
    """Extract reference quantities from ``AVLState``."""
    return ReferenceQuantities(
        sref=jnp.asarray(state.sref, dtype=jnp.float64),
        cref=jnp.asarray(state.cref, dtype=jnp.float64),
        bref=jnp.asarray(state.bref, dtype=jnp.float64),
        xyzref=jnp.asarray(state.xyzref, dtype=jnp.float64),
        cdref=jnp.asarray(state.cdref, dtype=jnp.float64),
        iysym=int(state.iysym),
    )


__all__ = [
    "AnalysisGeometry",
    "CirculationGeometry",
    "FlowCondition",
    "GeometryArrays",
    "GeometryStripMap",
    "ReferenceQuantities",
    "snapshot_analysis_geometry",
    "snapshot_circulation_geometry",
    "snapshot_flow",
    "snapshot_geometry",
    "snapshot_refs",
]

