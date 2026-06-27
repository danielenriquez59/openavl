"""Convert Geometry API objects to AVLModel dataclasses."""

from __future__ import annotations

from pathlib import Path

from openavl.fileio.parser import (
    AVLHeader,
    AVLModel,
    BodyDef,
    ControlDef,
    SectionDef,
    SurfaceDef,
    prepare_model,
)
from openavl.geometry.aircraft import Aircraft
from openavl.geometry.airfoil import AirfoilType
from openavl.geometry.cdcl_polar import CdclPolar


def _validate_aircraft(aircraft: Aircraft) -> None:
    """Raise ValueError if the aircraft definition is invalid for conversion."""
    if aircraft.sref <= 0:
        raise ValueError(f"sref must be > 0, got {aircraft.sref}")
    if aircraft.cref <= 0:
        raise ValueError(f"cref must be > 0, got {aircraft.cref}")
    if aircraft.bref <= 0:
        raise ValueError(f"bref must be > 0, got {aircraft.bref}")

    for wing in aircraft.wings:
        if len(wing.sections) < 2:
            raise ValueError(
                f"Wing {wing.name!r} must have at least 2 sections, got {len(wing.sections)}"
            )
        for sec in wing.sections:
            if sec.chord <= 0:
                raise ValueError(
                    f"Section chord must be > 0 on wing {wing.name!r}, got {sec.chord}"
                )


def _cdcl_as_list(cdcl: CdclPolar | None) -> list[float] | None:
    """Convert an optional geometry polar to the parser's six-value list."""
    return cdcl.as_list() if cdcl is not None else None


def _section_to_def(section, wing_cdcl: CdclPolar | None) -> SectionDef:
    """Build a SectionDef from a geometry Section."""
    sec_def = SectionDef(
        xle=section.xle,
        yle=section.yle,
        zle=section.zle,
        chord=section.chord,
        ainc=section.ainc,
        ainc_deg=section.ainc,
        n_span=section.n_span,
        s_space=section.s_space,
        claf=section.claf,
        cdcl=(
            _cdcl_as_list(section.cdcl)
            if section.cdcl is not None
            else _cdcl_as_list(wing_cdcl)
        ),
    )

    if section.airfoil is not None:
        if section.airfoil.af_type == AirfoilType.NACA:
            sec_def.naca = section.airfoil.naca
        elif section.airfoil.af_type == AirfoilType.FILE:
            sec_def.airfoil_file = section.airfoil.file_path
        elif section.airfoil.af_type == AirfoilType.COORDS:
            sec_def.airfoil_coords = [list(pt) for pt in (section.airfoil.coords or [])]

    for ctrl in section.controls:
        sec_def.controls.append(
            ControlDef(
                name=ctrl.name,
                gain=ctrl.gain,
                xhinge=ctrl.xhinge,
                vhinge=list(ctrl.vhinge),
                sgn_dup=ctrl.sgn_dup,
            )
        )

    return sec_def


def to_avl_model(aircraft: Aircraft, base_dir: str | Path | None = None) -> AVLModel:
    """Build and prepare an AVLModel from a Geometry API Aircraft.

    Copies each wing's :attr:`~openavl.geometry.Wing.clmax` into the internal
    surface definition for sectional lift capping during the solve.
    """
    _validate_aircraft(aircraft)

    header = AVLHeader(
        title=aircraft.name,
        mach=aircraft.mach,
        iysym=aircraft.iysym,
        izsym=aircraft.izsym,
        zsym=aircraft.zsym,
        sref=aircraft.sref,
        cref=aircraft.cref,
        bref=aircraft.bref,
        xref=aircraft.xref,
        yref=aircraft.yref,
        zref=aircraft.zref,
    )

    airfoil_files: list[str] = []
    body_files: list[str] = []
    surfaces: list[SurfaceDef] = []

    for wing in aircraft.wings:
        surf = SurfaceDef(
            name=wing.name,
            n_chord=wing.n_chord,
            c_space=wing.c_space,
            n_span=wing.n_span,
            s_space=wing.s_space,
            yduplicate=wing.yduplicate,
            scale=list(wing.scale),
            translate=list(wing.translate),
            angle_deg=wing.angle,
            component=wing.component,
            nowake=wing.nowake,
            noload=wing.noload,
            cdcl=_cdcl_as_list(wing.cdcl),
        )
        surf.clmax = wing.clmax
        for section in wing.sections:
            sec_def = _section_to_def(section, wing.cdcl)
            if sec_def.airfoil_file:
                airfoil_files.append(sec_def.airfoil_file)
            surf.sections.append(sec_def)
        surfaces.append(surf)

    bodies: list[BodyDef] = []
    for body in aircraft.bodies:
        body_def = BodyDef(
            name=body.name,
            n_body=body.n_body,
            b_space=body.b_space,
            scale=list(body.scale),
            translate=list(body.translate),
            yduplicate=body.yduplicate,
            body_file=body.body_file,
            body_coords=[list(pt) for pt in body.body_coords] if body.body_coords else None,
        )
        if body_def.body_file:
            body_files.append(body_def.body_file)
        bodies.append(body_def)

    model = AVLModel(
        header=header,
        surfaces=surfaces,
        bodies=bodies,
        airfoil_files=airfoil_files,
        body_files=body_files,
    )
    return prepare_model(model, base_dir=base_dir)
