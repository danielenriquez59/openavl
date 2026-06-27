"""AVL .avl configuration file parser."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from openavl.geom.airfoil import (
    AirfoilCamber,
    build_body_thread,
    build_camber_slope,
    build_naca_slope,
    parse_body_coords,
)
from openavl.fileio.mass import MassProperties

KEYWORD_PREFIXES = frozenset(
    {
        "SURF",
        "SECT",
        "BODY",
        "COMP",
        "INDE",
        "YDUP",
        "SCAL",
        "TRAN",
        "ANGL",
        "NOWA",
        "NOLO",
        "NACA",
        "AFIL",
        "BFIL",
        "AIRF",
        "CONT",
        "CDCL",
    }
)


def normalize_airfoil_path(raw_path: str) -> str:
    raw = str(raw_path or "").strip()
    if not raw:
        return ""
    return re.sub(r'^["\']+|["\']+$', "", raw).strip()


def strip_inline_comment(line: str) -> str:
    text = str(line or "")
    hash_idx = text.find("#")
    bang_idx = text.find("!")
    cut_idx = -1
    if hash_idx >= 0:
        cut_idx = hash_idx
    if bang_idx >= 0 and (cut_idx < 0 or bang_idx < cut_idx):
        cut_idx = bang_idx
    return text[:cut_idx] if cut_idx >= 0 else text


def parse_numbers(line: str) -> list[float]:
    clean = strip_inline_comment(line).strip()
    if not clean:
        return []
    return [float(v) for v in clean.split() if _is_finite_float(v)]


def parse_xy_coords_text(text: str) -> list[list[float]]:
    """Parse first-two-column coordinate pairs from AVL-style text."""
    coords: list[list[float]] = []
    for raw in str(text or "").splitlines():
        trimmed = raw.strip()
        if not trimmed or _is_comment_line(trimmed):
            continue
        nums = parse_numbers(trimmed)
        if len(nums) >= 2:
            coords.append([nums[0], nums[1]])
    return coords


def _is_finite_float(v: str) -> bool:
    try:
        x = float(v)
    except ValueError:
        return False
    return x == x and abs(x) != float("inf")


def _is_comment_line(line: str) -> bool:
    t = str(line or "").strip()
    return not t or t.startswith("#") or t.startswith("!") or t.startswith("%")


def _is_keyword_line(line: str) -> bool:
    t = strip_inline_comment(line).strip().upper()
    if not t:
        return False
    return t[:4] in KEYWORD_PREFIXES


@dataclass
class ControlDef:
    name: str = "CTRL"
    gain: float = 1.0
    xhinge: float = 0.75
    vhinge: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    sgn_dup: float = 1.0
    index: int = 0


@dataclass
class SectionDef:
    xle: float = 0.0
    yle: float = 0.0
    zle: float = 0.0
    chord: float = 1.0
    ainc: float = 0.0
    ainc_deg: float = 0.0
    n_span: int = 0
    s_space: float = 1.0
    controls: list[ControlDef] = field(default_factory=list)
    naca: str | None = None
    airfoil_file: str | None = None
    airfoil_coords: list[list[float]] | None = None
    cdcl: list[float] | None = None
    airfoil_camber: AirfoilCamber | None = None
    claf: float = 1.0


@dataclass
class SurfaceDef:
    name: str = "Surface"
    n_chord: int = 0
    c_space: float = 1.0
    n_span: int = 0
    s_space: float = 1.0
    sections: list[SectionDef] = field(default_factory=list)
    yduplicate: float | None = None
    scale: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    translate: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    angle_deg: float = 0.0
    component: int = 0
    lvalbe: bool = True
    cdcl: list[float] | None = None
    nowake: bool = False
    noload: bool = False
    imags: int = 1
    clmax: float = 0.0  # max local Cl per surface; 0 = disabled (Geometry API only)


@dataclass
class BodyDef:
    name: str = "Body"
    n_body: int = 0
    b_space: float = 1.0
    scale: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    translate: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    yduplicate: float | None = None
    body_file: str | None = None
    body_coords: list[list[float]] | None = None
    body_thread_x: list[float] | None = None
    body_thread_y: list[float] | None = None
    body_thread_t: list[float] | None = None


@dataclass
class AVLHeader:
    title: str = ""
    mach: float = 0.0
    iysym: int = 0
    izsym: int = 0
    zsym: float = 0.0
    sref: float = 1.0
    cref: float = 1.0
    bref: float = 1.0
    xref: float = 0.0
    yref: float = 0.0
    zref: float = 0.0


@dataclass
class AVLModel:
    header: AVLHeader = field(default_factory=AVLHeader)
    surfaces: list[SurfaceDef] = field(default_factory=list)
    bodies: list[BodyDef] = field(default_factory=list)
    airfoil_files: list[str] = field(default_factory=list)
    body_files: list[str] = field(default_factory=list)
    control_map: dict[str, int] = field(default_factory=dict)
    mass: MassProperties | None = None


def _parse_control_line(raw: str, read_next_line) -> ControlDef:
    parts = raw.split()[1:]
    if len(parts) < 2:
        next_line = read_next_line() or ""
        parts = next_line.strip().split()
    name = parts.pop(0) if parts else "CTRL"
    nums = [float(v) for v in parts if _is_finite_float(v)]
    return ControlDef(
        name=name,
        gain=nums[0] if len(nums) > 0 else 1.0,
        xhinge=nums[1] if len(nums) > 1 else 0.75,
        vhinge=[
            nums[2] if len(nums) > 2 else 0.0,
            nums[3] if len(nums) > 3 else 0.0,
            nums[4] if len(nums) > 4 else 0.0,
        ],
        sgn_dup=nums[5] if len(nums) > 5 else 1.0,
    )


def _parse_cdcl_line(trimmed_line: str, read_next_line) -> list[float] | None:
    inline = parse_numbers(trimmed_line[4:])
    vals = inline if len(inline) >= 6 else parse_numbers(read_next_line() or "")
    if len(vals) < 6:
        return None
    try:
        from openavl.geometry.cdcl_polar import CdclPolar

        return CdclPolar.from_points(
            (vals[0], vals[1]),
            (vals[2], vals[3]),
            (vals[4], vals[5]),
        ).as_list()
    except ValueError:
        return None


def _parse_airfoil_block(lines: list[str], index_ref: dict[str, int]) -> list[list[float]]:
    coords: list[list[float]] = []
    while index_ref["i"] < len(lines):
        raw = lines[index_ref["i"]]
        if not raw:
            index_ref["i"] += 1
            continue
        trimmed = raw.strip()
        if not trimmed:
            index_ref["i"] += 1
            if coords:
                break
            continue
        if _is_keyword_line(trimmed):
            break
        index_ref["i"] += 1
        if _is_comment_line(trimmed):
            continue
        nums = parse_numbers(trimmed)
        if len(nums) >= 2:
            coords.append([nums[0], nums[1]])
    return coords


def _parse_afil_inline_path(trimmed_line: str) -> str:
    parts = strip_inline_comment(trimmed_line).strip().split()[1:]
    if parts:
        path_tokens = [token for token in parts if not _is_finite_float(token)]
        if path_tokens:
            inline_path = normalize_airfoil_path(" ".join(path_tokens))
            if inline_path:
                return inline_path
    return ""


def parse_avl(text: str) -> AVLModel:
    """Parse AVL configuration text into an AVLModel tree."""
    lines = str(text or "").splitlines()
    surfaces: list[SurfaceDef] = []
    bodies: list[BodyDef] = []
    inline_airfoils: list[str] = []
    body_files: list[str] = []
    index_ref = {"i": 0}
    header = AVLHeader()

    def next_line() -> str | None:
        while index_ref["i"] < len(lines) and _is_comment_line(lines[index_ref["i"]]):
            index_ref["i"] += 1
        if index_ref["i"] >= len(lines):
            return None
        line = strip_inline_comment(lines[index_ref["i"]]).strip()
        index_ref["i"] += 1
        return line

    def read_value_line() -> list[float]:
        line = next_line()
        return parse_numbers(line or "")

    title_line = next_line()
    if title_line:
        header.title = title_line
    mach_line = read_value_line()
    if mach_line:
        header.mach = mach_line[0]
    sym_line = read_value_line()
    if len(sym_line) >= 3:
        header.iysym = int(sym_line[0])
        header.izsym = int(sym_line[1])
        header.zsym = sym_line[2]
    ref_line = read_value_line()
    if len(ref_line) >= 3:
        header.sref, header.cref, header.bref = ref_line[:3]
    xyz_line = read_value_line()
    if len(xyz_line) >= 3:
        header.xref, header.yref, header.zref = xyz_line[:3]

    while index_ref["i"] < len(lines):
        line = next_line()
        if not line:
            break
        key = line[:4].upper()
        if key not in ("SURF", "BODY"):
            continue

        if key == "BODY":
            body_name = next_line() or "Body"
            spacing = read_value_line()
            body = BodyDef(
                name=body_name,
                n_body=int(spacing[0]) if spacing else 0,
                b_space=spacing[1] if len(spacing) > 1 else 1.0,
            )
            while index_ref["i"] < len(lines):
                mark = lines[index_ref["i"]]
                trimmed = mark.strip()
                if not trimmed or _is_comment_line(trimmed):
                    index_ref["i"] += 1
                    continue
                subkey = trimmed[:4].upper()
                if subkey in ("SURF", "BODY"):
                    break
                index_ref["i"] += 1
                if subkey == "YDUP":
                    nums = parse_numbers(trimmed[4:])
                    vals = nums if nums else read_value_line()
                    body.yduplicate = vals[0] if vals else body.yduplicate
                elif subkey == "SCAL":
                    nums = parse_numbers(trimmed[4:])
                    vals = nums if nums else read_value_line()
                    if len(vals) >= 3:
                        body.scale = vals[:3]
                elif subkey == "TRAN":
                    nums = parse_numbers(trimmed[4:])
                    vals = nums if nums else read_value_line()
                    if len(vals) >= 3:
                        body.translate = vals[:3]
                elif subkey == "BFIL":
                    parts = strip_inline_comment(trimmed).strip().split()[1:]
                    path_value = " ".join(parts)
                    if not path_value:
                        path_value = next_line() or ""
                    normalized = normalize_airfoil_path(path_value)
                    body.body_file = normalized or None
                    if normalized:
                        body_files.append(normalized)
            bodies.append(body)
            continue

        surface_name = next_line() or "Surface"
        spacing = read_value_line()
        surface = SurfaceDef(
            name=surface_name,
            n_chord=int(spacing[0]) if spacing else 0,
            c_space=spacing[1] if len(spacing) > 1 else 1.0,
            n_span=int(spacing[2]) if len(spacing) > 2 else 0,
            s_space=spacing[3] if len(spacing) > 3 else 1.0,
        )
        current_section: SectionDef | None = None
        while index_ref["i"] < len(lines):
            mark = lines[index_ref["i"]]
            trimmed = mark.strip()
            if not trimmed or _is_comment_line(trimmed):
                index_ref["i"] += 1
                continue
            subkey = trimmed[:4].upper()
            if subkey in ("SURF", "BODY"):
                break
            index_ref["i"] += 1

            if subkey in ("COMP", "INDE"):
                nums = parse_numbers(trimmed[4:])
                vals = nums if nums else read_value_line()
                surface.component = int(vals[0]) if vals else surface.component
            elif subkey == "YDUP":
                nums = parse_numbers(trimmed[4:])
                vals = nums if nums else read_value_line()
                surface.yduplicate = vals[0] if vals else surface.yduplicate
            elif subkey == "SCAL":
                nums = parse_numbers(trimmed[4:])
                vals = nums if nums else read_value_line()
                if len(vals) >= 3:
                    surface.scale = vals[:3]
            elif subkey == "TRAN":
                nums = parse_numbers(trimmed[4:])
                vals = nums if nums else read_value_line()
                if len(vals) >= 3:
                    surface.translate = vals[:3]
            elif subkey == "NOWA":
                surface.nowake = True
            elif subkey == "NOLO":
                surface.noload = True
            elif subkey == "ANGL":
                nums = parse_numbers(trimmed[4:])
                vals = nums if nums else read_value_line()
                if vals:
                    surface.angle_deg = vals[0]
            elif subkey == "SECT":
                nums = parse_numbers(trimmed[4:])
                vals = nums if len(nums) >= 5 else read_value_line()
                if len(vals) >= 5:
                    current_section = SectionDef(
                        xle=vals[0],
                        yle=vals[1],
                        zle=vals[2],
                        chord=vals[3],
                        ainc=vals[4],
                        ainc_deg=vals[4],
                        n_span=int(vals[5]) if len(vals) > 5 else 0,
                        s_space=vals[6] if len(vals) > 6 else 1.0,
                        cdcl=list(surface.cdcl) if surface.cdcl else None,
                    )
                    surface.sections.append(current_section)
            elif subkey == "CDCL":
                cdcl = _parse_cdcl_line(trimmed, next_line)
                if not cdcl:
                    continue
                if current_section:
                    current_section.cdcl = cdcl
                else:
                    surface.cdcl = cdcl
            elif subkey == "NACA":
                if not current_section:
                    continue
                code = strip_inline_comment(trimmed[4:]).strip() or (next_line() or "")
                match = re.search(r"(\d{4})", code)
                current_section.naca = match.group(1) if match else None
            elif subkey == "CLAF":
                if not current_section:
                    continue
                nums = parse_numbers(trimmed[4:])
                vals = nums if nums else read_value_line()
                if vals:
                    claf = float(vals[0])
                    if 0.0 < claf < 2.0:
                        current_section.claf = claf
            elif subkey == "AFIL":
                if not current_section:
                    continue
                normalized = _parse_afil_inline_path(trimmed)
                if not normalized:
                    while index_ref["i"] < len(lines):
                        probe_raw = lines[index_ref["i"]]
                        probe = strip_inline_comment(probe_raw).strip()
                        if not probe or _is_comment_line(probe_raw):
                            index_ref["i"] += 1
                            continue
                        probe_key = probe[:4].upper()
                        if probe_key == "AFIL":
                            index_ref["i"] += 1
                            normalized = _parse_afil_inline_path(probe)
                            if normalized:
                                break
                            continue
                        if _is_keyword_line(probe):
                            normalized = ""
                            break
                        normalized = normalize_airfoil_path(probe)
                        index_ref["i"] += 1
                        break
                current_section.airfoil_file = normalized or None
                if normalized:
                    inline_airfoils.append(normalized)
            elif subkey == "AIRF":
                if not current_section:
                    continue
                current_section.airfoil_coords = _parse_airfoil_block(lines, index_ref)
            elif subkey == "CONT":
                if not current_section:
                    continue
                current_section.controls.append(_parse_control_line(trimmed, next_line))
        surfaces.append(surface)

    return AVLModel(
        header=header,
        surfaces=surfaces,
        bodies=bodies,
        airfoil_files=inline_airfoils,
        body_files=body_files,
    )


def parse_avl_file(path: str | Path) -> AVLModel:
    """Parse an .avl file from disk."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return parse_avl(text)


def prepare_model(model: AVLModel, base_dir: str | Path | None = None) -> AVLModel:
    """Assign control indices and default airfoil camber tables."""
    control_map: dict[str, int] = {}
    base = Path(base_dir) if base_dir else None

    for surf_idx, surf in enumerate(model.surfaces):
        if surf.component == 0:
            surf.component = surf_idx + 1
        if surf.imags == 0:
            surf.imags = 1

        for sec in surf.sections:
            for ctrl in sec.controls:
                if ctrl.name not in control_map:
                    control_map[ctrl.name] = len(control_map)
                ctrl.index = control_map[ctrl.name]

            coords = sec.airfoil_coords
            if not coords and sec.airfoil_file and base:
                af_path = Path(sec.airfoil_file)
                if not af_path.is_absolute():
                    af_path = base / af_path
                if af_path.is_file():
                    coords = _parse_airfoil_coords_file(af_path)
                    sec.airfoil_coords = coords

            if coords:
                sec.airfoil_camber = build_camber_slope(coords)
            elif sec.naca:
                sec.airfoil_camber = build_naca_slope(sec.naca)
            else:
                sec.airfoil_camber = build_naca_slope("0000")

    model.control_map = control_map

    for body in model.bodies:
        coords = body.body_coords
        if not coords and body.body_file and base:
            body_path = Path(body.body_file)
            if not body_path.is_absolute():
                body_path = base / body_path
            if body_path.is_file():
                coords = parse_body_coords(body_path.read_text(encoding="utf-8", errors="replace"))
                body.body_coords = coords
        if coords:
            thread = build_body_thread(coords)
            if thread is not None:
                xc, yc, tc = thread
                body.body_thread_x = xc.tolist()
                body.body_thread_y = yc.tolist()
                body.body_thread_t = tc.tolist()

    return model


def _parse_airfoil_coords_file(path: Path) -> list[list[float]]:
    return parse_xy_coords_text(path.read_text(encoding="utf-8", errors="replace"))
