"""AVL .avl configuration file parser."""

from __future__ import annotations

import re
import warnings
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

# Every keyword ainput.f dispatches on (first four characters).
KEYWORD_PREFIXES = frozenset(
    {
        "SURF",
        "SECT",
        "BODY",
        "BSEC",
        "COMP",
        "INDE",
        "YDUP",
        "SCAL",
        "TRAN",
        "ANGL",
        "AINC",
        "NOWA",
        "NOLO",
        "NOAL",
        "NACA",
        "AFIL",
        "BFIL",
        "AIRF",
        "CONT",
        "CDCL",
        "CLAF",
        "DESI",
        "CORE",
    }
)


def normalize_airfoil_path(raw_path: str) -> str:
    raw = str(raw_path or "").strip()
    if not raw:
        return ""
    # Double quotes delimit the name (ainput.f AFIL); text after the closing
    # quote, such as trailing numbers, is not part of it.
    match = re.search(r'"([^"]+)"', raw)
    if match:
        return match.group(1).strip()
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
    # Fortran list-directed input / GETFLT: commas separate values.
    return [x for x in map(_to_float, clean.replace(",", " ").split()) if x is not None]


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


# A whole token that is a Fortran D-exponent number, e.g. 1.0D0 or 2.5d-3.
_FORTRAN_D_NUMBER = re.compile(r"([+-]?(?:\d+\.?\d*|\.\d+))[dD]([+-]?\d+)")


def _to_float(v: str) -> float | None:
    """Return the finite number a token reads as (Fortran D exponents allowed), else None."""
    match = _FORTRAN_D_NUMBER.fullmatch(v)
    if match:
        v = f"{match.group(1)}e{match.group(2)}"
    try:
        x = float(v)
    except ValueError:
        return None
    return x if x == x and abs(x) != float("inf") else None


def _is_finite_float(v: str) -> bool:
    return _to_float(v) is not None


def _is_comment_line(line: str) -> bool:
    t = str(line or "").strip()
    return not t or t.startswith("#") or t.startswith("!") or t.startswith("%")


def _is_keyword_line(line: str) -> bool:
    t = strip_inline_comment(line).strip().upper()
    if not t:
        return False
    return t[:4] in KEYWORD_PREFIXES


def _looks_like_file_path(line: str) -> bool:
    """Return True when a line looks like a filesystem path, not an AVL keyword.

    AVL keywords are matched by a 4-character prefix, so a next-line path such as
    ``airfoils/GOE316.dat`` would otherwise be mistaken for the ``AIRF`` keyword.
    """
    token = strip_inline_comment(line).strip().split()
    if not token:
        return False
    first = token[0]
    return any(ch in first for ch in ("/", "\\", "."))


@dataclass
class ControlDef:
    name: str = "CTRL"
    gain: float = 1.0
    xhinge: float = 0.0
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
    # Airfoil x/c range from the NACA/AIRFOIL/AFILE keyword line (XFMIN XFMAX).
    xfmin: float = 0.0
    xfmax: float = 1.0


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
    lrange: bool = True  # False once any airfoil keyword gives a partial x/c range (LRANGE)


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
    cdp: float = 0.0  # optional 6th header line: baseline profile drag (Fortran CDREF0)


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
    parts = strip_inline_comment(raw).split()[1:]
    if len(parts) < 2:
        next_line = read_next_line() or ""
        parts = strip_inline_comment(next_line).split()
    name = parts.pop(0) if parts else "CTRL"
    nums = parse_numbers(" ".join(parts))
    # Short-line defaults follow ainput.f: gain 1, xhinge 0, hinge vector 0
    # unless all three components are given, sgn_dup 1.
    has_vhinge = len(nums) > 4
    return ControlDef(
        name=name,
        gain=nums[0] if len(nums) > 0 else 1.0,
        xhinge=nums[1] if len(nums) > 1 else 0.0,
        vhinge=nums[2:5] if has_vhinge else [0.0, 0.0, 0.0],
        sgn_dup=nums[5] if len(nums) > 5 else 1.0,
    )


def _parse_cdcl_line(trimmed_line: str, read_next_line) -> list[float] | None:
    inline = parse_numbers(trimmed_line[4:])
    vals = inline if len(inline) >= 6 else parse_numbers(read_next_line() or "")
    if len(vals) < 6:
        return None
    cl = [vals[0], vals[2], vals[4]]
    cd = [vals[1], vals[3], vals[5]]
    # ainput.f CDCL: no validation, only this ordering (strict comparisons,
    # so ties keep the first point as min and the first strictly larger as max).
    lmax = 0
    lmin = 0
    for k in (1, 2):
        if cl[k] > cl[lmax]:
            lmax = k
        if cl[k] < cl[lmin]:
            lmin = k
    lmid = 3 - (lmin + lmax)
    if lmin == lmax:
        # All three CLs equal: Fortran's LMID = 6-(LMIN+LMAX) indexes past the
        # end of CLX, so there is no defined AVL result; keep the input order.
        lmin, lmid, lmax = 0, 1, 2
    if len({cl[0], cl[1], cl[2]}) < 3 or min(cd) < 0.0:
        warnings.warn(
            f"CDCL polar {vals[:6]} has repeated CL values or a negative CD; "
            "kept as given, as AVL does.",
            UserWarning,
            stacklevel=3,
        )
    return [cl[lmin], cd[lmin], cl[lmid], cd[lmid], cl[lmax], cd[lmax]]


def _leading_xy_pair(line: str) -> list[float] | None:
    """Return the first two numbers of a coordinate line, or None.

    Mirrors ainput.f AIRF: GETFLT reads at most two values and the block ends
    at the first line on which two leading numbers cannot be read.
    """
    tokens = strip_inline_comment(line).replace(",", " ").split()
    if len(tokens) < 2 or not (_is_finite_float(tokens[0]) and _is_finite_float(tokens[1])):
        return None
    return [_to_float(tokens[0]), _to_float(tokens[1])]


def _parse_airfoil_block(lines: list[str], index_ref: dict[str, int]) -> list[list[float]]:
    """Read inline AIRFOIL coordinates.

    ainput.f reads lines with RDLINE (which skips blank and comment lines)
    until one does not give two numbers; that line is left unconsumed and is
    re-dispatched as a keyword ("GO TO 11").
    """
    coords: list[list[float]] = []
    while index_ref["i"] < len(lines):
        trimmed = lines[index_ref["i"]].strip()
        if not trimmed or _is_comment_line(trimmed):
            index_ref["i"] += 1
            continue
        pair = _leading_xy_pair(trimmed)
        if pair is None:
            break
        index_ref["i"] += 1
        coords.append(pair)
    return coords


def _parse_xf_limits(trimmed_line: str) -> tuple[float, float]:
    """Return optional ``XFMIN XFMAX`` from an airfoil keyword line.

    ainput.f (NACA/AIRF/AFIL/BFIL) reads two numbers after the keyword with
    GETFLT; if fewer than two numbers parse, the range is the whole airfoil.
    """
    parts = strip_inline_comment(trimmed_line).replace(",", " ").split()[1:3]
    if len(parts) < 2 or not all(_is_finite_float(p) for p in parts):
        return 0.0, 1.0
    return max(0.0, _to_float(parts[0])), min(1.0, _to_float(parts[1]))


def _is_partial_range(xfmin: float, xfmax: float) -> bool:
    """ainput.f sets LRANGE false only for XFMIN > 0.01 or XFMAX < 0.99."""
    return xfmin > 0.01 or xfmax < 0.99


def _sign_flag(value: float) -> int:
    """Clamp a symmetry flag to -1/0/+1 (ainput.f IYSYM/IZSYM)."""
    flag = int(value)
    return (flag > 0) - (flag < 0)


def _count(value: float) -> int:
    """Panel count as ainput.f reads it: ``INT( RINPUT + 0.001 )``."""
    return int(value + 0.001)


def _check_redundant_ysym(iysym: int, ydupl: float) -> None:
    """ainput.f YDUP: stop on YDUPLICATE 0.0 when IYSYM /= 0 (surface or body)."""
    if iysym != 0 and ydupl == 0.0:
        raise ValueError(
            "Redundant y-symmetry specifications: IYSYM /= 0 and YDUPLICATE 0.0. "
            "Can use one or the other, but not both."
        )


def _parse_naca_designator(line: str) -> str | None:
    """Return the NACA designator token, read as one integer like ainput.f."""
    tokens = strip_inline_comment(line).replace(",", " ").split()
    if tokens and tokens[0].isdigit():
        return tokens[0]
    warnings.warn(
        f"Unreadable NACA designator line: {line!r}; section gets no NACA airfoil.",
        UserWarning,
        stacklevel=3,
    )
    return None


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
    warned_design = False

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
        # ainput.f clamps both flags to -1/0/+1.
        header.iysym = _sign_flag(sym_line[0])
        header.izsym = _sign_flag(sym_line[1])
        header.zsym = sym_line[2]
    ref_line = read_value_line()
    if len(ref_line) >= 3:
        # ainput.f: IF(SREF .LE. 0.) SREF = 1.  (likewise CREF, BREF)
        header.sref, header.cref, header.bref = (v if v > 0.0 else 1.0 for v in ref_line[:3])
    xyz_line = read_value_line()
    if len(xyz_line) >= 3:
        header.xref, header.yref, header.zref = xyz_line[:3]
    # Optional CDp line: consume it only when it reads as a number, otherwise
    # leave it for the keyword loop (ainput.f: READ (LINE,*,ERR=8) CDREF0).
    while index_ref["i"] < len(lines) and _is_comment_line(lines[index_ref["i"]]):
        index_ref["i"] += 1
    if index_ref["i"] < len(lines):
        cdp_tokens = strip_inline_comment(lines[index_ref["i"]]).replace(",", " ").split()
        if cdp_tokens and _is_finite_float(cdp_tokens[0]):
            header.cdp = _to_float(cdp_tokens[0])
            index_ref["i"] += 1

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
                    if vals:
                        _check_redundant_ysym(header.iysym, vals[0])
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
                    # Optional XFMIN XFMAX on the keyword line are read by
                    # ainput.f but never used for bodies (GETCAM takes the
                    # whole shape), so numeric tokens are not part of the path.
                    path_value = _parse_afil_inline_path(trimmed)
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
        # ainput.f SURF: Nspan/Sspace are used only when both are present
        # (NINPUT >= 4); otherwise NVS = 0, SSPACE = 0 and the per-section
        # values set the spanwise panelling.
        has_span = len(spacing) >= 4
        surface = SurfaceDef(
            name=surface_name,
            n_chord=_count(spacing[0]) if spacing else 0,
            c_space=spacing[1] if len(spacing) > 1 else 1.0,
            n_span=_count(spacing[2]) if has_span else 0,
            s_space=spacing[3] if has_span else 0.0,
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
                if vals:
                    _check_redundant_ysym(header.iysym, vals[0])
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
            elif subkey == "NOAL":
                surface.lvalbe = False
            elif subkey in ("ANGL", "AINC"):
                nums = parse_numbers(trimmed[4:])
                vals = nums if nums else read_value_line()
                if vals:
                    surface.angle_deg = vals[0]
            elif subkey == "SECT":
                nums = parse_numbers(trimmed[4:])
                vals = nums if len(nums) >= 5 else read_value_line()
                if len(vals) >= 5:
                    # ainput.f SECT: NSPANS/SSPACES only when all 7 values are
                    # present (NINPUT >= 7), else both 0.
                    has_span = len(vals) >= 7
                    current_section = SectionDef(
                        xle=vals[0],
                        yle=vals[1],
                        zle=vals[2],
                        chord=vals[3],
                        ainc=vals[4],
                        ainc_deg=vals[4],
                        n_span=_count(vals[5]) if has_span else 0,
                        s_space=vals[6] if has_span else 0.0,
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
                # ainput.f: optional XFMIN XFMAX on the keyword line, designator
                # on the next line. A lone integer on the keyword line is kept
                # as an inline designator (OpenAVL extension; AVL rejects it).
                xfmin, xfmax = _parse_xf_limits(trimmed)
                inline = strip_inline_comment(trimmed).split()[1:]
                if len(inline) == 1 and inline[0].isdigit():
                    code_line = inline[0]
                else:
                    code_line = next_line() or ""
                current_section.naca = _parse_naca_designator(code_line)
                # NACA/AIRFOIL/AFILE each overwrite the section camber in
                # ainput.f, so the last one given wins.
                current_section.airfoil_file = None
                current_section.airfoil_coords = None
                current_section.xfmin, current_section.xfmax = xfmin, xfmax
                if _is_partial_range(xfmin, xfmax):
                    surface.lrange = False
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
                        # Path-like lines (e.g. airfoils/foo.dat) can share the
                        # AIRF prefix; treat those as AFIL paths, not keywords.
                        if _is_keyword_line(probe) and not _looks_like_file_path(probe):
                            normalized = ""
                            break
                        normalized = normalize_airfoil_path(probe)
                        index_ref["i"] += 1
                        break
                current_section.airfoil_file = normalized or None
                current_section.naca = None  # last airfoil keyword wins
                current_section.airfoil_coords = None
                if normalized:
                    inline_airfoils.append(normalized)
                # x/c limits come only from the AFIL keyword line (ainput.f).
                xfmin, xfmax = _parse_xf_limits(trimmed)
                current_section.xfmin, current_section.xfmax = xfmin, xfmax
                if _is_partial_range(xfmin, xfmax):
                    surface.lrange = False
            elif subkey == "AIRF":
                if not current_section:
                    continue
                xfmin, xfmax = _parse_xf_limits(trimmed)
                current_section.xfmin, current_section.xfmax = xfmin, xfmax
                if _is_partial_range(xfmin, xfmax):
                    surface.lrange = False
                current_section.airfoil_coords = _parse_airfoil_block(lines, index_ref)
                current_section.naca = None  # last airfoil keyword wins
                current_section.airfoil_file = None
            elif subkey == "CONT":
                if not current_section:
                    continue
                current_section.controls.append(_parse_control_line(trimmed, next_line))
            elif subkey == "DESI":
                # ainput.f DESI: the next line is "DName Wdes". Design variables
                # are not supported, so the line is consumed and ignored.
                if not current_section:
                    continue
                next_line()
                if not warned_design:
                    warned_design = True
                    warnings.warn(
                        "DESIGN variables are not supported by OpenAVL and are ignored.",
                        UserWarning,
                        stacklevel=2,
                    )
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

    # ainput.f sets LNCOMP(ISURF) = ISURF when the SURFACE starts, where ISURF
    # counts solver surfaces: a YDUPLICATE image (SDUPL) takes the next number,
    # and shares its parent's component (amake.f: LNCOMP(NNI) = LNCOMP(NN)).
    solver_isurf = 0
    for surf in model.surfaces:
        solver_isurf += 1
        if surf.component == 0:
            surf.component = solver_isurf
        if surf.yduplicate is not None:
            solver_isurf += 1
        if surf.imags == 0:
            surf.imags = 1

        for sec in surf.sections:
            for ctrl in sec.controls:
                if ctrl.name not in control_map:
                    control_map[ctrl.name] = len(control_map)
                ctrl.index = control_map[ctrl.name]

            coords = sec.airfoil_coords
            if not coords and sec.airfoil_file:
                af_path = Path(sec.airfoil_file)
                if not af_path.is_absolute():
                    if base is None:
                        warnings.warn(
                            f"Airfoil file not found: {sec.airfoil_file} "
                            "(relative path with no base_dir); "
                            f"using {_airfoil_fallback_label(sec)} instead.",
                            UserWarning,
                            stacklevel=2,
                        )
                        af_path = None
                    else:
                        af_path = base / af_path
                if af_path is not None:
                    if af_path.is_file():
                        coords = _parse_airfoil_coords_file(af_path)
                        sec.airfoil_coords = coords
                    else:
                        warnings.warn(
                            f"Airfoil file not found: {af_path}; "
                            f"using {_airfoil_fallback_label(sec)} instead.",
                            UserWarning,
                            stacklevel=2,
                        )

            if coords:
                sec.airfoil_camber = build_camber_slope(coords, xfmin=sec.xfmin, xfmax=sec.xfmax)
            elif sec.naca:
                sec.airfoil_camber = build_naca_slope(sec.naca, xfmin=sec.xfmin, xfmax=sec.xfmax)
            else:
                sec.airfoil_camber = build_naca_slope("0000")

    model.control_map = control_map

    for body in model.bodies:
        coords = body.body_coords
        if not coords and body.body_file:
            body_path = Path(body.body_file)
            if not body_path.is_absolute():
                if base is None:
                    warnings.warn(
                        f"Body file not found: {body.body_file} "
                        "(relative path with no base_dir); body omitted.",
                        UserWarning,
                        stacklevel=2,
                    )
                    body_path = None
                else:
                    body_path = base / body_path
            if body_path is not None:
                if body_path.is_file():
                    coords = parse_body_coords(body_path.read_text(encoding="utf-8", errors="replace"))
                    body.body_coords = coords
                else:
                    warnings.warn(
                        f"Body file not found: {body_path}; body omitted.",
                        UserWarning,
                        stacklevel=2,
                    )
        if coords:
            thread = build_body_thread(coords)
            if thread is not None:
                xc, yc, tc = thread
                body.body_thread_x = xc.tolist()
                body.body_thread_y = yc.tolist()
                body.body_thread_t = tc.tolist()

    return model


def _airfoil_fallback_label(sec: SectionDef) -> str:
    """Return a short label for the camber used when an airfoil file is missing."""
    if sec.naca:
        return f"NACA {sec.naca}"
    return "flat plate (NACA 0000)"


def _parse_airfoil_coords_file(path: Path) -> list[list[float]]:
    return parse_xy_coords_text(path.read_text(encoding="utf-8", errors="replace"))
