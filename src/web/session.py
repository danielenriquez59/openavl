"""Web GUI session state and solver orchestration."""

from __future__ import annotations

import asyncio
import math
import re
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from openavl import constants as C
from openavl.core.reporting import nasa_dir, reported_totals
from openavl.core.solver import AVLSolver
from openavl.fileio.mass import MassProperties, _apply_mass_properties, masini, masput, parse_mass_text, unitset
from openavl.fileio.parser import AVLModel, normalize_airfoil_path, parse_avl, parse_xy_coords_text, prepare_model
from openavl.geom.geometry import build_geometry, solver_surface_name
from openavl.core.state import AVLState
from openavl.analysis.deriv import compute_body_axis_derivatives

from openavl.web.geometry_export import model_to_geometry

_EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"

EXAMPLES: dict[str, dict[str, Any]] = {
    "supra": {
        "label": "Supra F3J",
        "avl": _EXAMPLES_DIR / "supra.avl",
        "mass": _EXAMPLES_DIR / "supra.mass",
    },
}

_SESSION_TTL_SECONDS = 30 * 60

_DEFAULT_FLIGHT_PARAMS = {
    "cd0": 0.015,
    "rho": 1.225,
    "gravity": 9.81,
}

@dataclass
class SessionState:
    """Isolated solver state for one WebSocket connection."""

    session_id: str
    solver: AVLSolver | None = None
    last_active: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    disconnected_at: float | None = None
    pending_avl_text: str | None = None
    pending_mass_text: str | None = None
    warnings: list[str] = field(default_factory=list)
    airfoil_base_dir: Path | None = None
    airfoil_files: dict[str, str] = field(default_factory=dict)
    extra_airfoil_paths: list[str] = field(default_factory=list)
    invalid_airfoil_paths: set[str] = field(default_factory=set)
    _airfoil_temp: tempfile.TemporaryDirectory[str] | None = field(default=None, repr=False)


sessions: dict[str, SessionState] = {}


def _parse_airfoil_coords_text(text: str) -> list[list[float]]:
    """Parse coordinate pairs from uploaded airfoil file text."""
    return parse_xy_coords_text(text)


def _airfoil_text_valid(text: str) -> bool:
    """Return ``True`` when airfoil text contains at least one coordinate pair."""
    return len(_parse_airfoil_coords_text(text)) > 0


def list_airfoil_dependencies(
    avl_text: str | None,
    extra_paths: list[str] | None = None,
) -> list[str]:
    """Return unique normalized ``AFIL`` paths referenced by AVL text and manual entries."""
    paths: list[str] = []
    seen: set[str] = set()
    if avl_text:
        model = parse_avl(avl_text)
        for entry in model.airfoil_files:
            norm = normalize_airfoil_path(entry)
            if norm and norm not in seen:
                seen.add(norm)
                paths.append(norm)
    for entry in extra_paths or []:
        norm = normalize_airfoil_path(entry)
        if norm and norm not in seen:
            seen.add(norm)
            paths.append(norm)
    return paths


def _airfoil_source_path(session: SessionState, path: str) -> Path | None:
    """Return an on-disk path for a dependency when one is available."""
    rel = Path(path)
    if path in session.airfoil_files:
        return None
    if session.airfoil_base_dir is not None:
        candidate = session.airfoil_base_dir / rel
        if candidate.is_file():
            return candidate
    return None


def _airfoil_is_ready(session: SessionState, path: str) -> bool:
    """Return ``True`` when an airfoil dependency can be resolved for the solver."""
    if path in session.invalid_airfoil_paths:
        return False
    if path in session.airfoil_files and _airfoil_text_valid(session.airfoil_files[path]):
        return True
    return _airfoil_source_path(session, path) is not None


def build_afil_dependencies(session: SessionState) -> list[dict[str, Any]]:
    """Build AFIL dependency rows for the frontend panel."""
    avl_paths = set(list_airfoil_dependencies(session.pending_avl_text))
    rows: list[dict[str, Any]] = []
    for path in list_airfoil_dependencies(session.pending_avl_text, session.extra_airfoil_paths):
        if path in session.invalid_airfoil_paths:
            status = "invalid"
        elif _airfoil_is_ready(session, path):
            status = "ready"
        else:
            status = "missing"
        rows.append(
            {
                "path": path,
                "status": status,
                "manual": path not in avl_paths,
            }
        )
    return rows


def _cleanup_airfoil_temp(session: SessionState) -> None:
    """Release any temporary directory used to stage uploaded airfoil files."""
    if session._airfoil_temp is not None:
        session._airfoil_temp.cleanup()
        session._airfoil_temp = None


def _materialize_airfoil_work_dir(session: SessionState, avl_text: str) -> Path | None:
    """Write resolvable airfoil dependencies to a temp directory for ``prepare_model``."""
    deps = list_airfoil_dependencies(avl_text, session.extra_airfoil_paths)
    if not deps:
        _cleanup_airfoil_temp(session)
        return session.airfoil_base_dir

    _cleanup_airfoil_temp(session)
    session._airfoil_temp = tempfile.TemporaryDirectory(prefix="openavl-afil-")
    root = Path(session._airfoil_temp.name)

    for dep in deps:
        dest = root / Path(dep)
        if dep in session.airfoil_files:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(session.airfoil_files[dep], encoding="utf-8", errors="replace")
            continue
        source = _airfoil_source_path(session, dep)
        if source is not None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(source.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")

    return root


def _append_missing_afil_warnings(session: SessionState, warnings: list[str]) -> None:
    """Add warnings for unresolved ``AFIL`` dependencies."""
    missing = [row["path"] for row in build_afil_dependencies(session) if row["status"] == "missing"]
    if missing:
        joined = ", ".join(missing[:5])
        suffix = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        warnings.append(f"Missing AFIL airfoil files: {joined}{suffix}")


def _create_solver(
    model: AVLModel,
    *,
    mass_props: MassProperties | None = None,
    debug: bool = False,
    **state_options: Any,
) -> AVLSolver:
    """Construct an :class:`AVLSolver` from parsed model data without a geometry file."""
    solver = object.__new__(AVLSolver)
    solver.debug = debug
    solver.geo_file = None
    solver.mass_file = None
    solver.model = model
    solver.state = AVLState.from_model(model, debug=debug, **state_options)
    build_geometry(solver.state, model)
    solver.state.lgeo = True
    solver.state.lenc = True

    masini(solver.state)
    solver._apply_default_mass_parameters()
    if mass_props is not None and mass_props.loaded:
        _apply_mass_properties(solver.state, mass_props)
        unitset(solver.state)
        masput(solver.state, 0, 0)
        solver.model.mass = mass_props
    solver._apply_parameter_options(state_options)
    return solver


def _reference_dims(model: AVLModel, state: AVLState) -> dict[str, float]:
    """Return Sref/Cref/Bref from the parsed header (geometry units)."""
    return {
        "sref": float(model.header.sref),
        "cref": float(model.header.cref),
        "bref": float(model.header.bref),
    }


def _model_meta(solver: AVLSolver, session: SessionState | None = None) -> dict[str, Any]:
    """Summarize loaded aircraft metadata for the frontend."""
    state = solver.state
    model = solver.model
    mass = getattr(model.mass, "mass", None)
    meta: dict[str, Any] = {
        "title": model.header.title,
        "surfaces": int(state.nsurf),
        "strips": int(state.nstrip),
        "vortices": int(state.nvor),
        **_reference_dims(model, state),
        "controls": list(state.control_names),
        "mass_kg": float(mass) if mass is not None else None,
        "warnings": [],
    }
    if session is not None:
        meta["afil_dependencies"] = build_afil_dependencies(session)
    return meta


def _mass_values_from_props(props: MassProperties | None) -> dict[str, Any] | None:
    """Return display-friendly mass properties using AVL mass-file signs."""
    if props is None or not props.loaded:
        return None
    inertia = props.inertia
    unitl = float(props.unitl) or 1.0
    values: dict[str, Any] = {
        "mass": float(props.mass),
        "xcg": float(props.cg[0]) / unitl,
        "ycg": float(props.cg[1]) / unitl,
        "zcg": float(props.cg[2]) / unitl,
        "ixx": float(inertia[0, 0]),
        "iyy": float(inertia[1, 1]),
        "izz": float(inertia[2, 2]),
        "ixy": -float(inertia[0, 1]),
        "ixz": -float(inertia[0, 2]),
        "iyz": -float(inertia[1, 2]),
        "lunit": float(props.unitl),
        "munit": float(props.unitm),
        "tunit": float(props.unitt),
        "gravity": float(props.gee),
        "rho": float(props.rho),
    }
    values["components"] = [
        {
            "mass": float(comp["mass"]),
            "x": float(comp["x"]) / unitl,
            "y": float(comp["y"]) / unitl,
            "z": float(comp["z"]) / unitl,
        }
        for comp in props.components
    ]
    return values


def _active_mass_values(solver: AVLSolver) -> dict[str, float]:
    """Return the active run-case mass and inertia values for display."""
    state = solver.state
    ir = 0
    parval = state.parval
    return {
        "mass": float(parval[C.IPMASS, ir]),
        "xcg": float(parval[C.IPXCG, ir]),
        "ycg": float(parval[C.IPYCG, ir]),
        "zcg": float(parval[C.IPZCG, ir]),
        "ixx": float(parval[C.IPIXX, ir]),
        "iyy": float(parval[C.IPIYY, ir]),
        "izz": float(parval[C.IPIZZ, ir]),
        "ixy": -float(parval[C.IPIXY, ir]),
        "ixz": -float(parval[C.IPIZX, ir]),
        "iyz": -float(parval[C.IPIYZ, ir]),
        "gravity": float(parval[C.IPGEE, ir]),
        "rho": float(parval[C.IPRHO, ir]),
    }


def _mass_properties_payload(solver: AVLSolver) -> dict[str, Any]:
    """Build the mass-properties payload shared by model and results messages."""
    return {
        "file": _mass_values_from_props(getattr(solver.model, "mass", None)),
        "active": _active_mass_values(solver),
    }


def _serialize_stability_derivatives(derivs: Any) -> dict[str, Any]:
    """Convert :class:`StabilityDerivatives` to a JSON-friendly mapping."""
    return asdict(derivs)


def _surface_labels(solver: AVLSolver) -> list[str]:
    """Map lattice surface indices to human-readable labels."""
    return [solver_surface_name(solver.model, isurf) for isurf in range(int(solver.state.nsurf))]


def _cg_payload(solver: AVLSolver) -> dict[str, float]:
    """Return the active run-case center-of-gravity in geometry units."""
    state = solver.state
    return {
        "x": float(state.parval[C.IPXCG, 0]),
        "y": float(state.parval[C.IPYCG, 0]),
        "z": float(state.parval[C.IPZCG, 0]),
    }


def _build_lift_distribution_3d(solver: AVLSolver) -> dict[str, Any]:
    """Build per-strip 3D loading data for the web viewer overlay."""
    state = solver.state
    nsurf = int(state.nsurf)
    if nsurf <= 0 or int(state.nvor) <= 0:
        return {"surfaces": []}

    labels = _surface_labels(solver)
    surfaces: list[dict[str, Any]] = []

    for isurf in range(nsurf):
        lfload = getattr(state, "lfload", None)
        if lfload is not None and not bool(lfload[isurf]):
            continue

        j0 = int(state.jfrst[isurf])
        nj = int(state.nj[isurf])
        if nj <= 0:
            continue

        strips: list[dict[str, Any]] = []
        for jj in range(nj):
            j = j0 + jj
            if bool(state.lstripoff[j]) or float(state.wstrip[j]) == 0.0:
                continue

            i0 = int(state.ijfrst[j])
            nvc = int(state.nvstrp[j])
            if nvc <= 0:
                continue

            cl = float(state.cl_lstrp[j])
            points: list[dict[str, float]] = []
            for ivc in range(nvc):
                i = i0 + ivc
                points.append(
                    {
                        "x": float(state.rv[0, i]),
                        "y": float(state.rv[1, i]),
                        "z": float(state.rv[2, i]),
                        "cl": cl,
                    }
                )

            # The viewer uses strip normals to orient the 3D lift glyphs.
            strips.append(
                {
                    "ensy": float(state.ensy[j]),
                    "ensz": float(state.ensz[j]),
                    "points": points,
                }
            )

        if strips:
            surfaces.append({"name": labels[isurf], "strips": strips})

    return {"surfaces": surfaces}


def _build_trefftz_surface(
    state: AVLState,
    isurf: int,
    *,
    name: str,
    vee: float,
    cref: float,
) -> dict[str, Any] | None:
    """Build sorted Trefftz-plane strip arrays for one lifting surface."""
    lfload = getattr(state, "lfload", None)
    if lfload is not None and not bool(lfload[isurf]):
        return None

    j0 = int(state.jfrst[isurf])
    nj = int(state.nj[isurf])
    if nj <= 0:
        return None

    strip_indices = list(range(j0, j0 + nj))
    strip_indices.sort(key=lambda j: float(state.rle[1, j]))

    y: list[float] = []
    cl: list[float] = []
    clnorm: list[float] = []
    cnc: list[float] = []
    ai: list[float] = []

    for j in strip_indices:
        y.append(float(state.rle[1, j]))
        cl.append(float(state.cl_lstrp[j]))
        clnorm.append(float(state.clt_lstrp[j]))
        cnc.append(float(state.cnc[j]) / cref if cref > 0.0 else float(state.cnc[j]))
        if state.dwwake.size > j:
            ai.append(float(state.dwwake[j]))
        else:
            # Older states may not carry Trefftz downwash; circulation gives a
            # conservative induced-angle proxy for plotting only.
            i0 = int(state.ijfrst[j])
            nvc = int(state.nvstrp[j])
            if nvc <= 0 or vee <= 0.0:
                ai.append(0.0)
            else:
                gam_sum = float(state.gam[i0 : i0 + nvc].sum())
                chord = float(state.chord[j]) if state.chord[j] > 0.0 else 1.0
                ai.append(gam_sum / (vee * chord))

    return {"name": name, "y": y, "cl": cl, "clnorm": clnorm, "cnc": cnc, "ai": ai}


def _build_trefftz_data(solver: AVLSolver) -> dict[str, Any]:
    """Build Trefftz-plane span loading grouped by lifting surface."""
    state = solver.state
    nsurf = int(state.nsurf)
    if nsurf <= 0:
        return {
            "surfaces": [],
            "cref": float(state.cref),
            "bref": float(state.bref),
            "lift_3d": {"surfaces": []},
            "cg": _cg_payload(solver),
        }

    vee = float(state.parval[C.IPVEE, 0])
    cref = float(state.cref)
    labels = _surface_labels(solver)
    surfaces: list[dict[str, Any]] = []

    for isurf in range(nsurf):
        entry = _build_trefftz_surface(
            state,
            isurf,
            name=labels[isurf],
            vee=vee,
            cref=cref,
        )
        if entry is not None:
            surfaces.append(entry)

    lift_3d = _build_lift_distribution_3d(solver)
    return {
        "surfaces": surfaces,
        "cref": cref,
        "bref": float(state.bref),
        "lift_3d": lift_3d,
        "cg": _cg_payload(solver),
    }


def _build_surface_forces(solver: AVLSolver) -> list[dict[str, Any]]:
    """Return per-surface force coefficients from the latest solve."""
    state = solver.state
    labels = _surface_labels(solver)

    forces: list[dict[str, Any]] = []
    for isurf in range(int(state.nsurf)):
        forces.append(
            {
                "name": labels[isurf],
                "CL": float(state.clsurf[isurf]),
                "CD": float(state.cdsurf[isurf]),
                "CY": float(state.cysurf[isurf]),
                "Cl": float(state.cmsurf[0, isurf]),
                "Cm": float(state.cmsurf[1, isurf]),
                "Cn": float(state.cmsurf[2, isurf]),
            }
        )
    return forces


def _optional_positive_finite(value: float) -> float | None:
    """Return a JSON-safe positive scalar, or ``None`` when not applicable."""
    if value > 0.0 and math.isfinite(value):
        return float(value)
    return None


def _build_eigen_data(solver: AVLSolver) -> dict[str, Any]:
    """Serialize eigenanalysis output for the frontend."""
    result = solver.eigenvalues()
    return {
        "eigenvalues": [{"re": float(ev.real), "im": float(ev.imag)} for ev in result.eigenvalues],
        "modes": [
            {
                "name": mode.name,
                "frequency_hz": float(mode.frequency_hz),
                "damping_ratio": float(mode.damping_ratio),
                "time_constant": _optional_positive_finite(mode.time_constant),
                "period_s": _optional_positive_finite(mode.period_s),
                "time_to_half_s": _optional_positive_finite(mode.time_to_half_s),
                "eigenvalue": {"re": float(mode.eigenvalue.real), "im": float(mode.eigenvalue.imag)},
            }
            for mode in result.modes
        ],
    }


def _serialize_body_axis_derivatives(solver: AVLSolver) -> dict[str, Any]:
    """Convert body-axis derivative matrix to a JSON-friendly mapping."""
    derivs = compute_body_axis_derivatives(solver.state)
    payload = asdict(derivs)
    ncontrol = int(solver.state.ncontrol)
    control_names = list(solver.state.control_names[:ncontrol])
    for i, name in enumerate(control_names):
        row_idx = 6 + i
        if row_idx < len(payload["rows"]):
            payload["rows"][row_idx] = name
    return payload


def _build_results_extras(solver: AVLSolver) -> dict[str, Any]:
    """Attach run-case scalars and control deflections to the results payload."""
    state = solver.state
    # Web totals use the same NASA/sign convention as text reporting.
    dir_ = nasa_dir(state)
    wrot = state.wrot
    ncontrol = int(state.ncontrol)
    controls: dict[str, float] = {}
    for idx, name in enumerate(state.control_names[:ncontrol]):
        controls[name] = float(state.delcon[idx])

    moments = reported_totals(state)
    cl, cm, cn = moments["CM"]
    ir = 0
    rho = float(state.parval[C.IPRHO, ir])
    velocity = float(state.parval[C.IPVEE, ir])
    mass = float(state.parval[C.IPMASS, ir])
    inertia = np.array(
        [
            [state.parval[C.IPIXX, ir], state.parval[C.IPIXY, ir], state.parval[C.IPIZX, ir]],
            [state.parval[C.IPIXY, ir], state.parval[C.IPIYY, ir], state.parval[C.IPIYZ, ir]],
            [state.parval[C.IPIZX, ir], state.parval[C.IPIYZ, ir], state.parval[C.IPIZZ, ir]],
        ],
        dtype=np.float64,
    )
    q_pressure = 0.5 * rho * velocity**2
    sref_d = float(state.sref) * state.unitl * state.unitl
    bref_d = float(state.bref) * state.unitl
    cref_d = float(state.cref) * state.unitl
    force_body = q_pressure * sref_d * np.array(moments["CF"], dtype=np.float64)
    moment_body = q_pressure * sref_d * np.array([bref_d * cl, cref_d * cm, bref_d * cn], dtype=np.float64)
    linear_accel = force_body / mass if mass > 0.0 else np.full(3, math.nan, dtype=np.float64)
    omega_body = state.wrot * velocity / state.unitl
    try:
        angular_momentum = inertia @ omega_body
        rotational_accel = np.linalg.solve(inertia, moment_body - np.cross(omega_body, angular_momentum))
    except np.linalg.LinAlgError:
        rotational_accel = np.full(3, math.nan, dtype=np.float64)

    def finite_vector(values: np.ndarray) -> list[float | None]:
        return [float(value) if math.isfinite(float(value)) else None for value in values]

    return {
        "pb2V": dir_ * float(wrot[0]),
        "qc2V": float(wrot[1]),
        "rb2V": dir_ * float(wrot[2]),
        "controls": controls,
        "CDi": float(state.cdtot - state.cdvtot),
        "CDp": float(state.cdvtot),
        "e": float(state.spanef),
        "Cl": cl,
        "Cm": cm,
        "Cn": cn,
        "cref": float(state.cref),
        "bref": float(state.bref),
        "sref": float(state.sref),
        "xcg": float(state.parval[C.IPXCG, ir]),
        "rho": rho,
        "velocity": velocity,
        "linear_acceleration_body": finite_vector(linear_accel),
        "rotational_acceleration_body": finite_vector(rotational_accel),
    }


def _hinge_moment_dimensional(state: AVLState) -> bool:
    """Return True when rho and velocity define a physical dynamic pressure."""
    ir = 0
    rho = float(state.parval[C.IPRHO, ir])
    vee = float(state.parval[C.IPVEE, ir])
    # rho=V=1 is the solver default and indicates coefficient-style display.
    return not (rho == 1.0 and vee == 1.0)


def _hinge_moment_physical(chinge: float, state: AVLState) -> float:
    """Convert a hinge-moment coefficient to a physical moment (force·length units).

    Uses dimensional reference area and chord (``Sref * unitl^2``, ``Cref * unitl``)
    so models defined in non-SI geometry units (e.g. inches with ``Lunit`` in a
    ``.mass`` file) match AVL's trim/eigenmode unit conventions.
    """
    ir = 0
    rho = float(state.parval[C.IPRHO, ir])
    vee = float(state.parval[C.IPVEE, ir])
    que = 0.5 * rho * vee * vee
    unitl = float(state.unitl)
    sref_d = float(state.sref) * unitl * unitl
    cref_d = float(state.cref) * unitl
    return chinge * que * sref_d * cref_d


def _build_hinge_moments(solver: AVLSolver) -> dict[str, Any]:
    """Return hinge-moment coefficients and optional physical moments for each control."""
    state = solver.state
    dimensional = _hinge_moment_dimensional(state)
    controls: list[dict[str, Any]] = []
    for idx, name in enumerate(state.control_names):
        if idx >= state.chinge.size:
            break
        chinge = float(state.chinge[idx])
        row: dict[str, Any] = {"name": name, "Chinge": chinge}
        if dimensional:
            row["moment"] = _hinge_moment_physical(chinge, state)
        controls.append(row)
    return {
        "dimensional": dimensional,
        "moment_units": "force*length" if dimensional else None,
        "controls": controls,
    }


def _apply_default_trim(solver: AVLSolver) -> None:
    """Apply the Supra example trim setup (CL = 0.7, elevator trims Cm to zero)."""
    solver.set_parameter("cl", 0.7)
    solver.setup_trim(mode=1)
    solver.set_constraint("elevator", "cm", 0.0)


def create_session() -> SessionState:
    """Allocate a new session and register it in the global table."""
    session_id = str(uuid.uuid4())
    session = SessionState(session_id=session_id)
    sessions[session_id] = session
    return session


def touch_session(session: SessionState) -> None:
    """Refresh activity timestamp and clear disconnect marker."""
    session.last_active = time.time()
    session.disconnected_at = None


def mark_disconnected(session: SessionState) -> None:
    """Record disconnect time so the session can be evicted later."""
    session.disconnected_at = time.time()


async def cleanup_sessions() -> None:
    """Evict stale sessions (disconnected longer than TTL)."""
    while True:
        await asyncio.sleep(60.0)
        now = time.time()
        expired = [
            sid
            for sid, sess in sessions.items()
            if sess.disconnected_at is not None and now - sess.disconnected_at > _SESSION_TTL_SECONDS
        ]
        for sid in expired:
            sessions.pop(sid, None)


def _example_flight_params(spec: dict[str, Any]) -> dict[str, Any]:
    """Return run-case defaults for a built-in example.

    When an example includes a ``.mass`` file, CG and inertia come from that
    file and must not be overwritten by hard-coded GUI defaults.
    """
    params = dict(_DEFAULT_FLIGHT_PARAMS)
    if not spec.get("mass"):
        params["xcg"] = 3.75
    return params


def _solver_from_example(name: str) -> AVLSolver:
    """Load a built-in example aircraft by name."""
    key = name.strip().lower()
    if key not in EXAMPLES:
        raise KeyError(f"Unknown example: {name}")
    spec = EXAMPLES[key]
    return AVLSolver(
        spec["avl"],
        mass_file=spec["mass"] if spec.get("mass") else None,
        **_example_flight_params(spec),
    )


def _solver_from_text(
    session: SessionState,
    avl_text: str,
    mass_text: str | None = None,
    *,
    warnings: list[str] | None = None,
) -> AVLSolver:
    """Create a solver from uploaded AVL (and optional mass) file text."""
    work_dir = _materialize_airfoil_work_dir(session, avl_text)
    model = prepare_model(parse_avl(avl_text), base_dir=work_dir)
    mass_props: MassProperties | None = None
    if mass_text:
        mass_props = parse_mass_text(mass_text)
        if not mass_props.loaded:
            raise ValueError("Mass file text could not be parsed.")

    if warnings is not None:
        _append_missing_afil_warnings(session, warnings)

    return _create_solver(model, mass_props=mass_props, **_DEFAULT_FLIGHT_PARAMS)


def load_example(session: SessionState, name: str) -> dict[str, Any]:
    """Load a built-in example, apply default trim, solve, and package responses."""
    session.warnings = []
    key = name.strip().lower()
    spec = EXAMPLES.get(key)
    avl_path = spec["avl"] if spec else None
    session.pending_avl_text = avl_path.read_text(encoding="utf-8", errors="replace") if avl_path else None
    mass_path = spec["mass"] if spec and spec.get("mass") else None
    session.pending_mass_text = (
        mass_path.read_text(encoding="utf-8", errors="replace") if mass_path else None
    )
    session.airfoil_base_dir = avl_path.parent if avl_path else None
    session.extra_airfoil_paths = []
    session.invalid_airfoil_paths.clear()
    session.solver = _solver_from_example(name)
    _apply_default_trim(session.solver)
    session.solver.execute_run(max_iter=20)
    return build_solve_responses(session, include_geometry=True, example_name=name)


def rebuild_solver_from_pending(session: SessionState) -> dict[str, Any]:
    """Create or refresh the solver from uploaded AVL/mass text (no solve)."""
    if not session.pending_avl_text:
        raise ValueError("No AVL text uploaded.")
    session.warnings = []
    session.solver = _solver_from_text(
        session,
        session.pending_avl_text,
        session.pending_mass_text,
        warnings=session.warnings,
    )
    geometry = model_to_geometry(session.solver.model, session.solver.state, include_dcp=False)
    meta = _model_meta(session.solver, session)
    meta["warnings"] = list(session.warnings)
    meta["mass_props"] = _mass_properties_payload(session.solver)
    response = {"type": "model_loaded", "geometry": geometry, "meta": meta}
    if session.pending_avl_text is not None:
        response["avl_text"] = session.pending_avl_text
    if session.pending_mass_text is not None:
        response["mass_text"] = session.pending_mass_text
    return response


def upload_airfoil(session: SessionState, path: str, text: str) -> dict[str, Any]:
    """Store uploaded airfoil file text and rebuild the pending model."""
    norm = normalize_airfoil_path(path)
    if not norm:
        raise ValueError("Airfoil path is empty.")
    if not _airfoil_text_valid(text):
        session.invalid_airfoil_paths.add(norm)
        session.airfoil_files.pop(norm, None)
        raise ValueError(f"Could not parse airfoil coordinates in '{norm}'.")

    session.invalid_airfoil_paths.discard(norm)
    session.airfoil_files[norm] = text
    if not session.pending_avl_text:
        return {
            "type": "afil_dependencies",
            "dependencies": build_afil_dependencies(session),
        }
    return rebuild_solver_from_pending(session)


def add_airfoil_dependency(session: SessionState, path: str) -> dict[str, Any]:
    """Register a manually added AFIL dependency path."""
    norm = normalize_airfoil_path(path)
    if not norm:
        raise ValueError("Airfoil path is empty.")
    if norm not in session.extra_airfoil_paths:
        session.extra_airfoil_paths.append(norm)
    if session.pending_avl_text:
        return rebuild_solver_from_pending(session)
    return {
        "type": "afil_dependencies",
        "dependencies": build_afil_dependencies(session),
    }


def load_uploaded_model(session: SessionState) -> dict[str, Any]:
    """Build a solver from pending uploaded AVL/mass text and run it.

    Trim constraints come from the WebSocket ``set_constraint`` messages sent
    by the client; do not apply example-specific defaults here.
    """
    if not session.pending_avl_text:
        raise ValueError("No AVL text uploaded.")
    session.warnings = []
    session.airfoil_base_dir = None
    session.solver = _solver_from_text(
        session,
        session.pending_avl_text,
        session.pending_mass_text,
        warnings=session.warnings,
    )
    session.solver.execute_run(max_iter=20)
    return build_solve_responses(session, include_geometry=True)


def run_solve(session: SessionState) -> dict[str, Any]:
    """Execute the solver and return all post-solve payloads."""
    if session.solver is None:
        raise RuntimeError("No model loaded.")
    session.solver.execute_run(max_iter=20)
    return build_solve_responses(session, include_geometry=False)


def build_solve_responses(
    session: SessionState,
    *,
    include_geometry: bool,
    example_name: str | None = None,
) -> dict[str, Any]:
    """Assemble server messages after a model load or solve."""
    if session.solver is None:
        raise RuntimeError("No model loaded.")

    solver = session.solver
    messages: list[dict[str, Any]] = []

    geometry = model_to_geometry(solver.model, solver.state, include_dcp=True)
    meta = _model_meta(solver, session)
    meta["warnings"] = list(session.warnings)
    meta["mass_props"] = _mass_properties_payload(solver)
    if example_name:
        meta["example"] = example_name

    # Message order is part of the UI protocol: geometry/Cp first, then scalar
    # results, derivative tables, Trefftz/eigen plots, and per-surface forces.
    if include_geometry:
        model_loaded = {"type": "model_loaded", "geometry": geometry, "meta": meta}
        if session.pending_avl_text is not None:
            model_loaded["avl_text"] = session.pending_avl_text
        if session.pending_mass_text is not None:
            model_loaded["mass_text"] = session.pending_mass_text
        messages.append(model_loaded)
    else:
        messages.append({"type": "cp_update", "geometry": geometry})

    body_axis = _serialize_body_axis_derivatives(solver)
    results_payload = {
        "type": "results",
        **solver.get_results(),
        **_build_results_extras(solver),
        "body_axis": body_axis,
        "hinge_moments": _build_hinge_moments(solver),
        "mass_props": _mass_properties_payload(solver),
    }
    messages.append(results_payload)
    messages.append(
        {
            "type": "stability_derivs",
            **_serialize_stability_derivatives(solver.get_stability_derivatives()),
            "body_axis": body_axis,
            "cref": float(solver.state.cref),
            "xcg": float(solver.state.parval[C.IPXCG, 0]),
        }
    )
    messages.append({"type": "trefftz_data", **_build_trefftz_data(solver)})
    messages.append({"type": "eigen_data", **_build_eigen_data(solver)})
    messages.append({"type": "surface_forces", "surfaces": _build_surface_forces(solver)})

    return {"messages": messages}
