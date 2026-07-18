/**
 * OpenAVL Web GUI — WebSocket client, panel wiring, and auto-solve debounce.
 */

import { AircraftViewer3D } from "./viewer3d.js";
import { initPlots, updateTrefftzPlot, updateEigenmodesPlot } from "./plots.js";

const DEBOUNCE_MS = 500;
const STABILITY_ROWS = ["CL", "CD", "CY", "Cl", "Cm", "Cn"];
const STABILITY_COLS = ["a", "b", "p", "q", "r"];
const MASS_FIELDS = ["mass", "xcg", "ycg", "zcg", "ixx", "iyy", "izz", "ixy", "ixz", "iyz"];
const FLIGHT_PARAM_KEYS = ["mass", "xcg", "ycg", "zcg", "rho", "gravity", "cd0", "velocity", "cl"];
const RUN_CASE_COLORS = ["#7fb3ff", "#6fcf97", "#f2c94c", "#bb86fc", "#ff8a65", "#56ccf2"];
const VARIABLE_OPTIONS = ["alpha", "beta", "pb/2V", "qc/2V", "rb/2V"];
const DEFAULT_CONTROL_OPTIONS = ["elevator", "aileron", "rudder", "flap"];
const CONSTRAINT_OPTIONS = ["cl", "cy", "cll", "cm", "cn", "alpha", "beta", "pb/2V", "qc/2V", "rb/2V"];
/** Body-axis state rows (u–r); control rows follow in the payload. */
const BODY_AXIS_STATE_ROW_COUNT = 6;
/** Columns whose backend values are per-radian (α, β); p/q/r are nondimensional. */
const ANGULAR_DERIV_COLS = new Set(["a", "b"]);
/** Body-axis rows equivalent to angle derivatives (v ≈ β, w ≈ α). */
const BODY_AXIS_ANGLE_ROWS = new Set(["v", "w"]);
const RAD_TO_DEG = Math.PI / 180;

/** @type {WebSocket|null} */
let ws = null;
let reconnectTimer = null;
let solveDebounce = null;
let isSolving = false;
let hasAutoLoadedExample = false;
/** @type {"per_rad"|"per_deg"} */
let derivDisplayUnit = "per_rad";
/** @type {"body"|"stability"} */
let controlDerivAxis = "stability";
let selectedRunCaseIndex = -1;
let controlOptions = [...DEFAULT_CONTROL_OPTIONS];
let nextSolveId = 1;
let latestExecutedSnapshot = null;
let activeModelKey = null;
const pendingSolves = new Map();
const SOLVE_RESULT_TYPES = new Set([
  "cp_update",
  "results",
  "stability_derivs",
  "trefftz_data",
  "eigen_data",
  "surface_forces",
  "hinge_moments",
]);

const viewer = new AircraftViewer3D(document.getElementById("viewer3d"));

const els = {
  wsStatus: document.getElementById("ws-status"),
  errorBanner: document.getElementById("error-banner"),
  avlEditor: document.getElementById("avl-editor"),
  massEditor: document.getElementById("mass-editor"),
  constraintsList: document.getElementById("constraints-list"),
  runCaseList: document.getElementById("run-case-list"),
  runCasesMeta: document.getElementById("run-cases-meta"),
  massPropsMeta: document.getElementById("mass-props-meta"),
  totalForcesGrid: document.getElementById("total-forces-grid"),
  stabilityTable: document.querySelector("#stability-table tbody"),
  bodyAxisGrid: document.getElementById("body-axis-grid"),
  controlSurfaceGrid: document.getElementById("control-surface-grid"),
  surfaceForcesTable: document.querySelector("#surface-forces-table tbody"),
  hingeMomentsGrid: document.getElementById("hinge-moments-grid"),
  eigenanalysisTable: document.querySelector("#eigenanalysis-table tbody"),
  afilDepsList: document.getElementById("afil-deps-list"),
  afilDepsEmpty: document.getElementById("afil-deps-empty"),
};

/**
 * Scale a per-radian derivative when the display unit is 1/deg.
 *
 * @param {boolean} isAngular
 * @param {number} raw
 * @returns {number}
 */
function displayAngularDerivValue(isAngular, raw) {
  const value = Number(raw);
  if (!Number.isFinite(value)) return 0;
  if (derivDisplayUnit === "per_deg" && isAngular) {
    return value * RAD_TO_DEG;
  }
  return value;
}

/**
 * Convert a stability derivative for display.
 *
 * OpenAVL returns angle derivatives (columns α, β) per radian from the chain
 * rule in ``compute_stability_derivatives``. Rate derivatives (p, q, r) are
 * already nondimensional w.r.t. pb/2V, qc/2V, rb/2V and must not be scaled.
 *
 * @param {string} col
 * @param {number} raw
 * @returns {number}
 */
function displayDerivValue(col, raw) {
  return displayAngularDerivValue(ANGULAR_DERIV_COLS.has(col), raw);
}

/**
 * Convert a body-axis state derivative row for display unit selection.
 *
 * Rows ``v`` and ``w`` are treated like β and α angle derivatives; rate rows
 * ``p``/``q``/``r`` are left unchanged.
 *
 * @param {string} rowLabel
 * @param {number} raw
 * @returns {number}
 */
function displayBodyAxisValue(rowLabel, raw) {
  const row = String(rowLabel ?? "").trim().toLowerCase();
  return displayAngularDerivValue(BODY_AXIS_ANGLE_ROWS.has(row), raw);
}

/**
 * Convert a control-surface derivative for display unit selection.
 *
 * Control deflection derivatives are per radian in the backend.
 *
 * @param {number} raw
 * @returns {number}
 */
function displayControlDerivValue(raw) {
  return displayAngularDerivValue(true, raw);
}

/**
 * Toggle collapsible panel bodies when a panel header is clicked.
 *
 * Native ``<select>`` menus often emit a trailing click on the header after an
 * option is chosen. Suppress that ghost click so axis/unit dropdowns can update
 * the panel body without immediately collapsing it.
 */
function initCollapsiblePanels() {
  document.querySelectorAll(".panel").forEach((panel) => {
    const header = panel.querySelector(".panel-header");
    if (!header) return;

    let suppressCollapseUntil = 0;

    header.querySelectorAll("select, button, input, textarea, a").forEach((el) => {
      for (const type of ["pointerdown", "mousedown", "click"]) {
        el.addEventListener(type, (ev) => ev.stopPropagation());
      }
      if (el instanceof HTMLSelectElement) {
        el.addEventListener("change", () => {
          suppressCollapseUntil = performance.now() + 500;
        });
      }
    });

    header.addEventListener("click", (ev) => {
      if (performance.now() < suppressCollapseUntil) return;
      if (ev.target.closest("select, button, input, textarea, a")) return;
      panel.classList.toggle("collapsed");
    });
  });
}

/** Default trim constraint: alpha → CL = 0.7 */
const constraints = [{ variable: "alpha", constraint: "cl", value: 0.7 }];
const runCases = [];
/** @type {"reset"|"run"|null} */
let modelLoadIntent = null;

/**
 * Send a JSON message to the server when the socket is open.
 *
 * @param {Record<string, unknown>} msg
 */
function send(msg) {
  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(msg));
  }
}

/** Capture the editable flight, mass, and constraint values currently shown. */
function captureEditableState() {
  const inputs = {};
  document.querySelectorAll("#flight-form input[data-key]").forEach((input) => {
    if (input.value.trim() === "") {
      inputs[input.dataset.key] = null;
      return;
    }
    const value = Number(input.value);
    if (Number.isFinite(value)) inputs[input.dataset.key] = value;
  });
  document.querySelectorAll("#mass-props-grid input[data-mass-key]").forEach((input) => {
    if (input.value.trim() === "") {
      inputs[input.dataset.massKey] = null;
      return;
    }
    const value = Number(input.value);
    if (Number.isFinite(value)) inputs[input.dataset.massKey] = value;
  });
  return {
    inputs,
    constraints: constraints.map((row) => ({ ...row })),
  };
}

/** Return only numeric values that can be applied to the solver. */
function solverInputs(inputs) {
  return Object.fromEntries(
    Object.entries(inputs ?? {}).filter(([, value]) => value !== null && Number.isFinite(Number(value))),
  );
}

/**
 * Send a correlated solver execution and remember the exact submitted state.
 *
 * @param {Record<string, unknown>} [command]
 * @param {{inputs: Record<string, number|null>, constraints: Array<Record<string, unknown>>}|null} [state]
 */
function requestExecution(command = { type: "solve" }, state = captureEditableState()) {
  const solveId = `solve-${nextSolveId++}`;
  const entry = selectedRunCaseIndex >= 0 ? runCases[selectedRunCaseIndex] : null;
  const caseId = command.type === "solve" ? (entry?.id ?? null) : null;
  pendingSolves.set(solveId, {
    caseId,
    state,
    messages: [],
  });
  if (command.type === "solve" && state) {
    send({
      type: "apply_run_case",
      inputs: solverInputs(state.inputs),
      constraints: state.constraints,
    });
  }
  send({ ...command, case_id: caseId, solve_id: solveId });
}

/** Schedule a debounced solve request (500 ms) so typing can finish first. */
function scheduleSolve() {
  clearTimeout(solveDebounce);
  solveDebounce = setTimeout(() => requestExecution(), DEBOUNCE_MS);
}

/**
 * Update solving state and disable example load while a solve is in flight.
 *
 * @param {boolean} solving
 */
function setSolving(solving) {
  isSolving = solving;
  document.body.classList.toggle("is-solving", solving);
  const btn = document.getElementById("btn-load-supra-demo");
  if (btn) btn.disabled = solving;
}

/**
 * Update the connection status pill in the header.
 *
 * @param {"connecting"|"connected"|"solving"|"error"|"disconnected"} state
 * @param {string} [label]
 */
function setStatus(state, label) {
  const pill = els.wsStatus;
  pill.className = `status-pill ${state}`;
  pill.querySelector(".label").textContent = label ?? state;
}

/**
 * Show or hide the error banner toast.
 *
 * @param {string|null} message
 */
function showError(message) {
  if (!message) {
    els.errorBanner.classList.remove("visible");
    els.errorBanner.textContent = "";
    return;
  }
  els.errorBanner.textContent = message;
  els.errorBanner.classList.add("visible");
  setStatus("error", "Error");
}

/**
 * Format a numeric value for display panels.
 *
 * @param {unknown} value
 * @param {number} [digits]
 * @returns {string}
 */
function fmt(value, digits = 4) {
  if (value === null || value === undefined || value === "") return "—";
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : String(value);
}

/**
 * Escape text before inserting it into HTML strings.
 *
 * @param {unknown} value
 * @returns {string}
 */
function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

/**
 * Format with a leading sign for compact derivative grids.
 *
 * @param {unknown} value
 * @param {number} [digits]
 * @returns {string}
 */
function fmtSigned(value, digits = 6) {
  if (value === null || value === undefined || value === "") return "-";
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  const abs = Math.abs(n).toFixed(digits);
  return `${n < 0 ? "-" : " "}${abs}`;
}

/**
 * Round a number to a fixed number of significant figures.
 *
 * @param {number} n
 * @param {number} sigFigs
 * @returns {number}
 */
function roundToSigFig(n, sigFigs) {
  if (n === 0) return 0;
  const abs = Math.abs(n);
  const order = Math.floor(Math.log10(abs));
  const scale = 10 ** (sigFigs - 1 - order);
  const rounded = Math.round(abs * scale) / scale;
  return n < 0 ? -rounded : rounded;
}

/**
 * Convert a positive magnitude to plain decimal with exactly sigFigs digits.
 *
 * Uses ``toExponential`` internally so the mantissa never gains extra digits from
 * ``toFixed`` padding (the source of 7–8 decimal-place control-surface values).
 *
 * @param {number} abs
 * @param {number} sigFigs
 * @returns {string}
 */
function formatAbsSigFigPlain(abs, sigFigs) {
  if (!Number.isFinite(abs) || abs < 5e-5) return "0.0000";

  const rounded = roundToSigFig(abs, sigFigs);
  if (rounded === 0 || rounded < 5e-5) return "0.0000";

  const [coeff, expStr] = rounded.toExponential(sigFigs - 1).split("e");
  const exponent = parseInt(expStr, 10);
  const digits = coeff.replace(".", "");

  if (exponent >= 0) {
    const point = exponent + 1;
    if (point >= digits.length) {
      return digits + "0".repeat(point - digits.length);
    }
    const head = digits.slice(0, point);
    const tail = digits.slice(point);
    return tail ? `${head}.${tail}` : head;
  }

  return `0.${"0".repeat(-exponent - 1)}${digits}`;
}

/**
 * Format a signed value to a fixed number of significant figures in plain decimal.
 *
 * Values below 0.00005 display as ``0.0000`` instead of scientific notation.
 *
 * @param {unknown} value
 * @param {number} [sigFigs]
 * @returns {string}
 */
function fmtSignedSigFig(value, sigFigs = 4) {
  if (value === null || value === undefined || value === "") return "-";
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  if (n === 0) return " 0.0000";

  const plain = formatAbsSigFigPlain(Math.abs(n), sigFigs);
  if (plain === "0.0000") return " 0.0000";

  const rounded = roundToSigFig(n, sigFigs);
  return `${rounded < 0 ? "-" : " "}${plain}`;
}

/**
 * Set a numeric input without adding trailing zeros.
 *
 * @param {HTMLInputElement|null} input
 * @param {unknown} value
 */
function setNumericInput(input, value) {
  if (!input) return;
  const n = Number(value);
  input.value = Number.isFinite(n) ? String(Number(n.toPrecision(10))) : "";
}

/**
 * Render one compact matrix cell
 *
 * @param {string} text
 * @param {string} cls
 * @param {string} [title]
 * @returns {string}
 */
function matrixCell(text, cls, title = "") {
  const titleAttr = title ? ` title="${escapeHtml(title)}"` : "";
  return `<div class="matrix-cell ${cls}"${titleAttr}>${text}</div>`;
}

/**
 * Build and render a constraint row in the constraints panel.
 *
 * @param {{ variable: string, constraint: string, value: number }} row
 * @param {number} index
 */
function renderConstraintRow(row, index) {
  const div = document.createElement("div");
  div.className = "constraint-row";
  div.dataset.index = String(index);

  const varSel = document.createElement("select");
  varSel.className = "constraint-variable";
  for (const v of [...VARIABLE_OPTIONS, ...controlOptions]) {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    if (v === row.variable) opt.selected = true;
    varSel.appendChild(opt);
  }

  const conSel = document.createElement("select");
  conSel.className = "constraint-type";
  for (const c of [...CONSTRAINT_OPTIONS, ...controlOptions]) {
    const opt = document.createElement("option");
    opt.value = c;
    opt.textContent = c;
    if (c === row.constraint) opt.selected = true;
    conSel.appendChild(opt);
  }

  const valInput = document.createElement("input");
  valInput.type = "number";
  valInput.className = "constraint-value";
  valInput.step = "any";
  valInput.value = String(row.value);

  const removeBtn = document.createElement("button");
  removeBtn.type = "button";
  removeBtn.textContent = "×";
  removeBtn.title = "Remove";
  removeBtn.disabled = constraints.length <= 1;
  removeBtn.addEventListener("click", () => {
    if (constraints.length <= 1) return;
    constraints.splice(index, 1);
    renderAllConstraints();
    scheduleSolve();
  });

  const onChange = () => {
    constraints[index] = {
      variable: varSel.value,
      constraint: conSel.value,
      value: Number(valInput.value),
    };
    scheduleSolve();
  };

  varSel.addEventListener("change", onChange);
  conSel.addEventListener("change", onChange);
  valInput.addEventListener("input", onChange);

  div.append(varSel, conSel, valInput, removeBtn);
  return div;
}

/** Re-render all constraint rows from the `constraints` array. */
function renderAllConstraints() {
  els.constraintsList.innerHTML = "";
  constraints.forEach((row, i) => {
    els.constraintsList.appendChild(renderConstraintRow(row, i));
  });
}

/**
 * Build default trim rows for a loaded model.
 *
 * Longitudinal level-flight trim always targets CL = 0.7 via alpha. When the
 * geometry defines an elevator control, also trim pitch moment with it.
 *
 * @param {Record<string, unknown>} [meta]
 * @returns {Array<{ variable: string, constraint: string, value: number }>}
 */
function defaultConstraintsForModel(meta = {}) {
  const rows = [{ variable: "alpha", constraint: "cl", value: 0.7 }];
  const controls = new Set(
    (Array.isArray(meta.controls) ? meta.controls : []).map((name) => String(name).toLowerCase()),
  );
  if (controls.has("elevator")) {
    rows.push({ variable: "elevator", constraint: "cm", value: 0 });
  }
  return rows;
}

/**
 * Replace the constraint panel with defaults appropriate for the loaded model.
 *
 * @param {Record<string, unknown>} [meta]
 */
function resetConstraintsFromModel(meta = {}) {
  const next = defaultConstraintsForModel(meta);
  constraints.splice(0, constraints.length, ...next);
  renderAllConstraints();
}

/**
 * Capture the last successful execution as a local run case.
 *
 * @param {string} name
 * @returns {Record<string, unknown>}
 */
function captureRunCase(name) {
  const current = captureEditableState();
  const executed = latestExecutedSnapshot
    ? structuredClone(latestExecutedSnapshot)
    : { ...current, messages: [] };
  return {
    id: crypto.randomUUID(),
    name,
    color: RUN_CASE_COLORS[runCases.length % RUN_CASE_COLORS.length],
    executed,
  };
}

/** Update the run-case metadata line. */
function updateRunCasesMeta() {
  if (!els.runCasesMeta) return;
  if (!runCases.length) {
    els.runCasesMeta.textContent = "No run cases loaded.";
    return;
  }
  els.runCasesMeta.textContent = `${runCases.length} run case(s)`;
}

/** Render the local run-case selector list. */
function renderRunCasesList() {
  if (!els.runCaseList) return;
  els.runCaseList.innerHTML = "";
  if (!runCases.length) {
    selectedRunCaseIndex = -1;
    updateRunCasesMeta();
    return;
  }
  if (selectedRunCaseIndex < 0 || selectedRunCaseIndex >= runCases.length) selectedRunCaseIndex = 0;

  runCases.forEach((entry, index) => {
    const row = document.createElement("div");
    row.className = `run-case-item${index === selectedRunCaseIndex ? " active" : ""}`;
    row.dataset.index = String(index);

    const title = document.createElement("input");
    title.type = "text";
    title.className = "run-case-title";
    title.value = entry.name || `Case ${index + 1}`;
    title.addEventListener("click", (ev) => ev.stopPropagation());
    title.addEventListener("input", () => {
      entry.name = title.value;
      updateRunCasesMeta();
    });
    title.addEventListener("change", () => {
      entry.name = title.value.trim() || `Case ${index + 1}`;
      title.value = entry.name;
    });

    const color = document.createElement("input");
    color.type = "color";
    color.className = "run-case-color";
    color.value = entry.color || RUN_CASE_COLORS[index % RUN_CASE_COLORS.length];
    color.addEventListener("click", (ev) => ev.stopPropagation());
    color.addEventListener("input", () => {
      entry.color = color.value;
    });

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "run-case-delete";
    remove.title = "Delete case";
    remove.textContent = "×";
    remove.addEventListener("click", (ev) => {
      ev.stopPropagation();
      const wasActive = index === selectedRunCaseIndex;
      runCases.splice(index, 1);
      if (!runCases.length) {
        selectedRunCaseIndex = -1;
        renderRunCasesList();
      } else if (wasActive) {
        selectedRunCaseIndex = -1;
        selectRunCase(Math.min(index, runCases.length - 1));
      } else {
        if (index < selectedRunCaseIndex) selectedRunCaseIndex -= 1;
        renderRunCasesList();
      }
    });

    row.addEventListener("click", () => selectRunCase(index));
    row.append(title, color, remove);
    els.runCaseList.appendChild(row);
  });
  updateRunCasesMeta();
}

/**
 * Apply one local run case to the UI and solver session.
 *
 * @param {number} index
 */
function selectRunCase(index) {
  const entry = runCases[index];
  if (!entry?.executed) return;
  clearTimeout(solveDebounce);
  selectedRunCaseIndex = index;

  for (const [key, value] of Object.entries(entry.executed.inputs ?? {})) {
    const flightInput = document.querySelector(`#flight-form input[data-key="${key}"]`);
    const massInput = document.querySelector(`#mass-props-grid input[data-mass-key="${key}"]`);
    if (value === null) {
      if (flightInput) flightInput.value = "";
      if (massInput) massInput.value = "";
    } else {
      setNumericInput(flightInput, value);
      setNumericInput(massInput, value);
    }
  }

  constraints.splice(
    0,
    constraints.length,
    ...(entry.executed.constraints ?? []).map((row) => ({ ...row })),
  );
  renderAllConstraints();
  send({
    type: "apply_run_case",
    inputs: solverInputs(entry.executed.inputs),
    constraints: entry.executed.constraints ?? [],
  });
  for (const message of entry.executed.messages ?? []) {
    handleMessage({ ...structuredClone(message), replay: true });
  }
  setSolving(false);
  setStatus("connected", "Connected");
  renderRunCasesList();
}

/** Wire flight-condition inputs to `set_flight_param` messages. */
function bindFlightInputs() {
  document.querySelectorAll("#flight-form input[data-key]").forEach((input) => {
    input.addEventListener("input", () => {
      const key = input.dataset.key;
      const raw = input.value.trim();
      if (raw === "") return;
      const value = Number(raw);
      if (!Number.isFinite(value)) return;
      const massInput = document.querySelector(`#mass-props-grid input[data-mass-key="${key}"]`);
      setNumericInput(massInput, value);
      scheduleSolve();
    });
  });
}

/** Wire active mass-property inputs to run-case parameters. */
function bindMassInputs() {
  document.querySelectorAll("#mass-props-grid input[data-mass-key]").forEach((input) => {
    input.addEventListener("input", () => {
      const key = input.dataset.massKey;
      const value = Number(input.value);
      if (!key || !Number.isFinite(value)) return;
      const flightInput = document.querySelector(`#flight-form input[data-key="${key}"]`);
      setNumericInput(flightInput, value);
      scheduleSolve();
    });
  });
}

/**
 * Mirror active run-case parameters into the flight-condition inputs.
 *
 * @param {Record<string, unknown>|null|undefined} active
 * @param {{ notify?: boolean }} [options]
 */
function applyFlightParamsFromActive(active, { notify = false } = {}) {
  if (!active) return;
  for (const key of FLIGHT_PARAM_KEYS) {
    const value = Number(active[key]);
    if (!Number.isFinite(value)) continue;
    const flightInput = document.querySelector(`#flight-form input[data-key="${key}"]`);
    setNumericInput(flightInput, value);
    if (notify) send({ type: "set_flight_param", key, value });
  }
}

/**
 * Render mass properties in "Mass file" and "Active" columns.
 *
 * @param {{ file?: Record<string, unknown>|null, active?: Record<string, unknown>|null }} payload
 * @param {{ syncFlight?: boolean }} [options]
 */
function updateMassProperties(payload = {}, { syncFlight = true } = {}) {
  const file = payload.file ?? null;
  const active = payload.active ?? null;
  for (const key of MASS_FIELDS) {
    setNumericInput(document.getElementById(`mass-file-${key}`), file?.[key]);
    setNumericInput(document.getElementById(`mass-active-${key}`), active?.[key]);
  }

  if (syncFlight) applyFlightParamsFromActive(active);
  viewer.updateComponentMasses(file?.components ?? []);

  if (els.massPropsMeta) {
    if (file) {
      const units = [file.lunit, file.munit, file.tunit].filter((v) => v !== undefined).join(", ");
      els.massPropsMeta.textContent = units ? `Mass file loaded. Units: ${units}` : "Mass file loaded.";
    } else {
      els.massPropsMeta.textContent = "No mass file loaded.";
    }
  }
}

/** @type {Array<{path: string, status: string, manual?: boolean}>} */
let afilDependencies = [];

/**
 * Normalize an AFIL dependency path for display and server messages.
 *
 * @param {string} path
 * @returns {string}
 */
function normalizeAfilPath(path) {
  return String(path ?? "").trim().replace(/^["']+|["']+$/g, "").trim();
}

/**
 * Human-readable status label for an AFIL dependency row.
 *
 * @param {string} status
 * @returns {string}
 */
function afilStatusLabel(status) {
  switch (status) {
    case "ready":
      return "Ready";
    case "invalid":
      return "Parse error";
    case "missing":
    default:
      return "Missing";
  }
}

/**
 * Render the AFIL file dependencies panel.
 *
 * @param {Array<{path?: string, status?: string, manual?: boolean}>} [deps]
 */
function renderAfilDependencies(deps = afilDependencies) {
  afilDependencies = Array.isArray(deps) ? deps : [];
  if (!els.afilDepsList) return;

  els.afilDepsList.innerHTML = "";
  if (els.afilDepsEmpty) {
    els.afilDepsEmpty.style.display = afilDependencies.length ? "none" : "block";
  }

  afilDependencies.forEach((dep) => {
    const path = normalizeAfilPath(dep.path);
    if (!path) return;

    const row = document.createElement("div");
    row.className = `afil-dep-row ${dep.status ?? "missing"}`;

    const name = document.createElement("div");
    name.className = "afil-dep-name";
    name.title = path;
    name.textContent = dep.manual ? `${path} (manual)` : path;

    const status = document.createElement("div");
    status.className = "afil-dep-status";
    status.textContent = afilStatusLabel(dep.status);

    const uploadBtn = document.createElement("button");
    uploadBtn.type = "button";
    uploadBtn.textContent = "Upload";
    uploadBtn.className = "btn-upload-afil";

    const uploadInput = document.createElement("input");
    uploadInput.type = "file";
    uploadInput.accept = ".dat,.txt,text/plain";
    uploadInput.hidden = true;

    uploadBtn.addEventListener("click", () => uploadInput.click());
    uploadInput.addEventListener("change", async () => {
      const file = uploadInput.files?.[0];
      uploadInput.value = "";
      if (!file) return;
      try {
        const text = await file.text();
        send({ type: "upload_airfoil", path, text });
        scheduleSolve();
      } catch (err) {
        showError(err instanceof Error ? err.message : "Failed to read airfoil file");
      }
    });

    row.appendChild(name);
    row.appendChild(status);
    row.appendChild(uploadBtn);
    row.appendChild(uploadInput);
    els.afilDepsList.appendChild(row);
  });
}

/** Last merged aircraft metadata shown in the AIRCRAFT panel. */
let aircraftMeta = {};

/** Update available control names from model metadata. */
function updateControlOptions(controls) {
  if (!Array.isArray(controls)) return;
  const seen = new Set();
  controlOptions = [...controls, ...DEFAULT_CONTROL_OPTIONS]
    .map((name) => String(name ?? "").trim())
    .filter((name) => {
      const key = name.toLowerCase();
      if (!name || seen.has(key)) return false;
      seen.add(key);
      return true;
    });
}

/** Populate aircraft metadata labels from a `model_loaded` meta object. */
function updateMeta(meta = {}) {
  aircraftMeta = { ...aircraftMeta, ...meta };
  updateControlOptions(meta.controls);
  const m = aircraftMeta;
  const set = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val ?? "—";
  };
  set("meta-name", m.name ?? m.title);
  set("meta-nsurf", m.nsurf ?? m.surfaces ?? m.NSURF);
  set("meta-nstrip", m.nstrip ?? m.strips ?? m.NSTRIP);
  set("meta-nvor", m.nvor ?? m.vortices ?? m.NVOR);
  set("meta-sref", fmt(m.sref ?? m.SREF, 3));
  set("meta-cref", fmt(m.cref ?? m.CREF, 3));
  set("meta-bref", fmt(m.bref ?? m.BREF, 3));
}

/**
 * Render one cell in the 4-column Total Forces grid.
 *
 * @param {string} label
 * @param {unknown} value
 * @param {number} [digits]
 * @param {string} [cls]
 * @returns {string}
 */
function forceCell(label, value, digits = 4, cls = "force-cell") {
  return `<div class="${cls}">${label} = <strong>${fmt(value, digits)}</strong></div>`;
}

/**
 * Compute neutral point from stability derivatives and run-case reference data.
 *
 * @param {Record<string, unknown>|null|undefined} d
 * @param {Record<string, unknown>|null|undefined} results
 * @returns {number|null}
 */
function neutralPointValue(d, results = {}) {
  const cla = Number(d?.CL_a ?? d?.CLa);
  const cma = Number(d?.Cm_a ?? d?.CMa);
  const cref = Number(results?.cref ?? results?.CREF ?? 1);
  const xcg = Number(results?.xcg ?? results?.XCG);
  if (Number.isFinite(cla) && Number.isFinite(cma) && cla !== 0 && Number.isFinite(cref) && Number.isFinite(xcg)) {
    return xcg - (cma / cla) * cref;
  }
  const xnp = Number(d?.Xnp);
  return Number.isFinite(xnp) ? xnp : null;
}

/**
 * Update the Total Forces output panel from a results payload (AVL 4-column layout).
 *
 * @param {Record<string, unknown>} r
 */
function updateTotalForces(r) {
  window._lastResults = r;
  const cm = Array.isArray(r.CM) ? r.CM : [r.Cl, r.Cm, r.Cn];
  const cl = r.Cl ?? cm[0];
  const cmVal = r.Cm ?? cm[1];
  const cn = r.Cn ?? cm[2];
  const controls = r.controls ?? r.delcon ?? r.control_deflections ?? {};
  const xnp = neutralPointValue(window._lastStability, r);
  viewer.updateNeutralPoint(xnp);
  const linearAccel = Array.isArray(r.linear_acceleration_body) ? r.linear_acceleration_body : [];
  const rotationalAccel = Array.isArray(r.rotational_acceleration_body) ? r.rotational_acceleration_body : [];

  const cells = [
    forceCell("α", r.alpha_deg, 3, "force-cell r1 c1"),
    forceCell("pb/2v", r.pb2V ?? r["pb/2V"], 3, "force-cell r1 c2"),
    forceCell("CL", r.CL, 4, "force-cell r1 c3"),
    forceCell("cl", cl, 4, "force-cell r1 c4"),
    forceCell("β", r.beta_deg, 3, "force-cell r2 c1"),
    forceCell("qc/2v", r.qc2V ?? r["qc/2V"], 3, "force-cell r2 c2"),
    forceCell("CY", r.CY, 4, "force-cell r2 c3"),
    forceCell("cm", cmVal, 4, "force-cell r2 c4"),
    forceCell("M", r.mach, 3, "force-cell r3 c1"),
    forceCell("rb/2v", r.rb2V ?? r["rb/2V"], 3, "force-cell r3 c2"),
    forceCell("CD", r.CD, 4, "force-cell r3 c3"),
    forceCell("cn", cn, 4, "force-cell r3 c4"),
    forceCell("ax", linearAccel[0], 3, "force-cell r4 c1"),
    forceCell("pdot", rotationalAccel[0], 3, "force-cell r4 c2"),
    forceCell("CDi", r.CDi ?? (Number(r.CD) - Number(r.CDV)), 4, "force-cell r4 c3"),
    forceCell("e", r.e ?? r.SPANEF, 4, "force-cell r4 c4"),
    forceCell("ay", linearAccel[1], 3, "force-cell r5 c1"),
    forceCell("qdot", rotationalAccel[1], 3, "force-cell r5 c2"),
    forceCell("CDp", r.CDp ?? r.CDV, 4, "force-cell r5 c3"),
    forceCell("az", linearAccel[2], 3, "force-cell r6 c1"),
    forceCell("rdot", rotationalAccel[2], 3, "force-cell r6 c2"),
    forceCell("Xnp", xnp, 3, "force-cell r6 c3"),
  ];

  let row = 7;
  if (controls && typeof controls === "object") {
    for (const [name, deg] of Object.entries(controls)) {
      cells.push(
        `<div class="force-cell" style="grid-row:${row};grid-column:1 / span 2;">${escapeHtml(name)} = <strong>${fmt(deg, 2)}</strong> deg</div>`,
      );
      row += 1;
    }
  }

  els.totalForcesGrid.innerHTML = cells.join("");
}

/**
 * Update the stability derivatives table.
 *
 * @param {Record<string, number>} d
 * @param {Record<string, unknown>} [results]
 */
function updateStabilityDerivs(d, results = {}) {
  els.stabilityTable.innerHTML = "";

  for (const col of STABILITY_COLS) {
    const tr = document.createElement("tr");
    const label = document.createElement("td");
    label.textContent = col;
    tr.appendChild(label);

    for (const row of STABILITY_ROWS) {
      const td = document.createElement("td");
      const key = `${row}_${col}`;
      const altKey = `${row}${col === "a" ? "_a" : col === "b" ? "_b" : `_${col}`}`;
      const raw = d[key] ?? d[altKey] ?? 0;
      td.textContent = fmtSignedSigFig(displayDerivValue(col, raw));
      tr.appendChild(td);
    }
    els.stabilityTable.appendChild(tr);
  }

  const xnp = neutralPointValue(d, results);
  viewer.updateNeutralPoint(xnp);
  if (window._lastResults) updateTotalForces(window._lastResults);
}

/**
 * Render a derivative matrix grid with 4 significant figures.
 *
 * @param {HTMLElement|null} grid
 * @param {string[]} rows
 * @param {number[][]} values
 * @param {string[]} cols
 * @param {(rowLabel: string, raw: number) => number} displayValue
 * @param {string} emptyMessage
 */
function renderDerivativeGrid(grid, rows, values, cols, displayValue, emptyMessage) {
  if (!grid) return;
  if (!rows.length || !values.length) {
    grid.innerHTML = `<div class="placeholder">${emptyMessage}</div>`;
    return;
  }

  const cells = [
    matrixCell("", "matrix-head"),
    ...cols.map((col) => matrixCell(escapeHtml(col), "matrix-head matrix-colhead")),
  ];
  rows.forEach((rowLabel, i) => {
    const label = String(rowLabel ?? "");
    const rowVals = values[i] ?? [];
    cells.push(matrixCell(escapeHtml(label), "matrix-head"));
    cols.forEach((col, j) => {
      const key = `${col}${label}`;
      const shown = displayValue(label, rowVals[j]);
      cells.push(
        matrixCell(`<strong class="matrix-num">${fmtSignedSigFig(shown)}</strong>`, "matrix-val", key),
      );
    });
  });
  grid.innerHTML = cells.join("");
}

/**
 * Cache and render control-surface derivatives for the selected axis.
 *
 * @param {{ stability?: object, body?: object }|null|undefined} payload
 */
function updateControlSurface(payload) {
  if (payload) window._lastControlSurface = payload;
  const axisPayload = window._lastControlSurface?.[controlDerivAxis] ?? {};
  const defaultCols =
    controlDerivAxis === "stability"
      ? ["CL", "CD", "CY", "Cl", "Cm", "Cn"]
      : ["CX", "CY", "CZ", "Cl", "Cm", "Cn"];
  renderDerivativeGrid(
    els.controlSurfaceGrid,
    axisPayload.rows ?? [],
    axisPayload.values ?? [],
    axisPayload.cols ?? defaultCols,
    (_rowLabel, raw) => displayControlDerivValue(raw),
    "No control-surface data",
  );
}

/**
 * Update the body-axis state-derivative panel from server payload.
 *
 * @param {{ rows?: string[], cols?: string[], values?: number[][] }} payload
 */
function updateBodyAxis(payload) {
  window._lastBodyAxis = payload;
  const rows = payload?.rows ?? [];
  const values = payload?.values ?? [];
  const cols = payload?.cols ?? ["CX", "CY", "CZ", "Cl", "Cm", "Cn"];

  renderDerivativeGrid(
    els.bodyAxisGrid,
    rows.slice(0, BODY_AXIS_STATE_ROW_COUNT),
    values.slice(0, BODY_AXIS_STATE_ROW_COUNT),
    cols,
    (rowLabel, raw) => displayBodyAxisValue(rowLabel, raw),
    "No body-axis data",
  );
}

/**
 * Update per-surface force table.
 *
 * @param {Array<Record<string, unknown>>} surfaces
 */
function updateSurfaceForces(surfaces) {
  if (!surfaces?.length) {
    els.surfaceForcesTable.innerHTML = '<tr><td colspan="7" class="placeholder">—</td></tr>';
    return;
  }
  els.surfaceForcesTable.innerHTML = surfaces
    .map(
      (s) => `<tr>
        <td>${escapeHtml(s.name ?? "—")}</td>
        <td>${fmt(s.CL)}</td>
        <td>${fmt(s.CD)}</td>
        <td>${fmt(s.CY)}</td>
        <td>${fmt(s.Cl)}</td>
        <td>${fmt(s.Cm)}</td>
        <td>${fmt(s.Cn)}</td>
      </tr>`,
    )
    .join("");
}

/**
 * Format a derived eigenmetric for table display.
 *
 * @param {number|null|undefined} value
 * @param {number} [digits]
 * @returns {string}
 */
function fmtEigenMetric(value, digits = 4) {
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? n.toFixed(digits) : "—";
}

/**
 * Update the eigenanalysis table from eigenvalue and mode payloads.
 *
 * @param {Array<{ re: number, im: number }>} eigenvalues
 * @param {Array<Record<string, unknown>>} [modes]
 */
function updateEigenanalysis(eigenvalues, modes = []) {
  const tbody = els.eigenanalysisTable;
  if (!tbody) return;
  if (!eigenvalues?.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="placeholder">—</td></tr>';
    return;
  }

  tbody.innerHTML = eigenvalues
    .map((ev, i) => {
      const mode = modes[i] ?? {};
      const evData = mode.eigenvalue ?? ev;
      const label = mode.name ?? `Mode ${i + 1}`;
      return `<tr>
        <td>${escapeHtml(label)}</td>
        <td>${fmtSigned(evData.re, 4)}</td>
        <td>${fmtSigned(evData.im, 4)}</td>
        <td>${fmtEigenMetric(mode.frequency_hz, 3)}</td>
        <td>${fmtEigenMetric(mode.time_to_half_s, 2)}</td>
        <td>${fmtEigenMetric(mode.period_s, 2)}</td>
      </tr>`;
    })
    .join("");
}

/**
 * Normalize hinge-moment payloads from the server or legacy array form.
 *
 * @param {Array<Record<string, unknown>> | Record<string, unknown>} payload
 * @returns {{ dimensional: boolean, momentUnits: string | null, controls: Array<Record<string, unknown>> }}
 */
function normalizeHingeMoments(payload) {
  if (Array.isArray(payload)) {
    const momentUnits = payload.find((row) => row?.moment_units)?.moment_units ?? null;
    const dimensional = momentUnits != null && momentUnits !== "coeff·Sref·Cref";
    return { dimensional, momentUnits: dimensional ? momentUnits : null, controls: payload };
  }
  const controls = Array.isArray(payload?.controls)
    ? payload.controls
    : Array.isArray(payload?.hinges)
      ? payload.hinges
      : Array.isArray(payload?.data)
        ? payload.data
        : [];
  const dimensional = Boolean(payload?.dimensional);
  const momentUnits = dimensional ? (payload?.moment_units ?? "force*length") : null;
  return { dimensional, momentUnits, controls };
}

/**
 * Update hinge moment table (optional payload).
 *
 * @param {Array<Record<string, unknown>> | Record<string, unknown>} payload
 */
function updateHingeMoments(payload) {
  const grid = els.hingeMomentsGrid;
  if (!grid) return;
  const { dimensional, momentUnits, controls } = normalizeHingeMoments(payload ?? {});
  if (!controls.length) {
    grid.style.gridTemplateColumns = "";
    grid.innerHTML = '<div class="placeholder">—</div>';
    return;
  }

  grid.style.gridTemplateColumns = dimensional
    ? "max-content max-content max-content"
    : "max-content max-content";

  const cells = [
    matrixCell("", "matrix-head"),
    matrixCell("Chinge", "matrix-head matrix-colhead"),
  ];
  if (dimensional) {
    const unitLabel = momentUnits ? `Moment (${momentUnits})` : "Moment";
    cells.push(matrixCell(unitLabel, "matrix-head matrix-colhead"));
  }

  controls.forEach((h) => {
    const chinge = h.Chinge ?? h.chinge ?? h.value;
    cells.push(matrixCell(escapeHtml(h.name ?? h.control ?? "—"), "matrix-head"));
    cells.push(matrixCell(`<strong class="matrix-num">${fmtSigned(chinge, 6)}</strong>`, "matrix-val"));
    if (dimensional) {
      cells.push(matrixCell(`<strong class="matrix-num">${fmtSigned(h.moment, 6)}</strong>`, "matrix-val"));
    }
  });
  grid.innerHTML = cells.join("");
}

/** Return a stable key for detecting when the loaded aircraft has changed. */
function modelKey(msg) {
  if (typeof msg.avl_text === "string") return msg.avl_text;
  const meta = msg.meta ?? {};
  return JSON.stringify([meta.title, meta.surfaces, meta.strips, meta.vortices]);
}

/** Return whether a correlated solve message belongs to the visible case. */
function isVisibleSolveMessage(msg) {
  if (msg.replay) return true;
  const activeCase = selectedRunCaseIndex >= 0 ? runCases[selectedRunCaseIndex] : null;
  if (msg.case_id == null) return activeCase == null;
  return activeCase?.id === msg.case_id;
}

/** Save a completed response batch as the case's last successful execution. */
function completeExecution(msg) {
  const pending = pendingSolves.get(msg.solve_id);
  if (!pending) return;
  pendingSolves.delete(msg.solve_id);

  const state = pending.state ?? captureEditableState();
  const executed = {
    inputs: structuredClone(state.inputs),
    constraints: structuredClone(state.constraints),
    messages: structuredClone(pending.messages),
  };
  const entry = pending.caseId
    ? runCases.find((candidate) => candidate.id === pending.caseId)
    : null;
  if (entry) entry.executed = executed;

  const activeCase = selectedRunCaseIndex >= 0 ? runCases[selectedRunCaseIndex] : null;
  if ((!pending.caseId && !activeCase) || activeCase?.id === pending.caseId) {
    latestExecutedSnapshot = structuredClone(executed);
    setSolving(false);
    setStatus("connected", "Connected");
  }
}

/**
 * Dispatch an incoming WebSocket message by type.
 *
 * @param {Record<string, unknown>} msg
 */
function handleMessage(msg) {
  if (msg.type === "solve_complete") {
    completeExecution(msg);
    return;
  }

  if (msg.type === "error" && msg.solve_id) {
    pendingSolves.delete(msg.solve_id);
    if (!isVisibleSolveMessage(msg)) return;
  }

  if (msg.type === "model_loaded" && msg.solve_id && msg.geometry) {
    const pending = pendingSolves.get(msg.solve_id);
    if (pending) pending.messages.push({ type: "cp_update", geometry: msg.geometry });
  }

  if (msg.solve_id && SOLVE_RESULT_TYPES.has(msg.type)) {
    const pending = pendingSolves.get(msg.solve_id);
    if (pending) {
      const stored = { ...msg };
      delete stored.case_id;
      delete stored.solve_id;
      delete stored.replay;
      pending.messages.push(stored);
    }
    if (!isVisibleSolveMessage(msg)) return;
  }

  switch (msg.type) {
    case "model_loaded": {
      const nextModelKey = modelKey(msg);
      if (activeModelKey !== null && activeModelKey !== nextModelKey) {
        runCases.splice(0, runCases.length);
        selectedRunCaseIndex = -1;
        latestExecutedSnapshot = null;
        renderRunCasesList();
      }
      activeModelKey = nextModelKey;
      if (msg.geometry) viewer.loadGeometry(msg.geometry);
      if (msg.meta) {
        updateMeta(msg.meta);
        if (modelLoadIntent === "run") {
          requestExecution();
        } else if (modelLoadIntent === "reset") {
          resetConstraintsFromModel(msg.meta);
          requestExecution();
        } else if (msg.meta.example) {
          resetConstraintsFromModel(msg.meta);
        } else {
          renderAllConstraints();
        }
        if (msg.meta.mass_props) {
          updateMassProperties(msg.meta.mass_props);
          const active = msg.meta.mass_props.active;
          if (active) {
            viewer.updateCg({ x: active.xcg, y: active.ycg, z: active.zcg });
          }
        }
        if (msg.meta.afil_dependencies) renderAfilDependencies(msg.meta.afil_dependencies);
      }
      modelLoadIntent = null;
      if (msg.avl_text) els.avlEditor.value = msg.avl_text;
      if (msg.mass_text) els.massEditor.value = msg.mass_text;
      if (Array.isArray(msg.meta?.warnings) && msg.meta.warnings.length) {
        showError(msg.meta.warnings.join(" "));
      } else {
        showError(null);
      }
      break;
    }

    case "afil_dependencies":
      renderAfilDependencies(msg.dependencies);
      break;

    case "cp_update":
      if (msg.geometry) viewer.loadGeometry(msg.geometry);
      break;

    case "solve_started":
      setSolving(true);
      setStatus("solving", "Solving…");
      break;

    case "results":
      setSolving(false);
      setStatus("connected", "Connected");
      if (msg.sref != null || msg.cref != null || msg.bref != null) {
        updateMeta({ sref: msg.sref, cref: msg.cref, bref: msg.bref });
      }
      updateTotalForces(msg);
      if (msg.dcp || msg.cp_data) {
        viewer.updateCpOverlay(msg.cp_data ?? msg.dcp);
      }
      if (msg.body_axis) updateBodyAxis(msg.body_axis);
      if (msg.control_surface) updateControlSurface(msg.control_surface);
      if (msg.hinge_moments) updateHingeMoments(msg.hinge_moments);
      if (msg.mass_props) {
        updateMassProperties(msg.mass_props);
        const active = msg.mass_props.active;
        if (active) {
          viewer.updateCg({ x: active.xcg, y: active.ycg, z: active.zcg });
        }
      }
      if (msg.cref || msg.xcg) {
        updateStabilityDerivs(window._lastStability ?? {}, msg);
      }
      break;

    case "stability_derivs":
      window._lastStability = msg;
      updateStabilityDerivs(msg, msg);
      if (msg.body_axis) updateBodyAxis(msg.body_axis);
      if (msg.control_surface) updateControlSurface(msg.control_surface);
      break;

    case "trefftz_data":
      updateTrefftzPlot("trefftz-plot", msg);
      if (msg.lift_3d || msg.cref != null) {
        viewer.updateLiftDistribution({
          surfaces: msg.lift_3d?.surfaces ?? [],
          cref: msg.cref,
          bref: msg.bref,
        });
      }
      viewer.updateWake(msg.wake_3d ?? { surfaces: [] });
      if (msg.cg) viewer.updateCg(msg.cg);
      break;

    case "eigen_data": {
      const modes = msg.modes ?? [];
      const eigenvalues = msg.eigenvalues ?? [];
      updateEigenanalysis(eigenvalues, modes);
      updateEigenmodesPlot("eigen-plot", {
        eigenvalues: eigenvalues.map((ev, i) => ({
          re: ev.re,
          im: ev.im,
          name: modes[i]?.name,
        })),
      });
      break;
    }

    case "surface_forces":
      updateSurfaceForces(Array.isArray(msg) ? msg : msg.surfaces ?? msg.data);
      break;

    case "hinge_moments":
      updateHingeMoments(msg);
      break;

    case "error":
      setSolving(false);
      showError(msg.message ?? "Unknown error");
      setStatus("connected", "Connected");
      break;

    default:
      console.debug("Unhandled WS message:", msg.type);
  }
}

/**
 * Open (or reopen) the WebSocket connection to the server.
 */
function connect() {
  if (ws?.readyState === WebSocket.OPEN || ws?.readyState === WebSocket.CONNECTING) return;

  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const url = `${proto}//${location.host}/ws`;
  setStatus("connecting", "Connecting…");

  ws = new WebSocket(url);

  ws.addEventListener("open", () => {
    setStatus("connected", "Connected");
    showError(null);
    if (!hasAutoLoadedExample) {
      hasAutoLoadedExample = true;
      requestExecution({ type: "load_example", name: "supra" }, null);
    }
  });

  ws.addEventListener("message", (ev) => {
    try {
      handleMessage(JSON.parse(ev.data));
    } catch (err) {
      console.error("Bad WS payload", err);
    }
  });

  ws.addEventListener("close", () => {
    setStatus("disconnected", "Disconnected");
    clearTimeout(reconnectTimer);
    reconnectTimer = setTimeout(connect, 2000);
  });

  ws.addEventListener("error", () => {
    setStatus("error", "Connection error");
  });
}

/**
 * Read the first selected file from a file input as UTF-8 text.
 *
 * @param {HTMLInputElement} input
 * @returns {Promise<{name: string, text: string}|null>}
 */
async function readSelectedFile(input) {
  const file = input.files?.[0];
  if (!file) return null;
  const text = await file.text();
  return { name: file.name, text };
}

/**
 * Wire a hidden file input to one or more buttons and invoke a callback when a file is chosen.
 *
 * @param {string} inputId
 * @param {string|string[]} buttonIds
 * @param {(file: {name: string, text: string}) => void|Promise<void>} onLoaded
 */
function bindFileLoad(inputId, buttonIds, onLoaded) {
  const input = document.getElementById(inputId);
  const buttons = (Array.isArray(buttonIds) ? buttonIds : [buttonIds])
    .map((id) => document.getElementById(id))
    .filter(Boolean);
  if (!input || !buttons.length) return;

  buttons.forEach((button) => {
    button.addEventListener("click", () => input.click());
  });
  input.addEventListener("change", async () => {
    try {
      const result = await readSelectedFile(input);
      if (!result) return;
      await onLoaded(result);
    } catch (err) {
      showError(err instanceof Error ? err.message : "Failed to read file");
    } finally {
      input.value = "";
    }
  });
}

/** Bind UI button and editor handlers. */
function bindUI() {
  document.getElementById("btn-load-supra-demo").addEventListener("click", () => {
    if (isSolving) return;
    requestExecution({ type: "load_example", name: "supra" }, null);
  });

  bindFileLoad("avl-file-input", ["btn-load-avl-file-aircraft", "btn-load-avl-file"], async ({ text }) => {
    els.avlEditor.value = text;
    if (!text.trim()) {
      showError("Selected AVL file is empty");
      return;
    }
    showError(null);
    modelLoadIntent = "reset";
    send({ type: "upload_avl", text });
  });

  bindFileLoad("mass-file-input", "btn-load-mass-file", async ({ text }) => {
    els.massEditor.value = text;
    showError(null);
    send({ type: "upload_mass", text });
    scheduleSolve();
  });

  document.getElementById("btn-upload-avl").addEventListener("click", () => {
    const text = els.avlEditor.value;
    if (!text.trim()) {
      showError("AVL editor is empty");
      return;
    }
    showError(null);
    modelLoadIntent = "run";
    send({ type: "upload_avl", text });
  });

  document.getElementById("btn-upload-mass").addEventListener("click", () => {
    const text = els.massEditor.value;
    send({ type: "upload_mass", text });
    scheduleSolve();
  });

  document.getElementById("btn-add-constraint").addEventListener("click", () => {
    constraints.push({ variable: "beta", constraint: "beta", value: 0 });
    renderAllConstraints();
    scheduleSolve();
  });

  document.getElementById("btn-add-run-case").addEventListener("click", () => {
    runCases.push(captureRunCase(`Case ${runCases.length + 1}`));
    selectedRunCaseIndex = runCases.length - 1;
    renderRunCasesList();
  });

  els.avlEditor.addEventListener("change", () => {
    send({ type: "upload_avl", text: els.avlEditor.value });
  });

  const derivUnits = document.getElementById("deriv-units");
  if (derivUnits) {
    derivUnits.addEventListener("change", () => {
      derivDisplayUnit = derivUnits.value === "per_deg" ? "per_deg" : "per_rad";
      if (window._lastStability) {
        updateStabilityDerivs(window._lastStability, window._lastStability);
      }
      if (window._lastBodyAxis) {
        updateBodyAxis(window._lastBodyAxis);
      }
      updateControlSurface();
    });
  }

  const controlAxis = document.getElementById("control-deriv-axis");
  if (controlAxis) {
    const onControlAxisChange = () => {
      controlDerivAxis = controlAxis.value === "body" ? "body" : "stability";
      updateControlSurface();
    };
    controlAxis.addEventListener("change", onControlAxisChange);
    controlAxis.addEventListener("input", onControlAxisChange);
  }
}

/** Application entry point. */
async function main() {
  initCollapsiblePanels();
  renderAllConstraints();
  renderRunCasesList();
  bindFlightInputs();
  bindMassInputs();
  bindUI();
  connect();
  initPlots("trefftz-plot", "eigen-plot").catch((err) => {
    console.warn("Plotly init failed:", err);
  });
}

main();
