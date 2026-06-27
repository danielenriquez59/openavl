/**
 * Plotly.js Trefftz-plane and eigenmode charts for the OpenAVL web GUI.
 */

const PLOT_LAYOUT_BASE = {
  paper_bgcolor: "#111821",
  plot_bgcolor: "#0a0d12",
  font: { family: "Segoe UI, system-ui, sans-serif", color: "#9aa7b5", size: 10 },
  margin: { l: 48, r: 48, t: 24, b: 40 },
  xaxis: {
    gridcolor: "#1f242c",
    zerolinecolor: "#2a313c",
    linecolor: "#2a313c",
  },
  yaxis: {
    gridcolor: "#1f242c",
    zerolinecolor: "#2a313c",
    linecolor: "#2a313c",
  },
  legend: {
    orientation: "h",
    y: 1.12,
    x: 0,
    bgcolor: "rgba(0,0,0,0)",
    font: { size: 9 },
  },
};

const PLOT_CONFIG = {
  responsive: true,
  displayModeBar: false,
};

const EIGEN_PLOT_CONFIG = {
  responsive: true,
  displayModeBar: true,
  modeBarButtonsToRemove: ["lasso2d", "select2d", "toImage", "sendDataToCloud"],
  scrollZoom: true,
  doubleClick: "reset",
};

/** @type {WeakMap<HTMLElement, boolean>} */
const eigenPlotRangeFitted = new WeakMap();

/**
 * Resolve a Plotly container id or element.
 *
 * @param {string|HTMLElement} container
 * @returns {HTMLElement|null}
 */
function getPlotDiv(container) {
  if (typeof container === "string") {
    return document.getElementById(container);
  }
  return container ?? null;
}

const TREFFTZ_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"];

/**
 * Build one Trefftz trace for a surface strip series.
 *
 * @param {string} name
 * @param {number[]} x
 * @param {number[]} y
 * @param {Record<string, unknown>} style
 */
function trefftzTrace(name, x, y, style) {
  return {
    x,
    y,
    type: "scatter",
    mode: "lines",
    name,
    connectgaps: false,
    ...style,
  };
}

/**
 * Concatenate surface strip arrays into a single Plotly trace with null gaps.
 *
 * @param {Array<Record<string, unknown>>} surfaces
 * @param {string} valueKey
 * @param {(value: number) => number} [transform]
 * @returns {{ x: Array<number|null>, y: Array<number|null> }}
 */
function aggregateSurfaceSeries(surfaces, valueKey, transform = (value) => value) {
  const x = [];
  const y = [];

  for (const surf of surfaces) {
    const span = Array.isArray(surf.y) ? surf.y : [];
    const values = Array.isArray(surf[valueKey]) ? surf[valueKey] : [];
    const pairs = [];

    for (let i = 0; i < Math.min(span.length, values.length); i += 1) {
      const spanValue = Number(span[i]);
      const rawValue = Number(values[i]);
      if (!Number.isFinite(spanValue) || !Number.isFinite(rawValue)) continue;
      pairs.push([spanValue, transform(rawValue)]);
    }

    pairs.sort((a, b) => a[0] - b[0]);
    for (const [spanValue, value] of pairs) {
      x.push(spanValue);
      y.push(value);
    }
    if (pairs.length) {
      x.push(null);
      y.push(null);
    }
  }

  return { x, y };
}

/**
 * Return true when an aggregated series contains at least one finite value.
 *
 * @param {{ x: Array<number|null>, y: Array<number|null> }} series
 */
function hasFiniteSeries(series) {
  return series.y.some((value) => Number.isFinite(Number(value)));
}

/**
 * Render or update the Trefftz-plane span loading chart.
 *
 * @param {string|HTMLElement} container
 * @param {{
 *   surfaces?: Array<{ name?: string, y?: number[], cl?: number[], clnorm?: number[], cnc?: number[], ai?: number[] }>,
 *   y?: number[],
 *   cl?: number[],
 *   clnorm?: number[],
 *   cnc?: number[],
 *   ai?: number[]
 * }} data
 */
export function updateTrefftzPlot(container, data) {
  if (!window.Plotly) return;

  const surfaces = data?.surfaces?.length
    ? data.surfaces
    : [{ name: "loading", y: data?.y ?? [], cl: data?.cl, clnorm: data?.clnorm, cnc: data?.cnc, ai: data?.ai }];

  const cl = aggregateSurfaceSeries(surfaces, "cl");
  const clnorm = aggregateSurfaceSeries(surfaces, "clnorm");
  const cnc = aggregateSurfaceSeries(surfaces, "cnc");
  const ai = aggregateSurfaceSeries(surfaces, "ai", (value) => -value);

  const traces = [];
  if (hasFiniteSeries(clnorm)) {
    traces.push(
      trefftzTrace("cl⊥", clnorm.x, clnorm.y, {
        line: { color: "#ef4444", width: 1.5, dash: "dash" },
        yaxis: "y",
      }),
    );
  }
  if (hasFiniteSeries(cl)) {
    traces.push(
      trefftzTrace("cl", cl.x, cl.y, {
        line: { color: "#fb923c", width: 1.5, dash: "dashdot" },
        yaxis: "y",
      }),
    );
  }
  if (hasFiniteSeries(cnc)) {
    traces.push(
      trefftzTrace("cl c / cref", cnc.x, cnc.y, {
        line: { color: "#22c55e", width: 1.7 },
        yaxis: "y",
      }),
    );
  }
  if (hasFiniteSeries(ai)) {
    traces.push(
      trefftzTrace("αi", ai.x, ai.y, {
        line: { color: "#3b82f6", width: 1.4, dash: "dot" },
        yaxis: "y2",
      }),
    );
  }

  if (!traces.length) {
    traces.push({ x: [], y: [], type: "scatter", mode: "lines", name: "—" });
  }

  const layout = {
    ...PLOT_LAYOUT_BASE,
    xaxis: { ...PLOT_LAYOUT_BASE.xaxis, title: "span y" },
    yaxis: { ...PLOT_LAYOUT_BASE.yaxis, title: "cl / cl c/cref", rangemode: "tozero" },
    yaxis2: {
      title: "αi",
      overlaying: "y",
      side: "right",
      gridcolor: "rgba(0,0,0,0)",
      zerolinecolor: "#2a313c",
      linecolor: "#2a313c",
      color: "#3b82f6",
    },
  };

  window.Plotly.react(container, traces, layout, PLOT_CONFIG);
}

/**
 * Render or update the eigenvalue scatter plot in the complex plane.
 *
 * @param {string|HTMLElement} container
 * @param {{ eigenvalues: Array<{ re: number, im: number, label?: string, name?: string }>, labels?: string[] }} data
 */
export function updateEigenmodesPlot(container, data) {
  if (!window.Plotly) return;

  const plotDiv = getPlotDiv(container);
  if (!plotDiv) return;

  const modes = data?.eigenvalues ?? [];
  const re = modes.map((m) => m.re);
  const im = modes.map((m) => m.im);
  const text = modes.map((m, i) => m.label ?? m.name ?? data?.labels?.[i] ?? `Mode ${i + 1}`);

  const trace = {
    x: re,
    y: im,
    text,
    type: "scatter",
    mode: "markers+text",
    textposition: "top center",
    textfont: { size: 8, color: "#e8edf2" },
    marker: {
      size: 8,
      color: re.map((r) => (r < 0 ? "#6fcf97" : "#eb5757")),
      line: { color: "#e8edf2", width: 0.5 },
    },
    hovertemplate: "%{text}<br>Re=%{x:.4f}<br>Im=%{y:.4f}<extra></extra>",
  };

  const xRange = re.length
    ? [Math.min(...re, 0) * 1.2 - 0.1, Math.max(...re, 0) * 1.2 + 0.1]
    : [-1, 1];
  const yMax = im.length ? Math.max(...im.map(Math.abs), 0.5) * 1.3 : 1;

  const hasLayout = Boolean(plotDiv._fullLayout);
  const shouldFitRange = modes.length > 0 && !eigenPlotRangeFitted.get(plotDiv);

  const xaxis = {
    ...PLOT_LAYOUT_BASE.xaxis,
    title: "Real",
    zeroline: true,
    zerolinecolor: "#7fb3ff",
    zerolinewidth: 1,
  };
  const yaxis = {
    ...PLOT_LAYOUT_BASE.yaxis,
    title: "Imag",
    zeroline: true,
    zerolinecolor: "#7fb3ff",
  };

  if (shouldFitRange) {
    xaxis.range = xRange;
    yaxis.range = [-yMax, yMax];
    eigenPlotRangeFitted.set(plotDiv, true);
  } else if (!hasLayout) {
    xaxis.range = [-1, 1];
    yaxis.range = [-1, 1];
  }

  const layout = {
    ...PLOT_LAYOUT_BASE,
    uirevision: "eigenmodes",
    dragmode: "zoom",
    xaxis,
    yaxis,
    shapes: [
      {
        type: "rect",
        xref: "paper",
        yref: "paper",
        x0: 0,
        x1: 0.5,
        y0: 0,
        y1: 1,
        fillcolor: "rgba(111, 207, 151, 0.04)",
        line: { width: 0 },
        layer: "below",
      },
      {
        type: "rect",
        xref: "paper",
        yref: "paper",
        x0: 0.5,
        x1: 1,
        y0: 0,
        y1: 1,
        fillcolor: "rgba(235, 87, 87, 0.04)",
        line: { width: 0 },
        layer: "below",
      },
    ],
    showlegend: false,
  };

  window.Plotly.react(plotDiv, [trace], layout, EIGEN_PLOT_CONFIG);
}

/**
 * Inject Plotly.js from CDN once and return a ready promise.
 *
 * @returns {Promise<void>}
 */
export function loadPlotly() {
  if (window.Plotly) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "https://cdn.plot.ly/plotly-2.27.0.min.js";
    script.onload = () => resolve();
    script.onerror = () => reject(new Error("Failed to load Plotly.js"));
    document.head.appendChild(script);
  });
}

/**
 * Initialize empty placeholder charts.
 *
 * @param {string} trefftzId
 * @param {string} eigenId
 */
export async function initPlots(trefftzId, eigenId) {
  await loadPlotly();
  updateTrefftzPlot(trefftzId, { surfaces: [] });
  updateEigenmodesPlot(eigenId, { eigenvalues: [] });
}
