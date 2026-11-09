/* Interactive charts for the mechanism-design results page.
 *
 * Parts 1 and 2 replay the committed simulation output (docs/data.json) so the
 * page plots exactly the numbers the figures and tests were built from.
 * Part 3 is recomputed live in the browser -- Dijkstra plus the VCG payment
 * rule -- because the whole point there is to let you move a report and watch
 * the mechanism respond.
 */
"use strict";

const CSS = getComputedStyle(document.documentElement);
const C = (name) => CSS.getPropertyValue(name).trim();

const STRATEGY_LABELS = {
  truthful: "bids truthfully",
  half: "bids half its value",
  random: "bids at random",
};

/* ---------------------------------------------------------------- charting */

const SVG_NS = "http://www.w3.org/2000/svg";

function el(name, attrs = {}, text) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}

/** Round a range outwards to "nice" tick values. */
function ticks(lo, hi, count) {
  const span = hi - lo || 1;
  const raw = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag;
  const out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) {
    out.push(Math.abs(t) < 1e-9 ? 0 : t);
  }
  return out;
}

const fmt = (v) => (Math.abs(v) < 1e-9 ? "0" : String(Number(v.toFixed(4))));

/**
 * Draw a line/scatter chart.
 *
 * @param {HTMLElement} host      container, cleared on each draw
 * @param {object} spec
 *   xs      {number[]} shared x values
 *   dots    {number[]} scatter series (may be null)
 *   line    {number[]} line series (may be null)
 *   vlines  [{x, colour, label}] reference lines
 *   point   {x, y, colour} single emphasised marker
 *   xLabel, yLabel {string}
 */
function drawChart(host, spec) {
  const W = 760;
  const H = 330;
  const M = { l: 64, r: 20, t: 18, b: 52 };
  const iw = W - M.l - M.r;
  const ih = H - M.t - M.b;

  const series = [spec.dots, spec.line].filter(Boolean);
  const all = series.flat();
  let lo = Math.min(...all);
  let hi = Math.max(...all);
  if (spec.point) {
    lo = Math.min(lo, spec.point.y);
    hi = Math.max(hi, spec.point.y);
  }
  const pad = (hi - lo || 1) * 0.12;
  lo -= pad;
  hi += pad;

  const xlo = Math.min(...spec.xs);
  const xhi = Math.max(...spec.xs);
  const X = (v) => M.l + ((v - xlo) / (xhi - xlo || 1)) * iw;
  const Y = (v) => M.t + ih - ((v - lo) / (hi - lo || 1)) * ih;

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });

  for (const t of ticks(lo, hi, 5)) {
    const y = Y(t);
    svg.appendChild(el("line", { x1: M.l, x2: M.l + iw, y1: y, y2: y, class: "axis" }));
    svg.appendChild(
      el("text", { x: M.l - 10, y: y + 3.5, "text-anchor": "end", class: "tick" }, fmt(t))
    );
  }
  for (const t of ticks(xlo, xhi, 6)) {
    const x = X(t);
    svg.appendChild(
      el("text", { x, y: M.t + ih + 20, "text-anchor": "middle", class: "tick" }, fmt(t))
    );
  }

  // Zero line, where the axis range straddles it.
  if (lo < 0 && hi > 0) {
    svg.appendChild(
      el("line", {
        x1: M.l, x2: M.l + iw, y1: Y(0), y2: Y(0),
        stroke: C("--dim"), "stroke-width": 1,
      })
    );
  }

  for (const v of spec.vlines ?? []) {
    const x = X(v.x);
    svg.appendChild(
      el("line", {
        x1: x, x2: x, y1: M.t, y2: M.t + ih,
        stroke: v.colour, "stroke-width": 1.4, "stroke-dasharray": "3 3",
      })
    );
    if (v.label) {
      const flip = x > M.l + iw * 0.62;
      svg.appendChild(
        el("text", {
          x: x + (flip ? -7 : 7), y: M.t + 13,
          "text-anchor": flip ? "end" : "start",
          class: "tick", "font-weight": "700", style: `fill:${v.colour}`,
        }, v.label)
      );
    }
  }

  if (spec.dots) {
    for (let i = 0; i < spec.dots.length; i++) {
      svg.appendChild(
        el("circle", { cx: X(spec.xs[i]), cy: Y(spec.dots[i]), r: 2.6, fill: C("--blue") })
      );
    }
  }
  if (spec.line) {
    const d = spec.line.map((y, i) => `${i ? "L" : "M"}${X(spec.xs[i])},${Y(y)}`).join("");
    svg.appendChild(
      el("path", {
        d, fill: "none", stroke: C("--text"),
        "stroke-width": 2, "stroke-dasharray": "6 4",
      })
    );
  }
  if (spec.point) {
    svg.appendChild(
      el("circle", {
        cx: X(spec.point.x), cy: Y(spec.point.y), r: 6,
        fill: spec.point.colour, stroke: C("--bg"), "stroke-width": 2,
      })
    );
  }

  svg.appendChild(
    el("line", { x1: M.l, x2: M.l + iw, y1: M.t + ih, y2: M.t + ih, class: "axis" })
  );
  svg.appendChild(
    el("text", {
      x: M.l + iw / 2, y: H - 8, "text-anchor": "middle", class: "axlab",
    }, spec.xLabel)
  );
  svg.appendChild(
    el("text", {
      x: 14, y: M.t + ih / 2, "text-anchor": "middle", class: "axlab",
      transform: `rotate(-90 14 ${M.t + ih / 2})`,
    }, spec.yLabel)
  );

  host.replaceChildren(svg);
}

/* ---------------------------------------------------------------- controls */

function segmented(host, options, initial, onChange) {
  host.replaceChildren();
  let current = initial;
  const buttons = options.map(({ value, label }) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = label;
    b.setAttribute("aria-pressed", String(value === current));
    b.addEventListener("click", () => {
      current = value;
      buttons.forEach((other, i) =>
        other.setAttribute("aria-pressed", String(options[i].value === current))
      );
      onChange(value);
    });
    host.appendChild(b);
    return b;
  });
  return () => current;
}

function readout(host, entries) {
  host.replaceChildren(
    ...entries.map(([k, v]) => {
      const wrap = document.createElement("div");
      const key = document.createElement("span");
      key.className = "k";
      key.textContent = k;
      const val = document.createElement("span");
      val.className = "v";
      val.textContent = v;
      wrap.append(key, val);
      return wrap;
    })
  );
}

/* ------------------------------------------------------------- second price */

function initSecondPrice(data) {
  const state = { value: 0.5, strategy: "truthful", rivals: 1 };

  const draw = () => {
    const key = `${state.rivals}|${state.value}|${state.strategy}`;
    const s = data.series[key];
    if (!s) return;

    drawChart(document.getElementById("sp-chart"), {
      xs: data.bids,
      dots: s.sim,
      line: s.exact,
      vlines: [{ x: state.value, colour: C("--accent"), label: `b = v = ${state.value}` }],
      xLabel: "bidder 1's bid, b",
      yLabel: "expected utility",
    });

    const best = Math.max(...s.exact);
    const optimal = data.bids.filter((_, i) => s.exact[i] >= best - 1e-9);
    const atValue = s.exact[data.bids.findIndex((b) => Math.abs(b - state.value) < 1e-9)];
    const tie = optimal.length > 1
      ? `${optimal.length} bids tie (${optimal[0]}–${optimal[optimal.length - 1]})`
      : "unique";

    readout(document.getElementById("sp-readout"), [
      ["utility at b = v", fmt(atValue)],
      ["best achievable", fmt(best)],
      ["is b = v optimal?", atValue >= best - 1e-9 ? "yes" : "no"],
      ["argmax", tie],
    ]);
  };

  segmented(
    document.getElementById("sp-value"),
    [0.2, 0.5, 0.8].map((v) => ({ value: v, label: `v = ${v}` })),
    state.value,
    (v) => { state.value = v; draw(); }
  );
  segmented(
    document.getElementById("sp-strategy"),
    Object.entries(STRATEGY_LABELS).map(([value, label]) => ({ value, label })),
    state.strategy,
    (v) => { state.strategy = v; draw(); }
  );
  segmented(
    document.getElementById("sp-rivals"),
    [1, 2].map((n) => ({ value: n, label: n === 1 ? "1 rival" : "2 rivals" })),
    state.rivals,
    (v) => { state.rivals = v; draw(); }
  );
  draw();
}

/* -------------------------------------------------------------- first price */

function initFirstPrice(data) {
  const slider = document.getElementById("fp-k");
  const label = document.getElementById("fp-k-val");
  let n = 2;

  slider.max = String(data.k.length - 1);
  slider.value = String(data.k.findIndex((k) => Math.abs(k - 0.5) < 1e-9));

  const draw = () => {
    const s = data.series[String(n)];
    const i = Number(slider.value);
    const k = data.k[i];
    label.textContent = k.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");

    drawChart(document.getElementById("fp-chart"), {
      xs: data.k,
      dots: s.sim,
      line: s.exact,
      vlines: [
        { x: s.equilibrium_k, colour: C("--green"), label: "equilibrium" },
        { x: 1, colour: C("--red"), label: "truthful" },
      ],
      point: { x: k, y: s.exact[i], colour: C("--accent") },
      xLabel: "shading factor k   (bidder 1 bids k·v)",
      yLabel: "expected utility",
    });

    const loss = s.peak - s.exact[i];
    readout(document.getElementById("fp-readout"), [
      ["expected utility", fmt(s.exact[i])],
      ["win rate", `${(s.win[i] * 100).toFixed(1)}%`],
      ["winner overpays by", `${(s.overpay_pct[i] * 100).toFixed(1)}% of bid`],
      ["best possible", `${fmt(s.peak)} at k = ${s.equilibrium_k}`],
      ["you are giving up", loss < 1e-9 ? "nothing — this is the optimum" : fmt(loss)],
    ]);
  };

  segmented(
    document.getElementById("fp-n"),
    [2, 3, 4].map((v) => ({ value: v, label: `n = ${v}` })),
    n,
    (v) => {
      n = v;
      // Jump the slider to this n's equilibrium so the optimum is the default view.
      const eq = data.series[String(v)].equilibrium_k;
      slider.value = String(data.k.findIndex((kk) => Math.abs(kk - eq) < 1e-9));
      draw();
    }
  );
  slider.addEventListener("input", draw);
  draw();
}

/* ---------------------------------------------------------------------- VCG */

const key = (u, v) => (u <= v ? `${u}|${v}` : `${v}|${u}`);

/**
 * Cheapest source-to-sink path, optionally with one edge removed.
 * Ties break on the lexicographically smallest node sequence, matching the
 * Python implementation -- the network has real ties (ABDF and ACF both cost 7).
 */
function shortestPath(costs, exclude) {
  const adjacency = new Map();
  for (const [k, cost] of costs) {
    if (exclude && k === exclude) continue;
    const [u, v] = k.split("|");
    if (!adjacency.has(u)) adjacency.set(u, []);
    if (!adjacency.has(v)) adjacency.set(v, []);
    adjacency.get(u).push([v, cost]);
    adjacency.get(v).push([u, cost]);
  }

  const queue = [[0, ["A"]]];
  const settled = new Set();
  while (queue.length) {
    queue.sort((a, b) => a[0] - b[0] || a[1].join("").localeCompare(b[1].join("")));
    const [cost, nodes] = queue.shift();
    const node = nodes[nodes.length - 1];
    if (settled.has(node)) continue;
    if (node === "F") {
      const edges = nodes.slice(1).map((n, i) => key(nodes[i], n));
      return { cost, nodes, edges, label: nodes.join("") };
    }
    settled.add(node);
    for (const [next, step] of adjacency.get(node) ?? []) {
      if (!settled.has(next)) queue.push([cost + step, [...nodes, next]]);
    }
  }
  return { cost: Infinity, nodes: [], edges: [], label: "no path" };
}

function runVCG(costs) {
  const chosen = shortestPath(costs, null);
  const onPath = new Set(chosen.edges);
  const payments = new Map();
  for (const [k, cost] of costs) {
    if (!onPath.has(k)) {
      payments.set(k, 0);
      continue;
    }
    // Payment uses only the OTHER agents' costs: the best path without this
    // edge, minus the cost of the rest of the path actually chosen.
    payments.set(k, shortestPath(costs, k).cost - (chosen.cost - cost));
  }
  return { chosen, payments, onPath };
}

function initVCG(data) {
  const trueCosts = new Map(data.edges.map((e) => [key(e.u, e.v), e.cost]));
  const slider = document.getElementById("vcg-report");
  const label = document.getElementById("vcg-report-val");
  let selected = "A|B";

  const positions = data.nodes;
  const xs = Object.values(positions).map((p) => p[0]);
  const ys = Object.values(positions).map((p) => p[1]);
  const W = 760;
  const H = 330;
  const pad = 54;
  const X = (x) =>
    pad + ((x - Math.min(...xs)) / (Math.max(...xs) - Math.min(...xs))) * (W - 2 * pad);
  const Y = (y) =>
    H - pad - ((y - Math.min(...ys)) / (Math.max(...ys) - Math.min(...ys))) * (H - 2 * pad);

  const draw = () => {
    const report = Number(slider.value) * 0.5;
    label.textContent = report.toFixed(1);

    const reported = new Map(trueCosts);
    reported.set(selected, report);
    const { chosen, payments, onPath } = runVCG(reported);

    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });

    for (const [k, cost] of reported) {
      const [u, v] = k.split("|");
      const used = onPath.has(k);
      const isSelected = k === selected;
      const [x1, y1] = positions[u];
      const [x2, y2] = positions[v];
      svg.appendChild(
        el("line", {
          x1: X(x1), y1: Y(y1), x2: X(x2), y2: Y(y2),
          stroke: used ? C("--accent") : C("--line"),
          "stroke-width": used ? 4 : 2,
          "stroke-linecap": "round",
        })
      );

      // CF runs diagonally across the whole figure: its midpoint lands on BE,
      // and anywhere near the middle collides with CE. Pin the two apart.
      const t = k === "C|F" ? 0.12 : k === "C|E" ? 0.74 : 0.5;
      const mx = X(x1 + t * (x2 - x1));
      const my = Y(y1 + t * (y2 - y1));
      const text = fmt(isSelected ? report : cost);
      const pay = payments.get(k);

      svg.appendChild(
        el("rect", {
          x: mx - 30, y: my - 15, width: 60, height: 30, rx: 6,
          fill: used ? C("--accent") : C("--panel-2"),
          stroke: isSelected ? C("--red") : used ? C("--accent") : C("--line"),
          "stroke-width": isSelected ? 2 : 1,
        })
      );
      const labelFill = `fill:${used ? "#120b1f" : C("--dim")}`;
      svg.appendChild(
        el("text", {
          x: mx, y: my - 2, "text-anchor": "middle", class: "tick",
          "font-weight": "700", style: labelFill,
        }, text)
      );
      svg.appendChild(
        el("text", {
          x: mx, y: my + 10, "text-anchor": "middle", class: "tick", style: labelFill,
        }, `paid ${fmt(pay)}`)
      );
    }

    for (const [node, [x, y]] of Object.entries(positions)) {
      svg.appendChild(
        el("circle", {
          cx: X(x), cy: Y(y), r: 17,
          fill: C("--bg"), stroke: C("--text"), "stroke-width": 1.6,
        })
      );
      svg.appendChild(
        el("text", {
          x: X(x), y: Y(y) + 5, "text-anchor": "middle", class: "tick",
          "font-weight": "700", "font-size": "14", style: `fill:${C("--text")}`,
        }, node)
      );
    }
    document.getElementById("vcg-net").replaceChildren(svg);

    const trueCost = trueCosts.get(selected);
    const used = onPath.has(selected);
    const utility = used ? payments.get(selected) - trueCost : 0;
    const honest = runVCG(trueCosts);
    const honestUtility = honest.onPath.has(selected)
      ? honest.payments.get(selected) - trueCost
      : 0;

    let verdict;
    if (Math.abs(utility - honestUtility) < 1e-9) {
      verdict = report === trueCost ? "telling the truth" : "no better than the truth";
    } else {
      verdict = utility > honestUtility ? "BETTER — a bug!" : "worse than the truth";
    }

    readout(document.getElementById("vcg-readout"), [
      ["chosen path", chosen.label],
      ["its true cost", String(trueCost)],
      [used ? "it is paid" : "not on the path", used ? fmt(payments.get(selected)) : "paid 0"],
      ["its profit", fmt(utility)],
      ["vs. honest profit", `${fmt(honestUtility)} — ${verdict}`],
    ]);
  };

  segmented(
    document.getElementById("vcg-edge"),
    data.edges.map((e) => ({ value: key(e.u, e.v), label: `${e.u}${e.v}` })),
    selected,
    (v) => {
      selected = v;
      slider.value = String(Math.round(trueCosts.get(v) / 0.5));
      draw();
    }
  );
  slider.addEventListener("input", draw);
  slider.value = String(Math.round(trueCosts.get(selected) / 0.5));
  draw();
}

/* --------------------------------------------------------------- bootstrap */

fetch("data.json")
  .then((r) => {
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  })
  .then((data) => {
    initSecondPrice(data.part1);
    initFirstPrice(data.part2);
    initVCG(data.part3);
  })
  .catch((err) => {
    for (const id of ["sp-chart", "fp-chart", "vcg-net"]) {
      const host = document.getElementById(id);
      if (host) {
        host.innerHTML =
          `<p class="err">Could not load data.json (${err.message}). ` +
          `Serve this directory over HTTP rather than opening the file directly: ` +
          `<code>python -m http.server -d docs</code></p>`;
      }
    }
  });
