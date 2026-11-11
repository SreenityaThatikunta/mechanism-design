/* Gambit: the interactive page.
 *
 * The playground runs a real seeded Monte Carlo in the browser, using the same
 * method as the repository's Python: one block of rival draws is reused across
 * every bid on the grid (common random numbers), so the counterfactual
 * "what would honesty have earned?" is measured on the very same auctions you
 * just played rather than on a separate run.
 *
 * The VCG tab recomputes Dijkstra and the VCG payment rule live. Its output was
 * diffed against the Python sweep at every point on the 0..10 grid.
 */
"use strict";

const CSS = getComputedStyle(document.documentElement);
const C = (n) => CSS.getPropertyValue(n).trim();

const N_AUCTIONS = 100_000;
const TAPE_ROWS = 10;

const num = (v, d = 4) => {
  if (!isFinite(v)) return "∞";
  let s = v.toFixed(d);
  // Only ever trim zeros that sit after a decimal point. Trimming unconditionally
  // turns 2900 into 29 and 5970 into 597, which is how this shipped at first.
  if (s.includes(".")) s = s.replace(/0+$/, "").replace(/\.$/, "");
  return s === "-0" ? "0" : s;
};
const pct = (v, d = 1) => `${(100 * v).toFixed(d)}%`;

/* ------------------------------------------------------------------ tabs */

function initTabs() {
  const links = [...document.querySelectorAll("nav a[data-tab]")];
  const show = (name) => {
    const known = links.some((l) => l.dataset.tab === name);
    const target = known ? name : "playground";
    for (const l of links) l.classList.toggle("on", l.dataset.tab === target);
    for (const s of document.querySelectorAll("section.tab")) {
      s.classList.toggle("on", s.id === `tab-${target}`);
    }
  };
  addEventListener("hashchange", () => {
    show(location.hash.slice(1));
    scrollTo({ top: 0 });
  });
  show(location.hash.slice(1));
}

/* ------------------------------------------------------------------- rng */

/** mulberry32: small, fast, seeded. Same seed gives the same run every time. */
function rng(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ---------------------------------------------------------------- charts */

const NS = "http://www.w3.org/2000/svg";
const svgEl = (name, attrs = {}, text) => {
  const n = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text !== undefined) n.textContent = text;
  return n;
};

function ticks(lo, hi, count) {
  const span = hi - lo || 1;
  const raw = span / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag;
  const out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) {
    out.push(Math.abs(t) < 1e-9 ? 0 : t);
  }
  return out;
}

function drawChart(host, spec) {
  const W = 820;
  const H = 300;
  const M = { l: 58, r: 16, t: 16, b: 44 };
  const iw = W - M.l - M.r;
  const ih = H - M.t - M.b;

  const all = [spec.dots, spec.line].filter(Boolean).flat();
  let lo = Math.min(...all);
  let hi = Math.max(...all);
  const pad = (hi - lo || 1) * 0.14;
  lo -= pad;
  hi += pad;

  const xlo = Math.min(...spec.xs);
  const xhi = Math.max(...spec.xs);
  const X = (v) => M.l + ((v - xlo) / (xhi - xlo || 1)) * iw;
  const Y = (v) => M.t + ih - ((v - lo) / (hi - lo || 1)) * ih;

  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  svg.appendChild(svgEl("title", {}, spec.title ?? "chart"));

  for (const t of ticks(lo, hi, 5)) {
    const y = Y(t);
    svg.appendChild(svgEl("line", { x1: M.l, x2: M.l + iw, y1: y, y2: y, class: "axis" }));
    svg.appendChild(
      svgEl("text", { x: M.l - 9, y: y + 3.5, "text-anchor": "end", class: "tick" }, num(t, 3))
    );
  }
  for (const t of ticks(xlo, xhi, 6)) {
    svg.appendChild(
      svgEl("text",
        { x: X(t), y: M.t + ih + 19, "text-anchor": "middle", class: "tick" }, num(t, 2))
    );
  }
  if (lo < 0 && hi > 0) {
    svg.appendChild(
      svgEl("line", {
        x1: M.l, x2: M.l + iw, y1: Y(0), y2: Y(0),
        stroke: C("--line-2"), "stroke-width": 1,
      })
    );
  }

  for (const v of spec.vlines ?? []) {
    const x = X(v.x);
    svg.appendChild(
      svgEl("line", {
        x1: x, x2: x, y1: M.t, y2: M.t + ih,
        stroke: v.colour, "stroke-width": 1.3, "stroke-dasharray": "3 3",
      })
    );
    if (v.label) {
      const flip = x > M.l + iw * 0.6;
      svg.appendChild(
        svgEl("text", {
          x: x + (flip ? -7 : 7), y: M.t + 12,
          "text-anchor": flip ? "end" : "start",
          class: "tick", style: `fill:${v.colour}`,
        }, v.label)
      );
    }
  }

  if (spec.dots) {
    for (let i = 0; i < spec.dots.length; i++) {
      svg.appendChild(
        svgEl("circle",
          { cx: X(spec.xs[i]), cy: Y(spec.dots[i]), r: 2.3, fill: C("--accent"), opacity: ".85" })
      );
    }
  }
  if (spec.line) {
    const d = spec.line.map((y, i) => `${i ? "L" : "M"}${X(spec.xs[i])},${Y(y)}`).join("");
    svg.appendChild(
      svgEl("path", {
        d, fill: "none", stroke: C("--text-2"),
        "stroke-width": 1.8, "stroke-dasharray": "6 4",
      })
    );
  }
  if (spec.marker) {
    const x = X(spec.marker.x);
    const y = Y(spec.marker.y);
    svg.appendChild(
      svgEl("line", {
        x1: x, x2: x, y1: y, y2: M.t + ih, stroke: C("--text"),
        "stroke-width": 1, opacity: ".35",
      })
    );
    svg.appendChild(
      svgEl("circle",
        { cx: x, cy: y, r: 5.5, fill: C("--text"), stroke: C("--bg"), "stroke-width": 2 })
    );
  }

  svg.appendChild(
    svgEl("line", { x1: M.l, x2: M.l + iw, y1: M.t + ih, y2: M.t + ih, class: "axis" })
  );
  svg.appendChild(
    svgEl("text",
      { x: M.l + iw / 2, y: H - 6, "text-anchor": "middle", class: "axlab" }, spec.xLabel)
  );
  svg.appendChild(
    svgEl("text", {
      x: 12, y: M.t + ih / 2, "text-anchor": "middle", class: "axlab",
      transform: `rotate(-90 12 ${M.t + ih / 2})`,
    }, spec.yLabel)
  );
  host.replaceChildren(svg);
}

/* --------------------------------------------------------------- widgets */

function segmented(host, options, current, onChange) {
  host.replaceChildren();
  const buttons = options.map((o) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = o.label;
    b.setAttribute("aria-pressed", String(o.value === current));
    b.addEventListener("click", () => {
      current = o.value;
      buttons.forEach((btn, i) =>
        btn.setAttribute("aria-pressed", String(options[i].value === current))
      );
      onChange(o.value);
    });
    host.appendChild(b);
    return b;
  });
}

function slider({ label, min, max, step, value, format, onInput }) {
  const wrap = document.createElement("div");
  wrap.className = "field";
  wrap.innerHTML =
    `<div class="row"><label></label><span class="val"></span></div>` +
    `<input type="range">`;
  wrap.querySelector("label").textContent = label;
  const out = wrap.querySelector(".val");
  const input = wrap.querySelector("input");
  Object.assign(input, { min, max, step, value });
  const sync = () => { out.textContent = format(Number(input.value)); };
  input.addEventListener("input", () => { sync(); onInput(Number(input.value)); });
  sync();
  return { wrap, input, sync };
}

function kpis(host, items) {
  host.replaceChildren(...items.map(({ k, v, cls }) => {
    const d = document.createElement("div");
    d.className = "kpi";
    const kk = document.createElement("div");
    kk.className = "k";
    kk.textContent = k;
    const vv = document.createElement("div");
    vv.className = `v ${cls ?? ""}`.trim();
    vv.textContent = v;
    d.append(kk, vv);
    return d;
  }));
}

function verdict(host, { mark, markClass, head, sub }) {
  host.replaceChildren();
  const m = document.createElement("div");
  m.className = `mark ${markClass}`;
  m.textContent = mark;
  const t = document.createElement("div");
  t.className = "txt";
  const h = document.createElement("div");
  h.className = "head";
  h.textContent = head;
  const s = document.createElement("div");
  s.className = "sub";
  s.textContent = sub;
  t.append(h, s);
  host.append(m, t);
}

function table(host, head, rows) {
  host.replaceChildren();
  const thead = document.createElement("thead");
  const hr = document.createElement("tr");
  for (const h of head) {
    const th = document.createElement("th");
    th.textContent = h;
    hr.appendChild(th);
  }
  thead.appendChild(hr);
  const tbody = document.createElement("tbody");
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const cell of row) {
      const td = document.createElement("td");
      if (cell && typeof cell === "object") {
        td.textContent = cell.text;
        if (cell.cls) td.className = cell.cls;
      } else {
        td.textContent = cell;
      }
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  host.append(thead, tbody);
}

/* ---------------------------------------------------------- second price */

const STRATEGIES = {
  truthful: { label: "truthful", support: 1, bid: (v, u) => v },
  half: { label: "half", support: 0.5, bid: (v, u) => 0.5 * v },
  random: { label: "random", support: 1, bid: (v, u) => u },
};

const BID_GRID = Array.from({ length: 101 }, (_, i) => +(i / 100).toFixed(2));

/** Exact E[u(b)]: k rivals bidding uniformly on [0, s]. */
const spExact = (v, s, k, b) => {
  const t = Math.min(b, s);
  return (t / s) ** k * (v - (k * t) / (k + 1));
};

function spBuild({ value, rivals, strategy, seed }) {
  const r = rng(seed);
  const strat = STRATEGIES[strategy];
  const m = new Float64Array(N_AUCTIONS);

  for (let i = 0; i < N_AUCTIONS; i++) {
    let top = -1;
    for (let j = 0; j < rivals; j++) {
      const bid = strat.bid(r(), r());
      if (bid > top) top = bid;
    }
    m[i] = top;
  }

  // Common random numbers: the same m[] is swept across the whole bid grid.
  const sim = BID_GRID.map((b) => {
    let sum = 0;
    for (let i = 0; i < N_AUCTIONS; i++) if (b > m[i]) sum += value - m[i];
    return sum / N_AUCTIONS;
  });
  const exact = BID_GRID.map((b) => spExact(value, strat.support, rivals, b));
  return { m, sim, exact };
}

function spTotals(cache, value, bid) {
  let won = 0;
  let profit = 0;
  for (let i = 0; i < N_AUCTIONS; i++) {
    if (bid > cache.m[i]) { won++; profit += value - cache.m[i]; }
  }
  return { won, profit };
}

/* ----------------------------------------------------------- first price */

const K_GRID = (() => {
  const g = new Set();
  for (let k = 0.1; k <= 2.0001; k += 0.05) g.add(+k.toFixed(4));
  for (const n of [2, 3, 4]) g.add(+((n - 1) / n).toFixed(4));
  return [...g].sort((a, b) => a - b);
})();

const fpEquilibrium = (n) => (n - 1) / n;

/** Exact E[U(k)] for the deviating bidder against n-1 equilibrium rivals. */
function fpExact(n, k) {
  const c = fpEquilibrium(n);
  if (k <= c) return ((1 - k) * (k / c) ** (n - 1)) / (n + 1);
  const ratioSq = (c / k) ** 2;
  return (1 - k) * (ratioSq / (n + 1) + (1 - ratioSq) / 2);
}

function fpBuild({ bidders, seed }) {
  const r = rng(seed);
  const c = fpEquilibrium(bidders);
  const own = new Float64Array(N_AUCTIONS);
  const topRival = new Float64Array(N_AUCTIONS);

  for (let i = 0; i < N_AUCTIONS; i++) {
    own[i] = r();
    let top = -1;
    for (let j = 0; j < bidders - 1; j++) {
      const bid = c * r();
      if (bid > top) top = bid;
    }
    topRival[i] = top;
  }

  const sim = K_GRID.map((k) => {
    let sum = 0;
    for (let i = 0; i < N_AUCTIONS; i++) {
      if (k * own[i] > topRival[i]) sum += own[i] * (1 - k);
    }
    return sum / N_AUCTIONS;
  });
  const exact = K_GRID.map((k) => fpExact(bidders, k));
  return { own, topRival, sim, exact, c };
}

function fpTotals(cache, k) {
  let won = 0;
  let profit = 0;
  for (let i = 0; i < N_AUCTIONS; i++) {
    if (k * cache.own[i] > cache.topRival[i]) { won++; profit += cache.own[i] * (1 - k); }
  }
  return { won, profit };
}

/* ------------------------------------------------------------ playground */

function initPlayground() {
  const state = {
    mech: "second",
    value: 0.5,
    bid: 0.5,
    rivals: 1,
    strategy: "truthful",
    bidders: 2,
    k: 0.5,
    seed: 1,
  };
  let cache = null;
  let choiceSlider = null;

  const fields = document.getElementById("pg-fields");
  const seedInput = document.getElementById("pg-seed");

  const rebuild = () => {
    cache = state.mech === "second"
      ? spBuild({ value: state.value, rivals: state.rivals,
                  strategy: state.strategy, seed: state.seed })
      : fpBuild({ bidders: state.bidders, seed: state.seed });
    refresh();
  };

  const refresh = () => {
    if (!cache) return;
    state.mech === "second" ? refreshSecond() : refreshFirst();
  };

  /* ---- second price ---- */
  function refreshSecond() {
    const { value, bid } = state;
    const mine = spTotals(cache, value, bid);
    const honest = spTotals(cache, value, value);

    const strat = STRATEGIES[state.strategy];
    const exactAtBid = spExact(value, strat.support, state.rivals, bid);
    const best = Math.max(...cache.exact);
    const gap = best - exactAtBid;
    const optimal = BID_GRID.filter((_, i) => cache.exact[i] >= best - 1e-12);

    let v;
    if (gap < 1e-12) {
      v = Math.abs(bid - value) < 1e-9
        ? { mark: "=", markClass: "ok", head: "You bid your value, and nothing beats it",
            sub: `No bid on the grid does better. ${optimal.length > 1
              ? `${optimal.length} bids tie here, because a rival that halves its value never bids above ${strat.support}.`
              : "It is the unique best bid against these rivals."}` }
        : { mark: "=", markClass: "neutral", head: "Tied with honest bidding",
            sub: `Bidding ${num(bid, 2)} earns exactly what bidding your value earns. ${optimal.length} bids tie for best, Truthfulness is optimal, just not uniquely so.` };
    } else {
      v = { mark: "↓", markClass: "bad",
            head: `Honesty wins: you gave up ${num(gap, 4)} per auction`,
            sub: bid > value
              ? `Bidding above your value wins auctions you then pay more for than they are worth.`
              : `Bidding below your value forfeits auctions you would have profited from.` };
    }
    verdict(document.getElementById("pg-verdict"), v);

    kpis(document.getElementById("pg-kpis"), [
      { k: "Auctions won", v: `${mine.won.toLocaleString()} · ${pct(mine.won / N_AUCTIONS, 0)}` },
      { k: "Your total profit", v: num(mine.profit, 0) },
      { k: "Honest, same draws", v: num(honest.profit, 0), cls: "ok" },
      { k: "Difference", v: `${mine.profit >= honest.profit ? "+" : ""}${num(mine.profit - honest.profit, 0)}`,
        cls: mine.profit >= honest.profit - 1e-9 ? "" : "bad" },
    ]);

    drawChart(document.getElementById("pg-chart"), {
      xs: BID_GRID, dots: cache.sim, line: cache.exact,
      vlines: [{ x: value, colour: C("--truth"), label: `honest b = ${value}` }],
      marker: { x: bid, y: exactAtBid },
      xLabel: "your bid, b", yLabel: "expected profit per auction",
      title: "Expected profit against your bid",
    });

    legend([
      [C("--accent"), "dot", `simulated (${N_AUCTIONS.toLocaleString()} auctions)`],
      [C("--text-2"), "line", "exact expected utility"],
      [C("--truth"), "line", "honest bid"],
    ]);

    const rows = [];
    for (let i = 0; i < TAPE_ROWS; i++) {
      const m = cache.m[i];
      const won = bid > m;
      const profit = won ? value - m : 0;
      rows.push([
        String(i + 1), num(bid, 2), num(m, 3),
        { text: won ? "won" : "lost", cls: won ? "won" : "lost" },
        won ? num(m, 3) : "-",
        { text: num(profit, 3), cls: profit < -1e-12 ? "loss" : won ? "won" : "lost" },
      ]);
    }
    table(document.getElementById("pg-tape"),
      ["#", "your bid", "top rival bid", "result", "you pay", "profit"], rows);
  }

  /* ---- first price ---- */
  function refreshFirst() {
    const { bidders, k } = state;
    const c = fpEquilibrium(bidders);
    const mine = fpTotals(cache, k);
    const honest = fpTotals(cache, 1);
    const equilibrium = fpTotals(cache, c);

    const exactAtK = fpExact(bidders, k);
    const peak = 1 / (bidders * (bidders + 1));
    const beatsHonest = exactAtK > 1e-12;

    verdict(document.getElementById("pg-verdict"), beatsHonest
      ? { mark: "↑", markClass: "bad",
          head: `Shading beats honesty by ${num(exactAtK, 4)} per auction`,
          sub: `Bidding your true value earns exactly 0 here: you only win by handing over the whole surplus. This mechanism is not strategyproof, and the best you can do is k = ${num(c, 4)}.` }
      : { mark: k > 1 ? "↓" : "=", markClass: k > 1 ? "bad" : "neutral",
          head: k > 1 ? "Bidding above your value loses money" : "You earn nothing here",
          sub: k > 1
            ? `Every auction you win costs more than the item is worth to you.`
            : `At k = 1 you bid your true value, win ${pct(honest.won / N_AUCTIONS, 0)} of auctions, and make exactly nothing on any of them.` });

    kpis(document.getElementById("pg-kpis"), [
      { k: "Auctions won", v: `${mine.won.toLocaleString()} · ${pct(mine.won / N_AUCTIONS, 0)}` },
      { k: "Your total profit", v: num(mine.profit, 0),
        cls: mine.profit < -1e-9 ? "bad" : "" },
      { k: "Honest k=1, same draws", v: num(honest.profit, 0), cls: "bad" },
      { k: `Best possible k=${num(c, 3)}`, v: num(equilibrium.profit, 0), cls: "ok" },
    ]);

    drawChart(document.getElementById("pg-chart"), {
      xs: K_GRID, dots: cache.sim, line: cache.exact,
      vlines: [
        { x: c, colour: C("--truth"), label: `equilibrium k = ${num(c, 3)}` },
        { x: 1, colour: C("--dev"), label: "honest k = 1" },
      ],
      marker: { x: k, y: exactAtK },
      xLabel: "your shading factor k   (you bid k × your value)",
      yLabel: "expected profit per auction",
      title: "Expected profit against your shading factor",
    });

    legend([
      [C("--accent"), "dot", `simulated (${N_AUCTIONS.toLocaleString()} auctions)`],
      [C("--text-2"), "line", "exact expected utility"],
      [C("--truth"), "line", `equilibrium (n−1)/n`],
      [C("--dev"), "line", "truthful bidding"],
    ]);
    void peak;

    const rows = [];
    for (let i = 0; i < TAPE_ROWS; i++) {
      const v1 = cache.own[i];
      const myBid = k * v1;
      const top = cache.topRival[i];
      const won = myBid > top;
      const profit = won ? v1 * (1 - k) : 0;
      rows.push([
        String(i + 1), num(v1, 3), num(myBid, 3), num(top, 3),
        { text: won ? "won" : "lost", cls: won ? "won" : "lost" },
        won ? num(myBid, 3) : "-",
        { text: num(profit, 3), cls: profit < -1e-12 ? "loss" : won ? "won" : "lost" },
      ]);
    }
    table(document.getElementById("pg-tape"),
      ["#", "your value", "your bid", "top rival bid", "result", "you pay", "profit"], rows);
  }

  function legend(items) {
    const host = document.getElementById("pg-legend");
    host.replaceChildren(...items.map(([colour, kind, text]) => {
      const s = document.createElement("span");
      const i = document.createElement("i");
      i.className = `sw ${kind === "dot" ? "dot" : ""}`.trim();
      i.style.background = colour;
      s.append(i, document.createTextNode(text));
      return s;
    }));
  }

  /* ---- control panel, rebuilt when the mechanism changes ---- */
  function renderFields() {
    fields.replaceChildren();

    if (state.mech === "second") {
      const value = slider({
        label: "Your value", min: 0.05, max: 0.95, step: 0.05, value: state.value,
        format: (v) => v.toFixed(2),
        onInput: (v) => {
          state.value = v;
          // Keep the bid on the value while they coincide, so the honest case
          // stays the default rather than silently drifting off it.
          if (Math.abs(state.bid - state.value) < 1e-9) state.bid = v;
          rebuild();
          if (choiceSlider) { choiceSlider.input.value = String(state.bid); choiceSlider.sync(); }
        },
      });
      const bid = slider({
        label: "Your bid", min: 0, max: 1, step: 0.01, value: state.bid,
        format: (v) => v.toFixed(2),
        onInput: (v) => { state.bid = v; refresh(); },
      });
      choiceSlider = bid;

      const rivalsWrap = groupSeg("Rivals",
        [1, 2, 3].map((n) => ({ value: n, label: String(n) })), state.rivals,
        (v) => { state.rivals = v; rebuild(); });
      const stratWrap = groupSeg("Rivals bid",
        Object.entries(STRATEGIES).map(([k, s]) => ({ value: k, label: s.label })),
        state.strategy, (v) => { state.strategy = v; rebuild(); });

      fields.append(value.wrap, bid.wrap, rivalsWrap, stratWrap);
    } else {
      const biddersWrap = groupSeg("Bidders",
        [2, 3, 4].map((n) => ({ value: n, label: String(n) })), state.bidders,
        (v) => {
          state.bidders = v;
          state.k = +fpEquilibrium(v).toFixed(4);
          rebuild();
          if (choiceSlider) {
            choiceSlider.input.value = String(K_GRID.indexOf(state.k));
            choiceSlider.sync();
          }
        });

      const idx = Math.max(0, K_GRID.indexOf(+state.k.toFixed(4)));
      const k = slider({
        label: "Your shading k", min: 0, max: K_GRID.length - 1, step: 1, value: idx,
        format: (i) => num(K_GRID[i], 4),
        onInput: (i) => { state.k = K_GRID[i]; refresh(); },
      });
      choiceSlider = k;

      fields.append(biddersWrap, k.wrap);
    }
  }

  function groupSeg(title, options, current, onChange) {
    const wrap = document.createElement("div");
    const h = document.createElement("h4");
    h.textContent = title;
    const seg = document.createElement("div");
    seg.className = "seg sm";
    wrap.append(h, seg);
    segmented(seg, options, current, onChange);
    return wrap;
  }

  segmented(document.getElementById("pg-mech"),
    [{ value: "second", label: "Second-price" }, { value: "first", label: "First-price" }],
    state.mech,
    (v) => {
      state.mech = v;
      if (v === "first") state.k = +fpEquilibrium(state.bidders).toFixed(4);
      renderFields();
      rebuild();
    });

  seedInput.addEventListener("change", () => {
    const parsed = parseInt(seedInput.value, 10);
    state.seed = Number.isFinite(parsed) ? parsed : 1;
    seedInput.value = String(state.seed);
    rebuild();
  });
  // A fresh run means fresh draws. Rebuilding on the same seed replays the
  // identical stream, so the numbers would not move and the button would look
  // broken; drawing a new seed first is what makes "run" mean anything. Typing
  // a seed into the box still reproduces that exact run.
  const runFresh = () => {
    state.seed = Math.floor(Math.random() * 99999) + 1;
    seedInput.value = String(state.seed);
    rebuild();
  };

  document.getElementById("pg-reseed").addEventListener("click", runFresh);
  document.getElementById("pg-run").addEventListener("click", runFresh);

  renderFields();
  rebuild();
}

/* ---------------------------------------------------------------- VCG */

const EDGE_COSTS = {
  "A|B": 3, "A|C": 2, "B|D": 2, "B|E": 1, "C|E": 3, "C|F": 5, "D|F": 2, "E|F": 1,
};
const NODES = { A: [0, 0], B: [1, 1], D: [2, 1], C: [1, -1], E: [2, -1], F: [3, 0] };
const ekey = (u, v) => (u <= v ? `${u}|${v}` : `${v}|${u}`);

/**
 * Cheapest A→F path, optionally with one edge removed. Ties break on the
 * lexicographically smallest node sequence, matching the Python: the network
 * has real ties (ABDF and ACF both cost 7).
 */
function shortestPath(costs, exclude) {
  const adj = new Map();
  for (const [k, cost] of Object.entries(costs)) {
    if (k === exclude) continue;
    const [u, v] = k.split("|");
    if (!adj.has(u)) adj.set(u, []);
    if (!adj.has(v)) adj.set(v, []);
    adj.get(u).push([v, cost]);
    adj.get(v).push([u, cost]);
  }
  const queue = [[0, ["A"]]];
  const settled = new Set();
  while (queue.length) {
    queue.sort((a, b) => a[0] - b[0] || a[1].join("").localeCompare(b[1].join("")));
    const [cost, nodes] = queue.shift();
    const node = nodes.at(-1);
    if (settled.has(node)) continue;
    if (node === "F") {
      return { cost, nodes, label: nodes.join(""),
               edges: nodes.slice(1).map((n, i) => ekey(nodes[i], n)) };
    }
    settled.add(node);
    for (const [next, step] of adj.get(node) ?? []) {
      if (!settled.has(next)) queue.push([cost + step, [...nodes, next]]);
    }
  }
  return { cost: Infinity, nodes: [], label: "no path", edges: [] };
}

function runVCG(costs) {
  const chosen = shortestPath(costs, null);
  const onPath = new Set(chosen.edges);
  const payments = {};
  for (const [k, cost] of Object.entries(costs)) {
    // Uses only the OTHER agents' costs: best path without this edge, minus the
    // cost of the rest of the path actually chosen.
    payments[k] = onPath.has(k) ? shortestPath(costs, k).cost - (chosen.cost - cost) : 0;
  }
  return { chosen, payments, onPath };
}

const LABEL_T = { "C|F": 0.12, "C|E": 0.74 };

function initVCG() {
  const slider = document.getElementById("vcg-report");
  const out = document.getElementById("vcg-report-val");
  let selected = "A|B";

  const W = 820;
  const H = 330;
  const padX = 62;
  const padY = 56;
  const X = (x) => padX + (x / 3) * (W - 2 * padX);
  const Y = (y) => H / 2 - (y / 1) * ((H - 2 * padY) / 2);

  const draw = () => {
    const report = Number(slider.value) * 0.5;
    out.textContent = report.toFixed(1);

    const reported = { ...EDGE_COSTS, [selected]: report };
    const { chosen, payments, onPath } = runVCG(reported);

    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
    svg.appendChild(svgEl("title", {}, `Chosen path ${chosen.label}`));

    for (const [k, cost] of Object.entries(reported)) {
      const [u, v] = k.split("|");
      const used = onPath.has(k);
      const isSel = k === selected;
      const [x1, y1] = NODES[u];
      const [x2, y2] = NODES[v];
      svg.appendChild(svgEl("line", {
        x1: X(x1), y1: Y(y1), x2: X(x2), y2: Y(y2),
        stroke: used ? C("--accent") : C("--line-2"),
        "stroke-width": used ? 3.4 : 1.6, "stroke-linecap": "round",
      }));

      const t = LABEL_T[k] ?? 0.5;
      const mx = X(x1 + t * (x2 - x1));
      const my = Y(y1 + t * (y2 - y1));
      svg.appendChild(svgEl("rect", {
        x: mx - 31, y: my - 15, width: 62, height: 30, rx: 7,
        fill: used ? C("--accent") : C("--panel-2"),
        stroke: isSel ? C("--dev") : used ? C("--accent") : C("--line-2"),
        "stroke-width": isSel ? 1.8 : 1,
      }));
      const fill = `fill:${used ? C("--bg") : C("--text-2")}`;
      svg.appendChild(svgEl("text", {
        x: mx, y: my - 2, "text-anchor": "middle", class: "tick",
        "font-weight": "500", style: fill,
      }, num(cost, 1)));
      svg.appendChild(svgEl("text", {
        x: mx, y: my + 10, "text-anchor": "middle", class: "tick", style: fill,
      }, `paid ${num(payments[k], 1)}`));
    }

    for (const [node, [x, y]] of Object.entries(NODES)) {
      svg.appendChild(svgEl("circle", {
        cx: X(x), cy: Y(y), r: 17, fill: C("--panel"),
        stroke: C("--text-2"), "stroke-width": 1.4,
      }));
      svg.appendChild(svgEl("text", {
        x: X(x), y: Y(y) + 5, "text-anchor": "middle", class: "tick",
        "font-weight": "500", "font-size": "13", style: `fill:${C("--text")}`,
      }, node));
    }
    document.getElementById("vcg-net").replaceChildren(svg);

    const trueCost = EDGE_COSTS[selected];
    const used = onPath.has(selected);
    const profit = used ? payments[selected] - trueCost : 0;
    const honest = runVCG(EDGE_COSTS);
    const honestProfit = honest.onPath.has(selected)
      ? honest.payments[selected] - trueCost
      : 0;
    const name = selected.replace("|", "");

    const lying = Math.abs(report - trueCost) > 1e-9;
    let v;
    if (!lying) {
      v = { mark: "=", markClass: "ok", head: `${name} is telling the truth`,
            sub: used
              ? `It is on the chosen path, paid ${num(payments[selected], 1)} against a true cost of ${trueCost}.`
              : `It is not on the chosen path, so it is paid nothing and earns nothing.` };
    } else if (profit > honestProfit + 1e-9) {
      v = { mark: "!", markClass: "bad", head: "Lying paid off, which would be a bug",
            sub: `VCG is strategyproof, so this should be unreachable.` };
    } else if (profit < honestProfit - 1e-9) {
      v = { mark: "↓", markClass: "bad",
            head: `Lying cost ${name} ${num(honestProfit - profit, 1)}`,
            sub: used
              ? `Under-reporting won it the contract, but it is paid ${num(payments[selected], 1)} against a true cost of ${trueCost}.`
              : `Over-reporting priced it off the path, so it earns nothing instead of ${num(honestProfit, 1)}.` };
    } else {
      v = { mark: "=", markClass: "neutral", head: "No better than the truth",
            sub: used
              ? `Its payment is fixed by the other agents' costs, so moving its own report changes nothing while it stays on the path.`
              : `It is off the path either way, earning nothing.` };
    }
    verdict(document.getElementById("vcg-verdict"), v);

    const total = Object.values(payments).reduce((a, b) => a + b, 0);
    kpis(document.getElementById("vcg-kpis"), [
      { k: "Chosen path", v: chosen.label },
      { k: `${name} is paid`, v: used ? num(payments[selected], 1) : "0" },
      { k: `${name} profit`, v: num(profit, 1),
        cls: profit > 1e-9 ? "ok" : profit < -1e-9 ? "bad" : "" },
      { k: "Total paid out", v: num(total, 1) },
    ]);
  };

  segmented(document.getElementById("vcg-edge"),
    Object.keys(EDGE_COSTS).map((k) => ({ value: k, label: k.replace("|", "") })),
    selected,
    (v) => {
      selected = v;
      slider.value = String(Math.round(EDGE_COSTS[v] / 0.5));
      draw();
    });
  slider.addEventListener("input", draw);
  document.getElementById("vcg-honest").addEventListener("click", () => {
    slider.value = String(Math.round(EDGE_COSTS[selected] / 0.5));
    draw();
  });
  slider.value = String(Math.round(EDGE_COSTS[selected] / 0.5));
  draw();
}

/* --------------------------------------------------------------- start */

initTabs();
initPlayground();
initVCG();

/* --------------------------------------------------------------- results */

/**
 * Re-render the results tables from the committed pipeline output.
 *
 * The markup ships with these numbers inline so the page still reads correctly
 * with no network and when opened straight off disk. When data.json does load,
 * it wins, which means a figure quoted here can never drift away from what the
 * pipeline actually produced.
 */
function initResults(part2) {
  const rows = (id, build) => {
    const t = document.getElementById(id);
    if (!t) return;
    const body = t.querySelector("tbody");
    body.replaceChildren(...[2, 3, 4].map((n) => {
      const tr = document.createElement("tr");
      for (const cell of build(n, part2.series[String(n)])) {
        const td = document.createElement("td");
        td.textContent = cell;
        tr.appendChild(td);
      }
      return tr;
    }));
  };

  rows("res-shading", (n, s) => [
    String(n),
    s.equilibrium_k.toFixed(4),
    s.best_k_simulated.toFixed(4),
    s.peak.toFixed(6),
    "0",
  ]);
  rows("res-overpay", (n, s) => [
    String(n),
    pct(s.could_pay_less_at_eq, 2),
    s.overpay_abs_at_eq.toFixed(4),
    pct(s.overpay_pct_at_eq, 1),
  ]);
}

fetch("data.json")
  .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
  .then((d) => initResults(d.part2))
  .catch(() => { /* inline numbers already in the markup stand in */ });
