# Design decisions

Why this project is built the way it is. Each entry records the choice, the alternative that
was rejected, and the reason — including the ones where the honest answer is less flattering
than it could have been.

---

### D1 — Validate the simulation against closed-form theory, not against itself

**Choice.** Every simulated quantity has an analytic counterpart derived by hand. The tests
assert the Monte Carlo converges to the formula; the figures plot both on the same axes.

**Rejected.** Snapshot testing — freeze the output of a known-good run and assert future runs
match. It is far less work.

**Why.** A snapshot test only proves the code still does what it did before, which is worthless
if the first version was wrong. For these three mechanisms the expected utilities are solvable
in closed form, so there is no excuse for not checking against the real answer:

```
second-price, k rivals uniform on [0, s]:  E[u(b)] = (t/s)^k (v − k·t/(k+1)),  t = min(b, s)
first-price, k ≤ (n−1)/n:                  E[U(k)] = (1−k)(k/c)^(n−1)/(n+1)
first-price peak:                          1/(n(n+1))
```

The agreement is what makes the result believable: max discrepancy across all 18 second-price
configurations is 0.0016.

**Caveat.** The closed forms assume each rival's bid is uniform on `[0, s]`. That assumption is
load-bearing for every analytic result here, so it is itself verified empirically rather than
asserted — `test_rival_bids_are_uniform_on_their_support` compares the empirical CDF against the
declared one.

---

### D2 — Report argmax *sets*, not a single best bid

**Choice.** `argmax_set` returns every bid attaining the maximum. The summary and the page both
say how many bids tie.

**Rejected.** `np.argmax`, which returns one index.

**Why.** It would have made the headline result look stronger than it is. Against a rival that
halves its value, bidder 1 with `v = 0.8` wins every auction once `b ≥ 0.5`, so 51 bids on the
grid are exactly equally good. Against a *fixed* rival bid, between 41 and 91 bids tie. Truthful
bidding is **weakly** dominant — always in the argmax, rarely the only element of it — and a
lone argmax would quietly misrepresent that as strict dominance.

---

### D3 — Demonstrate dominance by sweeping the rival's bid, not by fixing it

**Choice.** `dominance_profile` sweeps the rival bid across [0, 1] and reports, for each of
bidder 1's bids, the fraction of rival bids against which it is optimal.

**Why.** The brief asks to fix the rival bid at 0.1, 0.4 and 0.9 and confirm `b = v` is best.
It is — but so are dozens of other bids in each case, so those three checks alone cannot
single out truthfulness. Sweeping the rival's bid can: `b = v` is the *unique* bid optimal
against all of them, which is precisely the definition of weak dominance. Any `b < v` forfeits
auctions when the rival bids between `b` and `v`; any `b > v` wins at a loss when the rival bids
between `v` and `b`.

---

### D4 — Common random numbers across the bid grid

**Choice.** One block of rival draws is generated per configuration and reused for every bid on
the grid, in both the Python and the browser simulation.

**Rejected.** Independent draws per bid point.

**Why.** Each estimate stays unbiased either way, but reusing draws makes the *differences*
between neighbouring bids far less noisy. The shape of the curve is what the plot is for, and
this estimates it much more precisely at identical cost. It is also what makes the playground's
counterfactual honest: "what would honesty have earned?" is measured on the very same auctions
the user just played, not on a separate run.

---

### D5 — Union the exact equilibrium factors into the shading grid

**Choice.** The `k` grid is the regular 0.05 steps from 0.1 to 2.0, unioned with `(n−1)/n` for
each `n`.

**Why.** `2/3` is not on a 0.05 grid. Without this the n = 3 optimum reads as 0.65, which looks
like a near-miss of the theoretical prediction but is purely an artifact of grid resolution.
Including the exact points lets the simulated argmax land on the true optimum, which is the
entire purpose of the sweep.

---

### D6 — Break shortest-path ties on the node sequence

**Choice.** Dijkstra pushes `(cost, path)` onto the heap, so equal-cost paths are ordered
lexicographically by node sequence.

**Rejected.** Letting heap order decide.

**Why.** The Figure 7.3 network has genuine ties — ABDF and ACF both cost 7. Leaving the choice
to heap ordering makes the result depend on dictionary insertion order, which is stable in
practice and therefore produces a test that passes for the wrong reason. Making the rule
explicit means the JavaScript implementation can match the Python exactly, which it does at all
42 points of the misreport grid.

---

### D7 — Run the playground simulation live rather than replaying stored curves

**Choice.** The page runs a seeded `mulberry32` Monte Carlo in the browser — 100,000 auctions —
using the same method as the Python.

**Rejected.** Shipping precomputed curves as JSON and interpolating. This was in fact the first
version.

**Why.** The central feature is the counterfactual: *on the auctions you just played*, what
would honesty have earned? Replayed curves cannot answer that, because there is no "run you just
played" — only an average someone else computed. Running it live also means the seed box is
real: change it and every number moves while every conclusion holds.

**Cost.** ~10M operations per rebuild, which lands around 100 ms. Fast enough that structural
controls can rebuild on change, while moving the bid only re-reads the cached draws.

---

### D8 — Results tables on the page render from `web/data.json`

**Choice.** The published numbers are inlined in the markup *and* overwritten from `data.json`
when it loads.

**Why.** Hardcoded numbers in HTML drift from the pipeline silently. Rendering from the
committed output means a quoted figure cannot disagree with what the code produced, while the
inline copy keeps the page correct with no network and when opened off disk.

---

### D9 — CI checks numbers for reproducibility, not images

**Choice.** CI reruns the pipeline and fails if anything in `results/tables/` or `web/data.json`
differs from what is committed. Figures are excluded.

**Rejected.** Asserting the whole of the output directory is byte-identical. This was the first
version, and it failed on its first run.

**Why.** Matplotlib rasterises text through the platform font stack, so a PNG built on macOS and
the same PNG built on the Linux runner differ in bytes while being identical plots. Asserting on
them tests the font stack rather than the code. The CSVs and JSON are plain text and are stable
across platforms, so they carry the guarantee that actually matters.

---

### D10 — The site lives in `web/`, documentation in `docs/`

**Choice.** `web/` is the deployed static site (Vercel's output directory). `docs/` holds
written documentation. `results/` holds canonical numeric output.

**Rejected.** Serving the site from `docs/`, which is what GitHub Pages expects and what this
repository did initially.

**Why.** With the site in `docs/`, there is nowhere to put design documentation without
publishing it to the website. Separating them also means the result CSVs stop being served as
part of the site — nothing ever linked to them from the page.

---

### D11 — Declare the project a static site to Vercel

**Choice.** `vercel.json` sets `"framework": null` with an empty build command.

**Why.** The first deployment failed with `PYTHON_ENTRYPOINT_NOT_FOUND`. Vercel sees
`pyproject.toml` at the repository root, selects the Python runtime, and then looks for a
WSGI/ASGI entrypoint this project has no reason to have. `outputDirectory` only says where
output is read *from*; it does not affect runtime detection.

---

### D12 — Deviations from the brief, and why

Three places where the brief and the mathematics disagree. In each case the code does the
correct thing and says so rather than reproducing the expected answer:

- **Part 1B asks to "confirm `b = v` gives the highest utility" against a fixed rival bid.** It
  does, but not uniquely — see D2. Reported as weak dominance.
- **Part 2's expected optimum.** The peak is at `k = (n−1)/n`, *not* at `k = 1`. Truthful bidding
  in a first-price auction earns exactly zero while still winning 44–75% of auctions. This is
  the sharpest contrast with Part 1 and the reason first-price auctions are not strategyproof.
- **"Fraction of auctions where the winner could have paid less".** This is ~100% by
  construction: bids are drawn from continuous distributions, so the winner almost surely
  strictly outbid the runner-up. The fraction alone carries almost no information, so the
  *size* of the gap is reported alongside it.

Part 1C's "repeat parts A and C" is read as "A and B".
