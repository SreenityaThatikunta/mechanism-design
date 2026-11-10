"""Shared experiment constants and output locations."""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: Root of the repository (two levels above this file's package directory).
ROOT = Path(__file__).resolve().parents[2]

#: The static site. Figures live here because both the README and the hosted
#: page embed them, so there is only ever one copy.
WEB_DIR = ROOT / "web"
FIGURE_DIR = WEB_DIR / "figures"

#: Canonical numeric output. Kept out of the published site: nothing links to
#: these CSVs from the page, and they are the artifact CI checks for drift.
RESULTS_DIR = ROOT / "results"
TABLE_DIR = RESULTS_DIR / "tables"

#: Master seed. Every experiment derives an independent child stream from this,
#: so results are bit-reproducible and no two experiments share random draws.
SEED = 20260411

#: Auctions simulated per bid point. The brief asks for at least 100,000.
N_AUCTIONS = 200_000

#: Bidder 1's bid grid, b = 0.00, 0.01, ..., 1.00 (the horizontal axis).
BID_GRID = np.round(np.linspace(0.0, 1.0, 101), 10)

#: The three fixed private values assigned to bidder 1.
BIDDER1_VALUES = (0.2, 0.5, 0.8)

#: Fixed rival bids used for the dominance check in part 1B.
FIXED_RIVAL_BIDS = (0.1, 0.4, 0.9)

#: Bidder counts for part 2.
N_BIDDERS = (2, 3, 4)

#: Shading grid for part 2: bidder 1 bids k * v, with k from 0.1 to 2.0.
#:
#: The regular 0.05 step is unioned with the exact equilibrium shading factors
#: (n-1)/n for each n. Without that, 2/3 is simply absent from the grid and the
#: n = 3 peak would read as 0.65 -- an artifact of grid resolution rather than a
#: real result. Including them lets the simulated argmax land on the true
#: optimum, which is the whole point of the sweep.
_EQUILIBRIUM_K = tuple((n - 1) / n for n in N_BIDDERS)
K_GRID = np.unique(
    np.round(
        np.concatenate([np.arange(0.1, 2.0 + 1e-9, 0.05), np.array(_EQUILIBRIUM_K)]),
        10,
    )
)
