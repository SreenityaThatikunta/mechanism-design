"""Shared experiment constants and output locations."""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: Root of the repository (two levels above this file's package directory).
ROOT = Path(__file__).resolve().parents[2]

#: Figures and tables are written straight into ``docs/`` so that the same
#: artifacts serve the README, the test suite and the hosted demo page.
FIGURE_DIR = ROOT / "docs" / "figures"
TABLE_DIR = ROOT / "docs" / "tables"

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

#: Shading grid for part 2: bidder 1 bids k * v.
K_GRID = np.round(np.arange(0.1, 2.0 + 1e-9, 0.05), 10)

#: Bidder counts for part 2.
N_BIDDERS = (2, 3, 4)
