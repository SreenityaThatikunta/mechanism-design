"""Part 2 -- first-price sealed-bid auction with ``n`` bidders.

All ``n`` bidders draw values from U[0, 1]. Bidders ``2..n`` play the symmetric
Bayes-Nash equilibrium strategy ``b = ((n-1)/n) * v``. Bidder 1 deviates, bidding
``k * v`` for ``k`` on a grid. The winner pays its own bid, so bidder 1's utility
on a win is ``v - k*v = v(1 - k)``.

Two results are worth stating up front because they are easy to misread:

1. The curve peaks at ``k = (n-1)/n``, **not** at ``k = 1``. ``k = 1`` is truthful
   bidding, and it earns exactly zero: you only ever win by handing over your
   entire value. That is the whole reason first-price auctions are not
   strategyproof, and it is the sharpest possible contrast with part 1.

2. The fraction of auctions in which "the winner could have paid less and still
   won" is ~100% by construction, since bids are almost surely distinct and the
   winner almost surely strictly outbid the runner-up. The fraction alone is
   therefore not very informative -- the *size* of the gap is, so this module
   reports the overpayment distribution too.

Closed form. With ``c = (n-1)/n`` and ``M`` the highest of the ``n-1`` rival
values, bidder 1 wins iff ``M < k*v/c``, giving win probability
``min(1, k*v/c)^(n-1)`` and

    E[U(k)] = (1 - k) * integral_0^1 v * min(1, k*v/c)^(n-1) dv

which evaluates to ``(1-k) (k/c)^(n-1) / (n+1)`` when ``k <= c``, and to
``(1-k) [ (c/k)^2/(n+1) + (1 - (c/k)^2)/2 ]`` when ``k > c``. Maximising the
first branch over ``k`` gives ``d/dk[(1-k) k^(n-1)] = 0``, i.e. ``k = (n-1)/n``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import K_GRID, N_AUCTIONS


def equilibrium_shading(n_bidders: int) -> float:
    """The symmetric equilibrium shading factor ``(n-1)/n``."""
    return (n_bidders - 1) / n_bidders


@dataclass(frozen=True)
class ShadingSweep:
    """Bidder 1's simulated outcomes across the shading grid."""

    n_bidders: int
    k_grid: np.ndarray
    mean_utility: np.ndarray
    std_error: np.ndarray
    win_rate: np.ndarray
    could_have_paid_less: np.ndarray
    mean_overpayment: np.ndarray
    mean_overpayment_pct: np.ndarray
    efficient_allocation: np.ndarray
    n_auctions: int

    @property
    def best_k(self) -> float:
        return float(self.k_grid[int(np.argmax(self.mean_utility))])

    @property
    def equilibrium_k(self) -> float:
        return equilibrium_shading(self.n_bidders)


def simulate_shading_sweep(
    n_bidders: int,
    rng: np.random.Generator,
    k_grid: np.ndarray = K_GRID,
    n_auctions: int = N_AUCTIONS,
) -> ShadingSweep:
    """Sweep bidder 1's shading factor ``k`` against equilibrium rivals.

    Values are drawn once and reused across the whole grid (common random
    numbers), so differences between neighbouring ``k`` reflect the deviation
    rather than sampling noise.
    """
    c = equilibrium_shading(n_bidders)
    own_value = rng.uniform(size=n_auctions)
    rival_values = rng.uniform(size=(n_auctions, n_bidders - 1))
    rival_bids = c * rival_values
    highest_rival_bid = rival_bids.max(axis=1)
    highest_rival_value = rival_values.max(axis=1)

    n_k = k_grid.size
    out = {
        key: np.empty(n_k)
        for key in (
            "mean_utility",
            "std_error",
            "win_rate",
            "could_have_paid_less",
            "mean_overpayment",
            "mean_overpayment_pct",
            "efficient_allocation",
        )
    }

    for i, k in enumerate(k_grid):
        own_bid = k * own_value
        won = own_bid > highest_rival_bid
        utility = np.where(won, own_value * (1.0 - k), 0.0)

        out["mean_utility"][i] = utility.mean()
        out["std_error"][i] = utility.std(ddof=1) / np.sqrt(n_auctions)
        out["win_rate"][i] = won.mean()

        # Top two bids overall, to ask what the winner could have got away with.
        all_bids = np.concatenate([own_bid[:, None], rival_bids], axis=1)
        top_two = np.sort(all_bids, axis=1)[:, -2:]
        runner_up, winning_bid = top_two[:, 0], top_two[:, 1]

        overpayment = winning_bid - runner_up
        out["could_have_paid_less"][i] = (overpayment > 0).mean()
        out["mean_overpayment"][i] = overpayment.mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.where(winning_bid > 0, overpayment / winning_bid, 0.0)
        out["mean_overpayment_pct"][i] = pct.mean()

        # Bonus diagnostic: deviating also breaks allocative efficiency, since
        # bidder 1 no longer bids on the same scale as everyone else.
        out["efficient_allocation"][i] = np.where(
            won, own_value >= highest_rival_value, highest_rival_value >= own_value
        ).mean()

    return ShadingSweep(
        n_bidders=n_bidders,
        k_grid=k_grid,
        n_auctions=n_auctions,
        **out,
    )


def expected_utility_exact(n_bidders: int, k_grid: np.ndarray = K_GRID) -> np.ndarray:
    """Exact ``E[U(k)]`` for bidder 1, used to validate and overlay."""
    c = equilibrium_shading(n_bidders)
    k = np.asarray(k_grid, dtype=float)
    n = n_bidders

    below = (1.0 - k) * (k / c) ** (n - 1) / (n + 1)

    ratio_sq = np.divide(c, k, out=np.ones_like(k), where=k > 0) ** 2
    above = (1.0 - k) * (ratio_sq / (n + 1) + (1.0 - ratio_sq) / 2.0)

    return np.where(k <= c, below, above)


def optimal_utility_exact(n_bidders: int) -> float:
    """Utility at the equilibrium ``k = (n-1)/n``, which equals ``1/(n(n+1))``."""
    return 1.0 / (n_bidders * (n_bidders + 1))
