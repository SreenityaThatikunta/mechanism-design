"""Part 1 -- second-price sealed-bid (Vickrey) auction.

Bidder 1 holds a fixed private value ``v`` and submits a bid ``b``. Each rival
draws a value from U[0, 1] and bids according to its strategy. Bidder 1 wins
when its bid strictly exceeds every rival bid, and then pays the highest rival
bid, so its utility is ``v - m`` where ``m`` is that highest rival bid (and zero
when it loses).

The decisive structural fact is that ``b`` determines *whether* bidder 1 wins but
never *what it pays*. Raising ``b`` above ``v`` therefore only buys auctions that
lose money, and lowering it below ``v`` only forfeits auctions that would have
turned a profit.

Every simulation here has an exact counterpart in closed form. All three rival
strategies happen to produce bids that are uniform on ``[0, support_max]``
(truthful and value-ignoring-random are in fact *identically* distributed, which
is why their panels coincide), so a single formula covers all of them:

    E[u(b)] = (t / s)^k * (v - k * t / (k + 1)),   t = min(b, s)

for ``k`` rivals each bidding uniformly on ``[0, s]``. Differentiating gives
``dE/db = (v - b) * f(b)``, which changes sign exactly at ``b = v``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import BID_GRID, N_AUCTIONS
from .strategies import RivalStrategy


@dataclass(frozen=True)
class BidSweep:
    """Simulated expected utility of bidder 1 across its bid grid."""

    value: float
    strategy_key: str
    n_rivals: int
    bids: np.ndarray
    mean_utility: np.ndarray
    std_error: np.ndarray
    win_rate: np.ndarray
    n_auctions: int

    @property
    def best_bid(self) -> float:
        """The bid with the highest simulated expected utility."""
        return float(self.bids[int(np.argmax(self.mean_utility))])


def draw_highest_rival_bid(
    strategy: RivalStrategy,
    n_rivals: int,
    n_auctions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw the highest rival bid for each of ``n_auctions`` independent auctions.

    Rival values are always drawn from U[0, 1]; ``noise`` is an extra independent
    U[0, 1] stream consumed only by the value-ignoring strategy.
    """
    values = rng.uniform(size=(n_auctions, n_rivals))
    noise = rng.uniform(size=(n_auctions, n_rivals))
    bids = strategy.bid(values, noise)
    return bids.max(axis=1)


def simulate_bid_sweep(
    value: float,
    strategy: RivalStrategy,
    n_rivals: int,
    rng: np.random.Generator,
    bids: np.ndarray = BID_GRID,
    n_auctions: int = N_AUCTIONS,
) -> BidSweep:
    """Estimate bidder 1's expected utility for every bid on the grid.

    The same block of rival draws is reused for every bid (common random
    numbers). Each individual estimate stays unbiased, while the *differences*
    between neighbouring bids become far less noisy -- so the shape of the curve,
    which is what the plot is for, is estimated much more precisely than
    independent sampling would allow at the same cost.
    """
    highest_rival = draw_highest_rival_bid(strategy, n_rivals, n_auctions, rng)

    mean_utility = np.empty(bids.size)
    std_error = np.empty(bids.size)
    win_rate = np.empty(bids.size)

    for i, bid in enumerate(bids):
        won = bid > highest_rival
        utility = np.where(won, value - highest_rival, 0.0)
        mean_utility[i] = utility.mean()
        std_error[i] = utility.std(ddof=1) / np.sqrt(n_auctions)
        win_rate[i] = won.mean()

    return BidSweep(
        value=float(value),
        strategy_key=strategy.key,
        n_rivals=n_rivals,
        bids=bids,
        mean_utility=mean_utility,
        std_error=std_error,
        win_rate=win_rate,
        n_auctions=n_auctions,
    )


def expected_utility_exact(
    value: float,
    strategy: RivalStrategy,
    n_rivals: int,
    bids: np.ndarray = BID_GRID,
) -> np.ndarray:
    """Exact expected utility, for validating and overlaying on the simulation.

    Each rival bids uniformly on ``[0, s]`` with ``s = strategy.support_max``, so
    the highest of ``k`` rival bids has CDF ``(m / s)^k`` and

        E[u(b)] = integral_0^t (v - m) d[(m / s)^k] = (t / s)^k (v - k t / (k+1))

    with ``t = min(b, s)``.
    """
    s = strategy.support_max
    k = n_rivals
    t = np.minimum(np.asarray(bids, dtype=float), s)
    return (t / s) ** k * (value - k * t / (k + 1))


def fixed_rival_utility(
    value: float,
    rival_bid: float,
    bids: np.ndarray = BID_GRID,
) -> np.ndarray:
    """Bidder 1's utility against a rival whose bid is pinned at ``rival_bid``.

    With the rival bid held fixed there is nothing random left, so this is exact:
    bidder 1 wins iff ``b > rival_bid`` and then banks ``value - rival_bid``.
    """
    bids = np.asarray(bids, dtype=float)
    return np.where(bids > rival_bid, value - rival_bid, 0.0)


def argmax_set(bids: np.ndarray, utility: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Every bid that attains the maximum utility (within ``tol``).

    Returning the whole set rather than one winner matters here: against a
    *fixed* rival bid, truthful bidding is only ever *weakly* optimal. If the
    rival bids 0.1 and ``v = 0.5``, then 0.2, 0.5 and 1.0 all win and all pay
    0.1, so all three are equally good. Reporting a single argmax would hide
    that, and would make the mechanism look more sharply truthful than it is.
    """
    bids = np.asarray(bids, dtype=float)
    return bids[utility >= utility.max() - tol]


def dominance_profile(
    value: float,
    bids: np.ndarray = BID_GRID,
    rival_bids: np.ndarray | None = None,
    tol: float = 1e-12,
) -> np.ndarray:
    """Fraction of possible rival bids against which each bid is optimal.

    This is the honest version of "truthful bidding is best". Against any *one*
    fixed rival bid, many bids tie for optimal. But sweeping the rival's bid
    across its whole range, ``b = v`` is the *unique* bid that is optimal against
    every one of them -- which is exactly what weak dominance means.
    """
    if rival_bids is None:
        rival_bids = np.linspace(0.0, 1.0, 201)
    bids = np.asarray(bids, dtype=float)

    optimal_count = np.zeros(bids.size)
    for rival_bid in rival_bids:
        utility = fixed_rival_utility(value, float(rival_bid), bids)
        optimal_count += utility >= utility.max() - tol
    return optimal_count / rival_bids.size
