"""Part 1 -- second-price auction.

The point of these tests is that the Monte Carlo is checked against *exact
theory*, not against another simulation or a stored snapshot.
"""

from __future__ import annotations

import numpy as np
import pytest

from mechanism_design.config import BID_GRID, BIDDER1_VALUES, FIXED_RIVAL_BIDS, SEED
from mechanism_design.second_price import (
    argmax_set,
    dominance_profile,
    draw_highest_rival_bid,
    expected_utility_exact,
    fixed_rival_utility,
    simulate_bid_sweep,
)
from mechanism_design.strategies import HALF, RANDOM, STRATEGIES, TRUTHFUL

RIVAL_COUNTS = (1, 2)
COMBINATIONS = [(v, s, k) for v in BIDDER1_VALUES for s in STRATEGIES for k in RIVAL_COUNTS]


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.key)
def test_rival_bids_are_uniform_on_their_support(strategy):
    """The closed forms assume each rival's bid is uniform on [0, support_max].

    That assumption is load-bearing for every analytic result in this module, so
    it is verified rather than asserted: compare the empirical CDF against the
    strategy's declared CDF at a grid of points.
    """
    rng = np.random.default_rng(SEED)
    bids = draw_highest_rival_bid(strategy, n_rivals=1, n_auctions=200_000, rng=rng)

    grid = np.linspace(0.0, 1.0, 51)
    empirical = np.array([(bids <= x).mean() for x in grid])
    assert np.allclose(empirical, strategy.bid_cdf(grid), atol=5e-3)
    assert bids.max() <= strategy.support_max + 1e-12


def test_truthful_and_random_rivals_are_indistinguishable():
    """Strategies (a) and (c) induce the same bid distribution, U[0, 1].

    So bidder 1 cannot tell them apart and its utility curve must coincide --
    which is why two of the nine panels in part 1A are identical.
    """
    grid = np.linspace(0.0, 1.0, 51)
    assert np.allclose(TRUTHFUL.bid_cdf(grid), RANDOM.bid_cdf(grid))

    for value in BIDDER1_VALUES:
        for k in RIVAL_COUNTS:
            assert np.allclose(
                expected_utility_exact(value, TRUTHFUL, k),
                expected_utility_exact(value, RANDOM, k),
            )


@pytest.mark.parametrize(("value", "strategy", "n_rivals"), COMBINATIONS)
def test_simulation_matches_closed_form(value, strategy, n_rivals):
    """Simulated expected utility agrees with the exact formula everywhere."""
    rng = np.random.default_rng(SEED)
    sweep = simulate_bid_sweep(value, strategy, n_rivals, rng, n_auctions=100_000)
    exact = expected_utility_exact(value, strategy, n_rivals)

    # Allow five standard errors, plus a floor for the bids where the estimator
    # has (correctly) zero variance because bidder 1 never wins.
    tolerance = 5 * sweep.std_error + 1e-9
    assert np.all(np.abs(sweep.mean_utility - exact) <= tolerance)


@pytest.mark.parametrize(("value", "strategy", "n_rivals"), COMBINATIONS)
def test_truthful_bid_maximises_expected_utility(value, strategy, n_rivals):
    """b = v is optimal against every rival strategy and both rival counts.

    For the half-value rival the optimum is a plateau rather than a spike: once
    b exceeds 0.5 bidder 1 already wins every auction, so every b in [0.5, 1] is
    equally good. b = v must be *in* the argmax, not the unique element of it.
    """
    exact = expected_utility_exact(value, strategy, n_rivals)
    optimal_bids = argmax_set(BID_GRID, exact, tol=1e-12)

    assert np.any(np.isclose(optimal_bids, value))
    assert exact[np.isclose(BID_GRID, value)][0] == pytest.approx(exact.max())


@pytest.mark.parametrize("value", BIDDER1_VALUES)
@pytest.mark.parametrize("n_rivals", RIVAL_COUNTS)
def test_overbidding_and_underbidding_never_help(value, n_rivals):
    """No bid beats b = v, against any of the three strategies."""
    for strategy in STRATEGIES:
        exact = expected_utility_exact(value, strategy, n_rivals)
        at_value = exact[np.isclose(BID_GRID, value)][0]
        assert at_value >= exact.max() - 1e-12


@pytest.mark.parametrize("value", BIDDER1_VALUES)
@pytest.mark.parametrize("rival_bid", FIXED_RIVAL_BIDS)
def test_truthful_is_weakly_optimal_against_a_fixed_rival_bid(value, rival_bid):
    """Part 1B. Against a pinned rival bid, b = v is optimal but rarely unique.

    With rival bid 0.1 and v = 0.5, bids 0.2, 0.5 and 1.0 all win and all pay
    0.1. The test therefore asserts membership in the argmax, and separately
    asserts that the tie is real -- documenting *weak* dominance rather than
    overstating the result.
    """
    utility = fixed_rival_utility(value, rival_bid)
    optimal_bids = argmax_set(BID_GRID, utility)

    assert np.any(np.isclose(optimal_bids, value))
    assert optimal_bids.size > 1, "expected a tie: many bids are optimal against a fixed bid"


@pytest.mark.parametrize("value", BIDDER1_VALUES)
def test_truthful_is_the_unique_bid_optimal_against_every_rival_bid(value):
    """The honest statement of weak dominance.

    Sweeping the rival's bid across [0, 1], b = v is the *only* bid that is
    optimal against all of them: any b < v forfeits auctions worth winning when
    the rival bids between b and v, and any b > v wins auctions at a loss when
    the rival bids between v and b.
    """
    profile = dominance_profile(value, rival_bids=np.linspace(0.0, 1.0, 401))
    always_optimal = BID_GRID[np.isclose(profile, 1.0)]

    assert always_optimal.size == 1
    assert always_optimal[0] == pytest.approx(value)


def test_half_strategy_caps_the_price_bidder_one_pays():
    """A rival that halves its value can never bid above 0.5.

    So for v > 0.5 bidder 1 wins every auction once b >= 0.5 and its utility
    flattens at v - E[highest rival bid].
    """
    exact = expected_utility_exact(0.8, HALF, n_rivals=1)
    plateau = exact[BID_GRID >= 0.5]
    assert np.allclose(plateau, plateau[0])
    assert plateau[0] == pytest.approx(0.8 - 0.25)  # E[U[0, 0.5]] = 0.25
