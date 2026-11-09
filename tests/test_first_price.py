"""Part 2 -- first-price auction.

The headline assertions are that the optimum sits at k = (n-1)/n rather than at
k = 1, and that truthful bidding earns exactly zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from mechanism_design.config import K_GRID, N_BIDDERS, SEED
from mechanism_design.first_price import (
    equilibrium_shading,
    expected_utility_exact,
    optimal_utility_exact,
    simulate_shading_sweep,
)


@pytest.mark.parametrize(("n_bidders", "expected"), [(2, 0.5), (3, 2 / 3), (4, 0.75)])
def test_equilibrium_shading_factor(n_bidders, expected):
    assert equilibrium_shading(n_bidders) == pytest.approx(expected)


@pytest.mark.parametrize("n_bidders", N_BIDDERS)
def test_truthful_bidding_earns_exactly_zero(n_bidders):
    """k = 1 means bidding your value, and in a first-price auction that is
    self-defeating: you only win by surrendering your entire surplus."""
    exact = expected_utility_exact(n_bidders, np.array([1.0]))
    assert exact[0] == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("n_bidders", N_BIDDERS)
def test_optimum_is_at_the_equilibrium_not_at_truthful(n_bidders):
    """The curve peaks at (n-1)/n. This is the central result of part 2."""
    exact = expected_utility_exact(n_bidders, K_GRID)
    best_k = K_GRID[int(np.argmax(exact))]

    assert best_k == pytest.approx(equilibrium_shading(n_bidders))
    assert best_k < 1.0


@pytest.mark.parametrize("n_bidders", N_BIDDERS)
def test_peak_utility_equals_closed_form(n_bidders):
    """At k = (n-1)/n the expected utility is exactly 1/(n(n+1))."""
    c = equilibrium_shading(n_bidders)
    exact = expected_utility_exact(n_bidders, np.array([c]))

    assert exact[0] == pytest.approx(optimal_utility_exact(n_bidders))
    assert optimal_utility_exact(n_bidders) == pytest.approx(
        1.0 / (n_bidders * (n_bidders + 1))
    )


@pytest.mark.parametrize("n_bidders", N_BIDDERS)
def test_bidding_above_value_loses_money(n_bidders):
    """Any k > 1 pays more than the item is worth whenever it wins."""
    k_over = K_GRID[K_GRID > 1.0]
    exact = expected_utility_exact(n_bidders, k_over)
    assert np.all(exact < 0.0)


@pytest.mark.parametrize("n_bidders", N_BIDDERS)
def test_simulation_matches_closed_form(n_bidders):
    rng = np.random.default_rng(SEED)
    sweep = simulate_shading_sweep(n_bidders, rng, n_auctions=100_000)
    exact = expected_utility_exact(n_bidders, K_GRID)

    tolerance = 5 * sweep.std_error + 1e-9
    assert np.all(np.abs(sweep.mean_utility - exact) <= tolerance)
    assert sweep.best_k == pytest.approx(sweep.equilibrium_k)


@pytest.mark.parametrize("n_bidders", N_BIDDERS)
def test_winner_could_essentially_always_have_paid_less(n_bidders):
    """The brief's overpayment question.

    Bids are drawn from continuous distributions, so they are almost surely
    distinct and the winner almost surely strictly outbid the runner-up. The
    fraction is therefore ~1 by construction and carries little information;
    the mean gap is the statistic that actually says something, so it is
    asserted to be materially positive.
    """
    rng = np.random.default_rng(SEED)
    sweep = simulate_shading_sweep(n_bidders, rng, n_auctions=100_000)

    assert np.all(sweep.could_have_paid_less > 0.999)
    assert np.all(sweep.mean_overpayment > 0.0)

    at_equilibrium = int(np.argmin(np.abs(K_GRID - sweep.equilibrium_k)))
    assert sweep.mean_overpayment[at_equilibrium] > 0.05


@pytest.mark.parametrize("n_bidders", N_BIDDERS)
def test_deviating_breaks_allocative_efficiency(n_bidders):
    """Bidding on a different scale from everyone else sometimes hands the item
    to a bidder who did not value it most. At k = (n-1)/n all bidders use the
    same scale, so the allocation is efficient."""
    rng = np.random.default_rng(SEED)
    sweep = simulate_shading_sweep(n_bidders, rng, n_auctions=50_000)

    at_equilibrium = int(np.argmin(np.abs(K_GRID - sweep.equilibrium_k)))
    assert sweep.efficient_allocation[at_equilibrium] == pytest.approx(1.0)
    assert sweep.efficient_allocation.min() < 1.0
