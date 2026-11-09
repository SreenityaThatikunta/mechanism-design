"""Part 3 -- VCG for cheapest-path procurement.

These tests pin the implementation to the numbers printed in Shoham &
Leyton-Brown's worked example, then check the incentive property directly by
letting every edge in turn try every lie on the grid.
"""

from __future__ import annotations

import numpy as np
import pytest

from mechanism_design.vcg_path import (
    FIGURE_7_3_COSTS,
    edge_key,
    misreport_sweep,
    run_vcg,
    shortest_path,
)

MISREPORT_GRID = [float(x) for x in np.arange(0.0, 10.0 + 1e-9, 0.5)]


def test_chosen_path_is_abef():
    """The book: "x(v) will return the shortest path in the graph, which is ABEF"."""
    chosen = shortest_path(FIGURE_7_3_COSTS)
    assert chosen.label == "ABEF"
    assert chosen.cost == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("excluded", "expected_label", "expected_cost"),
    [
        (("A", "B"), "ACEF", 6.0),
        (("B", "E"), "ACEF", 6.0),
        (("E", "F"), "ABDF", 7.0),
    ],
)
def test_counterfactual_path_costs(excluded, expected_label, expected_cost):
    """The 5/6/7 figures the book's payment arithmetic is built on."""
    without = shortest_path(FIGURE_7_3_COSTS, exclude=excluded)
    assert without.label == expected_label
    assert without.cost == pytest.approx(expected_cost)


def test_payments_reproduce_the_textbook():
    """"AB is paid 4, BE is paid 2, EF is paid 3, every other edge 0"."""
    outcome = run_vcg(FIGURE_7_3_COSTS)

    assert outcome.payment("A", "B") == pytest.approx(4.0)
    assert outcome.payment("B", "E") == pytest.approx(2.0)
    assert outcome.payment("E", "F") == pytest.approx(3.0)

    on_path = {edge_key("A", "B"), edge_key("B", "E"), edge_key("E", "F")}
    for edge, payment in outcome.payments.items():
        if edge not in on_path:
            assert payment == pytest.approx(0.0), f"{edge} is not pivotal and must be paid 0"


def test_equal_cost_edges_are_paid_differently():
    """BE and EF both cost 1, yet EF is paid 3 and BE only 2.

    The book's explanation is market power: the network without EF (7) is worse
    than the network without BE (6), so EF's absence hurts more and it captures
    more of the surplus. Worth a test of its own -- it is the clearest sign that
    VCG prices marginal contribution, not cost.
    """
    outcome = run_vcg(FIGURE_7_3_COSTS)
    assert FIGURE_7_3_COSTS[edge_key("B", "E")] == FIGURE_7_3_COSTS[edge_key("E", "F")]
    assert outcome.payment("E", "F") > outcome.payment("B", "E")


def test_payment_does_not_depend_on_own_report():
    """The structural reason VCG is strategyproof.

    An edge's payment is (best path without it) - (cost of the rest of the
    chosen path); neither term involves its own report. So as long as AB stays
    on the path, its payment is pinned at 4 however it bids.
    """
    payments = [
        row["payment"]
        for row in misreport_sweep(("A", "B"), MISREPORT_GRID)
        if row["on_chosen_path"]
    ]
    assert len(payments) > 1, "expected AB to stay on the path for a range of reports"
    assert payments == pytest.approx([4.0] * len(payments))


def test_on_path_edge_cannot_gain_by_misreporting():
    """AB, true cost 3. Under-reporting changes nothing; over-reporting past its
    pivotal threshold loses it the contract and all of its surplus."""
    rows = misreport_sweep(("A", "B"), MISREPORT_GRID)
    utilities = {row["reported_cost"]: row["utility"] for row in rows}

    assert utilities[3.0] == pytest.approx(1.0)
    assert utilities[3.0] >= max(utilities.values()) - 1e-12
    assert utilities[10.0] == pytest.approx(0.0)


def test_off_path_edge_can_only_win_at_a_loss():
    """AC, true cost 2. It can buy its way onto the path by reporting below 1,
    but is then paid only 1 against a true cost of 2 -- a utility of -1."""
    rows = misreport_sweep(("A", "C"), MISREPORT_GRID)
    utilities = {row["reported_cost"]: row["utility"] for row in rows}

    assert utilities[2.0] == pytest.approx(0.0)
    assert utilities[0.0] == pytest.approx(-1.0)
    assert utilities[2.0] >= max(utilities.values()) - 1e-12


@pytest.mark.parametrize("edge", sorted(FIGURE_7_3_COSTS), ids=lambda e: f"{e[0]}{e[1]}")
def test_truthful_reporting_is_optimal_for_every_edge(edge):
    """The full dominant-strategy claim: no edge, on or off the path, can do
    better than reporting its true cost against any lie on the grid."""
    rows = misreport_sweep(edge, MISREPORT_GRID)
    true_cost = FIGURE_7_3_COSTS[edge_key(*edge)]

    truthful = next(r["utility"] for r in rows if r["reported_cost"] == pytest.approx(true_cost))
    assert truthful >= max(r["utility"] for r in rows) - 1e-12


def test_individual_rationality_under_truthful_reporting():
    """Nobody is made worse off by taking part: every edge's utility is >= 0."""
    outcome = run_vcg(FIGURE_7_3_COSTS)
    on_path = set(outcome.chosen.edges)
    for edge, payment in outcome.payments.items():
        utility = payment - FIGURE_7_3_COSTS[edge] if edge in on_path else 0.0
        assert utility >= -1e-12


def test_mechanism_runs_at_a_loss_relative_to_true_cost():
    """VCG is not budget balanced here: it pays out 9 for a path that costs 5.

    Worth asserting because it is the price of strategyproofness, and the reason
    VCG is not a free lunch in procurement settings.
    """
    outcome = run_vcg(FIGURE_7_3_COSTS)
    total_paid = sum(outcome.payments.values())
    assert total_paid == pytest.approx(9.0)
    assert total_paid > outcome.chosen.cost


def test_disconnecting_the_network_is_handled():
    """If an edge is indispensable, the counterfactual path is infinite."""
    bridge_only = {edge_key("A", "B"): 1.0, edge_key("B", "F"): 1.0}
    assert shortest_path(bridge_only).label == "ABF"
    assert shortest_path(bridge_only, exclude=("A", "B")).cost == float("inf")
