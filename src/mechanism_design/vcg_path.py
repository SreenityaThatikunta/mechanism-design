"""Part 3 -- VCG for buying the cheapest path from A to F.

This is the transportation network of Figure 7.3 in Shoham & Leyton-Brown,
*Multiagent Systems* (Ch. 7, sec. 4.2). Every **edge** is an agent holding a
private cost, and the mechanism procures a path from A to F. An edge's utility is
``payment - true_cost`` when its edge is used, and zero otherwise.

This is a reverse (procurement) auction, so the efficient choice minimises total
reported cost: ``x`` returns the shortest path. The VCG payment to an edge ``e``
on the chosen path is

    p_e = L(without e) - (L(chosen) - reported_cost_e)

that is, the cost of the best path available if ``e`` did not exist, minus the
cost of the *rest* of the path actually chosen. Note that ``L(chosen) -
reported_cost_e`` is just the cost of the other edges on the path, so ``p_e``
does not depend on ``e``'s own report -- which is precisely why misreporting
cannot raise an edge's payment. Edges off the chosen path are not pivotal:
removing them changes nothing, so the two terms cancel and they are paid zero.

Expected output, matching the book exactly: the chosen path is A-B-E-F, AB is
paid 4, BE is paid 2, EF is paid 3, and every other edge is paid 0. EF is paid
more than BE despite having the same cost, because EF has more market power --
the network without EF is worse than the network without BE.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

Edge = tuple[str, str]

#: Edge costs read off Figure 7.3. Verified against every number quoted in the
#: book's worked example: shortest path ABEF = 5, ACEF = 6, ABDF = ACF = 7.
FIGURE_7_3_COSTS: dict[Edge, float] = {
    ("A", "B"): 3.0,
    ("A", "C"): 2.0,
    ("B", "D"): 2.0,
    ("B", "E"): 1.0,
    ("C", "E"): 3.0,
    ("C", "F"): 5.0,
    ("D", "F"): 2.0,
    ("E", "F"): 1.0,
}

SOURCE = "A"
SINK = "F"


def edge_key(u: str, v: str) -> Edge:
    """Canonical (sorted) key, so edges are undirected and order-insensitive."""
    return (u, v) if u <= v else (v, u)


@dataclass(frozen=True)
class PathResult:
    """A shortest path: its total cost and the edges it uses."""

    cost: float
    nodes: tuple[str, ...]
    edges: tuple[Edge, ...]

    @property
    def label(self) -> str:
        return "".join(self.nodes) if self.nodes else "(no path)"


NO_PATH = PathResult(cost=math.inf, nodes=(), edges=())


def shortest_path(
    costs: dict[Edge, float],
    source: str = SOURCE,
    sink: str = SINK,
    exclude: Edge | None = None,
) -> PathResult:
    """Cheapest ``source``-to-``sink`` path, optionally with one edge removed.

    Ties are broken by the lexicographically smallest node sequence. The graph
    here has genuine ties (ABDF and ACF both cost 7), so leaving tie-breaking to
    heap ordering would make results depend on insertion order. Pushing the path
    itself into the heap key makes the choice explicit and deterministic.
    """
    adjacency: dict[str, list[tuple[str, float]]] = {}
    for (u, v), cost in costs.items():
        if exclude is not None and edge_key(u, v) == edge_key(*exclude):
            continue
        adjacency.setdefault(u, []).append((v, cost))
        adjacency.setdefault(v, []).append((u, cost))

    heap: list[tuple[float, tuple[str, ...]]] = [(0.0, (source,))]
    settled: set[str] = set()

    while heap:
        cost, nodes = heapq.heappop(heap)
        node = nodes[-1]
        if node in settled:
            continue
        if node == sink:
            edges = tuple(edge_key(a, b) for a, b in zip(nodes, nodes[1:]))
            return PathResult(cost=cost, nodes=nodes, edges=edges)
        settled.add(node)
        for neighbour, step in adjacency.get(node, []):
            if neighbour not in settled:
                heapq.heappush(heap, (cost + step, (*nodes, neighbour)))

    return NO_PATH


@dataclass(frozen=True)
class VCGOutcome:
    """The chosen path and the payment made to every edge."""

    chosen: PathResult
    payments: dict[Edge, float]

    def payment(self, u: str, v: str) -> float:
        return self.payments[edge_key(u, v)]


def run_vcg(reported_costs: dict[Edge, float]) -> VCGOutcome:
    """Pick the cheapest reported path and compute each edge's VCG payment.

    Payments are positive numbers flowing *to* the edges. The book writes them as
    negative charges (``p_AB = -4``) to keep one sign convention across all of
    its mechanisms; "AB is paid 4" is the same statement.
    """
    chosen = shortest_path(reported_costs)
    if chosen is NO_PATH or not chosen.edges:
        return VCGOutcome(chosen=chosen, payments=dict.fromkeys(reported_costs, 0.0))

    on_path = set(chosen.edges)
    payments: dict[Edge, float] = {}

    for raw_edge in reported_costs:
        key = edge_key(*raw_edge)
        if key not in on_path:
            # Not pivotal: the network without it yields the same path, so the
            # two terms of the VCG payment cancel exactly.
            payments[key] = 0.0
            continue
        without = shortest_path(reported_costs, exclude=key)
        rest_of_path = chosen.cost - reported_costs[raw_edge]
        payments[key] = without.cost - rest_of_path

    return VCGOutcome(chosen=chosen, payments=payments)


def edge_utility(
    edge: Edge,
    true_cost: float,
    reported_costs: dict[Edge, float],
) -> tuple[float, float, bool]:
    """Utility of one edge under a given profile of reports.

    Returns ``(utility, payment, was_used)``. An unused edge pays nothing and is
    paid nothing, so its utility is zero.
    """
    key = edge_key(*edge)
    outcome = run_vcg(reported_costs)
    used = key in set(outcome.chosen.edges)
    payment = outcome.payments[key]
    utility = payment - true_cost if used else 0.0
    return utility, payment, used


def misreport_sweep(
    edge: Edge,
    reports: list[float],
    costs: dict[Edge, float] | None = None,
) -> list[dict[str, float | bool | str]]:
    """Sweep one edge's reported cost while every other edge reports truthfully.

    This is the strategyproofness check: the edge's true cost stays fixed, only
    its *report* moves, and we watch what happens to its utility.
    """
    costs = dict(FIGURE_7_3_COSTS if costs is None else costs)
    key = edge_key(*edge)
    true_cost = costs[key]

    rows: list[dict[str, float | bool | str]] = []
    for report in reports:
        profile = dict(costs)
        profile[key] = report
        outcome = run_vcg(profile)
        used = key in set(outcome.chosen.edges)
        payment = outcome.payments[key]
        rows.append(
            {
                "reported_cost": float(report),
                "true_cost": true_cost,
                "on_chosen_path": used,
                "payment": payment if used else 0.0,
                "utility": (payment - true_cost) if used else 0.0,
                "chosen_path": outcome.chosen.label,
                "path_cost_true": sum(costs[e] for e in outcome.chosen.edges),
            }
        )
    return rows
