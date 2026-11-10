"""Reproducible experiment pipeline.

``python -m mechanism_design.cli all`` regenerates every figure and table in
``docs/`` from scratch. Each part draws from its own independent child stream of
the master seed, so parts can be run individually and still produce exactly the
results they produce in a full run.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import plotting
from .config import (
    BID_GRID,
    BIDDER1_VALUES,
    FIXED_RIVAL_BIDS,
    K_GRID,
    N_AUCTIONS,
    N_BIDDERS,
    RESULTS_DIR,
    ROOT,
    SEED,
    TABLE_DIR,
    WEB_DIR,
)
from .first_price import (
    equilibrium_shading,
    optimal_utility_exact,
    simulate_shading_sweep,
)
from .first_price import (
    expected_utility_exact as fp_exact,
)
from .second_price import (
    argmax_set,
    dominance_profile,
    fixed_rival_utility,
    simulate_bid_sweep,
)
from .second_price import (
    expected_utility_exact as sp_exact,
)
from .strategies import STRATEGIES
from .vcg_path import FIGURE_7_3_COSTS, misreport_sweep, run_vcg, shortest_path

RIVAL_COUNTS = (1, 2)
MISREPORT_GRID = [float(x) for x in np.arange(0.0, 10.0 + 1e-9, 0.5)]


def _write_csv(name: str, rows: Sequence[dict[str, Any]]) -> Path:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / name
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _fmt(values: Iterable[float]) -> str:
    items = [f"{v:g}" for v in values]
    if len(items) <= 4:
        return "{" + ", ".join(items) + "}"
    return "{" + f"{items[0]} ... {items[-1]}" + f"}} ({len(items)} bids)"


def _streams() -> dict[str, np.random.Generator]:
    """Independent child streams, one per part, derived from the master seed."""
    part1a, part1c, part2 = np.random.SeedSequence(SEED).spawn(3)
    return {
        "part1_1rival": np.random.default_rng(part1a),
        "part1_2rivals": np.random.default_rng(part1c),
        "part2": np.random.default_rng(part2),
    }


def run_part1(streams: dict[str, np.random.Generator]) -> dict[str, Any]:
    print("\n=== Part 1 - second-price sealed-bid auction ===")

    web: dict[str, Any] = {"bids": [round(float(b), 2) for b in BID_GRID], "series": {}}
    sweep_rows: list[dict[str, Any]] = []
    for n_rivals in RIVAL_COUNTS:
        rng = streams["part1_1rival" if n_rivals == 1 else "part1_2rivals"]
        sweeps = {}
        for value in BIDDER1_VALUES:
            for strategy in STRATEGIES:
                sweep = simulate_bid_sweep(value, strategy, n_rivals, rng)
                sweeps[(value, strategy.key)] = sweep
                exact = sp_exact(value, strategy, n_rivals)
                web["series"][f"{n_rivals}|{value}|{strategy.key}"] = {
                    "sim": [round(float(u), 5) for u in sweep.mean_utility],
                    "exact": [round(float(u), 5) for u in exact],
                    "win": [round(float(w), 4) for w in sweep.win_rate],
                }
                for i, bid in enumerate(sweep.bids):
                    sweep_rows.append(
                        {
                            "n_rivals": n_rivals,
                            "value": value,
                            "rival_strategy": strategy.key,
                            "bid": f"{bid:.2f}",
                            "mean_utility": f"{sweep.mean_utility[i]:.6f}",
                            "std_error": f"{sweep.std_error[i]:.6f}",
                            "win_rate": f"{sweep.win_rate[i]:.6f}",
                            "exact_utility": f"{exact[i]:.6f}",
                        }
                    )

        figure = plotting.plot_second_price_grid(sweeps, n_rivals)
        label = "one rival" if n_rivals == 1 else "two rivals"
        print(f"  [A] {label}: 9 panels -> {figure}")
        for value in BIDDER1_VALUES:
            for strategy in STRATEGIES:
                sweep = sweeps[(value, strategy.key)]
                exact = sp_exact(value, strategy, n_rivals)
                err = float(np.abs(sweep.mean_utility - exact).max())
                best = argmax_set(BID_GRID, exact)
                print(
                    f"      v={value}  {strategy.key:<9} best bid {_fmt(best):<28}"
                    f" max|sim-exact| = {err:.5f}"
                )

    _write_csv("part1a_bid_sweeps.csv", sweep_rows)

    fixed_rows: list[dict[str, Any]] = []
    for n_rivals in RIVAL_COUNTS:
        for value in BIDDER1_VALUES:
            for rival_bid in FIXED_RIVAL_BIDS:
                utility = fixed_rival_utility(value, rival_bid)
                optimal = argmax_set(BID_GRID, utility)
                fixed_rows.append(
                    {
                        "n_rivals": n_rivals,
                        "value": value,
                        "rival_bid": rival_bid,
                        "utility_at_truthful_bid": f"{utility[np.isclose(BID_GRID, value)][0]:.4f}",
                        "max_utility": f"{utility.max():.4f}",
                        "truthful_is_optimal": bool(np.any(np.isclose(optimal, value))),
                        "n_optimal_bids": int(optimal.size),
                        "optimal_bid_range": f"[{optimal.min():.2f}, {optimal.max():.2f}]",
                    }
                )

    _write_csv("part1b_fixed_rival_bids.csv", fixed_rows)
    print(f"  [B] fixed rival bids: {plotting.plot_fixed_rival_bids()}")
    for row in fixed_rows:
        if row["n_rivals"] == 1:
            print(
                f"      v={row['value']}  rival bid {row['rival_bid']}: "
                f"b=v optimal = {row['truthful_is_optimal']}  "
                f"(tied with {row['n_optimal_bids'] - 1} other bids in "
                f"{row['optimal_bid_range']})"
            )
    print(
        "      Note: with two rivals both pinned at the same bid the highest rival bid is\n"
        "      unchanged, so part 1C's version of this check is identical by construction."
    )

    dominance_rows = [
        {
            "value": value,
            "bid": f"{bid:.2f}",
            "pct_rival_bids_where_optimal": f"{100 * profile[i]:.3f}",
        }
        for value in BIDDER1_VALUES
        for i, bid in enumerate(BID_GRID)
        for profile in [dominance_profile(value, rival_bids=np.linspace(0.0, 1.0, 401))]
    ]
    _write_csv("part1_dominance_profile.csv", dominance_rows)
    print(f"  [*] dominance profile: {plotting.plot_dominance_profile()}")
    for value in BIDDER1_VALUES:
        profile = dominance_profile(value, rival_bids=np.linspace(0.0, 1.0, 401))
        always = BID_GRID[np.isclose(profile, 1.0)]
        print(f"      v={value}: optimal against 100% of rival bids only at b = {_fmt(always)}")

    return web


def run_part2(streams: dict[str, np.random.Generator]) -> dict[str, Any]:
    print("\n=== Part 2 - first-price auction with a deviating bidder ===")
    rng = streams["part2"]

    web: dict[str, Any] = {"k": [round(float(k), 4) for k in K_GRID], "series": {}}
    sweeps = {}
    rows: list[dict[str, Any]] = []
    for n_bidders in N_BIDDERS:
        sweep = simulate_shading_sweep(n_bidders, rng)
        sweeps[n_bidders] = sweep
        exact = fp_exact(n_bidders, K_GRID)
        at_eq = int(np.argmin(np.abs(K_GRID - equilibrium_shading(n_bidders))))
        web["series"][str(n_bidders)] = {
            "sim": [round(float(u), 5) for u in sweep.mean_utility],
            "exact": [round(float(u), 5) for u in exact],
            "win": [round(float(w), 4) for w in sweep.win_rate],
            "overpay_pct": [round(float(o), 4) for o in sweep.mean_overpayment_pct],
            "equilibrium_k": round(equilibrium_shading(n_bidders), 4),
            "peak": round(optimal_utility_exact(n_bidders), 6),
            # Headline numbers for the results page, taken at the equilibrium so
            # the page can never quote a figure the pipeline did not produce.
            "best_k_simulated": round(float(sweep.best_k), 4),
            "could_pay_less_at_eq": round(float(sweep.could_have_paid_less[at_eq]), 6),
            "overpay_abs_at_eq": round(float(sweep.mean_overpayment[at_eq]), 6),
            "overpay_pct_at_eq": round(float(sweep.mean_overpayment_pct[at_eq]), 6),
        }
        for i, k in enumerate(sweep.k_grid):
            rows.append(
                {
                    "n_bidders": n_bidders,
                    "k": f"{k:.4f}",
                    "mean_utility": f"{sweep.mean_utility[i]:.6f}",
                    "std_error": f"{sweep.std_error[i]:.6f}",
                    "exact_utility": f"{exact[i]:.6f}",
                    "win_rate": f"{sweep.win_rate[i]:.6f}",
                    "could_have_paid_less": f"{sweep.could_have_paid_less[i]:.6f}",
                    "mean_overpayment": f"{sweep.mean_overpayment[i]:.6f}",
                    "mean_overpayment_pct_of_bid": f"{sweep.mean_overpayment_pct[i]:.6f}",
                    "efficient_allocation": f"{sweep.efficient_allocation[i]:.6f}",
                }
            )

    _write_csv("part2_shading_sweep.csv", rows)
    print(f"  expected utility vs k: {plotting.plot_first_price(sweeps)}")
    print(f"  overpayment:           {plotting.plot_overpayment(sweeps)}")

    summary: list[dict[str, Any]] = []
    for n_bidders, sweep in sorted(sweeps.items()):
        c = equilibrium_shading(n_bidders)
        at_eq = int(np.argmin(np.abs(K_GRID - c)))
        exact = fp_exact(n_bidders, K_GRID)
        truthful = float(fp_exact(n_bidders, np.array([1.0]))[0])
        summary.append(
            {
                "n_bidders": n_bidders,
                "equilibrium_k": f"{c:.4f}",
                "simulated_best_k": f"{sweep.best_k:.4f}",
                "exact_best_k": f"{K_GRID[int(np.argmax(exact))]:.4f}",
                "utility_at_equilibrium": f"{sweep.mean_utility[at_eq]:.6f}",
                "exact_utility_at_equilibrium": f"{optimal_utility_exact(n_bidders):.6f}",
                "utility_at_truthful_k1": f"{truthful:.6f}",
                "could_have_paid_less_at_eq": f"{sweep.could_have_paid_less[at_eq]:.6f}",
                "mean_overpayment_at_eq": f"{sweep.mean_overpayment[at_eq]:.6f}",
                "mean_overpayment_pct_at_eq": f"{sweep.mean_overpayment_pct[at_eq]:.6f}",
            }
        )
        print(
            f"  n={n_bidders}: optimal k = {sweep.best_k:.4f} (equilibrium {c:.4f}), "
            f"E[U] = {optimal_utility_exact(n_bidders):.6f} = 1/(n(n+1)); "
            f"truthful k=1 gives {truthful:.6f}"
        )
        print(
            f"       winner could have paid less in "
            f"{100 * sweep.could_have_paid_less[at_eq]:.2f}% of auctions, "
            f"overpaying by {sweep.mean_overpayment[at_eq]:.4f} on average "
            f"({100 * sweep.mean_overpayment_pct[at_eq]:.1f}% of the winning bid)"
        )

    _write_csv("part2_summary.csv", summary)
    return web


def run_part3() -> dict[str, Any]:
    print("\n=== Part 3 - VCG for the cheapest path from A to F ===")
    outcome = run_vcg(FIGURE_7_3_COSTS)
    print(f"  chosen path: {outcome.chosen.label} (cost {outcome.chosen.cost:g})")

    on_path = set(outcome.chosen.edges)
    payment_rows: list[dict[str, Any]] = []
    for edge in sorted(FIGURE_7_3_COSTS):
        used = edge in on_path
        without = shortest_path(FIGURE_7_3_COSTS, exclude=edge)
        payment = outcome.payments[edge]
        payment_rows.append(
            {
                "edge": f"{edge[0]}{edge[1]}",
                "true_cost": f"{FIGURE_7_3_COSTS[edge]:g}",
                "on_chosen_path": used,
                "shortest_path_without_edge": without.label,
                "cost_without_edge": f"{without.cost:g}",
                "payment": f"{payment:g}",
                "utility": f"{payment - FIGURE_7_3_COSTS[edge] if used else 0.0:g}",
            }
        )
        marker = " <- on path" if used else ""
        print(
            f"    {edge[0]}{edge[1]}: cost {FIGURE_7_3_COSTS[edge]:g}, paid {payment:g}, "
            f"utility {payment - FIGURE_7_3_COSTS[edge] if used else 0.0:g}{marker}"
        )

    _write_csv("part3_payments.csv", payment_rows)
    print(f"  network figure: {plotting.plot_vcg_network()}")

    sweeps: dict[str, list[dict[str, Any]]] = {}
    misreport_rows: list[dict[str, Any]] = []
    for edge, description in ((("A", "B"), "on the chosen path"), (("A", "C"), "off the path")):
        rows = misreport_sweep(edge, MISREPORT_GRID)
        label = (
            f"{edge[0]}{edge[1]} - {description} "
            f"(true cost {FIGURE_7_3_COSTS[edge]:g})"
        )
        sweeps[label] = rows
        for row in rows:
            misreport_rows.append({"edge": f"{edge[0]}{edge[1]}", **row})

        truthful = next(r for r in rows if r["reported_cost"] == FIGURE_7_3_COSTS[edge])
        best = max(r["utility"] for r in rows)
        print(
            f"    {edge[0]}{edge[1]} ({description}): truthful utility "
            f"{truthful['utility']:g}, best achievable by lying {best:g} "
            f"-> truthful is optimal: {truthful['utility'] >= best - 1e-12}"
        )

    _write_csv("part3_misreports.csv", misreport_rows)
    print(f"  misreport figure: {plotting.plot_vcg_misreports(sweeps)}")

    return {
        "edges": [
            {"u": edge[0], "v": edge[1], "cost": FIGURE_7_3_COSTS[edge]}
            for edge in sorted(FIGURE_7_3_COSTS)
        ],
        "nodes": {k: list(v) for k, v in plotting.NODE_POSITIONS.items()},
    }


def _read_csv(name: str) -> list[dict[str, str]]:
    with (TABLE_DIR / name).open() as handle:
        return list(csv.DictReader(handle))


def write_summary() -> Path:
    """Write ``results/summary.md`` from the CSVs the run just produced.

    Deliberately derived from the written files rather than from in-memory
    values: the summary then cannot disagree with the tables it summarises.
    """
    sweeps = _read_csv("part1a_bid_sweeps.csv")
    fixed = _read_csv("part1b_fixed_rival_bids.csv")
    shading = _read_csv("part2_summary.csv")
    payments = _read_csv("part3_payments.csv")
    misreports = _read_csv("part3_misreports.csv")

    out: list[str] = [
        "# Results summary",
        "",
        f"Generated by `python -m mechanism_design.cli all` with seed `{SEED}` and "
        f"{N_AUCTIONS:,} auctions per bid point. Regenerating reproduces this file exactly.",
        "",
        "## Part 1 — second-price sealed-bid auction",
        "",
        "Best bid and worst-case gap between the simulation and the closed form, for every "
        "combination of bidder 1's value, the rival strategy and the number of rivals.",
        "",
        "| rivals | value | rival strategy | best bid (exact) | max \|sim − exact\| |",
        "|---|---|---|---|---|",
    ]

    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in sweeps:
        groups.setdefault((row["n_rivals"], row["value"], row["rival_strategy"]), []).append(row)

    for (rivals, value, strategy), rows in sorted(groups.items()):
        exact = [float(r["exact_utility"]) for r in rows]
        best = max(exact)
        optimal = [r["bid"] for r, e in zip(rows, exact, strict=True) if e >= best - 1e-12]
        shown = (
            optimal[0]
            if len(optimal) == 1
            else f"{optimal[0]}–{optimal[-1]} ({len(optimal)} tie)"
        )
        err = max(abs(float(r["mean_utility"]) - float(r["exact_utility"])) for r in rows)
        out.append(f"| {rivals} | {value} | {strategy} | {shown} | {err:.5f} |")

    truthful_ok = all(r["truthful_is_optimal"] == "True" for r in fixed)
    ties = [int(r["n_optimal_bids"]) for r in fixed]
    out += [
        "",
        "The peak sits at `b = v` in every row. Where a range appears, the rival halves its "
        "value and so never bids above 0.5, which makes every bid above that equally good.",
        "",
        "### Against a fixed rival bid",
        "",
        f"Truthful bidding is optimal in all {len(fixed)} combinations: "
        f"**{truthful_ok}**. It is rarely the *only* optimum — between {min(ties)} and "
        f"{max(ties)} bids tie for best, which is why this is weak rather than strict "
        "dominance.",
        "",
        "## Part 2 — first-price auction",
        "",
        "| n | equilibrium k | simulated optimum | E[U] at optimum | exact 1/(n(n+1)) | "
        "E[U] at truthful k=1 | winner could pay less | mean overpayment |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in shading:
        out.append(
            f"| {row['n_bidders']} | {row['equilibrium_k']} | {row['simulated_best_k']} | "
            f"{row['utility_at_equilibrium']} | {row['exact_utility_at_equilibrium']} | "
            f"{row['utility_at_truthful_k1']} | "
            f"{100 * float(row['could_have_paid_less_at_eq']):.2f}% | "
            f"{row['mean_overpayment_at_eq']} "
            f"({100 * float(row['mean_overpayment_pct_at_eq']):.1f}% of bid) |"
        )

    out += [
        "",
        "The simulated optimum lands on the exact equilibrium `(n-1)/n` for every `n`, and "
        "truthful bidding earns exactly zero.",
        "",
        "## Part 3 — VCG for the cheapest path",
        "",
        "| edge | true cost | on path | best path without it | payment | profit |",
        "|---|---|---|---|---|---|",
    ]
    for row in payments:
        out.append(
            f"| {row['edge']} | {row['true_cost']} | "
            f"{'yes' if row['on_chosen_path'] == 'True' else 'no'} | "
            f"{row['shortest_path_without_edge']} ({row['cost_without_edge']}) | "
            f"{row['payment']} | {row['utility']} |"
        )

    total = sum(float(r["payment"]) for r in payments)
    by_edge: dict[str, list[dict[str, str]]] = {}
    for row in misreports:
        by_edge.setdefault(row["edge"], []).append(row)
    verdicts = []
    for edge, rows in by_edge.items():
        truth = next(r for r in rows if r["reported_cost"] == r["true_cost"])
        best = max(float(r["utility"]) for r in rows)
        verdicts.append(
            f"- **{edge}** (true cost {truth['true_cost']}): truthful profit "
            f"{truth['utility']}, best achievable by any misreport on the 0–10 grid {best:g} "
            f"— truthful is optimal: **{float(truth['utility']) >= best - 1e-12}**"
        )

    out += [
        "",
        f"Reproduces Figure 7.3 of Shoham & Leyton-Brown exactly. The mechanism pays out "
        f"{total:g} in total for a path that truly costs "
        f"{sum(float(r['true_cost']) for r in payments if r['on_chosen_path'] == 'True'):g} "
        "— the price of strategyproofness.",
        "",
        "### Misreporting sweeps",
        "",
        *verdicts,
        "",
        "The test suite runs this sweep for all eight edges, not just these two.",
        "",
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "summary.md"
    path.write_text("\n".join(out))
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mechanism-design",
        description="Reproduce every figure and table for the auction experiments.",
    )
    parser.add_argument(
        "part",
        nargs="?",
        default="all",
        choices=("all", "part1", "part2", "part3"),
        help="which experiment to run (default: all)",
    )
    args = parser.parse_args(argv)

    streams = _streams()
    print(f"seed = {SEED}, {N_AUCTIONS:,} auctions per bid point")

    web: dict[str, Any] = {}
    if args.part in ("all", "part1"):
        web["part1"] = run_part1(streams)
    if args.part in ("all", "part2"):
        web["part2"] = run_part2(streams)
    if args.part in ("all", "part3"):
        web["part3"] = run_part3()

    if args.part == "all":
        # Compact payload for the interactive page, so it plots the same numbers
        # the figures and tables were built from rather than its own re-run.
        data_path = WEB_DIR / "data.json"
        data_path.write_text(json.dumps({"seed": SEED, "n_auctions": N_AUCTIONS, **web}))
        size_kb = data_path.stat().st_size / 1024
        print(f"\nWeb data -> {data_path.relative_to(ROOT)} ({size_kb:.0f} KB)")
        print(f"Summary  -> {write_summary().relative_to(ROOT)}")

    print("Figures -> web/figures/   Tables -> results/tables/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
