"""Figure rendering.

Every panel shows the Monte Carlo result *and* the closed-form curve on the same
axes. That is the point of the plots: agreement between a 200,000-auction
simulation and a formula derived by hand is the evidence that both are right.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .config import BID_GRID, BIDDER1_VALUES, FIGURE_DIR, FIXED_RIVAL_BIDS, K_GRID
from .first_price import ShadingSweep, equilibrium_shading, expected_utility_exact
from .second_price import BidSweep, dominance_profile, fixed_rival_utility
from .second_price import expected_utility_exact as sp_exact
from .strategies import STRATEGIES
from .vcg_path import FIGURE_7_3_COSTS, run_vcg

# Render without a display, so figures build identically on CI and locally.
matplotlib.use("Agg")

INK = "#1b1f23"
SUBTLE = "#54606e"
ACCENT = "#2a6f97"
HIGHLIGHT = "#d1495b"
MUTED = "#9aa5b1"
GREEN = "#2a9d8f"
GRID = "#e3e8ee"
SERIES = (ACCENT, GREEN, HIGHLIGHT)

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#c7d0da",
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "font.size": 10,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 160,
}


def _save(fig: plt.Figure, name: str) -> str:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return str(path.relative_to(FIGURE_DIR.parents[1]))


def _style_axes(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _header(
    fig: plt.Figure,
    title: str,
    caption: str,
    *,
    reserve: float = 0.17,
    bottom: float = 0.0,
) -> None:
    """Lay out the axes, then place a title and caption in reserved space above.

    Reserving the space *before* writing into it is what stops suptitles and
    captions from landing on top of each other, which is easy to do when the
    caption wraps to more than one line. ``bottom`` reserves matching space for
    a figure-level legend underneath the axes.
    """
    fig.tight_layout(rect=(0, bottom, 1, 1 - reserve))
    fig.text(0.5, 1 - reserve * 0.26, title, ha="center", va="center", fontsize=13.5,
             fontweight="bold")
    fig.text(0.5, 1 - reserve * 0.68, caption, ha="center", va="center", fontsize=9.5,
             color=SUBTLE, linespacing=1.6)


def _footer(fig: plt.Figure, caption: str, *, reserve: float = 0.16) -> None:
    fig.tight_layout(rect=(reserve * 0.05, reserve, 1, 1))
    fig.text(0.5, reserve * 0.45, caption, ha="center", va="center", fontsize=9.5,
             color=SUBTLE, linespacing=1.6)


def _mark_vline(ax: plt.Axes, x: float, label: str, colour: str, *, y: float = 0.95) -> None:
    """Draw a reference line and label it inside the axes, flipping the label to
    the other side of the line when it would otherwise run off the right edge."""
    ax.axvline(x, color=colour, lw=1.4, ls=":")
    right_edge = x > sum(ax.get_xlim()) / 2
    ax.annotate(
        label,
        xy=(x, y),
        xycoords=("data", "axes fraction"),
        xytext=(-6 if right_edge else 6, 0),
        textcoords="offset points",
        ha="right" if right_edge else "left",
        va="top",
        color=colour,
        fontsize=9,
        fontweight="bold",
    )


def _headroom(ax: plt.Axes, values: np.ndarray, *, factor: float = 1.35) -> None:
    """Leave room above the data for in-axes labels."""
    low, high = float(np.min(values)), float(np.max(values))
    span = max(high - low, 1e-9)
    ax.set_ylim(low - 0.08 * span, low + span * factor)


def plot_second_price_grid(sweeps: dict[tuple[float, str], BidSweep], n_rivals: int) -> str:
    """The 3x3 panel the brief asks for: one panel per (value, rival strategy)."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 3, figsize=(13, 10.2), sharex=True)

        for row, value in enumerate(BIDDER1_VALUES):
            for col, strategy in enumerate(STRATEGIES):
                ax = axes[row][col]
                _style_axes(ax)
                sweep = sweeps[(value, strategy.key)]
                exact = sp_exact(value, strategy, n_rivals)

                ax.axhline(0.0, color=MUTED, lw=0.9)
                ax.plot(sweep.bids, sweep.mean_utility, "o", ms=2.8, color=ACCENT, alpha=0.85,
                        label=f"simulated ({sweep.n_auctions:,} auctions)")
                ax.plot(BID_GRID, exact, "--", lw=1.6, color=INK, label="closed form")
                _headroom(ax, np.concatenate([exact, sweep.mean_utility]), factor=1.3)
                _mark_vline(ax, value, f"b = v = {value}", HIGHLIGHT)

                ax.text(0.03, 0.95, f"v = {value}", transform=ax.transAxes, va="top",
                        fontsize=9.5, fontweight="bold", color=SUBTLE,
                        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#f4f7fa",
                              "edgecolor": GRID})

                if row == 0:
                    ax.set_title(strategy.label, fontsize=11, pad=10)
                if col == 0:
                    ax.set_ylabel("expected utility")
                if row == 2:
                    ax.set_xlabel("bidder 1's bid, b")

        handles, labels = axes[0][0].get_legend_handles_labels()
        rivals = "one rival" if n_rivals == 1 else f"{n_rivals} rivals"
        _header(
            fig,
            f"Second-price auction: bidder 1's expected utility vs its own bid ({rivals})",
            "The maximum sits at b = v in all nine panels. Changing the rival's strategy moves "
            "the height of the curve, never the location of its peak --\nwhich is exactly why "
            "bidder 1 does not need to know what the rival is doing.",
            reserve=0.13,
            bottom=0.055,
        )
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9.5,
                   bbox_to_anchor=(0.5, 0.004))
    return _save(fig, f"part1a_second_price_{n_rivals}rival.png")


def plot_fixed_rival_bids() -> str:
    """Part 1B: rival bid pinned at 0.1 / 0.4 / 0.9."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)

        for ax, value in zip(axes, BIDDER1_VALUES, strict=True):
            _style_axes(ax)
            ax.axhline(0.0, color=MUTED, lw=0.9)
            for rival_bid, colour in zip(FIXED_RIVAL_BIDS, SERIES, strict=True):
                ax.step(BID_GRID, fixed_rival_utility(value, rival_bid), where="post",
                        lw=1.9, color=colour, label=f"rival bids {rival_bid}")
            ax.set_ylim(-0.85, 0.92)
            _mark_vline(ax, value, f"b = v = {value}", INK)
            ax.set_title(f"bidder 1's value v = {value}")
            ax.set_xlabel("bidder 1's bid, b")

        axes[0].set_ylabel("utility")
        handles, labels = axes[0].get_legend_handles_labels()
        _header(
            fig,
            "Second-price auction with the rival's bid held fixed",
            "b = v always lands on the best step -- but so do many other bids. Against a single "
            "fixed rival bid truthfulness is optimal, not uniquely optimal.",
            reserve=0.2,
            bottom=0.1,
        )
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9.5,
                   bbox_to_anchor=(0.5, 0.006))
    return _save(fig, "part1b_fixed_rival_bids.png")


def plot_dominance_profile() -> str:
    """Why b = v is special once you stop fixing the rival's bid."""
    rival_bids = np.linspace(0.0, 1.0, 401)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8.8, 4.6))
        _style_axes(ax)
        for value, colour in zip(BIDDER1_VALUES, SERIES, strict=True):
            profile = dominance_profile(value, rival_bids=rival_bids)
            ax.plot(BID_GRID, 100 * profile, lw=2.0, color=colour, label=f"v = {value}")
            ax.plot([value], [100], "o", ms=7, color=colour, mec="white", mew=1.4)

        ax.set_ylim(0, 112)
        ax.set_xlabel("bidder 1's bid, b")
        ax.set_ylabel("% of rival bids for which b is optimal")
        ax.legend(fontsize=9.5, loc="lower center", ncol=3)
        ax.set_title(
            "Truthful bidding is the only bid that is optimal against every rival bid",
            fontsize=12.5, pad=12,
        )
        _footer(
            fig,
            "Each curve reaches exactly 100% at b = v and nowhere else. Bid below your value and "
            "you forfeit auctions worth winning;\nbid above it and you win auctions at a loss. "
            "This is what 'weakly dominant' means, stated precisely.",
            reserve=0.21,
        )
    return _save(fig, "part1_dominance_profile.png")


def plot_first_price(sweeps: dict[int, ShadingSweep]) -> str:
    """Part 2: expected utility against the shading factor k, for n = 2, 3, 4."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharey=True)

        for ax, (n_bidders, sweep) in zip(axes, sorted(sweeps.items()), strict=True):
            _style_axes(ax)
            exact = expected_utility_exact(n_bidders, K_GRID)
            c = equilibrium_shading(n_bidders)
            peak = 1 / (n_bidders * (n_bidders + 1))

            ax.axhline(0.0, color=MUTED, lw=0.9)
            ax.plot(sweep.k_grid, sweep.mean_utility, "o", ms=3.6, color=ACCENT,
                    label="simulated")
            ax.plot(K_GRID, exact, "--", lw=1.6, color=INK, label="closed form")
            ax.set_ylim(-0.56, 0.34)

            ax.axvline(c, color=GREEN, lw=1.5, ls=":")
            ax.plot([c], [peak], "o", ms=7, color=GREEN, mec="white", mew=1.4, zorder=4)
            ax.annotate(
                f"optimum k = {c:.3g}\nE[U] = {peak:.4f}",
                xy=(c, peak), xytext=(0, 16), textcoords="offset points",
                ha="center", color=GREEN, fontsize=8.8, fontweight="bold",
            )
            ax.axvline(1.0, color=HIGHLIGHT, lw=1.5, ls=":")
            # Parked in the empty upper-right quadrant: the curve is strictly
            # negative to the right of k = 1, so nothing is covered.
            ax.annotate(
                "truthful k = 1\nutility = 0",
                xy=(1.0, 0.0), xycoords="data",
                xytext=(0.58, 0.95), textcoords="axes fraction",
                ha="left", va="top", color=HIGHLIGHT, fontsize=9, fontweight="bold",
                arrowprops={"arrowstyle": "-", "color": HIGHLIGHT, "lw": 0.9,
                            "shrinkA": 2, "shrinkB": 2},
            )
            ax.set_title(f"n = {n_bidders} bidders", fontsize=11.5, pad=10)
            ax.set_xlabel("shading factor k   (bidder 1 bids k·v)")

        axes[0].set_ylabel("bidder 1's expected utility")
        handles, labels = axes[0].get_legend_handles_labels()
        _header(
            fig,
            "First-price auction: deviating from the equilibrium bid",
            "The optimum is k = (n-1)/n, not k = 1. Bidding your true value earns exactly zero, "
            "because the only auctions you win are ones\nwhere you hand over your entire "
            "surplus. Contrast part 1, where b = v was the best possible bid.",
            reserve=0.19,
            bottom=0.1,
        )
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9.5,
                   bbox_to_anchor=(0.5, 0.006))
    return _save(fig, "part2_first_price_shading.png")


def plot_overpayment(sweeps: dict[int, ShadingSweep]) -> str:
    """Part 2: how much the winner left on the table."""
    with plt.rc_context(STYLE):
        fig, (left, right) = plt.subplots(1, 2, figsize=(12.6, 4.6))

        for (n_bidders, sweep), colour in zip(sorted(sweeps.items()), SERIES, strict=True):
            left.plot(sweep.k_grid, 100 * sweep.could_have_paid_less, lw=2.2, color=colour,
                      label=f"first-price, n = {n_bidders}")
            right.plot(sweep.k_grid, 100 * sweep.mean_overpayment_pct, lw=2.2, color=colour,
                       label=f"n = {n_bidders}")

        # The comparison that makes the left panel worth drawing: in a second-price
        # auction the winner pays the runner-up's bid exactly, so the gap is always 0.
        left.plot(K_GRID, np.zeros_like(K_GRID), lw=2.2, ls="--", color=INK,
                  label="second-price (for comparison)")
        left.annotate("all three curves sit at 100%", xy=(1.05, 100), xytext=(0, -26),
                      textcoords="offset points", ha="center", fontsize=9, fontweight="bold",
                      color=SUBTLE)

        for ax in (left, right):
            _style_axes(ax)
            ax.set_xlabel("shading factor k")
        left.set_ylim(-8, 118)
        left.set_ylabel("% of auctions where the winner\ncould have paid less and still won")
        left.set_title("How often the winner overpaid", fontsize=11.5, pad=10)
        left.legend(fontsize=9, loc="center left")
        right.set_ylabel("mean overpayment\n(% of the winning bid)")
        right.set_title("By how much", fontsize=11.5, pad=10)
        right.legend(fontsize=9.5, loc="lower right")

        _header(
            fig,
            "First-price auctions leave money on the table",
            "The fraction is ~100% at every k, which is why it says little on its own: bids are "
            "drawn from continuous distributions, so the winner\nalmost surely strictly outbid "
            "the runner-up. The size of the gap is the informative statistic -- and the "
            "second-price auction\nsets it to zero by construction, which is the whole point of "
            "charging the runner-up's bid.",
            reserve=0.23,
        )
    return _save(fig, "part2_overpayment.png")


NODE_POSITIONS = {
    "A": (0.0, 0.0),
    "B": (1.0, 1.0),
    "D": (2.0, 1.0),
    "C": (1.0, -1.0),
    "E": (2.0, -1.0),
    "F": (3.0, 0.0),
}

#: Where along each edge to place its label, as a fraction from the first node.
#: Most sit at the midpoint. CF runs diagonally across the whole figure, so its
#: midpoint lands on both the BE edge and the EF label; it is pulled back close
#: to C, and CE is pushed towards E to keep the two apart.
LABEL_POSITION = {("C", "F"): 0.12, ("C", "E"): 0.72}


def plot_vcg_network() -> str:
    """Figure 7.3 redrawn, annotated with the VCG outcome."""
    outcome = run_vcg(FIGURE_7_3_COSTS)
    on_path = set(outcome.chosen.edges)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9.6, 5.0))
        ax.set_axis_off()

        for edge, cost in sorted(FIGURE_7_3_COSTS.items()):
            (x1, y1), (x2, y2) = NODE_POSITIONS[edge[0]], NODE_POSITIONS[edge[1]]
            chosen = edge in on_path
            ax.plot([x1, x2], [y1, y2], lw=3.6 if chosen else 1.5,
                    color=ACCENT if chosen else MUTED, zorder=1, solid_capstyle="round")

            t = LABEL_POSITION.get(edge, 0.5)
            mx, my = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
            payment = outcome.payments[edge]
            ax.text(
                mx, my,
                f"cost {cost:g}\npaid {payment:g}",
                ha="center", va="center", fontsize=9,
                fontweight="bold" if chosen else "normal",
                color="white" if chosen else SUBTLE,
                bbox={"boxstyle": "round,pad=0.32",
                      "facecolor": ACCENT if chosen else "white",
                      "edgecolor": ACCENT if chosen else GRID},
                zorder=3,
            )

        for node, (x, y) in NODE_POSITIONS.items():
            ax.plot([x], [y], "o", ms=30, color="white", mec=INK, mew=1.6, zorder=4)
            ax.text(x, y, node, ha="center", va="center", fontsize=12, fontweight="bold",
                    zorder=5)

        ax.set_title("VCG for buying the cheapest path from A to F  (Figure 7.3)",
                     fontsize=13, fontweight="bold", pad=14)
        ax.set_xlim(-0.45, 3.45)
        ax.set_ylim(-1.35, 1.4)
        total = sum(outcome.payments.values())
        _footer(
            fig,
            f"Chosen path {outcome.chosen.label}, true cost {outcome.chosen.cost:g}; the "
            f"mechanism pays out {total:g} in total.\n"
            "BE and EF both cost 1, yet EF is paid 3 and BE only 2. The network without EF "
            "(cost 7) is worse than the network without BE (cost 6),\n"
            "so EF's marginal contribution is larger. VCG prices market power, not cost.",
            reserve=0.2,
        )
    return _save(fig, "part3_vcg_network.png")


def plot_vcg_misreports(sweeps: dict[str, list[dict]]) -> str:
    """Part 3: utility as one edge varies its reported cost."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey=True)

        for ax, (edge_label, rows) in zip(axes, sweeps.items(), strict=True):
            _style_axes(ax)
            reports = np.array([r["reported_cost"] for r in rows])
            utility = np.array([r["utility"] for r in rows])
            used = np.array([r["on_chosen_path"] for r in rows])
            true_cost = rows[0]["true_cost"]

            ax.axhline(0.0, color=MUTED, lw=0.9)
            ax.step(reports, utility, where="post", lw=2.2, color=ACCENT)
            ax.scatter(reports[used], utility[used], s=34, color=ACCENT, zorder=3,
                       label="edge is used")
            ax.scatter(reports[~used], utility[~used], s=34, facecolor="white",
                       edgecolor=MUTED, zorder=3, label="edge is not used")
            ax.set_ylim(-1.5, 1.75)
            _mark_vline(ax, true_cost, f"truthful report = {true_cost:g}", HIGHLIGHT)
            ax.set_xlabel("reported cost")
            ax.set_title(edge_label, fontsize=11.5, pad=10)

        axes[0].set_ylabel("edge's utility (payment − true cost)")
        handles, labels = axes[0].get_legend_handles_labels()
        _header(
            fig,
            "No edge can gain by misreporting its cost",
            "AB's payment is pinned at 4 however it bids, so lying can only cost it the "
            "contract. AC can buy its way onto the path by under-reporting,\nbut is then paid 1 "
            "against a true cost of 2. In both cases telling the truth is in the argmax.",
            reserve=0.2,
            bottom=0.1,
        )
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9.5,
                   bbox_to_anchor=(0.5, 0.006))
    return _save(fig, "part3_vcg_misreports.png")
