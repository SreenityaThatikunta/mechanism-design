"""Rival bidding strategies for the second-price experiments.

A strategy maps a rival's private value to the bid it submits. Two of the three
depend on the value; the third deliberately ignores it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class RivalStrategy:
    """A rival's bidding rule, plus the theory needed to validate it.

    Attributes:
        key: Short machine-readable identifier.
        label: Human-readable name used in figures and tables.
        bid: Maps ``(values, noise)`` to bids. ``noise`` is an independent
            U[0, 1] draw, used only by value-ignoring strategies.
        bid_cdf: CDF of a *single* rival's bid, evaluated elementwise. Having
            this in closed form is what lets the tests check the simulation
            against exact theory rather than against another simulation.
        support_max: Largest bid the strategy can ever submit.
    """

    key: str
    label: str
    bid: Callable[[np.ndarray, np.ndarray], np.ndarray]
    bid_cdf: Callable[[np.ndarray], np.ndarray]
    support_max: float


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


TRUTHFUL = RivalStrategy(
    key="truthful",
    label="Bids truthfully (b = v)",
    bid=lambda values, noise: values,
    bid_cdf=lambda x: _clip01(x),
    support_max=1.0,
)

HALF = RivalStrategy(
    key="half",
    label="Bids half its value (b = v/2)",
    bid=lambda values, noise: 0.5 * values,
    bid_cdf=lambda x: _clip01(2.0 * x),
    support_max=0.5,
)

RANDOM = RivalStrategy(
    key="random",
    label="Bids U[0,1], ignoring its value",
    bid=lambda values, noise: noise,
    bid_cdf=lambda x: _clip01(x),
    support_max=1.0,
)

#: Ordered as presented in the brief: (a) truthful, (b) half, (c) random.
STRATEGIES: tuple[RivalStrategy, ...] = (TRUTHFUL, HALF, RANDOM)

STRATEGIES_BY_KEY = {s.key: s for s in STRATEGIES}
