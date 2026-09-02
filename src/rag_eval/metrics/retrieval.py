"""Retrieval metrics against the human relevance judgments.

Every number comes back with a confidence interval. With 300 test queries a
one point difference is often noise, and reporting the mean alone is how a
portfolio ends up claiming a gain that is not there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 20260902
CONFIDENCE = 0.95


@dataclass(frozen=True)
class Interval:
    mean: float
    low: float
    high: float
    n: int

    def overlaps(self, other: Interval) -> bool:
        return self.low <= other.high and other.low <= self.high

    def __str__(self) -> str:
        return f"{self.mean:.4f} [{self.low:.4f}, {self.high:.4f}]"


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    found = len(set(ranked[:k]) & relevant)
    return found / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str], k: int) -> float:
    for position, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: list[str], relevance: dict[str, int], k: int) -> float:
    """Binary relevance, log2 discount, ideal taken from the judgments."""
    gains = [relevance.get(doc_id, 0) for doc_id in ranked[:k]]
    dcg = sum(gain / math.log2(position + 1) for position, gain in enumerate(gains, start=1))

    ideal_gains = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(gain / math.log2(position + 1) for position, gain in enumerate(ideal_gains, start=1))

    return dcg / idcg if idcg else 0.0


def bootstrap(values: list[float], *, samples: int = BOOTSTRAP_SAMPLES) -> Interval:
    """Percentile interval over queries, with a fixed seed so runs compare."""
    if not values:
        return Interval(0.0, 0.0, 0.0, 0)

    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draws = generator.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)

    tail = (1 - CONFIDENCE) / 2
    return Interval(
        mean=float(array.mean()),
        low=float(np.quantile(draws, tail)),
        high=float(np.quantile(draws, 1 - tail)),
        n=len(array),
    )
