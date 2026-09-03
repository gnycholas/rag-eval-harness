"""Comparing two configurations on the same queries."""

from __future__ import annotations

import numpy as np

from rag_eval.metrics.retrieval import BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, CONFIDENCE, Interval


def paired_delta(
    candidate: list[float], reference: list[float], *, samples: int = BOOTSTRAP_SAMPLES
) -> Interval:
    """Interval for the per query difference between two configurations.

    Both are scored on the same queries, so the difference can be bootstrapped
    directly. Reading two separate intervals and calling any overlap a tie is
    the conservative version of this test: the intervals of two configurations
    can overlap while every single query moved the same way.
    """
    if not candidate or len(candidate) != len(reference):
        return Interval(0.0, 0.0, 0.0, 0)

    differences = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draws = generator.choice(differences, size=(samples, len(differences)), replace=True).mean(
        axis=1
    )

    tail = (1 - CONFIDENCE) / 2
    return Interval(
        mean=float(differences.mean()),
        low=float(np.quantile(draws, tail)),
        high=float(np.quantile(draws, 1 - tail)),
        n=len(differences),
    )


def inconclusive(delta: Interval) -> bool:
    """A difference whose interval contains zero has not been shown."""
    return delta.low <= 0.0 <= delta.high
