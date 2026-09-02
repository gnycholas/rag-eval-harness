"""Metrics checked against values worked out by hand.

Testing a metric against its own output only proves it is self consistent.
"""

from __future__ import annotations

import math

import pytest

from rag_eval.metrics.retrieval import (
    Interval,
    bootstrap,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_counts_relevant_documents_found() -> None:
    assert recall_at_k(["a", "b", "c"], {"a", "d"}, k=3) == 0.5


def test_recall_ignores_position() -> None:
    assert recall_at_k(["x", "y", "a"], {"a"}, k=3) == recall_at_k(["a", "x", "y"], {"a"}, k=3)


def test_recall_respects_the_cutoff() -> None:
    assert recall_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0


def test_recall_with_nothing_relevant_is_zero() -> None:
    assert recall_at_k(["a"], set(), k=5) == 0.0


def test_reciprocal_rank_is_one_over_the_first_hit() -> None:
    assert reciprocal_rank(["x", "a", "y"], {"a"}, k=10) == 0.5
    assert reciprocal_rank(["a"], {"a"}, k=10) == 1.0


def test_reciprocal_rank_is_zero_when_nothing_is_found() -> None:
    assert reciprocal_rank(["x", "y"], {"a"}, k=10) == 0.0


def test_ndcg_is_one_for_a_perfect_ranking() -> None:
    assert ndcg_at_k(["a", "b"], {"a": 1, "b": 1}, k=10) == 1.0


def test_ndcg_of_a_relevant_document_in_second_place() -> None:
    """DCG = 1/log2(3); the ideal puts it first, so IDCG = 1."""
    assert ndcg_at_k(["x", "a"], {"a": 1}, k=10) == pytest.approx(1 / math.log2(3))


def test_ndcg_of_a_relevant_document_in_third_place() -> None:
    assert ndcg_at_k(["x", "y", "a"], {"a": 1}, k=10) == pytest.approx(1 / math.log2(4))


def test_ndcg_rewards_the_better_ordering() -> None:
    relevance = {"a": 1, "b": 1}
    assert ndcg_at_k(["a", "x", "b"], relevance, k=10) > ndcg_at_k(["x", "a", "b"], relevance, k=10)


def test_ndcg_is_zero_when_nothing_relevant_is_retrieved() -> None:
    assert ndcg_at_k(["x", "y"], {"a": 1}, k=10) == 0.0


def test_ndcg_without_judgments_is_zero() -> None:
    assert ndcg_at_k(["a"], {}, k=10) == 0.0


def test_bootstrap_reports_the_mean_and_the_count() -> None:
    result = bootstrap([0.0, 1.0, 0.5, 0.5])
    assert result.mean == 0.5
    assert result.n == 4


def test_bootstrap_of_a_constant_has_no_width() -> None:
    result = bootstrap([0.7] * 20)
    assert result.low == result.high
    assert result.mean == pytest.approx(0.7)


def test_bootstrap_is_reproducible() -> None:
    values = [0.1, 0.9, 0.4, 0.6, 0.5]
    assert bootstrap(values) == bootstrap(values)


def test_bootstrap_of_nothing_is_empty() -> None:
    assert bootstrap([]).n == 0


def test_overlapping_intervals_are_recognised() -> None:
    """The check that keeps a one point difference from being called a gain."""
    a = Interval(0.70, 0.66, 0.74, 300)
    b = Interval(0.72, 0.68, 0.76, 300)
    assert a.overlaps(b)
    assert not a.overlaps(Interval(0.80, 0.76, 0.84, 300))
