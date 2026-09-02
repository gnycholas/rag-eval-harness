"""Fusion, tested on plain rankings so no index is needed."""

from __future__ import annotations

from rag_eval.index.hybrid import DEFAULT_K, fuse


def ranking(*doc_ids: str) -> list[tuple[str, float]]:
    return [(doc_id, 1.0) for doc_id in doc_ids]


def test_a_document_ranked_well_by_both_wins() -> None:
    fused = fuse([ranking("a", "b", "c"), ranking("a", "x", "y")], top=3)
    assert fused[0][0] == "a"


def test_a_document_found_by_only_one_still_appears() -> None:
    fused = fuse([ranking("a", "b"), ranking("a", "c")], top=5)
    assert {doc_id for doc_id, _ in fused} == {"a", "b", "c"}


def test_score_magnitude_does_not_matter() -> None:
    """The reason for RRF: BM25 returns tens, cosine returns fractions."""
    huge = [("a", 90.0), ("b", 80.0)]
    tiny = [("a", 0.9), ("b", 0.8)]
    assert fuse([huge, tiny], top=2) == fuse([tiny, huge], top=2)


def test_only_position_counts() -> None:
    by_score = fuse([[("a", 100.0), ("b", 1.0)]], top=2)
    by_position = fuse([[("a", 1.0), ("b", 1.0)]], top=2)
    assert [d for d, _ in by_score] == [d for d, _ in by_position]


def test_the_constant_shifts_the_ordering() -> None:
    rankings = [ranking("a", "b", "c"), ranking("c", "b", "a")]
    assert fuse(rankings, k=1, top=3) != fuse(rankings, k=1000, top=3)


def test_the_expected_reciprocal_score() -> None:
    """First place in both rankings scores 2/(k+1)."""
    fused = fuse([ranking("a"), ranking("a")], k=DEFAULT_K, top=1)
    assert fused[0][1] == 2 / (DEFAULT_K + 1)


def test_an_empty_ranking_contributes_nothing() -> None:
    assert fuse([ranking("a", "b"), []], top=5) == fuse([ranking("a", "b")], top=5)


def test_fusion_is_deterministic() -> None:
    rankings = [ranking("a", "b", "c"), ranking("b", "c", "a")]
    assert fuse(rankings, top=3) == fuse(rankings, top=3)


def test_top_limits_the_result() -> None:
    assert len(fuse([ranking("a", "b", "c", "d")], top=2)) == 2
