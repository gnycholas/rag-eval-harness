"""Reciprocal Rank Fusion over two rankings.

Fusion works on positions, not scores. BM25 returns tens and cosine returns
fractions, and normalising two distributions to compare them is where weighted
sum fusion usually loses to whichever component it was trying to improve.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_eval.index.base import Hit, Index, rank, search_many

# The value the literature uses. It stays configurable because a constant
# copied from a paper is not a measurement on this corpus.
DEFAULT_K = 60


def fuse(rankings: list[list[Hit]], *, k: int = DEFAULT_K, top: int = 10) -> list[Hit]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, (doc_id, _) in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + position)
    return rank(scores, top)


@dataclass
class HybridIndex:
    sparse: Index
    dense: Index
    k: int = DEFAULT_K
    depth: int = 100

    def search(self, query: str, k: int) -> list[Hit]:
        """Fuse deeper than the cutoff, so a document ranked well by only one
        component still has a chance to surface."""
        return self.search_batch([query], k)[0]

    def search_batch(self, queries: list[str], k: int) -> list[list[Hit]]:
        depth = max(self.depth, k)
        sparse = search_many(self.sparse, queries, depth)
        dense = search_many(self.dense, queries, depth)
        return [fuse([s, d], k=self.k, top=k) for s, d in zip(sparse, dense, strict=True)]
