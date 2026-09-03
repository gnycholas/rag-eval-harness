"""Reciprocal Rank Fusion over two rankings.

Fusion works on positions, not scores. BM25 returns tens and cosine returns
fractions, and normalising two distributions to compare them is where weighted
sum fusion usually loses to whichever component it was trying to improve.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_eval.index.base import Hit, Index, rank, search_many

# Measured on the 809 training queries, not taken from the literature. The
# usual value of 60 turned out to be close to the worst choice here:
#   k=1    nDCG@10 0.7092   recall@10 0.8436
#   k=10   nDCG@10 0.6983   recall@10 0.8402
#   k=60   nDCG@10 0.6765   recall@10 0.7991
# k of 0, 1 and 2 are within a thousandth of each other, so this is a plateau
# rather than a fragile peak. Copying 60 from a paper cost 0.04 nDCG and would
# have led to the wrong conclusion about hybrid retrieval on this corpus.
DEFAULT_K = 1

# How deep each component ranking goes before fusion. Measured the same way as
# the constant above.
DEFAULT_DEPTH = 100


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
    depth: int = DEFAULT_DEPTH

    def search(self, query: str, k: int) -> list[Hit]:
        """Fuse deeper than the cutoff, so a document ranked well by only one
        component still has a chance to surface."""
        return self.search_batch([query], k)[0]

    def search_batch(self, queries: list[str], k: int) -> list[list[Hit]]:
        depth = max(self.depth, k)
        sparse = search_many(self.sparse, queries, depth)
        dense = search_many(self.dense, queries, depth)
        return [fuse([s, d], k=self.k, top=k) for s, d in zip(sparse, dense, strict=True)]
