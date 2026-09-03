"""Running an index over a split and reporting what it scored."""

from __future__ import annotations

from dataclasses import dataclass, field

from rag_eval.data.scifact import Dataset
from rag_eval.index.base import Index, search_many
from rag_eval.metrics.retrieval import (
    Interval,
    bootstrap,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

RECALL_CUTOFFS = (1, 5, 10, 20)
NDCG_CUTOFF = 10
MRR_CUTOFF = 10
SEARCH_DEPTH = max(max(RECALL_CUTOFFS), NDCG_CUTOFF, MRR_CUTOFF)


@dataclass(frozen=True)
class RetrievalReport:
    metrics: dict[str, Interval]
    queries: int
    # Kept so two configurations can be compared query by query. Comparing the
    # two intervals instead throws away the pairing and calls real differences
    # inconclusive.
    per_query: dict[str, list[float]] = field(default_factory=dict)

    def table(self) -> str:
        lines = ["| metric | value | 95% CI |", "|---|---|---|"]
        lines.extend(
            f"| {name} | {value.mean:.4f} | [{value.low:.4f}, {value.high:.4f}] |"
            for name, value in self.metrics.items()
        )
        return "\n".join(lines)


def evaluate(index: Index, dataset: Dataset, *, depth: int = SEARCH_DEPTH) -> RetrievalReport:
    """Score an index over every query the split has judgments for.

    Queries without judgments are skipped rather than counted as failures, and
    the number actually scored travels with the result.
    """
    per_query: dict[str, list[float]] = {
        **{f"recall@{k}": [] for k in RECALL_CUTOFFS},
        f"ndcg@{NDCG_CUTOFF}": [],
        f"mrr@{MRR_CUTOFF}": [],
    }

    judged = dataset.judged()
    rankings = search_many(index, [q.text for q in judged], depth)

    for query, hits in zip(judged, rankings, strict=True):
        relevance = dataset.qrels[query.query_id]
        relevant = {doc_id for doc_id, score in relevance.items() if score > 0}
        ranked = [doc_id for doc_id, _ in hits]

        for k in RECALL_CUTOFFS:
            per_query[f"recall@{k}"].append(recall_at_k(ranked, relevant, k))
        per_query[f"ndcg@{NDCG_CUTOFF}"].append(ndcg_at_k(ranked, relevance, NDCG_CUTOFF))
        per_query[f"mrr@{MRR_CUTOFF}"].append(reciprocal_rank(ranked, relevant, MRR_CUTOFF))

    return RetrievalReport(
        metrics={name: bootstrap(values) for name, values in per_query.items()},
        queries=len(judged),
        per_query=per_query,
    )
