"""Running several configurations over the same split and comparing them.

The table is the point. A configuration that scores higher than another on the
mean has not been shown to be better, and the column that says so is what keeps
a portfolio README from claiming a gain that is not there.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from rag_eval.data.scifact import Dataset
from rag_eval.evals.runner import evaluate
from rag_eval.index.base import Index
from rag_eval.metrics.compare import inconclusive, paired_delta
from rag_eval.metrics.retrieval import Interval

HEADLINE = "ndcg@10"
SECONDARY = "recall@10"


@dataclass(frozen=True)
class Variant:
    label: str
    build: Callable[[], Index]


@dataclass(frozen=True)
class Row:
    label: str
    metrics: dict[str, Interval]
    per_query: dict[str, list[float]]
    seconds: float


@dataclass(frozen=True)
class AblationReport:
    rows: list[Row]
    split: str
    queries: int

    @property
    def best(self) -> Row:
        return max(self.rows, key=lambda row: row.metrics[HEADLINE].mean)

    def table(self) -> str:
        best = self.best
        header = (
            f"| configuration | {HEADLINE} | {SECONDARY} | delta {HEADLINE} vs best | seconds |"
        )
        lines = [header, "|---|---|---|---|---|"]
        for row in sorted(self.rows, key=lambda r: -r.metrics[HEADLINE].mean):
            lines.append(
                f"| {row.label} | {row.metrics[HEADLINE]} | {row.metrics[SECONDARY]} | "
                f"{self._verdict(row, best)} | {row.seconds:.0f} |"
            )
        return "\n".join(lines)

    def _verdict(self, row: Row, best: Row) -> str:
        if row is best:
            return "best"
        delta = paired_delta(row.per_query[HEADLINE], best.per_query[HEADLINE])
        mark = " (inconclusive)" if inconclusive(delta) else ""
        return f"{delta}{mark}"


def run(dataset: Dataset, variants: list[Variant], *, split: str) -> AblationReport:
    rows = []
    for variant in variants:
        started = time.perf_counter()
        report = evaluate(variant.build(), dataset)
        rows.append(
            Row(
                label=variant.label,
                metrics=report.metrics,
                per_query=report.per_query,
                seconds=time.perf_counter() - started,
            )
        )
    return AblationReport(rows=rows, split=split, queries=rows[0].metrics[HEADLINE].n)
