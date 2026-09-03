"""Retrieve, verify, and judge, over the queries that have human labels."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rag_eval.data.scifact import Dataset, Document
from rag_eval.generate.provider import LlmProvider, ResponseError, Usage
from rag_eval.generate.schema import JudgeVerdict
from rag_eval.generate.verify import Verified, build_prompt, normalize_verdict, verify
from rag_eval.index.base import Index
from rag_eval.metrics.agreement import Agreement, compare

TOP_K = 5

JUDGE_SYSTEM = (
    "You are checking scientific claims against passages, independently.\n"
    "Answer SUPPORT, CONTRADICT, or NOINFO, and say whether the passages really carry it.\n"
    "Judge only what the passages say."
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Outcome:
    verified: Verified
    human_verdict: str
    relevant_retrieved: bool

    @property
    def correct(self) -> bool:
        return self.verified.verdict == self.human_verdict

    @property
    def failure_kind(self) -> str | None:
        """Where a wrong answer went wrong.

        'Accuracy 0.71' says nothing about where to invest. Splitting the
        errors does.
        """
        if self.correct:
            return None
        return "retrieval" if not self.relevant_retrieved else "reasoning"


@dataclass(frozen=True)
class GenerationReport:
    model_agreement: Agreement
    judge_agreement: Agreement | None
    outcomes: list[Outcome]
    usage: Usage
    provider: str
    model: str
    faithful_rate: float
    # Claims whose answer could not be used at all. Not the same as a wrong
    # answer, and left out of the accuracy rather than counted against it, so
    # the number travels next to the score instead of hiding inside it.
    unusable: dict[str, int] = field(default_factory=lambda: {"verifier": 0, "judge": 0})

    @property
    def failures(self) -> dict[str, int]:
        counts = {"retrieval": 0, "reasoning": 0}
        for outcome in self.outcomes:
            kind = outcome.failure_kind
            if kind:
                counts[kind] += 1
        return counts


def _retrieve(index: Index, dataset: Dataset, claim: str, k: int) -> list[Document]:
    by_id = dataset.by_id
    return [by_id[doc_id] for doc_id, _ in index.search(claim, k) if doc_id in by_id]


def run(
    index: Index,
    dataset: Dataset,
    provider: LlmProvider,
    *,
    top_k: int = TOP_K,
    judge: LlmProvider | None = None,
    limit: int | None = None,
) -> GenerationReport:
    queries = dataset.with_verdict()
    if limit:
        queries = queries[:limit]

    outcomes: list[Outcome] = []
    judged_human: list[str] = []
    judge_labels: list[str] = []
    unusable = {"verifier": 0, "judge": 0}
    total = Usage()

    for position, query in enumerate(queries, start=1):
        passages = _retrieve(index, dataset, query.text, top_k)

        try:
            result = verify(provider, query.query_id, query.text, passages)
        except (ResponseError, ValueError) as exc:
            unusable["verifier"] += 1
            log.warning("claim %s has no usable answer: %s", query.query_id, exc)
        else:
            total = total + result.usage
            relevant = {d for d, score in dataset.qrels[query.query_id].items() if score > 0}
            outcomes.append(
                Outcome(
                    verified=result,
                    human_verdict=query.verdict or "",
                    relevant_retrieved=bool(relevant & {d.doc_id for d in passages}),
                )
            )

        # The judge answers the same claim on its own, so its result does not
        # depend on the verifier having produced one.
        if judge is not None:
            try:
                verdict, usage = judge.complete(
                    JUDGE_SYSTEM, build_prompt(query.text, passages), JudgeVerdict
                )
                label = normalize_verdict(verdict.verdict)
            except (ResponseError, ValueError) as exc:
                unusable["judge"] += 1
                log.warning("claim %s has no usable judgment: %s", query.query_id, exc)
            else:
                judged_human.append(query.verdict or "")
                judge_labels.append(label)
                total = total + usage

        if position % 25 == 0:
            log.info("%d/%d", position, len(queries))

    human = [o.human_verdict for o in outcomes]
    predicted = [o.verified.verdict for o in outcomes]
    faithful = sum(1 for o in outcomes if o.verified.faithful_citations)

    return GenerationReport(
        model_agreement=compare(human, predicted),
        judge_agreement=compare(judged_human, judge_labels) if judge_labels else None,
        outcomes=outcomes,
        usage=total,
        provider=provider.name,
        model=provider.model,
        faithful_rate=faithful / len(outcomes) if outcomes else 0.0,
        unusable=unusable,
    )
