"""The end to end pipeline, on the stub provider and the fixture corpus."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_eval.data import scifact
from rag_eval.evals.pipeline import Outcome, run
from rag_eval.generate.provider import ProviderError, ResponseError, StubProvider, Usage
from rag_eval.generate.verify import Verified
from rag_eval.index.sparse import Bm25Index

FIXTURE = Path(__file__).parent / "fixtures" / "scifact"


@pytest.fixture(scope="module")
def dataset() -> scifact.Dataset:
    return scifact.load(FIXTURE, check_counts=False)


@pytest.fixture(scope="module")
def index(dataset: scifact.Dataset) -> Bm25Index:
    built = Bm25Index()
    built.build(dataset.documents)
    return built


def outcome(*, correct: bool, retrieved: bool) -> Outcome:
    verified = Verified(
        query_id="q",
        verdict="SUPPORT",
        rationale="",
        cited_indices=[1],
        retrieved_ids=["d1"],
        faithful_citations=True,
        usage=Usage(),
    )
    return Outcome(
        verified=verified,
        human_verdict="SUPPORT" if correct else "CONTRADICT",
        relevant_retrieved=retrieved,
    )


def test_a_correct_answer_has_no_failure_kind() -> None:
    assert outcome(correct=True, retrieved=True).failure_kind is None


def test_a_miss_with_nothing_relevant_retrieved_is_a_retrieval_failure() -> None:
    assert outcome(correct=False, retrieved=False).failure_kind == "retrieval"


def test_a_miss_with_the_right_passage_present_is_a_reasoning_failure() -> None:
    """Accuracy alone does not say where to invest; this split does."""
    assert outcome(correct=False, retrieved=True).failure_kind == "reasoning"


def test_only_queries_with_a_human_verdict_are_scored(
    index: Bm25Index, dataset: scifact.Dataset
) -> None:
    report = run(index, dataset, StubProvider())
    assert len(report.outcomes) == len(dataset.with_verdict())


def test_the_report_records_who_produced_it(index: Bm25Index, dataset: scifact.Dataset) -> None:
    report = run(index, dataset, StubProvider())
    assert report.provider == "stub"
    assert report.model == "stub"


def test_failures_are_split_by_kind(index: Bm25Index, dataset: scifact.Dataset) -> None:
    report = run(index, dataset, StubProvider())
    assert set(report.failures) == {"retrieval", "reasoning"}
    correct = sum(1 for o in report.outcomes if o.correct)
    assert correct + sum(report.failures.values()) == len(report.outcomes)


def test_the_judge_produces_its_own_agreement(index: Bm25Index, dataset: scifact.Dataset) -> None:
    report = run(index, dataset, StubProvider(), judge=StubProvider())
    assert report.judge_agreement is not None


def test_without_a_judge_there_is_no_judge_agreement(
    index: Bm25Index, dataset: scifact.Dataset
) -> None:
    assert run(index, dataset, StubProvider()).judge_agreement is None


def test_the_limit_truncates_the_run(index: Bm25Index, dataset: scifact.Dataset) -> None:
    assert len(run(index, dataset, StubProvider(), limit=2).outcomes) == 2


def test_faithfulness_is_reported(index: Bm25Index, dataset: scifact.Dataset) -> None:
    assert 0.0 <= run(index, dataset, StubProvider()).faithful_rate <= 1.0


class Flaky:
    """A provider that fails a given call, so the run has to survive it."""

    name = "stub"
    model = "flaky"

    def __init__(self, fail_on: int, error: Exception) -> None:
        self.fail_on = fail_on
        self.error = error
        self.calls = 0
        self.inner = StubProvider()

    def complete(self, system: str, prompt: str, schema):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls == self.fail_on:
            raise self.error
        return self.inner.complete(system, prompt, schema)


def test_an_unusable_answer_is_counted_and_the_run_goes_on(
    index: Bm25Index, dataset: scifact.Dataset
) -> None:
    provider = Flaky(2, ResponseError("did not answer in the schema"))
    report = run(index, dataset, provider, limit=4)
    assert report.unusable == {"verifier": 1, "judge": 0}
    assert len(report.outcomes) == 3
    assert report.model_agreement.accuracy.n == 3


def test_a_verdict_outside_the_label_set_is_counted_the_same_way(
    index: Bm25Index, dataset: scifact.Dataset
) -> None:
    provider = Flaky(1, ValueError("verdict outside the label set"))
    report = run(index, dataset, provider, limit=3)
    assert report.unusable["verifier"] == 1


def test_an_unusable_judgment_does_not_shift_the_pairing(
    index: Bm25Index, dataset: scifact.Dataset
) -> None:
    # The judge answers claim two badly. Its agreement has to be scored against
    # the labels of the claims it actually answered, not against a list that
    # has slid by one.
    judge = Flaky(2, ResponseError("no candidate"))
    report = run(index, dataset, StubProvider(), judge=judge, limit=4)
    assert report.unusable == {"verifier": 0, "judge": 1}
    assert report.judge_agreement is not None
    assert report.judge_agreement.accuracy.n == 3
    assert report.model_agreement.accuracy.n == 4


def test_an_error_that_ends_the_run_is_not_swallowed(
    index: Bm25Index, dataset: scifact.Dataset
) -> None:
    # A quota that is gone is not a bad answer. Counting it per claim would
    # produce a report built from nothing.
    provider = Flaky(2, ProviderError("out of its daily free tier quota"))
    with pytest.raises(ProviderError, match="daily"):
        run(index, dataset, provider, limit=4)
