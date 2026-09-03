"""The ablation table, and the column that says a difference was not shown."""

from __future__ import annotations

from rag_eval.data.scifact import Dataset, Document, Query
from rag_eval.evals import ablation
from rag_eval.metrics.compare import inconclusive, paired_delta


class FakeIndex:
    def __init__(self, order: list[str]) -> None:
        self.order = order

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        return [(doc_id, 1.0) for doc_id in self.order[:k]]


def dataset(queries: int = 40) -> Dataset:
    documents = [Document(doc_id=f"d{i}", title=f"t{i}", text=f"x{i}") for i in range(5)]
    return Dataset(
        documents=documents,
        queries=[Query(query_id=f"q{i}", text="claim", verdict=None) for i in range(queries)],
        qrels={f"q{i}": {"d0": 1} for i in range(queries)},
    )


def variants() -> list[ablation.Variant]:
    return [
        ablation.Variant("good", lambda: FakeIndex(["d0", "d1", "d2"])),
        ablation.Variant("bad", lambda: FakeIndex(["d1", "d2", "d0"])),
    ]


def test_the_best_row_is_the_one_with_the_highest_headline() -> None:
    report = ablation.run(dataset(), variants(), split="train")
    assert report.best.label == "good"


def test_the_table_carries_an_interval_for_every_row() -> None:
    table = ablation.run(dataset(), variants(), split="train").table()
    rows = [line for line in table.splitlines() if line.startswith("| ")][1:]
    assert len(rows) == 2
    assert all(line.count("[") >= 2 for line in rows)


def test_the_best_row_says_best_and_the_others_carry_a_delta() -> None:
    table = ablation.run(dataset(), variants(), split="train").table()
    assert "| best |" in table
    assert "-0." in table


def test_a_difference_that_was_not_shown_is_marked() -> None:
    same = [0.5, 0.4, 0.6, 0.5] * 10
    noisy = [0.5, 0.5, 0.5, 0.5] * 10
    assert inconclusive(paired_delta(noisy, same))


def test_a_difference_on_every_query_is_not_marked() -> None:
    lower = [0.4] * 40
    higher = [0.5] * 40
    delta = paired_delta(lower, higher)
    assert not inconclusive(delta)
    assert delta.mean < 0


def test_the_paired_test_sees_what_two_separate_intervals_miss() -> None:
    # Wide spread across queries, small but consistent difference. The two
    # intervals overlap heavily while every query moved the same way.
    spread = [i / 40 for i in range(40)]
    candidate = [value + 0.01 for value in spread]
    from rag_eval.metrics.retrieval import bootstrap

    assert bootstrap(candidate).overlaps(bootstrap(spread))
    assert not inconclusive(paired_delta(candidate, spread))


def test_running_the_same_variant_twice_gives_the_same_numbers() -> None:
    first = ablation.run(dataset(), variants(), split="train")
    second = ablation.run(dataset(), variants(), split="train")
    assert first.table() == second.table()


def test_an_empty_comparison_is_inconclusive() -> None:
    assert inconclusive(paired_delta([], []))
