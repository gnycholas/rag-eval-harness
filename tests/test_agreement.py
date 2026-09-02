"""Agreement metrics, against values worked out by hand."""

from __future__ import annotations

import pytest

from rag_eval.metrics.agreement import cohen_kappa, compare, interpret_kappa, macro_f1

SUPPORT, CONTRADICT, NOINFO = "SUPPORT", "CONTRADICT", "NOINFO"


def test_perfect_agreement_gives_kappa_one() -> None:
    labels = [SUPPORT, CONTRADICT, SUPPORT, NOINFO]
    assert cohen_kappa(labels, labels) == 1.0


def test_kappa_is_zero_when_agreement_is_only_chance() -> None:
    """Both raters call everything SUPPORT: chance explains all of it."""
    truth = [SUPPORT] * 10
    assert cohen_kappa(truth, [SUPPORT] * 10) == 0.0


def test_kappa_is_negative_below_chance() -> None:
    truth = [SUPPORT, CONTRADICT, SUPPORT, CONTRADICT]
    flipped = [CONTRADICT, SUPPORT, CONTRADICT, SUPPORT]
    assert cohen_kappa(truth, flipped) < 0


def test_kappa_worked_by_hand() -> None:
    """4 items, 3 agree. observed = 0.75.
    truth SUPPORT 2/4 CONTRADICT 2/4; predicted SUPPORT 3/4 CONTRADICT 1/4.
    expected = 0.5*0.75 + 0.5*0.25 = 0.5 -> kappa = (0.75-0.5)/0.5 = 0.5"""
    truth = [SUPPORT, SUPPORT, CONTRADICT, CONTRADICT]
    predicted = [SUPPORT, SUPPORT, CONTRADICT, SUPPORT]
    assert cohen_kappa(truth, predicted) == pytest.approx(0.5)


def test_accuracy_alone_flatters_an_unbalanced_classifier() -> None:
    """The reason kappa is reported next to accuracy."""
    truth = [SUPPORT] * 9 + [CONTRADICT]
    lazy = [SUPPORT] * 10

    result = compare(truth, lazy)
    assert result.accuracy.mean == pytest.approx(0.9)
    assert result.kappa == pytest.approx(0.0)


def test_macro_f1_punishes_ignoring_a_class() -> None:
    truth = [SUPPORT] * 9 + [CONTRADICT]
    lazy = [SUPPORT] * 10
    assert macro_f1(truth, lazy, [CONTRADICT, SUPPORT]) < 0.5


def test_confusion_counts_every_pair() -> None:
    result = compare([SUPPORT, CONTRADICT], [SUPPORT, SUPPORT])
    assert result.confusion[(SUPPORT, SUPPORT)] == 1
    assert result.confusion[(CONTRADICT, SUPPORT)] == 1


def test_the_confusion_table_renders() -> None:
    table = compare([SUPPORT, CONTRADICT], [SUPPORT, SUPPORT]).confusion_table()
    assert "human \\ predicted" in table
    assert SUPPORT in table


def test_kappa_carries_a_reading() -> None:
    assert interpret_kappa(0.9) == "almost perfect"
    assert interpret_kappa(0.5) == "moderate"
    assert interpret_kappa(-0.2) == "none or worse than chance"


def test_accuracy_comes_with_an_interval() -> None:
    result = compare([SUPPORT, CONTRADICT] * 20, [SUPPORT, SUPPORT] * 20)
    assert result.accuracy.low < result.accuracy.mean < result.accuracy.high
    assert result.accuracy.n == 40
