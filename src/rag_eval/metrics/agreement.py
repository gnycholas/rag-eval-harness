"""Comparing labels: the model against people, and the judge against people.

Accuracy alone flatters a classifier on unbalanced classes, and SUPPORT
outnumbers CONTRADICT close to two to one here. Cohen's kappa discounts the
agreement that chance alone would produce.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from rag_eval.metrics.retrieval import Interval, bootstrap

# Landis and Koch, the conventional reading. Quoted so the number is not left
# to the reader to interpret.
KAPPA_BANDS = (
    (0.81, "almost perfect"),
    (0.61, "substantial"),
    (0.41, "moderate"),
    (0.21, "fair"),
    (0.01, "slight"),
    (float("-inf"), "none or worse than chance"),
)


def interpret_kappa(kappa: float) -> str:
    return next(label for threshold, label in KAPPA_BANDS if kappa >= threshold)


@dataclass(frozen=True)
class Agreement:
    accuracy: Interval
    kappa: float
    macro_f1: float
    confusion: dict[tuple[str, str], int]
    labels: list[str] = field(default_factory=list)

    @property
    def kappa_reading(self) -> str:
        return interpret_kappa(self.kappa)

    def confusion_table(self) -> str:
        header = "| human \\ predicted | " + " | ".join(self.labels) + " |"
        divider = "|---" * (len(self.labels) + 1) + "|"
        rows = [
            "| "
            + truth
            + " | "
            + " | ".join(str(self.confusion.get((truth, pred), 0)) for pred in self.labels)
            + " |"
            for truth in self.labels
        ]
        return "\n".join([header, divider, *rows])


def cohen_kappa(truth: list[str], predicted: list[str]) -> float:
    if not truth:
        return 0.0

    total = len(truth)
    observed = sum(1 for t, p in zip(truth, predicted, strict=True) if t == p) / total

    truth_counts = Counter(truth)
    predicted_counts = Counter(predicted)
    expected = sum(
        (truth_counts[label] / total) * (predicted_counts[label] / total)
        for label in set(truth) | set(predicted)
    )

    if expected == 1.0:
        # Everything in one class: chance already explains all of it.
        return 0.0
    return (observed - expected) / (1 - expected)


def macro_f1(truth: list[str], predicted: list[str], labels: list[str]) -> float:
    scores = []
    for label in labels:
        true_positive = sum(
            1 for t, p in zip(truth, predicted, strict=True) if t == label and p == label
        )
        predicted_positive = sum(1 for p in predicted if p == label)
        actual_positive = sum(1 for t in truth if t == label)

        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def compare(truth: list[str], predicted: list[str]) -> Agreement:
    labels = sorted(set(truth) | set(predicted))
    hits = [1.0 if t == p else 0.0 for t, p in zip(truth, predicted, strict=True)]

    confusion = Counter(zip(truth, predicted, strict=True))

    return Agreement(
        accuracy=bootstrap(hits),
        kappa=cohen_kappa(truth, predicted),
        macro_f1=macro_f1(truth, predicted, labels),
        confusion=dict(confusion),
        labels=labels,
    )
