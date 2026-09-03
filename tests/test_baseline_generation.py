"""Recording the generation half of the baseline, and the spread it needs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from rag_eval import cli
from rag_eval.config import STUB, Config, Paths
from rag_eval.metrics.agreement import compare


class FakeReport:
    def __init__(self, accuracy: float, faithful: float) -> None:
        human = ["SUPPORT"] * 100
        predicted = ["SUPPORT"] * int(accuracy * 100) + ["NOINFO"] * (100 - int(accuracy * 100))
        self.model_agreement = compare(human, predicted)
        self.faithful_rate = faithful


def args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {"repeat": 3, "top_k": 5, "limit": None}
    values.update(overrides)
    return argparse.Namespace(**values)


def config() -> Config:
    return Config(
        paths=Paths(root=Path("data")),
        provider=STUB,
        model="",
        judge_model="",
        ollama_host="http://nowhere:1",
        rpm=0,
    )


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "build_index", lambda *a, **k: object())


def measurements(monkeypatch: pytest.MonkeyPatch, accuracies: list[float]) -> tuple[dict, dict]:
    queue = [FakeReport(value, 1.0) for value in accuracies]
    monkeypatch.setattr(cli, "run_generation", lambda *a, **k: queue.pop(0))
    return cli._measure_generation(config(), object(), args(repeat=len(accuracies)))


def test_the_recorded_number_is_the_average_of_the_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    means, _ = measurements(monkeypatch, [0.70, 0.72, 0.74])
    assert means["accuracy"] == pytest.approx(0.72)


def test_the_spread_comes_from_the_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    _, spread = measurements(monkeypatch, [0.70, 0.72, 0.74])
    assert spread["accuracy"] == pytest.approx(0.02)


def test_one_run_records_no_spread(monkeypatch: pytest.MonkeyPatch) -> None:
    # A single run says nothing about variance, and inventing a band from it
    # would be the guess the gate is meant to avoid.
    means, spread = measurements(monkeypatch, [0.70])
    assert means["accuracy"] == pytest.approx(0.70)
    assert spread == {}
