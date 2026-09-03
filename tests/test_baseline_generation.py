"""Recording the generation half of the baseline, and the spread it needs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from rag_eval import cli
from rag_eval.config import GOOGLE, Config, Paths
from rag_eval.evals import runs as run_store
from rag_eval.metrics.agreement import compare


class FakeReport:
    def __init__(self, accuracy: float, model: str = "gemini-test") -> None:
        correct = round(accuracy * 100)
        human = ["SUPPORT"] * 100
        predicted = ["SUPPORT"] * correct + ["NOINFO"] * (100 - correct)
        self.model_agreement = compare(human, predicted)
        self.faithful_rate = 1.0
        self.provider = GOOGLE
        self.model = model


def args(repeat: int = 0, limit: int | None = 100) -> argparse.Namespace:
    return argparse.Namespace(repeat=repeat, top_k=5, limit=limit)


def config(model: str = "gemini-test") -> Config:
    return Config(
        paths=Paths(root=Path("data")),
        provider=GOOGLE,
        model=model,
        judge_model="",
        ollama_host="http://nowhere:1",
        rpm=0,
    )


@pytest.fixture(autouse=True)
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "runs.jsonl"
    monkeypatch.setattr(run_store, "RUNS_PATH", path)
    monkeypatch.setattr(cli, "build_index", lambda *a, **k: object())
    monkeypatch.setattr(cli, "build_provider", lambda *a, **k: object())
    return path


def measure(monkeypatch: pytest.MonkeyPatch, accuracies: list[float]) -> tuple[dict, dict]:
    queue = [FakeReport(value) for value in accuracies]
    monkeypatch.setattr(cli, "run_generation", lambda *a, **k: queue.pop(0))
    return cli._measure_generation(config(), object(), args(repeat=len(accuracies)))


def test_the_recorded_number_is_the_average_of_the_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    means, _ = measure(monkeypatch, [0.70, 0.72, 0.74])
    assert means["accuracy"] == pytest.approx(0.72)


def test_the_spread_comes_from_the_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    _, spread = measure(monkeypatch, [0.70, 0.72, 0.74])
    assert spread["accuracy"] == pytest.approx(0.02)


def test_one_run_records_no_spread(monkeypatch: pytest.MonkeyPatch) -> None:
    # A single run says nothing about variance, and inventing a band from it
    # would be the guess the gate is meant to avoid.
    means, spread = measure(monkeypatch, [0.70])
    assert means["accuracy"] == pytest.approx(0.70)
    assert spread == {}


def test_runs_from_separate_invocations_are_averaged_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The quota ends in the middle of a measurement. Two runs today and two
    # tomorrow have to add up to the same thing as four in one go.
    measure(monkeypatch, [0.70, 0.72])
    means, spread = measure(monkeypatch, [0.74, 0.76])
    assert means["accuracy"] == pytest.approx(0.73)
    assert spread["accuracy"] == pytest.approx(0.0258, abs=1e-4)


def test_runs_on_another_model_are_not_averaged_in(monkeypatch: pytest.MonkeyPatch) -> None:
    queue = [FakeReport(0.90, model="other")]
    monkeypatch.setattr(cli, "run_generation", lambda *a, **k: queue.pop(0))
    cli._measure_generation(config("other"), object(), args(repeat=1))

    means, _ = measure(monkeypatch, [0.70, 0.72])
    assert means["accuracy"] == pytest.approx(0.71)


def test_averaging_nothing_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit, match="no recorded runs"):
        cli._measure_generation(config(), object(), args(repeat=0))
