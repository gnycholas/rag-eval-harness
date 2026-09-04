"""The gate as the command actually runs it.

check() was unit tested from the first commit and the command still handed it an
empty generation dict, so a measured band sat in the baseline file blocking
nothing. Testing the rule without testing the wiring is how that survives.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_eval import cli
from rag_eval.config import GOOGLE, Config, Paths
from rag_eval.evals import gate as gating
from rag_eval.evals import runs as run_store
from rag_eval.metrics.agreement import compare

MODEL = "gemini-test"
BASELINE_ACCURACY = 0.853191
STDDEV = 0.002913
CLAIMS = 100


def baseline() -> gating.Baseline:
    return gating.Baseline(
        retrieval={"ndcg@10": 0.693686},
        retrieval_config={
            "index": "hybrid",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "rrf_k": "1",
        },
        generation={"accuracy": BASELINE_ACCURACY, "faithful_citations": 1.0},
        generation_stddev={"accuracy": STDDEV, "faithful_citations": 0.0},
        generation_claims=CLAIMS,
        provider=GOOGLE,
        model=MODEL,
        prompt_version="v1",
        recorded_at="2026-09-04",
    )


class FakeGeneration:
    def __init__(self, accuracy: float, faithful: float = 1.0) -> None:
        correct = round(accuracy * CLAIMS)
        human = ["SUPPORT"] * CLAIMS
        predicted = ["SUPPORT"] * correct + ["NOINFO"] * (CLAIMS - correct)
        self.model_agreement = compare(human, predicted)
        self.faithful_rate = faithful


def args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {"generation": False, "top_k": 5, "limit": None, "yes": True}
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Everything the command reaches for, replaced by something free."""
    calls = SimpleNamespace(providers=0, generations=0)
    runs = tmp_path / "runs.jsonl"
    monkeypatch.setattr(run_store, "RUNS_PATH", runs)
    monkeypatch.setattr(gating.Baseline, "load", classmethod(lambda cls, path=None: baseline()))
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda: Config(
            paths=Paths(root=tmp_path),
            provider=GOOGLE,
            model=MODEL,
            judge_model="",
            ollama_host="http://nowhere:1",
            rpm=0,
        ),
    )

    dataset = SimpleNamespace(with_verdict=lambda: range(CLAIMS))
    monkeypatch.setattr(cli.scifact, "load", lambda *a, **k: dataset)
    monkeypatch.setattr(cli, "build_index", lambda *a, **k: object())
    monkeypatch.setattr(
        cli,
        "evaluate",
        lambda *a, **k: SimpleNamespace(metrics={"ndcg@10": SimpleNamespace(mean=0.693686)}),
    )

    def provider(*a: object, **k: object) -> SimpleNamespace:
        calls.providers += 1
        return SimpleNamespace(name=GOOGLE, model=MODEL)

    monkeypatch.setattr(cli, "build_provider", provider)

    def generate(*a: object, **k: object) -> FakeGeneration:
        calls.generations += 1
        return harness.report

    monkeypatch.setattr(cli, "run_generation", generate)

    harness = SimpleNamespace(calls=calls, runs=runs, report=FakeGeneration(BASELINE_ACCURACY))
    return harness


def test_the_default_gate_never_reaches_a_model(harness: SimpleNamespace) -> None:
    """Rescoring retrieval is free; a call per claim is not, so it stays opt in."""
    assert cli.cmd_gate(args()) == 0
    assert harness.calls.providers == 0
    assert harness.calls.generations == 0


def test_a_generation_run_inside_the_band_passes(harness: SimpleNamespace) -> None:
    harness.report = FakeGeneration(BASELINE_ACCURACY - STDDEV)
    assert cli.cmd_gate(args(generation=True)) == 0
    assert harness.calls.generations == 1


def test_a_generation_drop_outside_the_band_blocks(harness: SimpleNamespace) -> None:
    """The whole point of measuring the spread: this is what it has to catch."""
    harness.report = FakeGeneration(BASELINE_ACCURACY - 5 * STDDEV)
    with pytest.raises(gating.GateError, match="accuracy"):
        cli.cmd_gate(args(generation=True))


def test_a_citation_outside_the_passages_blocks(harness: SimpleNamespace) -> None:
    # Faithfulness measured zero spread over five runs, and a citation pointing
    # at a passage the model was never given is a defect rather than sampling.
    harness.report = FakeGeneration(BASELINE_ACCURACY, faithful=0.99)
    with pytest.raises(gating.GateError, match="faithful_citations"):
        cli.cmd_gate(args(generation=True))


def test_a_gate_run_is_not_folded_into_the_measured_sample(harness: SimpleNamespace) -> None:
    # A gate run happens after a change. Averaging it into the baseline sample
    # would widen the band using the very thing the band exists to catch.
    cli.cmd_gate(args(generation=True))
    assert not harness.runs.exists()


def test_the_cost_is_confirmed_before_the_calls(harness: SimpleNamespace) -> None:
    with pytest.raises(SystemExit, match="100 calls"):
        cli.cmd_gate(args(generation=True, yes=False))
    assert harness.calls.generations == 0


def test_a_cheaper_run_is_refused_before_the_calls(harness: SimpleNamespace) -> None:
    # Finding out after 188 requests that the comparison was never valid is a
    # lesson delivered in the wrong place.
    with pytest.raises(gating.GateError, match="different measurements"):
        cli.cmd_gate(args(generation=True, limit=20))
    assert harness.calls.generations == 0


def test_a_failing_gate_exits_nonzero_with_the_numbers(
    harness: SimpleNamespace, caplog: pytest.LogCaptureFixture
) -> None:
    """CI reads the exit code and a person reads the message."""
    harness.report = FakeGeneration(BASELINE_ACCURACY - 5 * STDDEV)
    assert cli.main(["gate", "--generation", "--yes"]) == 1
    assert "accuracy" in caplog.text and "0.8532" in caplog.text
