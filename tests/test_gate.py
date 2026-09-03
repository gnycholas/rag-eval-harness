"""The gate: what blocks, what only gets reported, and what it refuses."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_eval.evals.gate import Baseline, GateError, check, enforce


def baseline(**overrides: object) -> Baseline:
    values: dict[str, object] = {
        "retrieval": {"ndcg@10": 0.71},
        "generation": {"accuracy": 0.80},
        "generation_stddev": {},
        "provider": "anthropic",
        "model": "claude-opus-5",
        "prompt_version": "v1",
        "recorded_at": "2026-09-02",
    }
    values.update(overrides)
    return Baseline(**values)  # type: ignore[arg-type]


def run(base: Baseline, retrieval: dict, generation: dict | None = None):
    return check(
        base,
        retrieval,
        generation or {},
        provider=base.provider,
        model=base.model,
    )


def test_a_drop_in_retrieval_blocks() -> None:
    findings = run(baseline(), {"ndcg@10": 0.69})
    assert findings[0].blocking
    with pytest.raises(GateError, match="ndcg@10"):
        enforce(findings)


def test_retrieval_has_no_tolerance() -> None:
    """No model is involved, so any drop is a real change."""
    assert run(baseline(), {"ndcg@10": 0.7099})[0].blocking


def test_holding_steady_passes() -> None:
    enforce(run(baseline(), {"ndcg@10": 0.71}))


def test_an_improvement_passes() -> None:
    findings = run(baseline(), {"ndcg@10": 0.75})
    enforce(findings)
    assert findings[0].delta > 0


def test_generation_does_not_block_before_variance_is_measured() -> None:
    findings = run(baseline(), {"ndcg@10": 0.71}, {"accuracy": 0.40})
    generation = [f for f in findings if f.metric == "accuracy"]
    assert generation and not generation[0].blocking
    enforce(findings)


def test_generation_blocks_outside_a_measured_band() -> None:
    base = baseline(generation_stddev={"accuracy": 0.01})
    findings = run(base, {"ndcg@10": 0.71}, {"accuracy": 0.70})
    assert any(f.metric == "accuracy" and f.blocking for f in findings)


def test_generation_inside_the_band_passes() -> None:
    base = baseline(generation_stddev={"accuracy": 0.02})
    findings = run(base, {"ndcg@10": 0.71}, {"accuracy": 0.775})
    enforce(findings)


def test_comparing_across_models_is_refused() -> None:
    """A cheap run measured against an expensive baseline is two experiments."""
    with pytest.raises(GateError, match="different configurations"):
        check(
            baseline(),
            {"ndcg@10": 0.71},
            {"accuracy": 0.80},
            provider="ollama",
            model="qwen2.5:7b",
        )


def test_a_retrieval_only_run_may_cross_providers() -> None:
    """Retrieval never touches the model, so the provider is irrelevant there."""
    check(baseline(), {"ndcg@10": 0.71}, {}, provider="ollama", model="qwen2.5:7b")


def test_the_message_carries_the_numbers() -> None:
    with pytest.raises(GateError) as err:
        enforce(run(baseline(), {"ndcg@10": 0.60}))
    message = str(err.value)
    assert "0.6000" in message and "0.7100" in message and "-0.1100" in message


def test_a_baseline_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    baseline().save(path)
    assert Baseline.load(path) == baseline()


def test_a_missing_baseline_is_none(tmp_path: Path) -> None:
    assert Baseline.load(tmp_path / "absent.json") is None


def test_metrics_absent_from_the_run_are_skipped() -> None:
    assert run(baseline(), {}) == []


MINILM = {"index": "hybrid", "embedding_model": "all-MiniLM-L6-v2", "rrf_k": "1"}


def test_another_embedding_model_is_refused() -> None:
    base = baseline(retrieval_config=MINILM)
    with pytest.raises(GateError, match="different configurations"):
        check(
            base,
            {"ndcg@10": 0.71},
            {},
            provider=base.provider,
            model=base.model,
            retrieval_config=MINILM | {"embedding_model": "bge-small-en-v1.5"},
        )


def test_another_fusion_constant_is_refused() -> None:
    base = baseline(retrieval_config=MINILM)
    with pytest.raises(GateError, match="rrf_k"):
        check(
            base,
            {"ndcg@10": 0.71},
            {},
            provider=base.provider,
            model=base.model,
            retrieval_config=MINILM | {"rrf_k": "60"},
        )


def test_the_same_configuration_compares() -> None:
    base = baseline(retrieval_config=MINILM)
    findings = check(
        base,
        {"ndcg@10": 0.71},
        {},
        provider=base.provider,
        model=base.model,
        retrieval_config=dict(MINILM),
    )
    assert [f.blocking for f in findings] == [False]


def test_a_baseline_recorded_before_the_field_still_compares() -> None:
    # Nothing to compare against, so it falls back to the old behaviour instead
    # of refusing every older baseline.
    assert run(baseline(), {"ndcg@10": 0.71}) != []


def test_rerunning_the_same_configuration_does_not_block() -> None:
    # The baseline file is rounded, the run is not. A difference of 1e-9 is the
    # file format, not a regression.
    base = baseline(retrieval={"ndcg@10": 0.693686})
    findings = run(base, {"ndcg@10": 0.6936859999999999})
    assert not findings[0].blocking
