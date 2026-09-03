"""Configuration is environment driven, with no machine specific defaults."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_eval.config import ANTHROPIC, GOOGLE, OLLAMA, STUB, Config, Paths


def config(provider: str = STUB, model: str = "", judge_model: str = "") -> Config:
    return Config(
        paths=Paths(root=Path("data")),
        provider=provider,
        model=model,
        judge_model=judge_model,
        ollama_host="http://localhost:11434",
        rpm=15,
    )


def test_paths_hang_off_one_root() -> None:
    paths = Paths(root=Path("/tmp/x"))
    assert paths.dataset == Path("/tmp/x/scifact")
    assert paths.index == Path("/tmp/x/index")


def test_each_provider_has_its_own_default_model() -> None:
    assert config(ANTHROPIC).resolved_model().startswith("claude-")
    assert config(OLLAMA).resolved_model() == "qwen2.5:7b"
    assert config(GOOGLE).resolved_model().startswith("gemini-")


def test_the_judge_falls_back_to_the_model_under_test() -> None:
    assert config(GOOGLE).resolved_judge_model() == config(GOOGLE).resolved_model()


def test_a_judge_model_can_differ_from_the_model_under_test() -> None:
    cfg = config(GOOGLE, model="gemini-a", judge_model="gemini-b")
    assert (cfg.resolved_model(), cfg.resolved_judge_model()) == ("gemini-a", "gemini-b")


def test_an_explicit_model_wins() -> None:
    assert config(ANTHROPIC, model="claude-haiku-4-5").resolved_model() == "claude-haiku-4-5"


def test_an_unknown_provider_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_eval.config import load_config

    load_config.cache_clear()
    monkeypatch.setenv("RAG_PROVIDER", "gpt")
    with pytest.raises(ValueError, match="unknown provider"):
        load_config()
    load_config.cache_clear()


def test_the_default_provider_needs_no_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clone with no keys still runs the test suite."""
    from rag_eval.config import load_config

    load_config.cache_clear()
    monkeypatch.delenv("RAG_PROVIDER", raising=False)
    assert load_config().provider == STUB
    load_config.cache_clear()
