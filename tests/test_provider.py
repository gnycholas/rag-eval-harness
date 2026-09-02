"""The provider layer, on the stub."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_eval.config import ANTHROPIC, OLLAMA, STUB, Config, Paths
from rag_eval.data.scifact import VERDICTS, Document
from rag_eval.generate import provider as prov
from rag_eval.generate.schema import JudgeVerdict, Verification
from rag_eval.generate.verify import build_prompt, normalize_verdict, verify


def config(name: str) -> Config:
    return Config(
        paths=Paths(root=Path("data")), provider=name, model="", ollama_host="http://nowhere:1"
    )


def passages() -> list[Document]:
    return [Document("d1", "First", "body one"), Document("d2", "Second", "body two")]


def test_the_stub_is_deterministic() -> None:
    stub = prov.StubProvider()
    first, _ = stub.complete("sys", "prompt", Verification)
    second, _ = stub.complete("sys", "prompt", Verification)
    assert first == second


def test_different_prompts_give_different_answers() -> None:
    stub = prov.StubProvider()
    a, _ = stub.complete("sys", "one", Verification)
    b, _ = stub.complete("sys", "two", Verification)
    assert (a, b) == (a, b)  # both parsed
    assert a.rationale != b.rationale


def test_the_stub_fills_whatever_schema_it_is_given() -> None:
    stub = prov.StubProvider()
    judged, _ = stub.complete("sys", "prompt", JudgeVerdict)
    assert isinstance(judged.grounded, bool)


def test_the_stub_always_produces_a_label_in_the_set() -> None:
    stub = prov.StubProvider()
    seen = {stub.complete("s", f"p{i}", Verification)[0].verdict for i in range(30)}
    assert seen <= set(VERDICTS)


def test_the_default_provider_is_the_stub() -> None:
    assert prov.build_provider(config(STUB)).name == STUB


def test_an_unreachable_ollama_fails_at_construction() -> None:
    """Failing here beats failing two hundred queries in."""
    with pytest.raises(prov.ProviderError, match="not reachable"):
        prov.build_provider(config(OLLAMA))


def test_missing_anthropic_credentials_fail_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import anthropic

    def explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("no key")

    monkeypatch.setattr(anthropic, "Anthropic", explode)
    with pytest.raises(prov.ProviderError, match="no Anthropic credentials"):
        prov.build_provider(config(ANTHROPIC))


def test_usage_adds_up() -> None:
    total = prov.Usage(10, 5) + prov.Usage(1, 2)
    assert (total.input_tokens, total.output_tokens) == (11, 7)


def test_the_prompt_numbers_the_passages() -> None:
    prompt = build_prompt("a claim", passages())
    assert "[1] First" in prompt
    assert "[2] Second" in prompt
    assert "Claim: a claim" in prompt


def test_a_verdict_outside_the_label_set_is_refused() -> None:
    """Quietly mapping it to NOINFO would make accuracy describe a fallback."""
    with pytest.raises(ValueError, match="outside the label set"):
        normalize_verdict("MAYBE")


def test_common_spellings_are_accepted() -> None:
    assert normalize_verdict(" support ") == "SUPPORT"
    assert normalize_verdict("NO_INFO") == "NOINFO"


def test_verification_records_what_it_cited() -> None:
    result = verify(prov.StubProvider(), "q1", "a claim", passages())
    assert result.verdict in VERDICTS
    assert result.retrieved_ids == ["d1", "d2"]
    assert result.cited_doc_ids == ["d1"]
    assert result.faithful_citations


def test_a_citation_out_of_range_is_recorded_not_dropped() -> None:
    class Wild(prov.StubProvider):
        def complete(self, system: str, prompt: str, schema: type):  # type: ignore[override]
            parsed, usage = super().complete(system, prompt, schema)
            return parsed.model_copy(update={"cited_indices": [99]}), usage

    result = verify(Wild(), "q1", "a claim", passages())
    assert not result.faithful_citations
    assert result.cited_indices == [99]


def test_the_prompt_version_travels_with_the_result() -> None:
    assert verify(prov.StubProvider(), "q1", "c", passages()).prompt_version
