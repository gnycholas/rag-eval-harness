"""The provider layer, on the stub."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_eval.config import ANTHROPIC, OLLAMA, STUB, Config, Paths
from rag_eval.data.scifact import VERDICTS, Document
from rag_eval.generate import provider as prov
from rag_eval.generate.schema import JudgeVerdict, Verification
from rag_eval.generate.verify import build_prompt, normalize_verdict, verify


def config(name: str, model: str = "") -> Config:
    return Config(
        paths=Paths(root=Path("data")),
        provider=name,
        model=model,
        judge_model="",
        ollama_host="http://nowhere:1",
        rpm=0,
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


class FakeResponse:
    def __init__(self, status: int, payload: dict) -> None:
        self.status_code = status
        self._payload = payload
        self.content = b"x"
        self.text = str(payload)

    def json(self) -> dict:
        return self._payload


def ok_body(text: str = '{"verdict": "SUPPORT", "rationale": "r", "cited_indices": [1]}') -> dict:
    return {
        "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": text}]}}],
        "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 7},
    }


def gemini(monkeypatch: pytest.MonkeyPatch, responses: list[FakeResponse]) -> prov.GeminiProvider:
    monkeypatch.setenv(prov.GOOGLE_KEY_VARIABLE, "test-key")
    monkeypatch.setattr(prov.time, "sleep", lambda _: None)
    queue = list(responses)
    monkeypatch.setattr(prov.requests, "post", lambda *a, **k: queue.pop(0))
    return prov.GeminiProvider(model="gemini-test", rpm=0)


def test_the_response_schema_drops_what_gemini_rejects() -> None:
    schema = prov.gemini_schema(Verification)
    assert set(schema) <= set(prov.SCHEMA_KEYS)
    assert "title" not in schema["properties"]["verdict"]
    assert schema["properties"]["cited_indices"]["items"]["type"] == "integer"
    assert "verdict" in schema["required"]


def test_a_missing_key_fails_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(prov.GOOGLE_KEY_VARIABLE, raising=False)
    with pytest.raises(prov.ProviderError, match=prov.GOOGLE_KEY_VARIABLE):
        prov.GeminiProvider(model="gemini-test")


def test_a_completion_parses_and_carries_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = gemini(monkeypatch, [FakeResponse(200, ok_body())])
    parsed, usage = provider.complete("sys", "prompt", Verification)
    assert parsed.verdict == "SUPPORT"
    assert (usage.input_tokens, usage.output_tokens) == (11, 7)


def test_thoughts_are_not_part_of_the_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    body = ok_body()
    body["candidates"][0]["content"]["parts"].insert(0, {"text": "thinking", "thought": True})
    provider = gemini(monkeypatch, [FakeResponse(200, body)])
    parsed, _ = provider.complete("sys", "prompt", Verification)
    assert parsed.verdict == "SUPPORT"


def test_stopping_for_its_own_reasons_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {"candidates": [{"finishReason": "SAFETY", "content": {"parts": []}}]}
    provider = gemini(monkeypatch, [FakeResponse(200, body)])
    with pytest.raises(prov.ProviderError, match="SAFETY"):
        provider.complete("sys", "prompt", Verification)


def test_a_blocked_prompt_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = gemini(
        monkeypatch, [FakeResponse(200, {"promptFeedback": {"blockReason": "OTHER"}})]
    )
    with pytest.raises(prov.ProviderError, match="OTHER"):
        provider.complete("sys", "prompt", Verification)


def test_a_quota_error_is_retried_with_the_delay_it_asked_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quota = {
        "error": {
            "message": "quota",
            "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "47s"}],
        }
    }
    slept: list[float] = []
    monkeypatch.setenv(prov.GOOGLE_KEY_VARIABLE, "test-key")
    monkeypatch.setattr(prov.time, "sleep", slept.append)
    queue = [FakeResponse(429, quota), FakeResponse(200, ok_body())]
    monkeypatch.setattr(prov.requests, "post", lambda *a, **k: queue.pop(0))

    parsed, _ = prov.GeminiProvider(model="gemini-test", rpm=0).complete("s", "p", Verification)
    assert parsed.verdict == "SUPPORT"
    assert slept == [47.0]


def test_a_busy_model_is_retried_without_a_stated_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = gemini(
        monkeypatch,
        [FakeResponse(503, {"error": {"message": "high demand"}}), FakeResponse(200, ok_body())],
    )
    parsed, _ = provider.complete("s", "p", Verification)
    assert parsed.verdict == "SUPPORT"


def test_a_rejected_request_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = gemini(monkeypatch, [FakeResponse(400, {"error": {"message": "bad schema"}})])
    with pytest.raises(prov.ProviderError, match="bad schema"):
        provider.complete("s", "p", Verification)


def test_giving_up_says_how_many_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    busy = [FakeResponse(503, {"error": {"message": "busy"}}) for _ in range(prov.GOOGLE_ATTEMPTS)]
    provider = gemini(monkeypatch, busy)
    with pytest.raises(prov.ProviderError, match=str(prov.GOOGLE_ATTEMPTS)):
        provider.complete("s", "p", Verification)


def test_the_rate_limit_spaces_calls_out(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(prov.time, "sleep", slept.append)
    limit = prov.RateLimit(rpm=60)
    limit.wait()
    limit.wait()
    assert slept and 0.9 < slept[-1] <= 1.0


def test_no_rate_limit_when_it_is_switched_off() -> None:
    prov.RateLimit(rpm=0).wait()


def test_the_judge_can_run_on_another_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(prov.GOOGLE_KEY_VARIABLE, "test-key")
    built = prov.build_provider(config(prov.GOOGLE, model="gemini-a"), model="gemini-b")
    assert built.model == "gemini-b"


def test_a_daily_quota_is_not_slept_on(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {
        "error": {
            "message": "quota",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.RetryInfo",
                    "retryDelay": f"{prov.MAX_RETRY_DELAY + 1}s",
                }
            ],
        }
    }
    provider = gemini(monkeypatch, [FakeResponse(429, body)])
    with pytest.raises(prov.ProviderError, match="daily quota"):
        provider.complete("s", "p", Verification)
