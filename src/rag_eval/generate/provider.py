"""Where completions come from.

Three implementations behind one interface: a deterministic stub for the suite
and CI, Ollama so a clone runs without a key, and Anthropic for the numbers
that get published. Anything reported carries which one produced it.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Protocol, TypeVar

import requests
from pydantic import BaseModel

from rag_eval.config import ANTHROPIC, OLLAMA, STUB, Config
from rag_eval.data.scifact import CONTRADICT, NOINFO, SUPPORT

T = TypeVar("T", bound=BaseModel)

log = logging.getLogger(__name__)

OLLAMA_TIMEOUT = 300
ANTHROPIC_MAX_TOKENS = 2048


class ProviderError(RuntimeError):
    """The provider cannot run."""


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )


class LlmProvider(Protocol):
    name: str
    model: str

    def complete(self, system: str, prompt: str, schema: type[T]) -> tuple[T, Usage]: ...


@dataclass
class StubProvider:
    """Deterministic, offline, and never right by luck.

    The verdict comes from a hash of the prompt. That is meaningless as an
    answer and exactly right for a test: it is stable, needs no network, and
    nobody could mistake its accuracy for a result.
    """

    model: str = "stub"
    name: str = STUB

    def complete(self, system: str, prompt: str, schema: type[T]) -> tuple[T, Usage]:
        digest = hashlib.sha256(f"{system}\n{prompt}".encode()).hexdigest()
        verdict = (SUPPORT, CONTRADICT, NOINFO)[int(digest[:8], 16) % 3]
        payload = {
            "verdict": verdict,
            "rationale": f"stub response {digest[:8]}",
            "cited_indices": [1],
            "grounded": digest[8] in "02468ace",
            "note": "stub",
        }
        fields = {k: v for k, v in payload.items() if k in schema.model_fields}
        return schema.model_validate(fields), Usage()


@dataclass
class OllamaProvider:
    model: str
    host: str
    name: str = OLLAMA

    def __post_init__(self) -> None:
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=10)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ProviderError(f"ollama is not reachable at {self.host}: {exc}") from exc

        available = {m["name"] for m in response.json().get("models", [])}
        if self.model not in available:
            raise ProviderError(
                f"model {self.model!r} is not pulled. Available: {', '.join(sorted(available))}"
            )

    def complete(self, system: str, prompt: str, schema: type[T]) -> tuple[T, Usage]:
        response = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "format": schema.model_json_schema(),
                "stream": False,
            },
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        body = response.json()
        parsed = schema.model_validate(json.loads(body["message"]["content"]))
        return parsed, Usage(
            input_tokens=int(body.get("prompt_eval_count", 0)),
            output_tokens=int(body.get("eval_count", 0)),
        )


@dataclass
class AnthropicProvider:
    model: str
    name: str = ANTHROPIC

    def __post_init__(self) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ProviderError("the anthropic package is not installed") from exc

        try:
            self._client = anthropic.Anthropic()
        except Exception as exc:
            raise ProviderError(f"no Anthropic credentials available: {exc}") from exc

    def complete(self, system: str, prompt: str, schema: type[T]) -> tuple[T, Usage]:
        response = self._client.messages.parse(
            model=self.model,
            max_tokens=ANTHROPIC_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
        )

        # A refusal comes back as a normal 200. Left unchecked it would enter
        # the metrics as a wrong answer and quietly depress the score.
        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None)
            raise ProviderError(f"the model declined to answer ({category})")

        parsed = response.parsed_output
        if parsed is None:
            raise ProviderError(f"no parseable output (stop_reason={response.stop_reason})")

        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return parsed, usage


def build_provider(cfg: Config) -> LlmProvider:
    """Fail here rather than 200 queries in."""
    model = cfg.resolved_model()
    if cfg.provider == STUB:
        return StubProvider()
    if cfg.provider == OLLAMA:
        return OllamaProvider(model=model, host=cfg.ollama_host)
    return AnthropicProvider(model=model)
