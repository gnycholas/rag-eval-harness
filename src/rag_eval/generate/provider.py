"""Where completions come from.

Four implementations behind one interface: a deterministic stub for the suite
and CI, Ollama so a clone runs without a key, and Anthropic or Gemini for the
numbers that get published. Anything reported carries which one produced it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

import requests
from pydantic import BaseModel

from rag_eval.config import ANTHROPIC, DEFAULT_RPM, GOOGLE, OLLAMA, STUB, Config
from rag_eval.data.scifact import CONTRADICT, NOINFO, SUPPORT

T = TypeVar("T", bound=BaseModel)

log = logging.getLogger(__name__)

OLLAMA_TIMEOUT = 300
ANTHROPIC_MAX_TOKENS = 2048

GOOGLE_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GOOGLE_KEY_VARIABLE = "GOOGLE_GENAI_API_KEY"
GOOGLE_TIMEOUT = 180
GOOGLE_ATTEMPTS = 6

# A per minute quota asks for at most a minute. A delay far past that is the
# daily quota, and sleeping on it would hang the run until the reset instead of
# saying what happened.
MAX_RETRY_DELAY = 120

# Keys the response schema accepts. Gemini takes a subset of OpenAPI, and the
# title and default that pydantic emits are enough for a 400.
SCHEMA_KEYS = ("type", "description", "enum", "items", "properties", "required", "nullable")


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


@dataclass
class RateLimit:
    """One call every so often, because the quota is per minute and per model.

    Overshooting it does not fail fast: it earns a 429 with a retry delay of
    most of a minute, so the run goes slower than if it had waited its turn.
    """

    rpm: int = DEFAULT_RPM
    _last: float = field(default=0.0, init=False)

    def wait(self) -> None:
        if self.rpm <= 0:
            return
        gap = 60.0 / self.rpm - (time.monotonic() - self._last)
        if gap > 0:
            time.sleep(gap)
        self._last = time.monotonic()


def gemini_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Turn a pydantic schema into the subset Gemini accepts."""
    resolved = schema.model_json_schema()
    definitions = resolved.get("$defs", {})

    def convert(node: dict[str, Any]) -> dict[str, Any]:
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            node = definitions[name]

        out = {key: node[key] for key in SCHEMA_KEYS if key in node}
        if "properties" in out:
            out["properties"] = {k: convert(v) for k, v in out["properties"].items()}
        if "items" in out:
            out["items"] = convert(out["items"])
        return out

    return convert(resolved)


def quota_violation(body: dict[str, Any]) -> tuple[str, int] | None:
    """The quota the API says was exceeded, and the value it holds.

    The 429 states both, which beats guessing: the per minute allowance differs
    by model and the per day one is not published at all.
    """
    for detail in body.get("error", {}).get("details", []):
        for violation in detail.get("violations", []):
            quota_id = violation.get("quotaId", "")
            value = violation.get("quotaValue")
            if quota_id and value is not None:
                return quota_id, int(value)
    return None


def retry_delay(body: dict[str, Any]) -> float | None:
    """Seconds the API asked us to wait, when it said so."""
    for detail in body.get("error", {}).get("details", []):
        raw = detail.get("retryDelay")
        if isinstance(raw, str):
            match = re.fullmatch(r"([0-9.]+)s", raw)
            if match:
                return float(match.group(1))
    return None


@dataclass
class GeminiProvider:
    model: str
    rpm: int = DEFAULT_RPM
    name: str = GOOGLE

    def __post_init__(self) -> None:
        self._key = os.environ.get(GOOGLE_KEY_VARIABLE, "")
        if not self._key:
            raise ProviderError(f"{GOOGLE_KEY_VARIABLE} is not set")
        self._limit = RateLimit(self.rpm)

    def complete(self, system: str, prompt: str, schema: type[T]) -> tuple[T, Usage]:
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": gemini_schema(schema),
            },
        }
        body = self._post(payload)

        candidates = body.get("candidates") or []
        if not candidates:
            blocked = body.get("promptFeedback", {}).get("blockReason")
            raise ProviderError(f"no candidate returned (blockReason={blocked})")

        candidate = candidates[0]
        # SAFETY, RECITATION and the rest mean the model stopped for its own
        # reasons. Treated as an answer it would enter the metrics as a wrong
        # one and quietly depress the score.
        reason = candidate.get("finishReason")
        if reason not in (None, "STOP"):
            raise ProviderError(f"the model stopped early (finishReason={reason})")

        text = "".join(
            part["text"]
            for part in candidate.get("content", {}).get("parts", [])
            if "text" in part and not part.get("thought")
        )
        if not text.strip():
            raise ProviderError("empty response")

        usage_metadata = body.get("usageMetadata", {})
        usage = Usage(
            input_tokens=int(usage_metadata.get("promptTokenCount", 0)),
            output_tokens=int(usage_metadata.get("candidatesTokenCount", 0)),
        )
        return schema.model_validate(json.loads(text)), usage

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = GOOGLE_ENDPOINT.format(model=self.model)
        last = ""
        for attempt in range(1, GOOGLE_ATTEMPTS + 1):
            self._limit.wait()
            response = requests.post(
                url,
                headers={"x-goog-api-key": self._key, "Content-Type": "application/json"},
                json=payload,
                timeout=GOOGLE_TIMEOUT,
            )
            if response.status_code == 200:
                parsed: dict[str, Any] = response.json()
                return parsed

            body = response.json() if response.content else {}
            last = body.get("error", {}).get("message", response.text)[:200]
            if response.status_code not in (429, 500, 502, 503, 504):
                raise ProviderError(f"{self.model} returned {response.status_code}: {last}")

            # The free tier answers a burst with a 429 carrying its own delay,
            # and the busier models answer with a 503 that clears on its own.
            quota = quota_violation(body)
            if quota and "PerDay" in quota[0]:
                # Waiting this one out means waiting for the reset. Six retries
                # against it is how the run wasted its last minutes.
                raise ProviderError(
                    f"{self.model} is out of its daily free tier quota of {quota[1]} requests"
                )
            if quota and "PerMinute" in quota[0] and quota[1] < self._limit.rpm:
                log.warning(
                    "%s allows %d requests per minute, not %d; slowing down",
                    self.model,
                    quota[1],
                    self._limit.rpm,
                )
                self._limit.rpm = quota[1]

            asked = retry_delay(body)
            if asked is not None and asked > MAX_RETRY_DELAY:
                raise ProviderError(
                    f"{self.model} asked for {asked:.0f}s before the next call, which is longer "
                    f"than a retry makes sense: {last}"
                )

            delay = asked or min(2**attempt, 60) + random.uniform(0, 1)
            log.warning(
                "%s returned %d, waiting %.0fs (attempt %d/%d)",
                self.model,
                response.status_code,
                delay,
                attempt,
                GOOGLE_ATTEMPTS,
            )
            time.sleep(delay)

        raise ProviderError(f"{self.model} kept failing after {GOOGLE_ATTEMPTS} attempts: {last}")


def build_provider(cfg: Config, *, model: str | None = None) -> LlmProvider:
    """Fail here rather than 200 queries in."""
    chosen = model or cfg.resolved_model()
    if cfg.provider == STUB:
        return StubProvider()
    if cfg.provider == OLLAMA:
        return OllamaProvider(model=chosen, host=cfg.ollama_host)
    if cfg.provider == GOOGLE:
        return GeminiProvider(model=chosen, rpm=cfg.rpm)
    return AnthropicProvider(model=chosen)
