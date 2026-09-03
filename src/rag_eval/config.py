"""Configuration from the environment, with defaults that work out of the box."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

STUB = "stub"
OLLAMA = "ollama"
ANTHROPIC = "anthropic"
GOOGLE = "google"

PROVIDERS = (STUB, OLLAMA, ANTHROPIC, GOOGLE)

# The skill's default. An eval run over the test split is a few dollars, which
# the README states rather than hides.
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
DEFAULT_GOOGLE_MODEL = "gemini-3.1-flash-lite"
# A judge should not be the model it is grading. This one is also the only
# choice on the free tier that can see the whole set: gemini-3-flash-preview
# allows twenty requests a day, which does not reach 188 claims.
DEFAULT_GOOGLE_JUDGE_MODEL = "gemma-4-31b-it"

# Free tier quota, per model and per minute, measured against the API rather
# than read off a page: the 429 names it as 15 for this model.
DEFAULT_RPM = 15


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def dataset(self) -> Path:
        return self.root / "scifact"

    @property
    def index(self) -> Path:
        return self.root / "index"

    @property
    def runs(self) -> Path:
        return self.root / "runs"


DEFAULT_MODELS = {
    STUB: STUB,
    OLLAMA: DEFAULT_OLLAMA_MODEL,
    ANTHROPIC: DEFAULT_ANTHROPIC_MODEL,
    GOOGLE: DEFAULT_GOOGLE_MODEL,
}

DEFAULT_JUDGE_MODELS = {GOOGLE: DEFAULT_GOOGLE_JUDGE_MODEL}


@dataclass(frozen=True)
class Config:
    paths: Paths
    provider: str
    model: str
    judge_model: str
    ollama_host: str
    rpm: int

    def resolved_model(self) -> str:
        return self.model or DEFAULT_MODELS[self.provider]

    def resolved_judge_model(self) -> str:
        """A judge should not be grading its own answers.

        A model asked to score what it just produced rates it generously, so
        the judge runs on a second model where there is one to run on. The
        stub and Ollama fall back to the model under test, since there is
        nothing to gain from a second local model.
        """
        if self.judge_model:
            return self.judge_model
        return DEFAULT_JUDGE_MODELS.get(self.provider) or self.resolved_model()


ENV_FILE = Path(".env")


def load_env(path: Path = ENV_FILE) -> None:
    """Read KEY=value lines from .env, without overriding the real environment."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


@lru_cache(maxsize=1)
def load_config() -> Config:
    load_env()
    provider = os.environ.get("RAG_PROVIDER", STUB)
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider: {provider!r}")

    return Config(
        paths=Paths(root=Path(os.environ.get("RAG_DATA_DIR", "data"))),
        provider=provider,
        model=os.environ.get("RAG_MODEL", ""),
        judge_model=os.environ.get("RAG_JUDGE_MODEL", ""),
        ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        rpm=int(os.environ.get("RAG_RPM", DEFAULT_RPM)),
    )
