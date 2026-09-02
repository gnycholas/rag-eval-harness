"""Configuration from the environment, with defaults that work out of the box."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

STUB = "stub"
OLLAMA = "ollama"
ANTHROPIC = "anthropic"

# The skill's default. An eval run over the test split is a few dollars, which
# the README states rather than hides.
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"


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


@dataclass(frozen=True)
class Config:
    paths: Paths
    provider: str
    model: str
    ollama_host: str

    def resolved_model(self) -> str:
        if self.model:
            return self.model
        return DEFAULT_OLLAMA_MODEL if self.provider == OLLAMA else DEFAULT_ANTHROPIC_MODEL


@lru_cache(maxsize=1)
def load_config() -> Config:
    provider = os.environ.get("RAG_PROVIDER", STUB)
    if provider not in (STUB, OLLAMA, ANTHROPIC):
        raise ValueError(f"unknown provider: {provider!r}")

    return Config(
        paths=Paths(root=Path(os.environ.get("RAG_DATA_DIR", "data"))),
        provider=provider,
        model=os.environ.get("RAG_MODEL", ""),
        ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
    )
