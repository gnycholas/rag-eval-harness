"""Keeping the individual runs a variance estimate is built from.

A spread computed inside one process is a spread that cannot survive an
interruption, and on a quota that ends mid measurement it will meet one. Each
run is appended here with what produced it, and only runs that match on all of
that are averaged together.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

RUNS_PATH = Path("evals/runs.jsonl")


@dataclass(frozen=True)
class Run:
    provider: str
    model: str
    prompt_version: str
    claims: int
    scored: int
    accuracy: float
    faithful_citations: float
    recorded_at: str = ""

    @property
    def key(self) -> tuple[str, str, str, int]:
        """What has to match for two runs to be repetitions of one measurement.

        The claims attempted, not the ones scored: a run where the model
        produced one unusable answer is still a repetition of the same
        measurement, and keying on the usable count would split it off.
        """
        return self.provider, self.model, self.prompt_version, self.claims


def append(run: Run, path: Path | None = None) -> None:
    path = path or RUNS_PATH
    stamped = Run(**{**asdict(run), "recorded_at": datetime.now(UTC).isoformat(timespec="seconds")})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(stamped), sort_keys=True) + "\n")


def load(path: Path | None = None) -> list[Run]:
    path = path or RUNS_PATH
    if not path.exists():
        return []
    return [
        Run(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line
    ]


def matching(runs: list[Run], key: tuple[str, str, str, int]) -> list[Run]:
    return [run for run in runs if run.key == key]
