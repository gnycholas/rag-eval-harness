"""Comparing a run against the recorded baseline.

Two layers with different rules, because their nature differs. Retrieval is
deterministic, so any drop is a real change. Generation is not: sampling
controls were removed from the current models, so two identical runs disagree.
A band picked by feel there fails on noise, everyone learns to rerun until it
passes, and from then on the gate protects nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BASELINE_PATH = Path("evals/baseline.json")

# Retrieval involves no model. A drop is a change, not variance.
RETRIEVAL_TOLERANCE = 0.0

# The baseline file stores rounded numbers. Comparing a full precision run
# against them makes a rerun of the very same configuration fail by 1e-9, which
# is how a zero tolerance gate ends up being switched off by whoever hits it.
PRECISION = 6


class GateError(RuntimeError):
    """A metric fell below the baseline."""


@dataclass(frozen=True)
class Baseline:
    retrieval: dict[str, float]
    generation: dict[str, float]
    provider: str
    model: str
    prompt_version: str
    recorded_at: str
    # Retrieval involves no language model, so what pins those numbers is the
    # embedding model and the fusion constant, not the provider.
    retrieval_config: dict[str, str] = field(default_factory=dict)
    # Standard deviation across repeated runs. Absent until it has been
    # measured, and generation does not gate while it is absent.
    generation_stddev: dict[str, float] = field(default_factory=dict)
    # How many claims each of those runs covered. Accuracy over a cheap subset
    # is not a lower reading of the same measurement, it is a different one.
    generation_claims: int = 0

    @classmethod
    def load(cls, path: Path = BASELINE_PATH) -> Baseline | None:
        if not path.exists():
            return None
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            retrieval=payload.get("retrieval", {}),
            generation=payload.get("generation", {}),
            provider=payload.get("provider", ""),
            model=payload.get("model", ""),
            prompt_version=payload.get("prompt_version", ""),
            recorded_at=payload.get("recorded_at", ""),
            generation_stddev=payload.get("generation_stddev", {}),
            generation_claims=payload.get("generation_claims", 0),
            retrieval_config=payload.get("retrieval_config", {}),
        )

    def save(self, path: Path = BASELINE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "retrieval": self.retrieval,
            "retrieval_config": self.retrieval_config,
            "generation": self.generation,
            "generation_stddev": self.generation_stddev,
            "generation_claims": self.generation_claims,
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "recorded_at": self.recorded_at,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class Finding:
    metric: str
    observed: float
    baseline: float
    tolerance: float
    blocking: bool

    @property
    def delta(self) -> float:
        return self.observed - self.baseline

    def __str__(self) -> str:
        verdict = "FAILED" if self.blocking else "reported"
        return (
            f"[{verdict}] {self.metric}: {self.observed:.4f} against baseline "
            f"{self.baseline:.4f} (delta {self.delta:+.4f}, tolerance {self.tolerance:.4f})"
        )


def _describe(config: dict[str, str]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(config.items()))


def refuse_mismatched_claims(baseline: Baseline, claims: int) -> None:
    """A run over fewer claims is a different measurement, not a cheaper one.

    Raised before the calls are spent as well as inside check(), because
    discovering it after 188 requests is a lesson in the wrong place.
    """
    if baseline.generation_claims and claims and claims != baseline.generation_claims:
        raise GateError(
            f"baseline generation was measured over {baseline.generation_claims} claims and this "
            f"run covers {claims}; these are different measurements, not a regression"
        )


def check(
    baseline: Baseline,
    retrieval: dict[str, float],
    generation: dict[str, float],
    *,
    provider: str,
    model: str,
    retrieval_config: dict[str, str] | None = None,
    generation_claims: int = 0,
) -> list[Finding]:
    """Compare a run, refusing to compare across configurations.

    A baseline recorded on one model and a run on another are two different
    experiments, and treating one as a regression of the other is the most
    tempting way to get a wrong answer here.
    """
    if generation and (provider != baseline.provider or model != baseline.model):
        raise GateError(
            f"baseline was recorded on {baseline.provider}/{baseline.model} and this run is "
            f"{provider}/{model}; these are different configurations, not a regression"
        )
    if generation:
        refuse_mismatched_claims(baseline, generation_claims)

    observed_config = retrieval_config or {}
    # Baselines recorded before this field existed carry no config, and there is
    # nothing to compare them against.
    if retrieval and baseline.retrieval_config and observed_config != baseline.retrieval_config:
        raise GateError(
            f"baseline was recorded on {_describe(baseline.retrieval_config)} and this run is "
            f"{_describe(observed_config)}; these are different configurations, not a regression"
        )

    findings = []
    for metric, recorded in baseline.retrieval.items():
        if metric in retrieval:
            observed = round(retrieval[metric], PRECISION)
            findings.append(
                Finding(
                    metric=metric,
                    observed=observed,
                    baseline=recorded,
                    tolerance=RETRIEVAL_TOLERANCE,
                    blocking=observed < recorded - RETRIEVAL_TOLERANCE,
                )
            )

    for metric, recorded in baseline.generation.items():
        if metric not in generation:
            continue
        stddev = baseline.generation_stddev.get(metric)
        observed = round(generation[metric], PRECISION)
        if stddev is None:
            # Variance has not been measured, so there is no honest band. Report
            # and let it through rather than block on a guess.
            findings.append(Finding(metric, observed, recorded, 0.0, blocking=False))
        else:
            band = 2 * stddev
            findings.append(
                Finding(metric, observed, recorded, band, blocking=observed < recorded - band)
            )

    return findings


def enforce(findings: list[Finding]) -> None:
    blocking = [f for f in findings if f.blocking]
    if blocking:
        raise GateError("regression:\n" + "\n".join(f"  {f}" for f in blocking))
