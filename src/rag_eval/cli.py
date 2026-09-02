"""Command line entry points."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime

from rag_eval.config import Config, load_config
from rag_eval.data import scifact
from rag_eval.evals import gate as gating
from rag_eval.evals.pipeline import run as run_generation
from rag_eval.evals.runner import evaluate
from rag_eval.generate.provider import build_provider
from rag_eval.generate.schema import PROMPT_VERSION
from rag_eval.index.base import Index
from rag_eval.index.dense import DenseIndex
from rag_eval.index.hybrid import HybridIndex
from rag_eval.index.sparse import Bm25Index

log = logging.getLogger("rag_eval")


def cmd_data(args: argparse.Namespace) -> int:
    cfg = load_config()
    target = cfg.paths.dataset
    scifact.download(target, force=args.force)

    for split in ("train", "test"):
        dataset = scifact.load(target, split=split)
        log.info(
            "%s: %d documents, %d queries, %d judged, %d with a human verdict",
            split,
            len(dataset.documents),
            len(dataset.queries),
            len(dataset.judged()),
            len(dataset.with_verdict()),
        )
    return 0


CONFIGURATIONS = ("bm25", "dense", "hybrid")


def _sparse(dataset: scifact.Dataset) -> Bm25Index:
    index = Bm25Index()
    index.build(dataset.documents)
    return index


def _dense(dataset: scifact.Dataset, cfg: Config) -> DenseIndex:
    """Embedding the corpus takes minutes, so it is cached on disk."""
    cached = cfg.paths.index / f"dense-{dataset.documents[0].doc_id}.npz"
    if cached.exists():
        return DenseIndex.load(cached)

    index = DenseIndex()
    index.build(dataset.documents)
    index.save(cached)
    return index


def build_index(name: str, dataset: scifact.Dataset, cfg: Config) -> Index:
    """Build only what was asked for.

    Asking for bm25 and paying for five thousand embeddings anyway is the kind
    of waste that stops people from running the thing.
    """
    if name == "bm25":
        return _sparse(dataset)
    if name == "dense":
        return _dense(dataset, cfg)
    if name == "hybrid":
        return HybridIndex(sparse=_sparse(dataset), dense=_dense(dataset, cfg))
    raise ValueError(f"unknown configuration: {name!r}")


def cmd_retrieval(args: argparse.Namespace) -> int:
    cfg = load_config()
    dataset = scifact.load(cfg.paths.dataset, split=args.split)

    for name in args.configs or list(CONFIGURATIONS):
        report = evaluate(build_index(name, dataset, cfg), dataset)
        log.info("%s over %d queries", name, report.queries)
        print(f"\n### {name} ({args.split})\n{report.table()}")
    return 0


def cmd_generation(args: argparse.Namespace) -> int:
    cfg = load_config()
    dataset = scifact.load(cfg.paths.dataset, split=args.split)
    provider = build_provider(cfg)

    queries = len(dataset.with_verdict()) if not args.limit else args.limit
    calls = queries * (2 if args.judge else 1)
    if provider.name != "stub" and not args.yes:
        log.warning(
            "about to make %d calls to %s/%s. Re-run with --yes to proceed.",
            calls,
            provider.name,
            provider.model,
        )
        return 1

    judge = build_provider(cfg) if args.judge else None
    report = run_generation(
        build_index("hybrid", dataset, cfg),
        dataset,
        provider,
        top_k=args.top_k,
        judge=judge,
        limit=args.limit,
    )

    agreement = report.model_agreement
    print(f"\n### verdict against human labels ({provider.name}/{provider.model})\n")
    print(f"accuracy   {agreement.accuracy}  n={agreement.accuracy.n}")
    print(f"macro F1   {agreement.macro_f1:.4f}")
    print(f"kappa      {agreement.kappa:.4f} ({agreement.kappa_reading})")
    print(f"faithful citations {report.faithful_rate:.4f}")
    print(f"failures   {report.failures}")
    print(f"\n{agreement.confusion_table()}")

    if report.judge_agreement:
        judged = report.judge_agreement
        print("\n### the judge, measured against the same human labels\n")
        print(f"accuracy   {judged.accuracy}")
        print(f"kappa      {judged.kappa:.4f} ({judged.kappa_reading})")
        print("\nJudge scores are only worth what this agreement is worth.")

    print(f"\ntokens in={report.usage.input_tokens} out={report.usage.output_tokens}")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    baseline = gating.Baseline.load()
    if baseline is None:
        log.error("no baseline recorded at %s", gating.BASELINE_PATH)
        return 1

    cfg = load_config()
    dataset = scifact.load(cfg.paths.dataset, split="test")
    report = evaluate(build_index("hybrid", dataset, cfg), dataset)
    retrieval = {name: value.mean for name, value in report.metrics.items()}

    findings = gating.check(
        baseline, retrieval, {}, provider=cfg.provider, model=cfg.resolved_model()
    )
    for finding in findings:
        log.info("%s", finding)
    gating.enforce(findings)
    log.info("no regression against the baseline of %s", baseline.recorded_at)
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    """Record the current numbers as the line the gate compares against."""
    cfg = load_config()
    dataset = scifact.load(cfg.paths.dataset, split="test")
    report = evaluate(build_index(args.config, dataset, cfg), dataset)

    existing = gating.Baseline.load()
    baseline = gating.Baseline(
        retrieval={name: round(value.mean, 6) for name, value in report.metrics.items()},
        generation=existing.generation if existing else {},
        generation_stddev=existing.generation_stddev if existing else {},
        provider=cfg.provider,
        model=cfg.resolved_model(),
        prompt_version=PROMPT_VERSION,
        recorded_at=datetime.now(UTC).date().isoformat(),
    )
    baseline.save()

    log.info("baseline written to %s over %d queries", gating.BASELINE_PATH, report.queries)
    for name, value in baseline.retrieval.items():
        log.info("  %s %.4f", name, value)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-eval")
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="download and verify the dataset")
    data.add_argument("--force", action="store_true", help="ignore the cached copy")

    retrieval = sub.add_parser("retrieval", help="score the indexes against the qrels")
    retrieval.add_argument("--split", default="test", choices=["train", "test"])
    retrieval.add_argument("--configs", nargs="*", help="bm25, dense, hybrid")

    generation = sub.add_parser("generation", help="verify claims and score against human labels")
    generation.add_argument("--split", default="test", choices=["train", "test"])
    generation.add_argument("--top-k", type=int, default=5)
    generation.add_argument("--judge", action="store_true", help="also run and measure a judge")
    generation.add_argument("--limit", type=int, help="stop after this many queries")
    generation.add_argument("--yes", action="store_true", help="skip the cost confirmation")

    baseline = sub.add_parser("baseline", help="record the current numbers as the baseline")
    baseline.add_argument("--config", default="hybrid", choices=CONFIGURATIONS)

    sub.add_parser("gate", help="compare a run against the recorded baseline")

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    handlers = {
        "data": cmd_data,
        "retrieval": cmd_retrieval,
        "generation": cmd_generation,
        "baseline": cmd_baseline,
        "gate": cmd_gate,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
