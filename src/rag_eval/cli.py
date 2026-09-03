"""Command line entry points."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from statistics import mean, stdev

from rag_eval.config import Config, load_config
from rag_eval.data import scifact
from rag_eval.evals import ablation
from rag_eval.evals import gate as gating
from rag_eval.evals import runs as run_store
from rag_eval.evals.pipeline import GenerationReport
from rag_eval.evals.pipeline import run as run_generation
from rag_eval.evals.runner import evaluate
from rag_eval.generate.provider import build_provider
from rag_eval.generate.schema import PROMPT_VERSION
from rag_eval.index.base import Index
from rag_eval.index.dense import DEFAULT_MODEL, DenseIndex
from rag_eval.index.hybrid import DEFAULT_DEPTH, DEFAULT_K, HybridIndex
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


def _dense(dataset: scifact.Dataset, cfg: Config, model: str | None = None) -> DenseIndex:
    """Embedding the corpus takes minutes, so it is cached on disk.

    The model name is part of the cache key. Without it, switching models
    reloads the previous model's vectors and queries them with the new one's
    embeddings, which returns confident nonsense and raises nothing.
    """
    name = model or DEFAULT_MODEL
    slug = name.replace("/", "_")
    cached = cfg.paths.index / f"dense-{slug}.npz"
    if cached.exists():
        return DenseIndex.load(cached, expect_model=name)

    index = DenseIndex(model_name=name)
    index.build(dataset.documents)
    index.save(cached)
    return index


def build_index(
    name: str, dataset: scifact.Dataset, cfg: Config, model: str | None = None
) -> Index:
    """Build only what was asked for.

    Asking for bm25 and paying for five thousand embeddings anyway is the kind
    of waste that stops people from running the thing.
    """
    if name == "bm25":
        return _sparse(dataset)
    if name == "dense":
        return _dense(dataset, cfg, model)
    if name == "hybrid":
        return HybridIndex(sparse=_sparse(dataset), dense=_dense(dataset, cfg, model))
    raise ValueError(f"unknown configuration: {name!r}")


def retrieval_config(name: str, model: str | None) -> dict[str, str]:
    """What pins a retrieval number: the index, the embeddings and the fusion
    constant. No language model is involved."""
    return {
        "index": name,
        "embedding_model": model or DEFAULT_MODEL,
        "rrf_k": str(DEFAULT_K),
    }


def cmd_retrieval(args: argparse.Namespace) -> int:
    cfg = load_config()
    dataset = scifact.load(cfg.paths.dataset, split=args.split)

    for name in args.configs or list(CONFIGURATIONS):
        report = evaluate(build_index(name, dataset, cfg, args.model), dataset)
        log.info("%s over %d queries", name, report.queries)
        print(f"\n### {name} ({args.split})\n{report.table()}")
    return 0


DEFAULT_RRF_K = (0, 1, 2, 5, 10, 60)
DEFAULT_DEPTHS = (10, 20, 50, 100, 200)


def cmd_ablation(args: argparse.Namespace) -> int:
    """Score every configuration over the same queries and print the table."""
    cfg = load_config()
    dataset = scifact.load(cfg.paths.dataset, split=args.split)
    if args.split == "test":
        log.warning(
            "the test split is for reporting a configuration that was already chosen; "
            "choosing one by looking at these numbers is calibrating on the answer"
        )

    sparse = _sparse(dataset)
    dense = _dense(dataset, cfg, args.model)

    variants = [
        ablation.Variant("bm25", lambda: sparse),
        ablation.Variant("dense", lambda: dense),
    ]
    for k in args.rrf_k:
        variants.append(_hybrid_variant(sparse, dense, k=k, depth=DEFAULT_DEPTH))
    for depth in args.depth:
        variants.append(_hybrid_variant(sparse, dense, k=DEFAULT_K, depth=depth))

    # The default k and the default depth appear in both sweeps.
    unique: dict[str, ablation.Variant] = {v.label: v for v in variants}

    report = ablation.run(dataset, list(unique.values()), split=args.split)
    print(f"\n### ablation ({args.split}, {report.queries} queries)\n")
    print(report.table())
    return 0


def _hybrid_variant(sparse: Index, dense: Index, *, k: int, depth: int) -> ablation.Variant:
    return ablation.Variant(
        f"hybrid k={k} depth={depth}",
        lambda: HybridIndex(sparse=sparse, dense=dense, k=k, depth=depth),
    )


def cmd_generation(args: argparse.Namespace) -> int:
    cfg = load_config()
    dataset = scifact.load(cfg.paths.dataset, split=args.split)
    provider = build_provider(cfg)

    queries = len(dataset.with_verdict()) if not args.limit else args.limit
    if provider.name != "stub" and not args.yes:
        log.warning(
            "about to make %d calls to %s/%s%s. Re-run with --yes to proceed.",
            queries,
            provider.name,
            provider.model,
            f" and {queries} to {cfg.resolved_judge_model()}" if args.judge else "",
        )
        return 1

    judge = build_provider(cfg, model=cfg.resolved_judge_model()) if args.judge else None
    report = run_generation(
        build_index("hybrid", dataset, cfg),
        dataset,
        provider,
        top_k=args.top_k,
        judge=judge,
        limit=args.limit,
    )

    if provider.name != "stub":
        record_run(report, queries)

    agreement = report.model_agreement
    print(f"\n### verdict against human labels ({provider.name}/{provider.model})\n")
    print(f"accuracy   {agreement.accuracy}  n={agreement.accuracy.n}")
    print(f"macro F1   {agreement.macro_f1:.4f}")
    print(f"kappa      {agreement.kappa:.4f} ({agreement.kappa_reading})")
    print(f"faithful citations {report.faithful_rate:.4f}")
    print(f"failures   {report.failures}")
    print(f"unusable   {report.unusable}")
    print(f"\n{agreement.confusion_table()}")

    if report.judge_agreement and judge is not None:
        judged = report.judge_agreement
        print(f"\n### the judge ({judge.model}), measured against the same human labels\n")
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

    # Rebuild what the baseline was recorded on. Scoring hybrid against a
    # baseline recorded on bm25 would read as a huge gain and mean nothing.
    recorded = baseline.retrieval_config
    index_name = recorded.get("index", "hybrid")
    model = recorded.get("embedding_model") or None
    report = evaluate(build_index(index_name, dataset, cfg, model), dataset)
    retrieval = {name: value.mean for name, value in report.metrics.items()}

    findings = gating.check(
        baseline,
        retrieval,
        {},
        provider=cfg.provider,
        model=cfg.resolved_model(),
        retrieval_config=retrieval_config(index_name, model),
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
    existing = gating.Baseline.load()

    generation = existing.generation if existing else {}
    generation_stddev = existing.generation_stddev if existing else {}
    if args.generation:
        generation, generation_stddev = _measure_generation(cfg, dataset, args)

    report = evaluate(build_index(args.config, dataset, cfg, args.model), dataset)
    baseline = gating.Baseline(
        retrieval={name: round(value.mean, 6) for name, value in report.metrics.items()},
        retrieval_config=retrieval_config(args.config, args.model),
        generation=generation,
        generation_stddev=generation_stddev,
        provider=cfg.provider,
        model=cfg.resolved_model(),
        prompt_version=PROMPT_VERSION,
        recorded_at=datetime.now(UTC).date().isoformat(),
    )
    baseline.save()

    log.info("baseline written to %s over %d queries", gating.BASELINE_PATH, report.queries)
    for name, value in baseline.retrieval.items():
        log.info("  %s %.4f", name, value)
    for name, value in baseline.generation.items():
        log.info("  %s %.4f (sd %.4f)", name, value, baseline.generation_stddev.get(name, 0.0))
    return 0


def record_run(report: GenerationReport, claims: int) -> None:
    run_store.append(
        run_store.Run(
            provider=report.provider,
            model=report.model,
            prompt_version=PROMPT_VERSION,
            claims=claims,
            scored=report.model_agreement.accuracy.n,
            accuracy=report.model_agreement.accuracy.mean,
            faithful_citations=report.faithful_rate,
        )
    )


def _measure_generation(
    cfg: Config, dataset: scifact.Dataset, args: argparse.Namespace
) -> tuple[dict[str, float], dict[str, float]]:
    """Average the recorded runs, and add more first if asked to.

    One run gives a number with no idea how much of it is the model and how
    much is the sampling. The gate needs that spread more than it needs the
    number, and a spread computed inside a single process cannot survive the
    quota running out halfway, so the runs are accumulated on disk and the
    average is taken over every one that matches this configuration.
    """
    claims = args.limit or len(dataset.with_verdict())
    if args.repeat:
        provider = build_provider(cfg)
        index = build_index("hybrid", dataset, cfg)
        for attempt in range(1, args.repeat + 1):
            report = run_generation(index, dataset, provider, top_k=args.top_k, limit=args.limit)
            record_run(report, claims)
            log.info(
                "run %d/%d: accuracy %.4f, faithful %.4f",
                attempt,
                args.repeat,
                report.model_agreement.accuracy.mean,
                report.faithful_rate,
            )

    key = (cfg.provider, cfg.resolved_model(), PROMPT_VERSION, claims)
    recorded = run_store.matching(run_store.load(), key)
    if not recorded:
        raise SystemExit(f"no recorded runs for {key}; run with --repeat to make some")

    values = {
        "accuracy": [run.accuracy for run in recorded],
        "faithful_citations": [run.faithful_citations for run in recorded],
    }
    log.info("averaging %d recorded runs over %d claims", len(recorded), claims)

    means = {name: round(mean(numbers), 6) for name, numbers in values.items()}
    if len(recorded) < 2:
        return means, {}
    return means, {name: round(stdev(numbers), 6) for name, numbers in values.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-eval")
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="download and verify the dataset")
    data.add_argument("--force", action="store_true", help="ignore the cached copy")

    retrieval = sub.add_parser("retrieval", help="score the indexes against the qrels")
    retrieval.add_argument("--split", default="test", choices=["train", "test"])
    retrieval.add_argument("--configs", nargs="*", help="bm25, dense, hybrid")
    retrieval.add_argument("--model", help="embedding model, defaults to the fast one")

    generation = sub.add_parser("generation", help="verify claims and score against human labels")
    generation.add_argument("--split", default="test", choices=["train", "test"])
    generation.add_argument("--top-k", type=int, default=5)
    generation.add_argument("--judge", action="store_true", help="also run and measure a judge")
    generation.add_argument("--limit", type=int, help="stop after this many queries")
    generation.add_argument("--yes", action="store_true", help="skip the cost confirmation")

    ablate = sub.add_parser("ablation", help="compare configurations on the training split")
    ablate.add_argument("--split", default="train", choices=("train", "test"))
    ablate.add_argument("--model", help="embedding model, defaults to the fast one")
    ablate.add_argument("--rrf-k", type=int, nargs="+", default=list(DEFAULT_RRF_K))
    ablate.add_argument("--depth", type=int, nargs="+", default=list(DEFAULT_DEPTHS))

    baseline = sub.add_parser("baseline", help="record the current numbers as the baseline")
    baseline.add_argument("--config", default="hybrid", choices=CONFIGURATIONS)
    baseline.add_argument("--model", help="embedding model, defaults to the fast one")
    baseline.add_argument(
        "--generation", action="store_true", help="also measure the generation metrics"
    )
    baseline.add_argument(
        "--repeat", type=int, default=0, help="runs to add before averaging what is recorded"
    )
    baseline.add_argument("--top-k", type=int, default=5)
    baseline.add_argument("--limit", type=int, help="claims per run, for a cheaper measurement")

    sub.add_parser("gate", help="compare a run against the recorded baseline")

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    handlers = {
        "data": cmd_data,
        "retrieval": cmd_retrieval,
        "ablation": cmd_ablation,
        "generation": cmd_generation,
        "baseline": cmd_baseline,
        "gate": cmd_gate,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
