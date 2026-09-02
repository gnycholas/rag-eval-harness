"""Command line entry points."""

from __future__ import annotations

import argparse
import logging
import sys

from rag_eval.config import load_config
from rag_eval.data import scifact

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-eval")
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="download and verify the dataset")
    data.add_argument("--force", action="store_true", help="ignore the cached copy")

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    handlers = {"data": cmd_data}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
