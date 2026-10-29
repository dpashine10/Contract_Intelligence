"""Command-line interface.

  uv run python -m app.cli evaluate --config configs/baseline.yaml --splits train
  uv run python -m app.cli query "Can the licensee sublicense?"   # default: configs/final_frozen.yaml
"""

from __future__ import annotations

import argparse
import json
import sys

from app.config import ConfigError, load_config
from app.evaluate import print_summary, run_evaluation
from app.pipeline import DataError, build_pipeline


def cmd_evaluate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.output_root:
        config["output_root"] = args.output_root
    run_dir = run_evaluation(config, args.splits, config_path=args.config, label=args.label)
    print_summary(run_dir)
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    pipeline = build_pipeline(config)
    top_k = args.top_k or config["ui"]["display_top_k"]
    answer = pipeline.answer(args.question, top_k)
    texts = pipeline.texts

    if args.json:
        out = dict(answer)
        out["results"] = [r.__dict__ | {"text": texts[r.file_path][r.start : r.end]} for r in answer["results"]]
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0

    score = "none" if answer["abstention_score"] is None else f"{answer['abstention_score']:.3f}"
    label = answer["abstention_score_kind"]
    results = answer["results"]
    if answer["abstained"]:
        print(f"No sufficiently relevant passage found ({label} {score} < threshold {answer['threshold']:.3f}). Returning nothing.")
        if not args.show_anyway:
            return 0
        print("\n[diagnostic --show-anyway: these passages were NOT returned as an answer]\n")
        results = pipeline.search(args.question, top_k)
    else:
        print(f"{label} {score} >= threshold {answer['threshold']:.3f}\n")
    for r in results:
        passage = " ".join(texts[r.file_path][r.start : r.end].split())
        local = "" if r.rerank_score is None else f"  local={r.rerank_score:.3f}"
        print(f"#{r.rank}  bm25={r.score:.3f}{local}  {r.file_path}  chars [{r.start}, {r.end})")
        print(f"    {passage[: args.preview_chars]}{'…' if len(passage) > args.preview_chars else ''}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="contract-intelligence")
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("evaluate", help="evaluate a configuration and write a run folder")
    ev.add_argument("--config", default="configs/baseline.yaml")
    ev.add_argument("--splits", nargs="+", default=["train"], choices=["train", "test"], help="test runs are logged to experiments/test_set_runs.jsonl")
    ev.add_argument("--output-root", default=None, help="override config output_root")
    ev.add_argument("--label", default=None, help="free-text label stored in run metadata and the test ledger")
    ev.set_defaults(func=cmd_evaluate)

    q = sub.add_parser("query", help="search the contracts for one question")
    q.add_argument("question")
    q.add_argument("--config", default="configs/final_frozen.yaml")
    q.add_argument("--top-k", type=int, default=None)
    q.add_argument("--preview-chars", type=int, default=300)
    q.add_argument("--json", action="store_true")
    q.add_argument("--show-anyway", action="store_true", help="diagnostic: print ranked passages even when abstaining")
    q.set_defaults(func=cmd_query)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, DataError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
