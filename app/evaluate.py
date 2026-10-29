"""Run one evaluation of a configuration and write a self-contained run folder.

Order of work (keeps the test split honest):
  1. Build corpus + index; record path resolution; abort on any data problem.
  2. Check every gold span: document[start:end] == answer.
  3. Score TRAIN queries (answerable + unanswerable).
  4. If the config has no threshold, calibrate it on TRAIN only; freeze it.
  5. Only then, if requested, score TEST queries with the frozen threshold,
     and append a line to experiments/test_set_runs.jsonl (test-use ledger).

Retrieval metrics are computed on the full ranking (top ``retrieval_depth``)
regardless of the abstention decision, so retrieval quality and abstention
quality are measured separately.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from app.abstention import abstention_score, calibrate_threshold, should_abstain
from app.config import config_hash, with_threshold
from app.corpus import corpus_manifest
from app.metrics import abstention_metrics, aggregate, per_query_at_k
from app.pipeline import DataError, Pipeline, build_pipeline, load_unanswerable

TEST_LEDGER = Path("experiments/test_set_runs.jsonl")
SPLITS = ("train", "test")


def sha256_file(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def git_info() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "changed_or_untracked_paths": status.splitlines() if status else [],
    }


def versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("rank-bm25", "numpy", "pyyaml"):
        try:
            out[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            out[package] = "not installed"
    return out


def json_safe(value: Any) -> Any:
    """Replace +/-inf (possible calibrated thresholds) with strings for strict JSON."""
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(json_safe(data), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")


def check_gold_spans(pipeline: Pipeline) -> list[dict[str, Any]]:
    texts = pipeline.texts
    problems = []
    for test in pipeline.answerable:
        for s in test["snippets"]:
            text = texts.get(s["file_path"])
            start, end = s["span"]
            if text is None or text[start:end] != s["answer"]:
                problems.append({"type": "gold_span_mismatch", "query": test["query"], "file_path": s["file_path"], "span": s["span"]})
    return problems


def ranking_record(results) -> list[dict[str, Any]]:
    return [
        {"rank": r.rank, "score": r.score, "rerank_score": r.rerank_score, "chunk_id": r.chunk_id, "file_path": r.file_path, "start": r.start, "end": r.end}
        for r in results
    ]


def query_ids(pipeline: Pipeline) -> dict[str, str]:
    """Stable id per answerable query: position in the prescribed mini selection."""
    return {t["query"]: f"q{i:03d}" for i, t in enumerate(pipeline.answerable)}


def num_query_tokens(query: str) -> int:
    return len(re.findall(r"\w+", query))


def score_answerable(pipeline: Pipeline, tests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Answerable and unanswerable queries go through the same search + features."""
    ks = pipeline.config["evaluation"]["ks"]
    depth = pipeline.config["evaluation"]["retrieval_depth"]
    kind = pipeline.config["abstention"].get("score", "raw_top1")
    ids = query_ids(pipeline)
    rows = []
    for test in tests:
        results, features = pipeline.index.search_detailed(test["query"], depth)
        retrieved = [(r.file_path, r.start, r.end) for r in results]
        gold = [(s["file_path"], s["span"][0], s["span"][1]) for s in test["snippets"]]
        gold_files = {g[0] for g in gold}
        rows.append(
            {
                "query_id": ids[test["query"]],
                "query": test["query"],
                "split": test["split"],
                "num_query_tokens": num_query_tokens(test["query"]),
                "gold": [{"file_path": f, "start": s, "end": e} for f, s, e in gold],
                "top_score": features.top1,
                "score_features": features.as_dict(),
                "abstention_score": abstention_score(features.as_dict(), kind),
                "metrics_at_k": {str(k): v for k, v in per_query_at_k(retrieved, gold, ks).items()},
                # Diagnostic only (not a substitute for the character metrics):
                # rank of the first chunk from a gold document.
                "first_gold_doc_rank": next((r.rank for r in results if r.file_path in gold_files), None),
                "ranking": ranking_record(results),
            }
        )
    return rows


def score_unanswerable(pipeline: Pipeline, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    depth = pipeline.config["evaluation"]["retrieval_depth"]
    kind = pipeline.config["abstention"].get("score", "raw_top1")
    rows = []
    for item in items:
        results, features = pipeline.index.search_detailed(item["question"], depth)
        rows.append(
            {
                "id": item["id"],
                "query": item["question"],
                "split": item["split"],
                "num_query_tokens": num_query_tokens(item["question"]),
                "top_score": features.top1,
                "score_features": features.as_dict(),
                "abstention_score": abstention_score(features.as_dict(), kind),
                "ranking": ranking_record(results),
            }
        )
    return rows


def apply_threshold(rows: list[dict[str, Any]], threshold: float) -> None:
    for row in rows:
        row["abstained"] = should_abstain(row["abstention_score"], threshold)


def failure_bucket(row: dict[str, Any], k: int) -> str:
    """Retrieval outcome for one answerable query at k (ignores abstention)."""
    m = row["metrics_at_k"][str(k)]
    gold_files = {g["file_path"] for g in row["gold"]}
    if not any(r["file_path"] in gold_files for r in row["ranking"][:k]):
        return "wrong_document"
    if m["overlap_chars"] == 0:
        return "right_doc_wrong_section"
    if m["recall"] < 1.0:
        return "partial_span"
    return "full_coverage"


def diagnosis(ans: list[dict], unans: list[dict], k: int) -> dict[str, Any]:
    buckets = {"wrong_document": 0, "right_doc_wrong_section": 0, "partial_span": 0, "full_coverage": 0}
    for row in ans:
        row["failure_bucket_at_diagnostic_k"] = failure_bucket(row, k)
        buckets[row["failure_bucket_at_diagnostic_k"]] += 1
    false_abs = sum(r["abstained"] for r in ans)
    false_answers = sum(not r["abstained"] for r in unans)
    return {
        "k": k,
        "answerable_buckets": buckets,
        "false_abstentions": false_abs,
        "false_answers_on_unanswerable": false_answers,
        "incorrect_abstention": false_abs + false_answers,
    }


def length_correlation(rows: list[dict]) -> float | None:
    """Pearson r between abstention score and query length (None if undefined)."""
    pairs = [(r["num_query_tokens"], r["abstention_score"]) for r in rows if r["abstention_score"] is not None]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    try:
        return statistics.correlation(xs, ys)
    except statistics.StatisticsError:
        return None


def split_metrics(ans: list[dict], unans: list[dict], ks: list[int], diagnostic_k: int = 8) -> dict[str, Any]:
    per_query = [{int(k): v for k, v in r["metrics_at_k"].items()} for r in ans]
    doc_hit = {
        str(k): (sum(1 for r in ans if r["first_gold_doc_rank"] is not None and r["first_gold_doc_rank"] <= k) / len(ans) if ans else None)
        for k in ks
    }
    return {
        "retrieval": aggregate(per_query, ks),
        "diagnostic_gold_document_hit_rate": doc_hit,
        "abstention": abstention_metrics([r["abstained"] for r in ans], [r["abstained"] for r in unans]),
        "abstention_score_length_correlation_answerable": length_correlation(ans),
        "diagnosis": diagnosis(ans, unans, diagnostic_k),
    }


def make_run_dir(output_root: Path, config: dict[str, Any]) -> tuple[str, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{stamp}_{config_hash(config)[:8]}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)  # never overwrite an earlier run
    return run_id, run_dir


def run_evaluation(config: dict[str, Any], splits: list[str], config_path: str | None = None, label: str | None = None) -> Path:
    for s in splits:
        if s not in SPLITS:
            raise ValueError(f"unknown split {s!r}")
    if "train" not in splits and config["abstention"]["threshold"] is None:
        raise ValueError("a null threshold must be calibrated on train: include the train split")

    started = datetime.now(timezone.utc)
    run_id, run_dir = make_run_dir(Path(config["output_root"]), config)
    data = config["data"]
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "started_utc": started.isoformat(),
        "config_path": config_path,
        "config_sha256": config_hash(config),
        "splits_evaluated": splits,
        "label": label,
        "git": git_info(),
        "versions": versions(),
        "input_hashes": {
            "benchmark": sha256_file(Path(data["benchmark_path"])),
            "unanswerable": sha256_file(Path(data["unanswerable_path"])),
            "processed_split_file": sha256_file(Path("data/processed/cuad_mini_split.json")),
        },
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    def fail(errors: list[dict[str, Any]]) -> None:
        write_json(run_dir / "errors.json", errors)
        metadata.update(status="failed", finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(run_dir / "run_metadata.json", metadata)
        raise DataError(f"evaluation aborted, see {run_dir / 'errors.json'}")

    pipeline = build_pipeline(config, strict=False)
    unanswerable = load_unanswerable(Path(data["unanswerable_path"]))
    write_json(run_dir / "path_resolution.json", [r.__dict__ | {"candidates": list(r.candidates)} for r in pipeline.corpus.resolutions])
    write_json(run_dir / "corpus_manifest.json", corpus_manifest(pipeline.corpus))
    errors = pipeline.corpus.errors + check_gold_spans(pipeline)
    if errors:
        fail(errors)

    resolution_counts: dict[str, int] = {}
    for r in pipeline.corpus.resolutions:
        resolution_counts[r.status] = resolution_counts.get(r.status, 0) + 1
    metadata.update(
        num_answerable_queries=len(pipeline.answerable),
        num_annotations=sum(len(t["snippets"]) for t in pipeline.answerable),
        num_unanswerable_queries=len(unanswerable),
        num_indexed_documents=len(pipeline.corpus.documents),
        num_chunks=len(pipeline.chunks),
        corpus_combined_sha256=corpus_manifest(pipeline.corpus)["combined_sha256"],
        path_resolution_counts=resolution_counts,
    )

    ks = config["evaluation"]["ks"]
    dk = config["evaluation"].get("diagnostic_k", 8 if 8 in ks else max(ks))
    if dk not in ks:
        raise ValueError(f"evaluation.diagnostic_k={dk} must be one of evaluation.ks")
    answerable_rows: list[dict] = []
    unanswerable_rows: list[dict] = []
    metrics: dict[str, Any] = {"definitions": "see app/metrics.py docstring; macro = mean over queries (primary), pooled = summed characters (secondary)"}

    # --- TRAIN first -------------------------------------------------------
    train_ans = score_answerable(pipeline, [t for t in pipeline.answerable if t["split"] == "train"])
    train_un = score_unanswerable(pipeline, [u for u in unanswerable if u["split"] == "train"])

    threshold = config["abstention"]["threshold"]
    if threshold is None:
        calibration = calibrate_threshold([r["abstention_score"] for r in train_ans], [r["abstention_score"] for r in train_un])
        calibration["score"] = config["abstention"].get("score", "raw_top1")
        threshold = calibration["threshold"]
        calibration["calibrated_on"] = {"split": "train", "answerable": len(train_ans), "unanswerable": len(train_un)}
        write_json(run_dir / "calibration.json", calibration)
        frozen = with_threshold(config, threshold, f"calibrated on train in run {run_id}")
    else:
        frozen = config
    # Frozen before any test query is searched.
    (run_dir / "frozen_config.yaml").write_text(yaml.safe_dump(json_safe(frozen), sort_keys=False), encoding="utf-8")

    if "train" in splits:
        apply_threshold(train_ans, threshold)
        apply_threshold(train_un, threshold)
        metrics["train"] = split_metrics(train_ans, train_un, ks, dk)
        answerable_rows += train_ans
        unanswerable_rows += train_un

    # --- TEST only after the threshold is frozen ---------------------------
    if "test" in splits:
        test_ans = score_answerable(pipeline, [t for t in pipeline.answerable if t["split"] == "test"])
        test_un = score_unanswerable(pipeline, [u for u in unanswerable if u["split"] == "test"])
        apply_threshold(test_ans, threshold)
        apply_threshold(test_un, threshold)
        metrics["test"] = split_metrics(test_ans, test_un, ks, dk)
        answerable_rows += test_ans
        unanswerable_rows += test_un
        TEST_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with open(TEST_LEDGER, "a", encoding="utf-8") as ledger:
            ledger.write(json.dumps({"run_id": run_id, "label": label, "utc": started.isoformat(), "config": config["name"], "config_sha256": config_hash(config), "git_commit": metadata["git"]["commit"], "run_dir": str(run_dir)}) + "\n")

    metrics["abstention_rule"] = {"rule": config["abstention"]["rule"], "score": config["abstention"].get("score", "raw_top1"), "threshold": threshold, "source": frozen["abstention"].get("threshold_source", "set in config")}
    write_json(run_dir / "metrics.json", metrics)
    write_jsonl(run_dir / "per_query_answerable.jsonl", answerable_rows)
    write_jsonl(run_dir / "per_query_unanswerable.jsonl", unanswerable_rows)
    write_json(run_dir / "errors.json", [])
    metadata.update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(run_dir / "run_metadata.json", metadata)
    return run_dir


def print_summary(run_dir: Path, out=sys.stdout) -> None:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    print(f"Run: {run_dir}", file=out)
    print(f"Abstention: {metrics['abstention_rule']}", file=out)
    for split in SPLITS:
        if split not in metrics:
            continue
        m = metrics[split]
        n = next(iter(m["retrieval"].values()))["num_queries"]
        print(f"\n[{split}] answerable queries: {n}", file=out)
        print(f"{'k':>4} {'P@k':>8} {'R@k':>8} {'pooledP':>8} {'pooledR':>8} {'docHit':>7}", file=out)
        for k, row in m["retrieval"].items():
            print(f"{k:>4} {row['macro_precision']:8.4f} {row['macro_recall']:8.4f} {row['pooled_precision']:8.4f} {row['pooled_recall']:8.4f} {m['diagnostic_gold_document_hit_rate'][k]:7.3f}", file=out)
        a = m["abstention"]
        print(
            f"abstention: correct {a['correct_abstentions']}/{a['num_unanswerable']} unanswerable "
            f"(rate {a['correct_abstention_rate']:.3f}); false {a['false_abstentions']}/{a['num_answerable']} answerable "
            f"(rate {a['false_abstention_rate']:.3f}); F1 {a['abstention_f1']:.3f}",
            file=out,
        )
