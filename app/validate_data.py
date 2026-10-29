"""Validate CUAD benchmark annotations against the source corpus."""

import json
import random
import hashlib
from pathlib import Path
from typing import Any


DEFAULT_BENCHMARK = Path("data/raw/benchmarks/cuad.json")
DEFAULT_CORPUS = Path("data/raw/corpus/cuad")


def mini(tests: list[dict[str, Any]], n: int = 194) -> list[dict[str, Any]]:
    """Select the benchmark's deterministic mini subset.

    Uses the same algorithm as app.data.mini: seeds Python's random with
    the first snippet's file_path and picks the n tests with smallest
    random values.
    """
    if len(tests) <= n:
        return tests

    def key(test: dict[str, Any]) -> float:
        random.seed(test["snippets"][0]["file_path"])
        return random.random()

    return sorted(tests, key=key)[:n]


def load_benchmark(path: Path = DEFAULT_BENCHMARK) -> list[dict[str, Any]]:
    """Load the CUAD benchmark and select its mini subset."""
    raw = path.read_text(encoding="utf-8")
    tests = json.loads(raw)["tests"]
    return mini(tests)


def resolve_corpus_path(file_path: str, corpus_root: Path = DEFAULT_CORPUS) -> Path:
    """Resolve a benchmark file_path to the actual corpus file.

    Benchmark paths are prefixed with 'cuad/'. Strip that prefix to get
    the actual filename under the corpus root.
    """
    if file_path.startswith("cuad/"):
        file_name = file_path[5:]
    else:
        file_name = file_path
    return corpus_root / file_name


def validate_snippet(
    file_path: str,
    span: list[int],
    answer: str,
    corpus_root: Path = DEFAULT_CORPUS,
) -> dict[str, Any]:
    """Validate a single snippet against the source document.

    Returns a dict with:
    - "status": "match" | "mismatch" | "missing_file" | "malformed"
    - "details": dict with diagnostic information
    """
    corpus_path = resolve_corpus_path(file_path, corpus_root)

    if not corpus_path.exists():
        return {
            "status": "missing_file",
            "details": {"file_path": file_path, "resolved_path": str(corpus_path)},
        }

    try:
        content = corpus_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        return {
            "status": "malformed",
            "details": {
                "file_path": file_path,
                "resolved_path": str(corpus_path),
                "error": f"UTF-8 decode error: {e}",
            },
        }

    # Validate span structure
    if not isinstance(span, list) or len(span) != 2:
        return {
            "status": "malformed",
            "details": {
                "file_path": file_path,
                "resolved_path": str(corpus_path),
                "error": f"span must be a list of two integers, got {span!r}",
            },
        }

    start, end = span
    if not isinstance(start, int) or not isinstance(end, int):
        return {
            "status": "malformed",
            "details": {
                "file_path": file_path,
                "resolved_path": str(corpus_path),
                "error": f"span bounds must be integers, got {span!r}",
            },
        }

    if start < 0 or end < 0 or start > end:
        return {
            "status": "malformed",
            "details": {
                "file_path": file_path,
                "resolved_path": str(corpus_path),
                "error": f"invalid span bounds: start={start}, end={end}",
            },
        }

    if end > len(content):
        return {
            "status": "malformed",
            "details": {
                "file_path": file_path,
                "resolved_path": str(corpus_path),
                "error": f"span end={end} exceeds file length={len(content)}",
            },
        }

    extracted = content[start:end]

    if extracted == answer:
        return {
            "status": "match",
            "details": {
                "file_path": file_path,
                "span": span,
                "length": len(extracted),
            },
        }
    else:
        # Provide diagnostic info for mismatch
        return {
            "status": "mismatch",
            "details": {
                "file_path": file_path,
                "span": span,
                "expected_length": len(answer),
                "actual_length": len(extracted),
                "expected_preview": answer[:200],
                "actual_preview": extracted[:200],
                "expected_suffix": answer[-200:] if len(answer) > 200 else answer,
                "actual_suffix": extracted[-200:] if len(extracted) > 200 else extracted,
            },
        }


def run_validation(
    benchmark_path: Path = DEFAULT_BENCHMARK,
    corpus_root: Path = DEFAULT_CORPUS,
) -> dict[str, Any]:
    """Run full validation on the mini benchmark.

    Returns a summary dict with counts and mismatch details.
    """
    tests = load_benchmark(benchmark_path)

    total_queries = len(tests)
    total_annotations = 0
    matches = 0
    mismatches = 0
    missing_files = 0
    malformed = 0
    mismatch_details = []

    for test in tests:
        for snippet in test["snippets"]:
            total_annotations += 1
            result = validate_snippet(
                snippet["file_path"],
                snippet["span"],
                snippet["answer"],
                corpus_root,
            )

            status = result["status"]
            if status == "match":
                matches += 1
            elif status == "mismatch":
                mismatches += 1
                mismatch_details.append(
                    {
                        "query": test["query"],
                        **result["details"],
                    }
                )
            elif status == "missing_file":
                missing_files += 1
            elif status == "malformed":
                malformed += 1

    return {
        "total_queries": total_queries,
        "total_annotations": total_annotations,
        "matches": matches,
        "mismatches": mismatches,
        "missing_files": missing_files,
        "malformed": malformed,
        "mismatch_details": mismatch_details,
    }


def print_summary(summary: dict[str, Any]) -> None:
    """Print a human-readable validation summary."""
    print("=== CUAD Mini Benchmark Validation ===")
    print(f"Total selected queries:  {summary['total_queries']}")
    print(f"Total annotations checked: {summary['total_annotations']}")
    print(f"Exact matches:           {summary['matches']}")
    print(f"Mismatches:              {summary['mismatches']}")
    print(f"Missing files:           {summary['missing_files']}")
    print(f"Malformed annotations:   {summary['malformed']}")
    print()

    if summary["mismatch_details"]:
        print(f"--- Mismatch Details ({len(summary['mismatch_details'])}) ---")
        for i, d in enumerate(summary["mismatch_details"], 1):
            print(f"\n{i}. Query: {d['query'][:100]}...")
            print(f"   File: {d['file_path']}")
            print(f"   Span: {d['span']}")
            print(f"   Expected length: {d['expected_length']}, Actual length: {d['actual_length']}")
            print(f"   Expected: {d['expected_preview']!r}")
            print(f"   Actual:   {d['actual_preview']!r}")


def main() -> None:
    """Run validation and print summary."""
    summary = run_validation()
    print_summary(summary)

    # Exit with error code if any issues found
    if summary["mismatches"] > 0 or summary["missing_files"] > 0 or summary["malformed"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()