"""Load and deterministically split the LegalBench-RAG CUAD benchmark."""

import hashlib
import json
import random
from pathlib import Path
from typing import Any


DEFAULT_BENCHMARK = Path("data/raw/benchmarks/cuad.json")


def mini(tests: list[dict[str, Any]], n: int = 194) -> list[dict[str, Any]]:
    """Select the benchmark's deterministic mini subset."""
    if len(tests) <= n:
        return tests

    def key(test: dict[str, Any]) -> float:
        random.seed(test["snippets"][0]["file_path"])
        return random.random()

    return sorted(tests, key=key)[:n]


def assign_split(query: str) -> str:
    """Assign a query to train or test using the prescribed SHA-256 rule."""
    h = int(hashlib.sha256(query.strip().encode("utf-8")).hexdigest(), 16)
    return "test" if h % 10 < 3 else "train"


def load_benchmark(
    path: Path = DEFAULT_BENCHMARK,
) -> list[dict[str, Any]]:
    """Load the CUAD benchmark and select its mini subset."""
    raw = path.read_text(encoding="utf-8")
    tests = json.loads(raw)["tests"]
    return mini(tests)


def get_query(test: dict[str, Any]) -> str:
    """Return the query text, failing clearly if its field is missing."""
    for field in ("query", "question"):
        value = test.get(field)
        if isinstance(value, str):
            return value

    raise ValueError(
        f"Could not identify the query field. Available fields: {list(test)}"
    )


def add_splits(
    tests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return copies of the selected queries with deterministic split labels."""
    result = []

    for test in tests:
        query = get_query(test)
        item = dict(test)
        item["split"] = assign_split(query)
        result.append(item)

    return result


def split_counts(
    tests: list[dict[str, Any]],
) -> dict[str, int]:
    """Count queries in each split."""
    return {
        "train": sum(test["split"] == "train" for test in tests),
        "test": sum(test["split"] == "test" for test in tests),
    }


def main() -> None:
    tests = add_splits(load_benchmark())
    counts = split_counts(tests)

    print(f"Selected queries: {len(tests)}")
    print(f"Train queries: {counts['train']}")
    print(f"Test queries: {counts['test']}")

    if len(tests) != 194:
        raise ValueError(
            f"Expected 194 queries, found {len(tests)}. "
            "Check the downloaded benchmark."
        )

    output = Path("data/processed/cuad_mini_split.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(tests, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved split dataset to {output}")


if __name__ == "__main__":
    main()
