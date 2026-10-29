"""Guards for the prescribed data. Skipped when the raw data is not downloaded.

These do not pin the benchmark contents to a hash (it is downloaded, not
committed); they check that the prescribed selection and split are what the
assignment's functions produce and that the committed unanswerable set has
not drifted from its split rule.
"""

import json
from pathlib import Path

import pytest

from app.data import assign_split
from app.pipeline import load_answerable, load_unanswerable, referenced_files
from app.corpus import load_corpus

BENCHMARK = Path("data/raw/benchmarks/cuad.json")
CORPUS = Path("data/raw/corpus/cuad")
PROCESSED = Path("data/processed/cuad_mini_split.json")
UNANSWERABLE = Path("data/unanswerable_queries.json")

needs_benchmark = pytest.mark.skipif(not BENCHMARK.exists(), reason="benchmark not downloaded")


@needs_benchmark
def test_mini_selection_and_split():
    tests = load_answerable(BENCHMARK)
    assert len(tests) == 194
    assert all(t["split"] == assign_split(t["query"]) for t in tests)


@needs_benchmark
@pytest.mark.skipif(not PROCESSED.exists(), reason="processed split not generated")
def test_processed_split_file_matches_regeneration():
    assert json.loads(PROCESSED.read_text(encoding="utf-8")) == load_answerable(BENCHMARK)


@needs_benchmark
@pytest.mark.skipif(not CORPUS.exists(), reason="corpus not downloaded")
def test_every_gold_span_matches_its_answer_text():
    tests = load_answerable(BENCHMARK)
    corpus = load_corpus(referenced_files(tests), CORPUS)
    assert corpus.errors == []
    texts = {d.file_path: d.text for d in corpus.documents}
    for t in tests:
        for s in t["snippets"]:
            start, end = s["span"]
            assert texts[s["file_path"]][start:end] == s["answer"]


def test_unanswerable_set_is_intact():
    items = load_unanswerable(UNANSWERABLE)  # raises if any split label drifted
    assert len(items) >= 30
    assert len({i["id"] for i in items}) == len(items)
    assert len({i["question"].strip() for i in items}) == len(items)
