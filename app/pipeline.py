"""Assemble benchmark, corpus and BM25 index from a configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.abstention import abstention_score, should_abstain
from app.corpus import Corpus, load_corpus
from app.data import add_splits, assign_split, load_benchmark
from app.retrieval import BM25Index, Chunk, SearchResult, chunk_corpus


class DataError(RuntimeError):
    pass


def load_answerable(path: Path) -> list[dict[str, Any]]:
    """The prescribed 194 mini queries with their SHA-256 split labels."""
    tests = add_splits(load_benchmark(path))
    if len(tests) != 194:
        raise DataError(f"expected 194 mini queries, found {len(tests)}")
    return tests


def load_unanswerable(path: Path) -> list[dict[str, Any]]:
    """Committed unanswerable set; stored split labels must match the rule."""
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    for item in items:
        expected = assign_split(item["question"])
        if item.get("split") != expected:
            raise DataError(f"{item['id']}: stored split {item.get('split')!r} != assign_split {expected!r}")
        if item.get("label") != "unanswerable":
            raise DataError(f"{item['id']}: label is {item.get('label')!r}")
    return items


def referenced_files(tests: list[dict[str, Any]]) -> list[str]:
    """Every file referenced by any gold snippet of the mini benchmark."""
    return sorted({s["file_path"] for t in tests for s in t["snippets"]})


@dataclass
class Pipeline:
    config: dict[str, Any]
    answerable: list[dict[str, Any]]
    corpus: Corpus
    chunks: list[Chunk]
    index: BM25Index

    @property
    def texts(self) -> dict[str, str]:
        return {d.file_path: d.text for d in self.corpus.documents}

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        return self.index.search(query, top_k)

    def answer(self, query: str, top_k: int) -> dict[str, Any]:
        """Ranked passages plus the abstention decision (what users see)."""
        threshold = self.config["abstention"]["threshold"]
        if threshold is None:
            raise DataError("abstention.threshold is null: use a frozen config produced by an evaluation run")
        results, features = self.index.search_detailed(query, top_k)
        kind = self.config["abstention"].get("score", "raw_top1")
        score = abstention_score(features.as_dict(), kind)
        abstain = should_abstain(score, threshold)
        return {
            "query": query,
            "abstained": abstain,
            "abstention_score": score,
            "abstention_score_kind": kind,
            "top_score": features.top1,
            "threshold": threshold,
            "results": [] if abstain else results,
        }


def build_pipeline(config: dict[str, Any], strict: bool = True) -> Pipeline:
    data = config["data"]
    answerable = load_answerable(Path(data["benchmark_path"]))
    corpus = load_corpus(referenced_files(answerable), Path(data["corpus_root"]))
    if strict and corpus.errors:
        raise DataError(f"corpus problems: {corpus.errors}")
    chunking, retrieval = config["chunking"], config["retrieval"]
    chunks = chunk_corpus(corpus.documents, chunking["chunk_size"], chunking["chunk_overlap"])
    index = BM25Index(
        chunks,
        k1=retrieval["k1"],
        b=retrieval["b"],
        epsilon=retrieval["epsilon"],
        query_stopwords=retrieval.get("query_stopwords", False),
        rerank=retrieval.get("rerank", "none"),
        rerank_docs=retrieval.get("rerank_docs", 1),
    )
    return Pipeline(config, answerable, corpus, chunks, index)
