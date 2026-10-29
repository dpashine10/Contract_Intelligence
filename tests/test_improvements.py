"""Tests for the options added for the improvement loop."""

import copy

import pytest

from app.abstention import abstention_score
from app.config import ConfigError, validate_config
from app.corpus import Document
from app.retrieval import BM25Index, chunk_corpus, query_tokens


def doc(fp, text):
    return Document(fp, text, "x")


def test_query_stopwords_removed_only_when_enabled():
    q = "Consider the Agreement; what is the term of this contract?"
    assert "the" in query_tokens(q)
    assert query_tokens(q, drop_stopwords=True) == ["agreement", "term", "contract"]


def test_score_features_upper_bound_and_stats():
    chunks = chunk_corpus([doc("cuad/a", "alpha beta gamma"), doc("cuad/b", "delta epsilon zeta"), doc("cuad/c", "eta theta iota")], 8, 0)
    index = BM25Index(chunks, k1=1.5, b=0.75, epsilon=0.25)
    results, f = index.search_detailed("alpha unknownword", 3)
    assert f.num_query_terms == 2 and f.num_query_terms_in_vocab == 1
    assert f.max_possible == pytest.approx(index._bm25.idf["alpha"] * 2.5)
    assert 0 < f.top1 <= f.max_possible  # BM25 never exceeds its upper bound
    assert f.top1 == results[0].score
    assert f.std > 0


def test_empty_query_features():
    index = BM25Index(chunk_corpus([doc("cuad/a", "alpha")], 8, 0), k1=1.5, b=0.75, epsilon=0.25)
    results, f = index.search_detailed("?!", 3)
    assert results == [] and f.top1 is None


def test_abstention_score_kinds():
    f = {"top1": 6.0, "max_possible": 12.0, "mean": 2.0, "std": 2.0}
    assert abstention_score(f, "raw_top1") == 6.0
    assert abstention_score(f, "normalized_top1") == 0.5
    assert abstention_score(f, "zscore_top1") == 2.0
    assert abstention_score({"top1": None}, "normalized_top1") is None
    assert abstention_score({"top1": 1.0, "max_possible": 0.0}, "normalized_top1") is None
    with pytest.raises(ValueError):
        abstention_score(f, "nope")


def test_normalized_score_is_invariant_to_repeating_the_query():
    chunks = chunk_corpus([doc("cuad/a", "royalty rate five percent"), doc("cuad/b", "governing law delaware"), doc("cuad/c", "term two years")], 8, 0)
    index = BM25Index(chunks, k1=1.5, b=0.75, epsilon=0.25)
    _, short = index.search_detailed("royalty rate", 3)
    _, long = index.search_detailed("royalty rate royalty rate royalty rate", 3)
    assert long.top1 == pytest.approx(3 * short.top1)  # raw score scales with length
    assert abstention_score(long.as_dict(), "normalized_top1") == pytest.approx(abstention_score(short.as_dict(), "normalized_top1"))


def test_doc_local_rerank_reorders_within_top_document():
    # Party name appears everywhere in doc a; only one chunk has the clause term.
    a = " ".join(["acme"] * 6 + ["filler"] * 2) + " " + " ".join(["acme"] * 2 + ["termination", "notice"] + ["filler"] * 4)
    b = "unrelated text about something else entirely here"
    c = "acme termination"  # other document with the clause term
    chunks = chunk_corpus([doc("cuad/a", a), doc("cuad/b", b), doc("cuad/c", c)], 8, 0)
    plain = BM25Index(chunks, k1=1.5, b=0.75, epsilon=0.25)
    rerank = BM25Index(chunks, k1=1.5, b=0.75, epsilon=0.25, rerank="doc_local_bm25", rerank_docs=1)
    q = "acme acme acme termination"
    base = plain.search(q, 4)
    top_doc = base[0].file_path
    out = rerank.search(q, 4)
    # Re-ranked list starts with every chunk of the top document, then the rest.
    n_top = sum(1 for ch in chunks if ch.file_path == top_doc)
    assert [r.file_path for r in out[:n_top]] == [top_doc] * n_top
    assert all(r.rerank_score is not None for r in out[:n_top])
    assert all(r.rerank_score is None for r in out[n_top:])
    # Within the top document, results are ordered by the local score.
    local = [r.rerank_score for r in out[:n_top]]
    assert local == sorted(local, reverse=True)
    # Same set of chunks overall, deterministic.
    assert rerank.search(q, 4) == out
    assert {r.chunk_id for r in rerank.search(q, len(chunks))} == {c.chunk_id for c in chunks}


def test_invalid_rerank_rejected():
    with pytest.raises(ValueError):
        BM25Index([], k1=1.5, b=0.75, epsilon=0.25, rerank="neural")


BASE = {
    "name": "x",
    "output_root": "o",
    "data": {"benchmark_path": "b", "corpus_root": "c", "unanswerable_path": "u"},
    "chunking": {"method": "fixed_tokens", "chunk_size": 8, "chunk_overlap": 0},
    "retrieval": {"method": "bm25_okapi", "k1": 1.5, "b": 0.75, "epsilon": 0.25},
    "abstention": {"rule": "top1_score_threshold", "threshold": None},
    "evaluation": {"ks": [1, 8], "retrieval_depth": 8},
    "ui": {"display_top_k": 5},
}


@pytest.mark.parametrize(
    "section,key,value",
    [("retrieval", "rerank", "neural"), ("retrieval", "rerank_docs", 0), ("retrieval", "query_stopwords", "yes"), ("abstention", "score", "magic")],
)
def test_optional_keys_validated(section, key, value):
    cfg = copy.deepcopy(BASE)
    cfg[section][key] = value
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_old_config_without_optional_keys_is_valid():
    validate_config(copy.deepcopy(BASE))
