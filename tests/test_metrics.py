import pytest

from app.metrics import (
    abstention_metrics,
    aggregate,
    by_file,
    char_overlap,
    merge_intervals,
    per_query_at_k,
    precision_recall,
)


def test_merge_intervals_half_open():
    assert merge_intervals([(5, 10), (0, 3), (3, 4), (8, 12), (20, 20)]) == [(0, 4), (5, 12)]
    assert merge_intervals([]) == []


def test_half_open_lengths():
    # [10, 20) has 10 characters; [10, 20) and [20, 30) share none.
    counts = char_overlap([("f", 10, 20)], [("f", 20, 30)])
    assert counts == {"retrieved_chars": 10, "gold_chars": 10, "overlap_chars": 0}


def test_basic_overlap():
    counts = char_overlap([("f", 0, 100)], [("f", 50, 150)])
    assert counts == {"retrieved_chars": 100, "gold_chars": 100, "overlap_chars": 50}
    assert precision_recall(counts) == (0.5, 0.5)


def test_overlap_respects_documents():
    counts = char_overlap([("a", 0, 100)], [("b", 0, 100)])
    assert counts["overlap_chars"] == 0


def test_multiple_gold_spans():
    gold = [("f", 0, 10), ("f", 100, 110), ("g", 0, 10)]
    counts = char_overlap([("f", 5, 105)], gold)
    assert counts == {"retrieved_chars": 100, "gold_chars": 30, "overlap_chars": 10}
    p, r = precision_recall(counts)
    assert p == pytest.approx(0.1) and r == pytest.approx(1 / 3)


def test_overlapping_retrieved_chunks_not_double_counted():
    # Two overlapping chunks covering [0, 150) once each character.
    counts = char_overlap([("f", 0, 100), ("f", 50, 150)], [("f", 40, 60)])
    assert counts == {"retrieved_chars": 150, "gold_chars": 20, "overlap_chars": 20}


def official_formula(retrieved, gold):
    """Re-implementation of the LegalBench-RAG reference code (sums, no unions)."""
    tr = sum(e - s for _, s, e in retrieved)
    tg = sum(e - s for _, s, e in gold)
    inter = 0
    for fr, sr, er in retrieved:
        for fg, sg, eg in gold:
            if fr == fg:
                inter += max(0, min(er, eg) - max(sr, sg))
    return (inter / tr if tr else 0), (inter / tg if tg else 0)


def test_matches_official_formula_when_nothing_overlaps():
    retrieved = [("a", 0, 50), ("a", 60, 90), ("b", 0, 40)]
    gold = [("a", 40, 70), ("b", 30, 35)]
    assert precision_recall(char_overlap(retrieved, gold)) == pytest.approx(official_formula(retrieved, gold))


def test_empty_retrieval_gives_zero_not_error():
    assert precision_recall(char_overlap([], [("f", 0, 10)])) == (0.0, 0.0)


def test_invalid_span_rejected():
    with pytest.raises(ValueError):
        by_file([("f", 10, 5)])


def test_per_query_at_k_uses_prefix_of_ranking():
    ranking = [("f", 0, 10), ("f", 100, 110), ("f", 200, 210)]
    gold = [("f", 100, 110)]
    m = per_query_at_k(ranking, gold, [1, 2, 4])
    assert m[1]["recall"] == 0 and m[1]["precision"] == 0
    assert m[2]["recall"] == 1 and m[2]["precision"] == 0.5
    assert m[4]["recall"] == 1 and m[4]["precision"] == pytest.approx(1 / 3)


def test_aggregate_macro_vs_pooled():
    q1 = {1: {"precision": 1.0, "recall": 1.0, "retrieved_chars": 10, "gold_chars": 10, "overlap_chars": 10}}
    q2 = {1: {"precision": 0.0, "recall": 0.0, "retrieved_chars": 90, "gold_chars": 90, "overlap_chars": 0}}
    agg = aggregate([q1, q2], [1])["1"]
    assert agg["macro_precision"] == 0.5 and agg["macro_recall"] == 0.5
    assert agg["pooled_precision"] == 0.1 and agg["pooled_recall"] == 0.1
    assert agg["num_queries"] == 2


def test_abstention_metrics_definitions():
    m = abstention_metrics(answerable_abstained=[True, False, False, False], unanswerable_abstained=[True, True, False])
    assert m["false_abstention_rate"] == 0.25
    assert m["correct_abstention_rate"] == pytest.approx(2 / 3)
    assert m["abstention_precision"] == pytest.approx(2 / 3)
    assert m["abstention_f1"] == pytest.approx(2 / 3)
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 0.75) / 2)


def test_abstention_metrics_no_abstentions():
    m = abstention_metrics([False, False], [False])
    assert m["abstention_precision"] is None
    assert m["abstention_f1"] == 0.0
    assert m["correct_abstention_rate"] == 0.0 and m["false_abstention_rate"] == 0.0
