"""Character-level retrieval metrics (LegalBench-RAG) and abstention metrics.

Spans are zero-based, half-open ``[start, end)`` intervals of the original
contract text, keyed by file_path.

Per query, with R = union of retrieved intervals at k and G = union of gold
intervals (both per file):

    precision@k = |R ∩ G| / |R|      (0 if |R| == 0)
    recall@k    = |R ∩ G| / |G|      (0 if |G| == 0)

This is the formula in the official LegalBench-RAG code, which sums snippet
lengths and is only correct because its snippets never overlap. Our chunks
overlap by design, so we take unions first; on non-overlapping inputs the two
are identical (tested). Aggregation: the primary figure is the macro average
over queries (the official code's default, weight 1 per query). A pooled
(micro) figure is reported as a secondary number.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

Interval = tuple[int, int]
Span = tuple[str, int, int]  # (file_path, start, end)


def merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    """Merge overlapping or touching half-open intervals; drop empty ones."""
    merged: list[list[int]] = []
    for start, end in sorted(i for i in intervals if i[1] > i[0]):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def by_file(spans: Iterable[Span]) -> dict[str, list[Interval]]:
    grouped: dict[str, list[Interval]] = defaultdict(list)
    for file_path, start, end in spans:
        if start < 0 or end < start:
            raise ValueError(f"invalid span {(file_path, start, end)}")
        grouped[file_path].append((start, end))
    return {f: merge_intervals(v) for f, v in grouped.items()}


def total_length(intervals: Mapping[str, Sequence[Interval]]) -> int:
    return sum(e - s for spans in intervals.values() for s, e in spans)


def intersection_length(a: Mapping[str, Sequence[Interval]], b: Mapping[str, Sequence[Interval]]) -> int:
    """Characters covered by both (inputs must already be merged per file)."""
    total = 0
    for file_path in a.keys() & b.keys():
        x, y = a[file_path], b[file_path]
        i = j = 0
        while i < len(x) and j < len(y):
            lo, hi = max(x[i][0], y[j][0]), min(x[i][1], y[j][1])
            if hi > lo:
                total += hi - lo
            if x[i][1] < y[j][1]:
                i += 1
            else:
                j += 1
    return total


def char_overlap(retrieved: Iterable[Span], gold: Iterable[Span]) -> dict[str, int]:
    r, g = by_file(retrieved), by_file(gold)
    return {
        "retrieved_chars": total_length(r),
        "gold_chars": total_length(g),
        "overlap_chars": intersection_length(r, g),
    }


def precision_recall(counts: Mapping[str, int]) -> tuple[float, float]:
    p = counts["overlap_chars"] / counts["retrieved_chars"] if counts["retrieved_chars"] else 0.0
    r = counts["overlap_chars"] / counts["gold_chars"] if counts["gold_chars"] else 0.0
    return p, r


def per_query_at_k(retrieved_ranked: Sequence[Span], gold: Sequence[Span], ks: Sequence[int]) -> dict[int, dict]:
    """Precision/recall and raw character counts for each k (top-k prefix)."""
    out = {}
    for k in ks:
        counts = char_overlap(retrieved_ranked[:k], gold)
        p, r = precision_recall(counts)
        out[k] = {"precision": p, "recall": r, **counts}
    return out


def aggregate(per_query: Sequence[Mapping[int, Mapping]], ks: Sequence[int]) -> dict[str, dict]:
    """Macro (primary) and pooled (secondary) averages over queries."""
    n = len(per_query)
    result: dict[str, dict] = {}
    for k in ks:
        rows = [q[k] for q in per_query]
        retrieved = sum(r["retrieved_chars"] for r in rows)
        gold = sum(r["gold_chars"] for r in rows)
        overlap = sum(r["overlap_chars"] for r in rows)
        result[str(k)] = {
            "macro_precision": sum(r["precision"] for r in rows) / n if n else None,
            "macro_recall": sum(r["recall"] for r in rows) / n if n else None,
            "pooled_precision": overlap / retrieved if retrieved else None,
            "pooled_recall": overlap / gold if gold else None,
            "num_queries": n,
        }
    return result


# --------------------------------------------------------------------------
# Abstention
# --------------------------------------------------------------------------
# Positive class = "the system should return nothing" (unanswerable query).
#   correct_abstention_rate = abstained unanswerable / all unanswerable  (= abstention recall)
#   false_abstention_rate   = abstained answerable   / all answerable
#   abstention_precision    = abstained unanswerable / all abstentions   (None if no abstentions)
#   abstention_f1           = harmonic mean of abstention precision and recall
#   balanced_accuracy       = (correct_abstention_rate + (1 - false_abstention_rate)) / 2


def abstention_metrics(answerable_abstained: Sequence[bool], unanswerable_abstained: Sequence[bool]) -> dict:
    n_ans, n_un = len(answerable_abstained), len(unanswerable_abstained)
    false_abs = sum(answerable_abstained)
    correct_abs = sum(unanswerable_abstained)
    car = correct_abs / n_un if n_un else None
    far = false_abs / n_ans if n_ans else None
    total_abs = false_abs + correct_abs
    prec = correct_abs / total_abs if total_abs else None
    if car is None:
        f1 = None  # no unanswerable queries: recall undefined
    elif prec is None or prec + car == 0:
        f1 = 0.0  # no correct abstentions at all
    else:
        f1 = 2 * prec * car / (prec + car)
    bal = (car + (1 - far)) / 2 if car is not None and far is not None else None
    return {
        "num_answerable": n_ans,
        "num_unanswerable": n_un,
        "false_abstentions": false_abs,
        "correct_abstentions": correct_abs,
        "correct_abstention_rate": car,
        "false_abstention_rate": far,
        "abstention_precision": prec,
        "abstention_recall": car,
        "abstention_f1": f1,
        "balanced_accuracy": bal,
    }
