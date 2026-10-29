"""Abstention rule: return nothing when the best passage scores too low.

Rule ("top1_score_threshold"): abstain iff the ranking is empty or the
abstention score is strictly below ``threshold``. Score options
(``abstention.score``):

  raw_top1        top-1 corpus-level BM25 score (baseline). Grows with query length.
  normalized_top1 top1 / max_possible, where max_possible = sum over query terms
                  of idf(t) * (k1 + 1) is BM25's upper bound for that query.
                  In [0, 1]; does not grow just because the query is longer.
  zscore_top1     (top1 - mean) / std over all chunk scores for the query:
                  how far the best chunk stands out from the rest.

The threshold is calibrated on TRAIN queries only (answerable + unanswerable)
and frozen before any test query is scored. Calibration criterion: maximise
balanced accuracy = mean(correct-abstention rate, 1 - false-abstention rate),
which weighs both error directions equally despite 140 vs 20 train queries.
Ties -> lower false-abstention rate -> lower threshold (prefer answering).
"""

from __future__ import annotations

import math
from typing import Sequence

from app.metrics import abstention_metrics


SCORE_KINDS = ("raw_top1", "normalized_top1", "zscore_top1")


def abstention_score(features: dict, kind: str = "raw_top1") -> float | None:
    """Score used by the rule; None (=> abstain) when it is undefined."""
    top1 = features.get("top1")
    if top1 is None:
        return None
    if kind == "raw_top1":
        return top1
    if kind == "normalized_top1":
        mp = features.get("max_possible") or 0.0
        return top1 / mp if mp > 0 else None
    if kind == "zscore_top1":
        std = features.get("std") or 0.0
        return (top1 - features["mean"]) / std if std > 0 else None
    raise ValueError(f"unknown abstention score {kind!r}; options: {SCORE_KINDS}")


def should_abstain(top_score: float | None, threshold: float) -> bool:
    return top_score is None or top_score < threshold


def candidate_thresholds(scores: Sequence[float]) -> list[float]:
    """-inf (never abstain), midpoints between distinct scores, and +inf."""
    distinct = sorted(set(scores))
    mids = [(a + b) / 2 for a, b in zip(distinct, distinct[1:])]
    return [-math.inf, *mids, math.inf]


def calibrate_threshold(
    answerable_top_scores: Sequence[float | None],
    unanswerable_top_scores: Sequence[float | None],
) -> dict:
    """Pick the threshold that maximises balanced accuracy on the given (train) data."""
    if not answerable_top_scores or not unanswerable_top_scores:
        raise ValueError("calibration needs both answerable and unanswerable train queries")
    observed = [s for s in [*answerable_top_scores, *unanswerable_top_scores] if s is not None]
    best = None
    for t in candidate_thresholds(observed):
        m = abstention_metrics(
            [should_abstain(s, t) for s in answerable_top_scores],
            [should_abstain(s, t) for s in unanswerable_top_scores],
        )
        key = (m["balanced_accuracy"], -m["false_abstention_rate"], -t)
        if best is None or key > best[0]:
            best = (key, t, m)
    _, threshold, metrics = best
    return {
        "threshold": threshold,
        "criterion": "max balanced accuracy on train; ties -> lower false-abstention rate -> lower threshold",
        "train_metrics_at_threshold": metrics,
        "num_candidates": len(candidate_thresholds(observed)),
    }
