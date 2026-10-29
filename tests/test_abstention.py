import math

import pytest

from app.abstention import calibrate_threshold, candidate_thresholds, should_abstain


def test_rule():
    assert should_abstain(None, 1.0)  # empty ranking -> abstain
    assert should_abstain(0.99, 1.0)
    assert not should_abstain(1.0, 1.0)  # strictly below abstains


def test_candidates_include_never_and_always():
    c = candidate_thresholds([3.0, 1.0, 3.0, 2.0])
    assert c == [-math.inf, 1.5, 2.5, math.inf]


def test_calibration_separates_clean_data():
    result = calibrate_threshold([10.0, 12.0, 9.0], [2.0, 3.0])
    assert 3.0 < result["threshold"] < 9.0
    m = result["train_metrics_at_threshold"]
    assert m["false_abstention_rate"] == 0 and m["correct_abstention_rate"] == 1


def test_calibration_uses_midpoint_between_scores():
    assert calibrate_threshold([9.0], [3.0])["threshold"] == 6.0


def test_calibration_tie_break_prefers_lower_false_abstention():
    # Candidates: 3 -> BA 0.75 (FAR 0); 8.5 -> BA 0.75 (FAR 0.5). Pick 3.
    result = calibrate_threshold([5.0, 10.0], [1.0, 7.0])
    assert result["threshold"] == 3.0
    assert result["train_metrics_at_threshold"]["false_abstention_rate"] == 0.0


def test_calibration_requires_both_classes():
    with pytest.raises(ValueError):
        calibrate_threshold([1.0], [])
