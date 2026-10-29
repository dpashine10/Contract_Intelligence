import json
from pathlib import Path

import yaml

from app.loop import apply_patch, decide, deep_merge, propose, run_loop
from tests.test_evaluate import make_fixture

PLAN = {
    "objective": {"primary_metric": "macro_recall", "primary_k": 8, "min_improvement": 0.01},
    "guardrails": {"precision_k": 8, "max_relative_precision_drop": 0.10, "max_false_abstention_increase": 0.05, "max_correct_abstention_drop": 0.05},
    "abstention_acceptance": {"min_balanced_accuracy_gain": 0.02, "max_balanced_accuracy_loss_for_robustness": 0.01, "min_length_correlation_reduction": 0.30},
}


def S(r=0.40, p=0.016, far=0.25, car=0.85, corr=0.8):
    return {"primary": r, "precision_guard": p, "length_correlation": corr,
            "abstention": {"false_abstention_rate": far, "correct_abstention_rate": car, "balanced_accuracy": (car + 1 - far) / 2}}


def test_accept_primary_gain():
    ok, reason, _ = decide(S(r=0.42), S(), S(), PLAN)
    assert ok and "accepted" in reason


def test_reject_small_gain_as_noise():
    ok, _, _ = decide(S(r=0.405), S(), S(), PLAN)
    assert not ok


def test_reject_primary_loss():
    ok, reason, _ = decide(S(r=0.30, p=0.05), S(), S(), PLAN)
    assert not ok and "worse" in reason


def test_precision_guardrail_blocks_gain():
    ok, reason, d = decide(S(r=0.50, p=0.010), S(), S(), PLAN)
    assert not ok and "guardrail" in reason and d["guardrail_failures"]


def test_guardrail_checked_against_baseline_too():
    base, best = S(far=0.10), S(far=0.14)
    ok, reason, _ = decide(S(r=0.45, far=0.17), best, base, PLAN)  # +0.03 vs best ok, +0.07 vs baseline not
    assert not ok and "baseline" in reason


def test_correct_abstention_guardrail_allows_one_query():
    assert decide(S(r=0.45, car=0.80), S(), S(), PLAN)[0]  # -0.05 allowed
    assert not decide(S(r=0.45, car=0.75), S(), S(), PLAN)[0]


def test_abstention_only_gain_accepted():
    ok, reason, _ = decide(S(far=0.15), S(), S(), PLAN)
    assert ok and "balanced accuracy" in reason


def test_robustness_acceptance():
    ok, reason, _ = decide(S(corr=0.2, far=0.26), S(), S(), PLAN)  # BA -0.005, corr -0.6
    assert ok and "robustness" in reason
    ok, _, _ = decide(S(corr=0.6), S(), S(), PLAN)  # corr reduction only 0.2
    assert not ok


def test_propose_targets_largest_bucket_then_falls_back():
    summary = {"diagnosis": {"answerable_buckets": {"wrong_document": 0, "right_doc_wrong_section": 5, "partial_span": 9, "full_coverage": 3}, "incorrect_abstention": 2}}
    pool = [{"id": "a", "targets": "right_doc_wrong_section"}, {"id": "b", "targets": "partial_span"}, {"id": "c", "targets": "incorrect_abstention"}]
    assert propose(summary, pool, set())[0]["id"] == "b"
    assert propose(summary, pool, {"b"})[0]["id"] == "a"
    assert propose(summary, pool, {"a", "b"})[0]["id"] == "c"
    assert propose(summary, pool, {"a", "b", "c"})[0] is None


def test_apply_patch_resolves_overlap_fraction_and_resets_threshold(tmp_path):
    cfg = make_fixture(tmp_path)
    cfg["abstention"]["threshold"] = 3.0
    new = apply_patch(cfg, {"chunking": {"overlap_fraction": 0.5}}, "n", tmp_path / "r")
    assert new["chunking"]["chunk_overlap"] == 8 and "overlap_fraction" not in new["chunking"]
    assert new["abstention"]["threshold"] is None
    assert cfg["chunking"]["chunk_overlap"] == 4  # original untouched
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 5}}) == {"a": {"b": 5, "c": 2}}


def test_loop_end_to_end_train_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = make_fixture(tmp_path)
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(cfg))
    plan = dict(PLAN, base_config=str(tmp_path / "base.yaml"), output_root=str(tmp_path / "loop"), max_iterations=2,
                objective=dict(PLAN["objective"], primary_k=4), guardrails=dict(PLAN["guardrails"], precision_k=4),
                candidates=[{"id": "small", "targets": "partial_span", "rationale": "r", "patch": {"chunking": {"chunk_size": 8, "chunk_overlap": 0}}},
                            {"id": "z", "targets": "incorrect_abstention", "rationale": "r", "patch": {"abstention": {"score": "zscore_top1"}}}])
    cfg["evaluation"]["diagnostic_k"] = 4
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(cfg))
    (tmp_path / "plan.yaml").write_text(yaml.safe_dump(plan))
    loop_dir = run_loop(str(tmp_path / "plan.yaml"))
    rows = [json.loads(l) for l in (loop_dir / "iterations.jsonl").read_text().splitlines()]
    assert [r["iteration"] for r in rows] == [0, 1, 2]
    for r in rows:
        meta = json.loads((Path(r["summary"]["run_dir"]) / "run_metadata.json").read_text())
        assert meta["splits_evaluated"] == ["train"]
        assert {"config", "summary", "accepted", "reason", "proposed_by"} <= set(r)
    assert not Path("experiments/test_set_runs.jsonl").exists()  # loop never touches test
    for f in ("LOG.md", "summary.json", "final_config.yaml", "final_frozen_config.yaml", "plan.yaml"):
        assert (loop_dir / f).exists()
