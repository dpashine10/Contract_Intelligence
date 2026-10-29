"""Self-improvement loop (train split only).

  uv run python -m app.loop --plan configs/loop.yaml

Each iteration:
  1. Diagnose the current best configuration on train (failure buckets at the
     diagnostic k: wrong document / right document but wrong section / partial
     span; plus incorrect abstentions in either direction).
  2. Propose: the first untried candidate in the plan's pool that targets the
     largest bucket (rule-based; candidates were written before the loop ran).
  3. Evaluate current-best + patch on TRAIN (threshold re-calibrated on train).
  4. Accept or reject with the predeclared objective and guardrails.
  5. Append a full record to iterations.jsonl and LOG.md. Nothing is
     overwritten: every evaluation is a new run folder.

The loop never evaluates the test split; ``run_evaluation`` is always called
with ``["train"]``.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from app.config import load_config, validate_config
from app.evaluate import json_safe, run_evaluation

BUCKET_ORDER = ["right_doc_wrong_section", "partial_span", "wrong_document", "incorrect_abstention"]


def deep_merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_patch(config: dict, patch: dict, name: str, output_root: Path) -> dict:
    new = deep_merge(config, patch)
    chunking = new["chunking"]
    if "overlap_fraction" in chunking:  # resolve relative overlap against the current size
        chunking["chunk_overlap"] = int(chunking["chunk_size"] * chunking.pop("overlap_fraction"))
    new["abstention"]["threshold"] = None  # always re-calibrate on train
    new["abstention"].pop("threshold_source", None)
    new["name"] = name
    new["output_root"] = str(output_root)
    validate_config(new)
    return new


def summarise(run_dir: Path, plan: dict) -> dict[str, Any]:
    m = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    train = m["train"]
    retrieval = train["retrieval"]
    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "macro_recall": {k: v["macro_recall"] for k, v in retrieval.items()},
        "macro_precision": {k: v["macro_precision"] for k, v in retrieval.items()},
        "pooled_recall": {k: v["pooled_recall"] for k, v in retrieval.items()},
        "abstention": train["abstention"],
        "abstention_rule": m["abstention_rule"],
        "length_correlation": train["abstention_score_length_correlation_answerable"],
        "diagnosis": train["diagnosis"],
        "primary": retrieval[str(plan["objective"]["primary_k"])]["macro_recall"],
        "precision_guard": retrieval[str(plan["guardrails"]["precision_k"])]["macro_precision"],
    }


def check_guardrails(cand: dict, ref: dict, g: dict, label: str) -> list[str]:
    eps = 1e-12
    failures = []
    if cand["precision_guard"] < ref["precision_guard"] * (1 - g["max_relative_precision_drop"]) - eps:
        failures.append(f"P@{g['precision_k']} {cand['precision_guard']:.4f} < {1 - g['max_relative_precision_drop']:.2f} x {label} {ref['precision_guard']:.4f}")
    far_c, far_r = cand["abstention"]["false_abstention_rate"], ref["abstention"]["false_abstention_rate"]
    if far_c > far_r + g["max_false_abstention_increase"] + eps:
        failures.append(f"false-abstention {far_c:.3f} > {label} {far_r:.3f} + {g['max_false_abstention_increase']}")
    car_c, car_r = cand["abstention"]["correct_abstention_rate"], ref["abstention"]["correct_abstention_rate"]
    if car_c < car_r - g["max_correct_abstention_drop"] - eps:
        failures.append(f"correct-abstention {car_c:.3f} < {label} {car_r:.3f} - {g['max_correct_abstention_drop']}")
    return failures


def decide(cand: dict, best: dict, base: dict, plan: dict) -> tuple[bool, str, dict]:
    obj, g, ab = plan["objective"], plan["guardrails"], plan["abstention_acceptance"]
    d_primary = cand["primary"] - best["primary"]
    d_ba = cand["abstention"]["balanced_accuracy"] - best["abstention"]["balanced_accuracy"]
    corr_c, corr_b = cand["length_correlation"], best["length_correlation"]
    d_corr = (abs(corr_b) - abs(corr_c)) if corr_c is not None and corr_b is not None else None
    detail = {"delta_primary_vs_best": d_primary, "delta_primary_vs_baseline": cand["primary"] - base["primary"],
              "delta_balanced_accuracy_vs_best": d_ba, "length_correlation_reduction_vs_best": d_corr}

    failures = check_guardrails(cand, best, g, "best") + check_guardrails(cand, base, g, "baseline")
    detail["guardrail_failures"] = failures
    if failures:
        return False, "rejected: guardrail failed (" + "; ".join(failures) + ")", detail
    if d_primary >= obj["min_improvement"]:
        return True, f"accepted: R@{obj['primary_k']} +{d_primary:.4f} >= {obj['min_improvement']} and guardrails pass", detail
    if d_primary <= -obj["min_improvement"]:
        return False, f"rejected: R@{obj['primary_k']} changed {d_primary:+.4f} (worse than -{obj['min_improvement']})", detail
    # Primary unchanged within noise: judge the abstention side.
    if d_ba >= ab["min_balanced_accuracy_gain"]:
        return True, f"accepted: primary within noise ({d_primary:+.4f}); balanced accuracy +{d_ba:.3f}", detail
    if d_corr is not None and d_ba >= -ab["max_balanced_accuracy_loss_for_robustness"] and d_corr >= ab["min_length_correlation_reduction"]:
        return True, (f"accepted (robustness): primary within noise ({d_primary:+.4f}); balanced accuracy {d_ba:+.3f}; "
                      f"|corr(score, query length)| reduced by {d_corr:.2f}"), detail
    return False, f"rejected: primary within noise ({d_primary:+.4f}) and abstention gain insufficient (balanced accuracy {d_ba:+.3f}, length-corr reduction {d_corr if d_corr is None else round(d_corr, 2)})", detail


def failure_counts(summary: dict) -> dict[str, int]:
    d = summary["diagnosis"]
    counts = {k: v for k, v in d["answerable_buckets"].items() if k != "full_coverage"}
    counts["incorrect_abstention"] = d["incorrect_abstention"]
    return counts


def propose(summary: dict, pool: list[dict], tried: set[str]) -> tuple[dict | None, str]:
    counts = failure_counts(summary)
    ranked = sorted(counts, key=lambda b: (-counts[b], BUCKET_ORDER.index(b)))
    for bucket in ranked:
        if counts[bucket] == 0:
            continue
        for cand in pool:
            if cand["id"] not in tried and cand["targets"] == bucket:
                return cand, f"largest open failure bucket '{bucket}' ({counts[bucket]} queries; buckets {counts})"
    for cand in pool:
        if cand["id"] not in tried:
            return cand, f"no candidate left for non-empty buckets {counts}; next untried in pool order"
    return None, "pool exhausted"


def fmt(x: float | None, nd: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def run_loop(plan_path: str) -> Path:
    plan = yaml.safe_load(Path(plan_path).read_text(encoding="utf-8"))
    base_config = load_config(plan["base_config"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    loop_dir = Path(plan["output_root"]) / stamp
    loop_dir.mkdir(parents=True, exist_ok=False)
    runs_root = loop_dir / "runs"
    (loop_dir / "plan.yaml").write_text(Path(plan_path).read_text(encoding="utf-8"), encoding="utf-8")
    log_path, md_path = loop_dir / "iterations.jsonl", loop_dir / "LOG.md"

    def record(row: dict) -> None:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")

    pk, gk = plan["objective"]["primary_k"], plan["guardrails"]["precision_k"]
    md = [f"# Improvement loop {stamp}\n", f"Plan: `{plan_path}` · objective: macro R@{pk} (min gain {plan['objective']['min_improvement']}) · guardrails on P@{gk}, false/correct abstention · train split only.\n",
          f"| # | candidate | change | R@{pk} | P@{gk} | false abst. | correct abst. | bal. acc. | len-corr | decision |", "|---|---|---|---|---|---|---|---|---|---|"]

    def md_row(i, cid, change, s, decision):
        a = s["abstention"]
        md.append(f"| {i} | {cid} | {change} | {fmt(s['primary'])} | {fmt(s['precision_guard'])} | {fmt(a['false_abstention_rate'], 3)} | "
                  f"{fmt(a['correct_abstention_rate'], 3)} | {fmt(a['balanced_accuracy'], 3)} | {fmt(s['length_correlation'], 2)} | {decision} |")

    # Iteration 0: baseline, re-evaluated inside the loop.
    cfg0 = apply_patch(base_config, {}, base_config["name"], runs_root)
    base = summarise(run_evaluation(cfg0, ["train"], config_path=plan["base_config"], label="loop:baseline"), plan)
    record({"iteration": 0, "candidate": "baseline", "proposed_by": "baseline", "patch": {}, "config": cfg0, "summary": base, "accepted": True, "reason": "reference"})
    md_row(0, "baseline", f"`{plan['base_config']}`", base, "reference")
    best, best_cfg, tried, applied = base, cfg0, set(), []

    for i in range(1, plan["max_iterations"] + 1):
        cand, why = propose(best, plan["candidates"], tried)
        if cand is None:
            break
        tried.add(cand["id"])
        name = "+".join([base_config["name"], *applied, cand["id"]])
        cfg = apply_patch(best_cfg, cand["patch"], name, runs_root)
        summary = summarise(run_evaluation(cfg, ["train"], label=f"loop:{cand['id']}"), plan)
        ok, reason, detail = decide(summary, best, base, plan)
        record({"iteration": i, "candidate": cand["id"], "targets": cand["targets"], "rationale": cand["rationale"],
                "proposed_by": f"rule-based proposer: {why}", "patch": cand["patch"], "config": cfg, "summary": summary,
                "compared_to_best_run": best["run_id"], "compared_to_baseline_run": base["run_id"], **detail,
                "accepted": ok, "reason": reason})
        md_row(i, cand["id"], json.dumps(cand["patch"]), summary, ("**accept**" if ok else "reject") + f": {reason}")
        print(f"[{i}] {cand['id']}: R@{pk}={summary['primary']:.4f} P@{gk}={summary['precision_guard']:.4f} -> {reason}")
        if ok:
            best, best_cfg = summary, cfg
            applied.append(cand["id"])

    final_cfg = copy.deepcopy(best_cfg)
    (loop_dir / "final_config.yaml").write_text(yaml.safe_dump(final_cfg, sort_keys=False), encoding="utf-8")
    frozen = Path(best["run_dir"]) / "frozen_config.yaml"
    (loop_dir / "final_frozen_config.yaml").write_text(frozen.read_text(encoding="utf-8"), encoding="utf-8")
    final = {"final_run_id": best["run_id"], "applied_changes": applied, "baseline_run_id": base["run_id"],
             "final_train_summary": best, "baseline_train_summary": base, "tried": sorted(tried)}
    (loop_dir / "summary.json").write_text(json.dumps(json_safe(final), indent=2) + "\n", encoding="utf-8")
    md += ["", f"**Final configuration:** baseline + {applied or 'no changes'} (run `{best['run_id']}`).",
           f"Train R@{pk} {fmt(base['primary'])} → {fmt(best['primary'])}; P@{gk} {fmt(base['precision_guard'])} → {fmt(best['precision_guard'])}."]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nLoop folder: {loop_dir}\nFinal: baseline + {applied}")
    return loop_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="improvement-loop")
    parser.add_argument("--plan", default="configs/loop.yaml")
    args = parser.parse_args(argv)
    run_loop(args.plan)
    return 0


if __name__ == "__main__":
    sys.exit(main())
