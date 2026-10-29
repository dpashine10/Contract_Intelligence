"""End-to-end evaluation on a small synthetic benchmark (no real data needed)."""

import copy
import json
from pathlib import Path

import pytest

from app.config import ConfigError, load_config, validate_config
from app.data import assign_split
from app.evaluate import run_evaluation
from app.pipeline import DataError

TOPICS = ["sublicense", "termination", "royalty", "exclusivity", "warranty", "insurance"]


def make_fixture(root: Path) -> dict:
    corpus = root / "corpus"
    corpus.mkdir()
    texts = {}
    for i, topic in enumerate(TOPICS):
        filler = " ".join(f"word{j}" for j in range(40))
        texts[f"doc{i}.txt"] = f"{filler} The {topic} clause says {topic} rules apply here. {filler}"
        (corpus / f"doc{i}.txt").write_text(texts[f"doc{i}.txt"], encoding="utf-8")

    tests = []
    for q in range(194):  # mini() keeps all when there are exactly 194
        name = f"doc{q % len(TOPICS)}.txt"
        topic = TOPICS[q % len(TOPICS)]
        text = texts[name]
        start = text.index(f"The {topic} clause")
        end = text.index("apply here.") + len("apply here.")
        tests.append({"query": f"Q{q}: what does the {topic} clause say?", "snippets": [{"file_path": f"cuad/{name}", "span": [start, end], "answer": text[start:end]}]})
    (root / "bench.json").write_text(json.dumps({"tests": tests}), encoding="utf-8")

    unanswerable = []
    for i in range(30):
        question = f"What is the stock price of company {i} on the moon?"
        unanswerable.append({"id": f"u{i}", "question": question, "label": "unanswerable", "split": assign_split(question), "rationale": "synthetic"})
    (root / "unans.json").write_text(json.dumps(unanswerable), encoding="utf-8")

    return {
        "name": "test_cfg",
        "data": {"benchmark_path": str(root / "bench.json"), "corpus_root": str(corpus), "unanswerable_path": str(root / "unans.json")},
        "chunking": {"method": "fixed_tokens", "chunk_size": 16, "chunk_overlap": 4},
        "retrieval": {"method": "bm25_okapi", "k1": 1.5, "b": 0.75, "epsilon": 0.25},
        "abstention": {"rule": "top1_score_threshold", "threshold": None},
        "evaluation": {"ks": [1, 2, 4], "retrieval_depth": 4},
        "ui": {"display_top_k": 2},
        "output_root": str(root / "runs"),
    }


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # test ledger is written relative to cwd
    return make_fixture(tmp_path)


def test_train_run_writes_expected_artifacts(config):
    run_dir = run_evaluation(config, ["train"])
    names = {p.name for p in run_dir.iterdir()}
    assert {"config.yaml", "frozen_config.yaml", "run_metadata.json", "metrics.json", "calibration.json",
            "per_query_answerable.jsonl", "per_query_unanswerable.jsonl", "path_resolution.json",
            "corpus_manifest.json", "errors.json"} <= names
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    assert meta["status"] == "completed"
    assert meta["num_indexed_documents"] == len(TOPICS)
    assert meta["path_resolution_counts"] == {"exact": len(TOPICS)}
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert "test" not in metrics  # train-only run never touches test
    assert set(metrics["train"]["retrieval"]) == {"1", "2", "4"}
    row = json.loads((run_dir / "per_query_answerable.jsonl").read_text().splitlines()[0])
    assert {"query", "split", "gold", "top_score", "metrics_at_k", "ranking", "abstained", "first_gold_doc_rank"} <= set(row)
    assert all(json.loads(line)["split"] == "train" for line in (run_dir / "per_query_answerable.jsonl").read_text().splitlines())
    assert not Path("experiments/test_set_runs.jsonl").exists()


def test_repeated_runs_are_identical_and_never_overwrite(config):
    a = run_evaluation(config, ["train"])
    b = run_evaluation(config, ["train"])
    assert a != b  # new folder each time
    for name in ("per_query_answerable.jsonl", "per_query_unanswerable.jsonl", "calibration.json", "corpus_manifest.json"):
        assert (a / name).read_text() == (b / name).read_text()
    ma, mb = (json.loads((d / "metrics.json").read_text()) for d in (a, b))
    # the only run-specific field is the run id recorded as the threshold source
    ma["abstention_rule"].pop("source"), mb["abstention_rule"].pop("source")
    assert ma == mb


def test_test_split_logged_and_uses_train_threshold(config):
    run_dir = run_evaluation(config, ["train", "test"])
    ledger = Path("experiments/test_set_runs.jsonl").read_text().splitlines()
    assert len(ledger) == 1 and json.loads(ledger[0])["run_dir"] == str(run_dir)
    calibration = json.loads((run_dir / "calibration.json").read_text())
    assert calibration["calibrated_on"]["split"] == "train"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["abstention_rule"]["threshold"] == calibration["threshold"]
    assert {"train", "test"} <= set(metrics)


def test_synthetic_retrieval_finds_gold(config):
    run_dir = run_evaluation(config, ["train"])
    metrics = json.loads((run_dir / "metrics.json").read_text())["train"]
    assert metrics["retrieval"]["4"]["macro_recall"] == 1.0
    assert metrics["abstention"]["correct_abstention_rate"] == 1.0
    assert metrics["abstention"]["false_abstention_rate"] == 0.0


def test_missing_corpus_file_aborts_with_record(config):
    Path(config["data"]["corpus_root"], "doc0.txt").unlink()
    with pytest.raises(DataError):
        run_evaluation(config, ["train"])
    run_dir = next(Path(config["output_root"]).iterdir())
    assert json.loads((run_dir / "run_metadata.json").read_text())["status"] == "failed"
    assert json.loads((run_dir / "errors.json").read_text())[0]["type"] == "missing_file"


def test_tampered_unanswerable_split_is_rejected(config):
    path = Path(config["data"]["unanswerable_path"])
    items = json.loads(path.read_text())
    items[0]["split"] = "test" if items[0]["split"] == "train" else "train"
    path.write_text(json.dumps(items))
    with pytest.raises(DataError):
        run_evaluation(config, ["train"])


def test_test_only_with_null_threshold_is_refused(config):
    with pytest.raises(ValueError):
        run_evaluation(config, ["test"])


def test_config_validation(config):
    bad = copy.deepcopy(config)
    bad["evaluation"]["retrieval_depth"] = 2
    with pytest.raises(ConfigError):
        validate_config(bad)
    bad = copy.deepcopy(config)
    bad["retrieval"]["method"] = "dense"
    with pytest.raises(ConfigError):
        validate_config(bad)


def test_baseline_config_file_is_valid():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/baseline.yaml")
    assert cfg["evaluation"]["ks"] == [1, 2, 4, 8, 16, 32, 64]
    assert cfg["evaluation"]["retrieval_depth"] >= 64
    assert cfg["abstention"]["threshold"] is None
