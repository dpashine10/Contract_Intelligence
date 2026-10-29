"""Load and check a pipeline configuration (YAML)."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REQUIRED = {
    "data": ["benchmark_path", "corpus_root", "unanswerable_path"],
    "chunking": ["method", "chunk_size", "chunk_overlap"],
    "retrieval": ["method", "k1", "b", "epsilon"],
    "abstention": ["rule", "threshold"],
    "evaluation": ["ks", "retrieval_depth"],
    "ui": ["display_top_k"],
}
SUPPORTED = {
    ("chunking", "method"): {"fixed_tokens"},
    ("retrieval", "method"): {"bm25_okapi"},
    ("abstention", "rule"): {"top1_score_threshold"},
}


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict) or "name" not in config or "output_root" not in config:
        raise ConfigError("config needs 'name' and 'output_root'")
    for section, keys in REQUIRED.items():
        if section not in config:
            raise ConfigError(f"missing section '{section}'")
        for key in keys:
            if key not in config[section]:
                raise ConfigError(f"missing '{section}.{key}'")
    for (section, key), allowed in SUPPORTED.items():
        if config[section][key] not in allowed:
            raise ConfigError(f"{section}.{key}={config[section][key]!r}; supported: {sorted(allowed)}")
    # Optional keys (absent = baseline behaviour, so old configs stay valid).
    r, a = config["retrieval"], config["abstention"]
    if r.get("rerank", "none") not in {"none", "doc_local_bm25"}:
        raise ConfigError("retrieval.rerank must be 'none' or 'doc_local_bm25'")
    if not isinstance(r.get("rerank_docs", 1), int) or r.get("rerank_docs", 1) < 1:
        raise ConfigError("retrieval.rerank_docs must be a positive integer")
    if not isinstance(r.get("query_stopwords", False), bool):
        raise ConfigError("retrieval.query_stopwords must be true/false")
    if a.get("score", "raw_top1") not in {"raw_top1", "normalized_top1", "zscore_top1"}:
        raise ConfigError("abstention.score must be raw_top1, normalized_top1 or zscore_top1")
    ks = config["evaluation"]["ks"]
    if not ks or any(not isinstance(k, int) or k <= 0 for k in ks):
        raise ConfigError("evaluation.ks must be positive integers")
    if config["evaluation"]["retrieval_depth"] < max(ks):
        raise ConfigError("evaluation.retrieval_depth must be >= max(ks)")


def config_hash(config: dict[str, Any]) -> str:
    """Stable hash of the pipeline settings (data paths and output dir excluded)."""
    relevant = {k: v for k, v in config.items() if k not in {"data", "output_root"}}
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode("utf-8")).hexdigest()


def with_threshold(config: dict[str, Any], threshold: float, source: str) -> dict[str, Any]:
    frozen = copy.deepcopy(config)
    frozen["abstention"]["threshold"] = threshold
    frozen["abstention"]["threshold_source"] = source
    return frozen
