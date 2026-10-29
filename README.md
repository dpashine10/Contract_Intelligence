# Contract Intelligence Assistant

## Overview

A clause-search system over CUAD contracts (LegalBench-RAG-mini, 194 queries).
For a natural-language question it returns a ranked list of passages, each
with its source contract and original character offsets `[start, end)`, or
returns nothing when no passage scores above an abstention threshold.

It has three parts:

- a CLI and a Streamlit web app;
- an evaluator that reports character-level precision/recall@k and abstention metrics;
- a single-command, train-only self-improvement loop.

Results and analysis are in [REPORT.md](REPORT.md) (rendered copy: `REPORT.pdf`).

## Setup

- Python 3.11 (see `.python-version`); dependencies are locked with [uv](https://docs.astral.sh/uv/).
- No LLM, API key or local model is needed: the application makes no LLM calls.

```bash
uv sync                      # install locked dependencies into .venv

# Download the benchmark and corpus (only cuad.json and corpus/cuad are needed)
uv run hf download awinml/legalbench-rag \
  --repo-type dataset \
  --include "benchmarks/cuad.json" \
  --include "corpus/cuad/**" \
  --local-dir data/raw

# Optional: write the 194-query mini selection with split labels to data/processed/
uv run python -m app.data

# Validate: every gold span document[start:end] must equal its answer text
uv run python -m app.validate_data
```

## How to run

| Task | Command |
|---|---|
| Tests | `uv run pytest` |
| Ask a question (CLI) | `uv run python -m app.cli query "Consider the Endorsement Agreement between ABG-Shaq, LLC and Papa John's International, Inc.; What is the governing law for this contract?"` |
| CLI options | `--top-k N`, `--json`, `--config <file>`, `--show-anyway` (diagnostic: print ranked passages even when abstaining) |
| Web app | `uv run streamlit run interfaces/web_app.py` (final config) |
| Web app, baseline config | `uv run streamlit run interfaces/web_app.py -- --config configs/baseline_frozen.yaml` |
| Improvement loop (train only) | `uv run python -m app.loop --plan configs/loop.yaml` |
| Loop round 2 | `uv run python -m app.loop --plan configs/loop_round2.yaml` (its `base_config` points at the round-1 folder; edit the path if you re-run round 1) |
| Evaluate on train | `uv run python -m app.cli evaluate --config configs/final.yaml --splits train` |
| Evaluate on train + test | `uv run python -m app.cli evaluate --config configs/final_frozen.yaml --splits train test --label "..."` (appends to `experiments/test_set_runs.jsonl`) |

The CLI and web app default to `configs/final_frozen.yaml`.

## Architecture

```
benchmarks/cuad.json ──mini(194) + assign_split──► answerable queries (train/test)
data/unanswerable_queries.json ──────────────────► unanswerable queries (train/test)
corpus/cuad/*.txt (only the files the 194 queries reference)
  │ app/corpus.py     exact or unique-NFC filename match; UTF-8 text mode; text never modified
  ▼
  │ app/retrieval.py  regex word tokens (casefolded) with original offsets
  │                   → windows of N tokens with overlap → BM25Okapi over all chunks
  │                   → optional: re-score the top contract's chunks with a document-local BM25
  │                   → optional: drop query stop-words
  ▼
ranked chunks (file, start, end, score) + score features
  │ app/abstention.py abstain iff score < threshold (threshold calibrated on train)
  ▼
CLI (app/cli.py) · web app (interfaces/web_app.py) · evaluator (app/evaluate.py, app/metrics.py)
                                                       ▲
                    improvement loop (app/loop.py) ────┘ train split only
```

| Module | Role |
|---|---|
| `app/data.py` | Prescribed mini selection and SHA-256 train/test split |
| `app/validate_data.py` | Gold span validation against the corpus |
| `app/corpus.py` | Filename resolution, contract loading, corpus manifest and hashes |
| `app/retrieval.py` | Tokeniser, chunker, BM25 index, document-local re-scoring, score features |
| `app/abstention.py` | Abstention scores (raw / normalised / z-score) and train-only calibration |
| `app/metrics.py` | Character-level precision/recall@k and abstention metrics (formulas in docstring) |
| `app/evaluate.py` | Evaluation runner, failure buckets, run artifacts, test-set ledger |
| `app/loop.py` | Diagnose → propose → evaluate (train) → accept/reject → log |
| `app/pipeline.py`, `app/config.py` | Build the pipeline from YAML; validate configuration |

**Configuration** (`configs/`):

| File | Contents |
|---|---|
| `baseline.yaml` | baseline settings; threshold `null` (calibrate on train) |
| `baseline_frozen.yaml` | baseline with its train-calibrated threshold |
| `loop.yaml` | objective, guardrails and candidate pool; written before any experiment |
| `loop_round2.yaml` | round-2 candidates |
| `final.yaml` | the selected configuration |
| `final_frozen.yaml` | the selected configuration with its train-calibrated threshold |

**Experiment outputs** (`experiments/`; never overwritten, each run is a new folder):

| Path | Contents |
|---|---|
| `bm25_baseline/<run>/` | baseline runs (train calibration run; train+test run) |
| `improvement_loop/<loop>/` | `plan.yaml`, `iterations.jsonl` (full record per iteration), `LOG.md` (table), `summary.json`, `final_config.yaml`, `final_frozen_config.yaml`, `runs/<run>/` |
| `final/<run>/` | the single final train+test evaluation |
| `test_set_runs.jsonl` | ledger of every evaluation that touched the test split |

Each run folder contains:

- `config.yaml`, `frozen_config.yaml`, `calibration.json`;
- `metrics.json`: per split, retrieval at each k (macro and pooled), abstention, failure diagnosis;
- `per_query_answerable.jsonl`, `per_query_unanswerable.jsonl`: top-64 rankings, score features, per-k character counts, abstention decision, failure bucket;
- `path_resolution.json`, `corpus_manifest.json`, `errors.json`;
- `run_metadata.json`: git commit and dirty state, input file hashes, package versions, document and chunk counts.

## Reproducing results

Rankings and metrics are deterministic for a fixed corpus and configuration.

1. Run setup as above (`uv sync`, download, `uv run python -m app.validate_data`).
2. Baseline on train: `uv run python -m app.cli evaluate --config configs/baseline.yaml --splits train` (~2 s).
3. Improvement loop: `uv run python -m app.loop --plan configs/loop.yaml` (~25 s, 8 iterations), then round 2 (~10 s).
4. Final on train: `uv run python -m app.cli evaluate --config configs/final.yaml --splits train` (~3 s).
5. Train + test runs, only if you need to reproduce the test figures (each adds a ledger line):
   - `uv run python -m app.cli evaluate --config configs/baseline_frozen.yaml --splits train test`
   - `uv run python -m app.cli evaluate --config configs/final_frozen.yaml --splits train test`
6. Compare each new `metrics.json` with the run IDs cited in REPORT.md.

## Assumptions

- **Corpus scope.** Only contracts referenced by any gold snippet of the 194 mini queries are indexed (29; counted at run time). Unanswerable labels mean "not answerable from this indexed corpus".
- **Offsets.** Zero-based, half-open `[start, end)` over the file read in Python text mode with UTF-8. Tokenisation never changes offsets.
- **Filename normalisation.** One benchmark path is Unicode NFD while the file is NFC. Paths are matched exactly first, then by NFC name if exactly one file matches. Missing or ambiguous files abort the run and are recorded.
- **Metrics.**
  - k = number of retrieved chunks, k ∈ {1, 2, 4, 8, 16, 32, 64}.
  - Overlapping retrieved intervals are merged before scoring; this equals the LegalBench-RAG reference formula when nothing overlaps.
  - The primary aggregate is the macro average over queries; pooled is secondary.
  - Precision is 0 when nothing is retrieved.
  - Retrieval metrics use the full ranking even when the system abstains; abstention is measured separately in both directions.
- **Abstention threshold.** Calibrated on train only by maximising balanced accuracy (ties: lower false-abstention rate, then lower threshold), and frozen before any test query is scored.
- **Objective.** Macro Recall@8 on train, with guardrails on P@8 and on both abstention error rates. Declared in `configs/loop.yaml` before any experiment; see REPORT.md.
- **Baseline parameters.** k1 1.5, b 0.75, epsilon 0.25, 256-token chunks, 32 overlap. These are fixed engineering defaults chosen before seeing results, not values prescribed by the paper.
- **Test-set use.** Two evaluations: baseline once (before the loop existed) and the frozen final configuration once. Both are recorded in `experiments/test_set_runs.jsonl`.
- **Query IDs.** `qNNN` is a query's position in the prescribed mini selection.

## AI tool usage

**Development.** AI was used as a coding assistant through
the entire development process for all work from the BM25 baseline onward:

- inspecting the repository and the assignment specification;
- implementing the BM25 baseline, character-level metrics, abstention calibration, evaluation runner, improvement loop, CLI, Streamlit app and their tests;
- drafting the objective, guardrails and candidate pool in `configs/loop.yaml` before the loop ran;
- running the baseline, improvement-loop and final evaluations;
- drafting README.md and REPORT.pdf from the run artifacts, auditing the reported numbers against those artifacts, and exporting REPORT.pdf.

The project decisions were specified by me, and the assistant implemented them:

- index only the contracts referenced by the 194 queries;
- follow the LegalBench-RAG character-level metric protocol with k ∈ {1, 2, 4, 8, 16, 32, 64};
- report abstention in both directions;
- use the stated baseline BM25 and chunking parameters;
- build both a CLI and a web app.

This section covers work from the BM25 baseline onward. The earlier commits (`2634fe3`: data download, mini selection, split and validation; `4b27a8e`: the unanswerable query set) are not covered here.

**Runtime.** The evaluated system makes no LLM calls: it is BM25 retrieval plus a score threshold, and it does no query rewriting, reranking with a model, or answer generation.
