# Evaluation Report: Contract Clause Search

Tags used in this report: **[Measured]** = computed from a run artifact named in
the text; **[Hypothesis]** = my explanation of a measured pattern, not
verified by an experiment.

## 1. Summary

The system ranks fixed-size passages of the 29 CUAD contracts referenced by
LegalBench-RAG-mini with BM25. It returns nothing when the top score falls
below a threshold that was calibrated on train. A train-only improvement loop
ran 11 logged iterations beyond the baseline and accepted two changes:

- re-scoring chunks inside the top-ranked contract with document-local BM25;
- dropping stop-words from queries.

**[Measured] Headline test results** (54 answerable and 12 unanswerable test
queries, single final test run `experiments/final/20261029T191441576809Z_69310a62`):

| Metric | Baseline | Final |
|---|---|---|
| Macro character Recall@8 (primary) | 0.352 | **0.608** |
| Macro character Recall@64 (see Section 5.1) | 0.745 | 0.993 |
| Macro character Precision@8 | 0.015 | 0.024 |
| Correct abstention (unanswerable) | 11/12 | 10/12 |
| False abstention (answerable) | 14/54 | 13/54 |

Recall@8 is the primary retrieval comparison. Recall@64 should not be read as
"99.3% of questions answered". In the final configuration the top 64 chunks
are the entire gold contract for 172 of 194 queries, so that figure mostly
shows that the right contract was found (Section 5.1).

The final system is not better on every measure:

- On test it wrongly rejected one fewer answerable query.
- It also correctly rejected one fewer unanswerable query.
- With only 12 unanswerable test queries, neither difference means much.

The abstention threshold τ ≈ 46.336014 is a raw top-1 BM25 score, not a
probability.

The abstention rule is the weakest part:

- it rejects whole contracts;
- it cannot tell a "near-miss" question about a real contract from an answerable one;
- it abstains on short, natural questions.

## 2. System design

| Stage | Baseline | Final | Alternatives tried (Section 6) |
|------|----------------|----------------|------------|
| Corpus | 29 contracts referenced by the 194 mini queries, read as UTF-8 text, original offsets kept | same | n/a |
| Tokens | regex `\w+`, casefold, offsets into original text; no stemming | same | n/a |
| Chunks | 256 tokens, 32-token overlap (848 chunks; median retrieved chunk about 1,600 characters) | same | 64, 128, 384 tokens; 0 and 50% overlap |
| Ranking | BM25Okapi over all chunks (k1 1.5, b 0.75, epsilon 0.25); ties broken by file name, then offset | first stage the same, then the top-scoring contract's chunks are re-scored with a BM25 built only over that contract and placed first | re-score top 2 contracts |
| Query | all tokens | stop-words removed (fixed 119-word list including "consider") | n/a |
| Abstention | abstain if top-1 corpus BM25 score < τ; τ maximises balanced accuracy on train | same rule, τ re-calibrated on train = 46.336 | length-normalised top-1 score; z-score of top-1 |

**Why document-local re-scoring.**

- **[Measured]** The baseline found the right contract almost every time: a gold-document chunk at rank 1 for 140/140 train queries. But the top-1 chunk was the contract's opening chunk for 127/140 train queries.
- **[Hypothesis]** Every benchmark query begins with `Consider the <Agreement> between <A> and <B>`. Party names and title words dominate corpus-level BM25 because they are rare across contracts. Inside one contract they appear in almost every chunk, so document-local IDF down-weights them and the clause-specific words decide the ranking.

**Configuration:**

| File | Purpose |
|---|---|
| `configs/baseline.yaml` | baseline settings |
| `configs/baseline_frozen.yaml` | baseline with train threshold |
| `configs/loop.yaml` | objective, guardrails and candidate pool (written before any experiment) |
| `configs/loop_round2.yaml` | round-2 candidates |
| `configs/final.yaml` | final settings |
| `configs/final_frozen.yaml` | final with train threshold; used by the CLI, web app and final test run |

## 3. LLM use

**No LLM is used at runtime** (no query rewriting, reranking or answer
generation). The evaluated system is BM25 retrieval plus a score threshold.

- **Why not:** the pipeline is lexical and fully deterministic, so every number can be reproduced offline, and no contract text leaves the machine. That matters for confidential agreements.
- **What it would cost:** the obvious LLM use is a reranker over the chunks of the retrieved contract.
  - Final train Recall@32 is 0.93, but for 96 of 194 queries the top 32 already *is* the whole gold contract (Section 5.1). So that figure shows the right contract is found, not that the clause ranks near the top.
  - The remaining problem is ordering chunks within the right contract, which is what a reranker would address.
  - It would add per-query cost and latency and, with a hosted model, a data-privacy question.

AI assistance during development is disclosed in the README ("AI tool usage").

## 4. Evaluation setup

**Data [Measured]:**

- 194 queries and 272 gold spans; every `text[start:end]` equals the annotated answer, re-checked on every run.
- Train/test split by the prescribed SHA-256 rule:
  - answerable: 140 train / 54 test;
  - unanswerable: 20 train / 12 test (32 total, committed in `4b27a8e` before the loop).
- All gold spans for a query lie in one contract.
- Gold span length: min 67, median 316.5, max 1,756 characters.
- The index contains exactly the 29 contracts referenced by the 194 queries (848 chunks in both baseline and final configurations).

**Character-level metrics.** These follow the LegalBench-RAG reference code. k is the number of retrieved chunks, k ∈ {1, 2, 4, 8, 16, 32, 64}. For one query, with R the union of the top-k chunk intervals and G the union of gold intervals (both per file, half-open `[start, end)`):

```
precision@k = |R ∩ G| / |R|      (0 if nothing retrieved)
recall@k    = |R ∩ G| / |G|
```

- **Merging overlaps.** The reference code sums snippet lengths, which is only correct because its chunks never overlap. Our chunks overlap, so intervals are merged per contract first.
  - A unit test shows the two formulas agree when nothing overlaps.
  - **[Measured]** In the final configuration all 140 train queries have overlapping chunks in their top 8. Summing lengths instead of merging would inflate train R@8 from 0.607 to 0.650, with 19 queries above 1.0.
- **Aggregation.** Primary = **macro** mean over queries (one weight per query, as in the reference code). Pooled (Σ overlap / Σ retrieved, Σ overlap / Σ gold) is reported as secondary.
- **Retrieval vs abstention.** Retrieval metrics use the full ranking even when the system abstains, so the two are measured separately.

**Abstention metrics** (positive class = should abstain):

| Metric | Definition |
|---|---|
| correct-abstention rate | abstained unanswerable / all unanswerable (abstention recall) |
| false-abstention rate | abstained answerable / all answerable |
| abstention precision | abstained unanswerable / all abstentions |
| abstention F1 | harmonic mean of abstention precision and recall |
| balanced accuracy | (correct-abstention rate + 1 − false-abstention rate) / 2 |

**Use of the split:**

- All development, calibration and selection used train only.
- The test split was evaluated **twice in total**: once for the baseline before the loop existed, once for the frozen final configuration. Both runs are recorded in `experiments/test_set_runs.jsonl`.
- **[Measured]** Of the 16 recorded evaluation runs under `experiments/`, only those two touched the test split. The other 14 were train-only: the baseline calibration run, 9 round-1 loop runs and 4 round-2 loop runs.
- Train abstention numbers are in-sample, because τ is fitted on those same queries.

## 5. Results

**[Measured] Retrieval, macro average** (train n = 140, test n = 54):

| k | Baseline train P / R | Final train P / R | Baseline test P / R | Final test P / R |
|---|---|---|---|---|
| 1 | 0.0086 / 0.0439 | 0.0109 / 0.0462 | 0.0045 / 0.0441 | 0.0205 / 0.0812 |
| 2 | 0.0155 / 0.1078 | 0.0162 / 0.1162 | 0.0076 / 0.0948 | 0.0199 / 0.1415 |
| 4 | 0.0138 / 0.2044 | 0.0211 / 0.3138 | 0.0084 / 0.2171 | 0.0154 / 0.2391 |
| **8** | 0.0161 / **0.3858** | 0.0236 / **0.6073** | 0.0146 / **0.3524** | 0.0239 / **0.6076** |
| 16 | 0.0103 / 0.5008 | 0.0166 / 0.7883 | 0.0104 / 0.4424 | 0.0188 / 0.8444 |
| 32 | 0.0065 / 0.6319 | 0.0103 / 0.9327 | 0.0073 / 0.6180 | 0.0113 / 0.9788 |
| 64 | 0.0039 / 0.7138 | 0.0055 / 0.9874 | 0.0045 / 0.7450 | 0.0059 / 0.9929 |

Pooled Recall@8:

| | Baseline | Final |
|---|---|---|
| Train | 0.374 | 0.532 |
| Test | 0.317 | 0.502 |

### 5.1 How to read Recall@16 to Recall@64

**[Measured] Why the top 64 is often the whole contract.** In the final configuration:

- the first stage ranks a chunk of the gold contract first for every query (140/140 train, 54/54 test);
- re-scoring then places *all* chunks of that contract ahead of any other contract;
- 27 of the 29 indexed contracts have at most 64 chunks (the exceptions have 127 and 122).

So whenever the gold contract has at most k chunks, the top-k list is the entire contract and recall is 1 by construction.

| k | Queries (of 194) whose top-k is the entire gold contract |
|---|---|
| 8 | 9 |
| 16 | 43 |
| 32 | 96 |
| 64 | 172 |

**What this means.**

- **High R@64 is not precise retrieval.** The final Recall@64 of 0.993 mostly reflects finding the right contract and returning it in full. It does not mean that 99.3% of questions were answered, or that the answering clause ranks near the top.
- **Recall@16 and Recall@32 are partly affected** in the same way.
- **The baseline-vs-final comparison at large k is not like-for-like.** **[Measured]** The baseline ranked chunks from several contracts: its top 64 never consisted only of gold-contract chunks (0/140 train, 0/54 test).
- **The two large contracts** account for 22 queries. On those, final R@64 is 0.896 on train (17 queries) and 0.924 on test (5 queries).
- **Recall@8 is the primary comparison.** Only 9 of 194 queries have a gold contract small enough to be returned whole at k = 8.

### 5.2 Abstention

**[Measured] Abstention** (τ calibrated on train only):

| Run | τ | Correct abstention | False abstention | Precision | F1 | Balanced acc. |
|---|---|---|---|---|---|---|
| Baseline train | 60.97 | 17/20 (0.850) | 36/140 (0.257) | 0.321 | 0.466 | 0.796 |
| Final train | 46.34 | 16/20 (0.800) | 37/140 (0.264) | 0.302 | 0.438 | 0.768 |
| Baseline test | 60.97 | 11/12 (0.917) | 14/54 (0.259) | 0.440 | 0.595 | 0.829 |
| Final test | 46.34 | 10/12 (0.833) | 13/54 (0.241) | 0.435 | 0.571 | 0.796 |

- **What τ is.** τ is a threshold on the raw top-1 BM25 score: 46.336014 for the final configuration, 60.972 for the baseline. It is not a probability, and its scale depends on query length and corpus statistics.
- **A trade-off, not a win.** Between baseline and final, test shows one fewer false abstention (14 → 13 of 54) and one fewer correct abstention (11 → 10 of 12). Train balanced accuracy is lower for the final configuration (0.768 vs 0.796).

### 5.3 Train vs test

**[Measured]** Final Recall@8 is 0.607 on train and 0.608 on test, so the primary metric shows no sign that the two accepted changes overfit train. Recall@64 (0.987 vs 0.993) is not informative here, for the reasons in Section 5.1.

- **[Hypothesis]** Both changes exploit structure shared by every query (the "Consider the…" template), not train-specific content.
- Test abstention numbers rest on only 12 unanswerable queries; one query moves the correct-abstention rate by 8.3 points.
- Test P@1 / R@1 are noisy for the same reason (54 queries).

## 6. Iteration history

**Objective, declared in `configs/loop.yaml` before any experiment.** The primary metric is macro character Recall@8 on train answerable queries.

- Why recall: missing the governing clause costs a legal user more than reading extra text.
- Why k = 8: about what a reviewer will scan, and it is a LegalBench-RAG k value.
- A gain must be ≥ 0.01 to count; smaller changes are treated as noise.

**Guardrails.** Each is checked against both the baseline and the current best, so they cannot drift over iterations:

- macro P@8 may not fall by more than 10% (relative);
- false-abstention rate may not rise by more than 0.05;
- correct-abstention rate may not fall by more than 0.05 (one train query).

**Changes that leave retrieval unchanged** (abstention-only changes) are accepted only if balanced accuracy rises by ≥ 0.02, or if it falls by ≤ 0.01 while the absolute correlation between the abstention score and query length falls by ≥ 0.30.

**How the loop works** (`uv run python -m app.loop --plan configs/loop.yaml`):

1. Evaluate the current best configuration on train.
2. Bucket every failure: wrong document, right document but wrong section, partial span, incorrect abstention.
3. Pick the first untried candidate from a pool written before the loop ran, targeting the largest bucket. This is a rule-based proposer; the candidate pool was drafted with the AI coding assistant (see README).
4. Apply it as a patch to the current best, evaluate on train with τ re-calibrated, and accept or reject by the rule above.

Every iteration is logged with its full config, run ID, metrics, deltas, guardrail checks and reason, in `iterations.jsonl` and `LOG.md`. Each evaluation is a new run folder, and the loop calls the evaluator with the train split only.

- Round 1: `experiments/improvement_loop/20261029T191235Z`.
- Round 2: `experiments/improvement_loop/20261029T191347Z`. It was added after round 1 because the loop is greedy and chunk size had only been tested *before* re-scoring was accepted.
- **Known labelling issue.** Row 0 of round 2's `LOG.md` is labelled `configs/baseline.yaml`, but its actual reference is the round-1 best (run `20261029T191347475686Z_127bfd15`, re-evaluating the round-1 winner). `iterations.jsonl` and `run_metadata.json` record the actual config. I left the historical log unchanged; the label is fixed in `app/loop.py` for future runs.

**[Measured] All iterations (train):**

| # | Change (applied to current best) | R@1 | R@8 | R@64 | P@8 | False abst. | Correct abst. | Bal. acc. | len-corr | Decision |
|----|--------------|----|----|----|-----|----|----|----|----|--------------|
| 0 | baseline | 0.044 | 0.386 | 0.714 | 0.0161 | 0.257 | 0.850 | 0.796 | 0.78 | reference |
| 1 | chunks 128 / overlap 16 | 0.006 | 0.213 | 0.586 | 0.0175 | 0.257 | 0.900 | 0.821 | 0.73 | reject: R@8 −0.173 |
| 2 | document-local re-scoring (top 1 contract) | 0.058 | 0.574 | 0.986 | 0.0220 | 0.257 | 0.850 | 0.796 | 0.78 | **accept**: R@8 +0.188 |
| 3 | chunks 64 / overlap 8 | 0.003 | 0.223 | 0.777 | 0.0246 | 0.414 | 1.000 | 0.793 | 0.58 | reject: false abstention 0.414 |
| 4 | query stop-words removed | 0.046 | 0.607 | 0.987 | 0.0236 | 0.264 | 0.800 | 0.768 | 0.71 | **accept**: R@8 +0.033 (correct abstention −1 query, within guardrail) |
| 5 | abstain on length-normalised score | 0.046 | 0.607 | 0.987 | 0.0236 | 0.114 | 0.550 | 0.718 | −0.08 | reject: correct abstention 0.55 |
| 6 | abstain on z-score | 0.046 | 0.607 | 0.987 | 0.0236 | 0.229 | 0.450 | 0.611 | 0.28 | reject: correct abstention 0.45 |
| 7 | overlap 0 | 0.072 | 0.611 | 0.991 | 0.0229 | 0.271 | 0.800 | 0.764 | 0.71 | reject: R@8 +0.004 is noise |
| 8 | overlap 50% (128 tokens) | 0.059 | 0.461 | 0.969 | 0.0207 | 0.271 | 0.800 | 0.764 | 0.69 | reject: P@8 fell more than 10% |
| R2-1 | chunks 128 / overlap 16 (re-test with re-scoring) | 0.024 | 0.361 | 0.949 | 0.0266 | 0.350 | 0.900 | 0.775 | 0.67 | reject: false abstention 0.350 |
| R2-2 | re-score top 2 contracts | 0.046 | 0.607 | 0.987 | 0.0237 | 0.264 | 0.800 | 0.768 | 0.71 | reject: R@8 unchanged |
| R2-3 | chunks 384 / overlap 48 | 0.093 | 0.691 | 0.993 | 0.0186 | 0.279 | 0.800 | 0.761 | 0.66 | reject: P@8 fell more than 10% |

R@64 values from iteration 2 onward are affected by whole-contract retrieval (Section 5.1).

**What the rejected experiments taught:**

- **Smaller chunks.**
  - **[Measured]** 64- and 128-token chunks raised P@8 but cut R@8 sharply at fixed k. 128 tokens was tested both without re-scoring (R@8 0.213) and with it (0.361).
  - **[Measured]** With re-scoring, the re-calibrated τ also produced more false abstentions: 0.414 for 64 tokens and 0.350 for 128 tokens. Without re-scoring, 128 tokens left false abstention at 0.257.
  - **[Hypothesis]** Recall@k at a fixed number of chunks structurally favours larger chunks; a fixed character budget would be a fairer comparison.
- **Larger chunks.** 384 tokens raised R@8 to 0.691 only by retrieving more text, so P@8 fell below the guardrail. This is the case the precision guardrail exists for.
- **Re-scoring two contracts.** It left R@8 and all abstention metrics unchanged (P@8 0.0236 → 0.0237).
  - **[Measured]** The first stage already ranks a gold-contract chunk first for 140/140 train queries.
  - **[Measured]** The top contract's chunks fill the top 8 for 133 of 140 train queries. The second contract only matters for the 7 queries whose contract has fewer than 8 chunks, which is consistent with the 0.0001 change in P@8.

**Selection.** The final configuration is the baseline plus document-local re-scoring plus query stop-words: the best train configuration under the declared rule. Round 2 found no better candidate. The configuration was frozen (`configs/final_frozen.yaml`, τ = 46.336014 from train-only run `20261029T191249464881Z_127bfd15`) before the single final test run.

## 7. Error analysis (final configuration)

Evidence comes from `experiments/final/20261029T191441576809Z_69310a62/per_query_*.jsonl`. Intermediate ranks for q078 come from the round-1 loop runs.

- **Query IDs** (`qNNN`) are positions in the prescribed 194-query mini selection.
- **"Gold rank"** is the rank of the first chunk overlapping a gold span.
- **"Query terms in gold"** checks the question part's non-stop-words against the gold text (exact word match).
- In the tables below, the Evidence column is measured. The Diagnosis column is a hypothesis unless a statement is marked **[Measured]**.

**[Measured] Train failure buckets at k = 8:**

| Bucket | Count (of 140) |
|---|---|
| Wrong document | 0 |
| Right document, wrong section | 46 |
| Partial span | 18 |
| Full coverage | 76 |

The top-1 chunk is still the contract's opening chunk for 110/140 train queries (127/140 at baseline).

### A. Vocabulary mismatch: the clause shares no words with the question

| ID | Evidence [Measured] | Diagnosis [Hypothesis unless stated] | Next step |
|------|----------------------|--------------|-----------|
| q060 (train) | "governing law" question for the Purolator/Parcelway agreement. None of {governing, law, contract} occurs in the gold "shall be governed … in accordance with the laws of the Province of Ontario". Gold rank 17 (baseline: not in top 64); R@8 = 0. | Without stemming, *governing* does not match *governed* and *law* does not match *laws*. | Light stemming or lemmatisation (a loop candidate). |
| q054 (train) | Same question for the Premier Biomedical IP agreement. Same 0/3 term overlap; gold rank 17; R@8 = 0. Abstention score 64.4, so the system answered with wrong passages. | Same cause. The high score (a confident wrong answer) likely comes from party-name matches. | Stemming; domain synonyms ("governing law" → "governed", "laws of"). |
| q003 (train) | "exceptions to competitive restrictions" (BicycleTX service agreement). Gold: "save that nothing in this paragraph 13.2 shall prevent you from holding … three percent (3%) of the issued equity share capital". 0/3 terms present; gold rank 11; also abstained (34.0 < 46.3). | Answering an exception requires recognising "save that … shall prevent", which lexical matching cannot do. | Dense or hybrid retrieval, or a clause-type classifier. |
| q013 (train) | "expiration date" (Theravance service agreement). Gold: employment "shall continue … until terminated by either party giving … notice". Only "date" matches; gold rank 9; R@8 = 0. | The concept (term / duration) is phrased differently from the question. | Query expansion for common CUAD question types (expiration → term, until, continue). |

### B. Clause-type questions whose gold annotation is indirect

| ID | Evidence [Measured] | Diagnosis [Hypothesis unless stated] | Next step |
|------|----------------------|--------------|-----------|
| q004 (train) | Right of first refusal / offer / negotiation question (BicycleTX). Gold: two spans (1,700 characters) about an option grant over shares. Only "include" matches, and "first refusal" does not occur in the gold; gold rank 17; abstained. | The annotator judged the option clause the closest match, although it contains no right-of-first-refusal wording. | Accept as a lexical-retrieval ceiling; candidate for semantic reranking. |
| q019 (train) | "covenant not to sue" (Theravance). Gold: a 581-character clause about termination on reconstruction or amalgamation, ending "there shall be no claim against the Company". 0/3 terms present; gold rank 8; R@8 = 0.49 (baseline 0). | The concept is implied ("there shall be no claim against the Company") rather than named. | Same as above. |

### C. Long contracts and multi-span answers

| ID | Evidence [Measured] | Diagnosis [Hypothesis unless stated] | Next step |
|------|----------------------|--------------|-----------|
| q087 (train) | "What licenses are granted" (Depomed/King, a 173,240-character contract). **7 gold spans** spread from character 27,994 to 144,889; 0/2 question terms present ("licenses", "granted" vs "grants"). Gold rank 12; R@8 = 0; abstained. | The seven spans are far apart, so each would have to rank in the top 8 separately; none does, partly because *granted* does not match *grants*. | Stemming; a per-query result depth that adapts to contract length. |
| q078 (train) | "expiration date" (same contract). Gold "The term of this Agreement shall … continue … until June 27, 2011" at character 112,116. Gold rank 15 at baseline, 16 after document-local re-scoring, **37 after stop-word removal**. Abstained. | **[Measured]** The regression comes from the stop-word step, not from re-scoring. Why removing function words pushed this clause down was not investigated. | Inspect per-term BM25 contributions for q078 before changing the stop-word list. |
| q039 (train) | IP ownership (ISA/Rubicon). 3 gold spans, 1,805 characters; gold rank 6; R@8 = 0.58 (unchanged from baseline). | **[Measured]** Partial span: the first two gold spans are fully covered by the rank-6 chunk; the third span (characters 15,667–16,433) first appears at rank 19. | Merge neighbouring chunks around a hit ("expand to clause"). |

### D. Structure: the opening or a nearby section outranks the clause

| ID | Evidence [Measured] | Diagnosis [Hypothesis unless stated] | Next step |
|------|----------------------|--------------|-----------|
| q009 (train) | "anti-assignment clause" (HPIL/GINARES). Top-1 is `[0, 4251)`: title, parties and **table of contents**. Gold "shall not be assignable" has 0/4 term overlap. Gold rank 11 (baseline 41). | The table of contents contains section headings, so it matches words from many questions. *assignment* does not match *assignable*. | Down-weight or skip table-of-contents and preamble chunks; stemming. |
| q096 (train) | "notice period required to terminate the renewal" (PennTex). Top-1 `[31206, 32836)` is a records/books clause; gold (the term clause, at character 49,279) is at rank 11; 2/5 terms present. | **[Measured]** The question word "renewal" does not occur in the gold. "notice" and "period" probably occur in many other clauses. | Clause-aware chunking on section boundaries so headings travel with clause text. |

### E. Correct retrieval, wrongly withheld (false abstention)

| ID | Evidence [Measured] | Diagnosis [Hypothesis unless stated] | Next step |
|------|----------------------|--------------|-----------|
| q080 (train) | Governing law (Depomed/King). R@8 = **1.0** (gold rank 8) but abstained: score 26.5 < 46.3. All 11/11 Depomed/King train queries abstained. | **[Measured]** False abstentions cluster by contract: all 37 train false abstentions come from 7 contracts (12/12 Papa John's, 11/11 King, 4/4 BicycleTX, 4/4 On2, …). **[Hypothesis]** The raw score depends on how distinctive a contract's party and title words are in the corpus. | Per-document normalisation of the abstention score, or calibrating on the document-local score; test on train with the same guardrails. |
| q116 (train) | Irrevocable/perpetual licences (ABG/Papa John's). Gold at **rank 1**, R@8 = 1.0, still abstained (40.3 < 46.3). | Same cause. | Same as above. |

### F. Near-miss unanswerable questions answered

| ID | Evidence [Measured] | Diagnosis [Hypothesis unless stated] | Next step |
|------|----------------------|--------------|-----------|
| unans_020 (train) | "stock price of Veoneer AB on the effective date of the JV amendment". Score 57.0 ≥ 46.3; top-1 is the Veoneer amendment preamble. | The contract exists and is named, so lexical scores are high; nothing measures whether the *requested fact* is present. | Answerability check on the top passages (e.g. NLI or an LLM verifier), evaluated against cost and privacy. |
| unans_041 (train) | "How many AB Wately subscribers accessed the PCQ software … on October 11, 1996". Score 55.7; top-1 is the PCQuote agreement header. | Same: specific party and product names match. | Same. |

## 8. Abstention findings and limitations

- **[Measured] Raw score tracks query length.** The raw top-1 score correlates with query length (Pearson r = 0.78 baseline, 0.71 final, on train answerable queries).
- **[Measured] The query sets differ in shape.** All 140 train answerable queries use the long "Consider the … ;" template; none of the 32 unanswerable queries do.
- **[Hypothesis]** Part of the measured abstention accuracy comes from detecting query shape rather than relevance.
- **[Measured] Length-invariant scores failed the guardrails.** Normalised top-1 removes the length dependence (r = −0.08) and cuts false abstention to 0.114. But it lets 9/20 train unanswerable queries through (correct abstention 0.55). The z-score is worse (0.45).
- **Short questions still abstain (unresolved).** Three short natural questions about contracts in the corpus all abstained (raw scores 18.0–22.7 vs τ = 46.3). This is an illustrative probe, not used for any decision.
  - In one of them ("What is the governing law of the Papa John's endorsement agreement?"), the #1 ranked passage contains the governing-law clause, but it is withheld.
  - The CLI (`--show-anyway`) and web app (checkbox) can show the ranked passages in a clearly labelled diagnostic mode; they are not returned as answers.
- **Small unanswerable set.** 20 train and 12 test queries; τ and the abstention metrics carry large uncertainty.

## 9. Limitations and next steps

**What doesn't work yet:**

- Precision is low by construction: retrieved chunks are about 1,600 characters, while the median answer is 316.
- The opening chunk wins rank 1 for most queries.
- Recall at k ≥ 16 is inflated by whole-contract retrieval in the final configuration (Section 5.1), so it says little about clause-level ranking.
- There is no stemming, so morphological variants are missed (governing / governed, grants / granted).
- The abstention rule is not usable for real users' short questions.
- Selection used train only (140 queries) with a greedy, order-dependent loop. There is no train sub-split, so some winner's-curse optimism on train is possible. The final test run checks this, but on only 54 answerable and 12 unanswerable queries.

**Next experiments, in order:**

1. Stemming.
2. Skip or down-weight preamble and table-of-contents chunks.
3. Combine corpus-level and document-local scores instead of replacing one with the other (not yet tested).
4. Clause-aware chunking on section headings.
5. A per-document-normalised abstention score, judged by the same guardrails.
6. A reranker or answerability verifier, measured for gain against cost, latency and data privacy.

**Before legal users relied on it:**

- A much larger and more realistic unanswerable set, including short natural questions.
- Human review of every returned clause.
- Coverage of cross-references and definitions, which this system ignores.
- Answer extraction beyond whole chunks.
