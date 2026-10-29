# Improvement loop 20261029T191235Z

Plan: `configs/loop.yaml` · objective: macro R@8 (min gain 0.01) · guardrails on P@8, false/correct abstention · train split only.

| # | candidate | change | R@8 | P@8 | false abst. | correct abst. | bal. acc. | len-corr | decision |
|---|---|---|---|---|---|---|---|---|---|
| 0 | baseline | configs/baseline.yaml | 0.3858 | 0.0161 | 0.257 | 0.850 | 0.796 | 0.78 | reference |
| 1 | chunk_128 | {"chunking": {"chunk_size": 128, "chunk_overlap": 16}} | 0.2128 | 0.0175 | 0.257 | 0.900 | 0.821 | 0.73 | reject: rejected: R@8 changed -0.1729 (worse than -0.01) |
| 2 | doc_local_rerank | {"retrieval": {"rerank": "doc_local_bm25", "rerank_docs": 1}} | 0.5740 | 0.0220 | 0.257 | 0.850 | 0.796 | 0.78 | **accept**: accepted: R@8 +0.1882 >= 0.01 and guardrails pass |
| 3 | chunk_64 | {"chunking": {"chunk_size": 64, "chunk_overlap": 8}} | 0.2225 | 0.0246 | 0.414 | 1.000 | 0.793 | 0.58 | reject: rejected: guardrail failed (false-abstention 0.414 > best 0.257 + 0.05; false-abstention 0.414 > baseline 0.257 + 0.05) |
| 4 | query_stopwords | {"retrieval": {"query_stopwords": true}} | 0.6073 | 0.0236 | 0.264 | 0.800 | 0.768 | 0.71 | **accept**: accepted: R@8 +0.0333 >= 0.01 and guardrails pass |
| 5 | abstain_normalized | {"abstention": {"score": "normalized_top1"}} | 0.6073 | 0.0236 | 0.114 | 0.550 | 0.718 | -0.08 | reject: rejected: guardrail failed (correct-abstention 0.550 < best 0.800 - 0.05; correct-abstention 0.550 < baseline 0.850 - 0.05) |
| 6 | abstain_zscore | {"abstention": {"score": "zscore_top1"}} | 0.6073 | 0.0236 | 0.229 | 0.450 | 0.611 | 0.28 | reject: rejected: guardrail failed (correct-abstention 0.450 < best 0.800 - 0.05; correct-abstention 0.450 < baseline 0.850 - 0.05) |
| 7 | overlap_0 | {"chunking": {"chunk_overlap": 0}} | 0.6108 | 0.0229 | 0.271 | 0.800 | 0.764 | 0.71 | reject: rejected: primary within noise (+0.0036) and abstention gain insufficient (balanced accuracy -0.004, length-corr reduction 0.01) |
| 8 | overlap_half | {"chunking": {"overlap_fraction": 0.5}} | 0.4613 | 0.0207 | 0.271 | 0.800 | 0.764 | 0.69 | reject: rejected: guardrail failed (P@8 0.0207 < 0.90 x best 0.0236) |

**Final configuration:** baseline + ['doc_local_rerank', 'query_stopwords'] (run `20261029T191249464881Z_127bfd15`).
Train R@8 0.3858 → 0.6073; P@8 0.0161 → 0.0236.
