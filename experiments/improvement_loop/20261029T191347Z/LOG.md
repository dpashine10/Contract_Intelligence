# Improvement loop 20261029T191347Z

Plan: `configs/loop_round2.yaml` · objective: macro R@8 (min gain 0.01) · guardrails on P@8, false/correct abstention · train split only.

| # | candidate | change | R@8 | P@8 | false abst. | correct abst. | bal. acc. | len-corr | decision |
|---|---|---|---|---|---|---|---|---|---|
| 0 | baseline | configs/baseline.yaml | 0.6073 | 0.0236 | 0.264 | 0.800 | 0.768 | 0.71 | reference |
| 1 | chunk_128_retest | {"chunking": {"chunk_size": 128, "chunk_overlap": 16}} | 0.3611 | 0.0266 | 0.350 | 0.900 | 0.775 | 0.67 | reject: rejected: guardrail failed (false-abstention 0.350 > best 0.264 + 0.05; false-abstention 0.350 > baseline 0.264 + 0.05) |
| 2 | rerank_two_docs | {"retrieval": {"rerank_docs": 2}} | 0.6073 | 0.0237 | 0.264 | 0.800 | 0.768 | 0.71 | reject: rejected: primary within noise (+0.0000) and abstention gain insufficient (balanced accuracy +0.000, length-corr reduction 0.0) |
| 3 | chunk_384 | {"chunking": {"chunk_size": 384, "chunk_overlap": 48}} | 0.6906 | 0.0186 | 0.279 | 0.800 | 0.761 | 0.66 | reject: rejected: guardrail failed (P@8 0.0186 < 0.90 x best 0.0236; P@8 0.0186 < 0.90 x baseline 0.0236) |

**Final configuration:** baseline + no changes (run `20261029T191347475686Z_127bfd15`).
Train R@8 0.6073 → 0.6073; P@8 0.0236 → 0.0236.
