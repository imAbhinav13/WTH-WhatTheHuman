# Phase 1 Retrieval Evaluation

- Retrieval version: `phase1-concept-domain-retrieval-v1`
- Corpus version: `phase1_active_corpus_v1`
- Questions: 20
- Top-k per domain: 3
- Token budget per domain: 900
- Decision: **RETAIN CONCEPT-AWARE**

## Overall metrics

| Metric | Plain vector | Concept-aware | Delta |
|---|---:|---:|---:|
| Macro precision@k | 0.9417 | 1.0000 | +0.0583 |
| Macro recall@k | 0.0317 | 0.0364 | +0.0048 |
| Concept coverage | 0.9861 | 1.0000 | +0.0139 |
| MRR | 0.9556 | 1.0000 | +0.0444 |
| Source diversity | 0.8167 | 0.8556 | +0.0389 |

## Exit gate

- Precision not worse: `True`
- Concept coverage not worse: `True`
- Recall loss acceptable: `True`
- At least one quality metric improved: `True`
- Concept-aware retained: `True`

## Important limitation

This automatic retrieval evaluation uses the frozen question's expected concepts plus authoritative reviewed chunk labels as the relevance signal. It measures concept relevance and coverage; it is not a substitute for human judgment of question-specific passage usefulness.
