# Phase 1 Embedding Architecture Decision

**Project:** WTH — What The Human  
**Status:** Accepted / Frozen for Phase 1  
**Decision date:** 2026-08-08  
**Scope:** Consciousness, Self/Identity, Reality/Appearance

## 1. Purpose

This record freezes the Phase 1 embedding architecture after empirical benchmarking. Model selection used only the frozen Build and Development sets. The Held-out set was checksum-verified and kept read-only; its content was not parsed or used for model selection.

## 2. Evaluation context

| Dataset | Records | Role |
|---|---:|---|
| Build | 159 | Prototype construction, debugging, qualitative inspection |
| Development | 80 | Threshold, weighting, top-k, reranking and ambiguity calibration |
| Held-out | 79 | Final evaluation only |
| **Total** | **318** | |

The benchmark used 239 Build + Development documents and 22 benchmark queries. Similarity was cosine on L2-normalized vectors. Relevance grades were positive=2, partial=1, negative/uncertain=0.

## 3. Candidates

### Gemini
- Provider: Google Gemini API
- Model: `gemini-embedding-2`
- Dimensions: 768
- Document representation: `title: {title} | text: {content}`
- Query representation: `task: search result | query: {content}`
- Normalization: L2 before cosine evaluation

### Local
- Provider: Local SentenceTransformers
- Model: `intfloat/multilingual-e5-base`
- Dimensions: 768
- Document representation: `passage: title: {title} | text: {content}`
- Query representation: `query: {content}`
- Batch size: 32
- Normalization: L2 before cosine evaluation

### Optional hosted alternative
The benchmark harness defined Cohere `embed-v4.0`, but no Cohere result is present in the completed benchmark. This decision therefore does not claim empirical superiority over Cohere.

## 4. Aggregate benchmark results

| Metric | Gemini Embedding 2 | multilingual-e5-base | Winner |
|---|---:|---:|---|
| MRR | **0.9318** | 0.7565 | Gemini |
| nDCG@5 | **0.8088** | 0.6122 | Gemini |
| Precision@5 | **0.8636** | 0.6818 | Gemini |
| Recall@10 | **0.0891** | 0.0719 | Gemini |
| Hard-negative false-positive rate @10 | **0.0591** | 0.1227 | Gemini |
| Score margin | **0.03465** | 0.00356 | Gemini |

Gemini outperformed the local candidate on every aggregate retrieval-quality metric measured. The hard-negative result is particularly important because Phase 1 intentionally tests discrimination among adjacent concepts.

## 5. Concept-level results

| Concept | Gemini MRR | Gemini nDCG@5 | Local MRR | Local nDCG@5 |
|---|---:|---:|---:|---:|
| Consciousness | **1.0000** | **0.8705** | 0.7347 | 0.4875 |
| Reality/Appearance | **0.9286** | **0.7789** | 0.8333 | 0.7037 |
| Self/Identity | **0.8750** | **0.7810** | 0.7083 | 0.6413 |

Gemini also showed materially better hard-negative behavior for Consciousness and Self/Identity.

## 6. Domain robustness

### Gemini

| Domain | MRR | nDCG@5 | Precision@5 | Recall@10 | HN FP@10 |
|---|---:|---:|---:|---:|---:|
| Science | **1.0000** | **1.0000** | **1.0000** | 0.1273 | **0.0000** |
| Advaita | **1.0000** | 0.7506 | 0.7333 | 0.1491 | 0.0333 |
| Samkhya | 0.6667 | 0.4693 | 0.6667 | 0.1511 | 0.1667 |

**Known weakness:** Samkhya retrieval is materially weaker than Science and Advaita. This must remain an explicit Phase 1 risk. It should be addressed through prototypes, weighting, reranking, source-aware retrieval and hard-negative-aware calibration using Build + Development only.

## 7. Operational results

| Metric | Gemini | Local E5 |
|---|---:|---:|
| Generated vectors | 261 | 261 |
| API requests | 261 | 0 |
| Document embedding time | 384.91 s | 404.64 s |
| Query embedding time | 32.05 s | 8.30 s |
| Average query latency | 0.694 s | **0.377 s** |
| p95 query latency | 0.895 s | **0.520 s** |
| Rate-limit retries | **0** | N/A |
| Other retries | **0** | N/A |
| Reported input tokens | 71,020 | N/A |
| Additional local memory | N/A | ~601 MB |

Local E5 has lower query latency and no external API dependency. However, corpus embedding was not faster on the tested machine and retrieval quality was substantially weaker.

The correctly scoped Gemini benchmark completed all 261 API requests with zero rate-limit retries. This is evidence that the earlier quota failure was strongly associated with the accidentally oversized 7,469-chunk workflow rather than demonstrating that Gemini is unsuitable for the corrected Phase 1 architecture.

## 8. Decision

**Selected primary embedding architecture: Google Gemini API — `gemini-embedding-2`, 768 dimensions.**

Frozen configuration:

```text
Provider: Google Gemini API
Model: gemini-embedding-2
Dimensions: 768
Similarity: cosine similarity
Normalization: L2-normalized vectors

Document representation:
title: {title} | text: {content}

Query representation:
task: search result | query: {content}
```

The Phase 1 database/vector architecture remains 768-dimensional. Current evidence does not justify migration to 1536 or 3072 dimensions.

## 9. Quota and batching policy

The production embedding path must not recreate the original unbounded bulk-embedding behavior. It must use:

1. deterministic checksum-based embedding caching;
2. skip-if-already-embedded behavior;
3. bounded request scheduling;
4. resumable checkpoints;
5. request/token accounting;
6. exponential backoff with jitter for retryable 429/5xx responses;
7. maximum retry limits;
8. circuit breaking for sustained provider failures;
9. backpressure rather than uncontrolled concurrency;
10. persisted progress so completed embeddings are not regenerated after failure.

For the Phase 1 corpus size, a rate-aware bounded asynchronous worker or paced synchronous worker is sufficient. Distributed embedding infrastructure is not justified at this stage.

## 10. Cache identity

Persisted embedding identity must include at least:

```text
provider
model
dimensions
normalization/configuration version
document/query representation version
text checksum
```

A change in text, model, dimensions or representation invalidates the corresponding cached vector.

## 11. Local fallback

`intfloat/multilingual-e5-base` is retained as an offline/fallback model for local development, diagnostics, cost-free experimentation or provider outages.

It must not silently replace Gemini in evaluation or production. Vectors from different models are not interchangeable, so provider/model identity must accompany persisted embeddings.

## 12. Why local E5 was not selected

It showed:
- lower MRR;
- lower nDCG@5;
- lower Precision@5;
- lower Recall@10;
- approximately twice Gemini's aggregate hard-negative false-positive rate;
- weaker consciousness discrimination;
- weaker Advaita performance;
- a much smaller relevant-vs-irrelevant score margin.

Its lower query latency and API independence do not outweigh the retrieval-quality deficit for the Phase 1 research objective.

## 13. Decision constraints and limitations

This decision applies to the Phase 1 three-concept vertical slice and should be reconsidered if corpus scale, language coverage, provider economics, model availability, or the concept-family scope changes materially.

Limitations:
1. Only Gemini and local E5 produced completed benchmark results.
2. Cohere was configured but not empirically benchmarked.
3. The benchmark query suite contains 22 queries.
4. Recall@10 should be interpreted with the ranking metrics because of the benchmark relevance-set formulation.
5. Samkhya remains the weakest Gemini domain.
6. Benchmark cost was not calculated because no per-million-token price was configured.
7. Final untouched Held-out evaluation has not yet occurred.

The Held-out set may reveal a failure of the selected architecture, but it must not be used iteratively for tuning and then represented as untouched final evaluation.

## 14. Next stage

```text
Frozen Gemini Embedding 2 / 768d architecture
        ↓
Generate corpus embeddings
        ↓
Generate concept prototypes / anchors
        ↓
Calculate weighted concept tags
        ↓
Tune using Build + Development only
        ↓
Evaluate adjacent-concept discrimination
        ↓
Final locked Held-out evaluation
        ↓
Activate approved Phase 1 corpus
        ↓
Concept/domain retrieval
        ↓
Science / Advaita / Samkhya generation
        ↓
Synthesis and tension detection
        ↓
Coverage classification
```

## 15. Final decision statement

For the WTH Phase 1 Consciousness / Self / Reality-appearance vertical slice, **Google `gemini-embedding-2` at 768 dimensions is selected and frozen as the primary embedding model**.

The decision is supported by materially stronger ranking quality, adjacent-concept discrimination, hard-negative rejection and score separation than `intfloat/multilingual-e5-base`.

Local E5 remains the offline/fallback implementation.

The primary known retrieval risk is Samkhya. That risk will be addressed in subsequent prototype, weighting, reranking and calibration work using only the Build and Development sets before the untouched Held-out evaluation.

## 16. Benchmark provenance

Inputs and evidence:
- `artifacts/phase1/evaluation/embedding_benchmark_results.json`
- `artifacts/phase1/evaluation/embedding_benchmark_query_results.csv`
- `data/evaluation/phase1_build.jsonl`
- `data/evaluation/phase1_development.jsonl`
- `data/evaluation/phase1_heldout.jsonl`
- `data/evaluation/phase1_split_manifest.json`

Benchmark status: `benchmark_complete`  
Pre-decision status: `pending_human_review`

This document records the architecture decision following human review of the Phase 7 benchmark.
