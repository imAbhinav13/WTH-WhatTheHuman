# WTH Phase 1 — Supabase Production Schema Mapping

## Purpose

This document records the Stage 0 mapping from the frozen Phase 1 artifacts to the live Supabase production schema.

The objective is to preserve the frozen Phase 1 runtime semantics while changing only the storage/I/O layer.

## Frozen baseline

- Corpus version: `phase1_active_corpus_v1`
- Active chunks: `318`
- Approved sources: `10`
- Reviewed chunk-concept relations: `954`
- Phase 1 concepts:
  - `consciousness`
  - `self_identity`
  - `reality_appearance`
- Embedding provider: `Google Gemini API`
- Embedding model: `gemini-embedding-2`
- Model revision: `2`
- Dimensions: `768`
- Normalization: `provider_auto_l2`
- Prototype version: `phase1-prototype-v2`
- Mapping method: `hybrid:question:centroid`

## Design decisions

### Embeddings remain on `chunks`

Phase 1 has one selected production document embedding per active chunk, so the live schema keeps:

```text
chunks.embedding vector(768)
```

A separate embedding table is not required for the Phase 1 production runtime.

### Corpus version remains normalized

The live relationship is:

```text
chunks.source_id
    -> sources.id

sources.corpus_version_id
    -> corpus_versions.id
```

`chunks` does not duplicate `corpus_version_id`.

Every runtime retrieval query that needs corpus-version enforcement must join through `sources` and `corpus_versions`.

### `chunk_concepts.weight` means calibrated production weight

The existing `weight` column is retained for compatibility.

For the frozen Phase 1 production corpus:

```text
chunk_concepts.weight
    == Phase 12 calibrated_weight
```

Runtime Pydantic/service models may expose the database field as `calibrated_weight`.

Human-reviewed labels remain authoritative. `weight` is a ranking signal, not the source of inclusion truth.

## Artifact-to-database mapping

### Corpus version

| Frozen artifact value | Supabase destination |
|---|---|
| `phase1_active_corpus_v1` | `corpus_versions.version` |
| lifecycle active | `corpus_versions.is_active = true` |

Stage 1 creates the Phase 1 version and deactivates `phase0-v1` in the same transaction because the schema allows only one active corpus version.

### Sources

| Frozen/source-catalogue field | Supabase destination |
|---|---|
| `source_id` | `sources.id` |
| corpus version FK | `sources.corpus_version_id` |
| `title` | `sources.title` |
| `author` | `sources.author` |
| `translator` | `sources.translator` |
| `editor` | `sources.editor` |
| `edition` | `sources.edition` |
| `publication_year` | `sources.publication_year` |
| `source_type` | `sources.source_type` |
| `canonical_url` | `sources.source_url` |
| `download_url` | `sources.download_url` |
| `license_name` | `sources.license_name` |
| `license_url` | `sources.license_url` |
| frozen attribution text | `sources.license_attribution` |
| license/rights verified | `sources.license_verified` |
| frozen source rights status | `sources.rights_status` |
| `rights_statement` | `sources.rights_statement` |
| `rights_jurisdiction` | `sources.rights_jurisdiction` |
| `accessed_at` | `sources.accessed_at` |
| `checksum` / source checksum | `sources.source_checksum` |

Only the `10` approved sources represented by the frozen active corpus are loaded in Stage 1. The broader source catalogue contains additional candidates that are not automatically activated.

### Chunks

| Frozen active-chunk field | Supabase destination |
|---|---|
| `chunk_id` | `chunks.id` |
| `source_id` | `chunks.source_id` |
| `domain` | `chunks.domain` |
| `citation` | `chunks.citation` |
| final reviewed text | `chunks.full_text` |
| frozen claim type | `chunks.claim_type` |
| previous neighbor ID | `chunks.neighbor_prev_id` |
| next neighbor ID | `chunks.neighbor_next_id` |
| lifecycle active | `chunks.review_status = 'active'` |
| selected embedding model | `chunks.embedding_model` |
| dimensions | `chunks.embedding_dimension = 768` |
| selected vector | `chunks.embedding` |
| reviewed `text_checksum` | `chunks.content_hash` |
| review decision | `chunks.review_decision` |
| reviewer | `chunks.reviewer` |
| reviewed timestamp | `chunks.reviewed_at` |
| review notes | `chunks.review_notes` |
| provider | `chunks.embedding_provider` |
| model revision | `chunks.embedding_model_revision` |
| normalization | `chunks.embedding_normalization` |
| document task type | `chunks.embedding_task_type` |
| embedding checksum | `chunks.embedding_checksum` |
| embedding creation timestamp | `chunks.embedding_created_at` |

The database must preserve exactly the frozen stable `chunk_id`; Stage 1 must never generate new chunk IDs.

### Concepts

The database already contains all eight canonical concept seeds.

Stage 1 uses the existing UUIDs for:

| Concept slug | Existing UUID |
|---|---|
| `self_identity` | `10000000-0000-4000-8000-000000000001` |
| `consciousness` | `10000000-0000-4000-8000-000000000002` |
| `reality_appearance` | `10000000-0000-4000-8000-000000000003` |

The other five canonical concepts remain seeded but are not assigned Phase 1 chunk relations.

No concept rows are duplicated during Stage 1.

### Reviewed chunk-concept relations

| Frozen Phase 12/13 field | Supabase destination |
|---|---|
| `chunk_id` | `chunk_concepts.chunk_id` |
| concept slug | resolve to `concepts.id`, then `chunk_concepts.concept_id` |
| `calibrated_weight` | `chunk_concepts.weight` |
| `raw_similarity` | `chunk_concepts.raw_similarity` |
| `raw_mapping_score` | `chunk_concepts.raw_mapping_score` |
| `human_label` | `chunk_concepts.human_label` |
| `human_override` | `chunk_concepts.human_override` |
| `production_active` | `chunk_concepts.production_active` |
| `phase1_role` | `chunk_concepts.phase1_role` |
| `review_status` | `chunk_concepts.review_status` |
| `mapping_method` | `chunk_concepts.mapping_method` |
| `prototype_version` | `chunk_concepts.prototype_version` |
| `model_version` | `chunk_concepts.model_version` |

Every active Phase 1 chunk must have exactly three rows: one for each Phase 1 concept.

Expected total:

```text
318 chunks × 3 concepts = 954 chunk_concepts rows
```

`production_active`, not the numeric weight alone, controls concept eligibility.

## Runtime retrieval contract

A production Phase 14 database query must enforce:

```text
chunks.review_status = 'active'
corpus_versions.version = 'phase1_active_corpus_v1'
corpus_versions.is_active = true
chunk_concepts.production_active = true
```

Vector search uses:

```text
chunks.embedding vector(768)
chunks_embedding_hnsw_idx
vector_cosine_ops
```

Domain separation uses:

```text
chunks.domain IN ('science', 'advaita', 'samkhya')
```

## RLS/security decision

RLS is enabled on the corpus tables and currently has no anon/authenticated policies.

That is intentional for the public architecture:

```text
Browser
  -> FastAPI
      -> Supabase using backend-only service credentials
```

The frontend must not receive the Supabase service-role/secret key and must not query corpus tables directly.

Public citation access will later occur through:

```http
GET /api/chunk/{id}
```

## Stage 1 idempotency keys

- corpus version: unique `corpus_versions.version`
- source: stable `sources.id`
- chunk: stable `chunks.id`
- concept: existing unique `concepts.slug`
- relation: primary key `(chunk_id, concept_id)`

The Stage 1 loader must use upserts and must be safe to rerun.

## Stage 0 exit-gate mapping

After the Stage 0 migration is applied and the schema audit is rerun:

| Exit gate | Expected |
|---|---|
| Schema supports 318 active chunks | PASS |
| Schema supports 954 chunk-concept relations | PASS |
| Embedding dimension = 768 | PASS |
| All required provenance fields mapped | PASS |
| All required citations/source metadata mapped | PASS |
| Required indexes defined | PASS |
| No destructive schema gap | PASS |
| Migration/load mapping documented | PASS |

Stage 0 does not load the 318 chunks. Row-count reconciliation belongs to Stage 1.
