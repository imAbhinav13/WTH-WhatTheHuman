-- WTH Stage 0 — Phase 1 production schema hardening
-- Purpose:
--   Extend the existing Supabase schema so it can losslessly represent
--   the frozen Phase 1 runtime corpus before any corpus rows are loaded.
--
-- Preconditions verified by Stage 0B audit:
--   public.sources        = 0 rows
--   public.chunks         = 0 rows
--   public.chunk_concepts = 0 rows
--   public.concepts       = 8 seeded rows
--   pgvector              = installed
--   chunks.embedding      = vector(768)
--
-- This migration intentionally:
--   * does NOT load Phase 1 corpus data
--   * does NOT regenerate embeddings
--   * does NOT create public RLS policies
--   * does NOT duplicate corpus_version_id onto chunks
--   * preserves the current 8 canonical concept seeds

begin;

-- ---------------------------------------------------------------------------
-- 1. SOURCE PROVENANCE
-- ---------------------------------------------------------------------------

alter table public.sources
    add column if not exists editor text,
    add column if not exists download_url text,
    add column if not exists license_name text,
    add column if not exists license_url text,
    add column if not exists rights_status text,
    add column if not exists rights_statement text,
    add column if not exists rights_jurisdiction text,
    add column if not exists accessed_at date,
    add column if not exists source_checksum text;

alter table public.sources
    alter column rights_status set not null,
    alter column source_checksum set not null;

alter table public.sources
    drop constraint if exists sources_rights_status_not_blank,
    add constraint sources_rights_status_not_blank
        check (length(btrim(rights_status)) > 0);

alter table public.sources
    drop constraint if exists sources_source_checksum_sha256,
    add constraint sources_source_checksum_sha256
        check (source_checksum ~ '^[0-9a-f]{64}$');

create index if not exists sources_source_checksum_idx
    on public.sources (source_checksum);

comment on column public.sources.source_url is
    'Canonical source URL for citation/provenance.';

comment on column public.sources.download_url is
    'Acquisition URL used to obtain the reviewed source artifact, when available.';

comment on column public.sources.rights_status is
    'Frozen Phase 1 activation rights-status value. Must be populated before source activation.';

comment on column public.sources.source_checksum is
    'SHA-256 checksum of the acquired source artifact used to build the reviewed corpus.';

-- ---------------------------------------------------------------------------
-- 2. CHUNK REVIEW + EMBEDDING PROVENANCE
-- ---------------------------------------------------------------------------

alter table public.chunks
    add column if not exists review_decision text,
    add column if not exists reviewer text,
    add column if not exists reviewed_at timestamptz,
    add column if not exists review_notes text,
    add column if not exists embedding_provider text,
    add column if not exists embedding_model_revision text,
    add column if not exists embedding_normalization text,
    add column if not exists embedding_task_type text,
    add column if not exists embedding_checksum text,
    add column if not exists embedding_created_at timestamptz;

alter table public.chunks
    alter column review_decision set not null,
    alter column embedding_provider set not null,
    alter column embedding_model_revision set not null,
    alter column embedding_normalization set not null,
    alter column embedding_task_type set not null,
    alter column embedding_checksum set not null;

alter table public.chunks
    drop constraint if exists chunks_review_decision_not_blank,
    add constraint chunks_review_decision_not_blank
        check (length(btrim(review_decision)) > 0);

alter table public.chunks
    drop constraint if exists chunks_embedding_provider_not_blank,
    add constraint chunks_embedding_provider_not_blank
        check (length(btrim(embedding_provider)) > 0);

alter table public.chunks
    drop constraint if exists chunks_embedding_model_revision_not_blank,
    add constraint chunks_embedding_model_revision_not_blank
        check (length(btrim(embedding_model_revision)) > 0);

alter table public.chunks
    drop constraint if exists chunks_embedding_normalization_not_blank,
    add constraint chunks_embedding_normalization_not_blank
        check (length(btrim(embedding_normalization)) > 0);

alter table public.chunks
    drop constraint if exists chunks_embedding_task_type_not_blank,
    add constraint chunks_embedding_task_type_not_blank
        check (length(btrim(embedding_task_type)) > 0);

alter table public.chunks
    drop constraint if exists chunks_embedding_checksum_sha256,
    add constraint chunks_embedding_checksum_sha256
        check (embedding_checksum ~ '^[0-9a-f]{64}$');

comment on column public.chunks.content_hash is
    'Frozen reviewed-text checksum. Stage 1 maps Phase 1 text_checksum to this field.';

comment on column public.chunks.embedding_provider is
    'Embedding provider identity; frozen Phase 1 value is Google Gemini API.';

comment on column public.chunks.embedding_model_revision is
    'Exact embedding model revision; frozen Phase 1 value is 2.';

comment on column public.chunks.embedding_normalization is
    'Embedding normalization policy; frozen Phase 1 value is provider_auto_l2.';

comment on column public.chunks.embedding_task_type is
    'Provider task type used for the stored document embedding.';

comment on column public.chunks.embedding_checksum is
    'SHA-256 checksum of the serialized selected embedding record/vector provenance.';

-- ---------------------------------------------------------------------------
-- 3. REVIEWED CONCEPT RELATION PROVENANCE
-- ---------------------------------------------------------------------------
-- Existing public.chunk_concepts.weight is retained intentionally.
-- For the production Phase 1 corpus it is the frozen calibrated_weight
-- used for ranking. Runtime repositories should expose it as
-- calibrated_weight in domain models if desired.

alter table public.chunk_concepts
    add column if not exists raw_similarity double precision,
    add column if not exists raw_mapping_score double precision,
    add column if not exists human_label text,
    add column if not exists human_override boolean default false,
    add column if not exists production_active boolean,
    add column if not exists phase1_role text,
    add column if not exists review_status text,
    add column if not exists mapping_method text,
    add column if not exists prototype_version text,
    add column if not exists model_version text,
    add column if not exists updated_at timestamptz
        default timezone('utc'::text, now());

alter table public.chunk_concepts
    alter column raw_similarity set not null,
    alter column raw_mapping_score set not null,
    alter column human_label set not null,
    alter column human_override set not null,
    alter column production_active set not null,
    alter column phase1_role set not null,
    alter column review_status set not null,
    alter column mapping_method set not null,
    alter column prototype_version set not null,
    alter column model_version set not null,
    alter column updated_at set not null;

-- Phase 1 calibrated weights are explicitly 0..1.
alter table public.chunk_concepts
    drop constraint if exists chunk_concepts_weight_valid,
    add constraint chunk_concepts_weight_valid
        check (weight >= 0.0 and weight <= 1.0);

alter table public.chunk_concepts
    drop constraint if exists chunk_concepts_human_label_valid,
    add constraint chunk_concepts_human_label_valid
        check (human_label in ('positive', 'partial', 'negative'));

alter table public.chunk_concepts
    drop constraint if exists chunk_concepts_phase1_role_not_blank,
    add constraint chunk_concepts_phase1_role_not_blank
        check (length(btrim(phase1_role)) > 0);

alter table public.chunk_concepts
    drop constraint if exists chunk_concepts_review_status_not_blank,
    add constraint chunk_concepts_review_status_not_blank
        check (length(btrim(review_status)) > 0);

alter table public.chunk_concepts
    drop constraint if exists chunk_concepts_mapping_method_not_blank,
    add constraint chunk_concepts_mapping_method_not_blank
        check (length(btrim(mapping_method)) > 0);

alter table public.chunk_concepts
    drop constraint if exists chunk_concepts_prototype_version_not_blank,
    add constraint chunk_concepts_prototype_version_not_blank
        check (length(btrim(prototype_version)) > 0);

alter table public.chunk_concepts
    drop constraint if exists chunk_concepts_model_version_not_blank,
    add constraint chunk_concepts_model_version_not_blank
        check (length(btrim(model_version)) > 0);

create index if not exists chunk_concepts_active_concept_weight_idx
    on public.chunk_concepts (concept_id, weight desc)
    where production_active = true;

create index if not exists chunk_concepts_chunk_active_idx
    on public.chunk_concepts (chunk_id, production_active);

comment on column public.chunk_concepts.weight is
    'Frozen calibrated concept weight in [0,1]; equivalent to Phase 12 calibrated_weight.';

comment on column public.chunk_concepts.human_label is
    'Authoritative reviewed Phase 1 concept label: positive, partial, or negative.';

comment on column public.chunk_concepts.production_active is
    'Authoritative concept eligibility flag derived from human-reviewed labels.';

comment on column public.chunk_concepts.human_override is
    'True when frozen automated activation disagreed with the authoritative human-reviewed label.';

-- Keep updated_at behavior consistent with sources/chunks/concepts.
drop trigger if exists chunk_concepts_set_updated_at
    on public.chunk_concepts;

create trigger chunk_concepts_set_updated_at
before update on public.chunk_concepts
for each row
execute function public.set_updated_at();

-- ---------------------------------------------------------------------------
-- 4. CORPUS-VERSION RESOLUTION
-- ---------------------------------------------------------------------------
-- No corpus_version_id column is added to chunks.
--
-- A chunk resolves corpus version through:
--   chunks.source_id -> sources.id
--   sources.corpus_version_id -> corpus_versions.id
--
-- This prevents duplicated version fields from drifting out of sync.
-- Existing indexes already support:
--   sources(corpus_version_id)
--   chunks(source_id)
--
-- Stage 1 will create/activate corpus_versions.version =
--   'phase1_active_corpus_v1'
-- and load only its approved sources.

-- ---------------------------------------------------------------------------
-- 5. RLS SECURITY POSTURE
-- ---------------------------------------------------------------------------
-- RLS remains enabled with no anon/authenticated read policies.
-- The public frontend must NOT query these tables directly.
-- FastAPI will use backend-held service credentials.
-- No RLS policy changes are made in this migration.

commit;
