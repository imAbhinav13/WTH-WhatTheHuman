begin;

-- ---------------------------------------------------------------------------
-- Extensions
-- ---------------------------------------------------------------------------

create extension if not exists vector with schema extensions;
create extension if not exists pgcrypto with schema extensions;


-- ---------------------------------------------------------------------------
-- Shared trigger function
-- ---------------------------------------------------------------------------

create or replace function public.set_updated_at()
returns trigger
language plpgsql
security invoker
set search_path = public
as $$
begin
    new.updated_at = timezone('utc', now());
    return new;
end;
$$;


-- ---------------------------------------------------------------------------
-- Corpus versions
-- ---------------------------------------------------------------------------

create table public.corpus_versions (
    id uuid primary key default extensions.gen_random_uuid(),
    version text not null unique,
    description text,
    is_active boolean not null default false,
    created_at timestamptz not null default timezone('utc', now()),

    constraint corpus_versions_version_not_blank
        check (length(btrim(version)) > 0)
);

create unique index corpus_versions_one_active_idx
    on public.corpus_versions (is_active)
    where is_active = true;


-- ---------------------------------------------------------------------------
-- Sources
-- ---------------------------------------------------------------------------

create table public.sources (
    id text primary key,
    corpus_version_id uuid not null
        references public.corpus_versions(id)
        on update cascade
        on delete restrict,

    title text not null,
    author text,
    translator text,
    edition text,
    publication_year integer,
    source_type text not null,
    source_url text,
    license_attribution text not null,
    license_verified boolean not null default false,

    created_at timestamptz not null default timezone('utc', now()),
    updated_at timestamptz not null default timezone('utc', now()),

    constraint sources_id_not_blank
        check (length(btrim(id)) > 0),

    constraint sources_title_not_blank
        check (length(btrim(title)) > 0),

    constraint sources_source_type_valid
        check (
            source_type in (
                'paper',
                'primary_text',
                'commentary'
            )
        ),

    constraint sources_publication_year_valid
        check (
            publication_year is null
            or publication_year between 0 and 3000
        ),

    constraint sources_license_attribution_not_blank
        check (length(btrim(license_attribution)) > 0)
);

create index sources_corpus_version_idx
    on public.sources (corpus_version_id);

create index sources_source_type_idx
    on public.sources (source_type);

create trigger sources_set_updated_at
before update on public.sources
for each row
execute function public.set_updated_at();


-- ---------------------------------------------------------------------------
-- Concepts
-- ---------------------------------------------------------------------------

create table public.concepts (
    id uuid primary key default extensions.gen_random_uuid(),

    slug text not null unique,
    display_name text not null,
    description text not null,
    anchor_text text not null,
    anchor_embedding extensions.vector(768),

    is_active boolean not null default true,

    created_at timestamptz not null default timezone('utc', now()),
    updated_at timestamptz not null default timezone('utc', now()),

    constraint concepts_slug_not_blank
        check (length(btrim(slug)) > 0),

    constraint concepts_slug_format
        check (slug ~ '^[a-z][a-z0-9_]*$'),

    constraint concepts_display_name_not_blank
        check (length(btrim(display_name)) > 0),

    constraint concepts_description_not_blank
        check (length(btrim(description)) > 0),

    constraint concepts_anchor_text_not_blank
        check (length(btrim(anchor_text)) > 0)
);

create index concepts_active_idx
    on public.concepts (is_active);

create index concepts_anchor_embedding_hnsw_idx
    on public.concepts
    using hnsw (anchor_embedding extensions.vector_cosine_ops)
    where anchor_embedding is not null;

create trigger concepts_set_updated_at
before update on public.concepts
for each row
execute function public.set_updated_at();


-- ---------------------------------------------------------------------------
-- Chunks
-- ---------------------------------------------------------------------------

create table public.chunks (
    id text primary key,

    source_id text not null
        references public.sources(id)
        on update cascade
        on delete restrict,

    domain text not null,
    citation text not null,
    full_text text not null,
    claim_type text not null,

    neighbor_prev_id text,
    neighbor_next_id text,

    review_status text not null default 'draft',

    embedding_model text not null,
    embedding_dimension integer not null default 768,
    embedding extensions.vector(768) not null,

    content_hash text not null,

    created_at timestamptz not null default timezone('utc', now()),
    updated_at timestamptz not null default timezone('utc', now()),

    constraint chunks_neighbor_prev_fk
        foreign key (neighbor_prev_id)
        references public.chunks(id)
        on update cascade
        on delete set null
        deferrable initially deferred,

    constraint chunks_neighbor_next_fk
        foreign key (neighbor_next_id)
        references public.chunks(id)
        on update cascade
        on delete set null
        deferrable initially deferred,

    constraint chunks_id_not_blank
        check (length(btrim(id)) > 0),

    constraint chunks_domain_valid
        check (
            domain in (
                'science',
                'advaita',
                'samkhya'
            )
        ),

    constraint chunks_citation_not_blank
        check (length(btrim(citation)) > 0),

    constraint chunks_full_text_not_blank
        check (length(btrim(full_text)) > 0),

    constraint chunks_claim_type_valid
        check (
            claim_type in (
                'empirical',
                'metaphysical',
                'normative'
            )
        ),

    constraint chunks_review_status_valid
        check (
            review_status in (
                'draft',
                'reviewed',
                'active',
                'archived'
            )
        ),

    constraint chunks_embedding_model_not_blank
        check (length(btrim(embedding_model)) > 0),

    constraint chunks_embedding_dimension_locked
        check (embedding_dimension = 768),

    constraint chunks_content_hash_not_blank
        check (length(btrim(content_hash)) > 0),

    constraint chunks_neighbor_prev_not_self
        check (
            neighbor_prev_id is null
            or neighbor_prev_id <> id
        ),

    constraint chunks_neighbor_next_not_self
        check (
            neighbor_next_id is null
            or neighbor_next_id <> id
        )
);

create unique index chunks_source_content_hash_idx
    on public.chunks (source_id, content_hash);

create index chunks_source_id_idx
    on public.chunks (source_id);

create index chunks_domain_idx
    on public.chunks (domain);

create index chunks_review_status_idx
    on public.chunks (review_status);

create index chunks_claim_type_idx
    on public.chunks (claim_type);

create index chunks_domain_active_idx
    on public.chunks (domain)
    where review_status = 'active';

create index chunks_embedding_hnsw_idx
    on public.chunks
    using hnsw (embedding extensions.vector_cosine_ops)
    where review_status = 'active';

create trigger chunks_set_updated_at
before update on public.chunks
for each row
execute function public.set_updated_at();


-- ---------------------------------------------------------------------------
-- Weighted chunk-to-concept relationships
-- ---------------------------------------------------------------------------

create table public.chunk_concepts (
    chunk_id text not null
        references public.chunks(id)
        on update cascade
        on delete cascade,

    concept_id uuid not null
        references public.concepts(id)
        on update cascade
        on delete cascade,

    weight double precision not null,

    created_at timestamptz not null default timezone('utc', now()),

    primary key (chunk_id, concept_id),

    constraint chunk_concepts_weight_valid
        check (weight between -1.0 and 1.0)
);

create index chunk_concepts_concept_weight_idx
    on public.chunk_concepts (
        concept_id,
        weight desc
    );

create index chunk_concepts_chunk_weight_idx
    on public.chunk_concepts (
        chunk_id,
        weight desc
    );


-- ---------------------------------------------------------------------------
-- Queries
-- ---------------------------------------------------------------------------

create table public.queries (
    id uuid primary key default extensions.gen_random_uuid(),

    question_hash text not null,
    question_text text,

    concept_activations jsonb not null default '{}'::jsonb,
    mapping_method text not null,
    claim_type_breakdown jsonb not null default '{}'::jsonb,

    created_at timestamptz not null default timezone('utc', now()),

    constraint queries_question_hash_not_blank
        check (length(btrim(question_hash)) > 0),

    constraint queries_question_text_not_blank
        check (
            question_text is null
            or length(btrim(question_text)) > 0
        ),

    constraint queries_mapping_method_valid
        check (
            mapping_method in (
                'anchor_vector',
                'llm_fallback'
            )
        ),

    constraint queries_concept_activations_object
        check (jsonb_typeof(concept_activations) = 'object'),

    constraint queries_claim_type_breakdown_object
        check (jsonb_typeof(claim_type_breakdown) = 'object')
);

create index queries_question_hash_idx
    on public.queries (question_hash);

create index queries_created_at_idx
    on public.queries (created_at desc);


-- ---------------------------------------------------------------------------
-- Responses
-- ---------------------------------------------------------------------------

create table public.responses (
    id uuid primary key default extensions.gen_random_uuid(),

    query_id uuid not null unique
        references public.queries(id)
        on update cascade
        on delete cascade,

    corpus_version_id uuid not null
        references public.corpus_versions(id)
        on update cascade
        on delete restrict,

    overall_coverage text not null,
    domain_coverage jsonb not null default '{}'::jsonb,

    relationship_type text,
    tension_summary text,

    generation_model text not null,
    embedding_model text not null,

    mapper_latency_ms integer,
    retrieval_latency_ms integer,
    generation_latency_ms integer,
    synthesis_latency_ms integer,
    total_latency_ms integer not null,

    created_at timestamptz not null default timezone('utc', now()),

    constraint responses_overall_coverage_valid
        check (
            overall_coverage in (
                'supported',
                'partially_supported',
                'out_of_corpus'
            )
        ),

    constraint responses_domain_coverage_object
        check (jsonb_typeof(domain_coverage) = 'object'),

    constraint responses_relationship_type_valid
        check (
            relationship_type is null
            or relationship_type in (
                'genuine_disagreement',
                'surface_similarity_deep_difference',
                'not_comparable',
                'no_tension'
            )
        ),

    constraint responses_generation_model_not_blank
        check (length(btrim(generation_model)) > 0),

    constraint responses_embedding_model_not_blank
        check (length(btrim(embedding_model)) > 0),

    constraint responses_mapper_latency_valid
        check (
            mapper_latency_ms is null
            or mapper_latency_ms >= 0
        ),

    constraint responses_retrieval_latency_valid
        check (
            retrieval_latency_ms is null
            or retrieval_latency_ms >= 0
        ),

    constraint responses_generation_latency_valid
        check (
            generation_latency_ms is null
            or generation_latency_ms >= 0
        ),

    constraint responses_synthesis_latency_valid
        check (
            synthesis_latency_ms is null
            or synthesis_latency_ms >= 0
        ),

    constraint responses_total_latency_valid
        check (total_latency_ms >= 0)
);

create index responses_corpus_version_idx
    on public.responses (corpus_version_id);

create index responses_created_at_idx
    on public.responses (created_at desc);

create index responses_overall_coverage_idx
    on public.responses (overall_coverage);


-- ---------------------------------------------------------------------------
-- Claim-level generated output
-- ---------------------------------------------------------------------------

create table public.response_claims (
    id uuid primary key default extensions.gen_random_uuid(),

    response_id uuid not null
        references public.responses(id)
        on update cascade
        on delete cascade,

    domain text not null,
    claim_order integer not null,
    claim_text text not null,

    created_at timestamptz not null default timezone('utc', now()),

    constraint response_claims_domain_valid
        check (
            domain in (
                'science',
                'advaita',
                'samkhya'
            )
        ),

    constraint response_claims_claim_order_valid
        check (claim_order >= 0),

    constraint response_claims_claim_text_not_blank
        check (length(btrim(claim_text)) > 0),

    constraint response_claims_unique_order
        unique (
            response_id,
            domain,
            claim_order
        )
);

create index response_claims_response_idx
    on public.response_claims (
        response_id,
        domain,
        claim_order
    );


-- ---------------------------------------------------------------------------
-- Enforceable claim-to-citation grounding
-- ---------------------------------------------------------------------------

create table public.claim_citations (
    claim_id uuid not null
        references public.response_claims(id)
        on update cascade
        on delete cascade,

    chunk_id text not null
        references public.chunks(id)
        on update cascade
        on delete restrict,

    citation_order integer not null default 0,

    created_at timestamptz not null default timezone('utc', now()),

    primary key (claim_id, chunk_id),

    constraint claim_citations_order_valid
        check (citation_order >= 0)
);

create index claim_citations_claim_order_idx
    on public.claim_citations (
        claim_id,
        citation_order
    );

create index claim_citations_chunk_idx
    on public.claim_citations (chunk_id);


-- ---------------------------------------------------------------------------
-- Evaluation questions
-- ---------------------------------------------------------------------------

create table public.eval_questions (
    id uuid primary key default extensions.gen_random_uuid(),

    question_text text not null,
    expected_concepts jsonb not null default '{}'::jsonb,
    expected_claim_type text,

    expected_coverage text,
    reviewer_notes text,
    reviewed_by text,

    is_active boolean not null default true,

    created_at timestamptz not null default timezone('utc', now()),
    updated_at timestamptz not null default timezone('utc', now()),

    constraint eval_questions_question_text_not_blank
        check (length(btrim(question_text)) > 0),

    constraint eval_questions_expected_concepts_object
        check (jsonb_typeof(expected_concepts) = 'object'),

    constraint eval_questions_expected_claim_type_valid
        check (
            expected_claim_type is null
            or expected_claim_type in (
                'empirical',
                'metaphysical',
                'normative',
                'mixed'
            )
        ),

    constraint eval_questions_expected_coverage_valid
        check (
            expected_coverage is null
            or expected_coverage in (
                'supported',
                'partially_supported',
                'out_of_corpus'
            )
        )
);

create index eval_questions_active_idx
    on public.eval_questions (is_active);

create trigger eval_questions_set_updated_at
before update on public.eval_questions
for each row
execute function public.set_updated_at();


-- ---------------------------------------------------------------------------
-- Row-level security
--
-- No public policies are created. The FastAPI backend and controlled ingestion
-- process use the Supabase service-role credential, which bypasses RLS.
-- ---------------------------------------------------------------------------

alter table public.corpus_versions enable row level security;
alter table public.sources enable row level security;
alter table public.concepts enable row level security;
alter table public.chunks enable row level security;
alter table public.chunk_concepts enable row level security;
alter table public.queries enable row level security;
alter table public.responses enable row level security;
alter table public.response_claims enable row level security;
alter table public.claim_citations enable row level security;
alter table public.eval_questions enable row level security;


-- ---------------------------------------------------------------------------
-- Explicit grants
--
-- Public clients receive no direct table access. All application data access
-- goes through FastAPI.
-- ---------------------------------------------------------------------------

revoke all on table public.corpus_versions from anon, authenticated;
revoke all on table public.sources from anon, authenticated;
revoke all on table public.concepts from anon, authenticated;
revoke all on table public.chunks from anon, authenticated;
revoke all on table public.chunk_concepts from anon, authenticated;
revoke all on table public.queries from anon, authenticated;
revoke all on table public.responses from anon, authenticated;
revoke all on table public.response_claims from anon, authenticated;
revoke all on table public.claim_citations from anon, authenticated;
revoke all on table public.eval_questions from anon, authenticated;


-- ---------------------------------------------------------------------------
-- Initial corpus version
-- ---------------------------------------------------------------------------

insert into public.corpus_versions (
    version,
    description,
    is_active
)
values (
    'phase0-v1',
    'Initial WTH schema and Phase 0 corpus foundation.',
    true
);

commit;