begin;

-- ---------------------------------------------------------------------------
-- 0002: Query-time provenance chain + concept-level coverage
--
-- Adds runtime provenance for the weighted multi-concept Semantic Mapper:
--
--   queries
--      -> query_concepts
--      -> retrieval_results
--      -> claim_citations
--      -> response_claims
--
-- chunk_concepts continues to represent corpus-level affinity.
-- retrieval_results records which concept actually caused a chunk to be
-- retrieved for a specific query.
-- ---------------------------------------------------------------------------


-- ---------------------------------------------------------------------------
-- query_concepts
--
-- Relational source of truth for the concepts activated by the Semantic
-- Mapper for a particular query.
--
-- queries.concept_activations remains as a denormalized JSON snapshot.
-- queries.mapping_method remains the query-level record of whether the
-- anchor-vector or LLM-fallback path produced the concept bundle.
-- ---------------------------------------------------------------------------

create table public.query_concepts (
    query_id uuid not null
        references public.queries(id)
        on update cascade
        on delete cascade,

    concept_id uuid not null
        references public.concepts(id)
        on update cascade
        on delete restrict,

    activation_weight double precision not null,
    activation_rank integer not null,

    created_at timestamptz not null default timezone('utc', now()),

    primary key (query_id, concept_id),

    constraint query_concepts_activation_weight_valid
        check (activation_weight between -1.0 and 1.0),

    constraint query_concepts_activation_rank_valid
        check (activation_rank >= 1),

    constraint query_concepts_unique_rank_per_query
        unique (query_id, activation_rank)
);

create index query_concepts_concept_idx
    on public.query_concepts (concept_id);

create index query_concepts_query_rank_idx
    on public.query_concepts (
        query_id,
        activation_rank
    );


-- ---------------------------------------------------------------------------
-- retrieval_results
--
-- Records one query-specific retrieval event for each:
--
--   query + activated concept + domain + chunk
--
-- The scoring fields are deliberately separated:
--
-- query_concept_weight:
--   How strongly the user's question activated the concept.
--
-- chunk_concept_weight:
--   The corpus-level affinity between the chunk and the concept, copied from
--   chunk_concepts at retrieval time for reproducible historical provenance.
--
-- similarity_score:
--   Query-to-chunk cosine similarity.
--
-- combined_score:
--   Final normalized retrieval score used for ranking. The retrieval
--   implementation must keep this value within [-1, 1].
-- ---------------------------------------------------------------------------

create table public.retrieval_results (
    id uuid primary key default extensions.gen_random_uuid(),

    query_id uuid not null,
    concept_id uuid not null,

    domain text not null,

    chunk_id text not null
        references public.chunks(id)
        on update cascade
        on delete restrict,

    query_concept_weight double precision not null,
    chunk_concept_weight double precision not null,
    similarity_score double precision not null,
    combined_score double precision not null,

    retrieval_rank integer not null,

    created_at timestamptz not null default timezone('utc', now()),

    constraint retrieval_results_query_concept_fk
        foreign key (query_id, concept_id)
        references public.query_concepts(query_id, concept_id)
        on update cascade
        on delete cascade,

    constraint retrieval_results_domain_valid
        check (
            domain in (
                'science',
                'advaita',
                'samkhya'
            )
        ),

    constraint retrieval_results_query_concept_weight_valid
        check (
            query_concept_weight between -1.0 and 1.0
        ),

    constraint retrieval_results_chunk_concept_weight_valid
        check (
            chunk_concept_weight between -1.0 and 1.0
        ),

    constraint retrieval_results_similarity_score_valid
        check (
            similarity_score between -1.0 and 1.0
        ),

    constraint retrieval_results_combined_score_valid
        check (
            combined_score between -1.0 and 1.0
        ),

    constraint retrieval_results_retrieval_rank_valid
        check (retrieval_rank >= 1),

    constraint retrieval_results_unique_hit
        unique (
            query_id,
            concept_id,
            domain,
            chunk_id
        ),

    constraint retrieval_results_unique_rank
        unique (
            query_id,
            concept_id,
            domain,
            retrieval_rank
        )
);

create index retrieval_results_query_idx
    on public.retrieval_results (query_id);

create index retrieval_results_concept_idx
    on public.retrieval_results (concept_id);

create index retrieval_results_chunk_idx
    on public.retrieval_results (chunk_id);

create index retrieval_results_query_concept_domain_rank_idx
    on public.retrieval_results (
        query_id,
        concept_id,
        domain,
        retrieval_rank
    );

create index retrieval_results_query_domain_score_idx
    on public.retrieval_results (
        query_id,
        domain,
        combined_score desc
    );


-- ---------------------------------------------------------------------------
-- claim_citations rework
--
-- claim_citations previously referenced chunks directly.
--
-- It now references retrieval_results so each citation carries:
--
-- - query provenance
-- - concept provenance
-- - domain
-- - source chunk
-- - component scores
-- - final rank
--
-- A claim can cite retrieval results from multiple concepts, preserving
-- many-to-many claim-to-concept provenance.
-- ---------------------------------------------------------------------------

alter table public.claim_citations
    drop constraint claim_citations_pkey,
    drop constraint claim_citations_chunk_id_fkey,
    drop column chunk_id,
    add column retrieval_result_id uuid;

alter table public.claim_citations
    add constraint claim_citations_retrieval_result_fkey
        foreign key (retrieval_result_id)
        references public.retrieval_results(id)
        on update cascade
        on delete restrict;

alter table public.claim_citations
    alter column retrieval_result_id set not null;

alter table public.claim_citations
    add constraint claim_citations_pkey
        primary key (
            claim_id,
            retrieval_result_id
        );

drop index if exists public.claim_citations_chunk_idx;

create index claim_citations_retrieval_result_idx
    on public.claim_citations (retrieval_result_id);

create unique index claim_citations_claim_order_unique_idx
    on public.claim_citations (
        claim_id,
        citation_order
    );


-- ---------------------------------------------------------------------------
-- response_concept_coverage
--
-- Records deterministic coverage for each concept activated by the query.
--
-- This exists alongside responses.domain_coverage because domain-level and
-- concept-level support answer different questions.
--
-- Example:
--
-- Domain coverage:
--   science  = supported
--   advaita  = supported
--   samkhya  = partially_supported
--
-- Concept coverage:
--   morality = supported
--   agency   = partially_supported
--   suffering = unsupported
-- ---------------------------------------------------------------------------

create table public.response_concept_coverage (
    response_id uuid not null
        references public.responses(id)
        on update cascade
        on delete cascade,

    concept_id uuid not null
        references public.concepts(id)
        on update cascade
        on delete restrict,

    coverage_status text not null,

    supporting_domain_count integer not null default 0,
    supporting_chunk_count integer not null default 0,

    strongest_score double precision,

    created_at timestamptz not null default timezone('utc', now()),

    primary key (
        response_id,
        concept_id
    ),

    constraint response_concept_coverage_status_valid
        check (
            coverage_status in (
                'supported',
                'partially_supported',
                'unsupported'
            )
        ),

    constraint response_concept_coverage_domain_count_valid
        check (
            supporting_domain_count between 0 and 3
        ),

    constraint response_concept_coverage_chunk_count_valid
        check (
            supporting_chunk_count >= 0
        ),

    constraint response_concept_coverage_strongest_score_valid
        check (
            strongest_score is null
            or strongest_score between -1.0 and 1.0
        ),

    constraint response_concept_coverage_evidence_consistent
        check (
            (
                coverage_status = 'unsupported'
                and supporting_domain_count = 0
                and supporting_chunk_count = 0
                and strongest_score is null
            )
            or
            (
                coverage_status in (
                    'supported',
                    'partially_supported'
                )
                and supporting_domain_count >= 1
                and supporting_chunk_count >= 1
                and strongest_score is not null
            )
        )
);

create index response_concept_coverage_concept_idx
    on public.response_concept_coverage (concept_id);

create index response_concept_coverage_status_idx
    on public.response_concept_coverage (coverage_status);

create index response_concept_coverage_response_status_idx
    on public.response_concept_coverage (
        response_id,
        coverage_status
    );


-- ---------------------------------------------------------------------------
-- Convenience view for response rendering, debugging, and evaluation
--
-- Prevents application code from repeatedly rebuilding the complete:
--
-- claim -> retrieval event -> concept -> chunk
--
-- join chain.
-- ---------------------------------------------------------------------------

create view public.claim_citation_details
with (security_invoker = true)
as
select
    cc.claim_id,
    cc.retrieval_result_id,
    cc.citation_order,

    rc.response_id,
    rc.domain as claim_domain,
    rc.claim_order,
    rc.claim_text,

    r.query_id,

    rr.concept_id,
    c.slug as concept_slug,
    c.display_name as concept_display_name,

    rr.domain as retrieval_domain,
    rr.chunk_id,

    ch.citation,
    ch.full_text,

    rr.query_concept_weight,
    rr.chunk_concept_weight,
    rr.similarity_score,
    rr.combined_score,
    rr.retrieval_rank,

    cc.created_at
from public.claim_citations cc
join public.response_claims rc
    on rc.id = cc.claim_id
join public.responses r
    on r.id = rc.response_id
join public.retrieval_results rr
    on rr.id = cc.retrieval_result_id
join public.concepts c
    on c.id = rr.concept_id
join public.chunks ch
    on ch.id = rr.chunk_id;


-- ---------------------------------------------------------------------------
-- Row-level security and grants
--
-- Same posture as 0001:
-- service-role backend only, with no direct anon/authenticated access.
-- ---------------------------------------------------------------------------

alter table public.query_concepts
    enable row level security;

alter table public.retrieval_results
    enable row level security;

alter table public.response_concept_coverage
    enable row level security;

revoke all
    on table public.query_concepts
    from anon, authenticated;

revoke all
    on table public.retrieval_results
    from anon, authenticated;

revoke all
    on table public.response_concept_coverage
    from anon, authenticated;

revoke all
    on table public.claim_citation_details
    from anon, authenticated;

commit;