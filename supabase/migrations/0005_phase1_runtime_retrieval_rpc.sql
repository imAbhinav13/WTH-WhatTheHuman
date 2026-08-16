-- WTH Stage 3.5A — production Phase 14 pgvector candidate RPC
--
-- Purpose:
--   Expose the frozen Phase 1 active corpus to FastAPI runtime retrieval
--   without changing the validated Phase 14 Python ranking algorithm.
--
-- Critical parity rule:
--   Historical Phase 14 computes:
--
--     vector similarity
--       + concept alignment
--       + human relevance
--       + citation quality
--       -> base score
--       -> sort
--       -> candidate_pool_per_domain (30)
--       -> diversity / dedup / token-budget selection
--
--   Therefore this RPC MUST NOT pre-limit each domain to the top 30 rows by
--   vector similarity. For phase1_active_corpus_v1 it returns the complete
--   active domain candidate set (up to p_candidate_limit, runtime value 318).
--   Python then applies the original Phase 14 ranking unchanged.
--
-- Security:
--   Backend service-role only. No anon/authenticated execute permission.
--
-- Corpus invariants already established by Production Stage 1:
--   318 active chunks
--   954 chunk-concept relations
--   10 sources
--   vector(768)
--   phase1_active_corpus_v1

begin;

drop function if exists public.match_phase1_active_chunks(
    extensions.vector,
    text,
    text,
    integer
);

create function public.match_phase1_active_chunks(
    p_query_embedding extensions.vector(768),
    p_domain text,
    p_corpus_version text,
    p_candidate_limit integer default 318
)
returns table (
    chunk_id text,
    source_id text,
    domain text,
    citation text,
    reviewed_text text,
    corpus_version text,
    vector_similarity double precision,
    source_title text,
    translator text,
    concept_relations jsonb
)
language plpgsql
stable
security invoker
set search_path = public, extensions
as $function$
begin
    if p_query_embedding is null then
        raise exception 'p_query_embedding must not be null';
    end if;

    if p_domain not in ('science', 'advaita', 'samkhya') then
        raise exception 'Unsupported Phase 1 domain: %', p_domain;
    end if;

    if p_corpus_version is null or length(btrim(p_corpus_version)) = 0 then
        raise exception 'p_corpus_version must be non-empty';
    end if;

    if p_candidate_limit is null or p_candidate_limit < 1 then
        raise exception 'p_candidate_limit must be >= 1';
    end if;

    if p_candidate_limit > 1000 then
        raise exception 'p_candidate_limit must be <= 1000';
    end if;

    return query
    with vector_candidates as (
        select
            c.id as chunk_id,
            c.source_id,
            c.domain,
            c.citation,
            c.full_text as reviewed_text,
            cv.version as corpus_version,
            (
                1.0 - (c.embedding <=> p_query_embedding)
            )::double precision as vector_similarity,
            s.title as source_title,
            s.translator
        from public.chunks as c
        join public.sources as s
          on s.id = c.source_id
        join public.corpus_versions as cv
          on cv.id = s.corpus_version_id
        where c.review_status = 'active'
          and c.domain = p_domain
          and cv.version = p_corpus_version
          and cv.is_active = true
        order by
            c.embedding <=> p_query_embedding,
            c.source_id,
            c.id
        limit p_candidate_limit
    ),
    phase1_relations as (
        select
            vc.chunk_id,
            jsonb_agg(
                jsonb_build_object(
                    'concept_id', concept.slug,
                    'human_label', relation.human_label,
                    'production_active', relation.production_active,
                    'calibrated_weight', relation.weight,
                    'human_override', relation.human_override
                )
                order by concept.slug
            ) as concept_relations,
            count(*) as relation_count
        from vector_candidates as vc
        join public.chunk_concepts as relation
          on relation.chunk_id = vc.chunk_id
        join public.concepts as concept
          on concept.id = relation.concept_id
        where concept.slug in (
            'consciousness',
            'self_identity',
            'reality_appearance'
        )
        group by vc.chunk_id
    )
    select
        vc.chunk_id,
        vc.source_id,
        vc.domain,
        vc.citation,
        vc.reviewed_text,
        vc.corpus_version,
        vc.vector_similarity,
        vc.source_title,
        vc.translator,
        relations.concept_relations
    from vector_candidates as vc
    join phase1_relations as relations
      on relations.chunk_id = vc.chunk_id
    where relations.relation_count = 3
    order by
        vc.vector_similarity desc,
        vc.source_id,
        vc.chunk_id;
end;
$function$;

comment on function public.match_phase1_active_chunks(
    extensions.vector,
    text,
    text,
    integer
) is
'Stage 3.5 Phase 14 runtime candidate retrieval. Returns the active,
domain-separated frozen Phase 1 candidate set with pgvector cosine similarity
and all three reviewed concept relations. Python retains ownership of the
validated Phase 14 concept-aware ranking, candidate-pool truncation, source
diversity, deduplication, and context-budget selection.';

revoke all on function public.match_phase1_active_chunks(
    extensions.vector,
    text,
    text,
    integer
) from public;

revoke all on function public.match_phase1_active_chunks(
    extensions.vector,
    text,
    text,
    integer
) from anon;

revoke all on function public.match_phase1_active_chunks(
    extensions.vector,
    text,
    text,
    integer
) from authenticated;

grant execute on function public.match_phase1_active_chunks(
    extensions.vector,
    text,
    text,
    integer
) to service_role;

commit;
