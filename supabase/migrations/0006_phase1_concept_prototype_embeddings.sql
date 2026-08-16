begin;

create table if not exists public.concept_prototype_embeddings (
    record_id text primary key,

    concept_id uuid not null
        references public.concepts(id)
        on update cascade
        on delete restrict,

    prototype_version text not null,
    prototype_role text not null,
    record_type text not null,

    embedding extensions.vector(768) not null,

    provider text not null,
    model text not null,
    model_revision text not null,
    dimensions integer not null,
    normalization text not null,
    task_type text not null,
    task_instruction text not null,

    text_checksum text not null,
    embedding_input_checksum text not null,
    embedding_checksum text not null,
    embedding_origin text not null,

    source_artifact_sha256 text not null,

    chunk_id text,
    source_id text,
    domain text,
    evaluation_split text,
    citation text,
    title text,
    reviewed_labels jsonb,

    estimated_tokens integer,
    actual_tokens integer,
    embedding_created_at timestamptz not null,

    loaded_at timestamptz not null
        default timezone('utc'::text, now()),
    updated_at timestamptz not null
        default timezone('utc'::text, now()),

    constraint concept_prototype_embeddings_record_id_not_blank
        check (length(btrim(record_id)) > 0),

    constraint concept_prototype_embeddings_version_not_blank
        check (length(btrim(prototype_version)) > 0),

    constraint concept_prototype_embeddings_role_valid
        check (
            prototype_role in (
                'question',
                'positive',
                'hard_negative'
            )
        ),

    constraint concept_prototype_embeddings_record_type_valid
        check (
            record_type in (
                'query_prototype',
                'passage_prototype'
            )
        ),

    constraint concept_prototype_embeddings_role_type_consistent
        check (
            (
                prototype_role = 'question'
                and record_type = 'query_prototype'
                and task_type = 'search_query'
                and chunk_id is null
                and source_id is null
                and domain is null
                and evaluation_split is null
            )
            or
            (
                prototype_role in ('positive', 'hard_negative')
                and record_type = 'passage_prototype'
                and task_type = 'search_document'
                and chunk_id is not null
                and source_id is not null
                and domain in ('science', 'advaita', 'samkhya')
                and evaluation_split = 'build'
            )
        ),

    constraint concept_prototype_embeddings_provider_locked
        check (provider = 'Google Gemini API'),

    constraint concept_prototype_embeddings_model_locked
        check (model = 'gemini-embedding-2'),

    constraint concept_prototype_embeddings_model_revision_locked
        check (model_revision = '2'),

    constraint concept_prototype_embeddings_dimensions_locked
        check (dimensions = 768),

    constraint concept_prototype_embeddings_normalization_locked
        check (normalization = 'provider_auto_l2'),

    constraint concept_prototype_embeddings_origin_locked
        check (embedding_origin = 'provider'),

    constraint concept_prototype_embeddings_text_checksum_sha256
        check (text_checksum ~ '^[0-9a-f]{64}$'),

    constraint concept_prototype_embeddings_input_checksum_sha256
        check (embedding_input_checksum ~ '^[0-9a-f]{64}$'),

    constraint concept_prototype_embeddings_embedding_checksum_sha256
        check (embedding_checksum ~ '^[0-9a-f]{64}$'),

    constraint concept_prototype_embeddings_source_artifact_sha256
        check (source_artifact_sha256 ~ '^[0-9a-f]{64}$'),

    constraint concept_prototype_embeddings_estimated_tokens_valid
        check (
            estimated_tokens is null
            or estimated_tokens >= 0
        ),

    constraint concept_prototype_embeddings_actual_tokens_valid
        check (
            actual_tokens is null
            or actual_tokens >= 0
        )
);

create unique index if not exists
    concept_prototype_embeddings_version_record_idx
on public.concept_prototype_embeddings (
    prototype_version,
    record_id
);

create index if not exists
    concept_prototype_embeddings_lookup_idx
on public.concept_prototype_embeddings (
    prototype_version,
    concept_id,
    prototype_role,
    record_id
);

create index if not exists
    concept_prototype_embeddings_checksum_idx
on public.concept_prototype_embeddings (
    embedding_checksum
);

comment on table public.concept_prototype_embeddings is
'Frozen WTH Phase 1 concept-prototype embeddings used by the Phase 10 mapping
and Phase 14 runtime concept activation. These rows are production bootstrap
data, not query-time generated vectors.';

comment on column public.concept_prototype_embeddings.record_id is
'Stable Phase 9 prototype embedding record_id from the frozen JSONL artifact.';

comment on column public.concept_prototype_embeddings.prototype_role is
'question, positive, or hard_negative. Runtime ConceptActivationService maps
positive rows to its passage prototype bank.';

comment on column public.concept_prototype_embeddings.source_artifact_sha256 is
'Semantic SHA-256 of the frozen source JSONL from which this exact row was loaded.';

-- Keep updated_at semantics consistent with the rest of the schema.
create or replace function public.set_concept_prototype_embeddings_updated_at()
returns trigger
language plpgsql
as $function$
begin
    new.updated_at = timezone('utc'::text, now());
    return new;
end;
$function$;

drop trigger if exists concept_prototype_embeddings_set_updated_at
    on public.concept_prototype_embeddings;

create trigger concept_prototype_embeddings_set_updated_at
before update on public.concept_prototype_embeddings
for each row
execute function public.set_concept_prototype_embeddings_updated_at();

-- Backend-only posture.
alter table public.concept_prototype_embeddings
    enable row level security;

revoke all on table public.concept_prototype_embeddings
    from public;

revoke all on table public.concept_prototype_embeddings
    from anon;

revoke all on table public.concept_prototype_embeddings
    from authenticated;

grant select, insert, update, delete
    on table public.concept_prototype_embeddings
    to service_role;

commit;
