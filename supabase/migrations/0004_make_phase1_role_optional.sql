-- WTH Stage 1 pre-load schema correction
-- phase1_role is not part of the frozen Phase 12 per-relation contract.
-- Keep the column for optional future provenance, but do not require it.

begin;

alter table public.chunk_concepts
    alter column phase1_role drop not null;

alter table public.chunk_concepts
    drop constraint if exists chunk_concepts_phase1_role_not_blank;

comment on column public.chunk_concepts.phase1_role is
    'Optional relation-level role metadata. Frozen Phase 1 does not require this field on every chunk-concept row.';

commit;
