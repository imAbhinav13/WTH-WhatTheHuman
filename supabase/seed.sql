begin;

-- ---------------------------------------------------------------------------
-- Canonical WTH concept ontology
--
-- All eight concepts are seeded now so their identifiers remain stable across
-- local, test, and production environments.
--
-- Phase 1 initially builds corpus coverage for:
--   consciousness
--   self_identity
--   reality_appearance
--
-- The remaining concepts exist in the ontology but will not participate in
-- retrieval until they have anchor embeddings and active corpus coverage.
-- ---------------------------------------------------------------------------

insert into public.concepts (
    id,
    slug,
    display_name,
    description,
    anchor_text,
    is_active
)
values
    (
        '10000000-0000-4000-8000-000000000001'::uuid,
        'self_identity',
        'Self and Identity',
        'Questions about personal identity, the enduring or constructed self, ego, individuality, continuity through change, and what makes a person the same person over time.',
        'self identity personhood ego individuality personal continuity who am I enduring self constructed self illusion of self subject identity through time',
        true
    ),
    (
        '10000000-0000-4000-8000-000000000002'::uuid,
        'consciousness',
        'Consciousness',
        'Questions about awareness, subjective experience, sentience, the observer, phenomenal experience, and the origin or nature of consciousness.',
        'consciousness awareness subjective experience sentience observer witness phenomenal experience qualia wakefulness origin and nature of awareness',
        true
    ),
    (
        '10000000-0000-4000-8000-000000000003'::uuid,
        'reality_appearance',
        'Reality and Appearance',
        'Questions about what is fundamentally real, whether experienced reality is deceptive or constructed, and the distinction between appearance and underlying reality.',
        'reality appearance illusion perception fundamental reality maya seeming and being objective reality constructed experience what is ultimately real',
        true
    ),
    (
        '10000000-0000-4000-8000-000000000004'::uuid,
        'matter_mind',
        'Matter and Mind',
        'Questions about the relationship between physical matter and mental experience, including dualism, physicalism, emergence, and mind-body interaction.',
        'mind matter body brain mental physical dualism physicalism emergence mind body problem relationship between consciousness and material processes',
        true
    ),
    (
        '10000000-0000-4000-8000-000000000005'::uuid,
        'cosmology_origins',
        'Cosmology and Origins',
        'Questions about the beginning, structure, cause, duration, and ultimate nature of the universe and existence.',
        'cosmology origins universe beginning creation existence cause of the cosmos big bang eternal universe cycles of creation why there is something',
        true
    ),
    (
        '10000000-0000-4000-8000-000000000006'::uuid,
        'agency_free_will',
        'Agency and Free Will',
        'Questions about choice, intention, control, determinism, human agency, autonomy, and whether actions could have been otherwise.',
        'agency free will choice intention control autonomy determinism decision action responsibility ability to do otherwise voluntary action',
        true
    ),
    (
        '10000000-0000-4000-8000-000000000007'::uuid,
        'causation_karma',
        'Causation and Karma',
        'Questions about cause and effect across actions and consequences, conditioning, karma, recurrence, and whether events follow moral or impersonal causal structures.',
        'causation karma cause effect consequence conditioning action result recurrence past actions causal chain moral causation impersonal causality',
        true
    ),
    (
        '10000000-0000-4000-8000-000000000008'::uuid,
        'moral_responsibility_suffering',
        'Moral Responsibility and Suffering',
        'Questions about blame, guilt, justice, moral accountability, suffering, punishment, fairness, and responsibility for harmful outcomes.',
        'moral responsibility suffering guilt blame justice fairness punishment accountability harm innocence consequence why bad things happen responsibility for actions',
        true
    )
on conflict (slug)
do update set
    display_name = excluded.display_name,
    description = excluded.description,
    anchor_text = excluded.anchor_text,
    is_active = excluded.is_active,
    updated_at = timezone('utc', now());

commit;