# Phase 1 Candidate Corpus Status

## Classification

- **Corpus version:** `phase1_candidate_corpus_v1`
- **Classification:** Candidate corpus
- **Review status:** Not reviewed
- **Phase 1 activation status:** Not active
- **Production retrieval status:** Not eligible
- **Evaluation gold status:** Not gold-labelled
- **Direct activation eligible:** No
- **Database activation permitted:** No
- **Verified chunk count:** 7,469
- **Frozen and working copies match:** Yes

## Authoritative status statement

The 7,469 chunks in this corpus are broad,
source-derived candidate material.

They have not received passage-level Phase 1 relevance review and are
not approved as the active consciousness, self/identity, and
reality/appearance corpus.

These chunks must not be activated directly or used as production
retrieval evidence.

## Distribution by domain

- **advaita:** 1,657 chunks
- **samkhya:** 5,714 chunks
- **science:** 98 chunks

## Distribution by source

- `advaita_paramananda_1917_vedanta_in_practice`: 91 chunks
- `advaita_paramananda_1919_upanishads`: 193 chunks
- `advaita_sankara_johnston_1946_crest_jewel`: 255 chunks
- `advaita_sankara_thibaut_1890_vedanta_sutras`: 1,118 chunks
- `samkhya_dasgupta_1922_history_volume1_samkhya`: 1,193 chunks
- `samkhya_isvarakrishna_davies_1881_karika`: 416 chunks
- `samkhya_sinha_1915_samkhya_philosophy`: 3,574 chunks
- `samkhya_vachaspati_jha_1896_tattva_kaumudi`: 531 chunks
- `science_herzog_kammer_scharnowski_2016_time_slices`: 15 chunks
- `science_ionta_gassert_blanke_2011_bodily_self_consciousness`: 19 chunks
- `science_limanowski_blankenburg_2013_minimal_self_models`: 34 chunks
- `science_petkova_ehrsson_2008_body_swapping`: 30 chunks

## Intended uses

- Source-section scope analysis for the three-concept Phase 1 vertical slice.
- Rule-based selection of consciousness, self_identity, and reality_appearance candidate passages.
- Selection of adjacent-concept hard negatives.
- Future expansion to the remaining five canonical concept families.
- Reproducible parser, chunker, and corpus quality analysis.

## Prohibited uses

- Direct activation into the Phase 1 production retrieval corpus.
- Bulk approval without passage-level human review.
- Use as the held-out gold evaluation dataset.
- Use for final claim-cited generation before Phase 1 relevance review.
- Automatic selection based solely on the embedding or anchor model being evaluated.

## Why direct activation is blocked

- The corpus contains broad source-derived material outside consciousness, self_identity, and reality_appearance.
- The corpus has not received passage-level human review.
- The corpus is materially imbalanced across Science, Advaita Vedanta, and Samkhya.
- Concept relevance and hard-negative labels have not been frozen.
- Development and held-out evaluation splits have not been created.

## Required next steps

1. Create the source-structure report.
2. Create and approve source-section scope metadata.
3. Run independent rule-based Phase 1 candidate selection.
4. Perform human review and produce a balanced 250-350 chunk vertical slice.
5. Freeze build, development, and held-out evaluation sets.
6. Generate approved-corpus embeddings and concept weights.
7. Activate only the reviewed Phase 1 slice.

## Provenance

- Frozen snapshot: `E:\ABHINAV\Coding\Projects\WTH-WhatTheHuman\artifacts\archive\phase1_candidate_corpus_v1`
- Frozen chunks: `E:\ABHINAV\Coding\Projects\WTH-WhatTheHuman\artifacts\archive\phase1_candidate_corpus_v1\artifacts\phase1\chunks`
- Working chunks: `E:\ABHINAV\Coding\Projects\WTH-WhatTheHuman\artifacts\phase1\chunks`
- Freeze manifest: `E:\ABHINAV\Coding\Projects\WTH-WhatTheHuman\artifacts\archive\phase1_candidate_corpus_v1\freeze_manifest.json`
- Freeze-manifest SHA-256:
  `eea8359d45e4f0a16993cf69211d860d8b650c32b6ec761c60faf12a307e8ac2`
- Classification created:
  `2026-08-05T14:53:36.076323+00:00`
