# WTH Phase 1 Corpus Policy

**Project:** WTH — What The Human
**Phase:** Phase 1 — Three-Concept Corpus Slice
**Policy version:** 1.0.0
**Status:** Approved for Phase 1 implementation
**Effective date:** 2026-08-02
**Corpus owner:** WTH project team
**Applies to:** Science, Advaita Vedanta, and Samkhya corpus ingestion

---

## 1. Purpose

This policy defines how sources are selected, acquired, parsed, chunked, embedded, classified, reviewed, cited, versioned, and activated for the Phase 1 WTH corpus.

The Phase 1 corpus supports comparative retrieval across three concepts:

1. `consciousness`
2. `self_identity`
3. `reality_appearance`

and three domains:

1. `science`
2. `advaita`
3. `samkhya`

The policy is intended to ensure that every active corpus chunk is:

* attributable to a known source;
* legally and ethically usable;
* traceable to its original location;
* correctly assigned to a domain;
* represented by a 768-dimensional embedding;
* associated with reviewed concept weights;
* supported by a stable citation;
* reviewed before activation;
* reproducible from versioned source and processing metadata.

This policy governs corpus construction. It does not govern final answer generation, synthesis, or user-interface presentation except where those functions depend on corpus provenance.

---

## 2. Phase 1 scope

### 2.1 Included concepts

Phase 1 includes only:

| Concept slug         | Display name           | Scope                                                                                                                                    |
| -------------------- | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `consciousness`      | Consciousness          | Subjective experience, awareness, sentience, witnessing, and closely related accounts of conscious experience                            |
| `self_identity`      | Self and identity      | Personal identity, selfhood, ego, subjectivity, Atman, Purusha, self-models, and related distinctions                                    |
| `reality_appearance` | Reality and appearance | Perception, representation, illusion, appearance, Maya, manifestation, and distinctions between what appears and what is held to be real |

References to the other five canonical WTH concepts may remain in a source, but Phase 1 does not require assigning or evaluating those concepts.

### 2.2 Included domains

Phase 1 includes:

* modern scientific literature;
* Advaita Vedanta primary texts and clearly identified commentarial traditions;
* classical Samkhya texts and clearly identified commentarial traditions.

### 2.3 Excluded Phase 1 activities

The following are outside the Phase 1 corpus scope:

* production answer generation;
* comparative synthesis across domains;
* inclusion of all eight canonical concepts;
* automated activation without human review;
* unrestricted web scraping;
* ingestion of user-generated content;
* ingestion of personal or confidential data;
* OCR-heavy bulk ingestion without quality review;
* treating general Hindu philosophy as interchangeable with Advaita or Samkhya;
* using one domain to explain, reinterpret, or validate another domain.

---

## 3. Governing principles

### 3.1 Domain independence

Each domain must be represented according to its own terminology, source hierarchy, methods, and claims.

The corpus must not:

* translate philosophical claims into scientific claims;
* represent scientific hypotheses as confirmations of metaphysical positions;
* merge Advaita and Samkhya because they share Sanskrit terminology;
* suppress disagreement to create artificial convergence;
* use one tradition’s vocabulary as the default ontology for another.

### 3.2 Provenance before volume

A smaller corpus with complete provenance is preferable to a larger corpus with uncertain rights, weak citations, or unreliable parsing.

No source or chunk may become active merely because it improves corpus size.

### 3.3 Review before activation

All newly generated chunks begin in a non-active state.

The required lifecycle is:

```text
draft → reviewed → active
```

A chunk may move directly from `draft` to `rejected`, but it may not move directly from `draft` to `active`.

### 3.4 Reproducibility

Given the same:

* source files;
* source catalogue;
* parser versions;
* chunker versions;
* normalization rules;
* embedding model and configuration;
* concept anchors;
* review decisions;

the project should be able to reproduce the same active corpus, except where an external embedding service produces documented non-deterministic variation.

### 3.5 No hidden correction

Parsers and reviewers must not silently rewrite a source to make it clearer, more accurate, more modern, or more compatible with another source.

Normalization may correct machine-level formatting problems, but substantive wording must remain faithful to the selected edition.

---

## 4. Source classes

Every source must be assigned one `source_type`.

Recommended Phase 1 values are:

* `paper`
* `primary_text`
* `commentary`

Additional internal distinctions may be captured in source metadata.

### 4.1 Scientific sources

Scientific sources may include:

* peer-reviewed review articles;
* peer-reviewed empirical studies;
* peer-reviewed theoretical papers;
* scholarly reference articles with explicit reuse rights.

A scientific source must clearly identify:

* authors;
* title;
* publication year;
* journal or publisher;
* stable identifier, where available;
* licence or reuse status;
* canonical source location.

Preprints may be included only when:

* they are clearly labelled as preprints;
* their reuse terms permit ingestion;
* their non-peer-reviewed status is retained in metadata;
* their inclusion is justified by a corpus reviewer.

Retracted papers must not be active.

### 4.2 Advaita sources

Advaita sources may include:

* selected Upanishadic passages;
* Brahma Sutra material relevant to the Phase 1 concepts;
* works attributed to or traditionally associated with Shankara;
* identified Shankara commentaries;
* later Advaita commentaries or explanatory works.

Each source must distinguish among:

* root or primary text;
* traditional commentary;
* translator commentary;
* editor introduction;
* later interpretation.

An Upanishadic passage must not automatically be classified as Advaita doctrine. Its use in the Advaita corpus must be justified through an identified Advaita edition, commentary, or interpretive context.

### 4.3 Samkhya sources

Samkhya sources may include:

* Samkhya Karika;
* classical commentaries on Samkhya texts;
* historically significant translations;
* rights-approved scholarly explanations of classical Samkhya.

The source must preserve distinctions among concepts such as:

* Purusha;
* Prakriti;
* buddhi;
* ahamkara;
* manas;
* the gunas;
* the evolutes of Prakriti.

Yoga, Vedanta, Tantra, or general Indian philosophy must not be classified as Samkhya solely because they use overlapping terminology.

---

## 5. Source authority

Each source must receive an `authority_level`.

Recommended values are:

| Authority level        | Meaning                                                            |
| ---------------------- | ------------------------------------------------------------------ |
| `primary`              | Root text, original scientific article, or original scholarly work |
| `canonical_commentary` | Historically recognized commentary tied directly to a primary text |
| `scholarly_secondary`  | Academic analysis, review, translation, or interpretation          |
| `contextual`           | Background material used to clarify terminology or source history  |
| `excluded`             | Insufficient authority for active retrieval                        |

Contextual sources may help reviewers understand the corpus but should not normally supply active answer evidence unless their use is explicitly approved.

Popular articles, unsourced summaries, commercial blogs, AI-generated summaries, discussion forums, and anonymous web content are excluded from the active Phase 1 corpus.

---

## 6. Rights, licence, and reuse requirements

### 6.1 General rule

Access to a source does not by itself establish the right to ingest, transform, store, embed, redistribute, or expose its text.

Every source must have an item-level rights determination before ingestion.

The determination must record:

* licence name;
* licence URL or authoritative rights reference;
* rights statement;
* copyright holder, when known;
* permitted use;
* attribution requirements;
* redistribution restrictions;
* derivative-work restrictions;
* commercial-use restrictions;
* jurisdiction notes;
* date checked;
* reviewer or person who checked it.

### 6.2 Rights statuses

Each source must receive one of these statuses:

| Status                     | Meaning                                                         |
| -------------------------- | --------------------------------------------------------------- |
| `eligible`                 | Rights have been checked and Phase 1 use is permitted           |
| `eligible_with_conditions` | Use is permitted subject to recorded conditions                 |
| `pending_review`           | Rights have not been sufficiently verified                      |
| `restricted`               | Source may be consulted but not ingested into the active corpus |
| `rejected`                 | Source cannot be used                                           |

Only `eligible` and `eligible_with_conditions` sources may produce active chunks.

### 6.3 PMC sources

A scientific article being available through PubMed Central does not necessarily mean that its full text may be reused. PMC states that not every PMC article is available for text mining or reuse and identifies the PMC Open Access Subset as the collection intended for articles made available under reusable licence terms.

Therefore:

* inclusion in PMC alone is insufficient;
* the article should normally appear in the PMC Open Access Subset;
* the item-level licence must still be captured;
* licence conditions must be applied to stored text, review packets, and downstream displays;
* removed or rights-changed items must be capable of being deactivated.

### 6.4 Creative Commons sources

For Creative Commons material, the source record must preserve the exact licence variant.

Attribution metadata must include, where applicable:

* creator;
* title;
* source;
* copyright notice;
* licence;
* licence reference;
* indication of modifications.

Creative Commons licences commonly require appropriate credit, a licence reference, and disclosure of modifications. NonCommercial, NoDerivatives, and ShareAlike variants impose additional constraints and must not be treated as equivalent to CC BY.

A NoDerivatives licence requires legal review before transformed text, normalized extracts, review packets, or generated corpus derivatives are distributed beyond internal analysis.

### 6.5 Project Gutenberg sources

Project Gutenberg primarily evaluates copyright status under United States law. It warns users outside the United States to verify the law applicable in their own jurisdiction and does not guarantee that a work is unrestricted elsewhere.

For every Project Gutenberg source:

* retain its ebook identifier;
* retain the title, author, translator, and edition information;
* record the rights statement included with that specific ebook;
* record that Project Gutenberg’s determination is principally based on United States law;
* perform or document a separate jurisdictional check where corpus development, hosting, or redistribution occurs outside the United States;
* do not assume that the underlying translation is unrestricted merely because the original classical text is ancient;
* preserve any required Project Gutenberg licence notices when applicable;
* avoid implying endorsement by Project Gutenberg.

Where practical, the project should distinguish the underlying public-domain work from Project Gutenberg-specific headers, formatting, and trademark references.

### 6.6 Legal-review limitation

This policy defines project controls and is not a legal opinion.

Any source with uncertain jurisdiction, translation rights, database rights, contractual restrictions, or commercial-use limitations must remain `pending_review` or `restricted` until appropriately reviewed.

---

## 7. Source inclusion criteria

A source is eligible for Phase 1 when all applicable criteria are satisfied:

* it is relevant to at least one Phase 1 concept;
* it belongs clearly to one Phase 1 domain;
* its authority level is recorded;
* its reuse status is verified;
* its full text is obtainable in a reliable format;
* its edition or publication identity is known;
* its authorship or traditional attribution is recorded;
* its content can be cited with stable locators;
* its parsing quality can be reviewed;
* its inclusion does not create unacceptable corpus imbalance;
* its provenance can be preserved through ingestion.

Scientific sources should normally have a clear abstract, body structure, and stable identifier.

Classical-text sources should normally have chapter, section, verse, sutra, karika, or comparable structural divisions.

---

## 8. Source exclusion criteria

A source must be excluded or restricted when:

* reuse rights are unknown or incompatible;
* the source cannot be reliably attributed;
* the text is a machine-generated summary;
* the edition or translation cannot be identified;
* OCR quality prevents reliable reconstruction;
* the source contains substantial corruption or missing sections;
* it is a duplicate of a preferred edition;
* it is a retracted scientific article;
* it materially misrepresents its stated tradition;
* its relevance is only superficial keyword overlap;
* it cannot produce stable citations;
* it is primarily promotional, devotional, polemical, or popularized material without sufficient scholarly value;
* it would cause one author, translator, source, or school to dominate the corpus;
* it contains personal, confidential, or unlawfully obtained information.

An excluded source must retain a catalogue record when it was seriously considered for inclusion. The record should include an exclusion reason.

---

## 9. Corpus balance

Phase 1 should target:

* 4–6 sources per domain;
* 12–18 sources overall;
* 60–90 active chunks per domain;
* 180–270 active chunks overall;
* at least 15 relevant active chunks for each concept-domain cell.

The nine concept-domain cells are:

| Concept                | Science  | Advaita  | Samkhya  |
| ---------------------- | -------- | -------- | -------- |
| Consciousness          | Required | Required | Required |
| Self and identity      | Required | Required | Required |
| Reality and appearance | Required | Required | Required |

A chunk may contribute to more than one cell.

No single source should normally account for more than 25% of the active chunks in its domain.

A source exceeding this threshold requires a documented exception explaining why its structural importance outweighs the concentration risk.

Balance does not mean forcing equal doctrinal positions or equal numbers of claims. It means ensuring that retrieval outcomes are not determined mainly by source volume.

---

## 10. Source catalogue requirements

Every considered source must be registered in:

```text
data/catalogues/phase1_sources.yaml
```

before production ingestion.

Each catalogue record must include, where applicable:

```text
source_id
domain
title
subtitle
author
traditional_attribution
translator
editor
publication_year
edition
publisher
journal
volume
issue
page_range
source_type
authority_level
canonical_url
download_url
external_identifier
format
language
original_language
licence_name
licence_reference
rights_statement
rights_status
rights_jurisdiction
rights_checked_at
rights_checked_by
accessed_at
source_checksum
included_concepts
inclusion_status
inclusion_notes
exclusion_reason
```

A source ID must be stable and must not depend solely on a mutable URL.

---

## 11. Source acquisition and storage

### 11.1 Raw-source preservation

The original acquired file must be preserved unchanged in the ingestion workspace.

Recommended structure:

```text
data/
  raw/
    science/
    advaita/
    samkhya/
```

The raw source must be accompanied by:

* a cryptographic checksum;
* acquisition timestamp;
* source catalogue ID;
* original filename;
* media type;
* acquisition method.

### 11.2 No silent replacement

When the upstream source changes:

* preserve the earlier raw version;
* calculate a new checksum;
* create a new acquisition record;
* assess whether the change affects existing chunks;
* create a new corpus version when active content changes.

A mutable URL must not be treated as proof that two downloaded files are identical.

### 11.3 Automated acquisition

Automated downloading must respect:

* provider terms;
* documented APIs or bulk-download mechanisms;
* rate limits;
* robots and access restrictions where applicable;
* item-level reuse rights.

The project must not bypass authentication, paywalls, technical restrictions, or access controls.

---

## 12. Parsing policy

### 12.1 Parser separation

Parsers must be source-format specific.

Phase 1 parser classes are expected for:

* PMC JATS XML;
* Project Gutenberg or equivalent structured HTML;
* approved structured plain text.

PDF must not be the preferred ingestion format when reliable XML, HTML, EPUB, or structured text is available.

### 12.2 Normalized parser output

Every parser must produce a shared normalized document representation containing:

* source ID;
* document title;
* domain;
* source metadata;
* hierarchical sections;
* structural locators;
* normalized text;
* parser name;
* parser version;
* parser warnings.

Chunkers must consume this normalized representation rather than raw HTML, XML, or OCR output.

### 12.3 Permitted normalization

Permitted normalization includes:

* Unicode normalization;
* consistent line endings;
* removal of navigation menus;
* removal of duplicated headers and footers;
* removal of Project Gutenberg boilerplate where permitted;
* conversion of typographic whitespace;
* reconstruction of paragraphs split only by file formatting;
* preservation of meaningful headings and locators.

### 12.4 Prohibited normalization

Normalization must not:

* paraphrase the author;
* modernize substantive wording without documentation;
* replace philosophical terminology with assumed equivalents;
* merge translator commentary into primary text;
* convert hypotheses into factual statements;
* remove qualifications, uncertainty, negation, or counterarguments;
* correct a source’s doctrinal position;
* silently repair uncertain OCR text.

Uncertain text must be flagged for review.

---

## 13. Chunking policy

### 13.1 General requirements

A chunk must represent a coherent unit that can be retrieved and cited independently.

Each chunk must contain:

* stable chunk ID;
* source ID;
* domain;
* text;
* structural locator;
* citation;
* token count;
* parser version;
* chunker version;
* source checksum;
* review status.

Chunk boundaries should follow source structure before token count.

### 13.2 Scientific chunking

Scientific chunks should normally:

* contain approximately 250–450 tokens;
* remain within one major section;
* preserve complete arguments, findings, definitions, or qualifications;
* avoid separating a conclusion from its essential limitation;
* exclude bibliographies unless references themselves are under study;
* identify whether text came from an abstract, introduction, methods, results, discussion, conclusion, caption, or other section.

Overlap should normally remain between 0 and 60 tokens and should be used only when necessary to preserve context.

### 13.3 Classical-text chunking

Advaita and Samkhya chunks should normally:

* contain approximately 120–300 tokens;
* respect chapter, section, verse, sutra, karika, or commentary boundaries;
* preserve the distinction between root text and commentary;
* preserve translator and edition identity;
* avoid combining unrelated passages merely to reach a target size;
* use minimal overlap when stable textual divisions already exist.

A short verse or sutra may be combined with its directly associated commentary when the combined unit is explicitly labelled and remains attributable.

### 13.4 Chunk independence

A chunk should not depend on hidden preceding text to reverse its meaning.

Where context is essential, the chunk may include a limited contextual lead-in or be rejected as unsuitable for independent retrieval.

### 13.5 Duplicate control

Exact and near-duplicate chunks must be detected before activation.

When duplicate content exists across editions or mirrors, the preferred version should be selected according to:

1. rights clarity;
2. source authority;
3. citation stability;
4. text quality;
5. metadata completeness.

Alternate editions may remain when their differences are substantively important.

---

## 14. Citation policy

Every active chunk must have a human-readable citation and a machine-readable locator.

### 14.1 Scientific citation

A scientific citation should include:

* author or shortened author list;
* publication year;
* article title;
* journal or publisher;
* section or page locator where available;
* stable identifier such as DOI or PMCID.

Example pattern:

```text
Author et al. (Year), “Article Title,” Journal, section: Discussion, PMCID: PMC1234567.
```

### 14.2 Classical-text citation

A classical-text citation should include:

* work;
* author or traditional attribution;
* translator;
* edition or publication year;
* chapter, section, verse, sutra, or karika locator;
* commentary identity when applicable.

Example pattern:

```text
Samkhya Karika, trans. [Translator], Karika 17–19, [Edition].
```

### 14.3 Citation integrity

A citation must resolve to the chunk’s actual source location.

Reviewers must reject:

* fabricated page numbers;
* inferred verse numbers;
* locators taken from another edition;
* citations that identify only a website homepage;
* citations that obscure whether text is source, translation, or commentary.

---

## 15. Embedding policy

### 15.1 Phase 1 embedding configuration

Phase 1 uses:

```text
Provider: Google Gemini
Model: gemini-embedding-001
Dimensions: 768
Vector storage: pgvector
```

The embedding configuration must be frozen for the Phase 1 baseline experiment.

Both retrieval baselines must use the same query and chunk embeddings.

### 15.2 Embedding metadata

For each embedded chunk, the project must retain or be able to reconstruct:

* embedding provider;
* model;
* output dimension;
* task type;
* embedding configuration version;
* normalized-text checksum;
* embedding timestamp;
* embedding status;
* error information, if generation failed.

### 15.3 Embedding eligibility

Only chunks that have passed source and parsing validation may be embedded for activation.

An embedding must not be used when:

* its dimension differs from 768;
* the source text changed after embedding;
* the chunk checksum no longer matches;
* the embedding request failed partially;
* the vector contains invalid values;
* the model identity is unknown.

### 15.4 Embedding updates

Changing any of the following requires a new embedding version:

* model;
* output dimensionality;
* task type;
* normalization rule;
* chunk text;
* chunk boundaries.

Existing vectors must not be silently overwritten in a frozen experimental corpus.

---

## 16. Concept-anchor policy

Phase 1 uses reviewed anchor definitions for:

* `consciousness`
* `self_identity`
* `reality_appearance`

Each anchor must include:

* canonical definition;
* positive indicators;
* domain-specific terminology;
* related but non-equivalent terms;
* explicit exclusions;
* common confusions;
* anchor text;
* anchor version;
* review status.

Concept anchors must be reviewed before their embeddings are written to the concepts table.

An anchor must not define a concept using only one domain’s worldview.

It may contain domain-specific language, but it must distinguish terminology rather than collapse it.

---

## 17. Weighted chunk concepts

### 17.1 Plural assignment

A chunk may be relevant to multiple concepts.

Concept weights are independent relevance values and must not be forced to sum to one.

For example, a passage may be strongly relevant to both consciousness and self/identity.

### 17.2 Initial weight interpretation

Phase 1 may use the following provisional interpretation:

|    Weight | Interpretation               |
| --------: | ---------------------------- |
| 0.00–0.24 | Not relevant                 |
| 0.25–0.49 | Weak or contextual relevance |
| 0.50–0.74 | Relevant                     |
| 0.75–1.00 | Central or strongly relevant |

These ranges are calibration aids, not permanent semantic definitions.

### 17.3 Proposed versus approved weights

Automated similarity may generate proposed concept weights.

Proposed weights must remain distinguishable from reviewer-approved weights.

Only reviewed weights may be used in the frozen Phase 1 concept-aware baseline.

### 17.4 Minimum active requirement

Every active Phase 1 chunk must have at least one approved Phase 1 concept weight.

A chunk with no approved concept relevance must not be active in the Phase 1 corpus.

---

## 18. Review policy

### 18.1 Review packet

Reviewable chunks must be exported to:

```text
artifacts/review/phase1_review_packet.csv
artifacts/review/phase1_review_packet.html
```

The packet must show:

* source ID;
* source title;
* domain;
* source type;
* authority level;
* licence;
* rights status;
* citation;
* structural locator;
* chunk text;
* token count;
* parser warnings;
* embedding status;
* proposed concept weights;
* reviewer-adjusted concept weights;
* review decision;
* review notes.

### 18.2 Review decisions

Allowed review decisions are:

* `approve`
* `approve_with_edits`
* `reject`
* `needs_source_review`
* `needs_license_review`

### 18.3 Approval criteria

A chunk may be approved only when:

* the source is eligible;
* its domain assignment is correct;
* its text is faithful to the source;
* its citation is accurate;
* its boundaries are coherent;
* its concept relevance is defensible;
* any primary-text/commentary distinction is visible;
* its embedding is valid or ready to be generated;
* no unresolved parser warning affects meaning.

### 18.4 Rejection reasons

Recommended rejection reasons include:

* `rights_unclear`
* `wrong_domain`
* `insufficient_relevance`
* `citation_invalid`
* `parser_error`
* `ocr_uncertain`
* `duplicate`
* `chunk_too_fragmentary`
* `chunk_mixes_source_types`
* `translation_unknown`
* `source_authority_insufficient`
* `retracted_source`
* `other`

### 18.5 Reviewer identity

Review records must identify the reviewer or review process.

Automated review alone is insufficient for Phase 1 activation.

---

## 19. Activation policy

Activation must be performed by a separate, auditable operation after review.

The activation process must verify that each chunk has:

* an eligible source;
* a recorded licence or rights basis;
* a valid domain;
* a stable citation;
* a structural locator;
* a 768-dimensional embedding;
* at least one reviewed concept weight;
* an approved review decision;
* a target corpus version.

If any required property is missing, activation must fail for that chunk.

Partial activation is permitted only when failed chunks are clearly reported and do not silently disappear from the activation record.

---

## 20. Corpus versioning

Every active corpus release must have a unique corpus version.

The initial Phase 1 release is expected to use a version such as:

```text
phase1-three-concept-v1
```

A new corpus version is required when active retrieval content changes because of:

* source addition or removal;
* source rights change;
* parser change affecting text;
* chunk-boundary change;
* citation correction;
* concept-weight correction;
* embedding-model or configuration change;
* review reversal;
* retraction or source-quality issue.

A frozen experiment must refer to one immutable corpus version.

---

## 21. Baseline experiment controls

Phase 1 evaluates:

### Baseline A

Plain vector similarity within each domain.

### Baseline B

Vector similarity combined with weighted concept alignment.

To preserve experimental validity, both baselines must use:

* the same active corpus version;
* the same query set;
* the same chunk embeddings;
* the same query embeddings;
* the same domain filters;
* the same candidate pool, unless a reranking experiment explicitly says otherwise;
* the same relevance judgments;
* the same evaluation cutoffs.

Concept weights used in Baseline B must be frozen before held-out evaluation.

The held-out test set must not be used to:

* tune score weights;
* change activation thresholds;
* rewrite concept anchors;
* select sources;
* adjust chunk boundaries;
* revise relevance labels.

Any post-test changes require a newly identified experiment run.

---

## 22. Rights changes, corrections, and removals

A source or chunk must be deactivated when:

* its reuse rights are withdrawn or materially changed;
* a scientific source is retracted;
* its citation is shown to be materially incorrect;
* its translation or edition was misidentified;
* its parser output altered the source’s meaning;
* its domain assignment was materially wrong;
* its continued use creates a legal, ethical, or scholarly integrity risk.

Deactivation must preserve an audit record.

Affected embeddings, concept weights, retrieval results, and evaluations must remain traceable to the corpus version in which they were used.

---

## 23. Security and privacy

Phase 1 is limited to published scholarly and classical-text materials.

The corpus must not contain:

* personal account information;
* private correspondence;
* unpublished personal records;
* patient data;
* protected health information;
* authentication credentials;
* access tokens;
* confidential manuscripts;
* unlawfully obtained material.

Raw-source storage and review artifacts must not expose API keys or secret configuration values.

---

## 24. Required audit information

For every active chunk, the system must store directly or make reproducibly available through linked manifests:

```text
chunk_id
source_id
source_checksum
domain
source_type
authority_level
licence_name
rights_status
citation
structural_locator
normalized_text
normalized_text_checksum
parser_name
parser_version
chunker_name
chunker_version
token_count
embedding_provider
embedding_model
embedding_dimension
embedding_task_type
embedding_status
concept_weights
concept_weight_review_status
review_decision
reviewer
reviewed_at
corpus_version
activation_status
activated_at
```

---

## 25. Phase 1 exit gate

Phase 1 corpus construction is complete only when every active chunk satisfies all of the following:

```text
✓ Linked to a valid source
✓ Assigned to one domain
✓ Source authority recorded
✓ Licence or rights basis recorded
✓ Rights status eligible
✓ Source checksum recorded
✓ Parser and chunker versions recorded
✓ Stable citation recorded
✓ Structural locator recorded
✓ Normalized-text checksum recorded
✓ Gemini embedding present
✓ Embedding dimension equals 768
✓ Embedding configuration recorded
✓ At least one reviewed concept weight present
✓ Review decision approved
✓ Review status active
✓ Corpus version assigned
```

The corpus-level exit gate additionally requires:

```text
✓ All three domains represented
✓ All three Phase 1 concepts represented
✓ All nine concept-domain cells meet minimum coverage
✓ No unresolved rights issues among active sources
✓ No unresolved parser errors among active chunks
✓ Review packet archived
✓ Activation manifest archived
✓ Baseline A completed
✓ Baseline B completed
✓ Held-out evaluation completed
✓ Keep, modify, or reject decision recorded for concept-aware retrieval
```

---

## 26. Exceptions

Any exception to this policy must be documented with:

* affected source or chunk;
* policy provision being overridden;
* reason;
* risk assessment;
* compensating control;
* approver;
* approval date;
* expiry or review date.

Exceptions must not be used to bypass uncertain rights, fabricated provenance, or required human review.

---

## 27. Policy maintenance

This policy must be reviewed when:

* Phase 1 scope changes;
* a new source repository is introduced;
* a new source format is ingested;
* corpus hosting or jurisdiction changes;
* a new embedding model is adopted;
* active content is redistributed externally;
* a rights dispute occurs;
* the project moves from research use toward public or commercial deployment.

Policy changes must increment the policy version and must not silently alter the interpretation of a frozen corpus or experiment.
