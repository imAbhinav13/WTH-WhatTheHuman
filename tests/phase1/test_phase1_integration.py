from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final

from scripts.assemble_phase1_final_response import (
    DEFAULT_COVERAGE,
    DEFAULT_COVERAGE_MANIFEST,
    DEFAULT_DOMAIN_RESPONSES,
    DEFAULT_EVIDENCE_PACKAGE,
    DEFAULT_GENERATION_MANIFEST,
    DEFAULT_RETRIEVAL_MANIFEST,
    DEFAULT_SYNTHESIS,
    DEFAULT_SYNTHESIS_MANIFEST,
    build_claim_index,
    build_evidence_index,
    parse_domain_responses,
    parse_query_activation,
    parse_synthesis,
    run_phase18,
)

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]

SELECTION_CANDIDATES: Final = (
    PROJECT_ROOT / "artifacts" / "phase1" / "selection" / "phase1_selection_candidates.jsonl"
)
GOLD_CORPUS: Final = (
    PROJECT_ROOT / "artifacts" / "phase1" / "reviewed" / "phase1_reviewed_gold_corpus.jsonl"
)
REVIEW_MANIFEST: Final = (
    PROJECT_ROOT / "artifacts" / "phase1" / "reviewed" / "phase1_human_review_manifest.json"
)

BUILD_SPLIT: Final = PROJECT_ROOT / "data" / "evaluation" / "phase1_build.jsonl"
DEVELOPMENT_SPLIT: Final = PROJECT_ROOT / "data" / "evaluation" / "phase1_development.jsonl"
HELDOUT_SPLIT: Final = PROJECT_ROOT / "data" / "evaluation" / "phase1_heldout.jsonl"
SPLIT_MANIFEST: Final = PROJECT_ROOT / "data" / "evaluation" / "phase1_split_manifest.json"

APPROVED_EMBEDDINGS: Final = (
    PROJECT_ROOT / "artifacts" / "phase1" / "embeddings" / "approved_chunk_embeddings.jsonl"
)
EMBEDDING_MANIFEST: Final = (
    PROJECT_ROOT / "artifacts" / "phase1" / "embeddings" / "embedding_manifest.json"
)

ACTIVE_BUNDLES: Final = (
    PROJECT_ROOT / "artifacts" / "phase1" / "active" / "active_chunk_bundles.jsonl"
)
ACTIVE_CONCEPTS: Final = (
    PROJECT_ROOT / "artifacts" / "phase1" / "active" / "reviewed_chunk_concepts.jsonl"
)
ACTIVATION_MANIFEST: Final = (
    PROJECT_ROOT / "artifacts" / "phase1" / "active" / "activation_manifest.json"
)

EVIDENCE_PACKAGE: Final = PROJECT_ROOT / DEFAULT_EVIDENCE_PACKAGE
RETRIEVAL_MANIFEST: Final = PROJECT_ROOT / DEFAULT_RETRIEVAL_MANIFEST
DOMAIN_RESPONSES: Final = PROJECT_ROOT / DEFAULT_DOMAIN_RESPONSES
GENERATION_MANIFEST: Final = PROJECT_ROOT / DEFAULT_GENERATION_MANIFEST
SYNTHESIS: Final = PROJECT_ROOT / DEFAULT_SYNTHESIS
SYNTHESIS_MANIFEST: Final = PROJECT_ROOT / DEFAULT_SYNTHESIS_MANIFEST
COVERAGE: Final = PROJECT_ROOT / DEFAULT_COVERAGE
COVERAGE_MANIFEST: Final = PROJECT_ROOT / DEFAULT_COVERAGE_MANIFEST

FROZEN_FINAL_JSON: Final = PROJECT_ROOT / "artifacts" / "phase1" / "final" / "final_response.json"
FROZEN_FINAL_MARKDOWN: Final = PROJECT_ROOT / "artifacts" / "phase1" / "final" / "final_response.md"
FROZEN_FINAL_MANIFEST: Final = (
    PROJECT_ROOT / "artifacts" / "phase1" / "final" / "final_response_manifest.json"
)

EXPECTED_APPROVED_CHUNKS: Final = 318
EXPECTED_SELECTION_CANDIDATES: Final = 424
EXPECTED_CONCEPT_ROWS: Final = 954
EXPECTED_SPLIT_COUNTS: Final = {
    "build": 159,
    "development": 80,
    "heldout": 79,
}
EXPECTED_DOMAINS: Final = {
    "science",
    "advaita",
    "samkhya",
}
EXPECTED_CONCEPTS: Final = {
    "consciousness",
    "self_identity",
    "reality_appearance",
}
VALID_REVIEW_DECISIONS: Final = {
    "include",
    "include_with_edits",
}
VALID_CONCEPT_LABELS: Final = {
    "positive",
    "partial",
    "negative",
}
EXPECTED_CORPUS_VERSION: Final = "phase1_active_corpus_v1"


def require_file(path: Path) -> Path:
    assert path.is_file(), f"Required Phase 1 artifact missing: {path}"
    return path


def require_mapping(
    value: object,
    description: str,
) -> dict[str, object]:
    assert isinstance(value, Mapping), f"{description} must be an object"
    result: dict[str, object] = {}
    for key, nested in value.items():
        assert isinstance(key, str), f"{description} contains a non-string key"
        result[key] = nested
    return result


def require_list(
    value: object,
    description: str,
) -> list[object]:
    assert isinstance(value, list), f"{description} must be a list"
    return value


def require_string(
    value: object,
    description: str,
) -> str:
    assert isinstance(value, str), f"{description} must be a string"
    stripped = value.strip()
    assert stripped, f"{description} must be non-empty"
    return stripped


def load_json(path: Path) -> dict[str, object]:
    require_file(path)
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    return require_mapping(
        raw,
        f"JSON document {path}",
    )


def iter_jsonl(
    path: Path,
) -> Iterable[dict[str, object]]:
    require_file(path)
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, raw_line in enumerate(
            handle,
            start=1,
        ):
            stripped = raw_line.strip()
            if not stripped:
                continue

            raw: object = json.loads(stripped)
            yield require_mapping(
                raw,
                f"{path}:{line_number}",
            )


def load_jsonl(
    path: Path,
) -> list[dict[str, object]]:
    return list(iter_jsonl(path))


def chunk_id(
    record: Mapping[str, object],
    description: str,
) -> str:
    return require_string(
        record.get("chunk_id"),
        f"{description} chunk_id",
    )


def chunk_ids(
    records: Iterable[Mapping[str, object]],
    description: str,
) -> set[str]:
    result: set[str] = set()

    for index, record in enumerate(
        records,
        start=1,
    ):
        identifier = chunk_id(
            record,
            f"{description}[{index}]",
        )
        assert identifier not in result, f"Duplicate chunk_id {identifier!r} in {description}"
        result.add(identifier)

    return result


def nested_exit_gate(
    manifest: Mapping[str, object],
    description: str,
) -> dict[str, object]:
    return require_mapping(
        manifest.get("exit_gate"),
        f"{description} exit_gate",
    )


def assert_all_boolean_gate_values_true(
    gate: Mapping[str, object],
    description: str,
    *,
    ignored_keys: set[str] | None = None,
) -> None:
    ignored = ignored_keys or set()

    for key, value in gate.items():
        if key in ignored:
            continue
        if isinstance(value, bool):
            assert value is True, f"{description} exit gate {key!r} is not true"


def comparable_final_response(
    response: Mapping[str, object],
) -> dict[str, object]:
    """
    Remove only inherently run-specific metadata.

    Phase 18 is deterministic from Phases 14-17. The user-facing sections,
    citation registry, versions, question, corpus version, and validation
    should therefore reproduce exactly.
    """
    comparable = dict(response)
    comparable.pop("generated_at", None)
    return comparable


# ---------------------------------------------------------------------------
# Candidate -> scope/selection -> human review
# ---------------------------------------------------------------------------


def test_selection_to_reviewed_gold_lineage_is_preserved() -> None:
    selected = load_jsonl(SELECTION_CANDIDATES)
    gold = load_jsonl(GOLD_CORPUS)

    assert len(selected) == EXPECTED_SELECTION_CANDIDATES
    assert len(gold) == EXPECTED_APPROVED_CHUNKS

    selected_by_id = {chunk_id(record, "selection candidate"): record for record in selected}
    gold_by_id = {chunk_id(record, "gold record"): record for record in gold}

    assert set(gold_by_id).issubset(selected_by_id), (
        "Reviewed gold corpus contains chunks that did not pass Phase 3 selection"
    )

    for identifier, reviewed in gold_by_id.items():
        selected_record = selected_by_id[identifier]

        assert require_string(
            reviewed.get("source_id"),
            f"{identifier} source_id",
        ) == require_string(
            selected_record.get("source_id"),
            f"{identifier} selected source_id",
        )
        assert require_string(
            reviewed.get("domain"),
            f"{identifier} domain",
        ) == require_string(
            selected_record.get("domain"),
            f"{identifier} selected domain",
        )

        selection_rule_ids = reviewed.get("selection_rule_ids")
        assert selection_rule_ids not in (
            None,
            "",
            [],
        ), f"{identifier} lost transparent scope/selection rule provenance"

        review = require_mapping(
            reviewed.get("review"),
            f"{identifier} review",
        )

        labels = require_mapping(
            review.get("labels"),
            f"{identifier} review.labels",
        )

        for concept in EXPECTED_CONCEPTS:
            label = require_string(
                labels.get(concept),
                f"{identifier} review.labels.{concept}",
            )
            assert label in VALID_CONCEPT_LABELS

    review_manifest = load_json(REVIEW_MANIFEST)
    assert review_manifest.get("status") == "phase1_human_review_complete"
    assert review_manifest.get("strict_gate_passed") is True

    summary = require_mapping(
        review_manifest.get("summary"),
        "review manifest summary",
    )

    assert summary.get("approved_rows") == EXPECTED_APPROVED_CHUNKS


# ---------------------------------------------------------------------------
# Reviewed gold -> deterministic evaluation splits
# ---------------------------------------------------------------------------


def test_frozen_splits_are_an_exact_non_leaking_partition_of_gold() -> None:
    gold = load_jsonl(GOLD_CORPUS)
    build = load_jsonl(BUILD_SPLIT)
    development = load_jsonl(DEVELOPMENT_SPLIT)
    heldout = load_jsonl(HELDOUT_SPLIT)

    assert len(build) == EXPECTED_SPLIT_COUNTS["build"]
    assert len(development) == (EXPECTED_SPLIT_COUNTS["development"])
    assert len(heldout) == EXPECTED_SPLIT_COUNTS["heldout"]

    gold_ids = chunk_ids(
        gold,
        "gold",
    )
    build_ids = chunk_ids(
        build,
        "build",
    )
    development_ids = chunk_ids(
        development,
        "development",
    )
    heldout_ids = chunk_ids(
        heldout,
        "heldout",
    )

    assert build_ids.isdisjoint(development_ids)
    assert build_ids.isdisjoint(heldout_ids)
    assert development_ids.isdisjoint(heldout_ids)

    assert (build_ids | development_ids | heldout_ids) == gold_ids

    split_manifest = load_json(SPLIT_MANIFEST)
    gate = nested_exit_gate(
        split_manifest,
        "split manifest",
    )
    for required_true in (
        "splits_checksummed",
        "heldout_marked_read_only",
        "distribution_report_generated",
    ):
        assert gate.get(required_true) is True

    for required_false in (
        "passage_family_leakage",
        "exact_duplicate_leakage",
        "high_overlap_leakage",
    ):
        assert gate.get(required_false) is False


# ---------------------------------------------------------------------------
# Reviewed gold -> embedding
# ---------------------------------------------------------------------------


def test_every_reviewed_gold_chunk_has_exactly_one_frozen_embedding() -> None:
    gold = load_jsonl(GOLD_CORPUS)
    embeddings = load_jsonl(APPROVED_EMBEDDINGS)

    assert len(embeddings) == (EXPECTED_APPROVED_CHUNKS)

    gold_ids = chunk_ids(
        gold,
        "gold",
    )
    embedding_ids = chunk_ids(
        embeddings,
        "approved embeddings",
    )
    assert embedding_ids == gold_ids

    identities: set[tuple[str, str, str, int, str]] = set()

    for record in embeddings:
        provider = require_string(
            record.get("provider"),
            "embedding provider",
        )
        model = require_string(
            record.get("model"),
            "embedding model",
        )
        revision = require_string(
            record.get("model_revision"),
            "embedding revision",
        )
        dimensions_raw = record.get("dimensions")
        assert isinstance(
            dimensions_raw,
            int,
        )
        task_type = require_string(
            record.get("task_type"),
            "embedding task_type",
        )
        normalization = require_string(
            record.get("normalization"),
            "embedding normalization",
        )

        identities.add(
            (
                provider,
                model,
                revision,
                dimensions_raw,
                normalization,
            )
        )

        assert "mock" not in (provider.casefold())
        require_string(
            record.get("text_checksum"),
            "embedding text_checksum",
        )
        require_string(
            record.get("embedding_checksum"),
            "embedding checksum",
        )
        assert task_type

    assert len(identities) == 1

    (
        provider,
        model,
        revision,
        dimensions,
        _normalization,
    ) = next(iter(identities))
    assert provider == "Google Gemini API"
    assert model == "gemini-embedding-2"
    assert revision == "2"
    assert dimensions == 768

    embedding_manifest = load_json(EMBEDDING_MANIFEST)
    assert embedding_manifest.get("status") == "complete"
    gate = nested_exit_gate(
        embedding_manifest,
        "embedding manifest",
    )
    assert_all_boolean_gate_values_true(
        gate,
        "embedding manifest",
    )


# ---------------------------------------------------------------------------
# Embedding + reviewed concepts -> active corpus
# ---------------------------------------------------------------------------


def test_activation_preserves_gold_embedding_and_three_concept_rows() -> None:
    gold = load_jsonl(GOLD_CORPUS)
    embeddings = load_jsonl(APPROVED_EMBEDDINGS)
    bundles = load_jsonl(ACTIVE_BUNDLES)
    concept_rows = load_jsonl(ACTIVE_CONCEPTS)

    gold_ids = chunk_ids(
        gold,
        "gold",
    )
    embedding_ids = chunk_ids(
        embeddings,
        "approved embeddings",
    )
    bundle_ids = chunk_ids(
        bundles,
        "active bundles",
    )

    assert len(bundles) == (EXPECTED_APPROVED_CHUNKS)
    assert len(concept_rows) == (EXPECTED_CONCEPT_ROWS)
    assert gold_ids == embedding_ids == bundle_ids

    concepts_by_chunk: defaultdict[
        str,
        set[str],
    ] = defaultdict(set)

    for row in concept_rows:
        identifier = chunk_id(
            row,
            "reviewed chunk concept",
        )
        concept = require_string(
            row.get("concept_id"),
            f"{identifier} concept_id",
        )
        assert concept in EXPECTED_CONCEPTS
        concepts_by_chunk[identifier].add(concept)

    assert set(concepts_by_chunk) == gold_ids
    for identifier in gold_ids:
        assert concepts_by_chunk[identifier] == EXPECTED_CONCEPTS

    domain_counts = Counter(
        require_string(
            record.get("domain"),
            "gold domain",
        )
        for record in gold
    )
    assert domain_counts == {
        "science": 90,
        "advaita": 120,
        "samkhya": 108,
    }

    activation_manifest = load_json(ACTIVATION_MANIFEST)
    assert activation_manifest.get("status") == "activation_artifacts_complete"
    assert activation_manifest.get("lifecycle_status") == "active"
    assert activation_manifest.get("corpus_version") == EXPECTED_CORPUS_VERSION

    counts = require_mapping(
        activation_manifest.get("counts"),
        "activation counts",
    )
    assert counts.get("active_chunk_count") == EXPECTED_APPROVED_CHUNKS
    assert counts.get("reviewed_chunk_concept_count") == EXPECTED_CONCEPT_ROWS

    gate = nested_exit_gate(
        activation_manifest,
        "activation manifest",
    )
    # Database schema mapping was explicitly carried as a known Phase 13
    # implementation limitation. It does not invalidate artifact-level lineage.
    assert_all_boolean_gate_values_true(
        gate,
        "activation manifest",
        ignored_keys={
            "database_schema_mapping_pending",
        },
    )


# ---------------------------------------------------------------------------
# Active corpus -> retrieval
# ---------------------------------------------------------------------------


def test_retrieval_evidence_resolves_only_to_active_chunks() -> None:
    bundles = load_jsonl(ACTIVE_BUNDLES)
    active_by_id = {
        chunk_id(
            record,
            "active bundle",
        ): record
        for record in bundles
    }

    evidence_package = load_json(EVIDENCE_PACKAGE)
    corpus_version = require_string(
        evidence_package.get("corpus_version"),
        "retrieval corpus_version",
    )
    assert corpus_version == EXPECTED_CORPUS_VERSION

    (
        evidence_index,
        chunk_domains,
    ) = build_evidence_index(
        evidence_package,
        corpus_version=corpus_version,
    )

    assert evidence_index
    assert chunk_domains

    for identifier, domain in chunk_domains.items():
        assert identifier in active_by_id, f"Retrieval returned inactive chunk {identifier!r}"
        assert domain in EXPECTED_DOMAINS

    retrieval_manifest = load_json(RETRIEVAL_MANIFEST)
    assert retrieval_manifest.get("status") == "evaluation_complete"
    assert retrieval_manifest.get("corpus_version") == EXPECTED_CORPUS_VERSION

    gate = nested_exit_gate(
        retrieval_manifest,
        "retrieval manifest",
    )
    for required in (
        "only_active_chunks_retrieved",
        "domain_separation_enforced",
        "concept_aware_retained",
    ):
        assert gate.get(required) is True


# ---------------------------------------------------------------------------
# Retrieval -> generation -> synthesis -> coverage
# ---------------------------------------------------------------------------


def test_runtime_artifacts_share_question_corpus_and_passed_exit_gates() -> None:
    evidence = load_json(EVIDENCE_PACKAGE)
    generation = load_json(GENERATION_MANIFEST)
    synthesis = load_json(SYNTHESIS_MANIFEST)
    coverage = load_json(COVERAGE_MANIFEST)
    final_manifest = load_json(FROZEN_FINAL_MANIFEST)

    question = require_string(
        evidence.get("question"),
        "Phase 14 question",
    )
    corpus_version = require_string(
        evidence.get("corpus_version"),
        "Phase 14 corpus_version",
    )

    for name, artifact in (
        ("Phase 15", generation),
        ("Phase 16", synthesis),
        ("Phase 17", coverage),
        ("Phase 18", final_manifest),
    ):
        assert (
            require_string(
                artifact.get("question"),
                f"{name} question",
            )
            == question
        )
        assert (
            require_string(
                artifact.get("corpus_version"),
                f"{name} corpus_version",
            )
            == corpus_version
        )

    assert generation.get("status") == ("domain_generation_complete")
    assert synthesis.get("status") == ("synthesis_complete")
    assert coverage.get("status") == ("coverage_classification_complete")

    for name, artifact in (
        ("Phase 15", generation),
        ("Phase 16", synthesis),
        ("Phase 17", coverage),
        ("Phase 18", final_manifest),
    ):
        gate = nested_exit_gate(
            artifact,
            name,
        )
        assert_all_boolean_gate_values_true(
            gate,
            name,
        )


def test_domain_generation_and_synthesis_are_structurally_grounded() -> None:
    evidence = load_json(EVIDENCE_PACKAGE)
    question = require_string(
        evidence.get("question"),
        "evidence question",
    )
    corpus_version = require_string(
        evidence.get("corpus_version"),
        "evidence corpus_version",
    )
    (
        query_activation,
        active_concepts,
        _weights,
    ) = parse_query_activation(evidence)
    evidence_index, _chunk_domains = build_evidence_index(
        evidence,
        corpus_version=corpus_version,
    )

    domain_responses_document = load_json(DOMAIN_RESPONSES)
    (
        generation_query_activation,
        domains,
    ) = parse_domain_responses(
        domain_responses_document,
        question=question,
        corpus_version=corpus_version,
        evidence_index=evidence_index,
    )

    assert generation_query_activation == query_activation
    assert set(domains) == EXPECTED_DOMAINS

    claim_index = build_claim_index(domains)
    assert len(claim_index) == 9

    for domain in EXPECTED_DOMAINS:
        response = require_mapping(
            domains.get(domain),
            f"{domain} canonical response",
        )
        claims = require_list(
            response.get("claims"),
            f"{domain} canonical claims",
        )
        assert len(claims) == 3

        assert (
            require_string(
                response.get("domain"),
                f"{domain} canonical response domain",
            )
            == domain
        )

        for claim_raw in claims:
            claim = require_mapping(
                claim_raw,
                f"{domain} canonical claim",
            )

            citations = require_list(
                claim.get("citations"),
                f"{domain} canonical citations",
            )
            assert citations

            synthesis_document = load_json(SYNTHESIS)

        raw_validation = require_mapping(
            synthesis_document.get("validation"),
            "synthesis validation",
        )

        assert raw_validation.get("passed") is True

        canonical_synthesis = parse_synthesis(
            synthesis_document,
            question=question,
            corpus_version=corpus_version,
            claim_index=claim_index,
            evidence_index=evidence_index,
        )

    comparisons = require_list(
        canonical_synthesis.get("comparisons"),
        "canonical synthesis comparisons",
    )
    assert len(comparisons) == 9

    assert set(active_concepts) == {
        "consciousness",
        "self_identity",
        "reality_appearance",
    }


# ---------------------------------------------------------------------------
# Coverage -> final response
# ---------------------------------------------------------------------------


def test_coverage_and_final_response_agree() -> None:
    coverage = load_json(COVERAGE)
    final_response = load_json(FROZEN_FINAL_JSON)

    sections = require_mapping(
        final_response.get("sections"),
        "final sections",
    )
    final_coverage = require_mapping(
        sections.get("coverage"),
        "final coverage section",
    )

    assert final_coverage.get("coverage_status") == coverage.get("coverage_status")
    assert final_coverage.get("coverage_score") == coverage.get("coverage_score")

    assert (
        final_response.get("corpus_version")
        == coverage.get("corpus_version")
        == EXPECTED_CORPUS_VERSION
    )

    validation = require_mapping(
        final_response.get("validation"),
        "final response validation",
    )
    assert validation.get("passed") is True

    citations = require_list(
        final_response.get("claim_level_citations"),
        "final claim-level citations",
    )
    assert len(citations) == 8


# ---------------------------------------------------------------------------
# Full deterministic Phase 14 -> Phase 18 assembly
# ---------------------------------------------------------------------------


def test_phase14_to_phase18_reassembles_reproducibly(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "phase18-reassembled"

    manifest = run_phase18(
        project_root=PROJECT_ROOT,
        evidence_package_path=EVIDENCE_PACKAGE,
        retrieval_manifest_path=RETRIEVAL_MANIFEST,
        domain_responses_path=DOMAIN_RESPONSES,
        generation_manifest_path=GENERATION_MANIFEST,
        synthesis_path=SYNTHESIS,
        synthesis_manifest_path=SYNTHESIS_MANIFEST,
        coverage_path=COVERAGE,
        coverage_manifest_path=COVERAGE_MANIFEST,
        output_directory=output_directory,
        replace=True,
    )

    gate = nested_exit_gate(
        manifest,
        "reassembled Phase 18",
    )
    assert_all_boolean_gate_values_true(
        gate,
        "reassembled Phase 18",
    )

    actual_json = load_json(output_directory / "final_response.json")
    expected_json = load_json(FROZEN_FINAL_JSON)

    assert comparable_final_response(actual_json) == comparable_final_response(expected_json)

    actual_markdown = require_file(output_directory / "final_response.md").read_text(
        encoding="utf-8"
    )
    expected_markdown = require_file(FROZEN_FINAL_MARKDOWN).read_text(encoding="utf-8")

    assert actual_markdown == expected_markdown


def test_phase19_vertical_slice_artifact_chain_is_complete() -> None:
    required_artifacts = (
        SELECTION_CANDIDATES,
        GOLD_CORPUS,
        REVIEW_MANIFEST,
        BUILD_SPLIT,
        DEVELOPMENT_SPLIT,
        HELDOUT_SPLIT,
        SPLIT_MANIFEST,
        APPROVED_EMBEDDINGS,
        EMBEDDING_MANIFEST,
        ACTIVE_BUNDLES,
        ACTIVE_CONCEPTS,
        ACTIVATION_MANIFEST,
        EVIDENCE_PACKAGE,
        RETRIEVAL_MANIFEST,
        DOMAIN_RESPONSES,
        GENERATION_MANIFEST,
        SYNTHESIS,
        SYNTHESIS_MANIFEST,
        COVERAGE,
        COVERAGE_MANIFEST,
        FROZEN_FINAL_JSON,
        FROZEN_FINAL_MARKDOWN,
        FROZEN_FINAL_MANIFEST,
    )

    missing = [path for path in required_artifacts if not path.is_file()]

    assert not missing, "Phase 19 integration chain is incomplete: " + ", ".join(
        path.as_posix() for path in missing
    )
