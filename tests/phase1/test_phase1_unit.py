from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import pytest
from scripts.assemble_phase1_final_response import (
    AssemblyError,
    parse_claim_citations,
)
from scripts.classify_phase1_coverage import (
    ConceptCoverage,
    ConceptStatus,
    classify_overall,
    coverage_score_components,
)
from scripts.freeze_phase1_evaluation_sets import (
    SPLIT_RATIOS,
    UnionFind,
    calculate_feature_targets,
)
from scripts.select_phase1_vertical_slice import (
    ConceptRule,
    HardNegativeRule,
    ScopeConfiguration,
    SourceRule,
    TermMatcher,
    evaluate_concepts,
    evaluate_hard_negatives,
    parse_scope_configuration,
    read_structure_report,
    stable_tie_breaker,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PHASE1_SCOPE_PATH = PROJECT_ROOT / "docs" / "corpus" / "phase1_section_scope.yaml"

EMBEDDING_FILES = (
    PROJECT_ROOT / "artifacts" / "phase1" / "embeddings" / "approved_chunk_embeddings.jsonl",
    PROJECT_ROOT / "artifacts" / "phase1" / "embeddings" / "query_prototype_embeddings.jsonl",
    PROJECT_ROOT / "artifacts" / "phase1" / "embeddings" / "passage_prototype_embeddings.jsonl",
)


class LiteralMatcher:
    """Minimal deterministic matcher implementing the production matcher protocol."""

    def matched_terms(
        self,
        normalized_text: str,
        terms: Iterable[str],
    ) -> tuple[str, ...]:
        text = normalized_text.casefold()
        return tuple(term for term in terms if term.casefold() in text)


def minimal_source_rule(
    *,
    hard_negative_targets: tuple[str, ...] = (),
) -> SourceRule:
    return SourceRule(
        source_id="test_source",
        source_title="Test Source",
        domain="science",
        scope_status="approved",
        structural_strategy="test",
        structure_assessment="test",
        preprocessing_requirements=(),
        include_rules=(),
        exclude_rules=(),
        hard_negative_targets=hard_negative_targets,
        maximum_pre_review_chunks=100,
        maximum_chunks_per_section=100,
    )


def minimal_scope(
    *,
    concept_rules: dict[str, ConceptRule],
    hard_negative_rules: dict[str, HardNegativeRule] | None = None,
) -> ScopeConfiguration:
    return ScopeConfiguration(
        scope_version="phase1-test-scope-v1",
        corpus_version="phase1_candidate_corpus_v1",
        status="approved",
        concepts=tuple(concept_rules),
        domains=("science", "advaita", "samkhya"),
        target_minimum=1,
        target_maximum=100,
        domain_weights={
            "science": 1.0,
            "advaita": 1.0,
            "samkhya": 1.0,
        },
        domain_minimums={
            "science": 1,
            "advaita": 1,
            "samkhya": 1,
        },
        domain_maximums={
            "science": 100,
            "advaita": 100,
            "samkhya": 100,
        },
        hard_negative_minimum=1,
        hard_negative_target=1,
        hard_negative_maximum=10,
        hard_negative_total_cap=10,
        hard_negative_category_caps={},
        minimum_alphabetic_ratio=0.20,
        maximum_ocr_noise_score=0.20,
        concept_rules=concept_rules,
        source_rules={
            "test_source": minimal_source_rule(),
        },
        hard_negative_rules=hard_negative_rules or {},
        global_exclusion_rules=(),
        methodological_constraints={
            "selection_must_be_embedding_independent": True,
            "prohibit_anchor_similarity_selection": True,
            "prohibit_chunk_concept_weight_selection": True,
            "prohibit_llm_labels_as_final_authority": True,
            "require_human_review_before_activation": True,
        },
    )


def make_concept_coverage(
    *,
    concept: str,
    score: float,
    status: str,
    activation_weight: float = 1.0,
) -> ConceptCoverage:
    return ConceptCoverage(
        concept=concept,
        activation_weight=activation_weight,
        evidence_count=9,
        covered_domains=(
            "science",
            "advaita",
            "samkhya",
        ),
        claim_domains=(
            "science",
            "advaita",
            "samkhya",
        ),
        citation_quality=1.0,
        retrieval_confidence=0.90,
        retrieval_confidence_source="test",
        explicit_unsupported=False,
        insufficient_comparison=status == "Partially Supported",
        coverage_score=score,
        score_components={},
        hard_overrides=(),
        status=cast(ConceptStatus, status),
        reasons=(),
    )


def iter_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} must contain a JSON object")
            yield value


# ---------------------------------------------------------------------------
# 1. Scope-rule parsing
# ---------------------------------------------------------------------------


def test_phase1_scope_configuration_is_approved_and_frozen() -> None:
    scope = parse_scope_configuration(PHASE1_SCOPE_PATH)

    assert scope.status == "approved"
    assert set(scope.concepts) == {
        "consciousness",
        "self_identity",
        "reality_appearance",
    }
    assert set(scope.domains) == {
        "science",
        "advaita",
        "samkhya",
    }

    required_constraints = (
        "selection_must_be_embedding_independent",
        "prohibit_anchor_similarity_selection",
        "prohibit_chunk_concept_weight_selection",
        "prohibit_llm_labels_as_final_authority",
        "require_human_review_before_activation",
    )
    for key in required_constraints:
        assert scope.methodological_constraints[key] is True


# ---------------------------------------------------------------------------
# 2. Section matching
# ---------------------------------------------------------------------------


def test_structure_report_matches_source_and_section_and_avoids_ambiguous_ids(
    tmp_path: Path,
) -> None:
    report = tmp_path / "structure.csv"

    fieldnames = (
        "source_id",
        "section_id",
        "proposed_structure_action",
        "section_title",
        "parent_section",
        "structural_locator",
        "unit_type",
        "parser_warning_count",
        "ocr_noise_score",
    )
    rows = (
        {
            "source_id": "source_a",
            "section_id": "shared",
            "proposed_structure_action": "include_candidate",
            "section_title": "A",
            "parent_section": "",
            "structural_locator": "A:1",
            "unit_type": "paragraph",
            "parser_warning_count": "0",
            "ocr_noise_score": "0.01",
        },
        {
            "source_id": "source_b",
            "section_id": "shared",
            "proposed_structure_action": "include_candidate",
            "section_title": "B",
            "parent_section": "",
            "structural_locator": "B:1",
            "unit_type": "paragraph",
            "parser_warning_count": "0",
            "ocr_noise_score": "0.02",
        },
        {
            "source_id": "source_a",
            "section_id": "unique",
            "proposed_structure_action": "manual_section_review",
            "section_title": "Unique",
            "parent_section": "",
            "structural_locator": "A:2",
            "unit_type": "paragraph",
            "parser_warning_count": "1",
            "ocr_noise_score": "0.03",
        },
    )

    with report.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    by_source_section, unique_by_section = read_structure_report(report)

    assert by_source_section[("source_a", "shared")].section_title == "A"
    assert by_source_section[("source_b", "shared")].section_title == "B"
    assert "shared" not in unique_by_section
    assert unique_by_section["unique"].source_id == "source_a"
    assert unique_by_section["unique"].parser_warning_count == 1


# ---------------------------------------------------------------------------
# 3. Lexical selection
# 4. Concept scoring
# ---------------------------------------------------------------------------


def test_lexical_selection_activates_concept_with_substantive_context() -> None:
    concept_rule = ConceptRule(
        concept_id="consciousness",
        positive_terms=("consciousness",),
        contextual_terms=("experience",),
        exclusion_terms=("unconscious",),
        minimum_positive_term_matches=1,
        require_substantive_context=True,
    )
    scope = minimal_scope(
        concept_rules={
            "consciousness": concept_rule,
        }
    )
    matcher = cast(TermMatcher, LiteralMatcher())

    (
        active_concepts,
        scores,
        positive_hits,
        context_hits,
        exclusion_hits,
    ) = evaluate_concepts(
        "consciousness shapes experience",
        "consciousness shapes experience",
        scope,
        matcher,
        {},
    )

    assert active_concepts == ("consciousness",)
    assert positive_hits["consciousness"] == ("consciousness",)
    assert context_hits["consciousness"] == ("experience",)
    assert exclusion_hits["consciousness"] == ()
    assert scores["consciousness"] == pytest.approx(3.75)


def test_concept_scoring_applies_exclusion_penalty() -> None:
    concept_rule = ConceptRule(
        concept_id="consciousness",
        positive_terms=("consciousness",),
        contextual_terms=("experience",),
        exclusion_terms=("unconscious",),
        minimum_positive_term_matches=1,
        require_substantive_context=False,
    )
    scope = minimal_scope(
        concept_rules={
            "consciousness": concept_rule,
        }
    )
    matcher = cast(TermMatcher, LiteralMatcher())

    active_concepts, scores, *_ = evaluate_concepts(
        "consciousness experience unconscious",
        "consciousness experience unconscious",
        scope,
        matcher,
        {},
    )

    # 3.0 positive + 0.75 contextual - 2.0 exclusion = 1.75.
    assert scores["consciousness"] == pytest.approx(1.75)
    assert active_concepts == ("consciousness",)


# ---------------------------------------------------------------------------
# 5. Hard-negative selection
# ---------------------------------------------------------------------------


def test_hard_negative_selection_requires_target_not_already_active() -> None:
    concept_rule = ConceptRule(
        concept_id="consciousness",
        positive_terms=("awareness",),
        contextual_terms=(),
        exclusion_terms=(),
        minimum_positive_term_matches=1,
        require_substantive_context=False,
    )
    hard_negative = HardNegativeRule(
        category="consciousness_vs_cognition",
        target_concept="consciousness",
        positive_terms=("cognition",),
        exclusion_terms=("awareness",),
        required_domain_contrast=("science",),
    )
    scope = minimal_scope(
        concept_rules={
            "consciousness": concept_rule,
        },
        hard_negative_rules={
            "consciousness_vs_cognition": hard_negative,
        },
    )
    matcher = cast(TermMatcher, LiteralMatcher())
    source_rule = minimal_source_rule(hard_negative_targets=("consciousness_vs_cognition",))

    categories, targets = evaluate_hard_negatives(
        "cognition without subjective experience",
        "science",
        scope,
        matcher,
        (),
        source_rule,
    )

    assert categories == ("consciousness_vs_cognition",)
    assert targets == ("consciousness",)

    categories_when_active, targets_when_active = evaluate_hard_negatives(
        "cognition without subjective experience",
        "science",
        scope,
        matcher,
        ("consciousness",),
        source_rule,
    )

    assert categories_when_active == ()
    assert targets_when_active == ()


# ---------------------------------------------------------------------------
# 6. Deterministic sampling
# ---------------------------------------------------------------------------


def test_stable_tie_breaker_is_deterministic() -> None:
    first = stable_tie_breaker("chunk-123")
    second = stable_tie_breaker("chunk-123")
    other = stable_tie_breaker("chunk-124")

    assert first == second
    assert first != other
    assert len(first) == 64


def test_split_targets_are_deterministic_for_phase1_gold_count() -> None:
    first = calculate_feature_targets(
        318,
        ratios=SPLIT_RATIOS,
    )
    second = calculate_feature_targets(
        318,
        ratios=SPLIT_RATIOS,
    )

    assert first == second
    assert first == {
        "build": 159,
        "development": 80,
        "heldout": 79,
    }


# ---------------------------------------------------------------------------
# 7. Split leakage prevention
# ---------------------------------------------------------------------------


def test_union_find_keeps_overlapping_passages_in_one_family() -> None:
    families = UnionFind.create(4)

    families.union(0, 1)
    families.union(1, 2)

    assert families.find(0) == families.find(1)
    assert families.find(1) == families.find(2)
    assert families.find(3) != families.find(0)


# ---------------------------------------------------------------------------
# 8. Embedding-cache identity / key contract
# ---------------------------------------------------------------------------


def test_embedding_cache_identity_is_complete_and_collision_free() -> None:
    records: list[dict[str, object]] = []

    for path in EMBEDDING_FILES:
        assert path.is_file(), f"Missing frozen Phase 9 embedding artifact: {path}"
        records.extend(iter_jsonl(path))

    assert records, "No Phase 9 embedding records found"

    required_identity_fields = (
        "provider",
        "model",
        "model_revision",
        "dimensions",
        "task_type",
        "normalization",
        "text_checksum",
        "embedding_checksum",
    )

    cache_key_to_embedding_checksum: dict[
        tuple[object, ...],
        object,
    ] = {}

    for record in records:
        for field_name in required_identity_fields:
            assert field_name in record, (
                f"Embedding record is missing cache identity field {field_name!r}"
            )

        cache_key = (
            record["provider"],
            record["model"],
            record["model_revision"],
            record["dimensions"],
            record["task_type"],
            record["normalization"],
            record["text_checksum"],
        )
        embedding_checksum = record["embedding_checksum"]

        existing = cache_key_to_embedding_checksum.get(cache_key)
        if existing is None:
            cache_key_to_embedding_checksum[cache_key] = embedding_checksum
        else:
            assert existing == embedding_checksum, (
                "Same embedding cache identity resolves to different vectors"
            )


# ---------------------------------------------------------------------------
# 9. Coverage classification
# ---------------------------------------------------------------------------


def test_coverage_score_components_reach_100_for_full_support() -> None:
    components = coverage_score_components(
        activation_weight=1.0,
        evidence_count=9,
        covered_domain_count=3,
        citation_quality=1.0,
        retrieval_confidence_value=1.0,
        explicit_unsupported=False,
        insufficient_comparison=False,
    )

    assert sum(components.values()) == pytest.approx(100.0)


def test_overall_coverage_is_partial_when_one_major_concept_is_partial() -> None:
    concepts = (
        make_concept_coverage(
            concept="reality_appearance",
            score=88.21,
            status="Supported",
            activation_weight=0.8068,
        ),
        make_concept_coverage(
            concept="self_identity",
            score=87.25,
            status="Supported",
            activation_weight=0.7749,
        ),
        make_concept_coverage(
            concept="consciousness",
            score=74.54,
            status="Partially Supported",
            activation_weight=0.4846,
        ),
    )

    status, score, overrides, reason = classify_overall(
        query_unsupported=False,
        active_concepts=(
            "reality_appearance",
            "self_identity",
            "consciousness",
        ),
        concept_results=concepts,
        covered_domains=(
            "science",
            "advaita",
            "samkhya",
        ),
    )

    assert status == "Partially Supported"
    assert score >= 70.0
    assert "one_or_more_active_concepts_not_fully_supported" in overrides
    assert "Coverage score" in reason


def test_query_explicitly_marked_unsupported_is_out_of_corpus() -> None:
    status, score, overrides, _reason = classify_overall(
        query_unsupported=True,
        active_concepts=("consciousness",),
        concept_results=(
            make_concept_coverage(
                concept="consciousness",
                score=90.0,
                status="Supported",
            ),
        ),
        covered_domains=(
            "science",
            "advaita",
            "samkhya",
        ),
    )

    assert status == "Out of Corpus"
    assert score == 0.0
    assert "query_outside_phase1_concepts" in overrides


# ---------------------------------------------------------------------------
# 10. Citation validation
# ---------------------------------------------------------------------------


def test_claim_citation_resolves_to_canonical_active_evidence() -> None:
    corpus_version = "phase1_active_corpus_v1"
    evidence_index = {
        ("chunk-1", "source-1"): {
            "chunk_id": "chunk-1",
            "source_id": "source-1",
            "citation": "Test Author. Test Source. 2026.",
            "corpus_version": corpus_version,
            "domain": "science",
        }
    }
    claim = {
        "citations": [
            {
                "chunk_id": "chunk-1",
                "source_id": "source-1",
                "citation": "Test Author. Test Source. 2026.",
                "corpus_version": corpus_version,
            }
        ]
    }

    citations = parse_claim_citations(
        claim=claim,
        domain="science",
        claim_id="C1",
        corpus_version=corpus_version,
        evidence_index=evidence_index,
    )

    assert len(citations) == 1
    assert citations[0]["chunk_id"] == "chunk-1"
    assert citations[0]["domain"] == "science"


def test_claim_citation_rejects_unknown_chunk() -> None:
    corpus_version = "phase1_active_corpus_v1"
    evidence_index = {
        ("chunk-1", "source-1"): {
            "chunk_id": "chunk-1",
            "source_id": "source-1",
            "citation": "Test Author. Test Source. 2026.",
            "corpus_version": corpus_version,
            "domain": "science",
        }
    }
    claim = {
        "citations": [
            {
                "chunk_id": "missing-chunk",
                "source_id": "source-1",
                "citation": "Test Author. Test Source. 2026.",
                "corpus_version": corpus_version,
            }
        ]
    }

    with pytest.raises(
        AssemblyError,
        match="does not resolve",
    ):
        parse_claim_citations(
            claim=claim,
            domain="science",
            claim_id="C1",
            corpus_version=corpus_version,
            evidence_index=evidence_index,
        )


def test_claim_citation_rejects_domain_leakage() -> None:
    corpus_version = "phase1_active_corpus_v1"
    evidence_index = {
        ("chunk-1", "source-1"): {
            "chunk_id": "chunk-1",
            "source_id": "source-1",
            "citation": "Test Author. Test Source. 2026.",
            "corpus_version": corpus_version,
            "domain": "advaita",
        }
    }
    claim = {
        "citations": [
            {
                "chunk_id": "chunk-1",
                "source_id": "source-1",
                "citation": "Test Author. Test Source. 2026.",
                "corpus_version": corpus_version,
            }
        ]
    }

    with pytest.raises(
        AssemblyError,
        match="belongs to",
    ):
        parse_claim_citations(
            claim=claim,
            domain="science",
            claim_id="C1",
            corpus_version=corpus_version,
            evidence_index=evidence_index,
        )
