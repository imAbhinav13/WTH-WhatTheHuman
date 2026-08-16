from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from apps.api.models.runtime_contracts import (
    DomainResponses,
    EvidencePackage,
    GenerationManifest,
    RetrievalManifest,
    SynthesisManifest,
    SynthesisResult,
)
from apps.api.services.coverage import (
    CoverageService,
    CoverageServiceResult,
)

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
PHASE1_ARTIFACTS: Final = PROJECT_ROOT / "artifacts" / "phase1"

RETRIEVAL_DIR: Final = PHASE1_ARTIFACTS / "retrieval"
GENERATION_DIR: Final = PHASE1_ARTIFACTS / "generation"
SYNTHESIS_DIR: Final = PHASE1_ARTIFACTS / "synthesis"
COVERAGE_DIR: Final = PHASE1_ARTIFACTS / "coverage"


def load_json_object(
    path: Path,
) -> dict[str, object]:
    """Load one required frozen JSON object."""

    assert path.is_file(), f"Required frozen artifact is missing: {path}"

    raw: object = json.loads(path.read_text(encoding="utf-8"))

    assert isinstance(raw, Mapping), f"Expected JSON object: {path}"

    result: dict[str, object] = {}

    for key, value in raw.items():
        assert isinstance(key, str)
        result[key] = value

    return result


def load_runtime_inputs() -> tuple[
    EvidencePackage,
    RetrievalManifest,
    DomainResponses,
    GenerationManifest,
    SynthesisResult,
    SynthesisManifest,
]:
    """Load frozen Phase 14-16 inputs through Stage 3.0 contracts."""

    evidence_package = EvidencePackage.model_validate(
        load_json_object(RETRIEVAL_DIR / "evidence_package.json")
    )

    retrieval_manifest = RetrievalManifest.model_validate(
        load_json_object(RETRIEVAL_DIR / "retrieval_manifest.json")
    )

    domain_responses = DomainResponses.model_validate(
        load_json_object(GENERATION_DIR / "domain_responses.json")
    )

    generation_manifest = GenerationManifest.model_validate(
        load_json_object(GENERATION_DIR / "generation_manifest.json")
    )

    synthesis = SynthesisResult.model_validate(load_json_object(SYNTHESIS_DIR / "synthesis.json"))

    synthesis_manifest = SynthesisManifest.model_validate(
        load_json_object(SYNTHESIS_DIR / "synthesis_manifest.json")
    )

    return (
        evidence_package,
        retrieval_manifest,
        domain_responses,
        generation_manifest,
        synthesis,
        synthesis_manifest,
    )


def frozen_coverage() -> dict[str, object]:
    return load_json_object(COVERAGE_DIR / "coverage.json")


def frozen_manifest() -> dict[str, object]:
    return load_json_object(COVERAGE_DIR / "coverage_manifest.json")

def run_runtime_coverage() -> CoverageServiceResult:
    """Execute Phase 17 entirely through CoverageService."""

    (
        evidence_package,
        retrieval_manifest,
        domain_responses,
        generation_manifest,
        synthesis,
        synthesis_manifest,
    ) = load_runtime_inputs()

    frozen = frozen_coverage()

    generated_at = frozen.get("generated_at")
    assert isinstance(generated_at, str)
    assert generated_at

    frozen_manifest_document = frozen_manifest()

    outputs = frozen_manifest_document.get("outputs")
    assert isinstance(outputs, dict)

    coverage_output_path = outputs.get("coverage")
    assert isinstance(coverage_output_path, str)
    assert coverage_output_path

    return CoverageService().classify(
        evidence_package=evidence_package,
        retrieval_manifest=retrieval_manifest,
        domain_responses=domain_responses,
        generation_manifest=generation_manifest,
        synthesis=synthesis,
        synthesis_manifest=synthesis_manifest,
        generated_at=generated_at,
        coverage_output_path=coverage_output_path,
    )


def test_runtime_phase17_matches_frozen_coverage_exactly() -> None:
    """Runtime Phase 17 must reproduce coverage.json exactly."""

    result = run_runtime_coverage()

    actual = result.coverage.model_dump(
        mode="python",
        by_alias=True,
    )

    expected = frozen_coverage()

    assert actual == expected


def test_runtime_phase17_matches_frozen_manifest_exactly() -> None:
    """Runtime Phase 17 must reproduce coverage_manifest.json exactly."""

    result = run_runtime_coverage()

    actual = result.manifest.model_dump(
        mode="python",
        by_alias=True,
    )

    expected = frozen_manifest()

    assert actual == expected


def test_runtime_phase17_preserves_golden_status_and_score() -> None:
    """Golden Phase 17 classification must remain unchanged."""

    result = run_runtime_coverage()

    coverage = result.coverage

    assert coverage.coverage_status == ("Partially Supported")

    assert coverage.coverage_score == 84.64


def test_runtime_phase17_preserves_concept_classification() -> None:
    """Golden concept-level classifications must remain unchanged."""

    result = run_runtime_coverage()

    coverage = result.coverage

    assert coverage.supported_concepts == [
        "reality_appearance",
        "self_identity",   
    ]

    assert coverage.partially_supported_concepts == [
        "consciousness",
    ]

    assert coverage.unsupported_concepts == []


def test_runtime_phase17_preserves_per_concept_scores() -> None:
    """Per-concept scoring and classifications must remain unchanged."""

    result = run_runtime_coverage()

    actual = {
        item.concept: item.model_dump(
            mode="python",
            by_alias=True,
        )
        for item in result.coverage.concept_coverage
    }

    expected_document = frozen_coverage()

    expected_raw = expected_document.get("concept_coverage")

    assert isinstance(expected_raw, list)

    expected: dict[str, object] = {}

    for item_raw in expected_raw:
        assert isinstance(item_raw, dict)

        concept = item_raw.get("concept")

        assert isinstance(concept, str)

        expected[concept] = item_raw

    assert actual == expected


def test_runtime_phase17_preserves_hard_overrides() -> None:
    """Hard-override behavior must remain identical."""

    result = run_runtime_coverage()

    expected = frozen_coverage()

    expected_overrides = expected.get("hard_overrides")

    assert isinstance(
        expected_overrides,
        list,
    )

    assert result.coverage.hard_overrides == expected_overrides

    expected_concepts = expected.get("concept_coverage")

    assert isinstance(
        expected_concepts,
        list,
    )

    expected_by_concept: dict[
        str,
        list[object],
    ] = {}

    for item in expected_concepts:
        assert isinstance(item, dict)

        concept = item.get("concept")
        overrides = item.get("hard_overrides")

        assert isinstance(concept, str)
        assert isinstance(
            overrides,
            list,
        )

        expected_by_concept[concept] = overrides

    for concept_result in result.coverage.concept_coverage:
        assert concept_result.hard_overrides == expected_by_concept[concept_result.concept]


def test_runtime_phase17_preserves_retrieval_confidence() -> None:
    """Retrieval confidence values and source must remain unchanged."""

    result = run_runtime_coverage()
    expected = frozen_coverage()

    expected_concepts = expected.get("concept_coverage")

    assert isinstance(
        expected_concepts,
        list,
    )

    expected_by_concept: dict[
        str,
        tuple[object, object],
    ] = {}

    for item in expected_concepts:
        assert isinstance(item, dict)

        concept = item.get("concept")

        assert isinstance(concept, str)

        expected_by_concept[concept] = (
            item.get("retrieval_confidence"),
            item.get("retrieval_confidence_source"),
        )

    for item in result.coverage.concept_coverage:
        expected_confidence, expected_source = expected_by_concept[item.concept]

        assert item.retrieval_confidence == expected_confidence

        assert item.retrieval_confidence_source == expected_source


def test_runtime_phase17_preserves_citation_quality() -> None:
    """Citation quality calculation must remain unchanged."""

    result = run_runtime_coverage()
    expected = frozen_coverage()

    expected_concepts = expected.get("concept_coverage")

    assert isinstance(
        expected_concepts,
        list,
    )

    expected_quality: dict[
        str,
        object,
    ] = {}

    for item in expected_concepts:
        assert isinstance(item, dict)

        concept = item.get("concept")

        assert isinstance(concept, str)

        expected_quality[concept] = item.get("citation_quality")

    for item in result.coverage.concept_coverage:
        assert item.citation_quality == expected_quality[item.concept]


def test_runtime_phase17_preserves_domain_coverage() -> None:
    """Domain coverage and missing-domain classification must stay frozen."""

    result = run_runtime_coverage()

    expected = frozen_coverage()

    assert result.coverage.covered_domains == expected["covered_domains"]

    assert result.coverage.missing_domains == expected["missing_domains"]


def test_runtime_phase17_preserves_response_policy() -> None:
    """Fallback and corpus-answer policy must remain unchanged."""

    result = run_runtime_coverage()

    expected = frozen_coverage()

    expected_policy = expected.get("response_policy")

    assert isinstance(
        expected_policy,
        dict,
    )

    actual_policy = result.coverage.response_policy.model_dump(
        mode="python",
        by_alias=True,
    )

    assert actual_policy == expected_policy


def test_runtime_phase17_preserves_knowledge_boundary() -> None:
    """Reviewed corpus and general-knowledge boundaries must stay unchanged."""

    result = run_runtime_coverage()

    expected = frozen_coverage()

    expected_boundary = expected.get("boundary")

    assert isinstance(
        expected_boundary,
        dict,
    )

    actual_boundary = result.coverage.boundary.model_dump(
        mode="python",
        by_alias=True,
    )

    assert actual_boundary == expected_boundary


def test_runtime_phase17_preserves_exit_gate() -> None:
    """The deterministic Phase 17 safety exit gate must remain green."""

    result = run_runtime_coverage()

    gate = result.coverage.exit_gate

    assert gate.passed is True

    assert gate.no_corpus_fabrication_when_out_of_corpus is True

    assert gate.partial_answers_limited_to_supported_components is True

    assert gate.general_knowledge_boundary_explicit is True

    assert gate.corpus_citations_forbidden_for_general_fallback is True


def test_runtime_phase17_uses_zero_llm_calls() -> None:
    """Phase 17 must remain fully deterministic."""

    result = run_runtime_coverage()

    assert result.manifest.calculation_policy.llm_calls == 0

    assert result.manifest.calculation_policy.deterministic is True


def test_runtime_phase17_is_deterministic_for_same_inputs() -> None:
    """Same inputs and timestamp must produce identical outputs."""

    first = run_runtime_coverage()
    second = run_runtime_coverage()

    assert first.coverage.model_dump(
        mode="python",
        by_alias=True,
    ) == second.coverage.model_dump(
        mode="python",
        by_alias=True,
    )

    assert first.manifest.model_dump(
        mode="python",
        by_alias=True,
    ) == second.manifest.model_dump(
        mode="python",
        by_alias=True,
    )
