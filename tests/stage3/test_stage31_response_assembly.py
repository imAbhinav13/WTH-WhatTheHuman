from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final, TypedDict, TypeVar

from apps.api.models.runtime_contracts import (
    CoverageManifest,
    CoverageResult,
    DomainResponses,
    EvidencePackage,
    FinalResponse,
    GenerationManifest,
    RetrievalManifest,
    SynthesisManifest,
    SynthesisResult,
)
from apps.api.services.response_assembly import (
    ResponseAssemblyResult,
    ResponseAssemblyService,
)


PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
PHASE1_ARTIFACTS: Final = PROJECT_ROOT / "artifacts" / "phase1"

RETRIEVAL_DIR: Final = PHASE1_ARTIFACTS / "retrieval"
GENERATION_DIR: Final = PHASE1_ARTIFACTS / "generation"
SYNTHESIS_DIR: Final = PHASE1_ARTIFACTS / "synthesis"
COVERAGE_DIR: Final = PHASE1_ARTIFACTS / "coverage"
FINAL_DIR: Final = PHASE1_ARTIFACTS / "final"


ContractT = TypeVar(
    "ContractT",
    EvidencePackage,
    RetrievalManifest,
    DomainResponses,
    GenerationManifest,
    SynthesisResult,
    SynthesisManifest,
    CoverageResult,
    CoverageManifest,
)


class RuntimeInputs(TypedDict):
    evidence_package: EvidencePackage
    retrieval_manifest: RetrievalManifest
    domain_responses: DomainResponses
    generation_manifest: GenerationManifest
    synthesis: SynthesisResult
    synthesis_manifest: SynthesisManifest
    coverage: CoverageResult
    coverage_manifest: CoverageManifest


def load_json_object(path: Path) -> dict[str, object]:
    """Load one required frozen JSON object."""

    assert path.is_file(), f"Required frozen artifact is missing: {path}"

    raw: object = json.loads(path.read_text(encoding="utf-8"))

    assert isinstance(raw, Mapping), f"Expected JSON object in frozen artifact: {path}"

    result: dict[str, object] = {}

    for key, value in raw.items():
        assert isinstance(key, str)
        result[key] = value

    return result


def load_contract(
    path: Path,
    model_type: type[ContractT],
) -> ContractT:
    """Load a frozen artifact through its Stage 3.0 runtime contract."""

    return model_type.model_validate(load_json_object(path))


def build_runtime_inputs() -> RuntimeInputs:
    """Load all Phase 14-17 inputs required by Phase 18."""

    return {
        "evidence_package": load_contract(
            RETRIEVAL_DIR / "evidence_package.json",
            EvidencePackage,
        ),
        "retrieval_manifest": load_contract(
            RETRIEVAL_DIR / "retrieval_manifest.json",
            RetrievalManifest,
        ),
        "domain_responses": load_contract(
            GENERATION_DIR / "domain_responses.json",
            DomainResponses,
        ),
        "generation_manifest": load_contract(
            GENERATION_DIR / "generation_manifest.json",
            GenerationManifest,
        ),
        "synthesis": load_contract(
            SYNTHESIS_DIR / "synthesis.json",
            SynthesisResult,
        ),
        "synthesis_manifest": load_contract(
            SYNTHESIS_DIR / "synthesis_manifest.json",
            SynthesisManifest,
        ),
        "coverage": load_contract(
            COVERAGE_DIR / "coverage.json",
            CoverageResult,
        ),
        "coverage_manifest": load_contract(
            COVERAGE_DIR / "coverage_manifest.json",
            CoverageManifest,
        ),
    }


def frozen_final_response() -> dict[str, object]:
    """Return the authoritative frozen Phase 18 JSON response."""

    return load_json_object(FINAL_DIR / "final_response.json")


def run_runtime_assembly() -> ResponseAssemblyResult:
    """Execute Phase 18 entirely through the runtime service."""

    inputs = build_runtime_inputs()

    frozen_response = frozen_final_response()

    generated_at = frozen_response.get("generated_at")
    assert isinstance(generated_at, str)
    assert generated_at

    service = ResponseAssemblyService()

    return service.assemble(
        evidence_package=inputs["evidence_package"],
        retrieval_manifest=inputs["retrieval_manifest"],
        domain_responses=inputs["domain_responses"],
        generation_manifest=inputs["generation_manifest"],
        synthesis=inputs["synthesis"],
        synthesis_manifest=inputs["synthesis_manifest"],
        coverage=inputs["coverage"],
        coverage_manifest=inputs["coverage_manifest"],
        generated_at=generated_at,
    )


def test_runtime_phase18_matches_frozen_final_response_exactly() -> None:
    """Runtime Phase 18 must reproduce the frozen JSON response exactly."""

    result = run_runtime_assembly()

    actual = result.response.model_dump(
        mode="python",
        by_alias=True,
    )

    expected = frozen_final_response()

    assert actual == expected


def test_runtime_phase18_matches_frozen_markdown_exactly() -> None:
    """Runtime Phase 18 must reproduce the frozen Markdown exactly."""

    result = run_runtime_assembly()

    markdown_path = FINAL_DIR / "final_response.md"

    assert markdown_path.is_file(), f"Required frozen Markdown is missing: {markdown_path}"

    expected = markdown_path.read_text(encoding="utf-8")

    assert result.markdown == expected


def test_runtime_phase18_returns_frozen_contract() -> None:
    """Runtime assembly must return the canonical FinalResponse model."""

    result = run_runtime_assembly()

    assert isinstance(
        result.response,
        FinalResponse,
    )


def test_runtime_phase18_has_zero_provider_calls() -> None:
    """Phase 18 must remain deterministic and provider-free."""

    result = run_runtime_assembly()

    provider_calls = result.response.provider_calls

    assert provider_calls.phase18_llm_calls == 0
    assert provider_calls.phase18_embedding_calls == 0
    assert provider_calls.phase18_retrieval_calls == 0


def test_runtime_phase18_preserves_frozen_versions() -> None:
    """All upstream version provenance must survive runtime assembly."""

    result = run_runtime_assembly()
    expected = frozen_final_response()

    expected_versions = expected.get("versions")

    assert isinstance(expected_versions, dict)

    actual_versions = result.response.versions.model_dump(
        mode="python",
        by_alias=True,
    )

    assert actual_versions == expected_versions


def test_runtime_phase18_preserves_frozen_citation_registry() -> None:
    """Claim-level citation registry must remain exactly unchanged."""

    result = run_runtime_assembly()
    expected = frozen_final_response()

    expected_citations = expected.get("claim_level_citations")

    assert isinstance(expected_citations, list)

    actual_citations = [
        citation.model_dump(
            mode="python",
            by_alias=True,
        )
        for citation in result.response.claim_level_citations
    ]

    assert actual_citations == expected_citations


def test_runtime_phase18_preserves_coverage_classification() -> None:
    """Phase 17 coverage semantics must survive Phase 18 unchanged."""

    result = run_runtime_assembly()
    expected = frozen_final_response()

    sections = expected.get("sections")
    assert isinstance(sections, dict)

    expected_coverage = sections.get("coverage")
    assert isinstance(expected_coverage, dict)

    actual_coverage = result.response.sections.coverage.model_dump(
        mode="python",
        by_alias=True,
    )

    assert actual_coverage == expected_coverage


def test_runtime_phase18_is_deterministic_for_same_inputs() -> None:
    """Same inputs and timestamp must produce identical runtime outputs."""

    first = run_runtime_assembly()
    second = run_runtime_assembly()

    first_response = first.response.model_dump(
        mode="python",
        by_alias=True,
    )
    second_response = second.response.model_dump(
        mode="python",
        by_alias=True,
    )

    assert first_response == second_response
    assert first.markdown == second.markdown
