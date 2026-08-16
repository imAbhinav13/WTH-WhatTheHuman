from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pytest
from pydantic import ValidationError

from apps.api.models.runtime_contracts import (
    CoverageManifest,
    CoverageResult,
    DomainResponses,
    EvidencePackage,
    FinalResponse,
    FinalResponseManifest,
    FrozenRuntimeContract,
    GenerationManifest,
    RetrievalManifest,
    SynthesisManifest,
    SynthesisResult,
)

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
PHASE1_ARTIFACTS: Final = PROJECT_ROOT / "artifacts" / "phase1"

ArtifactCase = tuple[
    str,
    Path,
    type[FrozenRuntimeContract],
]

ARTIFACT_CASES: Final[tuple[ArtifactCase, ...]] = (
    (
        "phase14_evidence_package",
        PHASE1_ARTIFACTS / "retrieval" / "evidence_package.json",
        EvidencePackage,
    ),
    (
        "phase14_retrieval_manifest",
        PHASE1_ARTIFACTS / "retrieval" / "retrieval_manifest.json",
        RetrievalManifest,
    ),
    (
        "phase15_domain_responses",
        PHASE1_ARTIFACTS / "generation" / "domain_responses.json",
        DomainResponses,
    ),
    (
        "phase15_generation_manifest",
        PHASE1_ARTIFACTS / "generation" / "generation_manifest.json",
        GenerationManifest,
    ),
    (
        "phase16_synthesis",
        PHASE1_ARTIFACTS / "synthesis" / "synthesis.json",
        SynthesisResult,
    ),
    (
        "phase16_synthesis_manifest",
        PHASE1_ARTIFACTS / "synthesis" / "synthesis_manifest.json",
        SynthesisManifest,
    ),
    (
        "phase17_coverage",
        PHASE1_ARTIFACTS / "coverage" / "coverage.json",
        CoverageResult,
    ),
    (
        "phase17_coverage_manifest",
        PHASE1_ARTIFACTS / "coverage" / "coverage_manifest.json",
        CoverageManifest,
    ),
    (
        "phase18_final_response",
        PHASE1_ARTIFACTS / "final" / "final_response.json",
        FinalResponse,
    ),
    (
        "phase18_final_response_manifest",
        PHASE1_ARTIFACTS / "final" / "final_response_manifest.json",
        FinalResponseManifest,
    ),
)


def load_json_object(path: Path) -> dict[str, object]:
    """Load one required frozen Phase 14-18 JSON artifact."""

    assert path.is_file(), f"Required frozen artifact is missing: {path}"

    raw: object = json.loads(path.read_text(encoding="utf-8"))

    assert isinstance(raw, Mapping), f"Frozen artifact must contain a JSON object: {path}"

    result: dict[str, object] = {}

    for key, value in raw.items():
        assert isinstance(key, str), f"Frozen artifact contains a non-string key: {path}"
        result[key] = value

    return result


@pytest.mark.parametrize(
    ("artifact_name", "artifact_path", "model_type"),
    ARTIFACT_CASES,
    ids=[case[0] for case in ARTIFACT_CASES],
)
def test_frozen_artifact_python_round_trip_is_exact(
    artifact_name: str,
    artifact_path: Path,
    model_type: type[FrozenRuntimeContract],
) -> None:
    """Existing JSON -> model -> model_dump must preserve the artifact."""

    original = load_json_object(artifact_path)

    model = model_type.model_validate(original)

    round_tripped = model.model_dump(
        mode="python",
        by_alias=True,
    )

    assert round_tripped == original, (
        f"{artifact_name} changed during JSON -> model -> model_dump round trip"
    )


@pytest.mark.parametrize(
    ("artifact_name", "artifact_path", "model_type"),
    ARTIFACT_CASES,
    ids=[case[0] for case in ARTIFACT_CASES],
)
def test_frozen_artifact_json_round_trip_is_exact(
    artifact_name: str,
    artifact_path: Path,
    model_type: type[FrozenRuntimeContract],
) -> None:
    """JSON serialization through the runtime model must preserve semantics."""

    original = load_json_object(artifact_path)

    model = model_type.model_validate(original)

    serialized = model.model_dump_json(
        by_alias=True,
    )
    round_tripped: object = json.loads(serialized)

    assert round_tripped == original, (
        f"{artifact_name} changed during JSON -> model -> JSON round trip"
    )


def test_unreviewed_top_level_schema_drift_is_rejected() -> None:
    """A new unreviewed top-level artifact field must fail validation."""

    path = PHASE1_ARTIFACTS / "retrieval" / "evidence_package.json"
    payload = load_json_object(path)

    payload["unreviewed_stage3_field"] = "must-not-be-accepted"

    with pytest.raises(ValidationError):
        EvidencePackage.model_validate(payload)


def test_unreviewed_nested_schema_drift_is_rejected() -> None:
    """A new unreviewed nested field must also fail validation."""

    path = PHASE1_ARTIFACTS / "retrieval" / "evidence_package.json"
    original = load_json_object(path)
    payload = copy.deepcopy(original)

    domains = payload["domains"]
    assert isinstance(domains, dict)

    science = domains["science"]
    assert isinstance(science, dict)

    evidence = science["evidence"]
    assert isinstance(evidence, list)
    assert evidence, "Frozen Science evidence must not be empty"

    first_evidence = evidence[0]
    assert isinstance(first_evidence, dict)

    scores = first_evidence["scores"]
    assert isinstance(scores, dict)

    scores["unreviewed_stage3_score"] = 1.0

    with pytest.raises(ValidationError):
        EvidencePackage.model_validate(payload)
