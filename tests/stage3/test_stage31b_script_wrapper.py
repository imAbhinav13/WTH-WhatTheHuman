from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pytest
from scripts.assemble_phase1_final_response import (
    DEFAULT_COVERAGE,
    DEFAULT_COVERAGE_MANIFEST,
    DEFAULT_DOMAIN_RESPONSES,
    DEFAULT_EVIDENCE_PACKAGE,
    DEFAULT_GENERATION_MANIFEST,
    DEFAULT_RETRIEVAL_MANIFEST,
    DEFAULT_SYNTHESIS,
    DEFAULT_SYNTHESIS_MANIFEST,
    AssemblyError,
    run_phase18,
)

from apps.api.models.runtime_contracts import (
    FinalResponse,
    FinalResponseManifest,
)

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]

FROZEN_FINAL_DIR: Final = (
    PROJECT_ROOT / "artifacts" / "phase1" / "final"
)

FROZEN_FINAL_RESPONSE: Final = (
    FROZEN_FINAL_DIR / "final_response.json"
)

FROZEN_FINAL_MARKDOWN: Final = (
    FROZEN_FINAL_DIR / "final_response.md"
)

FROZEN_FINAL_MANIFEST: Final = (
    FROZEN_FINAL_DIR / "final_response_manifest.json"
)


def load_json_object(
    path: Path,
) -> dict[str, object]:
    """Load one required JSON artifact."""

    assert path.is_file(), (
        f"Required artifact is missing: {path}"
    )

    raw: object = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert isinstance(raw, Mapping), (
        f"Expected JSON object: {path}"
    )

    result: dict[str, object] = {}

    for key, value in raw.items():
        assert isinstance(key, str)
        result[key] = value

    return result


def comparable_final_response(
    response: Mapping[str, object],
) -> dict[str, object]:
    """Remove only inherently run-specific Phase 18 metadata."""

    comparable = dict(response)
    comparable.pop("generated_at", None)

    return comparable


def comparable_manifest(
    manifest: Mapping[str, object],
) -> dict[str, object]:
    """Remove only run-specific wrapper metadata."""

    comparable = dict(manifest)

    comparable.pop(
        "generated_at",
        None,
    )

    # Artifact-mode outputs are intentionally written into a temporary
    # directory during this test, so their paths cannot equal the
    # historical frozen production paths.
    comparable.pop(
        "outputs",
        None,
    )

    return comparable


def execute_wrapper(
    output_directory: Path,
    *,
    replace: bool = True,
) -> dict[str, object]:
    """Run the artifact-mode Phase 18 wrapper."""

    return run_phase18(
        project_root=PROJECT_ROOT,
        evidence_package_path=DEFAULT_EVIDENCE_PACKAGE,
        retrieval_manifest_path=DEFAULT_RETRIEVAL_MANIFEST,
        domain_responses_path=DEFAULT_DOMAIN_RESPONSES,
        generation_manifest_path=DEFAULT_GENERATION_MANIFEST,
        synthesis_path=DEFAULT_SYNTHESIS,
        synthesis_manifest_path=DEFAULT_SYNTHESIS_MANIFEST,
        coverage_path=DEFAULT_COVERAGE,
        coverage_manifest_path=DEFAULT_COVERAGE_MANIFEST,
        output_directory=output_directory,
        replace=replace,
    )


def test_phase18_script_wrapper_writes_all_outputs(
    tmp_path: Path,
) -> None:
    """The legacy script entry point must still create all Phase 18 files."""

    output_directory = tmp_path / "phase18-wrapper"

    returned_manifest = execute_wrapper(
        output_directory
    )

    final_response_path = (
        output_directory / "final_response.json"
    )
    markdown_path = (
        output_directory / "final_response.md"
    )
    manifest_path = (
        output_directory / "final_response_manifest.json"
    )

    assert final_response_path.is_file()
    assert markdown_path.is_file()
    assert manifest_path.is_file()

    persisted_manifest = load_json_object(
        manifest_path
    )

    assert returned_manifest == persisted_manifest


def test_phase18_script_wrapper_response_matches_frozen_response(
    tmp_path: Path,
) -> None:
    """Wrapper mode must preserve the validated Phase 18 JSON semantics."""

    output_directory = tmp_path / "phase18-wrapper"

    execute_wrapper(
        output_directory
    )

    actual = load_json_object(
        output_directory / "final_response.json"
    )

    expected = load_json_object(
        FROZEN_FINAL_RESPONSE
    )

    # Both documents must continue to satisfy the canonical
    # Stage 3.0 FinalResponse contract.
    FinalResponse.model_validate(actual)
    FinalResponse.model_validate(expected)

    assert comparable_final_response(
        actual
    ) == comparable_final_response(
        expected
    )


def test_phase18_script_wrapper_markdown_matches_frozen_exactly(
    tmp_path: Path,
) -> None:
    """Wrapper mode must reproduce frozen user-facing Markdown exactly."""

    output_directory = tmp_path / "phase18-wrapper"

    execute_wrapper(
        output_directory
    )

    actual_path = (
        output_directory / "final_response.md"
    )

    assert actual_path.is_file()
    assert FROZEN_FINAL_MARKDOWN.is_file()

    actual = actual_path.read_text(
        encoding="utf-8"
    )

    expected = FROZEN_FINAL_MARKDOWN.read_text(
        encoding="utf-8"
    )

    assert actual == expected


def test_phase18_script_wrapper_manifest_matches_frozen_semantics(
    tmp_path: Path,
) -> None:
    """Wrapper manifest must retain the frozen Phase 18 contract."""

    output_directory = tmp_path / "phase18-wrapper"

    returned_manifest = execute_wrapper(
        output_directory
    )

    actual = load_json_object(
        output_directory / "final_response_manifest.json"
    )

    expected = load_json_object(
        FROZEN_FINAL_MANIFEST
    )

    FinalResponseManifest.model_validate(actual)
    FinalResponseManifest.model_validate(expected)

    assert comparable_manifest(
        actual
    ) == comparable_manifest(
        expected
    )

    assert comparable_manifest(
        returned_manifest
    ) == comparable_manifest(
        expected
    )


def test_phase18_script_wrapper_preserves_zero_provider_calls(
    tmp_path: Path,
) -> None:
    """Artifact mode must remain provider-free after runtime refactoring."""

    output_directory = tmp_path / "phase18-wrapper"

    execute_wrapper(
        output_directory
    )

    response = FinalResponse.model_validate(
        load_json_object(
            output_directory / "final_response.json"
        )
    )

    assert (
        response.provider_calls.phase18_llm_calls
        == 0
    )

    assert (
        response.provider_calls.phase18_embedding_calls
        == 0
    )

    assert (
        response.provider_calls.phase18_retrieval_calls
        == 0
    )


def test_phase18_script_wrapper_preserves_exit_gate(
    tmp_path: Path,
) -> None:
    """The existing Phase 18 artifact exit gate must remain green."""

    output_directory = tmp_path / "phase18-wrapper"

    returned_manifest = execute_wrapper(
        output_directory
    )

    manifest = FinalResponseManifest.model_validate(
        returned_manifest
    )

    assert manifest.status == (
        "final_response_complete"
    )

    assert manifest.exit_gate.passed is True
    assert (
        manifest.exit_gate.all_claims_are_cited
        is True
    )
    assert (
        manifest.exit_gate.citations_resolve_to_active_chunks
        is True
    )
    assert (
        manifest.exit_gate.no_domain_leakage
        is True
    )
    assert (
        manifest.exit_gate.no_unsupported_equivalence
        is True
    )
    assert (
        manifest.exit_gate
        .coverage_status_matches_actual_phase17_evidence_classification
        is True
    )
    assert (
        manifest.exit_gate
        .corpus_and_prompt_versions_recorded
        is True
    )


def test_phase18_script_wrapper_preserves_replace_policy(
    tmp_path: Path,
) -> None:
    """Existing outputs must not be overwritten without replace=True."""

    output_directory = tmp_path / "phase18-wrapper"

    execute_wrapper(
        output_directory,
        replace=True,
    )

    with pytest.raises(
        AssemblyError,
        match="outputs already exist",
    ):
        execute_wrapper(
            output_directory,
            replace=False,
        )