from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import pytest

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]

PHASE14_MODULE: Final = "scripts.build_phase1_retrieval"
PHASE15_MODULE: Final = "scripts.build_phase1_domain_generation"
PHASE16_MODULE: Final = "scripts.build_phase1_synthesis"
PHASE17_MODULE: Final = "scripts.classify_phase1_coverage"
PHASE18_MODULE: Final = "scripts.assemble_phase1_final_response"

FROZEN_RETRIEVAL_MANIFEST: Final = (
    PROJECT_ROOT / "artifacts" / "phase1" / "retrieval" / "retrieval_manifest.json"
)
EXPECTED_DOMAINS: Final = {
    "science",
    "advaita",
    "samkhya",
}
EXPECTED_CORPUS_VERSION: Final = "phase1_active_corpus_v1"
ALLOWED_COVERAGE_STATUSES: Final = {
    "Supported",
    "Partially Supported",
    "Out of Corpus",
}

FULL_SMOKE_QUESTION: Final = "How is consciousness related to the self and experienced reality?"
HARD_NEGATIVE_QUESTION: Final = "Are Atman and Purusha the same concept?"

ATMAN_PURUSHA_EQUIVALENCE_RE: Final = re.compile(
    r"\b(?:atman|ātman)\b.{0,100}\b"
    r"(?:same|identical|equivalent|same concept|same entity)\b"
    r".{0,100}\b(?:purusha|puruṣa)\b"
    r"|"
    r"\b(?:purusha|puruṣa)\b.{0,100}\b"
    r"(?:same|identical|equivalent|same concept|same entity)\b"
    r".{0,100}\b(?:atman|ātman)\b",
    re.IGNORECASE | re.DOTALL,
)
SCIENCE_METAPHYSICS_PROOF_RE: Final = re.compile(
    r"\b(?:science|scientific|neuroscience|neurological|empirical)\b"
    r".{0,160}\b"
    r"(?:prove|proves|proven|establishes|confirms)\b"
    r".{0,120}\b"
    r"(?:atman|ātman|brahman|purusha|puruṣa|maya|māyā|nondual|non-dual)\b",
    re.IGNORECASE | re.DOTALL,
)


class LiveE2EError(AssertionError):
    """Raised when a live Phase 19 smoke step fails."""


def require_mapping(
    value: object,
    description: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise LiveE2EError(f"{description} must be an object")

    result: dict[str, object] = {}
    for key, nested in value.items():
        if not isinstance(key, str):
            raise LiveE2EError(f"{description} contains a non-string key")
        result[key] = nested
    return result


def require_list(
    value: object,
    description: str,
) -> list[object]:
    if not isinstance(value, list):
        raise LiveE2EError(f"{description} must be a list")
    return value


def require_string(
    value: object,
    description: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LiveE2EError(f"{description} must be a non-empty string")
    return value.strip()


def load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise LiveE2EError(f"Expected live artifact does not exist: {path}")

    raw: object = json.loads(path.read_text(encoding="utf-8"))
    return require_mapping(
        raw,
        f"JSON document {path}",
    )


def env_file_value(key_name: str) -> str:
    candidates = (
        PROJECT_ROOT / ".env",
        PROJECT_ROOT / ".env.local",
        PROJECT_ROOT / "apps" / "api" / ".env",
        PROJECT_ROOT / "apps" / "api" / ".env.local",
    )

    for path in candidates:
        if not path.is_file():
            continue

        for raw_line in path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue

            key, raw_value = stripped.split("=", 1)
            if key.strip() != key_name:
                continue

            value = raw_value.strip().strip('"').strip("'")
            if value:
                return value

    return ""


def provider_configuration_available() -> bool:
    groq = os.getenv("GROQ_API_KEY", "").strip() or env_file_value("GROQ_API_KEY")
    embedding = (
        os.getenv("GEMINI_API_KEY", "").strip()
        or env_file_value("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY", "").strip()
        or env_file_value("GOOGLE_API_KEY")
    )
    return bool(groq and embedding)


def module_help(module: str) -> str:
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            module,
            "--help",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )

    if process.returncode != 0:
        raise LiveE2EError(
            f"Could not inspect CLI for {module}.\n"
            f"STDOUT:\n{process.stdout}\n"
            f"STDERR:\n{process.stderr}"
        )

    return process.stdout + process.stderr


def choose_cli_option(
    help_text: str,
    candidates: Sequence[str],
    *,
    module: str,
    purpose: str,
) -> str:
    for option in candidates:
        if option in help_text:
            return option

    raise LiveE2EError(
        f"{module} exposes no recognized CLI option for {purpose}. Tried: {', '.join(candidates)}"
    )


def run_module(
    module: str,
    arguments: Sequence[str],
    *,
    timeout_seconds: int = 180,
) -> None:
    command = [
        sys.executable,
        "-m",
        module,
        *arguments,
    ]

    process = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )

    if process.returncode != 0:
        raise LiveE2EError(
            "Live E2E step failed.\n"
            f"Module: {module}\n"
            f"Command: {' '.join(command)}\n"
            f"Exit code: {process.returncode}\n"
            f"STDOUT:\n{process.stdout}\n"
            f"STDERR:\n{process.stderr}"
        )


def run_phase14_cli(
    *,
    question: str,
    output_directory: Path,
) -> tuple[Path, Path]:
    help_text = module_help(PHASE14_MODULE)

    question_option = choose_cli_option(
        help_text,
        (
            "--question",
            "--query",
            "--question-text",
        ),
        module=PHASE14_MODULE,
        purpose="question input",
    )
    output_option = choose_cli_option(
        help_text,
        (
            "--output-directory",
            "--output-dir",
        ),
        module=PHASE14_MODULE,
        purpose="output directory",
    )

    run_module(
        PHASE14_MODULE,
        [
            "--project-root",
            str(PROJECT_ROOT),
            question_option,
            question,
            output_option,
            str(output_directory),
            "--replace",
        ],
        timeout_seconds=180,
    )

    return (
        output_directory / "evidence_package.json",
        output_directory / "retrieval_manifest.json",
    )


def run_phase15_cli(
    *,
    evidence_package: Path,
    retrieval_manifest: Path,
    output_directory: Path,
) -> tuple[Path, Path]:
    run_module(
        PHASE15_MODULE,
        [
            "--project-root",
            str(PROJECT_ROOT),
            "--evidence-package",
            str(evidence_package),
            "--retrieval-manifest",
            str(retrieval_manifest),
            "--output-directory",
            str(output_directory),
            "--replace",
        ],
        timeout_seconds=240,
    )

    return (
        output_directory / "domain_responses.json",
        output_directory / "generation_manifest.json",
    )


def run_phase16_cli(
    *,
    domain_responses: Path,
    generation_manifest: Path,
    output_directory: Path,
) -> tuple[Path, Path]:
    run_module(
        PHASE16_MODULE,
        [
            "--project-root",
            str(PROJECT_ROOT),
            "--domain-responses",
            str(domain_responses),
            "--generation-manifest",
            str(generation_manifest),
            "--output-directory",
            str(output_directory),
            "--replace",
        ],
        timeout_seconds=180,
    )

    return (
        output_directory / "synthesis.json",
        output_directory / "synthesis_manifest.json",
    )


def run_phase17_cli(
    *,
    evidence_package: Path,
    retrieval_manifest: Path,
    domain_responses: Path,
    generation_manifest: Path,
    synthesis: Path,
    synthesis_manifest: Path,
    output_directory: Path,
) -> tuple[Path, Path]:
    help_text = module_help(PHASE17_MODULE)

    arguments = [
        "--project-root",
        str(PROJECT_ROOT),
    ]

    option_values: tuple[tuple[tuple[str, ...], Path, str], ...] = (
        (
            ("--evidence-package",),
            evidence_package,
            "evidence package",
        ),
        (
            ("--retrieval-manifest",),
            retrieval_manifest,
            "retrieval manifest",
        ),
        (
            ("--domain-responses",),
            domain_responses,
            "domain responses",
        ),
        (
            ("--generation-manifest",),
            generation_manifest,
            "generation manifest",
        ),
        (
            ("--synthesis",),
            synthesis,
            "synthesis",
        ),
        (
            ("--synthesis-manifest",),
            synthesis_manifest,
            "synthesis manifest",
        ),
        (
            ("--output-directory", "--output-dir"),
            output_directory,
            "output directory",
        ),
    )

    for candidate_options, value, purpose in option_values:
        option = choose_cli_option(
            help_text,
            candidate_options,
            module=PHASE17_MODULE,
            purpose=purpose,
        )
        arguments.extend([option, str(value)])

    arguments.append("--replace")

    run_module(
        PHASE17_MODULE,
        arguments,
        timeout_seconds=120,
    )

    return (
        output_directory / "coverage.json",
        output_directory / "coverage_manifest.json",
    )


def run_phase18_cli(
    *,
    evidence_package: Path,
    retrieval_manifest: Path,
    domain_responses: Path,
    generation_manifest: Path,
    synthesis: Path,
    synthesis_manifest: Path,
    coverage: Path,
    coverage_manifest: Path,
    output_directory: Path,
) -> tuple[Path, Path]:
    run_module(
        PHASE18_MODULE,
        [
            "--project-root",
            str(PROJECT_ROOT),
            "--evidence-package",
            str(evidence_package),
            "--retrieval-manifest",
            str(retrieval_manifest),
            "--domain-responses",
            str(domain_responses),
            "--generation-manifest",
            str(generation_manifest),
            "--synthesis",
            str(synthesis),
            "--synthesis-manifest",
            str(synthesis_manifest),
            "--coverage",
            str(coverage),
            "--coverage-manifest",
            str(coverage_manifest),
            "--output-directory",
            str(output_directory),
            "--replace",
        ],
        timeout_seconds=120,
    )

    return (
        output_directory / "final_response.json",
        output_directory / "final_response_manifest.json",
    )


def assert_manifest_passed(
    path: Path,
    *,
    expected_phase: str,
    expected_question: str | None,
) -> dict[str, object]:
    manifest = load_json(path)

    assert (
        require_string(
            manifest.get("phase"),
            f"{path.name} phase",
        )
        == expected_phase
    )
    if expected_question is not None:
        assert (
            require_string(
                manifest.get("question"),
                f"{path.name} question",
            )
            == expected_question
        )
    assert (
        require_string(
            manifest.get("corpus_version"),
            f"{path.name} corpus_version",
        )
        == EXPECTED_CORPUS_VERSION
    )

    gate = require_mapping(
        manifest.get("exit_gate"),
        f"{path.name} exit_gate",
    )
    boolean_values = [value for value in gate.values() if isinstance(value, bool)]
    assert boolean_values, f"{path.name} has no Boolean exit-gate values"

    return manifest


def assert_phase15_grounding(
    generation_manifest_path: Path,
) -> None:
    manifest = load_json(generation_manifest_path)
    gate = require_mapping(
        manifest.get("exit_gate"),
        "Phase 15 exit gate",
    )

    for key in (
        "every_substantive_claim_maps_to_retrieved_chunks",
        "citations_resolve_from_retrieved_evidence",
        "domain_leakage_validation_passed",
        "independently_grounded_and_claim_cited",
    ):
        assert gate.get(key) is True, f"Phase 15 live gate failed: {key}"


def assert_phase16_synthesis(
    synthesis_path: Path,
    synthesis_manifest_path: Path,
) -> None:
    synthesis = load_json(synthesis_path)
    validation = require_mapping(
        synthesis.get("validation"),
        "Phase 16 synthesis validation",
    )
    assert validation.get("passed") is True

    manifest = load_json(synthesis_manifest_path)
    gate = require_mapping(
        manifest.get("exit_gate"),
        "Phase 16 exit gate",
    )

    for key in (
        "domain_differences_preserved",
        "all_comparison_references_validated",
        "atman_purusha_false_equivalence_rejected",
        "science_metaphysical_proof_rejected",
    ):
        assert gate.get(key) is True, f"Phase 16 live gate failed: {key}"


def final_response_semantic_text(
    response: Mapping[str, object],
) -> str:
    sections = require_mapping(
        response.get("sections"),
        "final response sections",
    )
    return json.dumps(
        sections,
        ensure_ascii=False,
        sort_keys=True,
    )


def assert_final_response_contract(
    final_response_path: Path,
    final_manifest_path: Path,
    *,
    expected_question: str,
    require_all_domains: bool,
) -> dict[str, object]:
    response = load_json(final_response_path)
    manifest = load_json(final_manifest_path)

    assert (
        require_string(
            response.get("question"),
            "final response question",
        )
        == expected_question
    )
    assert (
        require_string(
            response.get("corpus_version"),
            "final response corpus_version",
        )
        == EXPECTED_CORPUS_VERSION
    )

    validation = require_mapping(
        response.get("validation"),
        "final response validation",
    )
    assert validation.get("passed") is True

    manifest_gate = require_mapping(
        manifest.get("exit_gate"),
        "Phase 18 manifest exit_gate",
    )
    for key in (
        "all_claims_are_cited",
        "citations_resolve_to_active_chunks",
        "no_domain_leakage",
        "no_unsupported_equivalence",
        "coverage_status_matches_actual_phase17_evidence_classification",
        "corpus_and_prompt_versions_recorded",
    ):
        assert manifest_gate.get(key) is True, f"Phase 18 live gate failed: {key}"

    sections = require_mapping(
        response.get("sections"),
        "final response sections",
    )
    coverage = require_mapping(
        sections.get("coverage"),
        "final coverage",
    )
    coverage_status = require_string(
        coverage.get("coverage_status"),
        "coverage status",
    )
    assert coverage_status in (ALLOWED_COVERAGE_STATUSES)

    if coverage_status != "Out of Corpus":
        citations = require_list(
            response.get("claim_level_citations"),
            "claim-level citations",
        )
        assert citations, "Corpus-grounded live response has no citations"

        domains = require_mapping(
            sections.get("domain_perspectives"),
            "domain perspectives",
        )
        if require_all_domains:
            assert set(domains) == EXPECTED_DOMAINS

    semantic_text = final_response_semantic_text(response)
    assert not ATMAN_PURUSHA_EQUIVALENCE_RE.search(semantic_text)
    assert not SCIENCE_METAPHYSICS_PROOF_RE.search(semantic_text)

    return response


def run_live_vertical_slice(
    *,
    question: str,
    tmp_path: Path,
) -> dict[str, object]:
    root = tmp_path / "live-e2e"
    phase14 = root / "phase14"
    phase15 = root / "phase15"
    phase16 = root / "phase16"
    phase17 = root / "phase17"
    phase18 = root / "phase18"

    evidence_package, runtime_retrieval_manifest = run_phase14_cli(
        question=question,
        output_directory=phase14,
    )
    assert_manifest_passed(
        runtime_retrieval_manifest,
        expected_phase=("phase_14_build_retrieval_by_concept_and_domain"),
        expected_question=None,
    )
    phase14_evidence = load_json(evidence_package)

    assert (
        require_string(phase14_evidence.get("question"), "Phase 14 evidence question") == question
    )
    assert (
        require_string(phase14_evidence.get("corpus_version"), "Phase 14 evidence corpus_version")
        == EXPECTED_CORPUS_VERSION
    )
    assert (
        require_string(phase14_evidence.get("retrieval_mode"), "Phase 14 retrieval_mode")
        == "concept_aware"
    )
    assert FROZEN_RETRIEVAL_MANIFEST.is_file(), (
        f"Frozen Phase 14 retrieval evaluation manifest is missing: {FROZEN_RETRIEVAL_MANIFEST}"
    )

    domain_responses, generation_manifest = run_phase15_cli(
        evidence_package=evidence_package,
        retrieval_manifest=FROZEN_RETRIEVAL_MANIFEST,
        output_directory=phase15,
    )
    assert_manifest_passed(
        generation_manifest,
        expected_phase=("phase_15_build_domain_specific_generation"),
        expected_question=question,
    )
    assert_phase15_grounding(generation_manifest)

    synthesis, synthesis_manifest = run_phase16_cli(
        domain_responses=domain_responses,
        generation_manifest=generation_manifest,
        output_directory=phase16,
    )
    assert_manifest_passed(
        synthesis_manifest,
        expected_phase=("phase_16_synthesis_and_tension_detection"),
        expected_question=question,
    )
    assert_phase16_synthesis(
        synthesis,
        synthesis_manifest,
    )

    coverage, coverage_manifest = run_phase17_cli(
        evidence_package=evidence_package,
        retrieval_manifest=FROZEN_RETRIEVAL_MANIFEST,
        domain_responses=domain_responses,
        generation_manifest=generation_manifest,
        synthesis=synthesis,
        synthesis_manifest=synthesis_manifest,
        output_directory=phase17,
    )
    assert_manifest_passed(
        coverage_manifest,
        expected_phase=("phase_17_coverage_classification"),
        expected_question=question,
    )

    final_response, final_manifest = run_phase18_cli(
        evidence_package=evidence_package,
        retrieval_manifest=FROZEN_RETRIEVAL_MANIFEST,
        domain_responses=domain_responses,
        generation_manifest=generation_manifest,
        synthesis=synthesis,
        synthesis_manifest=synthesis_manifest,
        coverage=coverage,
        coverage_manifest=coverage_manifest,
        output_directory=phase18,
    )

    return assert_final_response_contract(
        final_response,
        final_manifest,
        expected_question=question,
        require_all_domains=True,
    )


@pytest.fixture(scope="session", autouse=True)
def require_live_provider_configuration() -> None:
    if provider_configuration_available():
        return

    pytest.skip(
        "Live Phase 19 E2E requires GROQ_API_KEY and a "
        "Gemini/Google embedding API key in environment or project .env files."
    )


@pytest.mark.live
def test_live_full_vertical_slice_multi_concept(
    tmp_path: Path,
) -> None:
    response = run_live_vertical_slice(
        question=FULL_SMOKE_QUESTION,
        tmp_path=tmp_path,
    )

    sections = require_mapping(
        response.get("sections"),
        "final sections",
    )
    activated = require_list(
        sections.get("activated_concepts"),
        "activated concepts",
    )

    active_concepts = {
        require_string(
            require_mapping(
                item,
                "activated concept",
            ).get("concept"),
            "activated concept id",
        )
        for item in activated
    }

    assert {
        "consciousness",
        "self_identity",
        "reality_appearance",
    }.issubset(active_concepts)


@pytest.mark.live
@pytest.mark.skipif(
    os.getenv(
        "WTH_RUN_EXTENDED_LIVE_E2E",
        "",
    ).strip()
    not in {"1", "true", "TRUE", "yes", "YES"},
    reason=(
        "Set WTH_RUN_EXTENDED_LIVE_E2E=1 to run the "
        "additional provider-backed hard-negative smoke test."
    ),
)
def test_live_hard_negative_atman_purusha(
    tmp_path: Path,
) -> None:
    response = run_live_vertical_slice(
        question=HARD_NEGATIVE_QUESTION,
        tmp_path=tmp_path,
    )

    semantic_text = final_response_semantic_text(response)

    assert not ATMAN_PURUSHA_EQUIVALENCE_RE.search(semantic_text)

    sections = require_mapping(
        response.get("sections"),
        "final sections",
    )
    synthesis = require_mapping(
        sections.get("comparative_synthesis"),
        "comparative synthesis",
    )
    comparisons = require_list(
        synthesis.get("comparisons"),
        "comparisons",
    )

    assert comparisons, "Hard-negative live response produced no comparative synthesis"
