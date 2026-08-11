from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

LOGGER = logging.getLogger("wth.phase1.validate_completion")

SCRIPT_VERSION: Final = "phase19-completion-validator-v1"
PHASE: Final = "phase_19_end_to_end_testing"
STATUS_PASSED: Final = "phase19_complete"
STATUS_FAILED: Final = "phase19_validation_failed"

EXPECTED_PHASES: Final = {
    "retrieval": "phase_14_build_retrieval_by_concept_and_domain",
    "generation": "phase_15_build_domain_specific_generation",
    "synthesis": "phase_16_synthesis_and_tension_detection",
    "coverage": "phase_17_coverage_classification",
    "final": "phase_18_final_response_assembly",
}

EXPECTED_STATUSES: Final = {
    "retrieval": "evaluation_complete",
    "generation": "domain_generation_complete",
    "synthesis": "synthesis_complete",
    "coverage": "coverage_classification_complete",
    "final": "final_response_complete",
}

DEFAULT_OUTPUT_DIRECTORY: Final = Path("artifacts/phase1/testing")
DEFAULT_MANIFEST_NAME: Final = "phase19_test_manifest.json"

OFFLINE_TESTS: Final = {
    "unit": Path("tests/phase1/test_phase1_unit.py"),
    "regression": Path("tests/phase1/test_phase1_regression.py"),
    "integration": Path("tests/phase1/test_phase1_integration.py"),
}

TEST_SUPPORT_FILES: Final = (
    Path("tests/phase1/phase1_regression_questions.yaml"),
    Path("tests/phase1/test_phase1_live_e2e.py"),
)

ARTIFACTS: Final = {
    "retrieval_manifest": Path("artifacts/phase1/retrieval/retrieval_manifest.json"),
    "generation_manifest": Path("artifacts/phase1/generation/generation_manifest.json"),
    "synthesis_manifest": Path("artifacts/phase1/synthesis/synthesis_manifest.json"),
    "coverage_manifest": Path("artifacts/phase1/coverage/coverage_manifest.json"),
    "final_manifest": Path("artifacts/phase1/final/final_response_manifest.json"),
    "final_response_json": Path("artifacts/phase1/final/final_response.json"),
    "final_response_markdown": Path("artifacts/phase1/final/final_response.md"),
}

PHASE14_REQUIRED_TRUE: Final = (
    "only_active_chunks_retrieved",
    "domain_separation_enforced",
    "concept_aware_retained",
)

PHASE15_REQUIRED_TRUE: Final = (
    "every_substantive_claim_maps_to_retrieved_chunks",
    "citations_resolve_from_retrieved_evidence",
    "domain_leakage_validation_passed",
    "independently_grounded_and_claim_cited",
)

PHASE16_SAFETY_KEYS: Final = (
    "atman_purusha_false_equivalence_rejected",
    "science_metaphysical_proof_rejected",
)

PHASE18_REQUIRED_TRUE: Final = (
    "all_claims_are_cited",
    "citations_resolve_to_active_chunks",
    "no_domain_leakage",
    "no_unsupported_equivalence",
    "coverage_status_matches_actual_phase17_evidence_classification",
    "corpus_and_prompt_versions_recorded",
)


class CompletionValidationError(RuntimeError):
    """Raised for an invalid Phase 19 completion state."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def configure_logging(level: str) -> None:
    numeric = getattr(
        logging,
        level.upper(),
        logging.INFO,
    )
    logging.basicConfig(
        level=numeric,
        format=("%(asctime)s %(levelname)s %(name)s %(message)s"),
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Validate and record Phase 19 end-to-end testing completion.")
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--live-e2e-passed",
        action="store_true",
        help=(
            "Record that the provider-backed "
            "Phase 19 live smoke test was run "
            "successfully outside this validator."
        ),
    )
    parser.add_argument(
        "--live-e2e-note",
        default=("Provider-backed Phase 14→18 smoke test verified separately with pytest -m live."),
    )
    parser.add_argument(
        "--replace",
        action="store_true",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
    )
    return parser.parse_args()


def resolve(
    project_root: Path,
    path: Path,
) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise CompletionValidationError(f"Required file missing: {path}")
    return path


def require_mapping(
    value: object,
    description: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise CompletionValidationError(f"{description} must be an object")

    result: dict[str, object] = {}
    for key, nested in value.items():
        if not isinstance(key, str):
            raise CompletionValidationError(f"{description} contains a non-string key")
        result[key] = nested
    return result


def require_string(
    value: object,
    description: str,
) -> str:
    if not isinstance(value, str):
        raise CompletionValidationError(f"{description} must be a string")
    stripped = value.strip()
    if not stripped:
        raise CompletionValidationError(f"{description} must be non-empty")
    return stripped


def load_json(path: Path) -> dict[str, object]:
    require_file(path)
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    return require_mapping(
        raw,
        f"JSON document {path}",
    )


def sha256_file(path: Path) -> str:
    require_file(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(
    path: Path,
    payload: Mapping[str, object],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )

    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            handle.write(encoded)
        Path(temporary_name).replace(path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def output_tail(
    text: str,
    *,
    maximum_lines: int = 30,
) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-maximum_lines:])


def run_pytest_suite(
    *,
    project_root: Path,
    suite_name: str,
    test_path: Path,
) -> dict[str, object]:
    resolved = resolve(
        project_root,
        test_path,
    )
    require_file(resolved)

    command = [
        sys.executable,
        "-m",
        "pytest",
        str(resolved),
        "-q",
        "--no-cov",
    ]

    LOGGER.info(
        "Running Phase 19 %s tests",
        suite_name,
    )

    started = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=project_root,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    elapsed_ms = round(
        (time.perf_counter() - started) * 1000.0,
        2,
    )

    combined = process.stdout + ("\n" if process.stdout and process.stderr else "") + process.stderr

    result: dict[str, object] = {
        "passed": process.returncode == 0,
        "return_code": process.returncode,
        "elapsed_ms": elapsed_ms,
        "command": command,
        "test_file": (test_path.as_posix()),
        "test_file_sha256": (sha256_file(resolved)),
        "output_tail": output_tail(combined),
    }

    LOGGER.info(
        "Phase 19 %s tests passed=%s",
        suite_name,
        result["passed"],
    )
    return result


def validate_manifest_identity(
    manifest: Mapping[str, object],
    *,
    name: str,
    expected_phase: str,
    expected_status: str,
) -> None:
    phase = require_string(
        manifest.get("phase"),
        f"{name} phase",
    )
    status = require_string(
        manifest.get("status"),
        f"{name} status",
    )

    if phase != expected_phase:
        raise CompletionValidationError(f"{name} phase mismatch: {phase!r}")
    if status != expected_status:
        raise CompletionValidationError(f"{name} status mismatch: {status!r}")


def require_true_gate_fields(
    gate: Mapping[str, object],
    keys: Sequence[str],
    *,
    description: str,
) -> None:
    for key in keys:
        if gate.get(key) is not True:
            raise CompletionValidationError(f"{description} gate failed: {key}")


def validate_phase14(
    manifest: Mapping[str, object],
) -> str:
    validate_manifest_identity(
        manifest,
        name="Phase 14",
        expected_phase=(EXPECTED_PHASES["retrieval"]),
        expected_status=(EXPECTED_STATUSES["retrieval"]),
    )

    corpus_version = require_string(
        manifest.get("corpus_version"),
        "Phase 14 corpus_version",
    )
    gate = require_mapping(
        manifest.get("exit_gate"),
        "Phase 14 exit_gate",
    )
    require_true_gate_fields(
        gate,
        PHASE14_REQUIRED_TRUE,
        description="Phase 14",
    )

    return corpus_version


def validate_phase15(
    manifest: Mapping[str, object],
    *,
    corpus_version: str,
) -> str:
    validate_manifest_identity(
        manifest,
        name="Phase 15",
        expected_phase=(EXPECTED_PHASES["generation"]),
        expected_status=(EXPECTED_STATUSES["generation"]),
    )

    if (
        require_string(
            manifest.get("corpus_version"),
            "Phase 15 corpus_version",
        )
        != corpus_version
    ):
        raise CompletionValidationError("Phase 15 corpus version differs from Phase 14.")

    question = require_string(
        manifest.get("question"),
        "Phase 15 question",
    )

    gate = require_mapping(
        manifest.get("exit_gate"),
        "Phase 15 exit_gate",
    )
    require_true_gate_fields(
        gate,
        PHASE15_REQUIRED_TRUE,
        description="Phase 15",
    )

    return question


def validate_phase16(
    manifest: Mapping[str, object],
    *,
    question: str,
    corpus_version: str,
) -> None:
    validate_manifest_identity(
        manifest,
        name="Phase 16",
        expected_phase=(EXPECTED_PHASES["synthesis"]),
        expected_status=(EXPECTED_STATUSES["synthesis"]),
    )

    if (
        require_string(
            manifest.get("question"),
            "Phase 16 question",
        )
        != question
    ):
        raise CompletionValidationError("Phase 16 question differs from Phase 15.")

    if (
        require_string(
            manifest.get("corpus_version"),
            "Phase 16 corpus_version",
        )
        != corpus_version
    ):
        raise CompletionValidationError("Phase 16 corpus version differs from Phase 14.")

    gate = require_mapping(
        manifest.get("exit_gate"),
        "Phase 16 exit_gate",
    )


    require_true_gate_fields(
        gate,
        PHASE16_SAFETY_KEYS,
        description="Phase 16",
    )


def validate_phase17(
    manifest: Mapping[str, object],
    *,
    question: str,
    corpus_version: str,
) -> None:
    validate_manifest_identity(
        manifest,
        name="Phase 17",
        expected_phase=(EXPECTED_PHASES["coverage"]),
        expected_status=(EXPECTED_STATUSES["coverage"]),
    )

    if (
        require_string(
            manifest.get("question"),
            "Phase 17 question",
        )
        != question
    ):
        raise CompletionValidationError("Phase 17 question differs from Phase 15.")

    if (
        require_string(
            manifest.get("corpus_version"),
            "Phase 17 corpus_version",
        )
        != corpus_version
    ):
        raise CompletionValidationError("Phase 17 corpus version differs from Phase 14.")

    gate = require_mapping(
        manifest.get("exit_gate"),
        "Phase 17 exit_gate",
    )
    if gate.get("passed") is not True:
        raise CompletionValidationError("Phase 17 exit gate did not pass.")


def validate_phase18(
    manifest: Mapping[str, object],
    response: Mapping[str, object],
    *,
    question: str,
    corpus_version: str,
) -> None:
    validate_manifest_identity(
        manifest,
        name="Phase 18",
        expected_phase=(EXPECTED_PHASES["final"]),
        expected_status=(EXPECTED_STATUSES["final"]),
    )

    if (
        require_string(
            manifest.get("question"),
            "Phase 18 question",
        )
        != question
    ):
        raise CompletionValidationError("Phase 18 question differs from Phase 15.")

    if (
        require_string(
            manifest.get("corpus_version"),
            "Phase 18 corpus_version",
        )
        != corpus_version
    ):
        raise CompletionValidationError("Phase 18 corpus version differs from Phase 14.")

    gate = require_mapping(
        manifest.get("exit_gate"),
        "Phase 18 exit_gate",
    )
    require_true_gate_fields(
        gate,
        PHASE18_REQUIRED_TRUE,
        description="Phase 18",
    )

    response_validation = require_mapping(
        response.get("validation"),
        "final_response validation",
    )
    if response_validation.get("passed") is not True:
        raise CompletionValidationError("Final response validation did not pass.")

    if (
        require_string(
            response.get("question"),
            "final_response question",
        )
        != question
    ):
        raise CompletionValidationError("Final response question differs from Phase 15.")

    if (
        require_string(
            response.get("corpus_version"),
            "final_response corpus_version",
        )
        != corpus_version
    ):
        raise CompletionValidationError("Final response corpus version differs from Phase 14.")


def artifact_hashes(
    *,
    project_root: Path,
) -> dict[str, dict[str, object]]:
    paths: dict[str, Path] = {
        **ARTIFACTS,
        **{f"test_{name}": path for name, path in OFFLINE_TESTS.items()},
        "regression_questions": (TEST_SUPPORT_FILES[0]),
        "live_e2e_test": (TEST_SUPPORT_FILES[1]),
    }

    result: dict[
        str,
        dict[str, object],
    ] = {}

    for name, relative in paths.items():
        resolved = resolve(
            project_root,
            relative,
        )
        require_file(resolved)
        result[name] = {
            "path": relative.as_posix(),
            "sha256": sha256_file(resolved),
            "size_bytes": (resolved.stat().st_size),
        }

    return result


def validate_artifact_chain(
    *,
    project_root: Path,
) -> dict[str, object]:
    manifests = {
        name: load_json(
            resolve(
                project_root,
                ARTIFACTS[f"{name}_manifest"],
            )
        )
        for name in (
            "retrieval",
            "generation",
            "synthesis",
            "coverage",
            "final",
        )
    }

    final_response = load_json(
        resolve(
            project_root,
            ARTIFACTS["final_response_json"],
        )
    )

    corpus_version = validate_phase14(manifests["retrieval"])
    question = validate_phase15(
        manifests["generation"],
        corpus_version=corpus_version,
    )
    validate_phase16(
        manifests["synthesis"],
        question=question,
        corpus_version=corpus_version,
    )
    validate_phase17(
        manifests["coverage"],
        question=question,
        corpus_version=corpus_version,
    )
    validate_phase18(
        manifests["final"],
        final_response,
        question=question,
        corpus_version=corpus_version,
    )

    return {
        "passed": True,
        "question": question,
        "corpus_version": corpus_version,
        "phase14_status": (manifests["retrieval"]["status"]),
        "phase15_status": (manifests["generation"]["status"]),
        "phase16_status": (manifests["synthesis"]["status"]),
        "phase17_status": (manifests["coverage"]["status"]),
        "phase18_status": (manifests["final"]["status"]),
        "final_response_validation_passed": True,
    }


def run_phase19_validation(
    *,
    project_root: Path,
    output_directory: Path,
    live_e2e_passed: bool,
    live_e2e_note: str,
    replace: bool,
) -> dict[str, object]:
    project_root = project_root.resolve()
    output_directory = resolve(
        project_root,
        output_directory,
    )
    manifest_path = output_directory / DEFAULT_MANIFEST_NAME

    if manifest_path.exists() and not replace:
        raise CompletionValidationError("Phase 19 manifest already exists. Use --replace.")

    LOGGER.info("Phase 19 completion validation starting")

    suite_results = {
        name: run_pytest_suite(
            project_root=project_root,
            suite_name=name,
            test_path=path,
        )
        for name, path in OFFLINE_TESTS.items()
    }

    offline_tests_passed = all(result.get("passed") is True for result in suite_results.values())

    artifact_chain: dict[
        str,
        object,
    ]
    artifact_chain_error: str | None = None

    try:
        artifact_chain = validate_artifact_chain(project_root=project_root)
    except CompletionValidationError as exc:
        artifact_chain = {
            "passed": False,
        }
        artifact_chain_error = str(exc)

    hashes: dict[
        str,
        dict[str, object],
    ]
    hash_error: str | None = None
    try:
        hashes = artifact_hashes(project_root=project_root)
    except CompletionValidationError as exc:
        hashes = {}
        hash_error = str(exc)

    artifact_chain_passed = artifact_chain.get("passed") is True
    hashes_complete = hash_error is None and bool(hashes)

    vertical_slice_reproducible = offline_tests_passed and artifact_chain_passed and live_e2e_passed

    exit_gate_passed = vertical_slice_reproducible and hashes_complete

    issues: list[dict[str, str]] = []

    for name, result in suite_results.items():
        if result.get("passed") is not True:
            issues.append(
                {
                    "severity": "error",
                    "code": (f"{name}_tests_failed"),
                    "message": (f"Phase 19 {name} pytest suite failed."),
                }
            )

    if not live_e2e_passed:
        issues.append(
            {
                "severity": "error",
                "code": ("live_e2e_not_recorded"),
                "message": (
                    "Provider-backed live E2E "
                    "PASS was not recorded. "
                    "Run the live smoke test "
                    "separately and rerun this "
                    "validator with "
                    "--live-e2e-passed."
                ),
            }
        )

    if artifact_chain_error:
        issues.append(
            {
                "severity": "error",
                "code": ("artifact_chain_invalid"),
                "message": (artifact_chain_error),
            }
        )

    if hash_error:
        issues.append(
            {
                "severity": "error",
                "code": ("artifact_hashing_failed"),
                "message": hash_error,
            }
        )

    manifest: dict[str, object] = {
        "phase": PHASE,
        "status": (STATUS_PASSED if exit_gate_passed else STATUS_FAILED),
        "script_version": SCRIPT_VERSION,
        "generated_at": utc_now(),
        "project_root": (project_root.as_posix()),
        "test_evidence": {
            "unit": (suite_results["unit"]),
            "regression": (suite_results["regression"]),
            "integration": (suite_results["integration"]),
            "live_e2e": {
                "passed": live_e2e_passed,
                "execution_mode": ("provider_backed_manual_pytest_invocation"),
                "provider_calls_rerun_by_validator": False,
                "note": live_e2e_note,
                "test_file": (TEST_SUPPORT_FILES[1].as_posix()),
            },
        },
        "artifact_chain": artifact_chain,
        "artifact_hashes": hashes,
        "validation": {
            "issue_count": len(issues),
            "issues": issues,
        },
        "exit_gate": {
            "passed": exit_gate_passed,
            "unit_tests_passed": (suite_results["unit"].get("passed") is True),
            "regression_tests_passed": (suite_results["regression"].get("passed") is True),
            "integration_tests_passed": (suite_results["integration"].get("passed") is True),
            "live_e2e_passed": (live_e2e_passed),
            "artifact_chain_valid": (artifact_chain_passed),
            "artifact_hashes_recorded": (hashes_complete),
            "vertical_slice_reproducible": (vertical_slice_reproducible),
        },
        "next_step": (
            "If the exit gate passes, freeze "
            "the validated Phase 1 baseline "
            "in Phase 20 completion criteria."
        ),
    }

    atomic_json(
        manifest_path,
        manifest,
    )

    exit_gate = require_mapping(manifest.get("exit_gate"), "exit_gate")

    LOGGER.info(
        "unit_tests_passed=%s",
        exit_gate.get("unit_tests_passed"),
    )
    LOGGER.info(
        "regression_tests_passed=%s",
        exit_gate.get("regression_tests_passed"),
    )
    LOGGER.info(
        "integration_tests_passed=%s",
        exit_gate.get("integration_tests_passed"),
    )

    LOGGER.info(
        "live_e2e_passed=%s",
        live_e2e_passed,
    )
    LOGGER.info(
        "artifact_chain_valid=%s",
        artifact_chain_passed,
    )
    LOGGER.info(
        "vertical_slice_reproducible=%s",
        vertical_slice_reproducible,
    )
    LOGGER.info(
        "Exit gate passed: %s",
        exit_gate_passed,
    )
    LOGGER.info(
        "Phase 19 manifest: %s",
        manifest_path,
    )

    return manifest


def main() -> int:
    arguments = parse_arguments()
    configure_logging(arguments.log_level)

    try:
        manifest = run_phase19_validation(
            project_root=(arguments.project_root),
            output_directory=(arguments.output_directory),
            live_e2e_passed=(arguments.live_e2e_passed),
            live_e2e_note=(arguments.live_e2e_note),
            replace=arguments.replace,
        )
    except CompletionValidationError:
        LOGGER.exception("Phase 19 completion validation failed")
        return 1

    exit_gate = require_mapping(
        manifest.get("exit_gate"),
        "Phase 19 exit_gate",
    )
    return 0 if exit_gate.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
