from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

LOGGER = logging.getLogger("wth.phase1.freeze_baseline")

SCRIPT_VERSION: Final = "phase20-freeze-baseline-v1"
PHASE: Final = "phase_20_phase1_completion_and_freeze"
STATUS_COMPLETE: Final = "phase1_frozen_complete"
DEFAULT_OUTPUT_DIRECTORY: Final = Path("artifacts/phase1/completion")
DEFAULT_MANIFEST_NAME: Final = "phase1_completion_manifest.json"

EXPECTED_CORPUS_VERSION: Final = "phase1_active_corpus_v1"
EXPECTED_ACTIVE_CHUNKS: Final = 318
EXPECTED_CONCEPT_RELATIONS: Final = 954
EXPECTED_CONCEPTS: Final = (
    "consciousness",
    "self_identity",
    "reality_appearance",
)
EXPECTED_DOMAINS: Final = (
    "science",
    "advaita",
    "samkhya",
)

MANIFEST_PATHS: Final = {
    "phase19": Path("artifacts/phase1/testing/phase19_test_manifest.json"),
    "activation": Path("artifacts/phase1/active/activation_manifest.json"),
    "weighted_tags": Path(
        "artifacts/phase1/concepts/phase1_reviewed_weighted_concept_tags_manifest.json"
    ),
    "embedding": Path("artifacts/phase1/embeddings/embedding_manifest.json"),
    "split": Path("data/evaluation/phase1_split_manifest.json"),
    "retrieval": Path("artifacts/phase1/retrieval/retrieval_manifest.json"),
    "generation": Path("artifacts/phase1/generation/generation_manifest.json"),
    "synthesis": Path("artifacts/phase1/synthesis/synthesis_manifest.json"),
    "coverage": Path("artifacts/phase1/coverage/coverage_manifest.json"),
    "final": Path("artifacts/phase1/final/final_response_manifest.json"),
    "final_response": Path("artifacts/phase1/final/final_response.json"),
}

FROZEN_ARTIFACTS: Final = {
    "scope_rules": Path("docs/corpus/phase1_section_scope.yaml"),
    "gold_corpus": Path("artifacts/phase1/reviewed/phase1_reviewed_gold_corpus.jsonl"),
    "review_manifest": Path("artifacts/phase1/reviewed/phase1_human_review_manifest.json"),
    "build_split": Path("data/evaluation/phase1_build.jsonl"),
    "development_split": Path("data/evaluation/phase1_development.jsonl"),
    "heldout_split": Path("data/evaluation/phase1_heldout.jsonl"),
    "split_manifest": MANIFEST_PATHS["split"],
    "concept_prototypes": Path("data/concepts/phase1_concept_prototypes.yaml"),
    "embedding_manifest": MANIFEST_PATHS["embedding"],
    "approved_chunk_embeddings": Path(
        "artifacts/phase1/embeddings/approved_chunk_embeddings.jsonl"
    ),
    "query_prototype_embeddings": Path(
        "artifacts/phase1/embeddings/query_prototype_embeddings.jsonl"
    ),
    "passage_prototype_embeddings": Path(
        "artifacts/phase1/embeddings/passage_prototype_embeddings.jsonl"
    ),
    "concept_mapping_dev_results": Path(
        "artifacts/phase1/evaluation/concept_mapping_dev_results.json"
    ),
    "heldout_results": Path("artifacts/phase1/evaluation/heldout_results.json"),
    "weighted_concept_tags": Path(
        "artifacts/phase1/concepts/phase1_reviewed_weighted_concept_tags.jsonl"
    ),
    "weighted_concept_tags_manifest": MANIFEST_PATHS["weighted_tags"],
    "active_chunk_bundles": Path("artifacts/phase1/active/active_chunk_bundles.jsonl"),
    "reviewed_chunk_concepts": Path("artifacts/phase1/active/reviewed_chunk_concepts.jsonl"),
    "activation_manifest": MANIFEST_PATHS["activation"],
    "retrieval_manifest": MANIFEST_PATHS["retrieval"],
    "evidence_package": Path("artifacts/phase1/retrieval/evidence_package.json"),
    "generation_manifest": MANIFEST_PATHS["generation"],
    "domain_responses": Path("artifacts/phase1/generation/domain_responses.json"),
    "synthesis_manifest": MANIFEST_PATHS["synthesis"],
    "synthesis": Path("artifacts/phase1/synthesis/synthesis.json"),
    "coverage_manifest": MANIFEST_PATHS["coverage"],
    "coverage": Path("artifacts/phase1/coverage/coverage.json"),
    "final_manifest": MANIFEST_PATHS["final"],
    "final_response_json": MANIFEST_PATHS["final_response"],
    "final_response_markdown": Path("artifacts/phase1/final/final_response.md"),
    "phase19_test_manifest": MANIFEST_PATHS["phase19"],
    "phase19_unit_tests": Path("tests/phase1/test_phase1_unit.py"),
    "phase19_regression_questions": Path("tests/phase1/phase1_regression_questions.yaml"),
    "phase19_regression_tests": Path("tests/phase1/test_phase1_regression.py"),
    "phase19_integration_tests": Path("tests/phase1/test_phase1_integration.py"),
    "phase19_live_e2e_tests": Path("tests/phase1/test_phase1_live_e2e.py"),
}


class FreezeError(RuntimeError):
    """Raised when the validated Phase 1 baseline cannot be frozen."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def configure_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 20: freeze the validated WTH Phase 1 vertical slice."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def resolve(project_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FreezeError(f"Required Phase 1 file missing: {path}")
    return path


def require_mapping(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise FreezeError(f"{description} must be an object")
    result: dict[str, object] = {}
    for key, nested in value.items():
        if not isinstance(key, str):
            raise FreezeError(f"{description} contains a non-string key")
        result[key] = nested
    return result


def require_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FreezeError(f"{description} must be a non-empty string")
    return value.strip()


def require_int(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FreezeError(f"{description} must be an integer")
    return value


def load_json(path: Path) -> dict[str, object]:
    require_file(path)
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    return require_mapping(raw, f"JSON document {path}")


def sha256_file(path: Path) -> str:
    require_file(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
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
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        Path(temporary_name).replace(path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def require_true_fields(
    mapping: Mapping[str, object],
    fields: tuple[str, ...],
    *,
    description: str,
) -> None:
    for field in fields:
        if mapping.get(field) is not True:
            raise FreezeError(f"{description} failed: {field}")


def get_manifest(project_root: Path, name: str) -> dict[str, object]:
    return load_json(resolve(project_root, MANIFEST_PATHS[name]))


def validate_phase19(phase19: Mapping[str, object]) -> None:
    if require_string(phase19.get("phase"), "Phase 19 phase") != ("phase_19_end_to_end_testing"):
        raise FreezeError("Unexpected Phase 19 manifest phase")
    if require_string(phase19.get("status"), "Phase 19 status") != ("phase19_complete"):
        raise FreezeError("Phase 19 is not complete")
    gate = require_mapping(phase19.get("exit_gate"), "Phase 19 exit_gate")
    require_true_fields(
        gate,
        (
            "passed",
            "unit_tests_passed",
            "regression_tests_passed",
            "integration_tests_passed",
            "live_e2e_passed",
            "artifact_chain_valid",
            "artifact_hashes_recorded",
            "vertical_slice_reproducible",
        ),
        description="Phase 19 exit gate",
    )


def validate_activation(
    activation: Mapping[str, object],
) -> tuple[str, dict[str, object], dict[str, object]]:
    if require_string(activation.get("status"), "activation status") != (
        "activation_artifacts_complete"
    ):
        raise FreezeError("Phase 13 activation is incomplete")
    if (
        require_string(
            activation.get("lifecycle_status"),
            "activation lifecycle_status",
        )
        != "active"
    ):
        raise FreezeError("Phase 1 corpus is not active")

    corpus_version = require_string(
        activation.get("corpus_version"),
        "activation corpus_version",
    )
    if corpus_version != EXPECTED_CORPUS_VERSION:
        raise FreezeError("Unexpected active corpus version")

    counts = require_mapping(activation.get("counts"), "activation counts")
    if require_int(counts.get("active_chunk_count"), "active_chunk_count") != (
        EXPECTED_ACTIVE_CHUNKS
    ):
        raise FreezeError("Active chunk count changed")
    if (
        require_int(
            counts.get("reviewed_chunk_concept_count"),
            "reviewed_chunk_concept_count",
        )
        != EXPECTED_CONCEPT_RELATIONS
    ):
        raise FreezeError("Reviewed concept relation count changed")

    embedding_identity = require_mapping(
        activation.get("embedding_identity"),
        "activation embedding_identity",
    )
    if require_string(embedding_identity.get("model"), "embedding model") != ("gemini-embedding-2"):
        raise FreezeError("Frozen embedding model changed")
    if (
        require_string(
            embedding_identity.get("model_revision"),
            "embedding model_revision",
        )
        != "2"
    ):
        raise FreezeError("Frozen embedding revision changed")
    if (
        require_int(
            embedding_identity.get("dimensions"),
            "embedding dimensions",
        )
        != 768
    ):
        raise FreezeError("Frozen embedding dimensions changed")

    database_step = require_mapping(
        activation.get("database_step"),
        "activation database_step",
    )
    return corpus_version, embedding_identity, database_step


def validate_weighted_tags(weighted_tags: Mapping[str, object]) -> dict[str, object]:
    if require_string(weighted_tags.get("status"), "weighted-tags status") != (
        "production_concept_weights_complete"
    ):
        raise FreezeError("Phase 12 weighted tags are incomplete")
    counts = require_mapping(weighted_tags.get("counts"), "weighted-tags counts")
    if require_int(counts.get("approved_chunks"), "approved_chunks") != (EXPECTED_ACTIVE_CHUNKS):
        raise FreezeError("Weighted-tag chunk count changed")
    if require_int(counts.get("tag_count"), "tag_count") != EXPECTED_CONCEPT_RELATIONS:
        raise FreezeError("Weighted-tag relation count changed")

    gate = require_mapping(weighted_tags.get("exit_gate"), "weighted-tags exit_gate")
    require_true_fields(
        gate,
        (
            "all_approved_chunks_present",
            "all_calibrated_weights_in_0_1",
            "all_chunks_have_valid_phase1_role",
            "all_reviewed_labels_present",
            "exactly_three_concept_rows_per_chunk",
            "full_model_provenance_present",
            "full_prototype_provenance_present",
            "human_labels_authoritative",
        ),
        description="Phase 12 exit gate",
    )

    provenance = require_mapping(
        weighted_tags.get("mapping_provenance"),
        "mapping_provenance",
    )
    if require_string(provenance.get("prototype_version"), "prototype_version") != (
        "phase1-prototype-v2"
    ):
        raise FreezeError("Frozen prototype version changed")
    return provenance


def validate_runtime_manifests(
    *,
    retrieval: Mapping[str, object],
    generation: Mapping[str, object],
    synthesis: Mapping[str, object],
    coverage: Mapping[str, object],
    final: Mapping[str, object],
    final_response: Mapping[str, object],
    corpus_version: str,
) -> dict[str, object]:
    expected = (
        (
            "retrieval",
            retrieval,
            "phase_14_build_retrieval_by_concept_and_domain",
            "evaluation_complete",
        ),
        (
            "generation",
            generation,
            "phase_15_build_domain_specific_generation",
            "domain_generation_complete",
        ),
        (
            "synthesis",
            synthesis,
            "phase_16_synthesis_and_tension_detection",
            "synthesis_complete",
        ),
        (
            "coverage",
            coverage,
            "phase_17_coverage_classification",
            "coverage_classification_complete",
        ),
        (
            "final",
            final,
            "phase_18_final_response_assembly",
            "final_response_complete",
        ),
    )
    for name, item, expected_phase, expected_status in expected:
        if require_string(item.get("phase"), f"{name} phase") != expected_phase:
            raise FreezeError(f"{name} phase mismatch")
        if require_string(item.get("status"), f"{name} status") != expected_status:
            raise FreezeError(f"{name} status mismatch")
        if require_string(item.get("corpus_version"), f"{name} corpus_version") != (corpus_version):
            raise FreezeError(f"{name} corpus version mismatch")

    final_validation = require_mapping(
        final_response.get("validation"),
        "final_response validation",
    )
    if final_validation.get("passed") is not True:
        raise FreezeError("Final response validation did not pass")

    versions = require_mapping(final_response.get("versions"), "final_response versions")
    generation_version = require_string(
        generation.get("generation_version"),
        "generation_version",
    )
    generation_prompt_version = require_string(
        generation.get("prompt_version"),
        "generation prompt_version",
    )
    synthesis_version = require_string(
        synthesis.get("synthesis_version"),
        "synthesis_version",
    )
    synthesis_prompt_version = require_string(
        synthesis.get("prompt_version"),
        "synthesis prompt_version",
    )
    coverage_version = require_string(
        coverage.get("coverage_version"),
        "coverage_version",
    )
    assembly_version = require_string(
        final_response.get("assembly_version"),
        "assembly_version",
    )

    expected_versions = {
        "corpus_version": corpus_version,
        "generation_version": generation_version,
        "generation_prompt_version": generation_prompt_version,
        "synthesis_version": synthesis_version,
        "synthesis_prompt_version": synthesis_prompt_version,
        "coverage_version": coverage_version,
    }
    for key, expected_value in expected_versions.items():
        if require_string(versions.get(key), f"final versions {key}") != expected_value:
            raise FreezeError(f"Final response version mismatch: {key}")

    return {
        **expected_versions,
        "assembly_version": assembly_version,
        "generation_provider": generation.get("provider"),
        "synthesis_provider": synthesis.get("provider"),
        "retrieval_configuration": retrieval.get(
            "retrieval_configuration",
            retrieval.get("configuration", retrieval.get("policy")),
        ),
        "coverage_policy": coverage.get(
            "coverage_policy",
            coverage.get("policy", coverage.get("thresholds")),
        ),
    }


def freeze_hashes(project_root: Path) -> dict[str, dict[str, object]]:
    frozen: dict[str, dict[str, object]] = {}
    for name, relative in FROZEN_ARTIFACTS.items():
        resolved = resolve(project_root, relative)
        require_file(resolved)
        frozen[name] = {
            "path": relative.as_posix(),
            "sha256": sha256_file(resolved),
            "size_bytes": resolved.stat().st_size,
        }
    return frozen


def run_phase20(
    *,
    project_root: Path,
    output_directory: Path,
    replace: bool,
) -> dict[str, object]:
    project_root = project_root.resolve()
    output_directory = resolve(project_root, output_directory)
    output_path = output_directory / DEFAULT_MANIFEST_NAME

    if output_path.exists() and not replace:
        raise FreezeError("Phase 20 manifest already exists. Use --replace.")

    LOGGER.info("Phase 20 starting: freeze_version=%s", SCRIPT_VERSION)

    phase19 = get_manifest(project_root, "phase19")
    activation = get_manifest(project_root, "activation")
    weighted_tags = get_manifest(project_root, "weighted_tags")
    embedding = get_manifest(project_root, "embedding")
    split = get_manifest(project_root, "split")
    retrieval = get_manifest(project_root, "retrieval")
    generation = get_manifest(project_root, "generation")
    synthesis = get_manifest(project_root, "synthesis")
    coverage = get_manifest(project_root, "coverage")
    final = get_manifest(project_root, "final")
    final_response = get_manifest(project_root, "final_response")

    validate_phase19(phase19)
    corpus_version, embedding_identity, database_step = validate_activation(activation)
    mapping_provenance = validate_weighted_tags(weighted_tags)

    if require_string(embedding.get("status"), "embedding manifest status") != "complete":
        raise FreezeError("Embedding manifest is not complete")

    split_gate = require_mapping(split.get("exit_gate"), "split exit_gate")
    for key in (
        "passage_family_leakage",
        "exact_duplicate_leakage",
        "high_overlap_leakage",
    ):
        if split_gate.get(key) is not False:
            raise FreezeError(f"Evaluation split leakage check failed: {key}")

    runtime_versions = validate_runtime_manifests(
        retrieval=retrieval,
        generation=generation,
        synthesis=synthesis,
        coverage=coverage,
        final=final,
        final_response=final_response,
        corpus_version=corpus_version,
    )

    frozen_hashes = freeze_hashes(project_root)
    phase19_path = resolve(project_root, MANIFEST_PATHS["phase19"])
    phase19_sha256 = sha256_file(phase19_path)

    freeze_payload: dict[str, object] = {
        "corpus_version": corpus_version,
        "concepts": list(EXPECTED_CONCEPTS),
        "domains": list(EXPECTED_DOMAINS),
        "active_chunk_count": EXPECTED_ACTIVE_CHUNKS,
        "reviewed_concept_relation_count": EXPECTED_CONCEPT_RELATIONS,
        "embedding_identity": embedding_identity,
        "mapping_provenance": mapping_provenance,
        "runtime_versions": runtime_versions,
        "phase19_manifest_sha256": phase19_sha256,
        "artifact_hashes": frozen_hashes,
    }
    freeze_fingerprint = canonical_sha256(freeze_payload)

    database_status = require_string(database_step.get("status"), "database_step status")
    known_limitations: list[dict[str, object]] = []
    if database_status != "complete":
        known_limitations.append(
            {
                "code": "phase13_database_upsert_pending",
                "status": database_status,
                "blocking_phase1_artifact_vertical_slice": False,
                "blocking_database_backed_production_activation": True,
                "reason": database_step.get("reason_pending"),
                "required_post_upsert_validation": database_step.get(
                    "required_post_upsert_validation"
                ),
            }
        )

    completion_manifest: dict[str, object] = {
        "phase": PHASE,
        "status": STATUS_COMPLETE,
        "script_version": SCRIPT_VERSION,
        "generated_at": utc_now(),
        "freeze_fingerprint_sha256": freeze_fingerprint,
        "baseline": freeze_payload,
        "source_manifests": {
            name: {
                "path": path.as_posix(),
                "sha256": sha256_file(resolve(project_root, path)),
            }
            for name, path in MANIFEST_PATHS.items()
        },
        "known_limitations": known_limitations,
        "freeze_policy": {
            "phase19_must_pass_before_freeze": True,
            "no_provider_calls_in_phase20": True,
            "no_embeddings_regenerated": True,
            "no_retrieval_rerun": True,
            "no_generation_rerun": True,
            "no_synthesis_rerun": True,
            "heldout_must_remain_read_only": True,
            "future_changes_require_new_version": True,
            "artifact_hash_change_invalidates_this_freeze": True,
        },
        "exit_gate": {
            "passed": True,
            "phase19_passed": True,
            "corpus_version_frozen": True,
            "embedding_identity_frozen": True,
            "prototype_and_mapping_frozen": True,
            "evaluation_splits_frozen": True,
            "retrieval_configuration_frozen": True,
            "generation_version_and_prompt_frozen": True,
            "synthesis_version_and_prompt_frozen": True,
            "coverage_policy_frozen": True,
            "final_response_contract_frozen": True,
            "artifact_hashes_recorded": True,
            "phase1_freeze_fingerprint_recorded": True,
        },
        "next_step": (
            "Phase 1 is frozen. Start future work from a new version or the next "
            "planned expansion phase; do not mutate this baseline in place."
        ),
    }

    atomic_json(output_path, completion_manifest)

    LOGGER.info("Phase 20 completion freeze complete")
    LOGGER.info(
        "corpus_version=%s active_chunks=%d concept_relations=%d",
        corpus_version,
        EXPECTED_ACTIVE_CHUNKS,
        EXPECTED_CONCEPT_RELATIONS,
    )
    LOGGER.info("freeze_fingerprint_sha256=%s", freeze_fingerprint)
    LOGGER.info("known_limitations=%d", len(known_limitations))
    LOGGER.info("Exit gate passed: True")
    LOGGER.info("Phase 1 completion manifest: %s", output_path)
    return completion_manifest


def main() -> int:
    arguments = parse_arguments()
    configure_logging(arguments.log_level)
    try:
        run_phase20(
            project_root=arguments.project_root,
            output_directory=arguments.output_directory,
            replace=arguments.replace,
        )
    except FreezeError:
        LOGGER.exception("Phase 20 freeze failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
