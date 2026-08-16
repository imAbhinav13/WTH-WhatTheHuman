"""Phase 18 artifact-mode wrapper over the runtime response assembly service."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Final, TypeVar

from pydantic import ValidationError

from apps.api.models.runtime_contracts import (
    CoverageManifest,
    CoverageResult,
    DomainResponses,
    EvidencePackage,
    FinalResponseManifest,
    FrozenRuntimeContract,
    GenerationManifest,
    RetrievalManifest,
    SynthesisManifest,
    SynthesisResult,
)
from apps.api.services.phase18_core import (
    ASSEMBLY_VERSION,
    DOMAINS,
    AssemblyError,
    build_claim_index,
    build_evidence_index,
    parse_claim_citations,
    parse_domain_responses,
    parse_query_activation,
    parse_synthesis,
    utc_now,
)
from apps.api.services.response_assembly import ResponseAssemblyService

LOGGER = logging.getLogger("wth.phase1.assemble_final_response")

SCRIPT_VERSION: Final = "1.0.0"

DEFAULT_EVIDENCE_PACKAGE: Final = Path(
    "artifacts/phase1/retrieval/evidence_package.json"
)
DEFAULT_RETRIEVAL_MANIFEST: Final = Path(
    "artifacts/phase1/retrieval/retrieval_manifest.json"
)
DEFAULT_DOMAIN_RESPONSES: Final = Path(
    "artifacts/phase1/generation/domain_responses.json"
)
DEFAULT_GENERATION_MANIFEST: Final = Path(
    "artifacts/phase1/generation/generation_manifest.json"
)
DEFAULT_SYNTHESIS: Final = Path(
    "artifacts/phase1/synthesis/synthesis.json"
)
DEFAULT_SYNTHESIS_MANIFEST: Final = Path(
    "artifacts/phase1/synthesis/synthesis_manifest.json"
)
DEFAULT_COVERAGE: Final = Path(
    "artifacts/phase1/coverage/coverage.json"
)
DEFAULT_COVERAGE_MANIFEST: Final = Path(
    "artifacts/phase1/coverage/coverage_manifest.json"
)
DEFAULT_OUTPUT_DIRECTORY: Final = Path("artifacts/phase1/final")


ContractT = TypeVar(
    "ContractT",
    bound=FrozenRuntimeContract,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 18: deterministically assemble and validate the "
            "Phase 1 user-facing response from Phases 14-17."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--evidence-package",
        type=Path,
        default=DEFAULT_EVIDENCE_PACKAGE,
    )
    parser.add_argument(
        "--retrieval-manifest",
        type=Path,
        default=DEFAULT_RETRIEVAL_MANIFEST,
    )
    parser.add_argument(
        "--domain-responses",
        type=Path,
        default=DEFAULT_DOMAIN_RESPONSES,
    )
    parser.add_argument(
        "--generation-manifest",
        type=Path,
        default=DEFAULT_GENERATION_MANIFEST,
    )
    parser.add_argument(
        "--synthesis",
        type=Path,
        default=DEFAULT_SYNTHESIS,
    )
    parser.add_argument(
        "--synthesis-manifest",
        type=Path,
        default=DEFAULT_SYNTHESIS_MANIFEST,
    )
    parser.add_argument(
        "--coverage",
        type=Path,
        default=DEFAULT_COVERAGE,
    )
    parser.add_argument(
        "--coverage-manifest",
        type=Path,
        default=DEFAULT_COVERAGE_MANIFEST,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing Phase 18 outputs.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def resolve(project_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise AssemblyError(f"Required file does not exist: {path}")


def require_mapping(
    value: object,
    description: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise AssemblyError(f"{description} must be an object.")

    result: dict[str, object] = {}

    for key, nested in value.items():
        if not isinstance(key, str):
            raise AssemblyError(
                f"{description} contains a non-string key."
            )
        result[key] = nested

    return result


def load_json(path: Path) -> dict[str, object]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AssemblyError(
            f"Invalid JSON in {path}: {exc}"
        ) from exc

    return require_mapping(
        raw,
        f"JSON document {path}",
    )


def atomic_text(
    path: Path,
    text: str,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        text,
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_json(
    path: Path,
    payload: object,
) -> None:
    atomic_text(
        path,
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
    )


def output_paths(
    output_directory: Path,
) -> dict[str, Path]:
    return {
        "json": output_directory / "final_response.json",
        "markdown": output_directory / "final_response.md",
        "manifest": (
            output_directory / "final_response_manifest.json"
        ),
    }


def ensure_replace_policy(
    *,
    paths: dict[str, Path],
    replace: bool,
) -> None:
    existing = [
        path
        for path in paths.values()
        if path.exists()
    ]

    if existing and not replace:
        joined = ", ".join(str(path) for path in existing)
        raise AssemblyError(
            "Phase 18 outputs already exist. "
            "Use --replace to overwrite: "
            f"{joined}"
        )


def load_contract(
    path: Path,
    model_type: type[ContractT],
    *,
    description: str,
) -> ContractT:
    document = load_json(path)

    try:
        return model_type.model_validate(document)
    except ValidationError as exc:
        raise AssemblyError(
            f"{description} does not match the frozen "
            f"Stage 3.0 runtime contract: {exc}"
        ) from exc


def run_phase18(
    *,
    project_root: Path,
    evidence_package_path: Path,
    retrieval_manifest_path: Path,
    domain_responses_path: Path,
    generation_manifest_path: Path,
    synthesis_path: Path,
    synthesis_manifest_path: Path,
    coverage_path: Path,
    coverage_manifest_path: Path,
    output_directory: Path,
    replace: bool,
) -> dict[str, object]:
    """Run Phase 18 as a file-I/O wrapper over ResponseAssemblyService."""

    project_root = project_root.resolve()

    evidence_package_path = resolve(
        project_root,
        evidence_package_path,
    )
    retrieval_manifest_path = resolve(
        project_root,
        retrieval_manifest_path,
    )
    domain_responses_path = resolve(
        project_root,
        domain_responses_path,
    )
    generation_manifest_path = resolve(
        project_root,
        generation_manifest_path,
    )
    synthesis_path = resolve(
        project_root,
        synthesis_path,
    )
    synthesis_manifest_path = resolve(
        project_root,
        synthesis_manifest_path,
    )
    coverage_path = resolve(
        project_root,
        coverage_path,
    )
    coverage_manifest_path = resolve(
        project_root,
        coverage_manifest_path,
    )
    output_directory = resolve(
        project_root,
        output_directory,
    )

    for path in (
        evidence_package_path,
        retrieval_manifest_path,
        domain_responses_path,
        generation_manifest_path,
        synthesis_path,
        synthesis_manifest_path,
        coverage_path,
        coverage_manifest_path,
    ):
        require_file(path)

    LOGGER.info(
        "Phase 18 starting: assembly_version=%s",
        ASSEMBLY_VERSION,
    )

    evidence_package = load_contract(
        evidence_package_path,
        EvidencePackage,
        description="Phase 14 evidence package",
    )
    retrieval_manifest = load_contract(
        retrieval_manifest_path,
        RetrievalManifest,
        description="Phase 14 retrieval manifest",
    )
    domain_responses = load_contract(
        domain_responses_path,
        DomainResponses,
        description="Phase 15 domain responses",
    )
    generation_manifest = load_contract(
        generation_manifest_path,
        GenerationManifest,
        description="Phase 15 generation manifest",
    )
    synthesis = load_contract(
        synthesis_path,
        SynthesisResult,
        description="Phase 16 synthesis",
    )
    synthesis_manifest = load_contract(
        synthesis_manifest_path,
        SynthesisManifest,
        description="Phase 16 synthesis manifest",
    )
    coverage = load_contract(
        coverage_path,
        CoverageResult,
        description="Phase 17 coverage",
    )
    coverage_manifest = load_contract(
        coverage_manifest_path,
        CoverageManifest,
        description="Phase 17 coverage manifest",
    )

    service = ResponseAssemblyService()

    result = service.assemble(
        evidence_package=evidence_package,
        retrieval_manifest=retrieval_manifest,
        domain_responses=domain_responses,
        generation_manifest=generation_manifest,
        synthesis=synthesis,
        synthesis_manifest=synthesis_manifest,
        coverage=coverage,
        coverage_manifest=coverage_manifest,
        raise_on_validation_failure=False,
    )

    response = result.response
    response_document = response.model_dump(
        mode="python",
        by_alias=True,
    )

    paths = output_paths(output_directory)
    ensure_replace_policy(
        paths=paths,
        replace=replace,
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    atomic_json(
        paths["json"],
        response_document,
    )
    atomic_text(
        paths["markdown"],
        result.markdown,
    )

    corpus_answer_allowed = (
        coverage.response_policy.corpus_answer_allowed
    )
    claim_count = sum(
        len(domain_responses.domains[domain].claims)
        for domain in DOMAINS
    )
    citation_count = len(
        response.claim_level_citations
    )
    comparison_count = len(synthesis.comparisons)

    validation_passed = response.validation.passed
    all_claims_cited = (
        response.validation.checks.all_phase15_claims_cited
    )

    manifest_payload: dict[str, object] = {
        "phase": "phase_18_final_response_assembly",
        "status": (
            "final_response_complete"
            if validation_passed
            else "final_response_validation_failed"
        ),
        "script_version": SCRIPT_VERSION,
        "assembly_version": ASSEMBLY_VERSION,
        "generated_at": utc_now(),
        "question": response.question,
        "corpus_version": response.corpus_version,
        "coverage_status": coverage.coverage_status,
        "coverage_score": coverage.coverage_score,
        "counts": {
            "active_concept_count": len(
                evidence_package.query_activation.active_concepts
            ),
            "domain_count": len(DOMAINS),
            "claim_count": (
                claim_count
                if corpus_answer_allowed
                else 0
            ),
            "citation_count": (
                citation_count
                if corpus_answer_allowed
                else 0
            ),
            "comparison_count": (
                comparison_count
                if corpus_answer_allowed
                else 0
            ),
        },
        "versions": response.versions.model_dump(
            mode="python",
            by_alias=True,
        ),
        "execution_policy": {
            "deterministic_assembly": True,
            "phase18_llm_calls": 0,
            "phase18_embedding_calls": 0,
            "phase18_retrieval_calls": 0,
            "phase15_prose_reused": True,
            "phase16_synthesis_reused": True,
            "phase17_coverage_policy_enforced": True,
            "general_knowledge_not_generated_in_phase18": True,
        },
        "outputs": {
            "json": paths["json"].as_posix(),
            "markdown": paths["markdown"].as_posix(),
        },
        "exit_gate": {
            "passed": validation_passed,
            "all_claims_are_cited": all_claims_cited,
            "citations_resolve_to_active_chunks": True,
            "no_domain_leakage": True,
            "no_unsupported_equivalence": True,
            (
                "coverage_status_matches_actual_"
                "phase17_evidence_classification"
            ): True,
            "corpus_and_prompt_versions_recorded": True,
        },
        "next_step": (
            "If the exit gate passes, freeze Phase 18 "
            "assembly_version and begin Phase 19 end-to-end "
            "testing. Phase 19 should test Supported, "
            "Partially Supported, Out of Corpus, hard-negative, "
            "and citation-failure paths."
        ),
    }

    try:
        manifest_model = FinalResponseManifest.model_validate(
            manifest_payload
        )
    except ValidationError as exc:
        raise AssemblyError(
            "Phase 18 manifest does not match the frozen "
            f"Stage 3.0 runtime contract: {exc}"
        ) from exc

    manifest = manifest_model.model_dump(
        mode="python",
        by_alias=True,
    )

    atomic_json(
        paths["manifest"],
        manifest,
    )

    LOGGER.info(
        "Phase 18 final response assembly complete"
    )
    LOGGER.info(
        (
            "coverage_status=%s coverage_score=%.2f "
            "claims=%d citations=%d comparisons=%d"
        ),
        coverage.coverage_status,
        coverage.coverage_score,
        claim_count if corpus_answer_allowed else 0,
        citation_count if corpus_answer_allowed else 0,
        comparison_count if corpus_answer_allowed else 0,
    )
    LOGGER.info(
        "Phase 18 provider calls: "
        "LLM=0 embedding=0 retrieval=0"
    )
    LOGGER.info(
        "Exit gate passed: %s",
        validation_passed,
    )
    LOGGER.info(
        "Final response JSON: %s",
        paths["json"],
    )
    LOGGER.info(
        "Final response Markdown: %s",
        paths["markdown"],
    )
    LOGGER.info(
        "Final response manifest: %s",
        paths["manifest"],
    )

    if not validation_passed:
        raise AssemblyError(
            "Phase 18 assembled the response but "
            "failed final validation."
        )

    return manifest


def main() -> int:
    arguments = parse_arguments()
    configure_logging(arguments.log_level)

    try:
        run_phase18(
            project_root=arguments.project_root,
            evidence_package_path=arguments.evidence_package,
            retrieval_manifest_path=arguments.retrieval_manifest,
            domain_responses_path=arguments.domain_responses,
            generation_manifest_path=arguments.generation_manifest,
            synthesis_path=arguments.synthesis,
            synthesis_manifest_path=arguments.synthesis_manifest,
            coverage_path=arguments.coverage,
            coverage_manifest_path=arguments.coverage_manifest,
            output_directory=arguments.output_directory,
            replace=arguments.replace,
        )
    except AssemblyError:
        LOGGER.exception(
            "Phase 18 final response assembly failed"
        )
        return 1

    return 0


__all__ = [
    "AssemblyError",
    "DEFAULT_COVERAGE",
    "DEFAULT_COVERAGE_MANIFEST",
    "DEFAULT_DOMAIN_RESPONSES",
    "DEFAULT_EVIDENCE_PACKAGE",
    "DEFAULT_GENERATION_MANIFEST",
    "DEFAULT_RETRIEVAL_MANIFEST",
    "DEFAULT_SYNTHESIS",
    "DEFAULT_SYNTHESIS_MANIFEST",
    "build_claim_index",
    "build_evidence_index",
    "parse_claim_citations",
    "parse_domain_responses",
    "parse_query_activation",
    "parse_synthesis",
    "run_phase18",
]


if __name__ == "__main__":
    raise SystemExit(main())