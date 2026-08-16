from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Final, TypeVar

from pydantic import ValidationError

from apps.api.models.runtime_contracts import (
    DomainResponses,
    EvidencePackage,
    FrozenRuntimeContract,
    GenerationManifest,
    RetrievalManifest,
    SynthesisManifest,
    SynthesisResult,
)
from apps.api.services.coverage import (
    COVERAGE_VERSION,
    CoverageError,
    CoverageService,
)

LOGGER = logging.getLogger("wth.phase1.classify_coverage")

SCRIPT_VERSION: Final = "1.1.0"

DEFAULT_EVIDENCE_PACKAGE: Final = Path("artifacts/phase1/retrieval/evidence_package.json")
DEFAULT_RETRIEVAL_MANIFEST: Final = Path("artifacts/phase1/retrieval/retrieval_manifest.json")
DEFAULT_DOMAIN_RESPONSES: Final = Path("artifacts/phase1/generation/domain_responses.json")
DEFAULT_GENERATION_MANIFEST: Final = Path("artifacts/phase1/generation/generation_manifest.json")
DEFAULT_SYNTHESIS: Final = Path("artifacts/phase1/synthesis/synthesis.json")
DEFAULT_SYNTHESIS_MANIFEST: Final = Path("artifacts/phase1/synthesis/synthesis_manifest.json")
DEFAULT_OUTPUT_DIRECTORY: Final = Path("artifacts/phase1/coverage")

ContractT = TypeVar(
    "ContractT",
    bound=FrozenRuntimeContract,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 17: deterministically classify reviewed-corpus "
            "coverage from validated Phase 14-16 artifacts."
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
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing Phase 17 outputs.",
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


def resolve(
    project_root: Path,
    path: Path,
) -> Path:
    if path.is_absolute():
        return path.resolve()

    return (project_root / path).resolve()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise CoverageError(f"Required file does not exist: {path}")


def require_mapping(
    value: object,
    description: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise CoverageError(f"{description} must be an object.")

    result: dict[str, object] = {}

    for key, nested in value.items():
        if not isinstance(key, str):
            raise CoverageError(f"{description} contains a non-string key.")
        result[key] = nested

    return result


def load_json(
    path: Path,
) -> dict[str, object]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CoverageError(f"Invalid JSON in {path}: {exc}") from exc

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
        "coverage": (output_directory / "coverage.json"),
        "manifest": (output_directory / "coverage_manifest.json"),
    }


def ensure_replace_policy(
    *,
    paths: Mapping[str, Path],
    replace: bool,
) -> None:
    if replace:
        return

    existing = [path for path in paths.values() if path.exists()]

    if existing:
        raise CoverageError(
            "Phase 17 outputs already exist. "
            "Use --replace: " + ", ".join(path.as_posix() for path in existing)
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
        raise CoverageError(
            f"{description} does not match the frozen Stage 3.0 runtime contract: {exc}"
        ) from exc


def run_phase17(
    *,
    project_root: Path,
    evidence_package_path: Path,
    retrieval_manifest_path: Path,
    domain_responses_path: Path,
    generation_manifest_path: Path,
    synthesis_path: Path,
    synthesis_manifest_path: Path,
    output_directory: Path,
    replace: bool,
) -> dict[str, object]:
    """Run Phase 17 as a file-I/O wrapper over CoverageService."""

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
    ):
        require_file(path)

    LOGGER.info(
        "Phase 17 starting: coverage_version=%s",
        COVERAGE_VERSION,
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

    paths = output_paths(output_directory)

    ensure_replace_policy(
        paths=paths,
        replace=replace,
    )

    service = CoverageService()

    result = service.classify(
        evidence_package=evidence_package,
        retrieval_manifest=retrieval_manifest,
        domain_responses=domain_responses,
        generation_manifest=generation_manifest,
        synthesis=synthesis,
        synthesis_manifest=synthesis_manifest,
        coverage_output_path=(paths["coverage"].as_posix()),
        raise_on_exit_gate_failure=False,
    )

    coverage_document = result.coverage.model_dump(
        mode="python",
        by_alias=True,
    )

    manifest_document = result.manifest.model_dump(
        mode="python",
        by_alias=True,
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    atomic_json(
        paths["coverage"],
        coverage_document,
    )

    atomic_json(
        paths["manifest"],
        manifest_document,
    )

    coverage = result.coverage
    manifest = result.manifest

    LOGGER.info("Phase 17 coverage classification complete")
    LOGGER.info(
        (
            "coverage_status=%s coverage_score=%.2f "
            "supported_concepts=%d "
            "partially_supported_concepts=%d "
            "unsupported_concepts=%d "
            "covered_domains=%d missing_domains=%d"
        ),
        coverage.coverage_status,
        coverage.coverage_score,
        len(coverage.supported_concepts),
        len(coverage.partially_supported_concepts),
        len(coverage.unsupported_concepts),
        len(coverage.covered_domains),
        len(coverage.missing_domains),
    )
    LOGGER.info(
        "Exit gate passed: %s",
        manifest.exit_gate.passed,
    )
    LOGGER.info(
        "Coverage output: %s",
        paths["coverage"],
    )
    LOGGER.info(
        "Coverage manifest: %s",
        paths["manifest"],
    )

    if not manifest.exit_gate.passed:
        raise CoverageError("Phase 17 failed its anti-fabrication exit gate.")

    return manifest_document


def main() -> int:
    arguments = parse_arguments()

    configure_logging(arguments.log_level)

    try:
        run_phase17(
            project_root=arguments.project_root,
            evidence_package_path=(arguments.evidence_package),
            retrieval_manifest_path=(arguments.retrieval_manifest),
            domain_responses_path=(arguments.domain_responses),
            generation_manifest_path=(arguments.generation_manifest),
            synthesis_path=(arguments.synthesis),
            synthesis_manifest_path=(arguments.synthesis_manifest),
            output_directory=(arguments.output_directory),
            replace=arguments.replace,
        )
    except CoverageError:
        LOGGER.exception("Phase 17 coverage classification failed")
        return 1

    return 0


__all__ = [
    "DEFAULT_DOMAIN_RESPONSES",
    "DEFAULT_EVIDENCE_PACKAGE",
    "DEFAULT_GENERATION_MANIFEST",
    "DEFAULT_RETRIEVAL_MANIFEST",
    "DEFAULT_SYNTHESIS",
    "DEFAULT_SYNTHESIS_MANIFEST",
    "CoverageError",
    "run_phase17",
]


if __name__ == "__main__":
    raise SystemExit(main())
