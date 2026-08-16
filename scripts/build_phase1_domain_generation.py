from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Final, TypeVar

from pydantic import ValidationError

from apps.api.models.runtime_contracts import (
    EvidencePackage,
    FrozenRuntimeContract,
    RetrievalManifest,
)
from apps.api.services.domain_generation import (
    DEFAULT_GROQ_MODEL,
    DEFAULT_MAX_COMPLETION_TOKENS,
    DEFAULT_MAX_PROVIDER_ATTEMPTS,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS,
    GENERATION_VERSION,
    PROMPT_VERSION,
    DomainGenerationService,
    GenerationError,
    ProviderConfig,
    validate_provider_config,
)

LOGGER = logging.getLogger("wth.phase1.build_domain_generation")

DEFAULT_EVIDENCE_PACKAGE: Final = Path("artifacts/phase1/retrieval/evidence_package.json")
DEFAULT_RETRIEVAL_MANIFEST: Final = Path("artifacts/phase1/retrieval/retrieval_manifest.json")
DEFAULT_OUTPUT_DIRECTORY: Final = Path("artifacts/phase1/generation")

ContractT = TypeVar(
    "ContractT",
    bound=FrozenRuntimeContract,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 15: generate three independently grounded domain "
            "responses from a Phase 14 evidence package using parallel "
            "Groq calls."
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
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(f"Groq model ID. Defaults to GROQ_MODEL or {DEFAULT_GROQ_MODEL}."),
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default=DEFAULT_REASONING_EFFORT,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=DEFAULT_MAX_COMPLETION_TOKENS,
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--max-provider-attempts",
        type=int,
        default=DEFAULT_MAX_PROVIDER_ATTEMPTS,
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing Phase 15 derived outputs.",
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
        raise GenerationError(f"Required file does not exist: {path}")


def require_mapping(
    value: object,
    description: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise GenerationError(f"{description} must be an object.")

    result: dict[str, object] = {}

    for key, nested in value.items():
        if not isinstance(key, str):
            raise GenerationError(f"{description} contains a non-string key.")
        result[key] = nested

    return result


def load_json(
    path: Path,
) -> dict[str, object]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GenerationError(f"Invalid JSON in {path}: {exc}") from exc

    return require_mapping(
        raw,
        f"JSON document {path}",
    )


def atomic_json(
    path: Path,
    value: Mapping[str, object],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = path.with_suffix(f"{path.suffix}.tmp")

    temp.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    temp.replace(path)


def output_paths(
    output_directory: Path,
) -> dict[str, Path]:
    return {
        "science": (output_directory / "science_response.json"),
        "advaita": (output_directory / "advaita_response.json"),
        "samkhya": (output_directory / "samkhya_response.json"),
        "combined": (output_directory / "domain_responses.json"),
        "manifest": (output_directory / "generation_manifest.json"),
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
        raise GenerationError(
            "Phase 15 outputs already exist. "
            "Use --replace: " + ", ".join(path.as_posix() for path in existing)
        )


def env_file_value(
    project_root: Path,
    key_name: str,
) -> str:
    candidates = (
        project_root / ".env",
        project_root / ".env.local",
        project_root / "apps" / "api" / ".env",
        project_root / "apps" / "api" / ".env.local",
    )

    for path in candidates:
        if not path.is_file():
            continue

        for line in path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines():
            stripped = line.strip()

            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue

            key, raw_value = stripped.split(
                "=",
                1,
            )

            if key.strip() != key_name:
                continue

            parsed = raw_value.strip().strip('"').strip("'")

            if parsed:
                return parsed

    return ""


def api_key_from_env(
    project_root: Path,
) -> str:
    value = os.getenv("GROQ_API_KEY")

    if value and value.strip():
        return value.strip()

    value = env_file_value(
        project_root,
        "GROQ_API_KEY",
    )

    if value:
        return value

    raise GenerationError("GROQ_API_KEY not found in environment or project .env files.")


def model_from_configuration(
    project_root: Path,
    explicit_model: str | None,
) -> str:
    if explicit_model and explicit_model.strip():
        return explicit_model.strip()

    env_value = os.getenv("GROQ_MODEL")

    if env_value and env_value.strip():
        return env_value.strip()

    env_value = env_file_value(
        project_root,
        "GROQ_MODEL",
    )

    if env_value:
        return env_value

    return DEFAULT_GROQ_MODEL


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
        raise GenerationError(
            f"{description} does not match the frozen Stage 3.0 runtime contract: {exc}"
        ) from exc


def run_phase15(
    *,
    project_root: Path,
    evidence_package_path: Path,
    retrieval_manifest_path: Path,
    output_directory: Path,
    provider_config: ProviderConfig,
    replace: bool,
) -> dict[str, object]:
    """Run Phase 15 as a file-I/O wrapper over DomainGenerationService."""

    # Preserve the legacy behavior: reject invalid provider configuration
    # before doing any artifact processing or provider calls.
    validate_provider_config(provider_config)

    project_root = project_root.resolve()

    evidence_package_path = resolve(
        project_root,
        evidence_package_path,
    )

    retrieval_manifest_path = resolve(
        project_root,
        retrieval_manifest_path,
    )

    output_directory = resolve(
        project_root,
        output_directory,
    )

    require_file(evidence_package_path)

    require_file(retrieval_manifest_path)

    paths = output_paths(output_directory)

    # Preserve the legacy behavior: do not call Groq if Phase 15 outputs
    # already exist and --replace was not supplied.
    ensure_replace_policy(
        paths=paths,
        replace=replace,
    )

    LOGGER.info(
        "Phase 15 starting: generation_version=%s model=%s",
        GENERATION_VERSION,
        provider_config.model,
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

    service = DomainGenerationService()

    result = service.generate(
        evidence_package=evidence_package,
        retrieval_manifest=retrieval_manifest,
        provider_config=provider_config,
        output_paths={
            "science": (paths["science"].as_posix()),
            "advaita": (paths["advaita"].as_posix()),
            "samkhya": (paths["samkhya"].as_posix()),
            "combined": (paths["combined"].as_posix()),
        },
        # Preserve legacy artifact-mode behavior: write outputs and manifest
        # first, then raise if the generated domain validation failed.
        raise_on_validation_failure=False,
    )

    domain_responses = result.domain_responses
    manifest = result.manifest

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for domain in (
        "science",
        "advaita",
        "samkhya",
    ):
        domain_response = domain_responses.domains[domain].model_dump(
            mode="python",
            by_alias=True,
        )

        atomic_json(
            paths[domain],
            domain_response,
        )

    combined_document = domain_responses.model_dump(
        mode="python",
        by_alias=True,
    )

    atomic_json(
        paths["combined"],
        combined_document,
    )

    manifest_document = manifest.model_dump(
        mode="python",
        by_alias=True,
    )

    atomic_json(
        paths["manifest"],
        manifest_document,
    )

    LOGGER.info("Phase 15 domain generation complete")

    LOGGER.info(
        "Parallel generation elapsed: %.2f ms",
        manifest.timing.parallel_generation_elapsed_ms,
    )

    for domain in (
        "science",
        "advaita",
        "samkhya",
    ):
        response = domain_responses.domains[domain]

        LOGGER.info(
            "%s claims=%d validation_passed=%s",
            domain,
            len(response.claims),
            response.validation.passed,
        )

    exit_gate_passed = manifest.exit_gate.independently_grounded_and_claim_cited

    LOGGER.info(
        "Exit gate passed: %s",
        exit_gate_passed,
    )

    LOGGER.info(
        "Generation manifest: %s",
        paths["manifest"],
    )

    if not exit_gate_passed:
        raise GenerationError(
            "Phase 15 generated outputs but failed "
            "grounding/domain-leakage validation. "
            "Inspect generation_manifest.json and "
            "per-domain validation issues."
        )

    return manifest_document


def main() -> int:
    arguments = parse_arguments()

    configure_logging(arguments.log_level)

    project_root = arguments.project_root.resolve()

    provider_config = ProviderConfig(
        api_key=api_key_from_env(project_root),
        model=model_from_configuration(
            project_root,
            arguments.model,
        ),
        reasoning_effort=(arguments.reasoning_effort),
        temperature=(arguments.temperature),
        max_completion_tokens=(arguments.max_completion_tokens),
        timeout_seconds=(arguments.timeout_seconds),
        max_attempts=(arguments.max_provider_attempts),
    )

    try:
        run_phase15(
            project_root=project_root,
            evidence_package_path=(arguments.evidence_package),
            retrieval_manifest_path=(arguments.retrieval_manifest),
            output_directory=(arguments.output_directory),
            provider_config=(provider_config),
            replace=arguments.replace,
        )
    except GenerationError:
        LOGGER.exception("Phase 15 generation failed")
        return 1

    return 0


__all__ = [
    "DEFAULT_EVIDENCE_PACKAGE",
    "DEFAULT_OUTPUT_DIRECTORY",
    "DEFAULT_RETRIEVAL_MANIFEST",
    "GENERATION_VERSION",
    "PROMPT_VERSION",
    "GenerationError",
    "ProviderConfig",
    "api_key_from_env",
    "model_from_configuration",
    "run_phase15",
]


if __name__ == "__main__":
    raise SystemExit(main())
