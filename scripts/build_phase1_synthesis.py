from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Final, TypeVar

from pydantic import ValidationError

from apps.api.models.runtime_contracts import (
    DomainResponses,
    FrozenRuntimeContract,
    GenerationManifest,
)
from apps.api.services.synthesis import (
    DEFAULT_MAX_COMPLETION_TOKENS,
    DEFAULT_MAX_PROVIDER_ATTEMPTS,
    DEFAULT_SYNTHESIS_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS,
    PROMPT_VERSION,
    SYNTHESIS_VERSION,
    SynthesisError,
    SynthesisProviderConfig,
    SynthesisService,
)

LOGGER = logging.getLogger("wth.phase1.build_synthesis")

SCRIPT_VERSION: Final = "1.1.0"

DEFAULT_DOMAIN_RESPONSES: Final = Path(
    "artifacts/phase1/generation/domain_responses.json"
)
DEFAULT_GENERATION_MANIFEST: Final = Path(
    "artifacts/phase1/generation/generation_manifest.json"
)
DEFAULT_OUTPUT_DIRECTORY: Final = Path(
    "artifacts/phase1/synthesis"
)

ContractT = TypeVar(
    "ContractT",
    bound=FrozenRuntimeContract,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 16: compare validated Science, Advaita Vedanta, "
            "and Samkhya domain claims without collapsing domain "
            "differences."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
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
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Groq synthesis model. Defaults to PHASE16_GROQ_MODEL "
            f"or {DEFAULT_SYNTHESIS_MODEL}."
        ),
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
        help="Replace existing Phase 16 derived outputs.",
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

    return (
        project_root / path
    ).resolve()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise SynthesisError(
            f"Required file does not exist: {path}"
        )


def require_mapping(
    value: object,
    description: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise SynthesisError(
            f"{description} must be an object."
        )

    result: dict[str, object] = {}

    for key, nested in value.items():
        if not isinstance(key, str):
            raise SynthesisError(
                f"{description} contains a non-string key."
            )
        result[key] = nested

    return result


def load_json(
    path: Path,
) -> dict[str, object]:
    try:
        raw: object = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as exc:
        raise SynthesisError(
            f"Invalid JSON in {path}: {exc}"
        ) from exc

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

    temp = path.with_suffix(
        f"{path.suffix}.tmp"
    )

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
        "synthesis": (
            output_directory / "synthesis.json"
        ),
        "manifest": (
            output_directory / "synthesis_manifest.json"
        ),
    }


def ensure_replace_policy(
    *,
    paths: Mapping[str, Path],
    replace: bool,
) -> None:
    if replace:
        return

    existing = [
        path
        for path in paths.values()
        if path.exists()
    ]

    if existing:
        raise SynthesisError(
            "Phase 16 outputs already exist. "
            "Use --replace: "
            + ", ".join(
                path.as_posix()
                for path in existing
            )
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

            if (
                not stripped
                or stripped.startswith("#")
                or "=" not in stripped
            ):
                continue

            key, raw_value = stripped.split(
                "=",
                1,
            )

            if key.strip() != key_name:
                continue

            parsed = (
                raw_value.strip()
                .strip('"')
                .strip("'")
            )

            if parsed:
                return parsed

    return ""


def api_key_from_env(
    project_root: Path,
) -> str:
    value = os.getenv(
        "GROQ_API_KEY"
    )

    if value and value.strip():
        return value.strip()

    value = env_file_value(
        project_root,
        "GROQ_API_KEY",
    )

    if value:
        return value

    raise SynthesisError(
        "GROQ_API_KEY not found in environment "
        "or project .env files."
    )


def synthesis_model_from_configuration(
    project_root: Path,
    explicit_model: str | None,
) -> str:
    if (
        explicit_model
        and explicit_model.strip()
    ):
        return explicit_model.strip()

    env_value = os.getenv(
        "PHASE16_GROQ_MODEL"
    )

    if (
        env_value
        and env_value.strip()
    ):
        return env_value.strip()

    env_value = env_file_value(
        project_root,
        "PHASE16_GROQ_MODEL",
    )

    if env_value:
        return env_value

    return DEFAULT_SYNTHESIS_MODEL


def load_contract(
    path: Path,
    model_type: type[ContractT],
    *,
    description: str,
) -> ContractT:
    document = load_json(path)

    try:
        return model_type.model_validate(
            document
        )
    except ValidationError as exc:
        raise SynthesisError(
            f"{description} does not match the frozen "
            f"Stage 3.0 runtime contract: {exc}"
        ) from exc


def run_phase16(
    *,
    project_root: Path,
    domain_responses_path: Path,
    generation_manifest_path: Path,
    output_directory: Path,
    provider_config: SynthesisProviderConfig,
    replace: bool,
) -> dict[str, object]:
    """Run Phase 16 as a file-I/O wrapper over SynthesisService."""

    project_root = project_root.resolve()

    domain_responses_path = resolve(
        project_root,
        domain_responses_path,
    )

    generation_manifest_path = resolve(
        project_root,
        generation_manifest_path,
    )

    output_directory = resolve(
        project_root,
        output_directory,
    )

    require_file(
        domain_responses_path
    )
    require_file(
        generation_manifest_path
    )

    paths = output_paths(
        output_directory
    )

    # Preserve the legacy behavior: do not make a provider call if outputs
    # already exist and --replace was not supplied.
    ensure_replace_policy(
        paths=paths,
        replace=replace,
    )

    LOGGER.info(
        "Phase 16 starting: synthesis_version=%s model=%s",
        SYNTHESIS_VERSION,
        provider_config.model,
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

    service = SynthesisService()

    result = asyncio.run(
        service.synthesize(
            domain_responses=domain_responses,
            generation_manifest=generation_manifest,
            provider_config=provider_config,
            synthesis_output_path=(
                paths["synthesis"].as_posix()
            ),
            # Preserve the legacy script behavior: write the synthesis and
            # manifest first, then raise if the exit gate failed.
            raise_on_validation_failure=False,
        )
    )

    synthesis_document = (
        result.synthesis.model_dump(
            mode="python",
            by_alias=True,
        )
    )

    manifest_document = (
        result.manifest.model_dump(
            mode="python",
            by_alias=True,
        )
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    atomic_json(
        paths["synthesis"],
        synthesis_document,
    )

    atomic_json(
        paths["manifest"],
        manifest_document,
    )

    manifest = result.manifest

    LOGGER.info(
        "Phase 16 synthesis complete"
    )

    LOGGER.info(
        "Synthesis generation elapsed: %.2f ms",
        manifest.timing.synthesis_generation_elapsed_ms,
    )

    LOGGER.info(
        (
            "pairwise_comparisons=%d tensions=%d "
            "non_equivalences=%d insufficient_coverage=%d"
        ),
        manifest.counts.comparison_count,
        manifest.counts.direct_tension_count,
        manifest.counts.non_equivalence_count,
        manifest.counts.insufficient_coverage_count,
    )

    validation_passed = (
        result.synthesis.validation.passed
    )

    LOGGER.info(
        "Exit gate passed: %s",
        validation_passed,
    )

    LOGGER.info(
        "Synthesis manifest: %s",
        paths["manifest"],
    )

    if not validation_passed:
        raise SynthesisError(
            "Phase 16 generated synthesis but failed comparison "
            "validation. Inspect synthesis.json validation issues."
        )

    return manifest_document


def main() -> int:
    arguments = parse_arguments()

    configure_logging(
        arguments.log_level
    )

    project_root = (
        arguments.project_root.resolve()
    )

    provider_config = (
        SynthesisProviderConfig(
            api_key=api_key_from_env(
                project_root
            ),
            model=synthesis_model_from_configuration(
                project_root,
                arguments.model,
            ),
            temperature=arguments.temperature,
            max_completion_tokens=(
                arguments.max_completion_tokens
            ),
            timeout_seconds=(
                arguments.timeout_seconds
            ),
            max_attempts=(
                arguments.max_provider_attempts
            ),
        )
    )

    try:
        run_phase16(
            project_root=project_root,
            domain_responses_path=(
                arguments.domain_responses
            ),
            generation_manifest_path=(
                arguments.generation_manifest
            ),
            output_directory=(
                arguments.output_directory
            ),
            provider_config=provider_config,
            replace=arguments.replace,
        )
    except SynthesisError:
        LOGGER.exception(
            "Phase 16 synthesis failed"
        )
        return 1

    return 0


__all__ = [
    "DEFAULT_DOMAIN_RESPONSES",
    "DEFAULT_GENERATION_MANIFEST",
    "PROMPT_VERSION",
    "SYNTHESIS_VERSION",
    "SynthesisError",
    "SynthesisProviderConfig",
    "api_key_from_env",
    "run_phase16",
    "synthesis_model_from_configuration",
]


if __name__ == "__main__":
    raise SystemExit(main())
