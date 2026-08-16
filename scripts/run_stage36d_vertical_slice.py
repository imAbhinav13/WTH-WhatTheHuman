from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from collections.abc import Mapping
from pathlib import Path

from apps.api.clients.supabase_runtime import (
    get_supabase_runtime_client,
)
from apps.api.repositories.concept_repository import (
    ConceptRepository,
)
from apps.api.repositories.retrieval_repository import (
    RetrievalRepository,
)
from apps.api.services.coverage import (
    CoverageService,
)
from apps.api.services.domain_generation import (
    DEFAULT_GROQ_MODEL,
    DEFAULT_MAX_COMPLETION_TOKENS,
    DEFAULT_MAX_PROVIDER_ATTEMPTS,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS,
    DomainGenerationService,
    ProviderConfig,
)
from apps.api.services.query_orchestrator import (
    QueryOrchestrator,
    QueryPipelineError,
    QueryPipelineProviderConfig,
    QueryPipelineServices,
)
from apps.api.services.response_assembly import (
    ResponseAssemblyService,
)
from apps.api.services.retrieval import (
    QueryEmbeddingConfig,
    RetrievalService,
)
from apps.api.services.synthesis import (
    DEFAULT_MAX_COMPLETION_TOKENS as SYNTHESIS_MAX_COMPLETION_TOKENS,
)
from apps.api.services.synthesis import (
    DEFAULT_MAX_PROVIDER_ATTEMPTS as SYNTHESIS_MAX_PROVIDER_ATTEMPTS,
)
from apps.api.services.synthesis import (
    DEFAULT_SYNTHESIS_MODEL,
    SynthesisProviderConfig,
    SynthesisService,
)
from apps.api.services.synthesis import (
    DEFAULT_TEMPERATURE as SYNTHESIS_TEMPERATURE,
)
from apps.api.services.synthesis import (
    DEFAULT_TIMEOUT_SECONDS as SYNTHESIS_TIMEOUT_SECONDS,
)

LOGGER = logging.getLogger("wth.stage3.6d.vertical_slice")

DEFAULT_QUESTION = "What constitutes the self or personal identity?"


class VerticalSliceConfigurationError(RuntimeError):
    """Raised before the live query starts when credentials are incomplete."""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run the Stage 3.6D live WTH Phase 14-18 vertical slice entirely in memory.")
    )

    parser.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
    )

    parser.add_argument(
        "--show-markdown",
        action="store_true",
        help=(
            "Print the complete deterministic Phase 18 Markdown "
            "after the compact smoke-test summary."
        ),
    )

    parser.add_argument(
        "--log-level",
        choices=(
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
        ),
        default="INFO",
    )

    return parser.parse_args()


def configure_logging(
    level: str,
) -> None:
    logging.basicConfig(
        level=getattr(
            logging,
            level,
        ),
        format=("%(asctime)s %(levelname)s %(name)s %(message)s"),
    )


def load_environment_file(
    path: Path,
) -> None:
    """Startup-only environment loading.

    This function is deliberately outside QueryOrchestrator and is completed
    before ``execute()`` starts. It never reads or writes Phase 14-18 artifacts.
    """

    if not path.is_file():
        return

    for raw_line in path.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, raw_value = line.split(
            "=",
            1,
        )

        key = key.strip()
        value = raw_value.strip()

        if not key or key in os.environ:
            continue

        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0]
            in {
                "'",
                '"',
            }
        ):
            value = value[1:-1]

        os.environ[key] = value


def load_startup_environment() -> None:
    project_root = Path.cwd().resolve()

    for path in (
        project_root / ".env",
        project_root / ".env.local",
        project_root / "apps" / "api" / ".env",
        project_root / "apps" / "api" / ".env.local",
    ):
        load_environment_file(path)

    # Shared Settings currently requires GOOGLE_API_KEY in live provider mode.
    # The embedding path accepts the historical GEMINI_API_KEY alias too, so
    # normalize it once during process composition if needed.
    google_key = os.getenv(
        "GOOGLE_API_KEY",
        "",
    ).strip()

    gemini_key = os.getenv(
        "GEMINI_API_KEY",
        "",
    ).strip()

    if not google_key and gemini_key:
        os.environ["GOOGLE_API_KEY"] = gemini_key


def required_environment(
    name: str,
) -> str:
    value = os.getenv(
        name,
        "",
    ).strip()

    if not value:
        raise VerticalSliceConfigurationError(f"{name} is not configured.")

    return value


def optional_environment(
    name: str,
    default: str,
) -> str:
    value = os.getenv(
        name,
        "",
    ).strip()

    return value or default


def build_orchestrator() -> QueryOrchestrator:
    """Compose runtime dependencies before the request starts."""

    load_startup_environment()

    google_api_key = required_environment("GOOGLE_API_KEY")

    groq_api_key = required_environment("GROQ_API_KEY")

    client = get_supabase_runtime_client()

    retrieval_service = RetrievalService(
        retrieval_repository=(RetrievalRepository(client)),
        concept_repository=(ConceptRepository(client)),
        embedding_config=(
            QueryEmbeddingConfig(
                api_key=(google_api_key),
            )
        ),
    )

    services = QueryPipelineServices(
        retrieval=retrieval_service,
        domain_generation=(DomainGenerationService()),
        synthesis=(SynthesisService()),
        coverage=(CoverageService()),
        response_assembly=(ResponseAssemblyService()),
    )

    provider_config = QueryPipelineProviderConfig(
        domain_generation=(
            ProviderConfig(
                api_key=(groq_api_key),
                model=(
                    optional_environment(
                        "GROQ_MODEL",
                        DEFAULT_GROQ_MODEL,
                    )
                ),
                reasoning_effort=(DEFAULT_REASONING_EFFORT),
                temperature=(DEFAULT_TEMPERATURE),
                max_completion_tokens=(DEFAULT_MAX_COMPLETION_TOKENS),
                timeout_seconds=(DEFAULT_TIMEOUT_SECONDS),
                max_attempts=(DEFAULT_MAX_PROVIDER_ATTEMPTS),
            )
        ),
        synthesis=(
            SynthesisProviderConfig(
                api_key=(groq_api_key),
                model=(
                    optional_environment(
                        "PHASE16_GROQ_MODEL",
                        DEFAULT_SYNTHESIS_MODEL,
                    )
                ),
                temperature=(SYNTHESIS_TEMPERATURE),
                max_completion_tokens=(SYNTHESIS_MAX_COMPLETION_TOKENS),
                timeout_seconds=(SYNTHESIS_TIMEOUT_SECONDS),
                max_attempts=(SYNTHESIS_MAX_PROVIDER_ATTEMPTS),
            )
        ),
    )

    return QueryOrchestrator(
        services=services,
        provider_config=(provider_config),
    )


def compact_final_summary(
    final_response: Mapping[str, object],
    *,
    elapsed_seconds: float,
) -> str:
    sections_raw = final_response.get("sections")
    sections: Mapping[str, object] = sections_raw if isinstance(sections_raw, Mapping) else {}

    coverage_raw = sections.get("coverage")
    coverage: Mapping[str, object] = coverage_raw if isinstance(coverage_raw, Mapping) else {}

    activated_raw = sections.get("activated_concepts")
    activated_concepts = activated_raw if isinstance(activated_raw, list) else []

    domain_perspectives_raw = sections.get("domain_perspectives")
    domain_perspectives: Mapping[str, object] = (
        domain_perspectives_raw if isinstance(domain_perspectives_raw, Mapping) else {}
    )

    comparative_raw = sections.get("comparative_synthesis")
    comparative: Mapping[str, object] = (
        comparative_raw if isinstance(comparative_raw, Mapping) else {}
    )

    comparisons_raw = comparative.get("comparisons")
    comparisons = comparisons_raw if isinstance(comparisons_raw, list) else []

    claim_count = 0
    for perspective_raw in domain_perspectives.values():
        if not isinstance(perspective_raw, Mapping):
            continue
        claims_raw = perspective_raw.get("claims")
        if isinstance(claims_raw, list):
            claim_count += len(claims_raw)

    citations_raw = final_response.get("claim_level_citations")
    citations = citations_raw if isinstance(citations_raw, list) else []

    coverage_status = coverage.get("coverage_status")
    coverage_score = coverage.get("coverage_score")
    question = final_response.get("question")
    corpus_version = final_response.get("corpus_version")

    validation_raw = final_response.get("validation")
    validation: Mapping[str, object] = validation_raw if isinstance(validation_raw, Mapping) else {}
    validation_passed = validation.get("passed")

    return "\n".join(
        (
            "",
            "=" * 72,
            "STAGE 3.6D LIVE IN-MEMORY VERTICAL SLICE: PASS",
            "=" * 72,
            f"Question: {question}",
            f"Corpus: {corpus_version}",
            f"Coverage: {coverage_status} (score={coverage_score})",
            (
                f"Counts: concepts={len(activated_concepts)} "
                f"domains={len(domain_perspectives)} "
                f"claims={claim_count} "
                f"citations={len(citations)} "
                f"comparisons={len(comparisons)}"
            ),
            f"Final validation passed: {validation_passed}",
            f"Elapsed wall time: {elapsed_seconds:.2f}s",
            "",
            "Request-path architecture:",
            "  Phase 14 -> Phase 15 -> Phase 16 -> Phase 17 -> Phase 18",
            "  Python objects only between phases",
            "  No intermediate artifact files written by this runner",
            "=" * 72,
        )
    )


async def execute_live_query(
    *,
    question: str,
    show_markdown: bool,
) -> int:
    orchestrator = build_orchestrator()

    normalized_question = question.strip()

    if not normalized_question:
        raise VerticalSliceConfigurationError("Question must be non-empty.")

    LOGGER.info("Stage 3.6D live query starting")

    started = time.perf_counter()

    # ------------------------------------------------------------------
    # This is the production-style request boundary being proven.
    # No artifact wrapper or intermediate file operation occurs below.
    # ------------------------------------------------------------------
    result = await orchestrator.execute(normalized_question)

    elapsed_seconds = time.perf_counter() - started

    final_document = result.final_response.model_dump(
        mode="python",
        by_alias=True,
    )

    print(
        compact_final_summary(
            final_document,
            elapsed_seconds=(elapsed_seconds),
        )
    )

    if show_markdown:
        print()
        print(result.markdown)

    return 0


def main() -> int:
    arguments = parse_arguments()

    configure_logging(arguments.log_level)

    try:
        return asyncio.run(
            execute_live_query(
                question=(arguments.question),
                show_markdown=(arguments.show_markdown),
            )
        )

    except (
        QueryPipelineError,
        VerticalSliceConfigurationError,
    ):
        LOGGER.exception("Stage 3.6D live vertical slice failed")
        return 1

    except KeyboardInterrupt:
        LOGGER.warning("Stage 3.6D live vertical slice interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
