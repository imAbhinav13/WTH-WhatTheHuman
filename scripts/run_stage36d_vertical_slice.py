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
    DomainGenerationService,
    default_domain_provider_config,
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
    DEFAULT_MAX_PROVIDER_ATTEMPTS as SYNTHESIS_MAX_PROVIDER_ATTEMPTS,
)
from apps.api.services.synthesis import (
    DEFAULT_TEMPERATURE as SYNTHESIS_TEMPERATURE,
)
from apps.api.services.synthesis import (
    DEFAULT_TIMEOUT_SECONDS as SYNTHESIS_TIMEOUT_SECONDS,
)
from apps.api.services.synthesis import (
    SynthesisProviderConfig,
    SynthesisService,
)

LOGGER = logging.getLogger("wth.stage3.6d.vertical_slice")

DEFAULT_QUESTION = "What constitutes the self or personal identity?"

# Stage 3.7 starting values for Phase 16. SynthesisProviderConfig currently
# has no reasoning_effort field; the agreed ``high`` setting must be wired in
# synthesis.py itself rather than being simulated at composition time.
PHASE16_STARTING_MODEL = "openai/gpt-oss-120b"
PHASE16_STARTING_MAX_COMPLETION_TOKENS = 4500


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

    # Use the same centrally defined Phase 15 mapping as Artifact Mode.
    # DomainGenerationService owns model-lane scheduling:
    #   Science  -> GPT-OSS 20B  / medium / 2500
    #   Advaita  -> GPT-OSS 120B / high   / 3000
    #   Samkhya  -> GPT-OSS 20B  / medium / 2500
    # Science + Advaita can start together; Samkhya waits for Science to
    # release the shared GPT-OSS 20B lane.
    domain_provider_config = default_domain_provider_config(
        api_key=groq_api_key,
    )

    provider_config = QueryPipelineProviderConfig(
        domain_generation=(domain_provider_config),
        synthesis=(
            SynthesisProviderConfig(
                api_key=(groq_api_key),
                model=(
                    optional_environment(
                        "PHASE16_GROQ_MODEL",
                        PHASE16_STARTING_MODEL,
                    )
                ),
                temperature=(SYNTHESIS_TEMPERATURE),
                max_completion_tokens=(PHASE16_STARTING_MAX_COMPLETION_TOKENS),
                timeout_seconds=(SYNTHESIS_TIMEOUT_SECONDS),
                max_attempts=(SYNTHESIS_MAX_PROVIDER_ATTEMPTS),
            )
        ),
    )

    LOGGER.info(
        "Stage 3.6D Phase 15 provider lanes: science=%s/%s/%d advaita=%s/%s/%d samkhya=%s/%s/%d",
        domain_provider_config.science.model,
        domain_provider_config.science.reasoning_effort,
        domain_provider_config.science.max_completion_tokens,
        domain_provider_config.advaita.model,
        domain_provider_config.advaita.reasoning_effort,
        domain_provider_config.advaita.max_completion_tokens,
        domain_provider_config.samkhya.model,
        domain_provider_config.samkhya.reasoning_effort,
        domain_provider_config.samkhya.max_completion_tokens,
    )
    LOGGER.info(
        "Stage 3.6D Phase 16 provider: model=%s max_completion_tokens=%d "
        "reasoning_effort=pending-synthesis-config-support",
        provider_config.synthesis.model,
        provider_config.synthesis.max_completion_tokens,
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
    sections: Mapping[str, object] = (
        sections_raw
        if isinstance(
            sections_raw,
            Mapping,
        )
        else {}
    )

    coverage_raw = sections.get("coverage")
    coverage: Mapping[str, object] = (
        coverage_raw
        if isinstance(
            coverage_raw,
            Mapping,
        )
        else {}
    )

    active_concepts_raw = sections.get("activated_concepts")
    active_concepts = (
        active_concepts_raw
        if isinstance(
            active_concepts_raw,
            list,
        )
        else []
    )

    domain_perspectives_raw = sections.get("domain_perspectives")
    domain_perspectives: Mapping[str, object] = (
        domain_perspectives_raw
        if isinstance(
            domain_perspectives_raw,
            Mapping,
        )
        else {}
    )

    claim_count = 0
    for raw_domain in domain_perspectives.values():
        if not isinstance(
            raw_domain,
            Mapping,
        ):
            continue

        claims = raw_domain.get("claims")
        if isinstance(
            claims,
            list,
        ):
            claim_count += len(claims)

    comparative_raw = sections.get("comparative_synthesis")
    comparative: Mapping[str, object] = (
        comparative_raw
        if isinstance(
            comparative_raw,
            Mapping,
        )
        else {}
    )

    comparisons_raw = comparative.get("comparisons")
    comparisons = (
        comparisons_raw
        if isinstance(
            comparisons_raw,
            list,
        )
        else []
    )

    citations_raw = final_response.get("claim_level_citations")
    citations = (
        citations_raw
        if isinstance(
            citations_raw,
            list,
        )
        else []
    )

    validation_raw = final_response.get("validation")
    validation: Mapping[str, object] = (
        validation_raw
        if isinstance(
            validation_raw,
            Mapping,
        )
        else {}
    )

    question = final_response.get("question")
    corpus_version = final_response.get("corpus_version")

    return "\n".join(
        (
            "",
            "=" * 72,
            "STAGE 3.6D LIVE IN-MEMORY VERTICAL SLICE: PASS",
            "=" * 72,
            f"Question: {question}",
            f"Corpus: {corpus_version}",
            (
                "Coverage: "
                f"{coverage.get('coverage_status')} "
                f"(score={coverage.get('coverage_score')})"
            ),
            (
                "Counts: "
                f"concepts={len(active_concepts)} "
                f"domains={len(domain_perspectives)} "
                f"claims={claim_count} "
                f"citations={len(citations)} "
                f"comparisons={len(comparisons)}"
            ),
            (f"Final validation passed: {validation.get('passed')}"),
            (f"Elapsed wall time: {elapsed_seconds:.2f}s"),
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
        LOGGER.exception(
            "Stage 3.6D live vertical slice failed",
        )
        return 1

    except KeyboardInterrupt:
        LOGGER.warning("Stage 3.6D live vertical slice interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
