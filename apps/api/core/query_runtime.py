from __future__ import annotations

from typing import Any, cast

from pydantic import SecretStr
from supabase import Client

from apps.api.clients.supabase_runtime import (
    get_supabase_runtime_client,
)
from apps.api.core.config import (
    ProviderMode,
    Settings,
    get_settings,
)
from apps.api.core.performance import (
    PerformanceInstrumentedOrchestrator,
    TimedServiceProxy,
    instrument_retrieval_embedding,
)
from apps.api.repositories.concept_repository import (
    ConceptRepository,
)
from apps.api.repositories.retrieval_repository import (
    RetrievalRepository,
)
from apps.api.services.coverage import CoverageService
from apps.api.services.domain_generation import (
    DomainGenerationService,
    default_domain_provider_config,
)
from apps.api.services.query_orchestrator import (
    QueryOrchestrator,
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
    DEFAULT_REASONING_EFFORT as SYNTHESIS_REASONING_EFFORT,
)
from apps.api.services.synthesis import (
    DEFAULT_SYNTHESIS_MODEL,
    DEFAULT_TEMPERATURE as SYNTHESIS_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS as SYNTHESIS_TIMEOUT_SECONDS,
    SynthesisProviderConfig,
    SynthesisService,
)


class QueryRuntimeConfigurationError(
    RuntimeError
):
    """Raised when production query-runtime composition is not possible."""


def build_query_orchestrator(
    *,
    settings: Settings | None = None,
    supabase_client: Client | None = None,
) -> PerformanceInstrumentedOrchestrator:
    """Construct the production Phase 14-18 query runtime."""

    resolved_settings = (
        settings
        if settings is not None
        else get_settings()
    )

    if (
        resolved_settings.provider_mode
        is not ProviderMode.LIVE
    ):
        raise QueryRuntimeConfigurationError(
            "POST /api/query requires PROVIDER_MODE=live."
        )

    google_api_key = _required_secret(
        resolved_settings.google_api_key,
        field_name="GOOGLE_API_KEY",
    )
    groq_api_key = _required_secret(
        resolved_settings.groq_api_key,
        field_name="GROQ_API_KEY",
    )

    client = (
        supabase_client
        if supabase_client is not None
        else get_supabase_runtime_client()
    )

    concept_repository = ConceptRepository(client)
    retrieval_repository = RetrievalRepository(client)

    retrieval_service = RetrievalService(
        retrieval_repository=(
            retrieval_repository
        ),
        concept_repository=(
            concept_repository
        ),
        embedding_config=QueryEmbeddingConfig(
            api_key=google_api_key,
        ),
    )

    # Phase 14 embedding is timed separately inside the existing retrieval
    # service. The surrounding proxy times the whole Phase 14 call; the
    # performance module subtracts embedding time from retrieval_ms.
    instrument_retrieval_embedding(
        retrieval_service
    )

    domain_provider_config = (
        default_domain_provider_config(
            api_key=groq_api_key,
        )
    )

    synthesis_provider_config = (
        SynthesisProviderConfig(
            api_key=groq_api_key,
            model=DEFAULT_SYNTHESIS_MODEL,
            reasoning_effort=(
                SYNTHESIS_REASONING_EFFORT
            ),
            temperature=(
                SYNTHESIS_TEMPERATURE
            ),
            max_completion_tokens=(
                SYNTHESIS_MAX_COMPLETION_TOKENS
            ),
            timeout_seconds=(
                SYNTHESIS_TIMEOUT_SECONDS
            ),
            max_attempts=(
                SYNTHESIS_MAX_PROVIDER_ATTEMPTS
            ),
        )
    )

    services = QueryPipelineServices(
        retrieval=cast(
            Any,
            TimedServiceProxy(
                retrieval_service,
                metric="retrieval_ms",
            ),
        ),
        domain_generation=cast(
            Any,
            TimedServiceProxy(
                DomainGenerationService(),
                metric="generation_ms",
            ),
        ),
        synthesis=cast(
            Any,
            TimedServiceProxy(
                SynthesisService(),
                metric="synthesis_ms",
            ),
        ),
        coverage=cast(
            Any,
            TimedServiceProxy(
                CoverageService(),
                metric="coverage_ms",
            ),
        ),
        response_assembly=cast(
            Any,
            TimedServiceProxy(
                ResponseAssemblyService(),
                metric="assembly_ms",
            ),
        ),
    )

    provider_config = (
        QueryPipelineProviderConfig(
            domain_generation=(
                domain_provider_config
            ),
            synthesis=(
                synthesis_provider_config
            ),
        )
    )

    orchestrator = QueryOrchestrator(
        services=services,
        provider_config=provider_config,
    )

    return PerformanceInstrumentedOrchestrator(
        orchestrator
    )


def _required_secret(
    value: SecretStr | None,
    *,
    field_name: str,
) -> str:
    if value is None:
        raise QueryRuntimeConfigurationError(
            f"{field_name} is required for POST /api/query."
        )

    secret = (
        value.get_secret_value().strip()
    )

    if not secret:
        raise QueryRuntimeConfigurationError(
            f"{field_name} must not be empty."
        )

    return secret


__all__ = [
    "QueryRuntimeConfigurationError",
    "build_query_orchestrator",
]