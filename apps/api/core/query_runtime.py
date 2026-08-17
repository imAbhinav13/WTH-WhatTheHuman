"""Production composition for the WTH query runtime.

This is the only Stage 4 module that knows how the Phase 14-18 runtime is
assembled for ``POST /api/query``.

The router must not know:
- Supabase repositories;
- Gemini embedding configuration;
- Groq model routing;
- Phase 15/16 provider settings;
- Phase 17/18 service implementations.

Those details are composed here and injected as one ``QueryOrchestrator``.
"""

from __future__ import annotations

from typing import cast

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
) -> QueryOrchestrator:
    """Construct the production Phase 14-18 ``QueryOrchestrator``.

    ``supabase_client`` is injectable only to keep composition tests offline.
    Normal application startup uses the backend-only cached runtime client.
    """

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

    # Phase 15 model/reasoning/token routing remains centralized in the
    # domain-generation service defaults:
    #
    # Science  -> GPT-OSS 20B  / medium / 2500
    # Advaita  -> GPT-OSS 120B / medium / 3000
    # Samkhya  -> GPT-OSS 20B  / medium / 2500
    #
    # The composition layer consumes that frozen mapping rather than making
    # model choices in the API router.
    domain_provider_config = (
        default_domain_provider_config(
            api_key=groq_api_key,
        )
    )

    # Phase 16:
    # GPT-OSS 120B / high / 4500, strict structured output as implemented by
    # SynthesisService.
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
        retrieval=retrieval_service,
        domain_generation=(
            DomainGenerationService()
        ),
        synthesis=SynthesisService(),
        coverage=CoverageService(),
        response_assembly=(
            ResponseAssemblyService()
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

    return QueryOrchestrator(
        services=services,
        provider_config=provider_config,
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
