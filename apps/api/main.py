"""FastAPI application entry point for the WTH API."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import SecretStr

from apps.api.clients.base import (
    DatabaseProvider,
    EmbeddingProvider,
    GenerationProvider,
)
from apps.api.clients.embedding import (
    GeminiEmbeddingProvider,
    MockEmbeddingProvider,
)
from apps.api.clients.generation import (
    GroqGenerationProvider,
    MockGenerationProvider,
)
from apps.api.clients.supabase import (
    MockDatabaseProvider,
    SupabaseDatabaseProvider,
)
from apps.api.core.config import (
    ProviderMode,
    Settings,
    get_settings,
)
from apps.api.core.query_runtime import (
    build_query_orchestrator,
)
from apps.api.routers.health import (
    router as health_router,
)
from apps.api.routers.query import (
    query_request_validation_exception_handler,
)
from apps.api.routers.query import (
    router as query_router,
)
from apps.api.routes.chunks import (
    router as chunks_router,
)
from apps.api.routes.readiness import (
    router as readiness_router,
)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncIterator[None]:
    """Initialize and close application dependencies."""

    settings = get_settings()

    # Existing Stage 0/2 provider objects remain available for health/readiness
    # and earlier API capabilities.
    database_provider = (
        _create_database_provider(
            settings
        )
    )
    embedding_provider = (
        _create_embedding_provider(
            settings
        )
    )
    generation_provider = (
        _create_generation_provider(
            settings
        )
    )

    app.state.database_provider = (
        database_provider
    )
    app.state.embedding_provider = (
        embedding_provider
    )
    app.state.generation_provider = (
        generation_provider
    )

    # Stage 4 production query runtime.
    #
    # In live mode this creates the complete in-memory Phase 14-18 pipeline.
    # In mock mode the older health/chunk development paths remain usable, but
    # POST /api/query returns the controlled 503 defined by the query router.
    app.state.query_orchestrator = (
        build_query_orchestrator(
            settings=settings,
        )
        if (
            settings.provider_mode
            is ProviderMode.LIVE
        )
        else None
    )

    app.state.service_name = (
        settings.app_name
    )
    app.state.service_version = (
        settings.app_version
    )
    app.state.environment_name = (
        settings.app_env
    )

    try:
        yield
    finally:
        await generation_provider.close()
        await embedding_provider.close()
        await database_provider.close()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Concept-aware comparative reasoning API across "
            "Science, Advaita Vedanta, and Samkhya."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(
            settings.cors_origins
        ),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Existing Stage 2 routes retain their current /api/v1 contract.
    application.include_router(
        health_router,
        prefix="/api/v1",
    )
    application.include_router(
        chunks_router,
        prefix="/api/v1",
    )
    application.include_router(
        readiness_router,
        prefix="/api/v1",
    )

    # Stage 4 public query contract is deliberately /api/query.
    # The router contains no Phase 14-18 implementation logic.
    application.include_router(
        query_router
    )

    # Preserve the frozen Stage 4.1 controlled error envelope for malformed
    # POST /api/query requests while the handler delegates unrelated 422s back
    # to FastAPI's default validation handler.
    application.add_exception_handler(
        RequestValidationError,
        query_request_validation_exception_handler,
    )

    return application


def _create_database_provider(
    settings: Settings,
) -> DatabaseProvider:
    """Create the configured database provider."""

    if (
        settings.provider_mode
        is ProviderMode.MOCK
    ):
        return MockDatabaseProvider(
            concept_count=8,
            corpus_version="phase0-v1",
            schema_available=True,
        )

    return SupabaseDatabaseProvider(
        url=str(
            settings.supabase_url
        ),
        secret_key=_secret_value(
            settings.supabase_secret_key,
            field_name=(
                "SUPABASE_SECRET_KEY"
            ),
        ),
    )


def _create_embedding_provider(
    settings: Settings,
) -> EmbeddingProvider:
    """Create the configured embedding provider."""

    if (
        settings.provider_mode
        is ProviderMode.MOCK
    ):
        return MockEmbeddingProvider(
            model=settings.embedding_model,
            dimensions=(
                settings.embedding_dimension
            ),
        )

    return GeminiEmbeddingProvider(
        api_key=_secret_value(
            settings.google_api_key,
            field_name="GOOGLE_API_KEY",
        ),
        model=settings.embedding_model,
        dimensions=(
            settings.embedding_dimension
        ),
    )


def _create_generation_provider(
    settings: Settings,
) -> GenerationProvider:
    """Create the legacy/general text-generation provider.

    Stage 4 query model routing does NOT use ``settings.groq_model``. The
    Phase 15/16 model split is owned by ``core.query_runtime`` and the
    respective runtime service configs.
    """

    if (
        settings.provider_mode
        is ProviderMode.MOCK
    ):
        return MockGenerationProvider(
            model=settings.groq_model,
        )

    return GroqGenerationProvider(
        api_key=_secret_value(
            settings.groq_api_key,
            field_name="GROQ_API_KEY",
        ),
        model=settings.groq_model,
        timeout_seconds=(
            settings.groq_timeout_seconds
        ),
    )


def _secret_value(
    value: SecretStr | None,
    *,
    field_name: str,
) -> str:
    """Return a configured secret or fail during application startup."""

    if value is None:
        raise RuntimeError(
            f"{field_name} is required in live provider mode"
        )

    secret = (
        value.get_secret_value().strip()
    )

    if not secret:
        raise RuntimeError(
            f"{field_name} must not be empty"
        )

    return secret


app = create_app()
