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
from apps.api.core.production import (
    ProductionPolicy,
    build_production_policy,
)
from apps.api.core.query_runtime import build_query_orchestrator
from apps.api.middleware.production import (
    InMemoryRateLimitMiddleware,
    QueryBodySizeLimitMiddleware,
    StructuredAccessLogMiddleware,
)
from apps.api.routers.ops import router as ops_router
from apps.api.routers.query import (
    query_request_validation_exception_handler,
)
from apps.api.routers.query import router as query_router
from apps.api.routes.chunks import router as chunks_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize and close application dependencies."""

    settings = get_settings()

    # Re-run policy validation at startup so unsafe production settings fail
    # before the service accepts traffic.
    build_production_policy(settings)

    database_provider = _create_database_provider(settings)
    embedding_provider = _create_embedding_provider(settings)
    generation_provider = _create_generation_provider(settings)

    app.state.database_provider = database_provider
    app.state.embedding_provider = embedding_provider
    app.state.generation_provider = generation_provider

    app.state.query_orchestrator = (
        build_query_orchestrator(settings=settings)
        if settings.provider_mode is ProviderMode.LIVE
        else None
    )

    app.state.service_name = settings.app_name
    app.state.service_version = settings.app_version
    app.state.environment_name = settings.app_env

    try:
        yield
    finally:
        await generation_provider.close()
        await embedding_provider.close()
        await database_provider.close()


def create_app() -> FastAPI:
    """Create the Stage 6 production-hardened FastAPI application."""

    settings = get_settings()
    policy = build_production_policy(settings)

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Concept-aware comparative reasoning API across Science, Advaita Vedanta, and Samkhya."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    _install_edge_middleware(
        application=application,
        policy=policy,
    )

    application.include_router(
        ops_router,
        prefix="/api",
    )
    application.include_router(
        chunks_router,
        prefix="/api",
    )
    application.include_router(query_router)

    application.add_exception_handler(
        RequestValidationError,
        query_request_validation_exception_handler,
    )

    return application


def _install_edge_middleware(
    *,
    application: FastAPI,
    policy: ProductionPolicy,
) -> None:
    # Inner controls first. CORS is intentionally added last so it wraps
    # middleware-generated 413/429 responses as well.
    application.add_middleware(
        QueryBodySizeLimitMiddleware,
        max_bytes=policy.max_query_body_bytes,
    )
    # Abuse throttling is production-only. Local development remains
    # convenient while the same limiter is exercised deterministically by
    # Stage 6 tests.
    if policy.production:
        application.add_middleware(
            InMemoryRateLimitMiddleware,
            query_requests=policy.query_rate_limit_requests,
            query_window_seconds=policy.query_rate_limit_window_seconds,
            chunk_requests=policy.chunk_rate_limit_requests,
            chunk_window_seconds=policy.chunk_rate_limit_window_seconds,
        )

    application.add_middleware(
        StructuredAccessLogMiddleware,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(policy.cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Accept"],
        expose_headers=[
            "X-Request-ID",
            "Retry-After",
            "Server-Timing",
        ],
    )


def _create_database_provider(settings: Settings) -> DatabaseProvider:
    if settings.provider_mode is ProviderMode.MOCK:
        return MockDatabaseProvider(
            concept_count=8,
            corpus_version="phase0-v1",
            schema_available=True,
        )

    return SupabaseDatabaseProvider(
        url=str(settings.supabase_url),
        secret_key=_secret_value(
            settings.supabase_secret_key,
            field_name="SUPABASE_SECRET_KEY",
        ),
    )


def _create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.provider_mode is ProviderMode.MOCK:
        return MockEmbeddingProvider(
            model=settings.embedding_model,
            dimensions=settings.embedding_dimension,
        )

    return GeminiEmbeddingProvider(
        api_key=_secret_value(
            settings.google_api_key,
            field_name="GOOGLE_API_KEY",
        ),
        model=settings.embedding_model,
        dimensions=settings.embedding_dimension,
    )


def _create_generation_provider(settings: Settings) -> GenerationProvider:
    if settings.provider_mode is ProviderMode.MOCK:
        return MockGenerationProvider(
            model=settings.groq_model,
        )

    return GroqGenerationProvider(
        api_key=_secret_value(
            settings.groq_api_key,
            field_name="GROQ_API_KEY",
        ),
        model=settings.groq_model,
        timeout_seconds=settings.groq_timeout_seconds,
    )


def _secret_value(
    value: SecretStr | None,
    *,
    field_name: str,
) -> str:
    if value is None:
        raise RuntimeError(f"{field_name} is required in live provider mode")

    secret = value.get_secret_value().strip()

    if not secret:
        raise RuntimeError(f"{field_name} must not be empty")

    return secret


app = create_app()
