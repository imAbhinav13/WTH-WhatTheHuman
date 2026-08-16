"""FastAPI application entry point for the WTH API."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
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
from apps.api.routers.health import router as health_router
from apps.api.routes.chunks import router as chunks_router
from apps.api.routes.readiness import router as readiness_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize and close application dependencies."""

    settings = get_settings()

    database_provider = _create_database_provider(settings)
    embedding_provider = _create_embedding_provider(settings)
    generation_provider = _create_generation_provider(settings)

    app.state.database_provider = database_provider
    app.state.embedding_provider = embedding_provider
    app.state.generation_provider = generation_provider

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
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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

    return application


def _create_database_provider(
    settings: Settings,
) -> DatabaseProvider:
    """Create the configured database provider."""

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


def _create_embedding_provider(
    settings: Settings,
) -> EmbeddingProvider:
    """Create the configured embedding provider."""

    if settings.provider_mode is ProviderMode.MOCK:
        return MockEmbeddingProvider(
            model=settings.embedding_model,
            dimensions=settings.embedding_dimension,
        )

    return GeminiEmbeddingProvider(
        api_key=_secret_value(
            settings.google_api_key,
            field_name="GEMINI_API_KEY",
        ),
        model=settings.embedding_model,
        dimensions=settings.embedding_dimension,
    )


def _create_generation_provider(
    settings: Settings,
) -> GenerationProvider:
    """Create the configured text-generation provider."""

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
    """Return a configured secret or fail during application startup."""

    if value is None:
        raise RuntimeError(
            f"{field_name} is required in live provider mode"
        )

    secret = value.get_secret_value().strip()

    if not secret:
        raise RuntimeError(
            f"{field_name} must not be empty"
        )

    return secret


app = create_app()