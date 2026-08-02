"""Health and readiness endpoints for the WTH API."""

from __future__ import annotations

import asyncio
from typing import cast

from fastapi import APIRouter, Request, status

from apps.api.clients.base import (
    DatabaseProvider,
    EmbeddingProvider,
    GenerationProvider,
)
from apps.api.models.common import (
    HealthResponse,
    ProviderHealth,
)
from apps.api.models.enums import (
    HealthStatus,
    ProviderKind,
    ProviderStatus,
)
from apps.api.models.providers import (
    DatabaseProbeRequest,
    DatabaseProbeResponse,
    ProviderProbeResult,
)

router = APIRouter(
    tags=["health"],
)


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Liveness check",
)
async def health(request: Request) -> HealthResponse:
    """Confirm that the FastAPI process is running."""

    return HealthResponse(
        status=HealthStatus.HEALTHY,
        service=_service_name(request),
        version=_service_version(request),
        environment=_environment_name(request),
        checks=(),
    )


@router.get(
    "/ready",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Readiness check",
)
async def readiness(request: Request) -> HealthResponse:
    """Check whether all application dependencies are ready."""

    database_provider = cast(
        DatabaseProvider,
        request.app.state.database_provider,
    )
    embedding_provider = cast(
        EmbeddingProvider,
        request.app.state.embedding_provider,
    )
    generation_provider = cast(
        GenerationProvider,
        request.app.state.generation_provider,
    )

    database_result, embedding_result, generation_result = (
        await asyncio.gather(
            database_provider.probe(
                DatabaseProbeRequest(
                    verify_schema=True,
                    verify_concepts=True,
                    expected_concept_count=8,
                )
            ),
            embedding_provider.probe(),
            generation_provider.probe(),
        )
    )

    checks = (
        _database_health(database_result),
        _provider_health(embedding_result),
        _provider_health(generation_result),
    )

    return HealthResponse(
        status=_overall_status(checks),
        service=_service_name(request),
        version=_service_version(request),
        environment=_environment_name(request),
        checks=checks,
    )


def _database_health(
    result: DatabaseProbeResponse,
) -> ProviderHealth:
    """Convert a database probe into the common health model."""

    details: list[str] = []

    if result.detail is not None:
        details.append(result.detail)

    details.append(
        f"schema_available={result.schema_available}"
    )

    if result.concept_count is not None:
        details.append(
            f"active_concepts={result.concept_count}"
        )

    if result.active_corpus_version is not None:
        details.append(
            "corpus_version="
            f"{result.active_corpus_version}"
        )

    return ProviderHealth(
        provider=ProviderKind.DATABASE,
        status=result.status,
        latency_ms=result.latency_ms,
        detail="; ".join(details),
    )


def _provider_health(
    result: ProviderProbeResult,
) -> ProviderHealth:
    """Convert a provider probe into the common health model."""

    return ProviderHealth(
        provider=result.provider,
        status=result.status,
        latency_ms=result.latency_ms,
        detail=result.detail,
    )


def _overall_status(
    checks: tuple[ProviderHealth, ...],
) -> HealthStatus:
    """Calculate the API readiness state from dependency checks."""

    statuses = {
        check.status
        for check in checks
    }

    if ProviderStatus.UNAVAILABLE in statuses:
        return HealthStatus.UNHEALTHY

    if ProviderStatus.DEGRADED in statuses:
        return HealthStatus.DEGRADED

    return HealthStatus.HEALTHY


def _service_name(request: Request) -> str:
    """Return the service name configured during application startup."""

    value = getattr(
        request.app.state,
        "service_name",
        None,
    )

    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(
            "Application state is missing service_name"
        )

    return value


def _service_version(request: Request) -> str:
    """Return the service version configured during startup."""

    value = getattr(
        request.app.state,
        "service_version",
        None,
    )

    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(
            "Application state is missing service_version"
        )

    return value


def _environment_name(request: Request) -> str:
    """Return the deployment environment configured at startup."""

    value = getattr(
        request.app.state,
        "environment_name",
        None,
    )

    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(
            "Application state is missing environment_name"
        )

    return value