"""Production health/readiness endpoints for Stage 6.

Readiness deliberately does NOT generate Groq text. Provider generation is
validated by the real /api/query smoke test, not by an uptime probe that would
consume quota and can fail on transient structured-output behavior.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse

from apps.api.core.config import ProviderMode, Settings, get_settings
from apps.api.repositories.chunk_repository import (
    ChunkRepository,
    ChunkRepositoryError,
)
from apps.api.services.chunk_service import get_chunk_repository

router = APIRouter(tags=["operations"])


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@router.get("/health")
async def health(request: Request) -> dict[str, object]:
    """Liveness only: no database or provider network calls."""

    return {
        "status": "healthy",
        "service": getattr(
            request.app.state,
            "service_name",
            "WTH: What The Human",
        ),
        "version": getattr(
            request.app.state,
            "service_version",
            "unknown",
        ),
        "environment": getattr(
            request.app.state,
            "environment_name",
            "unknown",
        ),
        "checks": {},
        "timestamp": _timestamp(),
    }

@router.head("/health")
async def health_head() -> Response:
    """Lightweight HEAD liveness probe for uptime monitors."""
    return Response(status_code=200)


@router.get("/ready",response_model=None)
async def ready(
    request: Request,
    repository: ChunkRepository = Depends(get_chunk_repository), # noqa: B008
    settings: Settings = Depends(get_settings), # noqa: B008
) -> dict[str, object] | JSONResponse:
    """Check database/corpus plus provider/runtime configuration.

    Groq and Gemini are reported as configured here; this endpoint does not
    spend provider quota. Real reachability is proven by Stage 6.5 /api/query.
    """

    checks: list[dict[str, object]] = []
    healthy = True

    started = time.perf_counter()
    try:
        record = repository.check_ready()
        latency_ms = round(
            (time.perf_counter() - started) * 1000.0
        )
        checks.append(
            {
                "provider": "database",
                "status": "ready",
                "latency_ms": latency_ms,
                "detail": (
                    "Supabase corpus database is reachable; "
                    f"corpus_version={record.corpus_version}"
                ),
            }
        )
    except ChunkRepositoryError:
        healthy = False
        latency_ms = round(
            (time.perf_counter() - started) * 1000.0
        )
        checks.append(
            {
                "provider": "database",
                "status": "unavailable",
                "latency_ms": latency_ms,
                "detail": "Corpus database is unavailable.",
            }
        )

    embedding_configured = _secret_configured(
        settings.google_api_key
    )
    generation_configured = _secret_configured(
        settings.groq_api_key
    )

    checks.append(
        {
            "provider": "embedding",
            "status": (
                "configured"
                if embedding_configured
                else "unavailable"
            ),
            "latency_ms": 0,
            "detail": (
                "Gemini embedding configuration present; "
                f"model={settings.embedding_model}, "
                f"dimensions={settings.embedding_dimension}"
                if embedding_configured
                else "Gemini embedding configuration is missing."
            ),
        }
    )

    checks.append(
        {
            "provider": "generation",
            "status": (
                "configured"
                if generation_configured
                else "unavailable"
            ),
            "latency_ms": 0,
            "detail": (
                "Groq generation configuration present; "
                "no generation probe executed."
                if generation_configured
                else "Groq generation configuration is missing."
            ),
        }
    )

    if settings.provider_mode is ProviderMode.LIVE:
        orchestrator = getattr(
            request.app.state,
            "query_orchestrator",
            None,
        )
        runtime_ready = callable(
            getattr(orchestrator, "execute", None)
        )
    else:
        runtime_ready = True

    checks.append(
        {
            "provider": "query_runtime",
            "status": "ready" if runtime_ready else "unavailable",
            "latency_ms": 0,
            "detail": (
                "Query runtime is composed."
                if runtime_ready
                else "Query runtime is not composed."
            ),
        }
    )

    healthy = (
        healthy
        and embedding_configured
        and generation_configured
        and runtime_ready
    )

    payload: dict[str, object] = {
        "status": "healthy" if healthy else "unhealthy",
        "service": getattr(
            request.app.state,
            "service_name",
            settings.app_name,
        ),
        "version": getattr(
            request.app.state,
            "service_version",
            settings.app_version,
        ),
        "environment": getattr(
            request.app.state,
            "environment_name",
            settings.app_env,
        ),
        "checks": checks,
        "timestamp": _timestamp(),
    }

    if healthy:
        return payload

    return JSONResponse(
        status_code=503,
        content=payload,
    )


def _secret_configured(value: object) -> bool:
    if value is None:
        return False

    getter = getattr(value, "get_secret_value", None)
    raw = str(getter()) if callable(getter) else str(value)

    return bool(raw.strip())


__all__ = ["health", "ready", "router"]
