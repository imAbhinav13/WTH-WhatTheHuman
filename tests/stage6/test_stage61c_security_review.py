"""Stage 6.1C deployment/security review.

All tests are offline. They prove the public HTTP boundary does not expose
synthetic secret-bearing exception text, and that the Stage 6 edge contract
is documented and browser-usable.
"""

from __future__ import annotations

import logging
import os

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

os.environ["PROVIDER_MODE"] = "mock"

from apps.api.core.config import get_settings
from apps.api.middleware.production import (
    InMemoryRateLimitMiddleware,
    QueryBodySizeLimitMiddleware,
    StructuredAccessLogMiddleware,
)
from apps.api.models.query_api import (
    ERROR_HTTP_STATUS,
    QueryApiErrorCode,
)
from apps.api.routers import query as query_router
from apps.api.services.query_orchestrator import (
    QueryPipelineError,
)


class SecretBearingUnexpectedOrchestrator:
    async def execute(self, question: str) -> object:
        del question
        raise RuntimeError(
            "GROQ_API_KEY=gsk_NEVER_LOG_ME "
            "SUPABASE_SECRET_KEY=NEVER_LOG_ME "
            "GOOGLE_API_KEY=AIza_NEVER_LOG_ME"
        )


class SecretBearingPipelineOrchestrator:
    async def execute(self, question: str) -> object:
        del question
        raise QueryPipelineError("Authorization: Bearer NEVER_LOG_ME")


def _query_app(orchestrator: object) -> FastAPI:
    app = FastAPI()
    app.state.query_orchestrator = orchestrator
    app.include_router(query_router.router)
    app.add_exception_handler(
        RequestValidationError,
        query_router.query_request_validation_exception_handler,
    )
    return app


@pytest.mark.parametrize(
    "orchestrator",
    [
        SecretBearingUnexpectedOrchestrator(),
        SecretBearingPipelineOrchestrator(),
    ],
)
def test_query_error_response_and_logs_do_not_leak_secret_exception_text(
    orchestrator: object,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR, logger="wth.api.query")

    with TestClient(_query_app(orchestrator)) as client:
        response = client.post(
            "/api/query",
            json={"question": "What is consciousness?"},
        )

    assert response.status_code == 500

    combined = (
        response.text
        + "\n"
        + "\n".join(
            record.getMessage() for record in caplog.records if record.name == "wth.api.query"
        )
    )

    forbidden = (
        "gsk_NEVER_LOG_ME",
        "NEVER_LOG_ME",
        "AIza_NEVER_LOG_ME",
        "Authorization: Bearer",
        "SUPABASE_SECRET_KEY",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
    )

    for marker in forbidden:
        assert marker not in combined


def test_stage6_edge_error_codes_are_part_of_public_error_model() -> None:
    assert QueryApiErrorCode.REQUEST_TOO_LARGE.value == "request_too_large"
    assert QueryApiErrorCode.API_RATE_LIMITED.value == "api_rate_limited"
    assert ERROR_HTTP_STATUS[QueryApiErrorCode.REQUEST_TOO_LARGE] == 413
    assert ERROR_HTTP_STATUS[QueryApiErrorCode.API_RATE_LIMITED] == 429


def test_query_openapi_documents_413_and_429() -> None:
    app = _query_app(SecretBearingUnexpectedOrchestrator())
    operation = app.openapi()["paths"]["/api/query"]["post"]
    responses = operation["responses"]

    assert "413" in responses
    assert "429" in responses


def _edge_app() -> FastAPI:
    app = FastAPI()

    @app.post("/api/query")
    async def query() -> dict[str, bool]:
        return {"ok": True}

    app.add_middleware(
        QueryBodySizeLimitMiddleware,
        max_bytes=64,
    )
    app.add_middleware(
        InMemoryRateLimitMiddleware,
        query_requests=1,
        query_window_seconds=60,
        chunk_requests=10,
        chunk_window_seconds=60,
    )
    app.add_middleware(StructuredAccessLogMiddleware)

    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://frontend.example"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Accept"],
        expose_headers=[
            "X-Request-ID",
            "Retry-After",
            "Server-Timing",
        ],
    )

    return app


def test_allowed_origin_can_read_request_id_on_413() -> None:
    with TestClient(_edge_app()) as client:
        response = client.post(
            "/api/query",
            content=b"x" * 65,
            headers={
                "content-type": "application/json",
                "origin": "https://frontend.example",
            },
        )

    assert response.status_code == 413
    assert response.headers["access-control-allow-origin"] == "https://frontend.example"
    exposed = response.headers["access-control-expose-headers"].lower()
    assert "x-request-id" in exposed
    assert response.headers["x-request-id"].startswith("req_")


def test_allowed_origin_can_read_retry_after_on_api_429() -> None:
    with TestClient(_edge_app()) as client:
        headers = {"origin": "https://frontend.example"}

        first = client.post(
            "/api/query",
            json={"question": "abc"},
            headers=headers,
        )
        second = client.post(
            "/api/query",
            json={"question": "abc"},
            headers=headers,
        )

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "api_rate_limited"
    assert second.headers["retry-after"]

    exposed = second.headers["access-control-expose-headers"].lower()
    assert "retry-after" in exposed
    assert "x-request-id" in exposed


def test_structured_access_log_has_no_headers_or_body(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="wth.api.access")

    secret_body = "SUPER_PRIVATE_QUESTION_TEXT"
    secret_header = "Bearer SHOULD_NOT_APPEAR"

    with TestClient(_edge_app()) as client:
        response = client.post(
            "/api/query",
            json={"question": secret_body},
            headers={
                "authorization": secret_header,
            },
        )

    assert response.status_code == 200

    access_logs = "\n".join(
        record.getMessage() for record in caplog.records if record.name == "wth.api.access"
    )

    assert "http_request_complete" in access_logs
    assert secret_body not in access_logs
    assert secret_header not in access_logs
    assert "authorization" not in access_logs.lower()


def test_api_v1_is_absent_after_security_changes() -> None:
    get_settings.cache_clear()

    from apps.api.main import create_app

    app = create_app()
    paths = app.openapi()["paths"]

    assert not any(path.startswith("/api/v1") for path in paths)
    assert "/api/query" in paths
    assert "/api/chunk/{chunk_id}" in paths
    assert "/api/health" in paths
    assert "/api/ready" in paths
