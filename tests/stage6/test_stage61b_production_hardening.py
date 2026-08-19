"""Offline Stage 6.1B hardening regression tests."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from pydantic import SecretStr

from apps.api.core.production import (
    ProductionConfigurationError,
    build_production_policy,
)
from apps.api.middleware.production import (
    InMemoryRateLimitMiddleware,
    QueryBodySizeLimitMiddleware,
    StructuredAccessLogMiddleware,
)


def _settings(
    *,
    app_env: str = "development",
    origins: tuple[str, ...] = ("http://localhost:3000",),
) -> SimpleNamespace:
    return SimpleNamespace(
        app_env=app_env,
        provider_mode=SimpleNamespace(value="live"),
        cors_origins=origins,
        supabase_url="https://real-project.supabase.co",
        supabase_secret_key=SecretStr("service-role-test-value"),
        groq_api_key=SecretStr("groq-test-value"),
        google_api_key=SecretStr("google-test-value"),
    )


def _edge_app(
    *,
    max_bytes: int = 128,
    query_limit: int = 2,
) -> FastAPI:
    app = FastAPI()

    @app.post("/api/query")
    async def query() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/api/chunk/{chunk_id}")
    async def chunk(chunk_id: str) -> dict[str, str]:
        return {"chunk_id": chunk_id}

    app.add_middleware(
        QueryBodySizeLimitMiddleware,
        max_bytes=max_bytes,
    )
    app.add_middleware(
        InMemoryRateLimitMiddleware,
        query_requests=query_limit,
        query_window_seconds=60,
        chunk_requests=20,
        chunk_window_seconds=60,
    )
    app.add_middleware(StructuredAccessLogMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://frontend.example"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Accept"],
    )
    return app


def test_production_rejects_wildcard_cors() -> None:
    with pytest.raises(
        ProductionConfigurationError,
        match="must not contain",
    ):
        build_production_policy(
            _settings(
                app_env="production",
                origins=("*",),
            )
        )


def test_production_requires_https_cors() -> None:
    with pytest.raises(
        ProductionConfigurationError,
        match="HTTPS",
    ):
        build_production_policy(
            _settings(
                app_env="production",
                origins=("http://frontend.example",),
            )
        )


def test_production_accepts_valid_server_side_configuration() -> None:
    policy = build_production_policy(
        _settings(
            app_env="production",
            origins=("https://frontend.example",),
        )
    )

    assert policy.production is True
    assert policy.cors_origins == (
        "https://frontend.example",
    )


def test_query_body_over_limit_returns_controlled_413() -> None:
    with TestClient(
        _edge_app(max_bytes=64)
    ) as client:
        response = client.post(
            "/api/query",
            content=b"x" * 65,
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 413
    body = response.json()
    assert body["error"]["code"] == "request_too_large"
    assert body["error"]["retryable"] is False
    assert body["request_id"].startswith("req_")


def test_query_rate_limit_has_distinct_api_429_code() -> None:
    with TestClient(
        _edge_app(query_limit=1)
    ) as client:
        first = client.post(
            "/api/query",
            json={"question": "abc"},
        )
        second = client.post(
            "/api/query",
            json={"question": "abc"},
        )

    assert first.status_code == 200
    assert second.status_code == 429
    body = second.json()
    assert body["error"]["code"] == "api_rate_limited"
    assert body["error"]["phase"] is None
    assert second.headers["retry-after"]


def test_cors_is_present_on_middleware_generated_rate_limit() -> None:
    app = _edge_app(query_limit=1)

    with TestClient(app) as client:
        headers = {
            "origin": "https://frontend.example",
        }
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
    assert (
        second.headers["access-control-allow-origin"]
        == "https://frontend.example"
    )


def test_disallowed_origin_is_not_granted_cors_permission() -> None:
    with TestClient(_edge_app()) as client:
        response = client.post(
            "/api/query",
            json={"question": "abc"},
            headers={
                "origin": "https://attacker.example",
            },
        )

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_access_log_does_not_include_request_body(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret_question = "DO_NOT_LOG_THIS_QUESTION"

    caplog.set_level(
        logging.INFO,
        logger="wth.api.access",
    )

    with TestClient(_edge_app()) as client:
        response = client.post(
            "/api/query",
            json={"question": secret_question},
        )

    assert response.status_code == 200

    text = "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.name == "wth.api.access"
    )

    assert "http_request_complete" in text
    assert secret_question not in text
