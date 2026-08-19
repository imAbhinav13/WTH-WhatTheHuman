from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from apps.api.routers import query as query_router
from apps.api.services.query_orchestrator import (
    QueryPhase,
    QueryPhaseExecutionError,
)


@dataclass(frozen=True, slots=True)
class FailureSpec:
    phase: QueryPhase
    cause_message: str


class RaisingOrchestrator:
    """Minimal orchestrator double that fails exactly at one public phase."""

    def __init__(self, spec: FailureSpec) -> None:
        self._spec = spec
        self.calls = 0

    async def execute(self, question: str) -> object:
        del question
        self.calls += 1

        try:
            raise RuntimeError(self._spec.cause_message)
        except RuntimeError as cause:
            raise QueryPhaseExecutionError(
                phase=self._spec.phase,
                message="Injected Stage 5.3 failure.",
            ) from cause


class SlowOrchestrator:
    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, question: str) -> object:
        del question
        self.calls += 1
        await asyncio.sleep(1.0)
        raise AssertionError("outer timeout should have cancelled the request")


def _app(orchestrator: object | None) -> FastAPI:
    app = FastAPI()

    app.include_router(
        query_router.router
    )
    app.add_exception_handler(
        RequestValidationError,
        query_router.query_request_validation_exception_handler,
    )

    if orchestrator is not None:
        app.state.query_orchestrator = orchestrator

    return app


def _post(
    app: FastAPI,
    question: str,
) -> object:
    with TestClient(app) as client:
        return client.post(
            "/api/query",
            json={"question": question},
        )


def _assert_controlled_error(
    response: object,
    *,
    status_code: int,
    code: str,
    phase: str | None,
    retryable: bool,
) -> dict[str, object]:
    assert getattr(response, "status_code") == status_code

    body = getattr(response, "json")()
    assert body["request_id"].startswith("req_")
    assert body["error"]["code"] == code
    assert body["error"]["phase"] == phase
    assert body["error"]["retryable"] is retryable

    header_request_id = getattr(response, "headers")["X-Request-ID"]
    assert header_request_id == body["request_id"]

    return body


def test_supabase_retrieval_failure_returns_controlled_503() -> None:
    secret_text = "SUPABASE_SECRET_KEY=never-leak-this"
    orchestrator = RaisingOrchestrator(
        FailureSpec(
            phase="phase_14_retrieval",
            cause_message=(
                "Supabase RPC failed connection refused "
                + secret_text
            ),
        )
    )

    response = _post(
        _app(orchestrator),
        "How is consciousness understood?",
    )

    body = _assert_controlled_error(
        response,
        status_code=503,
        code="dependency_unavailable",
        phase="retrieval",
        retryable=True,
    )

    assert secret_text not in str(body)
    assert orchestrator.calls == 1


def test_gemini_embedding_failure_returns_controlled_503() -> None:
    raw_provider_error = (
        "Gemini embedding request failed status=400 "
        "API_KEY_INVALID key=never-leak-this"
    )
    orchestrator = RaisingOrchestrator(
        FailureSpec(
            phase="phase_14_retrieval",
            cause_message=raw_provider_error,
        )
    )

    response = _post(
        _app(orchestrator),
        "How is consciousness understood?",
    )

    body = _assert_controlled_error(
        response,
        status_code=503,
        code="dependency_unavailable",
        phase="retrieval",
        retryable=True,
    )

    assert raw_provider_error not in str(body)
    assert "never-leak-this" not in str(body)


@pytest.mark.parametrize(
    ("phase", "public_phase"),
    [
        ("phase_15_domain_generation", "domain_generation"),
        ("phase_16_synthesis", "synthesis"),
    ],
)
def test_groq_429_returns_controlled_429(
    phase: QueryPhase,
    public_phase: str,
) -> None:
    orchestrator = RaisingOrchestrator(
        FailureSpec(
            phase=phase,
            cause_message=(
                "status=429 error_code=rate_limit_exceeded "
                "Please try again in 13.7175s"
            ),
        )
    )

    response = _post(
        _app(orchestrator),
        "How is consciousness understood?",
    )

    body = _assert_controlled_error(
        response,
        status_code=429,
        code="provider_rate_limited",
        phase=public_phase,
        retryable=True,
    )

    assert body["error"]["retry_after_seconds"] == pytest.approx(
        13.7175
    )
    assert response.headers["Retry-After"] == "14"


@pytest.mark.parametrize(
    ("cause_message", "phase", "public_phase"),
    [
        (
            "Groq request timed out after 60 seconds",
            "phase_15_domain_generation",
            "domain_generation",
        ),
        (
            "Groq request failed status=500 internal provider error",
            "phase_15_domain_generation",
            "domain_generation",
        ),
        (
            "Groq request failed status=400 "
            "error_code=json_validate_failed",
            "phase_16_synthesis",
            "synthesis",
        ),
    ],
)
def test_generation_provider_failures_return_controlled_502(
    cause_message: str,
    phase: QueryPhase,
    public_phase: str,
) -> None:
    orchestrator = RaisingOrchestrator(
        FailureSpec(
            phase=phase,
            cause_message=cause_message,
        )
    )

    response = _post(
        _app(orchestrator),
        "How is consciousness understood?",
    )

    body = _assert_controlled_error(
        response,
        status_code=502,
        code="upstream_provider_error",
        phase=public_phase,
        retryable=True,
    )

    assert cause_message not in str(body)


def test_whole_request_timeout_returns_controlled_504(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        query_router,
        "QUERY_TIMEOUT_SECONDS",
        0.01,
    )

    orchestrator = SlowOrchestrator()

    response = _post(
        _app(orchestrator),
        "How is consciousness understood?",
    )

    _assert_controlled_error(
        response,
        status_code=504,
        code="query_timeout",
        phase=None,
        retryable=True,
    )

    assert orchestrator.calls == 1


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "  "},
        {"question": "x"},
        {"question": "ok", "unexpected": True},
    ],
)
def test_malformed_requests_return_controlled_422(
    payload: dict[str, object],
) -> None:
    with TestClient(_app(None)) as client:
        response = client.post(
            "/api/query",
            json=payload,
        )

    _assert_controlled_error(
        response,
        status_code=422,
        code="invalid_request",
        phase=None,
        retryable=False,
    )


def test_oversized_question_returns_controlled_422() -> None:
    response = _post(
        _app(None),
        "q" * 1001,
    )

    _assert_controlled_error(
        response,
        status_code=422,
        code="invalid_request",
        phase=None,
        retryable=False,
    )


def test_missing_query_orchestrator_returns_controlled_503() -> None:
    response = _post(
        _app(None),
        "How is consciousness understood?",
    )

    _assert_controlled_error(
        response,
        status_code=503,
        code="dependency_unavailable",
        phase=None,
        retryable=True,
    )
