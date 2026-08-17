"""Focused Stage 4.2 tests for ``POST /api/query``.

These tests intentionally stub QueryOrchestrator. They verify only the public
HTTP adapter; they do not re-test Phase 14-18 reasoning.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from apps.api.models.runtime_contracts import FinalResponse
from apps.api.routers import query as query_router
from apps.api.services.query_orchestrator import (
    QueryPhaseExecutionError,
    QueryPipelineInvariantError,
)


def _final_response() -> FinalResponse:
    return FinalResponse.model_validate(
        {
            "assembly_version": "stage4-test",
            "claim_level_citations": [],
            "corpus_version": "phase1_active_corpus_v1",
            "generated_at": "2026-08-16T00:00:00Z",
            "provider_calls": {
                "phase18_embedding_calls": 0,
                "phase18_llm_calls": 0,
                "phase18_retrieval_calls": 0,
            },
            "question": "What is consciousness?",
            "sections": {
                "activated_concepts": [],
                "comparative_synthesis": {
                    "comparisons": [],
                    "non_conclusion": "No corpus-grounded conclusion.",
                    "summary": "No corpus-grounded synthesis.",
                    "three_way_overview": "No supported comparison.",
                },
                "coverage": {
                    "coverage_reason": "Out-of-Corpus test fixture.",
                    "coverage_score": 0.0,
                    "coverage_status": "Out of Corpus",
                    "covered_domains": [],
                    "hard_overrides": [],
                    "missing_domains": [
                        "science",
                        "advaita",
                        "samkhya",
                    ],
                    "partially_supported_concepts": [],
                    "supported_concepts": [],
                    "unsupported_concepts": [],
                },
                "domain_perspectives": {
                    "science": {
                        "claims": [],
                        "display_name": "Science",
                        "domain": "science",
                        "limitations": [],
                        "summary": "No supported corpus answer.",
                        "unsupported_aspects": [],
                    },
                    "advaita": {
                        "claims": [],
                        "display_name": "Advaita Vedanta",
                        "domain": "advaita",
                        "limitations": [],
                        "summary": "No supported corpus answer.",
                        "unsupported_aspects": [],
                    },
                    "samkhya": {
                        "claims": [],
                        "display_name": "Samkhya",
                        "domain": "samkhya",
                        "limitations": [],
                        "summary": "No supported corpus answer.",
                        "unsupported_aspects": [],
                    },
                },
                "general_knowledge_fallback": {
                    "allowed": True,
                    "generated_in_phase18": False,
                    "instruction": (
                        "General knowledge, if added later, must be "
                        "clearly separated from reviewed-corpus support."
                    ),
                    "may_use_wth_corpus_citations": False,
                    "must_be_clearly_labeled": True,
                },
                "interpretation": "The reviewed corpus does not support this request.",
                "key_tensions": [],
                "non_equivalences": [],
            },
            "validation": {
                "checks": {
                    "all_phase15_claims_cited": True,
                    "citation_domains_match_claim_domains": True,
                    "citations_resolve_to_phase14_active_retrieval_evidence": True,
                    "corpus_and_prompt_versions_recorded": True,
                    "coverage_status_consistent_with_phase17_concept_statuses": True,
                    "out_of_corpus_blocks_corpus_answer": True,
                    "phase15_domain_leakage_validation_passed": True,
                    "phase16_synthesis_validation_passed": True,
                    "reviewed_corpus_and_general_knowledge_separated": True,
                    "unsupported_atman_purusha_equivalence_rejected": True,
                },
                "issue_count": 0,
                "issues": [],
                "passed": True,
            },
            "versions": {
                "corpus_version": "phase1_active_corpus_v1",
                "coverage_version": "test",
                "generation_prompt_version": "test",
                "generation_version": "test",
                "synthesis_prompt_version": "test",
                "synthesis_version": "test",
            },
        }
    )


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


class _SuccessOrchestrator:
    async def execute(
        self,
        question: str,
    ) -> SimpleNamespace:
        response = _final_response().model_copy(
            update={
                "question": question,
            }
        )
        return SimpleNamespace(
            final_response=response
        )


class _PhaseErrorOrchestrator:
    def __init__(
        self,
        *,
        phase: str,
        cause_message: str,
    ) -> None:
        self.phase = phase
        self.cause_message = cause_message

    async def execute(
        self,
        question: str,
    ) -> SimpleNamespace:
        del question

        try:
            raise RuntimeError(
                self.cause_message
            )
        except RuntimeError as cause:
            raise QueryPhaseExecutionError(
                phase=self.phase,  # type: ignore[arg-type]
                message=str(cause),
            ) from cause


class _InvariantOrchestrator:
    async def execute(
        self,
        question: str,
    ) -> SimpleNamespace:
        del question

        raise QueryPipelineInvariantError(
            phase="phase_18_response_assembly",
            invariant="final_validation",
            detail="test invariant",
        )


class _SlowOrchestrator:
    async def execute(
        self,
        question: str,
    ) -> SimpleNamespace:
        del question
        await asyncio.sleep(
            1.0
        )
        return SimpleNamespace(
            final_response=_final_response()
        )


def test_success_returns_final_response_directly() -> None:
    with TestClient(
        _app(
            _SuccessOrchestrator()
        )
    ) as client:
        response = client.post(
            "/api/query",
            json={
                "question": "What is consciousness?",
            },
        )

    assert response.status_code == 200
    assert response.json()["question"] == "What is consciousness?"
    assert "final_response" not in response.json()
    assert response.headers[
        query_router.REQUEST_ID_HEADER
    ].startswith("req_")


def test_out_of_corpus_is_http_200() -> None:
    with TestClient(
        _app(
            _SuccessOrchestrator()
        )
    ) as client:
        response = client.post(
            "/api/query",
            json={
                "question": "What is consciousness?",
            },
        )

    assert response.status_code == 200
    assert (
        response.json()["sections"]["coverage"]["coverage_status"]
        == "Out of Corpus"
    )


def test_invalid_question_uses_controlled_422_shape() -> None:
    with TestClient(
        _app(
            _SuccessOrchestrator()
        )
    ) as client:
        response = client.post(
            "/api/query",
            json={
                "question": "x",
            },
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
    assert response.json()["error"]["retryable"] is False


def test_rate_limit_maps_to_429_and_retry_after() -> None:
    orchestrator = _PhaseErrorOrchestrator(
        phase="phase_16_synthesis",
        cause_message=(
            "status=429 error_code=rate_limit_exceeded "
            "Please try again in 12.4575s"
        ),
    )

    with TestClient(
        _app(
            orchestrator
        )
    ) as client:
        response = client.post(
            "/api/query",
            json={
                "question": "What is consciousness?",
            },
        )

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "provider_rate_limited"
    assert response.json()["error"]["phase"] == "synthesis"
    assert response.headers["Retry-After"] == "13"


def test_generation_failure_maps_to_controlled_502() -> None:
    orchestrator = _PhaseErrorOrchestrator(
        phase="phase_15_domain_generation",
        cause_message=(
            "Groq request failed status=400 "
            "error_code=json_validate_failed"
        ),
    )

    with TestClient(
        _app(
            orchestrator
        )
    ) as client:
        response = client.post(
            "/api/query",
            json={
                "question": "What is consciousness?",
            },
        )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_provider_error"
    assert response.json()["error"]["phase"] == "domain_generation"


def test_retrieval_dependency_failure_maps_to_503() -> None:
    orchestrator = _PhaseErrorOrchestrator(
        phase="phase_14_retrieval",
        cause_message="Runtime Phase 14 retrieval dependency failed.",
    )

    with TestClient(
        _app(
            orchestrator
        )
    ) as client:
        response = client.post(
            "/api/query",
            json={
                "question": "What is consciousness?",
            },
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "dependency_unavailable"
    assert response.json()["error"]["phase"] == "retrieval"


def test_invariant_failure_maps_to_controlled_500() -> None:
    with TestClient(
        _app(
            _InvariantOrchestrator()
        )
    ) as client:
        response = client.post(
            "/api/query",
            json={
                "question": "What is consciousness?",
            },
        )

    assert response.status_code == 500
    assert (
        response.json()["error"]["code"]
        == "pipeline_invariant_failed"
    )
    assert response.json()["error"]["retryable"] is False


def test_missing_orchestrator_maps_to_503() -> None:
    with TestClient(
        _app(
            None
        )
    ) as client:
        response = client.post(
            "/api/query",
            json={
                "question": "What is consciousness?",
            },
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "dependency_unavailable"


def test_whole_request_timeout_maps_to_504(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        query_router,
        "QUERY_TIMEOUT_SECONDS",
        0.01,
    )

    with TestClient(
        _app(
            _SlowOrchestrator()
        )
    ) as client:
        response = client.post(
            "/api/query",
            json={
                "question": "What is consciousness?",
            },
        )

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "query_timeout"
    assert response.json()["error"]["retryable"] is True
