"""Stage 5.4 performance and offline-concurrency tests."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from apps.api.core.performance import (
    PerformanceInstrumentedOrchestrator,
    QueryTimings,
    TimedServiceProxy,
    begin_query_timings,
    finish_query_timings,
    instrument_retrieval_embedding,
)
from apps.api.models.runtime_contracts import FinalResponse
from apps.api.routers import query as query_router


def _final_response(question: str) -> FinalResponse:
    return FinalResponse.model_validate(
        {
            "assembly_version": "stage5-test",
            "claim_level_citations": [],
            "corpus_version": "phase1_active_corpus_v1",
            "generated_at": "2026-08-17T00:00:00Z",
            "provider_calls": {
                "phase18_embedding_calls": 0,
                "phase18_llm_calls": 0,
                "phase18_retrieval_calls": 0,
            },
            "question": question,
            "sections": {
                "activated_concepts": [],
                "comparative_synthesis": {
                    "comparisons": [],
                    "non_conclusion": "No corpus-grounded conclusion.",
                    "summary": "No corpus-grounded synthesis.",
                    "three_way_overview": "No supported comparison.",
                },
                "coverage": {
                    "coverage_reason": "test",
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
                        "summary": "test",
                        "unsupported_aspects": [],
                    },
                    "advaita": {
                        "claims": [],
                        "display_name": "Advaita Vedanta",
                        "domain": "advaita",
                        "limitations": [],
                        "summary": "test",
                        "unsupported_aspects": [],
                    },
                    "samkhya": {
                        "claims": [],
                        "display_name": "Samkhya",
                        "domain": "samkhya",
                        "limitations": [],
                        "summary": "test",
                        "unsupported_aspects": [],
                    },
                },
                "general_knowledge_fallback": {
                    "allowed": True,
                    "generated_in_phase18": False,
                    "instruction": "test",
                    "may_use_wth_corpus_citations": False,
                    "must_be_clearly_labeled": True,
                },
                "interpretation": "test",
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


class _UnderlyingOrchestrator:
    async def execute(self, question: str) -> SimpleNamespace:
        await asyncio.sleep(0.01)
        return SimpleNamespace(
            final_response=_final_response(question)
        )


class _FakeRetrieval:
    def __init__(self) -> None:
        self._embedding_runner = self._embed

    def _embed(self) -> str:
        time.sleep(0.01)
        return "embedding"

    def retrieve(self) -> str:
        self._embedding_runner()
        time.sleep(0.01)
        return "retrieved"


class _SyncService:
    def run(self) -> str:
        time.sleep(0.005)
        return "ok"


def _app() -> FastAPI:
    app = FastAPI()
    app.state.query_orchestrator = (
        PerformanceInstrumentedOrchestrator(
            _UnderlyingOrchestrator()
        )
    )
    app.include_router(query_router.router)
    app.add_exception_handler(
        RequestValidationError,
        query_router.query_request_validation_exception_handler,
    )
    return app


def test_query_timings_server_header_contains_all_metrics() -> None:
    timings = QueryTimings(
        embedding_ms=1.0,
        retrieval_ms=2.0,
        generation_ms=3.0,
        synthesis_ms=4.0,
        coverage_ms=5.0,
        assembly_ms=6.0,
        total_ms=21.0,
    )

    header = timings.server_timing_header()

    for name in (
        "embedding",
        "retrieval",
        "generation",
        "synthesis",
        "coverage",
        "assembly",
        "total",
    ):
        assert f"{name};dur=" in header


def test_embedding_is_separated_from_phase14_retrieval_time() -> None:
    retrieval = _FakeRetrieval()
    instrument_retrieval_embedding(retrieval)

    proxy = TimedServiceProxy(
        retrieval,
        metric="retrieval_ms",
    )

    started = time.perf_counter()
    timings, token = begin_query_timings()

    assert proxy.retrieve() == "retrieved"

    final = finish_query_timings(
        timings,
        token,
        started_at=started,
    )

    assert final.embedding_ms > 0
    assert final.retrieval_ms > 0
    assert final.total_ms >= (
        final.embedding_ms + final.retrieval_ms
    )


@pytest.mark.parametrize(
    "metric",
    [
        "generation_ms",
        "synthesis_ms",
        "coverage_ms",
        "assembly_ms",
    ],
)
def test_service_proxy_records_phase_duration(metric: str) -> None:
    proxy = TimedServiceProxy(
        _SyncService(),
        metric=metric,
    )

    started = time.perf_counter()
    timings, token = begin_query_timings()

    assert proxy.run() == "ok"

    final = finish_query_timings(
        timings,
        token,
        started_at=started,
    )

    assert getattr(final, metric) > 0


def test_offline_concurrent_http_requests_do_not_bleed_state() -> None:
    async def run() -> None:
        app = _app()
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            questions = [
                f"Concurrent consciousness question {index}"
                for index in range(8)
            ]

            responses = await asyncio.gather(
                *[
                    client.post(
                        "/api/query",
                        json={"question": question},
                    )
                    for question in questions
                ]
            )

        assert all(
            response.status_code == 200
            for response in responses
        )

        request_ids = {
            response.headers["X-Request-ID"]
            for response in responses
        }
        assert len(request_ids) == len(responses)

        echoed_questions = [
            response.json()["question"]
            for response in responses
        ]
        assert echoed_questions == questions

        for response in responses:
            server_timing = response.headers.get(
                "Server-Timing"
            )
            assert server_timing is not None
            assert "total;dur=" in server_timing

    asyncio.run(run())
