"""Focused offline tests for Stage 5.2 core/safety regression logic."""

from __future__ import annotations

import httpx
import pytest

from scripts import run_stage5_regression as stage5


def _response(
    body: dict[str, object],
    status_code: int = 200,
) -> httpx.Response:
    request = httpx.Request(
        "POST",
        "http://test/api/query",
    )
    return httpx.Response(
        status_code,
        json=body,
        request=request,
        headers={"X-Request-ID": "req_12345678"},
    )


def _base_success(
    *,
    question: str,
    coverage_status: str = "Supported",
    concepts: list[str] | None = None,
    comparisons: int = 0,
) -> dict[str, object]:
    concepts = concepts or []

    inline_citation = {
        "chunk_id": "chunk-1",
        "source_id": "source-1",
        "citation": "Source 1",
        "corpus_version": "phase1_active_corpus_v1",
        "domain": "science",
    }

    registry_citation = {
        "citation_ref": "C1",
        **inline_citation,
    }

    return {
        "assembly_version": "test",
        "generated_at": "2026-08-17T00:00:00Z",
        "question": question,
        "corpus_version": "phase1_active_corpus_v1",
        "sections": {
            "interpretation": "test",
            "activated_concepts": [
                {
                    "concept": concept,
                    "display_name": concept,
                    "score": 1.0,
                    "activation_reason": "test",
                }
                for concept in concepts
            ],
            "domain_perspectives": {
                "science": {
                    "domain": "science",
                    "display_name": "Science",
                    "summary": "test",
                    "claims": [
                        {
                            "claim_id": "science:1",
                            "claim": "test",
                            "citation_refs": ["C1"],
                            "citations": [inline_citation],
                            "grounding_status": "grounded",
                        }
                    ],
                    "limitations": [],
                    "unsupported_aspects": [],
                },
                "advaita": {
                    "domain": "advaita",
                    "display_name": "Advaita Vedanta",
                    "summary": "test",
                    "claims": [],
                    "limitations": [],
                    "unsupported_aspects": [],
                },
                "samkhya": {
                    "domain": "samkhya",
                    "display_name": "Samkhya",
                    "summary": "test",
                    "claims": [],
                    "limitations": [],
                    "unsupported_aspects": [],
                },
            },
            "comparative_synthesis": {
                "summary": "test",
                "three_way_overview": "test",
                "comparisons": [
                    {
                        "comparison_id": f"cmp-{index}",
                        "concept": "consciousness",
                        "left_domain": "science",
                        "right_domain": "advaita",
                        "category": "partial_overlap",
                        "explanation": "test",
                        "claim_refs": [],
                        "citation_refs": ["C1"],
                        "limitations": [],
                    }
                    for index in range(comparisons)
                ],
                "non_conclusion": "test",
            },
            "key_tensions": [],
            "non_equivalences": [],
            "coverage": {
                "coverage_status": coverage_status,
                "coverage_score": 90.0,
                "coverage_reason": "test",
                "supported_concepts": concepts,
                "partially_supported_concepts": [],
                "unsupported_concepts": [],
                "covered_domains": ["science", "advaita", "samkhya"],
                "missing_domains": [],
                "hard_overrides": [],
            },
            "general_knowledge_fallback": {
                "allowed": coverage_status == "Out of Corpus",
                "must_be_clearly_labeled": True,
                "may_use_wth_corpus_citations": False,
                "generated_in_phase18": False,
                "instruction": "test",
            },
        },
        "claim_level_citations": [registry_citation],
        "validation": {
            "passed": True,
            "issue_count": 0,
            "issues": [],
            "checks": {},
        },
        "versions": {},
        "provider_calls": {},
    }


def test_multi_requires_three_concepts_and_nine_comparisons() -> None:
    case = stage5.CASES["multi"]
    body = _base_success(
        question=case.question,
        concepts=[
            "consciousness",
            "self_identity",
            "reality_appearance",
        ],
        comparisons=9,
    )

    result = stage5.classify_response(
        case=case,
        response=_response(body),
        elapsed_ms=10.0,
    )

    assert result.status is stage5.RegressionStatus.PASS
    assert result.comparison_count == 9


def test_missing_required_concept_is_true_regression() -> None:
    case = stage5.CASES["multi"]
    body = _base_success(
        question=case.question,
        concepts=[
            "consciousness",
            "self_identity",
        ],
        comparisons=9,
    )

    result = stage5.classify_response(
        case=case,
        response=_response(body),
        elapsed_ms=10.0,
    )

    assert (
        result.status
        is stage5.RegressionStatus.TRUE_REGRESSION
    )
    assert "Missing required activated concepts" in result.detail


def test_cross_domain_claim_citation_is_true_regression() -> None:
    case = stage5.CASES["consciousness"]
    body = _base_success(
        question=case.question,
        concepts=["consciousness"],
    )

    science = body["sections"]["domain_perspectives"]["science"]  # type: ignore[index]
    science["claims"][0]["citations"][0]["domain"] = "advaita"  # type: ignore[index]

    result = stage5.classify_response(
        case=case,
        response=_response(body),
        elapsed_ms=10.0,
    )

    assert (
        result.status
        is stage5.RegressionStatus.TRUE_REGRESSION
    )


def test_out_of_corpus_requires_no_phase18_fallback_generation() -> None:
    case = stage5.CASES["out_of_corpus"]
    body = _base_success(
        question=case.question,
        coverage_status="Out of Corpus",
    )

    fallback = body["sections"]["general_knowledge_fallback"]  # type: ignore[index]
    fallback["generated_in_phase18"] = True  # type: ignore[index]

    result = stage5.classify_response(
        case=case,
        response=_response(body),
        elapsed_ms=10.0,
    )

    assert (
        result.status
        is stage5.RegressionStatus.TRUE_REGRESSION
    )
    assert "general-knowledge fallback" in result.detail


def test_atman_purusha_false_equivalence_is_true_regression() -> None:
    case = stage5.CASES["atman_purusha"]
    body = _base_success(
        question=case.question,
        concepts=["self_identity"],
    )
    body["sections"]["interpretation"] = (  # type: ignore[index]
        "Atman is the same concept as Purusha."
    )

    result = stage5.classify_response(
        case=case,
        response=_response(body),
        elapsed_ms=10.0,
    )

    assert (
        result.status
        is stage5.RegressionStatus.TRUE_REGRESSION
    )


def test_maya_science_proof_is_true_regression() -> None:
    case = stage5.CASES["maya_science"]
    body = _base_success(
        question=case.question,
        coverage_status="Partially Supported",
    )
    body["sections"]["interpretation"] = (  # type: ignore[index]
        "Scientific evidence proves Maya."
    )

    result = stage5.classify_response(
        case=case,
        response=_response(body),
        elapsed_ms=10.0,
    )

    assert (
        result.status
        is stage5.RegressionStatus.TRUE_REGRESSION
    )


def test_429_still_remains_capacity_blocked() -> None:
    case = stage5.CASES["atman_purusha"]
    response = _response(
        {
            "request_id": "req_12345678",
            "error": {
                "code": "provider_rate_limited",
                "message": "rate limited",
                "retryable": True,
                "phase": "synthesis",
                "retry_after_seconds": 12.0,
            },
        },
        status_code=429,
    )

    result = stage5.classify_response(
        case=case,
        response=response,
        elapsed_ms=10.0,
    )

    assert (
        result.status
        is stage5.RegressionStatus.PROVIDER_CAPACITY_BLOCKED
    )


def test_inline_citation_must_match_registry_entry() -> None:
    case = stage5.CASES["consciousness"]
    body = _base_success(
        question=case.question,
        concepts=["consciousness"],
    )

    science = body["sections"]["domain_perspectives"]["science"]  # type: ignore[index]
    science["claims"][0]["citations"][0]["chunk_id"] = "wrong-chunk"  # type: ignore[index]

    result = stage5.classify_response(
        case=case,
        response=_response(body),
        elapsed_ms=10.0,
    )

    assert (
        result.status
        is stage5.RegressionStatus.TRUE_REGRESSION
    )
    assert "does not match registry entry" in result.detail
