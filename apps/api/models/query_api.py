"""Frozen public API contracts for ``POST /api/query``.

Stage 4.1 defines only the HTTP boundary. It does not execute or reimplement
the WTH reasoning pipeline.

Contract decisions
------------------
Request
    ``{"question": "..."}``
    - trimmed
    - 3..1000 characters
    - extra fields rejected

Success
    HTTP 200 with the canonical Phase 18 ``FinalResponse`` as the response
    body. There is deliberately no second public success DTO and no
    ``{"final_response": ...}`` wrapper.

Out of Corpus
    A valid pipeline outcome, not an HTTP error. It returns HTTP 200 using the
    same ``FinalResponse`` contract. Phase 18/17 policy remains authoritative:
    corpus-grounded content is blocked when coverage is Out of Corpus, WTH
    citations are not attached to general-knowledge fallback, and Phase 18
    itself does not generate general-knowledge prose.

Errors
    All controlled non-success responses use ``QueryApiErrorResponse``.
    Raw provider bodies, prompts, stack traces, API keys, and internal
    exception strings must never be returned to the browser.

Timeout
    One request has a 120-second application execution budget covering the
    complete Phase 14 -> 18 pipeline, including internal provider retries.
    Timeout returns HTTP 504 and never returns a partial FinalResponse.

Citations
    Reuse Phase 18 exactly. ``claim_level_citations`` is the canonical citation
    registry. Each entry is a ``FinalCitation`` containing ``citation_ref``,
    ``chunk_id``, ``source_id``, ``citation``, ``corpus_version``, and
    ``domain``. ``citation_ref`` is response-scoped; clients must not treat it
    as a permanent database identifier.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from apps.api.models.query_execution import QueryExecutionRequest
from apps.api.models.runtime_contracts import FinalCitation, FinalResponse

QUERY_API_VERSION: Final = "v1"
QUERY_API_PATH: Final = "/api/query"

QUESTION_MIN_LENGTH: Final = 3
QUESTION_MAX_LENGTH: Final = 1_000

# This is the public request budget, not an individual provider HTTP timeout.
# Stage 4.2 should enforce it around QueryOrchestrator.execute().
QUERY_TIMEOUT_SECONDS: Final = 120.0

HTTP_STATUS_SUCCESS: Final = 200
HTTP_STATUS_REQUEST_TOO_LARGE: Final = 413
HTTP_STATUS_VALIDATION_ERROR: Final = 422
HTTP_STATUS_RATE_LIMITED: Final = 429
HTTP_STATUS_INTERNAL_ERROR: Final = 500
HTTP_STATUS_UPSTREAM_ERROR: Final = 502
HTTP_STATUS_DEPENDENCY_UNAVAILABLE: Final = 503
HTTP_STATUS_TIMEOUT: Final = 504

OUT_OF_CORPUS_HTTP_STATUS: Final = HTTP_STATUS_SUCCESS


class QueryApiContract(BaseModel):
    """Strict base model for the public query HTTP boundary."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )


class QueryApiRequest(QueryExecutionRequest):
    """Public request accepted by ``POST /api/query``.

    The existing orchestration request contract remains authoritative for
    normalization/non-empty validation. Stage 4 adds the public size limits
    required before the request enters the provider-backed pipeline.
    """

    question: str = Field(
        min_length=QUESTION_MIN_LENGTH,
        max_length=QUESTION_MAX_LENGTH,
    )


# Success is intentionally the frozen Phase 18 contract itself.
QueryApiSuccessResponse: TypeAlias = FinalResponse
QueryApiCitation: TypeAlias = FinalCitation


class QueryApiPhase(StrEnum):
    """Stable public phase labels used only for controlled error context."""

    RETRIEVAL = "retrieval"
    DOMAIN_GENERATION = "domain_generation"
    SYNTHESIS = "synthesis"
    COVERAGE = "coverage"
    RESPONSE_ASSEMBLY = "response_assembly"


class QueryApiErrorCode(StrEnum):
    """Stable machine-readable error codes for the frontend."""

    INVALID_REQUEST = "invalid_request"
    REQUEST_TOO_LARGE = "request_too_large"
    API_RATE_LIMITED = "api_rate_limited"
    PROVIDER_RATE_LIMITED = "provider_rate_limited"
    UPSTREAM_PROVIDER_ERROR = "upstream_provider_error"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    QUERY_TIMEOUT = "query_timeout"
    PIPELINE_INVARIANT_FAILED = "pipeline_invariant_failed"
    INTERNAL_ERROR = "internal_error"


class QueryApiError(QueryApiContract):
    """Controlled error payload safe to expose to the browser."""

    code: QueryApiErrorCode
    message: str = Field(
        min_length=1,
        max_length=512,
    )
    retryable: bool
    phase: QueryApiPhase | None = None
    retry_after_seconds: float | None = Field(
        default=None,
        ge=0.0,
    )


class QueryApiErrorResponse(QueryApiContract):
    """Uniform envelope for all controlled non-2xx responses."""

    request_id: str = Field(
        min_length=8,
        max_length=128,
    )
    error: QueryApiError


ERROR_HTTP_STATUS: Final[dict[QueryApiErrorCode, int]] = {
    QueryApiErrorCode.INVALID_REQUEST: (HTTP_STATUS_VALIDATION_ERROR),
    QueryApiErrorCode.REQUEST_TOO_LARGE: (HTTP_STATUS_REQUEST_TOO_LARGE),
    QueryApiErrorCode.API_RATE_LIMITED: (HTTP_STATUS_RATE_LIMITED),
    QueryApiErrorCode.PROVIDER_RATE_LIMITED: (HTTP_STATUS_RATE_LIMITED),
    QueryApiErrorCode.UPSTREAM_PROVIDER_ERROR: (HTTP_STATUS_UPSTREAM_ERROR),
    QueryApiErrorCode.DEPENDENCY_UNAVAILABLE: (HTTP_STATUS_DEPENDENCY_UNAVAILABLE),
    QueryApiErrorCode.QUERY_TIMEOUT: (HTTP_STATUS_TIMEOUT),
    QueryApiErrorCode.PIPELINE_INVARIANT_FAILED: (HTTP_STATUS_INTERNAL_ERROR),
    QueryApiErrorCode.INTERNAL_ERROR: (HTTP_STATUS_INTERNAL_ERROR),
}


__all__ = [
    "ERROR_HTTP_STATUS",
    "HTTP_STATUS_DEPENDENCY_UNAVAILABLE",
    "HTTP_STATUS_INTERNAL_ERROR",
    "HTTP_STATUS_RATE_LIMITED",
    "HTTP_STATUS_REQUEST_TOO_LARGE",
    "HTTP_STATUS_SUCCESS",
    "HTTP_STATUS_TIMEOUT",
    "HTTP_STATUS_UPSTREAM_ERROR",
    "HTTP_STATUS_VALIDATION_ERROR",
    "OUT_OF_CORPUS_HTTP_STATUS",
    "QUERY_API_PATH",
    "QUERY_API_VERSION",
    "QUERY_TIMEOUT_SECONDS",
    "QUESTION_MAX_LENGTH",
    "QUESTION_MIN_LENGTH",
    "QueryApiCitation",
    "QueryApiContract",
    "QueryApiError",
    "QueryApiErrorCode",
    "QueryApiErrorResponse",
    "QueryApiPhase",
    "QueryApiRequest",
    "QueryApiSuccessResponse",
]
