"""Public ``POST /api/query`` route for the complete WTH pipeline.

Thin HTTP adapter. Stage 5.4 only adds optional Server-Timing exposure from an
instrumented orchestrator; Phase 14-18 implementation remains outside router.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from collections.abc import Iterator
from typing import cast
from uuid import uuid4

from fastapi import APIRouter, Request, Response
from fastapi.exception_handlers import (
    request_validation_exception_handler as default_request_validation_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from apps.api.models.query_api import (
    ERROR_HTTP_STATUS,
    HTTP_STATUS_DEPENDENCY_UNAVAILABLE,
    HTTP_STATUS_INTERNAL_ERROR,
    HTTP_STATUS_RATE_LIMITED,
    HTTP_STATUS_REQUEST_TOO_LARGE,
    HTTP_STATUS_SUCCESS,
    HTTP_STATUS_TIMEOUT,
    HTTP_STATUS_UPSTREAM_ERROR,
    HTTP_STATUS_VALIDATION_ERROR,
    QUERY_API_PATH,
    QUERY_TIMEOUT_SECONDS,
    QueryApiError,
    QueryApiErrorCode,
    QueryApiErrorResponse,
    QueryApiPhase,
    QueryApiRequest,
    QueryApiSuccessResponse,
)
from apps.api.services.query_orchestrator import (
    QueryOrchestrator,
    QueryPhase,
    QueryPhaseExecutionError,
    QueryPipelineError,
    QueryPipelineInvariantError,
)

LOGGER = logging.getLogger("wth.api.query")

REQUEST_ID_HEADER = "X-Request-ID"
RETRY_AFTER_HEADER = "Retry-After"
SERVER_TIMING_HEADER = "Server-Timing"

_RETRY_AFTER_PATTERNS = (
    re.compile(
        r"try\s+again\s+in\s+([0-9]+(?:\.[0-9]+)?)s",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"wait_seconds=([0-9]+(?:\.[0-9]+)?)",
        flags=re.IGNORECASE,
    ),
)

_PHASE_MAP: dict[QueryPhase, QueryApiPhase] = {
    "phase_14_retrieval": QueryApiPhase.RETRIEVAL,
    "phase_15_domain_generation": QueryApiPhase.DOMAIN_GENERATION,
    "phase_16_synthesis": QueryApiPhase.SYNTHESIS,
    "phase_17_coverage": QueryApiPhase.COVERAGE,
    "phase_18_response_assembly": QueryApiPhase.RESPONSE_ASSEMBLY,
}

router = APIRouter(
    prefix="/api",
    tags=["query"],
)


@router.post(
    "/query",
    response_model=QueryApiSuccessResponse,
    status_code=HTTP_STATUS_SUCCESS,
    summary="Execute one complete WTH query",
    responses={
        HTTP_STATUS_REQUEST_TOO_LARGE: {
            "model": QueryApiErrorResponse,
            "description": ("The request body exceeded the public API size limit."),
        },
        HTTP_STATUS_VALIDATION_ERROR: {
            "model": QueryApiErrorResponse,
            "description": "Invalid request body or question.",
        },
        HTTP_STATUS_RATE_LIMITED: {
            "model": QueryApiErrorResponse,
            "description": (
                "The WTH API or an upstream generation provider is temporarily rate limited."
            ),
        },
        HTTP_STATUS_INTERNAL_ERROR: {
            "model": QueryApiErrorResponse,
            "description": (
                "A deterministic pipeline invariant or unexpected backend error occurred."
            ),
        },
        HTTP_STATUS_UPSTREAM_ERROR: {
            "model": QueryApiErrorResponse,
            "description": ("An upstream generation provider failed to produce a usable response."),
        },
        HTTP_STATUS_DEPENDENCY_UNAVAILABLE: {
            "model": QueryApiErrorResponse,
            "description": ("A required retrieval/runtime dependency is unavailable."),
        },
        HTTP_STATUS_TIMEOUT: {
            "model": QueryApiErrorResponse,
            "description": ("The complete query exceeded the public request budget."),
        },
    },
)
async def query(
    payload: QueryApiRequest,
    request: Request,
    response: Response,
) -> QueryApiSuccessResponse | JSONResponse:
    """Execute one WTH query and return the canonical Phase 18 response."""

    request_id = _new_request_id()
    response.headers[REQUEST_ID_HEADER] = request_id

    try:
        orchestrator = _query_orchestrator(request)

        async with asyncio.timeout(QUERY_TIMEOUT_SECONDS):
            result = await orchestrator.execute(payload.question)
    except TimeoutError:
        LOGGER.warning(
            "Query timed out request_id=%s budget_seconds=%.1f",
            request_id,
            QUERY_TIMEOUT_SECONDS,
        )
        return _error_response(
            request_id=request_id,
            code=QueryApiErrorCode.QUERY_TIMEOUT,
            message=("The query exceeded the allowed execution time."),
            retryable=True,
            status_code=HTTP_STATUS_TIMEOUT,
        )

    except QueryPipelineInvariantError as exc:
        phase = _api_phase(exc.phase)
        LOGGER.exception(
            "Query pipeline invariant failed request_id=%s phase=%s invariant=%s",
            request_id,
            phase.value,
            exc.invariant,
        )
        return _error_response(
            request_id=request_id,
            code=QueryApiErrorCode.PIPELINE_INVARIANT_FAILED,
            message=("The query pipeline failed an internal validation check."),
            retryable=False,
            status_code=HTTP_STATUS_INTERNAL_ERROR,
            phase=phase,
        )

    except QueryPhaseExecutionError as exc:
        return _phase_execution_error_response(
            request_id=request_id,
            exc=exc,
        )

    except QueryPipelineError:
        # Do not emit exception text/tracebacks at the public HTTP boundary.
        # Lower layers may log sanitized operational details, while this
        # adapter records only the response-scoped request identifier.
        LOGGER.exception(
            "Unhandled query pipeline error request_id=%s",
            request_id,
        )
        return _error_response(
            request_id=request_id,
            code=QueryApiErrorCode.INTERNAL_ERROR,
            message=("The query could not be completed because of an internal pipeline error."),
            retryable=False,
            status_code=HTTP_STATUS_INTERNAL_ERROR,
        )

    except _QueryOrchestratorUnavailableError:
        LOGGER.exception(
            "QueryOrchestrator unavailable request_id=%s",
            request_id,
        )
        return _error_response(
            request_id=request_id,
            code=QueryApiErrorCode.DEPENDENCY_UNAVAILABLE,
            message=("The query service is temporarily unavailable."),
            retryable=True,
            status_code=HTTP_STATUS_DEPENDENCY_UNAVAILABLE,
        )

    except Exception:
        # Intentionally omit exc_info and exception text. An unexpected
        # exception can contain provider/database details or credentials.
        LOGGER.exception(
            "Unexpected query API failure request_id=%s",
            request_id,
        )
        return _error_response(
            request_id=request_id,
            code=QueryApiErrorCode.INTERNAL_ERROR,
            message=("The query could not be completed because of an unexpected backend error."),
            retryable=False,
            status_code=HTTP_STATUS_INTERNAL_ERROR,
        )
    else:
        _attach_server_timing(
            response=response,
            orchestrator=orchestrator,
        )

        return result.final_response


async def query_request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> Response:
    """Return Stage 4.1 controlled 422 for POST /api/query only."""

    if request.url.path != QUERY_API_PATH:
        return await default_request_validation_handler(
            request,
            exc,
        )

    request_id = _new_request_id()

    LOGGER.info(
        "Query request validation failed request_id=%s",
        request_id,
    )

    return _error_response(
        request_id=request_id,
        code=QueryApiErrorCode.INVALID_REQUEST,
        message=(
            "Invalid query request. Provide JSON containing a "
            "'question' between 3 and 1000 characters."
        ),
        retryable=False,
        status_code=HTTP_STATUS_VALIDATION_ERROR,
    )


class _QueryOrchestratorUnavailableError(RuntimeError):
    """Raised when composition did not install the orchestrator."""


def _query_orchestrator(
    request: Request,
) -> QueryOrchestrator:
    value = getattr(
        request.app.state,
        "query_orchestrator",
        None,
    )

    execute = getattr(
        value,
        "execute",
        None,
    )

    if value is None or not callable(execute):
        raise _QueryOrchestratorUnavailableError("Application state is missing query_orchestrator.")

    return cast(
        QueryOrchestrator,
        value,
    )


def _attach_server_timing(
    *,
    response: Response,
    orchestrator: object,
) -> None:
    """Attach optional Stage 5.4 timings without changing FinalResponse."""

    getter = getattr(
        orchestrator,
        "get_last_timings",
        None,
    )

    if getter is None or not callable(getter):
        return

    timings = getter()
    if timings is None:
        return

    header_builder = getattr(
        timings,
        "server_timing_header",
        None,
    )
    if header_builder is None or not callable(header_builder):
        return

    response.headers[SERVER_TIMING_HEADER] = header_builder()


def _phase_execution_error_response(
    *,
    request_id: str,
    exc: QueryPhaseExecutionError,
) -> JSONResponse:
    phase = _api_phase(exc.phase)

    if _is_rate_limited(exc):
        retry_after_seconds = _retry_after_seconds(exc)

        LOGGER.warning(
            "Query provider rate limited request_id=%s phase=%s retry_after_seconds=%s",
            request_id,
            phase.value,
            retry_after_seconds,
        )

        return _error_response(
            request_id=request_id,
            code=QueryApiErrorCode.PROVIDER_RATE_LIMITED,
            message=("A model provider is temporarily rate limited."),
            retryable=True,
            status_code=HTTP_STATUS_RATE_LIMITED,
            phase=phase,
            retry_after_seconds=retry_after_seconds,
        )

    if exc.phase == "phase_14_retrieval" and not _looks_like_phase14_contract_failure(exc):
        LOGGER.warning(
            "Query retrieval dependency failed request_id=%s",
            request_id,
        )
        return _error_response(
            request_id=request_id,
            code=QueryApiErrorCode.DEPENDENCY_UNAVAILABLE,
            message=("A retrieval dependency is temporarily unavailable."),
            retryable=True,
            status_code=HTTP_STATUS_DEPENDENCY_UNAVAILABLE,
            phase=phase,
        )

    if exc.phase in {
        "phase_15_domain_generation",
        "phase_16_synthesis",
    }:
        LOGGER.warning(
            "Query generation provider failed request_id=%s phase=%s",
            request_id,
            phase.value,
        )
        return _error_response(
            request_id=request_id,
            code=QueryApiErrorCode.UPSTREAM_PROVIDER_ERROR,
            message=("A model provider failed to produce a usable response."),
            retryable=True,
            status_code=HTTP_STATUS_UPSTREAM_ERROR,
            phase=phase,
        )

    LOGGER.error(
        "Deterministic query phase failed request_id=%s phase=%s",
        request_id,
        phase.value,
    )

    return _error_response(
        request_id=request_id,
        code=QueryApiErrorCode.PIPELINE_INVARIANT_FAILED,
        message=("The query pipeline failed an internal validation check."),
        retryable=False,
        status_code=HTTP_STATUS_INTERNAL_ERROR,
        phase=phase,
    )


def _error_response(
    *,
    request_id: str,
    code: QueryApiErrorCode,
    message: str,
    retryable: bool,
    status_code: int,
    phase: QueryApiPhase | None = None,
    retry_after_seconds: float | None = None,
) -> JSONResponse:
    expected_status = ERROR_HTTP_STATUS[code]

    if status_code != expected_status:
        raise RuntimeError(
            "Query API error/status mapping drifted from the frozen Stage 4.1 contract."
        )

    payload = QueryApiErrorResponse(
        request_id=request_id,
        error=QueryApiError(
            code=code,
            message=message,
            retryable=retryable,
            phase=phase,
            retry_after_seconds=retry_after_seconds,
        ),
    )

    headers = {
        REQUEST_ID_HEADER: request_id,
    }

    if status_code == HTTP_STATUS_RATE_LIMITED and retry_after_seconds is not None:
        headers[RETRY_AFTER_HEADER] = str(
            max(
                1,
                math.ceil(retry_after_seconds),
            )
        )

    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(
            mode="json",
        ),
        headers=headers,
    )


def _new_request_id() -> str:
    return f"req_{uuid4().hex}"


def _api_phase(
    phase: QueryPhase,
) -> QueryApiPhase:
    return _PHASE_MAP[phase]


def _exception_chain(
    exc: BaseException,
) -> Iterator[BaseException]:
    current: BaseException | None = exc
    seen: set[int] = set()

    while current is not None:
        identity = id(current)

        if identity in seen:
            return

        seen.add(identity)
        yield current

        current = current.__cause__ if current.__cause__ is not None else current.__context__


def _exception_text(
    exc: BaseException,
) -> str:
    return " ".join(str(item) for item in _exception_chain(exc) if str(item))


def _is_rate_limited(
    exc: BaseException,
) -> bool:
    text = _exception_text(exc).lower()

    markers = (
        "status=429",
        "429 too many requests",
        "rate_limit_exceeded",
        "rate limit reached",
        "rate limited",
    )

    return any(marker in text for marker in markers)


def _retry_after_seconds(
    exc: BaseException,
) -> float | None:
    text = _exception_text(exc)

    for pattern in _RETRY_AFTER_PATTERNS:
        match = pattern.search(text)

        if match is None:
            continue

        try:
            value = float(match.group(1))
        except ValueError:
            continue

        if value >= 0.0:
            return value

    return None


def _looks_like_phase14_contract_failure(
    exc: BaseException,
) -> bool:
    text = _exception_text(exc).lower()

    return "frozen stage 3.0 contract" in text or "question must be non-empty" in text


__all__ = [
    "REQUEST_ID_HEADER",
    "RETRY_AFTER_HEADER",
    "SERVER_TIMING_HEADER",
    "query",
    "query_request_validation_exception_handler",
    "router",
]
