"""Request-local performance instrumentation for the WTH query pipeline.

Stage 5.4 keeps performance metadata outside the frozen FinalResponse contract.

Stable metrics
--------------
embedding_ms
retrieval_ms
generation_ms
synthesis_ms
coverage_ms
assembly_ms
total_ms

``retrieval_ms`` is Phase 14 time excluding the separately measured embedding
call when the retrieval service exposes its existing ``_embedding_runner``.
No reasoning, provider, retrieval, coverage, or assembly behavior is changed.
"""

from __future__ import annotations

import inspect
import logging
import time
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass
from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar, cast


LOGGER = logging.getLogger("wth.performance")

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(slots=True)
class QueryTimings:
    embedding_ms: float = 0.0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    synthesis_ms: float = 0.0
    coverage_ms: float = 0.0
    assembly_ms: float = 0.0
    total_ms: float = 0.0

    def rounded(self) -> QueryTimings:
        return QueryTimings(
            **{
                key: round(float(value), 2)
                for key, value in asdict(self).items()
            }
        )

    def as_dict(self) -> dict[str, float]:
        return {
            key: float(value)
            for key, value in asdict(self.rounded()).items()
        }

    def server_timing_header(self) -> str:
        labels = (
            ("embedding", self.embedding_ms),
            ("retrieval", self.retrieval_ms),
            ("generation", self.generation_ms),
            ("synthesis", self.synthesis_ms),
            ("coverage", self.coverage_ms),
            ("assembly", self.assembly_ms),
            ("total", self.total_ms),
        )
        return ", ".join(
            f"{name};dur={max(0.0, value):.2f}"
            for name, value in labels
        )


_CURRENT_TIMINGS: ContextVar[QueryTimings | None] = ContextVar(
    "wth_query_timings",
    default=None,
)

_LAST_TIMINGS: ContextVar[QueryTimings | None] = ContextVar(
    "wth_last_query_timings",
    default=None,
)


class PerformanceInstrumentationError(RuntimeError):
    """Raised only for invalid instrumentation composition."""


def begin_query_timings() -> tuple[QueryTimings, Token[QueryTimings | None]]:
    timings = QueryTimings()
    token = _CURRENT_TIMINGS.set(timings)
    return timings, token


def finish_query_timings(
    timings: QueryTimings,
    token: Token[QueryTimings | None],
    *,
    started_at: float,
) -> QueryTimings:
    timings.total_ms = (
        time.perf_counter() - started_at
    ) * 1000.0

    # TimedServiceProxy measures the complete Phase 14 call while the
    # embedding runner is measured separately. Report retrieval_ms as the
    # remaining non-embedding Phase 14 work to avoid double counting.
    normalize_retrieval_timing(timings)

    final = timings.rounded()
    _LAST_TIMINGS.set(final)
    _CURRENT_TIMINGS.reset(token)
    return final


def get_last_query_timings() -> QueryTimings | None:
    """Return timings for the most recent query in the current async context."""

    value = _LAST_TIMINGS.get()
    return value.rounded() if value is not None else None


def _record(metric: str, elapsed_ms: float) -> None:
    timings = _CURRENT_TIMINGS.get()
    if timings is None:
        return

    current = float(getattr(timings, metric))
    setattr(timings, metric, current + elapsed_ms)


def timed_callable(
    func: Callable[P, R],
    *,
    metric: str,
) -> Callable[P, R]:
    """Wrap a sync or async callable and accumulate request-local duration."""

    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
            started = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                _record(
                    metric,
                    (time.perf_counter() - started) * 1000.0,
                )

        return cast(Callable[P, R], async_wrapper)

    @wraps(func)
    def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
        started = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            _record(
                metric,
                (time.perf_counter() - started) * 1000.0,
            )

    return cast(Callable[P, R], sync_wrapper)


class TimedServiceProxy:
    """Transparent service proxy that times one named pipeline phase."""

    def __init__(
        self,
        service: object,
        *,
        metric: str,
    ) -> None:
        self._service = service
        self._metric = metric

    def __getattr__(self, name: str) -> object:
        value = getattr(self._service, name)
        if not callable(value):
            return value

        return timed_callable(
            cast(Callable[..., Any], value),
            metric=self._metric,
        )


def instrument_retrieval_embedding(
    retrieval_service: object,
) -> None:
    """Time the existing RetrievalService embedding runner in-place.

    RetrievalService already delegates query embedding through
    ``_embedding_runner``. Wrapping that callable lets Stage 5.4 report
    embedding separately without changing Phase 14 behavior.
    """

    runner = getattr(
        retrieval_service,
        "_embedding_runner",
        None,
    )

    if runner is None or not callable(runner):
        raise PerformanceInstrumentationError(
            "Retrieval service does not expose callable _embedding_runner."
        )

    setattr(
        retrieval_service,
        "_embedding_runner",
        timed_callable(
            cast(Callable[..., Any], runner),
            metric="embedding_ms",
        ),
    )


class PerformanceInstrumentedOrchestrator:
    """Wrap QueryOrchestrator execution with request-local timing lifecycle."""

    def __init__(self, orchestrator: object) -> None:
        execute = getattr(orchestrator, "execute", None)
        if not callable(execute):
            raise PerformanceInstrumentationError(
                "Wrapped orchestrator must expose execute()."
            )
        self._orchestrator = orchestrator

    async def execute(self, question: str) -> object:
        started = time.perf_counter()
        timings, token = begin_query_timings()

        try:
            return await cast(Any, self._orchestrator).execute(question)
        finally:
            final = finish_query_timings(
                timings,
                token,
                started_at=started,
            )
            LOGGER.info(
                "WTH query timings embedding_ms=%.2f retrieval_ms=%.2f "
                "generation_ms=%.2f synthesis_ms=%.2f coverage_ms=%.2f "
                "assembly_ms=%.2f total_ms=%.2f",
                final.embedding_ms,
                final.retrieval_ms,
                final.generation_ms,
                final.synthesis_ms,
                final.coverage_ms,
                final.assembly_ms,
                final.total_ms,
            )

    def get_last_timings(self) -> QueryTimings | None:
        return get_last_query_timings()


def normalize_retrieval_timing(
    timings: QueryTimings,
) -> QueryTimings:
    """Convert Phase-14 inclusive timing into retrieval-exclusive timing."""

    timings.retrieval_ms = max(
        0.0,
        timings.retrieval_ms - timings.embedding_ms,
    )
    return timings


__all__ = [
    "PerformanceInstrumentationError",
    "PerformanceInstrumentedOrchestrator",
    "QueryTimings",
    "TimedServiceProxy",
    "begin_query_timings",
    "finish_query_timings",
    "get_last_query_timings",
    "instrument_retrieval_embedding",
    "normalize_retrieval_timing",
    "timed_callable",
]
