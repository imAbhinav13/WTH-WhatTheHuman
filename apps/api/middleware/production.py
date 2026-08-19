"""Stage 6.1B public-edge middleware.

Controls:
- exact request-body ceiling for POST /api/query;
- small, single-instance in-memory IP rate limits;
- structured request completion logs.

This is intentionally not a distributed rate limiter. It is appropriate for
the initial single-instance Render deployment. A shared store can replace it
later if WTH scales horizontally.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from collections.abc import MutableMapping
from typing import Any
from uuid import uuid4

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

LOGGER = logging.getLogger("wth.api.access")

QUERY_PATH = "/api/query"
CHUNK_PREFIX = "/api/chunk/"
REQUEST_ID_HEADER = b"x-request-id"


class RequestBodyTooLargeError(RuntimeError):
    """Internal sentinel raised when a streamed request exceeds the ceiling."""


class QueryBodySizeLimitMiddleware:
    """Enforce a byte ceiling on POST /api/query, including chunked bodies."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_bytes: int,
    ) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != QUERY_PATH
        ):
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        content_length = headers.get("content-length")

        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                declared = 0

            if declared > self.max_bytes:
                await _send_json_error(
                    send,
                    status=413,
                    code="request_too_large",
                    message="The query request body is too large.",
                    retryable=False,
                )
                return

        seen = 0

        async def limited_receive() -> Message:
            nonlocal seen
            message = await receive()

            if message["type"] == "http.request":
                body = message.get("body", b"")
                seen += len(body)

                if seen > self.max_bytes:
                    raise RequestBodyTooLargeError

            return message

        try:
            await self.app(scope, limited_receive, send)
        except RequestBodyTooLargeError:
            await _send_json_error(
                send,
                status=413,
                code="request_too_large",
                message="The query request body is too large.",
                retryable=False,
            )


class InMemoryRateLimitMiddleware:
    """Simple fixed-window-like sliding limiter for the initial deployment."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        query_requests: int,
        query_window_seconds: int,
        chunk_requests: int,
        chunk_window_seconds: int,
    ) -> None:
        self.app = app
        self.query_requests = query_requests
        self.query_window_seconds = query_window_seconds
        self.chunk_requests = chunk_requests
        self.chunk_window_seconds = chunk_window_seconds
        self._events: MutableMapping[
            tuple[str, str],
            deque[float],
        ] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = str(scope.get("method", ""))
        path = str(scope.get("path", ""))

        if method == "POST" and path == QUERY_PATH:
            bucket = "query"
            limit = self.query_requests
            window = self.query_window_seconds
        elif method == "GET" and path.startswith(CHUNK_PREFIX):
            bucket = "chunk"
            limit = self.chunk_requests
            window = self.chunk_window_seconds
        else:
            await self.app(scope, receive, send)
            return

        client_key = _client_key(scope)

        allowed, retry_after = await self._consume(
            key=(bucket, client_key),
            limit=limit,
            window_seconds=window,
        )

        if not allowed:
            await _send_json_error(
                send,
                status=429,
                code="api_rate_limited",
                message="Too many requests. Please try again later.",
                retryable=True,
                retry_after_seconds=retry_after,
                extra_headers=[
                    (
                        b"retry-after",
                        str(max(1, int(retry_after + 0.999))).encode("ascii"),
                    ),
                ],
            )
            return

        await self.app(scope, receive, send)

    async def _consume(
        self,
        *,
        key: tuple[str, str],
        limit: int,
        window_seconds: int,
    ) -> tuple[bool, float]:
        now = time.monotonic()
        cutoff = now - window_seconds

        async with self._lock:
            events = self._events[key]

            while events and events[0] <= cutoff:
                events.popleft()

            if len(events) >= limit:
                retry_after = max(
                    0.0,
                    window_seconds - (now - events[0]),
                )
                return False, retry_after

            events.append(now)
            return True, 0.0


class StructuredAccessLogMiddleware:
    """Emit one sanitized structured completion event per HTTP request."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        status_code = 500
        response_request_id: str | None = None

        async def logging_send(message: Message) -> None:
            nonlocal status_code, response_request_id

            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                headers = list(message.get("headers", []))

                for name, value in headers:
                    if name.lower() == REQUEST_ID_HEADER:
                        response_request_id = value.decode(
                            "latin-1",
                            errors="replace",
                        )
                        break

            await send(message)

        try:
            await self.app(scope, receive, logging_send)
        finally:
            elapsed_ms = round(
                (time.perf_counter() - started) * 1000.0,
                2,
            )

            event = {
                "event": "http_request_complete",
                "request_id": response_request_id,
                "method": scope.get("method"),
                "path": scope.get("path"),
                "status": status_code,
                "elapsed_ms": elapsed_ms,
            }

            # No headers, request body, query text, provider body, or secrets.
            LOGGER.info(
                "%s",
                json.dumps(
                    event,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            )


def _client_key(scope: Scope) -> str:
    headers = Headers(scope=scope)
    forwarded = headers.get("x-forwarded-for")

    if forwarded:
        # Render/reverse proxies normally append the original client first.
        first = forwarded.split(",", 1)[0].strip()
        if first:
            return first

    client = scope.get("client")
    if client:
        return str(client[0])

    return "unknown"


async def _send_json_error(
    send: Send,
    *,
    status: int,
    code: str,
    message: str,
    retryable: bool,
    retry_after_seconds: float | None = None,
    extra_headers: list[tuple[bytes, bytes]] | None = None,
) -> None:
    request_id = f"req_{uuid4().hex}"
    payload: dict[str, Any] = {
        "request_id": request_id,
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
            "phase": None,
            "retry_after_seconds": (
                round(retry_after_seconds, 3)
                if retry_after_seconds is not None
                else None
            ),
        },
    }

    body = json.dumps(
        payload,
        separators=(",", ":"),
    ).encode("utf-8")

    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
        (REQUEST_ID_HEADER, request_id.encode("ascii")),
    ]
    if extra_headers:
        headers.extend(extra_headers)

    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": headers,
        }
    )
    await send(
        {
            "type": "http.response.body",
            "body": body,
            "more_body": False,
        }
    )


__all__ = [
    "InMemoryRateLimitMiddleware",
    "QueryBodySizeLimitMiddleware",
    "StructuredAccessLogMiddleware",
]
