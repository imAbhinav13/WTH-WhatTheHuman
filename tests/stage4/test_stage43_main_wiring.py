"""FastAPI registration tests after Stage 6.1A route cleanup.

No lifespan is entered, so these tests perform no external provider calls.
"""

from __future__ import annotations

import os

os.environ["PROVIDER_MODE"] = "mock"

from fastapi.exceptions import RequestValidationError

from apps.api.core.config import get_settings

get_settings.cache_clear()

from apps.api.main import create_app
from apps.api.routers.query import (
    query_request_validation_exception_handler,
)


def test_create_app_registers_public_query_route() -> None:
    app = create_app()
    paths = app.openapi()["paths"]

    assert "/api/query" in paths
    assert "post" in paths["/api/query"]


def test_create_app_registers_clean_public_routes() -> None:
    app = create_app()
    paths = app.openapi()["paths"]

    assert "/api/health" in paths
    assert "get" in paths["/api/health"]

    assert "/api/ready" in paths
    assert "get" in paths["/api/ready"]

    assert "/api/chunk/{chunk_id}" in paths
    assert "get" in paths["/api/chunk/{chunk_id}"]


def test_create_app_does_not_register_api_v1_routes() -> None:
    app = create_app()
    paths = app.openapi()["paths"]

    assert not any(
        path.startswith("/api/v1")
        for path in paths
    )


def test_ready_route_is_registered_exactly_once() -> None:
    app = create_app()

    schema = app.openapi()
    paths = schema["paths"]

    assert "/api/ready" in paths

    ready_operations = paths["/api/ready"]

    assert "get" in ready_operations

    http_methods = {
        method
        for method in ready_operations
        if method
        in {
            "get",
            "post",
            "put",
            "patch",
            "delete",
            "head",
            "options",
        }
    }

    assert http_methods == {"get"}


def test_create_app_registers_controlled_query_validation_handler() -> None:
    app = create_app()

    assert (
        app.exception_handlers[
            RequestValidationError
        ]
        is query_request_validation_exception_handler
    )
