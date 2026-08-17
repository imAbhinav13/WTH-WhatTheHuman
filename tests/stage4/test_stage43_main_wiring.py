"""Stage 4.3 FastAPI registration tests.

No lifespan is entered and no external provider calls are made.
"""

from __future__ import annotations

import os

# IMPORTANT:
# apps.api.main creates `app = create_app()` at import time.
# Force mock mode before importing it so test collection never requires
# GOOGLE_API_KEY / GROQ_API_KEY / Supabase credentials.
os.environ["PROVIDER_MODE"] = "mock"

from fastapi.exceptions import RequestValidationError

from apps.api.core.config import get_settings

# Clear any Settings object cached by earlier imports/tests.
get_settings.cache_clear()

from apps.api.main import create_app
from apps.api.routers.query import (
    query_request_validation_exception_handler,
)


def test_create_app_registers_public_query_route() -> None:
    app = create_app()

    schema = app.openapi()

    paths = schema["paths"]

    assert "/api/query" in paths
    assert "post" in paths["/api/query"]

def test_create_app_keeps_existing_stage2_routes() -> None:
    app = create_app()

    schema = app.openapi()

    paths = schema["paths"]

    assert "/api/v1/health" in paths
    assert "/api/v1/ready" in paths
    assert "/api/v1/chunk/{chunk_id}" in paths

def test_create_app_registers_controlled_query_validation_handler() -> None:
    app = create_app()

    assert (
        app.exception_handlers[
            RequestValidationError
        ]
        is query_request_validation_exception_handler
    )
