"""Stage 6.1A public API contract regression tests."""

from __future__ import annotations

import os

os.environ["PROVIDER_MODE"] = "mock"

from apps.api.core.config import get_settings

get_settings.cache_clear()

from apps.api.main import create_app


EXPECTED_PUBLIC_METHODS = {
    "/api/query": {"post"},
    "/api/chunk/{chunk_id}": {"get"},
    "/api/health": {"get"},
    "/api/ready": {"get"},
}


def test_expected_public_api_surface_is_present() -> None:
    app = create_app()
    paths = app.openapi()["paths"]

    for path, methods in EXPECTED_PUBLIC_METHODS.items():
        assert path in paths

        actual_methods = {
            method.lower()
            for method in paths[path]
            if method.lower()
            in {
                "get",
                "post",
                "put",
                "patch",
                "delete",
                "options",
                "head",
            }
        }

        assert methods <= actual_methods


def test_historical_api_v1_surface_is_absent() -> None:
    app = create_app()
    paths = app.openapi()["paths"]

    legacy_paths = [
        path
        for path in paths
        if path.startswith("/api/v1")
    ]

    assert legacy_paths == []


def test_no_duplicate_public_method_path_registrations() -> None:
    app = create_app()

    registrations: list[tuple[str, str]] = []

    for route in app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)

        if not isinstance(path, str) or not methods:
            continue

        if not path.startswith("/api/"):
            continue

        for method in methods:
            if method in {"HEAD", "OPTIONS"}:
                continue
            registrations.append((method, path))

    assert len(registrations) == len(set(registrations))


def test_frontend_facing_route_names_are_stable() -> None:
    app = create_app()
    paths = set(app.openapi()["paths"])

    assert {
        "/api/query",
        "/api/chunk/{chunk_id}",
    } <= paths

    assert "/api/v1/query" not in paths
    assert "/api/v1/chunk/{chunk_id}" not in paths
