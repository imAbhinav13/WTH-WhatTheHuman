"""Print the Stage 6.1C public API/security checklist from OpenAPI.

This script performs no provider calls.
"""

from __future__ import annotations

import os

os.environ.setdefault("PROVIDER_MODE", "mock")

from apps.api.core.config import get_settings
from apps.api.main import create_app

get_settings.cache_clear()


def main() -> int:
    app = create_app()
    paths = app.openapi()["paths"]

    checks = {
        "POST /api/query": ("/api/query" in paths and "post" in paths["/api/query"]),
        "GET /api/chunk/{chunk_id}": (
            "/api/chunk/{chunk_id}" in paths and "get" in paths["/api/chunk/{chunk_id}"]
        ),
        "GET /api/health": ("/api/health" in paths and "get" in paths["/api/health"]),
        "GET /api/ready": ("/api/ready" in paths and "get" in paths["/api/ready"]),
        "No /api/v1 routes": not any(path.startswith("/api/v1") for path in paths),
        "Query documents 413": ("413" in paths["/api/query"]["post"]["responses"]),
        "Query documents 429": ("429" in paths["/api/query"]["post"]["responses"]),
    }

    print("Stage 6.1C API/security contract review")
    failed = False

    for label, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL':<5} {label}")
        failed = failed or not passed

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
