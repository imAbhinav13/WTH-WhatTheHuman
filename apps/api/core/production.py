"""Stage 6.1B production policy and startup validation.

This module deliberately sits beside the existing Settings model instead of
expanding it during deployment hardening. Existing Phase 0 configuration
remains authoritative; Stage 6-only controls are read from environment
variables with conservative defaults.

No secret value is ever included in a validation error.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from pydantic import SecretStr

DEFAULT_MAX_QUERY_BODY_BYTES = 16 * 1024
DEFAULT_QUERY_RATE_LIMIT_REQUESTS = 5
DEFAULT_QUERY_RATE_LIMIT_WINDOW_SECONDS = 10 * 60
DEFAULT_CHUNK_RATE_LIMIT_REQUESTS = 60
DEFAULT_CHUNK_RATE_LIMIT_WINDOW_SECONDS = 60


class ProductionConfigurationError(RuntimeError):
    """Raised when a production deployment is unsafe or incomplete."""


@dataclass(frozen=True, slots=True)
class ProductionPolicy:
    production: bool
    cors_origins: tuple[str, ...]
    max_query_body_bytes: int
    query_rate_limit_requests: int
    query_rate_limit_window_seconds: int
    chunk_rate_limit_requests: int
    chunk_rate_limit_window_seconds: int


def build_production_policy(settings: object) -> ProductionPolicy:
    """Build Stage 6 policy and fail fast on unsafe production settings."""

    app_env = str(getattr(settings, "app_env", "")).strip().lower()
    production = app_env == "production"

    origins = _normalize_origins(
        getattr(settings, "cors_origins", ())
    )

    policy = ProductionPolicy(
        production=production,
        cors_origins=origins,
        max_query_body_bytes=_positive_int_env(
            "WTH_MAX_QUERY_BODY_BYTES",
            DEFAULT_MAX_QUERY_BODY_BYTES,
        ),
        query_rate_limit_requests=_positive_int_env(
            "WTH_QUERY_RATE_LIMIT_REQUESTS",
            DEFAULT_QUERY_RATE_LIMIT_REQUESTS,
        ),
        query_rate_limit_window_seconds=_positive_int_env(
            "WTH_QUERY_RATE_LIMIT_WINDOW_SECONDS",
            DEFAULT_QUERY_RATE_LIMIT_WINDOW_SECONDS,
        ),
        chunk_rate_limit_requests=_positive_int_env(
            "WTH_CHUNK_RATE_LIMIT_REQUESTS",
            DEFAULT_CHUNK_RATE_LIMIT_REQUESTS,
        ),
        chunk_rate_limit_window_seconds=_positive_int_env(
            "WTH_CHUNK_RATE_LIMIT_WINDOW_SECONDS",
            DEFAULT_CHUNK_RATE_LIMIT_WINDOW_SECONDS,
        ),
    )

    if production:
        _validate_production_settings(
            settings=settings,
            policy=policy,
        )

    return policy


def _validate_production_settings(
    *,
    settings: object,
    policy: ProductionPolicy,
) -> None:
    provider_mode = getattr(
        getattr(settings, "provider_mode", None),
        "value",
        getattr(settings, "provider_mode", ""),
    )

    problems: list[str] = []

    if str(provider_mode).strip().lower() != "live":
        problems.append("PROVIDER_MODE must be 'live'")

    required = (
        ("SUPABASE_URL", getattr(settings, "supabase_url", None)),
        (
            "SUPABASE_SECRET_KEY",
            getattr(settings, "supabase_secret_key", None),
        ),
        ("GROQ_API_KEY", getattr(settings, "groq_api_key", None)),
        ("GOOGLE_API_KEY", getattr(settings, "google_api_key", None)),
    )

    for field_name, value in required:
        if not _configured(value):
            problems.append(f"{field_name} is required")

    if not policy.cors_origins:
        problems.append("CORS_ORIGINS must contain at least one frontend origin")

    for origin in policy.cors_origins:
        if origin == "*":
            problems.append("CORS_ORIGINS must not contain '*' in production")
            continue

        parsed = urlparse(origin)
        if parsed.scheme != "https" or not parsed.netloc:
            problems.append(
                "production CORS origins must be absolute HTTPS origins"
            )

    if problems:
        # Intentionally report field names/rules only, never configured values.
        joined = "; ".join(dict.fromkeys(problems))
        raise ProductionConfigurationError(
            f"Unsafe or incomplete production configuration: {joined}"
        )


def _normalize_origins(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()

    if isinstance(raw, str):
        values = [
            item.strip()
            for item in raw.split(",")
            if item.strip()
        ]
    else:
        values = [
            str(item).strip()
            for item in raw
            if str(item).strip()
        ]

    return tuple(dict.fromkeys(values))


def _configured(value: object) -> bool:
    if value is None:
        return False

    raw = value.get_secret_value() if isinstance(value, SecretStr) else str(value)

    normalized = raw.strip().lower()

    if not normalized:
        return False

    placeholders = (
        "your-",
        "replace-me",
        "changeme",
        "example",
        "placeholder",
    )
    return not any(
        marker in normalized
        for marker in placeholders
    )


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)

    if raw is None or not raw.strip():
        return default

    try:
        value = int(raw)
    except ValueError as exc:
        raise ProductionConfigurationError(
            f"{name} must be a positive integer"
        ) from exc

    if value <= 0:
        raise ProductionConfigurationError(
            f"{name} must be a positive integer"
        )

    return value


__all__ = [
    "DEFAULT_CHUNK_RATE_LIMIT_REQUESTS",
    "DEFAULT_CHUNK_RATE_LIMIT_WINDOW_SECONDS",
    "DEFAULT_MAX_QUERY_BODY_BYTES",
    "DEFAULT_QUERY_RATE_LIMIT_REQUESTS",
    "DEFAULT_QUERY_RATE_LIMIT_WINDOW_SECONDS",
    "ProductionConfigurationError",
    "ProductionPolicy",
    "build_production_policy",
]
