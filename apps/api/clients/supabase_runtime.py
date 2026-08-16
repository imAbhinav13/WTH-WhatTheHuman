from __future__ import annotations

from functools import lru_cache

from supabase import Client, create_client

from apps.api.core.config import get_settings


class SupabaseRuntimeConfigurationError(RuntimeError):
    """Raised when the backend Supabase runtime configuration is invalid."""


@lru_cache(maxsize=1)
def get_supabase_runtime_client() -> Client:
    """Return the backend-only Supabase client used by runtime repositories.

    The secret/service-role credential never leaves the backend process.
    """
    settings = get_settings()

    if settings.supabase_url is None:
        raise SupabaseRuntimeConfigurationError(
            "SUPABASE_URL is not configured."
        )

    if settings.supabase_secret_key is None:
        raise SupabaseRuntimeConfigurationError(
            "SUPABASE_SECRET_KEY is not configured."
        )

    url = str(settings.supabase_url).strip()
    secret_key = settings.supabase_secret_key.get_secret_value().strip()

    if not url:
        raise SupabaseRuntimeConfigurationError(
            "SUPABASE_URL is not configured."
        )

    if not secret_key:
        raise SupabaseRuntimeConfigurationError(
            "SUPABASE_SECRET_KEY is not configured."
        )

    return create_client(url, secret_key)