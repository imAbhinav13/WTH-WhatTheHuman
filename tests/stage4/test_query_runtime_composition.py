"""Focused Stage 4.3 composition tests.

These tests do not call Supabase, Gemini, or Groq.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import SecretStr

from apps.api.core.config import (
    ProviderMode,
    Settings,
)
from apps.api.core.query_runtime import (
    QueryRuntimeConfigurationError,
    build_query_orchestrator,
)


def _live_settings() -> Settings:
    return Settings(
        provider_mode=ProviderMode.LIVE,
        supabase_url="https://example.supabase.co",
        supabase_secret_key=SecretStr(
            "test-service-role"
        ),
        google_api_key=SecretStr(
            "test-google-key"
        ),
        groq_api_key=SecretStr(
            "test-groq-key"
        ),
    )


def test_query_runtime_uses_current_frozen_provider_split() -> None:
    fake_supabase_client = cast(
        Any,
        object(),
    )

    orchestrator = build_query_orchestrator(
        settings=_live_settings(),
        supabase_client=(
            fake_supabase_client
        ),
    )

    provider_config = (
        orchestrator._provider_config
    )

    phase15 = (
        provider_config.domain_generation
    )

    assert phase15.science.model == (
        "openai/gpt-oss-20b"
    )
    assert (
        phase15.science.reasoning_effort
        == "medium"
    )
    assert (
        phase15.science.max_completion_tokens
        == 2500
    )

    assert phase15.advaita.model == (
        "openai/gpt-oss-120b"
    )
    assert (
        phase15.advaita.reasoning_effort
        == "medium"
    )
    assert (
        phase15.advaita.max_completion_tokens
        == 3000
    )

    assert phase15.samkhya.model == (
        "openai/gpt-oss-20b"
    )
    assert (
        phase15.samkhya.reasoning_effort
        == "medium"
    )
    assert (
        phase15.samkhya.max_completion_tokens
        == 2500
    )

    phase16 = provider_config.synthesis

    assert phase16.model == (
        "openai/gpt-oss-120b"
    )
    assert (
        phase16.reasoning_effort
        == "high"
    )
    assert (
        phase16.max_completion_tokens
        == 4500
    )


def test_query_runtime_uses_frozen_gemini_embedding_config() -> None:
    fake_supabase_client = cast(
        Any,
        object(),
    )

    orchestrator = build_query_orchestrator(
        settings=_live_settings(),
        supabase_client=(
            fake_supabase_client
        ),
    )

    retrieval = (
        orchestrator._services.retrieval
    )
    config = retrieval._embedding_config

    assert config.model == "gemini-embedding-2"
    assert config.dimensions == 768


def test_query_runtime_rejects_mock_mode() -> None:
    settings = Settings(
        provider_mode=ProviderMode.MOCK,
    )

    with pytest.raises(
        QueryRuntimeConfigurationError,
        match="PROVIDER_MODE=live",
    ):
        build_query_orchestrator(
            settings=settings,
            supabase_client=cast(
                Any,
                object(),
            ),
        )


def test_query_runtime_contains_all_five_phase_services() -> None:
    orchestrator = build_query_orchestrator(
        settings=_live_settings(),
        supabase_client=cast(
            Any,
            object(),
        ),
    )

    services = orchestrator._services

    assert services.retrieval is not None
    assert services.domain_generation is not None
    assert services.synthesis is not None
    assert services.coverage is not None
    assert services.response_assembly is not None
