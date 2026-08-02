"""Supabase and mock database-provider implementations."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from time import perf_counter
from typing import Any, Final, cast

from supabase import Client, create_client

from apps.api.clients.base import DatabaseProvider
from apps.api.models.enums import ProviderKind, ProviderStatus
from apps.api.models.providers import (
    DatabaseProbeRequest,
    DatabaseProbeResponse,
    ProviderMetadata,
)


DEFAULT_EXPECTED_CONCEPT_COUNT: Final = 8


class SupabaseDatabaseProvider(DatabaseProvider):
    """Provide database readiness checks through Supabase."""

    def __init__(
        self,
        *,
        url: str,
        secret_key: str,
    ) -> None:
        """Initialize the Supabase database provider."""

        normalized_url = url.strip()
        normalized_secret_key = secret_key.strip()

        if not normalized_url:
            raise ValueError("Supabase URL must not be empty")

        if not normalized_secret_key:
            raise ValueError(
                "Supabase secret key must not be empty"
            )

        self._client: Client = create_client(
            normalized_url,
            normalized_secret_key,
        )

    @property
    def metadata(self) -> ProviderMetadata:
        """Return Supabase provider metadata."""

        return ProviderMetadata(
            provider=ProviderKind.DATABASE,
            implementation=self.__class__.__name__,
            model=None,
            is_mock=False,
        )

    async def probe(
        self,
        request: DatabaseProbeRequest,
    ) -> DatabaseProbeResponse:
        """Verify Supabase connectivity and required reference data."""

        started_at = perf_counter()

        try:
            return await asyncio.to_thread(
                self._probe_sync,
                request,
                started_at,
            )
        except Exception as exc:  # noqa: BLE001
            return DatabaseProbeResponse(
                status=ProviderStatus.UNAVAILABLE,
                latency_ms=_elapsed_ms(started_at),
                schema_available=False,
                concept_count=None,
                active_corpus_version=None,
                detail=(
                    "Supabase database probe failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )

    def _probe_sync(
        self,
        request: DatabaseProbeRequest,
        started_at: float,
    ) -> DatabaseProbeResponse:
        """Execute the readiness checks using the synchronous SDK."""

        schema_available = True
        concept_count: int | None = None
        active_corpus_version: str | None = None
        details: list[str] = []

        if request.verify_schema:
            schema_available = self._verify_schema()

            if not schema_available:
                return DatabaseProbeResponse(
                    status=ProviderStatus.UNAVAILABLE,
                    latency_ms=_elapsed_ms(started_at),
                    schema_available=False,
                    concept_count=None,
                    active_corpus_version=None,
                    detail=(
                        "Required Supabase tables are unavailable"
                    ),
                )

        if request.verify_concepts:
            concept_count = self._count_active_concepts()

            if concept_count != request.expected_concept_count:
                details.append(
                    "Expected "
                    f"{request.expected_concept_count} active concepts, "
                    f"found {concept_count}"
                )

        active_corpus_version = self._get_latest_corpus_version()

        if active_corpus_version is None:
            details.append(
                "No corpus version could be identified"
            )

        status = (
            ProviderStatus.DEGRADED
            if details
            else ProviderStatus.READY
        )

        if not details:
            details.append(
                "Supabase schema and reference data are ready"
            )

        return DatabaseProbeResponse(
            status=status,
            latency_ms=_elapsed_ms(started_at),
            schema_available=schema_available,
            concept_count=concept_count,
            active_corpus_version=active_corpus_version,
            detail="; ".join(details),
        )

    def _verify_schema(self) -> bool:
        """Verify that all Phase 0 tables are queryable."""

        required_tables = (
            "corpus_versions",
            "sources",
            "concepts",
            "chunks",
            "chunk_concepts",
            "queries",
            "query_concepts",
            "responses",
            "response_claims",
            "retrieval_results",
            "claim_citations",
            "response_concept_coverage",
            "eval_questions",
        )

        for table_name in required_tables:
            self._client.table(table_name).select(
                "*"
            ).limit(1).execute()

        return True

    def _count_active_concepts(self) -> int:
        """Return the number of active canonical concepts."""

        response = (
            self._client.table("concepts")
            .select("id")
            .eq("is_active", True)
            .execute()
        )

        rows = _response_rows(response.data)
        return len(rows)

    def _get_latest_corpus_version(self) -> str | None:
        """Return the latest known corpus-version identifier."""

        response = (
            self._client.table("corpus_versions")
            .select("*")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )

        rows = _response_rows(response.data)

        if not rows:
            return None

        row = rows[0]

        for field_name in (
            "version",
            "version_name",
            "name",
            "slug",
            "label",
        ):
            value = row.get(field_name)

            if isinstance(value, str) and value.strip():
                return value.strip()

        identifier = row.get("id")

        if identifier is None:
            return None

        return str(identifier)

    async def close(self) -> None:
        """Release resources owned by the Supabase client.

        The synchronous Supabase client does not expose a stable public
        asynchronous close method.
        """


class MockDatabaseProvider(DatabaseProvider):
    """Provide deterministic database readiness without Supabase."""

    def __init__(
        self,
        *,
        concept_count: int = DEFAULT_EXPECTED_CONCEPT_COUNT,
        corpus_version: str = "phase0-v1",
        schema_available: bool = True,
    ) -> None:
        """Initialize the mock database provider."""

        if concept_count < 0:
            raise ValueError(
                "Mock concept count must not be negative"
            )

        if not corpus_version.strip():
            raise ValueError(
                "Mock corpus version must not be empty"
            )

        self._concept_count = concept_count
        self._corpus_version = corpus_version
        self._schema_available = schema_available

    @property
    def metadata(self) -> ProviderMetadata:
        """Return mock database-provider metadata."""

        return ProviderMetadata(
            provider=ProviderKind.DATABASE,
            implementation=self.__class__.__name__,
            model=None,
            is_mock=True,
        )

    async def probe(
        self,
        request: DatabaseProbeRequest,
    ) -> DatabaseProbeResponse:
        """Return a deterministic readiness result."""

        started_at = perf_counter()

        if request.verify_schema and not self._schema_available:
            return DatabaseProbeResponse(
                status=ProviderStatus.UNAVAILABLE,
                latency_ms=_elapsed_ms(started_at),
                schema_available=False,
                concept_count=None,
                active_corpus_version=None,
                detail="Mock database schema is unavailable",
            )

        concept_count = (
            self._concept_count
            if request.verify_concepts
            else None
        )

        if (
            request.verify_concepts
            and concept_count != request.expected_concept_count
        ):
            return DatabaseProbeResponse(
                status=ProviderStatus.DEGRADED,
                latency_ms=_elapsed_ms(started_at),
                schema_available=self._schema_available,
                concept_count=concept_count,
                active_corpus_version=self._corpus_version,
                detail=(
                    "Expected "
                    f"{request.expected_concept_count} active concepts, "
                    f"found {concept_count}"
                ),
            )

        return DatabaseProbeResponse(
            status=ProviderStatus.READY,
            latency_ms=_elapsed_ms(started_at),
            schema_available=self._schema_available,
            concept_count=concept_count,
            active_corpus_version=self._corpus_version,
            detail="Mock database provider ready",
        )


def _response_rows(
    data: object,
) -> list[Mapping[str, Any]]:
    """Normalize Supabase response data into mapping rows."""

    if not isinstance(data, list):
        return []

    rows: list[Mapping[str, Any]] = []

    for item in data:
        if isinstance(item, Mapping):
            rows.append(
                cast(Mapping[str, Any], item)
            )

    return rows


def _elapsed_ms(started_at: float) -> int:
    """Return elapsed whole milliseconds."""

    return max(
        0,
        int((perf_counter() - started_at) * 1_000),
    )