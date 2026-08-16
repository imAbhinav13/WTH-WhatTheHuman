from __future__ import annotations

from dataclasses import dataclass

from supabase import Client


class ChunkRepositoryError(RuntimeError):
    """Raised when the live corpus cannot be read safely."""


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    chunk_id: str
    source_id: str
    domain: str
    text: str
    citation: str
    corpus_version: str


@dataclass(frozen=True, slots=True)
class ReadinessRecord:
    corpus_version: str
    sample_chunk_id: str


class ChunkRepository:
    """Read-only access to the active production corpus."""

    def __init__(self, client: Client) -> None:
        self._client = client

    def get_active_chunk(self, chunk_id: str) -> ChunkRecord | None:
        try:
            chunk_response = (
                self._client.table("chunks")
                .select("id,source_id,domain,citation,full_text,review_status")
                .eq("id", chunk_id)
                .eq("review_status", "active")
                .maybe_single()
                .execute()
            )
        except Exception as exc:
            raise ChunkRepositoryError("Chunk lookup failed.") from exc

        if chunk_response is None:
            raise ChunkRepositoryError("Chunk query returned no API response.")

        chunk = chunk_response.data
        if chunk is None:
            return None
        if not isinstance(chunk, dict):
            raise ChunkRepositoryError("Chunk lookup returned an invalid payload.")

        source_id = _required_string(chunk, "source_id")

        try:
            source_response = (
                self._client.table("sources")
                .select("id,corpus_version_id")
                .eq("id", source_id)
                .maybe_single()
                .execute()
            )
        except Exception as exc:
            raise ChunkRepositoryError("Source lookup failed.") from exc

        if source_response is None:
            raise ChunkRepositoryError("Source query returned no API response.")

        source = source_response.data
        if source is None or not isinstance(source, dict):
            raise ChunkRepositoryError("Active chunk has no resolvable source.")

        corpus_version_id = _required_string(source, "corpus_version_id")

        try:
            version_response = (
                self._client.table("corpus_versions")
                .select("id,version,is_active")
                .eq("id", corpus_version_id)
                .maybe_single()
                .execute()
            )
        except Exception as exc:
            raise ChunkRepositoryError("Corpus version lookup failed.") from exc

        if version_response is None:
            raise ChunkRepositoryError("Corpus version query returned no API response.")

        version = version_response.data
        if version is None or not isinstance(version, dict):
            raise ChunkRepositoryError("Active chunk has no resolvable corpus version.")

        if version.get("is_active") is not True:
            return None

        return ChunkRecord(
            chunk_id=_required_string(chunk, "id"),
            source_id=source_id,
            domain=_required_string(chunk, "domain"),
            text=_required_string(chunk, "full_text"),
            citation=_required_string(chunk, "citation"),
            corpus_version=_required_string(version, "version"),
        )

    def check_ready(self) -> ReadinessRecord:
        """Prove that the active corpus is reachable through the live DB."""

        try:
            version_response = (
                self._client.table("corpus_versions")
                .select("id,version,is_active")
                .eq("is_active", True)
                .maybe_single()
                .execute()
            )
        except Exception as exc:
            raise ChunkRepositoryError("Readiness corpus-version lookup failed.") from exc

        if version_response is None:
            raise ChunkRepositoryError("Readiness corpus-version query returned no API response.")

        version = version_response.data
        if version is None or not isinstance(version, dict):
            raise ChunkRepositoryError("No active corpus version is available.")

        corpus_version_id = _required_string(version, "id")
        corpus_version = _required_string(version, "version")

        try:
            source_response = (
                self._client.table("sources")
                .select("id")
                .eq("corpus_version_id", corpus_version_id)
                .limit(1)
                .execute()
            )
        except Exception as exc:
            raise ChunkRepositoryError("Readiness source lookup failed.") from exc

        if source_response is None:
            raise ChunkRepositoryError("Readiness source query returned no API response.")

        source_rows = source_response.data
        if not isinstance(source_rows, list) or not source_rows:
            raise ChunkRepositoryError("Active corpus version has no sources.")
        source = source_rows[0]
        if not isinstance(source, dict):
            raise ChunkRepositoryError("Readiness source payload is invalid.")
        source_id = _required_string(source, "id")

        try:
            chunk_response = (
                self._client.table("chunks")
                .select("id")
                .eq("source_id", source_id)
                .eq("review_status", "active")
                .limit(1)
                .execute()
            )
        except Exception as exc:
            raise ChunkRepositoryError("Readiness chunk lookup failed.") from exc

        if chunk_response is None:
            raise ChunkRepositoryError("Readiness chunk query returned no API response.")

        chunk_rows = chunk_response.data
        if not isinstance(chunk_rows, list) or not chunk_rows:
            raise ChunkRepositoryError("Active corpus has no active chunks.")
        chunk = chunk_rows[0]
        if not isinstance(chunk, dict):
            raise ChunkRepositoryError("Readiness chunk payload is invalid.")

        return ReadinessRecord(
            corpus_version=corpus_version,
            sample_chunk_id=_required_string(chunk, "id"),
        )


def _required_string(record: dict[str, object], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ChunkRepositoryError(f"Database field {field!r} is missing or invalid.")
    return value.strip()