from __future__ import annotations

from functools import lru_cache

from apps.api.clients.supabase_runtime import get_supabase_runtime_client
from apps.api.models.chunk import ChunkResponse
from apps.api.repositories.chunk_repository import ChunkRepository


class ChunkNotFoundError(LookupError):
    """Raised when a chunk is absent from the active production corpus."""


class InvalidChunkIdError(ValueError):
    """Raised when a chunk ID is not safe/valid for lookup."""


_ALLOWED_CHUNK_ID_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-"
)


class ChunkService:
    def __init__(self, repository: ChunkRepository) -> None:
        self._repository = repository

    def get_chunk(self, chunk_id: str) -> ChunkResponse:
        normalized = validate_chunk_id(chunk_id)
        record = self._repository.get_active_chunk(normalized)

        if record is None:
            raise ChunkNotFoundError(normalized)

        if record.domain not in {"science", "advaita", "samkhya"}:
            raise RuntimeError("Database returned an unsupported domain.")

        return ChunkResponse(
            chunk_id=record.chunk_id,
            source_id=record.source_id,
            domain=record.domain,  # type: ignore[arg-type]
            text=record.text,
            citation=record.citation,
            corpus_version=record.corpus_version,
        )


def validate_chunk_id(chunk_id: str) -> str:
    value = chunk_id.strip()

    if not value or len(value) > 512:
        raise InvalidChunkIdError("Invalid chunk ID.")

    if value[0] not in _ALLOWED_CHUNK_ID_CHARS:
        raise InvalidChunkIdError("Invalid chunk ID.")

    if any(character not in _ALLOWED_CHUNK_ID_CHARS for character in value):
        raise InvalidChunkIdError("Invalid chunk ID.")

    return value


@lru_cache(maxsize=1)
def get_chunk_repository() -> ChunkRepository:
    return ChunkRepository(get_supabase_runtime_client())


@lru_cache(maxsize=1)
def get_chunk_service() -> ChunkService:
    return ChunkService(get_chunk_repository())
