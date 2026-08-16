from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.repositories.chunk_repository import (
    ChunkRecord,
    ChunkRepositoryError,
    ReadinessRecord,
)
from apps.api.routes.chunks import router as chunks_router
from apps.api.routes.readiness import router as readiness_router
from apps.api.services.chunk_service import (
    ChunkService,
    get_chunk_repository,
    get_chunk_service,
)

CHUNK_API_PREFIX = "/api/v1/chunk"
READINESS_API_PREFIX = "/api/v1"
VALID_CHUNK_ID = "science_herzog_kammer_scharnowski_2016_time_slices:chunk:ae7186593aa7242988ef1db1"


@dataclass
class FakeRepository:
    record: ChunkRecord | None = None
    fail: bool = False

    def get_active_chunk(self, chunk_id: str) -> ChunkRecord | None:
        if self.fail:
            raise ChunkRepositoryError("secret=https://should-not-leak.example")
        if self.record is None:
            return None
        if self.record.chunk_id != chunk_id:
            return None
        return self.record

    def check_ready(self) -> ReadinessRecord:
        if self.fail:
            raise ChunkRepositoryError("SUPABASE_SECRET_KEY=must-not-leak")
        return ReadinessRecord(
            corpus_version="phase1_active_corpus_v1",
            sample_chunk_id=VALID_CHUNK_ID,
        )


def build_client(repository: FakeRepository) -> TestClient:
    app = FastAPI()
    app.include_router(chunks_router, prefix="/api/v1")
    app.include_router(readiness_router, prefix="/api/v1")

    service = ChunkService(repository)  # type: ignore[arg-type]

    app.dependency_overrides[get_chunk_service] = lambda: service
    app.dependency_overrides[get_chunk_repository] = lambda: repository
    return TestClient(app)


def sample_record() -> ChunkRecord:
    return ChunkRecord(
        chunk_id=VALID_CHUNK_ID,
        source_id="science_herzog_kammer_scharnowski_2016_time_slices",
        domain="science",
        text="A reviewed frozen corpus passage.",
        citation=(
            "Michael H. Herzog; Thomas Kammer; Frank Scharnowski. "
            "Time Slices: What Is the Duration of a Percept?. 2016"
        ),
        corpus_version="phase1_active_corpus_v1",
    )


def test_valid_chunk_returns_public_safe_response() -> None:
    client = build_client(FakeRepository(record=sample_record()))

    response = client.get(f"{CHUNK_API_PREFIX}/{VALID_CHUNK_ID}")

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "chunk_id": VALID_CHUNK_ID,
        "source_id": "science_herzog_kammer_scharnowski_2016_time_slices",
        "domain": "science",
        "text": "A reviewed frozen corpus passage.",
        "citation": (
            "Michael H. Herzog; Thomas Kammer; Frank Scharnowski. "
            "Time Slices: What Is the Duration of a Percept?. 2016"
        ),
        "corpus_version": "phase1_active_corpus_v1",
    }

    serialized = response.text.lower()
    assert "embedding" not in serialized
    assert "supabase" not in serialized
    assert "secret" not in serialized
    assert "provider" not in serialized


def test_unknown_chunk_returns_404() -> None:
    client = build_client(FakeRepository())

    response = client.get(f"{CHUNK_API_PREFIX}/unknown:chunk:123")

    assert response.status_code == 404
    assert response.json() == {"detail": "Chunk not found."}


def test_malformed_chunk_id_returns_400() -> None:
    client = build_client(FakeRepository())

    response = client.get(f"{CHUNK_API_PREFIX}/not%20allowed")

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid chunk ID."}


def test_database_failure_returns_sanitized_503() -> None:
    client = build_client(FakeRepository(fail=True))

    response = client.get(f"{CHUNK_API_PREFIX}/{VALID_CHUNK_ID}")

    assert response.status_code == 503
    assert response.json() == {"detail": "Corpus database is temporarily unavailable."}
    assert "secret" not in response.text.lower()
    assert "supabase" not in response.text.lower()


def test_ready_confirms_database_and_active_corpus() -> None:
    client = build_client(FakeRepository())

    response = client.get(f"{READINESS_API_PREFIX}/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "database": "ready",
        "corpus_version": "phase1_active_corpus_v1",
    }


def test_ready_database_failure_is_sanitized() -> None:
    client = build_client(FakeRepository(fail=True))

    response = client.get(f"{READINESS_API_PREFIX}/ready")

    assert response.status_code == 503
    assert response.json() == {"detail": "Service dependencies are not ready."}
    assert "secret" not in response.text.lower()
    assert "supabase" not in response.text.lower()