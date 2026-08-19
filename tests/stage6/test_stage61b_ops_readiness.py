"""Stage 6.1B readiness semantics tests."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from apps.api.repositories.chunk_repository import ReadinessRecord
from apps.api.routers.ops import router
from apps.api.services.chunk_service import get_chunk_repository


@dataclass
class FakeRepository:
    calls: int = 0

    def check_ready(self) -> ReadinessRecord:
        self.calls += 1
        return ReadinessRecord(
            corpus_version="phase1_active_corpus_v1",
            sample_chunk_id="sample:chunk:1",
        )


class FakeOrchestrator:
    async def execute(self, question: str) -> object:
        return question


def test_ready_does_not_call_generation_provider() -> None:
    repository = FakeRepository()

    settings = SimpleNamespace(
        app_name="WTH: What The Human",
        app_version="0.1.0",
        app_env="development",
        provider_mode=SimpleNamespace(value="live"),
        google_api_key=SecretStr("google-test"),
        groq_api_key=SecretStr("groq-test"),
        embedding_model="gemini-embedding-2",
        embedding_dimension=768,
    )

    # Match ProviderMode comparison used by the real route without creating
    # provider network clients.
    from apps.api.core.config import ProviderMode

    settings.provider_mode = ProviderMode.LIVE

    app = FastAPI()
    app.state.query_orchestrator = FakeOrchestrator()
    app.state.service_name = settings.app_name
    app.state.service_version = settings.app_version
    app.state.environment_name = settings.app_env
    app.include_router(router, prefix="/api")

    app.dependency_overrides[get_chunk_repository] = lambda: repository

    from apps.api.core.config import get_settings

    app.dependency_overrides[get_settings] = lambda: settings

    with TestClient(app) as client:
        response = client.get("/api/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert repository.calls == 1

    generation = next(
        check
        for check in body["checks"]
        if check["provider"] == "generation"
    )

    assert generation["status"] == "configured"
    assert "no generation probe executed" in generation["detail"].lower()


def test_health_is_liveness_only() -> None:
    app = FastAPI()
    app.state.service_name = "WTH"
    app.state.service_version = "0.1.0"
    app.state.environment_name = "production"
    app.include_router(router, prefix="/api")

    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["checks"] == {}
