from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app


pytestmark = pytest.mark.live

VALID_CHUNK_ID = (
    "science_herzog_kammer_scharnowski_2016_time_slices:"
    "chunk:ae7186593aa7242988ef1db1"
)


@pytest.mark.skipif(
    os.getenv("WTH_RUN_STAGE2_LIVE") != "1",
    reason="Set WTH_RUN_STAGE2_LIVE=1 to run the live Supabase API test.",
)
def test_stage2_live_chunk_and_readiness() -> None:
    with TestClient(app) as client:
        ready = client.get("/api/ready")
        assert ready.status_code == 200

        ready_body = ready.json()
        assert ready_body["status"] == "healthy"

        database_check = next(
            check
            for check in ready_body["checks"]
            if check["provider"] == "database"
        )
        assert database_check["status"] == "ready"

        response = client.get(f"/api/chunk/{VALID_CHUNK_ID}")
        assert response.status_code == 200

        body = response.json()
        assert body["chunk_id"] == VALID_CHUNK_ID
        assert body["domain"] == "science"
        assert body["corpus_version"] == "phase1_active_corpus_v1"
        assert body["citation"] == (
            "Michael H. Herzog; Thomas Kammer; Frank Scharnowski. "
            "Time Slices: What Is the Duration of a Percept?. 2016"
        )
        assert body["text"].strip()

        serialized = response.text.lower()
        assert "embedding" not in serialized
        assert "secret" not in serialized
        assert "supabase" not in serialized
