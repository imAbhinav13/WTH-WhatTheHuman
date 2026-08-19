"""Static Stage 6.2 Docker packaging contract tests.

These do not require Docker Desktop and perform no provider calls.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "Dockerfile"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"
ENTRYPOINT = REPO_ROOT / "deploy" / "docker-entrypoint.sh"


def _dockerfile() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def _dockerignore() -> str:
    return DOCKERIGNORE.read_text(encoding="utf-8")


def _entrypoint() -> str:
    return ENTRYPOINT.read_text(encoding="utf-8")


def test_required_docker_files_exist() -> None:
    assert DOCKERFILE.is_file()
    assert DOCKERIGNORE.is_file()
    assert ENTRYPOINT.is_file()


def test_runtime_is_python_311_and_multi_stage() -> None:
    text = _dockerfile()

    assert "python:3.11-slim" in text
    assert " AS builder" in text
    assert " AS runtime" in text


def test_uv_uses_locked_environment_without_dev_dependencies() -> None:
    text = _dockerfile()

    assert "uv sync" in text
    assert "--frozen" in text
    assert "--no-dev" in text
    assert "--no-install-project" in text


def test_dockerfile_never_bakes_runtime_secrets() -> None:
    text = _dockerfile()

    forbidden_assignments = (
        "SUPABASE_SECRET_KEY=",
        "GROQ_API_KEY=",
        "GOOGLE_API_KEY=",
        "SUPABASE_URL=https://",
    )

    for marker in forbidden_assignments:
        assert marker not in text


def test_image_runs_non_root() -> None:
    text = _dockerfile()

    assert "USER 10001:10001" in text
    assert "useradd" in text


def test_no_reload_and_single_initial_worker() -> None:
    text = _entrypoint()

    assert "--reload" not in text
    assert "--workers 1" in text


def test_entrypoint_binds_public_interface_and_runtime_port() -> None:
    text = _entrypoint()

    assert "--host 0.0.0.0" in text
    assert '--port "$PORT"' in text
    assert "PORT=" in text


def test_healthcheck_uses_liveness_endpoint() -> None:
    text = _dockerfile()

    assert "HEALTHCHECK" in text
    assert "/api/health" in text


def test_dockerignore_excludes_local_secrets_and_venv() -> None:
    text = _dockerignore()

    required_patterns = (
        ".env",
        ".venv",
        "__pycache__",
        "tests",
        "artifacts",
        ".git",
    )

    for pattern in required_patterns:
        assert pattern in text


def test_dockerignore_does_not_exclude_all_runtime_data() -> None:
    lines = {
        line.strip()
        for line in _dockerignore().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "data" not in lines
    assert "data/**" not in lines


def test_uvicorn_access_log_is_disabled_because_wth_has_structured_logs() -> None:
    assert "--no-access-log" in _entrypoint()
