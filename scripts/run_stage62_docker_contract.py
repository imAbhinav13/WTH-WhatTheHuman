"""Static Stage 6.2 Docker packaging review.

No Docker daemon and no provider access are required.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    entrypoint = (
        ROOT / "deploy" / "docker-entrypoint.sh"
    ).read_text(encoding="utf-8")

    checks = {
        "Dockerfile exists": (ROOT / "Dockerfile").is_file(),
        ".dockerignore exists": (ROOT / ".dockerignore").is_file(),
        "Entrypoint exists": (
            ROOT / "deploy" / "docker-entrypoint.sh"
        ).is_file(),
        "Python 3.11 runtime": "python:3.11-slim" in dockerfile,
        "uv frozen sync": "--frozen" in dockerfile,
        "No dev dependencies": "--no-dev" in dockerfile,
        "Non-root runtime": "USER 10001:10001" in dockerfile,
        "No --reload": "--reload" not in entrypoint,
        "Binds 0.0.0.0": "--host 0.0.0.0" in entrypoint,
        "Uses runtime PORT": '--port "$PORT"' in entrypoint,
        "Container health /api/health": "/api/health" in dockerfile,
        ".env excluded": ".env" in dockerignore,
        ".venv excluded": ".venv" in dockerignore,
        ".git excluded": ".git" in dockerignore,
        "tests excluded": "tests" in dockerignore,
        "artifacts excluded": "artifacts" in dockerignore,
        "No baked Groq secret": "GROQ_API_KEY=" not in dockerfile,
        "No baked Google secret": "GOOGLE_API_KEY=" not in dockerfile,
        "No baked Supabase secret": (
            "SUPABASE_SECRET_KEY=" not in dockerfile
        ),
    }

    print("Stage 6.2 Docker contract review")

    failed = False
    for label, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL':<5} {label}")
        failed = failed or not passed

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
