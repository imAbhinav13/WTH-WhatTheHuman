# syntax=docker/dockerfile:1.7

# WTH Stage 6.2 production image.
#
# Build:
#   docker build --pull -t wth-api:stage6.2 .
#
# Run:
#   docker run --rm -p 10000:10000 --env-file .env wth-api:stage6.2
#
# Secrets are supplied only at runtime. They are never ARG/ENV values here.

ARG PYTHON_IMAGE=python:3.11-slim-bookworm
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.32

FROM ${UV_IMAGE} AS uv_source

FROM ${PYTHON_IMAGE} AS builder

COPY --from=uv_source /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_NO_DEV=1

WORKDIR /app

# Install locked third-party dependencies in a cache-friendly layer before
# copying frequently changing application source.
COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --frozen \
        --no-dev \
        --no-install-project \
        --no-editable

# .dockerignore is the security boundary for local secrets, tests, caches,
# artifacts, raw corpus downloads, frontend build output, etc.
COPY . /app

# Install the WTH project itself into the already-created environment.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --frozen \
        --no-dev \
        --no-editable


FROM ${PYTHON_IMAGE} AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:${PATH}" \
    PORT=10000

WORKDIR /app

# Use a fixed unprivileged identity. No provider/database credential needs
# filesystem write access.
RUN groupadd --system --gid 10001 wth \
    && useradd \
        --system \
        --uid 10001 \
        --gid wth \
        --home-dir /app \
        --shell /usr/sbin/nologin \
        wth

# The builder context has already been pruned by .dockerignore.
# Copy the prepared virtual environment and runtime source together.
COPY --from=builder --chown=wth:wth /app /app

COPY --chown=root:root deploy/docker-entrypoint.sh /usr/local/bin/wth-entrypoint

RUN chmod 0555 /usr/local/bin/wth-entrypoint

USER 10001:10001

EXPOSE 10000

STOPSIGNAL SIGTERM

# Container-level liveness only. Render's own healthCheckPath will be set to
# /api/health in Stage 6.3.
HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=20s \
    --retries=3 \
    CMD python -c 'import os, urllib.request; urllib.request.urlopen("http://127.0.0.1:%s/api/health" % os.environ.get("PORT", "10000"), timeout=3).read()' || exit 1

ENTRYPOINT ["/usr/local/bin/wth-entrypoint"]
