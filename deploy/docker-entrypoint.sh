#!/bin/sh
set -eu

PORT="${PORT:-10000}"

case "$PORT" in
    ""|*[!0-9]*)
        echo "PORT must be a positive integer." >&2
        exit 2
        ;;
esac

if [ "$PORT" -le 0 ] || [ "$PORT" -gt 65535 ]; then
    echo "PORT must be between 1 and 65535." >&2
    exit 2
fi

# One Uvicorn worker is deliberate for the initial deployment:
# - Stage 6.1B's lightweight IP limiter is process-local.
# - Horizontal/multi-worker scaling is deferred until a shared limiter exists.
# This is not an application concurrency semaphore.
exec python -m uvicorn \
    apps.api.main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers 1 \
    --no-access-log
