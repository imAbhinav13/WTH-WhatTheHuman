from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from apps.api.models.chunk import ReadinessResponse
from apps.api.repositories.chunk_repository import (
    ChunkRepository,
    ChunkRepositoryError,
)
from apps.api.services.chunk_service import get_chunk_repository

LOGGER = logging.getLogger("wth.api.readiness")

router = APIRouter(tags=["health"])


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Check production runtime readiness",
)
def ready(
    repository: Annotated[
        ChunkRepository,
        Depends(get_chunk_repository),
    ],
) -> ReadinessResponse:
    try:
        result = repository.check_ready()
    except ChunkRepositoryError:
        LOGGER.exception("Runtime readiness database check failed.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service dependencies are not ready.",
        ) from None

    return ReadinessResponse(
        status="ready",
        database="ready",
        corpus_version=result.corpus_version,
    )
