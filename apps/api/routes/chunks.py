from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, status

from apps.api.models.chunk import ChunkResponse
from apps.api.repositories.chunk_repository import ChunkRepositoryError
from apps.api.services.chunk_service import (
    ChunkNotFoundError,
    ChunkService,
    InvalidChunkIdError,
    get_chunk_service,
)

LOGGER = logging.getLogger("wth.api.chunks")

router = APIRouter(prefix="/chunk", tags=["chunks"])

@router.get(
    "/{chunk_id}",
    response_model=ChunkResponse,
    response_model_exclude_none=True,
    summary="Get an active corpus chunk",
)
def get_chunk(
    chunk_id: Annotated[
        str,
        Path(
            min_length=1,
            max_length=512,
            description="Stable frozen WTH chunk ID.",
        ),
    ],
    service: Annotated[ChunkService, Depends(get_chunk_service)],
) -> ChunkResponse:
    try:
        return service.get_chunk(chunk_id)
    except InvalidChunkIdError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid chunk ID.",
        ) from exc
    except ChunkNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chunk not found.",
        ) from exc
    except ChunkRepositoryError:
        LOGGER.exception("Chunk database lookup failed.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Corpus database is temporarily unavailable.",
        ) from None
