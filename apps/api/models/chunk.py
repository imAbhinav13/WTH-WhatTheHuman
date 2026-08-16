from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Domain = Literal["science", "advaita", "samkhya"]


class ChunkResponse(BaseModel):
    """Public-safe citation expansion payload."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1, max_length=512)
    source_id: str = Field(min_length=1, max_length=512)
    domain: Domain
    text: str = Field(min_length=1)
    citation: str = Field(min_length=1)
    corpus_version: str = Field(min_length=1, max_length=128)


class ReadinessResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ready"]
    database: Literal["ready"]
    corpus_version: str
