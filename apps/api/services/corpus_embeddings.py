"""Embedding services for Phase 1 chunks and concept anchors."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Final
from uuid import UUID

from apps.api.clients.base import EmbeddingProvider
from apps.api.models.corpus import (
    ChunkDraft,
    ChunkEmbeddingRecord,
    ConceptAnchorDefinition,
    ConceptAnchorEmbeddingRecord,
)
from apps.api.models.providers import (
    EmbeddingRequest,
    EmbeddingResponse,
)

DEFAULT_EMBEDDING_MODEL: Final = "gemini-embedding-001"
DEFAULT_EMBEDDING_DIMENSIONS: Final = 768
DEFAULT_EMBEDDING_TASK_TYPE: Final = "RETRIEVAL_DOCUMENT"

_NORMALIZATION_TOLERANCE: Final = 1e-6
_ZERO_MAGNITUDE_TOLERANCE: Final = 1e-12


class CorpusEmbeddingError(RuntimeError):
    """Base exception for corpus-embedding failures."""


class EmbeddingConfigurationError(CorpusEmbeddingError):
    """Raised when the embedding service is configured incorrectly."""


class EmbeddingInputError(CorpusEmbeddingError):
    """Raised when chunks or concept anchors are inconsistent."""


class EmbeddingProviderError(CorpusEmbeddingError):
    """Raised when the embedding provider cannot complete a request."""


class EmbeddingResponseError(CorpusEmbeddingError):
    """Raised when the provider returns invalid embedding data."""


@dataclass(frozen=True, slots=True)
class CorpusEmbeddingConfig:
    """Frozen embedding configuration for one corpus version."""

    model: str = DEFAULT_EMBEDDING_MODEL
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS
    task_type: str = DEFAULT_EMBEDDING_TASK_TYPE

    batch_size: int = 32
    maximum_attempts: int = 3

    initial_retry_delay_seconds: float = 1.0
    maximum_retry_delay_seconds: float = 8.0

    def __post_init__(self) -> None:
        """Validate the embedding configuration."""

        if not self.model.strip():
            raise EmbeddingConfigurationError("Embedding model must not be empty")

        if self.dimensions != DEFAULT_EMBEDDING_DIMENSIONS:
            raise EmbeddingConfigurationError("Phase 1 embedding dimensions must equal 768")

        if not self.task_type.strip():
            raise EmbeddingConfigurationError("Embedding task type must not be empty")

        if self.batch_size < 1:
            raise EmbeddingConfigurationError("Embedding batch size must be at least 1")

        if self.maximum_attempts < 1:
            raise EmbeddingConfigurationError("maximum_attempts must be at least 1")

        if self.initial_retry_delay_seconds < 0:
            raise EmbeddingConfigurationError("Initial retry delay cannot be negative")

        if self.maximum_retry_delay_seconds < 0:
            raise EmbeddingConfigurationError("Maximum retry delay cannot be negative")

        if self.maximum_retry_delay_seconds < self.initial_retry_delay_seconds:
            raise EmbeddingConfigurationError(
                "Maximum retry delay cannot be below the initial retry delay"
            )


@dataclass(frozen=True, slots=True)
class ChunkEmbeddingBatchResult:
    """Result of embedding or reusing a collection of chunks."""

    records: tuple[ChunkEmbeddingRecord, ...]
    generated_count: int
    reused_count: int

    def __post_init__(self) -> None:
        """Validate batch-result counts."""

        if self.generated_count < 0:
            raise ValueError("generated_count cannot be negative")

        if self.reused_count < 0:
            raise ValueError("reused_count cannot be negative")

        if self.generated_count + self.reused_count != len(self.records):
            raise ValueError("Generated and reused counts must equal the number of records")


@dataclass(frozen=True, slots=True)
class AnchorEmbeddingBatchResult:
    """Result of embedding or reusing concept anchors."""

    records: tuple[ConceptAnchorEmbeddingRecord, ...]
    generated_count: int
    reused_count: int

    def __post_init__(self) -> None:
        """Validate batch-result counts."""

        if self.generated_count < 0:
            raise ValueError("generated_count cannot be negative")

        if self.reused_count < 0:
            raise ValueError("reused_count cannot be negative")

        if self.generated_count + self.reused_count != len(self.records):
            raise ValueError("Generated and reused counts must equal the number of records")


class CorpusEmbeddingService:
    """Generate normalized Phase 1 embeddings.

    The service embeds both corpus chunks and reviewed concept anchors
    through the same provider, model, dimension, and task type.

    Existing records are reused only when all relevant provenance fields
    match the frozen configuration.
    """

    def __init__(
        self,
        *,
        provider: EmbeddingProvider,
        config: CorpusEmbeddingConfig | None = None,
    ) -> None:
        """Initialize the corpus embedding service."""

        self._provider = provider
        self._config = config if config is not None else CorpusEmbeddingConfig()

        metadata = provider.metadata

        if metadata.model is None:
            raise EmbeddingConfigurationError("Embedding provider metadata must include a model")

        if metadata.model != self._config.model:
            raise EmbeddingConfigurationError(
                "Provider model does not match corpus configuration: "
                f"{metadata.model!r} != {self._config.model!r}"
            )

        self._provider_name = metadata.implementation

    @property
    def config(self) -> CorpusEmbeddingConfig:
        """Return the frozen corpus-embedding configuration."""

        return self._config

    async def embed_chunks(
        self,
        *,
        chunks: Sequence[ChunkDraft],
        existing_records: Iterable[ChunkEmbeddingRecord] = (),
    ) -> ChunkEmbeddingBatchResult:
        """Embed chunks and reuse records whose provenance still matches."""

        self._validate_chunks(chunks)

        existing_by_chunk_id = _index_chunk_records(existing_records)

        resolved: dict[str, ChunkEmbeddingRecord] = {}
        pending: list[ChunkDraft] = []
        reused_count = 0

        for chunk in chunks:
            existing = existing_by_chunk_id.get(chunk.chunk_id)

            if existing is not None and self._chunk_record_is_reusable(
                chunk=chunk,
                record=existing,
            ):
                resolved[chunk.chunk_id] = existing
                reused_count += 1
                continue

            pending.append(chunk)

        vectors_by_text = await self._embed_unique_texts(tuple(chunk.text for chunk in pending))

        for chunk in pending:
            vector = vectors_by_text.get(chunk.text)

            if vector is None:
                raise EmbeddingResponseError(
                    f"No embedding was returned for chunk {chunk.chunk_id}"
                )

            resolved[chunk.chunk_id] = ChunkEmbeddingRecord(
                chunk_id=chunk.chunk_id,
                text_checksum=chunk.text_checksum,
                provider=self._provider_name,
                model=self._config.model,
                dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
                task_type=self._config.task_type,
                embedding=vector,
                is_l2_normalized=True,
            )

        ordered_records = tuple(resolved[chunk.chunk_id] for chunk in chunks)

        return ChunkEmbeddingBatchResult(
            records=ordered_records,
            generated_count=len(pending),
            reused_count=reused_count,
        )

    async def embed_concept_anchors(
        self,
        *,
        anchors: Sequence[ConceptAnchorDefinition],
        existing_records: Iterable[ConceptAnchorEmbeddingRecord] = (),
    ) -> AnchorEmbeddingBatchResult:
        """Embed reviewed concept anchors using the corpus configuration."""

        self._validate_anchors(anchors)

        existing_by_concept_id = _index_anchor_records(existing_records)

        resolved: dict[
            UUID,
            ConceptAnchorEmbeddingRecord,
        ] = {}
        pending: list[ConceptAnchorDefinition] = []
        reused_count = 0

        for anchor in anchors:
            existing = existing_by_concept_id.get(anchor.concept_id)

            if existing is not None and self._anchor_record_is_reusable(
                anchor=anchor,
                record=existing,
            ):
                resolved[anchor.concept_id] = existing
                reused_count += 1
                continue

            pending.append(anchor)

        vectors_by_text = await self._embed_unique_texts(
            tuple(anchor.anchor_text for anchor in pending)
        )

        for anchor in pending:
            vector = vectors_by_text.get(anchor.anchor_text)

            if vector is None:
                raise EmbeddingResponseError(
                    f"No embedding was returned for concept anchor {anchor.concept_slug.value}"
                )

            resolved[anchor.concept_id] = ConceptAnchorEmbeddingRecord(
                concept_id=anchor.concept_id,
                concept_slug=anchor.concept_slug,
                anchor_version=anchor.anchor_version,
                provider=self._provider_name,
                model=self._config.model,
                dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
                task_type=self._config.task_type,
                embedding=vector,
            )

        ordered_records = tuple(resolved[anchor.concept_id] for anchor in anchors)

        return AnchorEmbeddingBatchResult(
            records=ordered_records,
            generated_count=len(pending),
            reused_count=reused_count,
        )

    async def _embed_unique_texts(
        self,
        texts: Sequence[str],
    ) -> dict[str, tuple[float, ...]]:
        """Embed unique texts while preserving first-occurrence order."""

        if not texts:
            return {}

        unique_texts = tuple(dict.fromkeys(texts))

        vectors_by_text: dict[
            str,
            tuple[float, ...],
        ] = {}

        for batch in _batched(
            unique_texts,
            batch_size=self._config.batch_size,
        ):
            response = await self._request_batch_with_retry(batch)

            batch_vectors = self._normalize_response(
                texts=batch,
                response=response,
            )

            vectors_by_text.update(batch_vectors)

        return vectors_by_text

    async def _request_batch_with_retry(
        self,
        texts: tuple[str, ...],
    ) -> EmbeddingResponse:
        """Request one provider batch with bounded retry behavior."""

        last_error: Exception | None = None

        for attempt in range(
            1,
            self._config.maximum_attempts + 1,
        ):
            try:
                return await self._provider.embed(
                    EmbeddingRequest(
                        texts=texts,
                        task_type=self._config.task_type,
                    )
                )
            except Exception as exc:
                last_error = exc

                if attempt >= self._config.maximum_attempts:
                    break

                delay = min(
                    self._config.initial_retry_delay_seconds * (2 ** (attempt - 1)),
                    self._config.maximum_retry_delay_seconds,
                )

                if delay > 0:
                    await asyncio.sleep(delay)

        if last_error is None:
            raise EmbeddingProviderError("Embedding request failed without an exception")

        raise EmbeddingProviderError(
            "Embedding provider failed after "
            f"{self._config.maximum_attempts} attempt(s): "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

    def _normalize_response(
        self,
        *,
        texts: tuple[str, ...],
        response: EmbeddingResponse,
    ) -> dict[str, tuple[float, ...]]:
        """Validate provider output and manually L2-normalize vectors."""

        if response.model != self._config.model:
            raise EmbeddingResponseError(
                "Embedding response model does not match "
                "the frozen corpus model: "
                f"{response.model!r} != {self._config.model!r}"
            )

        if response.dimensions != self._config.dimensions:
            raise EmbeddingResponseError(
                "Embedding response dimension does not match "
                "the frozen corpus dimension: "
                f"{response.dimensions} != "
                f"{self._config.dimensions}"
            )

        if len(response.results) != len(texts):
            raise EmbeddingResponseError(
                "Embedding response count does not match the request count"
            )

        vectors_by_text: dict[
            str,
            tuple[float, ...],
        ] = {}

        for result in response.results:
            if result.index >= len(texts):
                raise EmbeddingResponseError("Embedding result index is outside the request range")

            if result.model != self._config.model:
                raise EmbeddingResponseError(
                    "Embedding result model does not match the frozen corpus model"
                )

            if result.dimensions != self._config.dimensions:
                raise EmbeddingResponseError(
                    "Embedding result dimension does not match the frozen corpus dimension"
                )

            text = texts[result.index]

            vectors_by_text[text] = _l2_normalize(
                result.embedding,
                expected_dimensions=self._config.dimensions,
            )

        if len(vectors_by_text) != len(texts):
            raise EmbeddingResponseError(
                "Embedding response did not resolve every unique input text"
            )

        return vectors_by_text

    def _chunk_record_is_reusable(
        self,
        *,
        chunk: ChunkDraft,
        record: ChunkEmbeddingRecord,
    ) -> bool:
        """Return whether an existing chunk embedding remains valid."""

        return (
            record.chunk_id == chunk.chunk_id
            and record.text_checksum == chunk.text_checksum
            and record.provider == self._provider_name
            and record.model == self._config.model
            and record.dimensions == self._config.dimensions
            and record.task_type == self._config.task_type
            and record.is_l2_normalized
            and _is_valid_normalized_vector(
                record.embedding,
                expected_dimensions=self._config.dimensions,
            )
        )

    def _anchor_record_is_reusable(
        self,
        *,
        anchor: ConceptAnchorDefinition,
        record: ConceptAnchorEmbeddingRecord,
    ) -> bool:
        """Return whether an existing anchor embedding remains valid."""

        return (
            record.concept_id == anchor.concept_id
            and record.concept_slug is anchor.concept_slug
            and record.anchor_version == anchor.anchor_version
            and record.provider == self._provider_name
            and record.model == self._config.model
            and record.dimensions == self._config.dimensions
            and record.task_type == self._config.task_type
            and _is_valid_normalized_vector(
                record.embedding,
                expected_dimensions=self._config.dimensions,
            )
        )

    @staticmethod
    def _validate_chunks(
        chunks: Sequence[ChunkDraft],
    ) -> None:
        """Validate chunk identity before embedding."""

        chunk_ids = [chunk.chunk_id for chunk in chunks]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise EmbeddingInputError("Chunk embedding input contains duplicate chunk IDs")

        for chunk in chunks:
            if not chunk.text.strip():
                raise EmbeddingInputError(f"Chunk {chunk.chunk_id} has empty text")

    @staticmethod
    def _validate_anchors(
        anchors: Sequence[ConceptAnchorDefinition],
    ) -> None:
        """Validate concept-anchor identity before embedding."""

        concept_ids = [anchor.concept_id for anchor in anchors]
        concept_slugs = [anchor.concept_slug for anchor in anchors]

        if len(concept_ids) != len(set(concept_ids)):
            raise EmbeddingInputError("Anchor embedding input contains duplicate concept IDs")

        if len(concept_slugs) != len(set(concept_slugs)):
            raise EmbeddingInputError("Anchor embedding input contains duplicate concept slugs")

        for anchor in anchors:
            if not anchor.anchor_text.strip():
                raise EmbeddingInputError(
                    f"Concept anchor has empty anchor text: {anchor.concept_slug.value}"
                )


def _index_chunk_records(
    records: Iterable[ChunkEmbeddingRecord],
) -> dict[str, ChunkEmbeddingRecord]:
    """Index existing chunk records and reject duplicates."""

    indexed: dict[str, ChunkEmbeddingRecord] = {}

    for record in records:
        if record.chunk_id in indexed:
            raise EmbeddingInputError(
                f"Existing embeddings contain duplicate chunk ID {record.chunk_id}"
            )

        indexed[record.chunk_id] = record

    return indexed


def _index_anchor_records(
    records: Iterable[ConceptAnchorEmbeddingRecord],
) -> dict[UUID, ConceptAnchorEmbeddingRecord]:
    """Index existing concept-anchor records and reject duplicates."""

    indexed: dict[
        UUID,
        ConceptAnchorEmbeddingRecord,
    ] = {}

    for record in records:
        if record.concept_id in indexed:
            raise EmbeddingInputError(
                f"Existing anchor embeddings contain duplicate concept ID {record.concept_id}"
            )

        indexed[record.concept_id] = record

    return indexed


def _batched(
    values: Sequence[str],
    *,
    batch_size: int,
) -> Iterator[tuple[str, ...]]:
    """Yield stable batches from a sequence."""

    for start_index in range(
        0,
        len(values),
        batch_size,
    ):
        yield tuple(values[start_index : start_index + batch_size])


def _l2_normalize(
    vector: Sequence[float],
    *,
    expected_dimensions: int,
) -> tuple[float, ...]:
    """Return a finite unit-length vector."""

    if len(vector) != expected_dimensions:
        raise EmbeddingResponseError(
            "Embedding vector dimension mismatch: "
            f"expected {expected_dimensions}, "
            f"received {len(vector)}"
        )

    if not all(math.isfinite(component) for component in vector):
        raise EmbeddingResponseError("Embedding vector contains a non-finite value")

    magnitude_squared = math.fsum(component * component for component in vector)
    magnitude = math.sqrt(magnitude_squared)

    if magnitude <= _ZERO_MAGNITUDE_TOLERANCE:
        raise EmbeddingResponseError("Embedding vector has zero or near-zero magnitude")

    normalized = tuple(float(component / magnitude) for component in vector)

    if not _is_valid_normalized_vector(
        normalized,
        expected_dimensions=expected_dimensions,
    ):
        raise EmbeddingResponseError("Embedding vector could not be normalized")

    return normalized


def _is_valid_normalized_vector(
    vector: Sequence[float],
    *,
    expected_dimensions: int,
) -> bool:
    """Return whether a vector is finite and approximately unit length."""

    if len(vector) != expected_dimensions:
        return False

    if not all(math.isfinite(component) for component in vector):
        return False

    magnitude_squared = math.fsum(component * component for component in vector)

    return math.isclose(
        magnitude_squared,
        1.0,
        rel_tol=_NORMALIZATION_TOLERANCE,
        abs_tol=_NORMALIZATION_TOLERANCE,
    )
