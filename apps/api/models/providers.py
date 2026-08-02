"""Pydantic schemas used by provider interfaces and implementations."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from pydantic import Field, model_validator

from apps.api.models.common import (
    APIModel,
    NonEmptyString,
    NonNegativeInteger,
    SimilarityScore,
)
from apps.api.models.enums import (
    ConceptSlug,
    Domain,
    ProviderKind,
    ProviderStatus,
)

EmbeddingVector = Annotated[
    tuple[float, ...],
    Field(min_length=1),
]

GenerationTemperature = Annotated[
    float,
    Field(ge=0.0, le=2.0),
]

TokenLimit = Annotated[
    int,
    Field(ge=1, le=131_072),
]


class EmbeddingRequest(APIModel):
    """Request for one or more text embeddings."""

    texts: tuple[NonEmptyString, ...]
    task_type: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_texts(self) -> EmbeddingRequest:
        """Require at least one unique text input."""

        if not self.texts:
            raise ValueError(
                "Embedding request must contain at least one text"
            )

        if len(self.texts) != len(set(self.texts)):
            raise ValueError(
                "Embedding request must not contain duplicate texts"
            )

        return self


class EmbeddingResult(APIModel):
    """Embedding generated for one input text."""

    index: NonNegativeInteger
    embedding: EmbeddingVector

    model: NonEmptyString
    dimensions: int = Field(ge=1, le=4_096)

    @model_validator(mode="after")
    def validate_dimensions(self) -> EmbeddingResult:
        """Ensure the declared dimension matches the vector length."""

        if len(self.embedding) != self.dimensions:
            raise ValueError(
                "Embedding length must match declared dimensions"
            )

        return self


class EmbeddingResponse(APIModel):
    """Batch embedding-provider response."""

    results: tuple[EmbeddingResult, ...]
    model: NonEmptyString
    dimensions: int = Field(ge=1, le=4_096)

    @model_validator(mode="after")
    def validate_results(self) -> EmbeddingResponse:
        """Validate result indexes, model names, and dimensions."""

        if not self.results:
            raise ValueError(
                "Embedding response must contain at least one result"
            )

        indexes = [result.index for result in self.results]

        if len(indexes) != len(set(indexes)):
            raise ValueError(
                "Embedding response contains duplicate result indexes"
            )

        expected_indexes = list(range(len(indexes)))

        if sorted(indexes) != expected_indexes:
            raise ValueError(
                "Embedding result indexes must be contiguous and start at 0"
            )

        for result in self.results:
            if result.model != self.model:
                raise ValueError(
                    "Embedding result model must match response model"
                )

            if result.dimensions != self.dimensions:
                raise ValueError(
                    "Embedding result dimensions must match response dimensions"
                )

        return self


class GenerationMessage(APIModel):
    """Single message passed to a generation provider."""

    role: NonEmptyString
    content: NonEmptyString


class GenerationRequest(APIModel):
    """Structured request sent to a text-generation provider."""

    messages: tuple[GenerationMessage, ...]

    model: NonEmptyString
    temperature: GenerationTemperature = 0.0
    max_output_tokens: TokenLimit = 4_096

    response_schema_name: NonEmptyString | None = None
    response_schema: dict[str, object] | None = None

    @model_validator(mode="after")
    def validate_messages_and_schema(self) -> GenerationRequest:
        """Validate message ordering and structured-output settings."""

        if not self.messages:
            raise ValueError(
                "Generation request must contain at least one message"
            )

        allowed_roles = {
            "system",
            "developer",
            "user",
            "assistant",
        }

        for message in self.messages:
            if message.role not in allowed_roles:
                raise ValueError(
                    f"Unsupported generation message role: {message.role}"
                )

        if (
            self.response_schema_name is None
            and self.response_schema is not None
        ):
            raise ValueError(
                "response_schema_name is required when response_schema is set"
            )

        if (
            self.response_schema_name is not None
            and self.response_schema is None
        ):
            raise ValueError(
                "response_schema is required when response_schema_name is set"
            )

        return self


class GenerationUsage(APIModel):
    """Token usage reported by a generation provider."""

    input_tokens: NonNegativeInteger
    output_tokens: NonNegativeInteger
    total_tokens: NonNegativeInteger

    @model_validator(mode="after")
    def validate_total_tokens(self) -> GenerationUsage:
        """Ensure total token count is internally consistent."""

        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError(
                "total_tokens must equal input_tokens plus output_tokens"
            )

        return self


class GenerationResponse(APIModel):
    """Normalized generation-provider response."""

    content: NonEmptyString
    model: NonEmptyString
    usage: GenerationUsage | None = None
    provider_request_id: NonEmptyString | None = None


class ProviderProbeResult(APIModel):
    """Normalized health-check result from a provider client."""

    provider: ProviderKind
    status: ProviderStatus
    latency_ms: NonNegativeInteger | None = None
    detail: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_status(self) -> ProviderProbeResult:
        """Validate probe status and latency consistency."""

        if (
            self.status is ProviderStatus.SKIPPED
            and self.latency_ms is not None
        ):
            raise ValueError(
                "Skipped provider probes must not report latency"
            )

        if (
            self.status is ProviderStatus.READY
            and self.latency_ms is None
        ):
            raise ValueError(
                "Ready provider probes must report latency"
            )

        return self


class DatabaseProbeRequest(APIModel):
    """Database readiness probe configuration."""

    verify_schema: bool = True
    verify_concepts: bool = True
    expected_concept_count: int = Field(default=8, ge=1)


class DatabaseProbeResponse(APIModel):
    """Database readiness probe details."""

    status: ProviderStatus
    latency_ms: NonNegativeInteger | None = None

    schema_available: bool
    concept_count: NonNegativeInteger | None = None
    active_corpus_version: NonEmptyString | None = None

    detail: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_database_probe(self) -> DatabaseProbeResponse:
        """Ensure successful probes contain required database evidence."""

        if self.status is ProviderStatus.READY:
            if self.latency_ms is None:
                raise ValueError(
                    "Ready database probes must report latency"
                )

            if not self.schema_available:
                raise ValueError(
                    "Ready database probes require the schema to be available"
                )

            if self.concept_count is None:
                raise ValueError(
                    "Ready database probes must report concept count"
                )

        return self


class ConceptAnchorEmbeddingRequest(APIModel):
    """Request to generate an embedding for a concept anchor."""

    concept_id: UUID
    concept_slug: ConceptSlug
    anchor_text: NonEmptyString


class ConceptAnchorEmbeddingResult(APIModel):
    """Generated embedding for one canonical concept anchor."""

    concept_id: UUID
    concept_slug: ConceptSlug
    embedding: EmbeddingVector

    model: NonEmptyString
    dimensions: int = Field(ge=1, le=4_096)

    @model_validator(mode="after")
    def validate_anchor_dimensions(
        self,
    ) -> ConceptAnchorEmbeddingResult:
        """Ensure anchor vector length matches declared dimensions."""

        if len(self.embedding) != self.dimensions:
            raise ValueError(
                "Concept anchor embedding length must match dimensions"
            )

        return self


class ProviderMetadata(APIModel):
    """Provider metadata exposed for diagnostics."""

    provider: ProviderKind
    implementation: NonEmptyString
    model: NonEmptyString | None = None
    is_mock: bool


class RetrievalScoreInput(APIModel):
    """Inputs used to calculate a concept-aware retrieval score."""

    query_id: UUID
    concept_id: UUID
    concept_slug: ConceptSlug
    domain: Domain

    query_concept_weight: SimilarityScore
    chunk_concept_weight: SimilarityScore
    similarity_score: SimilarityScore