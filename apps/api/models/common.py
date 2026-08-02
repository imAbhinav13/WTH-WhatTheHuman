"""Shared Pydantic models used across WTH API contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from apps.api.models.enums import (
    ConceptCoverageStatus,
    ConceptSlug,
    Domain,
    HealthStatus,
    ProviderKind,
    ProviderStatus,
)

NonEmptyString = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
    ),
]

NonNegativeInteger = Annotated[
    int,
    Field(ge=0),
]

SimilarityScore = Annotated[
    float,
    Field(ge=-1.0, le=1.0),
]

PositiveRank = Annotated[
    int,
    Field(ge=1),
]


class APIModel(BaseModel):
    """Base model for all WTH API request and response schemas."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        populate_by_name=True,
        from_attributes=True,
    )


class TimestampedModel(APIModel):
    """Base model for schemas containing creation timestamps."""

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )


class ConceptActivation(APIModel):
    """A concept activated by the Semantic Mapper for a query."""

    concept_id: UUID
    slug: ConceptSlug
    display_name: NonEmptyString

    activation_weight: SimilarityScore
    activation_rank: PositiveRank


class ConceptCoverage(APIModel):
    """Coverage result for one activated concept."""

    concept_id: UUID
    slug: ConceptSlug

    coverage_status: ConceptCoverageStatus

    supporting_domain_count: int = Field(
        ge=0,
        le=3,
    )
    supporting_chunk_count: NonNegativeInteger

    strongest_score: SimilarityScore | None = None

    @model_validator(mode="after")
    def validate_evidence_consistency(self) -> ConceptCoverage:
        """Ensure coverage status is consistent with evidence counts."""

        if self.coverage_status is ConceptCoverageStatus.UNSUPPORTED:
            if self.supporting_domain_count != 0:
                raise ValueError(
                    "Unsupported concepts must have zero supporting domains"
                )

            if self.supporting_chunk_count != 0:
                raise ValueError(
                    "Unsupported concepts must have zero supporting chunks"
                )

            if self.strongest_score is not None:
                raise ValueError(
                    "Unsupported concepts must not have a strongest score"
                )

            return self

        if self.supporting_domain_count == 0:
            raise ValueError(
                "Supported concepts must have at least one supporting domain"
            )

        if self.supporting_chunk_count == 0:
            raise ValueError(
                "Supported concepts must have at least one supporting chunk"
            )

        if self.strongest_score is None:
            raise ValueError(
                "Supported concepts must have a strongest score"
            )

        return self


class DomainCoverage(APIModel):
    """Coverage result for one knowledge domain."""

    domain: Domain
    supported: bool

    supporting_chunk_count: NonNegativeInteger
    strongest_score: SimilarityScore | None = None

    @model_validator(mode="after")
    def validate_support_consistency(self) -> DomainCoverage:
        """Ensure support status matches available evidence."""

        if self.supported:
            if self.supporting_chunk_count == 0:
                raise ValueError(
                    "Supported domains must have at least one chunk"
                )

            if self.strongest_score is None:
                raise ValueError(
                    "Supported domains must have a strongest score"
                )

            return self

        if self.supporting_chunk_count != 0:
            raise ValueError(
                "Unsupported domains must have zero supporting chunks"
            )

        if self.strongest_score is not None:
            raise ValueError(
                "Unsupported domains must not have a strongest score"
            )

        return self


class ProviderHealth(APIModel):
    """Readiness result for one application dependency."""

    provider: ProviderKind
    status: ProviderStatus

    latency_ms: NonNegativeInteger | None = None
    detail: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_skipped_provider(self) -> ProviderHealth:
        """Prevent skipped providers from reporting latency."""

        if (
            self.status is ProviderStatus.SKIPPED
            and self.latency_ms is not None
        ):
            raise ValueError(
                "Skipped providers must not report latency"
            )

        return self


class HealthResponse(APIModel):
    """Health or readiness response returned by the API."""

    status: HealthStatus
    service: NonEmptyString
    version: NonEmptyString
    environment: NonEmptyString

    checks: tuple[ProviderHealth, ...] = ()

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )


class PipelineTimings(APIModel):
    """Measured latency for each query-processing stage."""

    mapper_latency_ms: NonNegativeInteger | None = None
    retrieval_latency_ms: NonNegativeInteger | None = None
    generation_latency_ms: NonNegativeInteger | None = None
    synthesis_latency_ms: NonNegativeInteger | None = None
    total_latency_ms: NonNegativeInteger


class ErrorDetail(APIModel):
    """Structured API error information."""

    code: NonEmptyString
    message: NonEmptyString
    request_id: UUID | None = None


class ErrorResponse(APIModel):
    """Standard error response returned by API routes."""

    error: ErrorDetail
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )