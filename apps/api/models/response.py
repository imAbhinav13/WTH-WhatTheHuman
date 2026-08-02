"""Pydantic schemas for generated claims and final query responses."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from pydantic import Field, model_validator

from apps.api.models.common import (
    APIModel,
    ConceptCoverage,
    DomainCoverage,
    NonEmptyString,
    PipelineTimings,
    PositiveRank,
    SimilarityScore,
)
from apps.api.models.enums import (
    CoverageStatus,
    Domain,
    MatchStrength,
    RelationshipType,
)
from apps.api.models.query import (
    ClaimTypeBreakdown,
    SemanticMappingResult,
)


class CitationReference(APIModel):
    """Citation linking a generated claim to a retrieval event."""

    retrieval_result_id: UUID
    chunk_id: NonEmptyString

    concept_id: UUID
    concept_slug: NonEmptyString

    domain: Domain

    citation: NonEmptyString
    full_text: NonEmptyString

    query_concept_weight: SimilarityScore
    chunk_concept_weight: SimilarityScore
    similarity_score: SimilarityScore
    combined_score: SimilarityScore

    retrieval_rank: PositiveRank
    citation_order: int = Field(ge=0)


class GeneratedClaim(APIModel):
    """One grounded claim generated for a knowledge domain."""

    claim_id: UUID | None = None
    domain: Domain

    claim_order: int = Field(ge=0)
    claim_text: NonEmptyString

    citations: tuple[CitationReference, ...]

    @model_validator(mode="after")
    def validate_claim_citations(self) -> GeneratedClaim:
        """Ensure every generated claim has valid, unique citations."""

        if not self.citations:
            raise ValueError(
                "Every generated claim must have at least one citation"
            )

        retrieval_result_ids = [
            citation.retrieval_result_id
            for citation in self.citations
        ]

        citation_orders = [
            citation.citation_order
            for citation in self.citations
        ]

        if len(retrieval_result_ids) != len(set(retrieval_result_ids)):
            raise ValueError(
                "Generated claim contains duplicate retrieval citations"
            )

        if len(citation_orders) != len(set(citation_orders)):
            raise ValueError(
                "Generated claim contains duplicate citation orders"
            )

        expected_orders = list(range(len(citation_orders)))

        if sorted(citation_orders) != expected_orders:
            raise ValueError(
                "Citation orders must be contiguous and start at 0"
            )

        for citation in self.citations:
            if citation.domain is not self.domain:
                raise ValueError(
                    "Citation domain must match the generated claim domain"
                )

        return self


class DomainPanel(APIModel):
    """Grounded response panel for one knowledge domain."""

    domain: Domain

    summary: NonEmptyString | None = None
    claims: tuple[GeneratedClaim, ...] = ()

    match_strength: MatchStrength
    coverage: DomainCoverage

    @model_validator(mode="after")
    def validate_panel_consistency(self) -> DomainPanel:
        """Ensure panel content matches retrieval and coverage state."""

        if self.coverage.domain is not self.domain:
            raise ValueError(
                "Panel coverage domain must match panel domain"
            )

        if self.match_strength is MatchStrength.NONE:
            if self.claims:
                raise ValueError(
                    "A panel with no retrieval match cannot contain claims"
                )

            if self.summary is not None:
                raise ValueError(
                    "A panel with no retrieval match cannot contain a summary"
                )

            if self.coverage.supported:
                raise ValueError(
                    "A panel with no retrieval match cannot be supported"
                )

            return self

        if not self.claims:
            raise ValueError(
                "A panel with retrieval evidence must contain claims"
            )

        if self.summary is None:
            raise ValueError(
                "A panel with retrieval evidence must contain a summary"
            )

        if not self.coverage.supported:
            raise ValueError(
                "A panel with generated claims must be marked supported"
            )

        claim_orders = [
            claim.claim_order
            for claim in self.claims
        ]

        if len(claim_orders) != len(set(claim_orders)):
            raise ValueError(
                "Domain panel contains duplicate claim orders"
            )

        expected_orders = list(range(len(claim_orders)))

        if sorted(claim_orders) != expected_orders:
            raise ValueError(
                "Claim orders must be contiguous and start at 0"
            )

        for claim in self.claims:
            if claim.domain is not self.domain:
                raise ValueError(
                    "Generated claim domain must match panel domain"
                )

        return self


class TensionAnalysis(APIModel):
    """Comparison across the three generated knowledge-domain panels."""

    relationship_type: RelationshipType
    summary: NonEmptyString

    agreements: tuple[NonEmptyString, ...] = ()
    disagreements: tuple[NonEmptyString, ...] = ()
    incommensurabilities: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def validate_relationship_content(self) -> TensionAnalysis:
        """Ensure comparison details align with the relationship type."""

        if (
            self.relationship_type
            is RelationshipType.GENUINE_DISAGREEMENT
            and not self.disagreements
        ):
            raise ValueError(
                "Genuine disagreement must include disagreement details"
            )

        if (
            self.relationship_type
            is RelationshipType.NOT_COMPARABLE
            and not self.incommensurabilities
        ):
            raise ValueError(
                "Not-comparable responses must explain incommensurability"
            )

        if (
            self.relationship_type
            is RelationshipType.NO_TENSION
            and self.disagreements
        ):
            raise ValueError(
                "No-tension responses cannot contain disagreements"
            )

        return self


class QueryResponse(APIModel):
    """Complete response returned by the WTH query endpoint."""

    query_id: UUID
    response_id: UUID

    question: NonEmptyString | None = None

    semantic_mapping: SemanticMappingResult
    claim_type_breakdown: ClaimTypeBreakdown

    overall_coverage: CoverageStatus

    science: DomainPanel
    advaita: DomainPanel
    samkhya: DomainPanel

    concept_coverage: tuple[ConceptCoverage, ...]

    tension: TensionAnalysis | None = None

    generation_model: NonEmptyString
    embedding_model: NonEmptyString
    corpus_version: NonEmptyString

    timings: PipelineTimings

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    @model_validator(mode="after")
    def validate_response(self) -> QueryResponse:
        """Validate panel identity, concept coverage, and final synthesis."""

        if self.science.domain is not Domain.SCIENCE:
            raise ValueError(
                "science panel must use the science domain"
            )

        if self.advaita.domain is not Domain.ADVAITA:
            raise ValueError(
                "advaita panel must use the advaita domain"
            )

        if self.samkhya.domain is not Domain.SAMKHYA:
            raise ValueError(
                "samkhya panel must use the samkhya domain"
            )

        activated_concept_ids = {
            activation.concept_id
            for activation in self.semantic_mapping.activations
        }

        covered_concept_ids = {
            coverage.concept_id
            for coverage in self.concept_coverage
        }

        if activated_concept_ids != covered_concept_ids:
            raise ValueError(
                "Concept coverage must exist for every activated concept"
            )

        if self.overall_coverage is CoverageStatus.OUT_OF_CORPUS:
            if self.tension is not None:
                raise ValueError(
                    "Out-of-corpus responses must not include synthesis"
                )

            if any(
                panel.claims
                for panel in (
                    self.science,
                    self.advaita,
                    self.samkhya,
                )
            ):
                raise ValueError(
                    "Out-of-corpus responses must not contain claims"
                )

            return self

        supported_panels = sum(
            panel.coverage.supported
            for panel in (
                self.science,
                self.advaita,
                self.samkhya,
            )
        )

        if supported_panels == 0:
            raise ValueError(
                "Supported responses require evidence from at least one domain"
            )

        if self.tension is None:
            raise ValueError(
                "Supported responses must include tension analysis"
            )

        return self


class QueryAcceptedEvent(APIModel):
    """SSE payload emitted when query processing begins."""

    query_id: UUID
    event: str = "query_accepted"


class ConceptsMappedEvent(APIModel):
    """SSE payload emitted after semantic mapping."""

    query_id: UUID
    semantic_mapping: SemanticMappingResult
    event: str = "concepts_mapped"


class DomainCompletedEvent(APIModel):
    """SSE payload emitted when one domain panel is complete."""

    query_id: UUID
    panel: DomainPanel
    event: str = "domain_completed"


class QueryCompletedEvent(APIModel):
    """SSE payload emitted when the full query response is complete."""

    query_id: UUID
    response: QueryResponse
    event: str = "query_completed"