"""Pydantic schemas for incoming questions and query-time provenance."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from pydantic import StringConstraints, model_validator

from apps.api.models.common import (
    APIModel,
    ConceptActivation,
    NonEmptyString,
    PositiveRank,
    SimilarityScore,
)
from apps.api.models.enums import (
    ClaimType,
    ConceptSlug,
    Domain,
    MappingMethod,
)

QuestionText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=3,
        max_length=1_000,
    ),
]


class QueryRequest(APIModel):
    """Natural-language question submitted to WTH."""

    question: QuestionText


class ClaimTypeBreakdown(APIModel):
    """Claim-type framing applied to each knowledge domain."""

    science: ClaimType
    advaita: ClaimType
    samkhya: ClaimType

    def for_domain(self, domain: Domain) -> ClaimType:
        """Return the configured claim type for a domain."""

        match domain:
            case Domain.SCIENCE:
                return self.science
            case Domain.ADVAITA:
                return self.advaita
            case Domain.SAMKHYA:
                return self.samkhya


class SemanticMappingResult(APIModel):
    """Weighted multi-concept output from the Semantic Mapper."""

    mapping_method: MappingMethod
    activations: tuple[ConceptActivation, ...]

    @model_validator(mode="after")
    def validate_activations(self) -> SemanticMappingResult:
        """Validate uniqueness, ranking, and descending activation order."""

        if not self.activations:
            raise ValueError(
                "Semantic mapping must contain at least one concept activation"
            )

        concept_ids = [
            activation.concept_id
            for activation in self.activations
        ]
        concept_slugs = [
            activation.slug
            for activation in self.activations
        ]
        ranks = [
            activation.activation_rank
            for activation in self.activations
        ]

        if len(concept_ids) != len(set(concept_ids)):
            raise ValueError(
                "Semantic mapping contains duplicate concept IDs"
            )

        if len(concept_slugs) != len(set(concept_slugs)):
            raise ValueError(
                "Semantic mapping contains duplicate concept slugs"
            )

        if len(ranks) != len(set(ranks)):
            raise ValueError(
                "Semantic mapping contains duplicate activation ranks"
            )

        expected_ranks = list(range(1, len(ranks) + 1))

        if sorted(ranks) != expected_ranks:
            raise ValueError(
                "Activation ranks must be contiguous and start at 1"
            )

        ordered = sorted(
            self.activations,
            key=lambda activation: activation.activation_rank,
        )

        weights = [
            activation.activation_weight
            for activation in ordered
        ]

        if weights != sorted(weights, reverse=True):
            raise ValueError(
                "Activation weights must descend by activation rank"
            )

        return self


class QueryRecord(APIModel):
    """Persisted query metadata returned after query creation."""

    query_id: UUID
    question_hash: NonEmptyString
    mapping_method: MappingMethod
    concept_activations: tuple[ConceptActivation, ...]
    claim_type_breakdown: ClaimTypeBreakdown


class RetrievalCandidate(APIModel):
    """Candidate chunk returned by concept-aware retrieval."""

    retrieval_result_id: UUID | None = None

    query_id: UUID
    concept_id: UUID
    concept_slug: ConceptSlug
    domain: Domain

    chunk_id: NonEmptyString
    citation: NonEmptyString
    full_text: NonEmptyString

    query_concept_weight: SimilarityScore
    chunk_concept_weight: SimilarityScore
    similarity_score: SimilarityScore
    combined_score: SimilarityScore

    retrieval_rank: PositiveRank


class RetrievalBundle(APIModel):
    """Retrieved evidence for one activated concept and domain."""

    query_id: UUID
    concept_id: UUID
    concept_slug: ConceptSlug
    domain: Domain

    candidates: tuple[RetrievalCandidate, ...] = ()

    @model_validator(mode="after")
    def validate_candidates(self) -> RetrievalBundle:
        """Ensure candidates belong to the bundle and have valid ranks."""

        if not self.candidates:
            return self

        chunk_ids: list[str] = []
        ranks: list[int] = []

        for candidate in self.candidates:
            if candidate.query_id != self.query_id:
                raise ValueError(
                    "Retrieval candidate query_id does not match bundle"
                )

            if candidate.concept_id != self.concept_id:
                raise ValueError(
                    "Retrieval candidate concept_id does not match bundle"
                )

            if candidate.concept_slug is not self.concept_slug:
                raise ValueError(
                    "Retrieval candidate concept_slug does not match bundle"
                )

            if candidate.domain is not self.domain:
                raise ValueError(
                    "Retrieval candidate domain does not match bundle"
                )

            chunk_ids.append(candidate.chunk_id)
            ranks.append(candidate.retrieval_rank)

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(
                "Retrieval bundle contains duplicate chunk IDs"
            )

        if len(ranks) != len(set(ranks)):
            raise ValueError(
                "Retrieval bundle contains duplicate ranks"
            )

        expected_ranks = list(range(1, len(ranks) + 1))

        if sorted(ranks) != expected_ranks:
            raise ValueError(
                "Retrieval ranks must be contiguous and start at 1"
            )

        ordered = sorted(
            self.candidates,
            key=lambda candidate: candidate.retrieval_rank,
        )

        scores = [
            candidate.combined_score
            for candidate in ordered
        ]

        if scores != sorted(scores, reverse=True):
            raise ValueError(
                "Combined scores must descend by retrieval rank"
            )

        return self


class QueryPipelineInput(APIModel):
    """Internal validated input passed into the query pipeline."""

    question: QuestionText
    question_hash: NonEmptyString
    semantic_mapping: SemanticMappingResult
    claim_type_breakdown: ClaimTypeBreakdown


class QueryConceptRecord(APIModel):
    """Database-ready representation of one activated query concept."""

    query_id: UUID
    concept_id: UUID
    activation_weight: SimilarityScore
    activation_rank: PositiveRank


class RetrievalResultRecord(APIModel):
    """Database-ready representation of one retrieval provenance row."""

    query_id: UUID
    concept_id: UUID
    domain: Domain
    chunk_id: NonEmptyString

    query_concept_weight: SimilarityScore
    chunk_concept_weight: SimilarityScore
    similarity_score: SimilarityScore
    combined_score: SimilarityScore

    retrieval_rank: PositiveRank