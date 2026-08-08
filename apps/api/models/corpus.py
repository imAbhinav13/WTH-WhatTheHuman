"""Pydantic models for  corpus ingestion and review."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self, TypeAlias
from uuid import UUID

from pydantic import (
    AnyHttpUrl,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from apps.api.models.common import (
    APIModel,
    NonEmptyString,
    NonNegativeInteger,
    PositiveRank,
    SimilarityScore,
)
from apps.api.models.enums import (
    ConceptSlug,
    Domain,
    MappingMethod,
    ReviewStatus,
    SourceType,
)

SourceIdentifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=3,
        max_length=160,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    ),
]

VersionIdentifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    ),
]

Sha256Checksum = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-fA-F]{64}$",
    ),
]

ConceptWeight = Annotated[
    float,
    Field(ge=0.0, le=1.0),
]

EmbeddingVector768 = Annotated[
    tuple[float, ...],
    Field(min_length=768, max_length=768),
]


PHASE1_CONCEPTS: frozenset[ConceptSlug] = frozenset(
    {
        ConceptSlug.CONSCIOUSNESS,
        ConceptSlug.SELF_IDENTITY,
        ConceptSlug.REALITY_APPEARANCE,
    }
)


class SourceFormat(StrEnum):
    """Supported Phase 1 source formats."""

    JATS_XML = "jats_xml"
    GUTENBERG_HTML = "gutenberg_html"
    STRUCTURED_TEXT = "structured_text"
    OCR_TEXT = "ocr_text"
    SCANNED_BOOK_WITH_OCR = "scanned_book_with_ocr"


class SourceInclusionStatus(StrEnum):
    """Catalogue-level inclusion state."""

    APPROVED_FOR_ACQUISITION = "approved_for_acquisition"
    CANDIDATE_PENDING_JURISDICTION_REVIEW = "candidate_pending_jurisdiction_review"
    CANDIDATE_PENDING_RIGHTS_AND_TEXT_QUALITY_REVIEW = (
        "candidate_pending_rights_and_text_quality_review"
    )
    PENDING_REVIEW = "pending_review"
    RESTRICTED = "restricted"
    REJECTED = "rejected"


class RightsStatus(StrEnum):
    """Rights determination applied during source review."""

    ELIGIBLE = "eligible"
    ELIGIBLE_WITH_CONDITIONS = "eligible_with_conditions"
    PENDING_REVIEW = "pending_review"
    RESTRICTED = "restricted"
    REJECTED = "rejected"


class AcquisitionMethod(StrEnum):
    """Method used to acquire a raw source artifact."""

    DIRECT_HTTP = "direct_http"
    MANUAL_DOWNLOAD = "manual_download"
    REPOSITORY_EXPORT = "repository_export"


class DocumentUnitType(StrEnum):
    """Semantic role of a parsed document unit."""

    ABSTRACT = "abstract"
    BODY = "body"
    ROOT_TEXT = "root_text"
    COMMENTARY = "commentary"
    TRANSLATOR_NOTE = "translator_note"
    EDITOR_NOTE = "editor_note"
    FOOTNOTE = "footnote"
    CAPTION = "caption"
    TABLE = "table"
    APPENDIX = "appendix"
    OTHER = "other"


class ParserWarningSeverity(StrEnum):
    """Severity assigned to parser warnings."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ReviewDecisionType(StrEnum):
    """Allowed human review decisions."""

    APPROVE = "approve"
    APPROVE_WITH_EDITS = "approve_with_edits"
    REJECT = "reject"
    NEEDS_SOURCE_REVIEW = "needs_source_review"
    NEEDS_LICENSE_REVIEW = "needs_license_review"


class RejectionReason(StrEnum):
    """Standardized chunk-rejection reasons."""

    RIGHTS_UNCLEAR = "rights_unclear"
    WRONG_DOMAIN = "wrong_domain"
    INSUFFICIENT_RELEVANCE = "insufficient_relevance"
    CITATION_INVALID = "citation_invalid"
    PARSER_ERROR = "parser_error"
    OCR_UNCERTAIN = "ocr_uncertain"
    DUPLICATE = "duplicate"
    CHUNK_TOO_FRAGMENTARY = "chunk_too_fragmentary"
    CHUNK_MIXES_SOURCE_TYPES = "chunk_mixes_source_types"
    TRANSLATION_UNKNOWN = "translation_unknown"
    SOURCE_AUTHORITY_INSUFFICIENT = "source_authority_insufficient"
    RETRACTED_SOURCE = "retracted_source"
    OTHER = "other"


class IngestionRunStatus(StrEnum):
    """Lifecycle state of one ingestion run."""

    STARTED = "started"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"


class SourceCatalogueEntry(APIModel):
    """One source declared in the Phase 1 YAML catalogue."""

    source_id: SourceIdentifier
    domain: Domain

    title: NonEmptyString
    author: NonEmptyString
    translator: NonEmptyString | None = None
    editor: NonEmptyString | None = None

    publication_year: int = Field(ge=1000, le=2100)
    source_type: SourceType

    canonical_url: AnyHttpUrl
    download_url: AnyHttpUrl

    format: SourceFormat

    license_name: NonEmptyString
    license_url: AnyHttpUrl
    rights_statement: NonEmptyString
    rights_jurisdiction: NonEmptyString

    accessed_at: date
    checksum: Sha256Checksum | None = None

    included_concepts: tuple[ConceptSlug, ...]

    authority_notes: NonEmptyString
    inclusion_status: SourceInclusionStatus

    enabled: bool = True

    @field_validator("checksum")
    @classmethod
    def normalize_checksum(
        cls,
        value: str | None,
    ) -> str | None:
        """Normalize SHA-256 checksums to lowercase."""

        if value is None:
            return None

        return value.lower()

    @model_validator(mode="after")
    def validate_phase1_concepts(self) -> Self:
        """Require unique concepts from the Phase 1 concept slice."""

        if not self.included_concepts:
            raise ValueError("A Phase 1 source must include at least one concept")

        if len(self.included_concepts) != len(set(self.included_concepts)):
            raise ValueError("included_concepts must not contain duplicates")

        unsupported = set(self.included_concepts) - PHASE1_CONCEPTS

        if unsupported:
            values = ", ".join(sorted(concept.value for concept in unsupported))
            raise ValueError(f"Source includes non-Phase 1 concepts: {values}")

        return self


SourceCatalogue: TypeAlias = tuple[SourceCatalogueEntry, ...]


class AcquiredSourceArtifact(APIModel):
    """Metadata for one downloaded and frozen raw source file."""

    source_id: SourceIdentifier
    source_url: AnyHttpUrl

    local_path: Path
    media_type: NonEmptyString
    file_size_bytes: int = Field(gt=0)

    checksum: Sha256Checksum
    acquisition_method: AcquisitionMethod
    acquired_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("checksum")
    @classmethod
    def normalize_checksum(cls, value: str) -> str:
        """Normalize the acquired-file checksum."""

        return value.lower()


class SourceRightsReview(APIModel):
    """Item-level rights determination for a source."""

    source_id: SourceIdentifier
    status: RightsStatus

    license_name: NonEmptyString
    license_url: AnyHttpUrl
    rights_statement: NonEmptyString
    rights_jurisdiction: NonEmptyString

    conditions: tuple[NonEmptyString, ...] = ()
    reviewed_by: NonEmptyString
    reviewed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_conditions(self) -> Self:
        """Require conditions for conditional eligibility."""

        if self.status is RightsStatus.ELIGIBLE_WITH_CONDITIONS and not self.conditions:
            raise ValueError("Conditional eligibility requires at least one condition")

        return self


class ParserWarning(APIModel):
    """Warning emitted while parsing a raw source."""

    code: NonEmptyString
    message: NonEmptyString
    severity: ParserWarningSeverity

    structural_locator: NonEmptyString | None = None


class DocumentUnit(APIModel):
    """Smallest normalized unit emitted by a source parser."""

    unit_id: NonEmptyString
    order: NonNegativeInteger

    unit_type: DocumentUnitType

    heading: NonEmptyString | None = None
    structural_locator: NonEmptyString
    text: NonEmptyString

    parent_section_id: NonEmptyString | None = None


class DocumentSection(APIModel):
    """Normalized document section containing ordered units."""

    section_id: NonEmptyString
    order: NonNegativeInteger
    level: int = Field(ge=0, le=12)

    title: NonEmptyString | None = None
    structural_locator: NonEmptyString

    units: tuple[DocumentUnit, ...]

    @model_validator(mode="after")
    def validate_units(self) -> Self:
        """Validate unit identity and ordering within the section."""

        if not self.units:
            raise ValueError("A document section must contain at least one unit")

        unit_ids = [unit.unit_id for unit in self.units]
        unit_orders = [unit.order for unit in self.units]

        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("Document section contains duplicate unit IDs")

        if len(unit_orders) != len(set(unit_orders)):
            raise ValueError("Document section contains duplicate unit orders")

        expected_orders = list(range(len(unit_orders)))

        if sorted(unit_orders) != expected_orders:
            raise ValueError("Document-unit orders must be contiguous and start at 0")

        return self


class ParsedDocument(APIModel):
    """Normalized output produced by a source-specific parser."""

    source_id: SourceIdentifier
    source_checksum: Sha256Checksum

    domain: Domain
    title: NonEmptyString

    parser_name: NonEmptyString
    parser_version: VersionIdentifier

    sections: tuple[DocumentSection, ...]
    warnings: tuple[ParserWarning, ...] = ()

    parsed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("source_checksum")
    @classmethod
    def normalize_checksum(cls, value: str) -> str:
        """Normalize the source checksum."""

        return value.lower()

    @model_validator(mode="after")
    def validate_sections(self) -> Self:
        """Require unique and consistently ordered sections."""

        if not self.sections:
            raise ValueError("A parsed document must contain at least one section")

        section_ids = [section.section_id for section in self.sections]
        section_orders = [section.order for section in self.sections]

        if len(section_ids) != len(set(section_ids)):
            raise ValueError("Parsed document contains duplicate section IDs")

        if len(section_orders) != len(set(section_orders)):
            raise ValueError("Parsed document contains duplicate section orders")

        expected_orders = list(range(len(section_orders)))

        if sorted(section_orders) != expected_orders:
            raise ValueError("Section orders must be contiguous and start at 0")

        return self


class ChunkCitation(APIModel):
    """Human-readable and machine-readable chunk citation."""

    display_text: NonEmptyString
    structural_locator: NonEmptyString

    canonical_url: AnyHttpUrl
    external_identifier: NonEmptyString | None = None


class ChunkDraft(APIModel):
    """Chunk produced by a domain-aware chunker before review."""

    chunk_id: NonEmptyString

    source_id: SourceIdentifier
    source_checksum: Sha256Checksum

    domain: Domain
    source_type: SourceType
    unit_type: DocumentUnitType

    section_id: NonEmptyString
    source_unit_ids: tuple[NonEmptyString, ...]

    text: NonEmptyString
    text_checksum: Sha256Checksum
    token_count: int = Field(gt=0)

    citation: ChunkCitation

    parser_name: NonEmptyString
    parser_version: VersionIdentifier
    chunker_name: NonEmptyString
    chunker_version: VersionIdentifier

    review_status: ReviewStatus = ReviewStatus.DRAFT

    @field_validator("source_checksum", "text_checksum")
    @classmethod
    def normalize_checksums(cls, value: str) -> str:
        """Normalize chunk checksums."""

        return value.lower()

    @model_validator(mode="after")
    def validate_source_units(self) -> Self:
        """Require at least one unique source unit."""

        if not self.source_unit_ids:
            raise ValueError("A chunk must reference at least one source unit")

        if len(self.source_unit_ids) != len(set(self.source_unit_ids)):
            raise ValueError("source_unit_ids must not contain duplicates")

        return self


class ChunkEmbeddingRecord(APIModel):
    """Frozen 768-dimensional embedding for one chunk."""

    chunk_id: NonEmptyString
    text_checksum: Sha256Checksum

    provider: NonEmptyString = "google"
    model: NonEmptyString = "gemini-embedding-001"
    dimensions: Literal[768] = 768
    task_type: NonEmptyString

    embedding: EmbeddingVector768
    is_l2_normalized: bool

    embedded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("text_checksum")
    @classmethod
    def normalize_checksum(cls, value: str) -> str:
        """Normalize the embedded-text checksum."""

        return value.lower()

    @field_validator("embedding")
    @classmethod
    def validate_vector_values(
        cls,
        value: tuple[float, ...],
    ) -> tuple[float, ...]:
        """Reject vectors containing non-finite values."""

        if not all(math.isfinite(component) for component in value):
            raise ValueError("Embedding values must all be finite")

        return value


class ConceptAnchorDefinition(APIModel):
    """Reviewed textual definition for one Phase 1 concept anchor."""

    concept_id: UUID
    concept_slug: ConceptSlug

    canonical_definition: NonEmptyString
    positive_indicators: tuple[NonEmptyString, ...]
    domain_terminology: dict[Domain, tuple[NonEmptyString, ...]]

    related_non_equivalent_terms: tuple[NonEmptyString, ...] = ()
    explicit_exclusions: tuple[NonEmptyString, ...]
    common_confusions: tuple[NonEmptyString, ...] = ()

    anchor_text: NonEmptyString
    anchor_version: VersionIdentifier

    reviewed_by: NonEmptyString
    reviewed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_concept(self) -> Self:
        """Restrict anchor definitions to the Phase 1 concepts."""

        if self.concept_slug not in PHASE1_CONCEPTS:
            raise ValueError("Concept anchor is outside the Phase 1 concept slice")

        if not self.positive_indicators:
            raise ValueError("Concept anchor requires positive indicators")

        if not self.explicit_exclusions:
            raise ValueError("Concept anchor requires explicit exclusions")

        return self


class ConceptAnchorEmbeddingRecord(APIModel):
    """Frozen 768-dimensional embedding for a concept anchor."""

    concept_id: UUID
    concept_slug: ConceptSlug
    anchor_version: VersionIdentifier

    provider: NonEmptyString = "google"
    model: NonEmptyString = "gemini-embedding-001"
    dimensions: Literal[768] = 768
    task_type: NonEmptyString

    embedding: EmbeddingVector768
    embedded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("embedding")
    @classmethod
    def validate_vector_values(
        cls,
        value: tuple[float, ...],
    ) -> tuple[float, ...]:
        """Reject concept vectors containing non-finite values."""

        if not all(math.isfinite(component) for component in value):
            raise ValueError("Anchor embedding values must all be finite")

        return value

    @model_validator(mode="after")
    def validate_concept(self) -> Self:
        """Restrict anchor embeddings to Phase 1 concepts."""

        if self.concept_slug not in PHASE1_CONCEPTS:
            raise ValueError("Concept anchor embedding is outside Phase 1")

        return self


class ChunkConceptProposal(APIModel):
    """Automated concept-weight proposal for one chunk."""

    chunk_id: NonEmptyString
    concept_id: UUID
    concept_slug: ConceptSlug

    mapping_method: MappingMethod
    anchor_version: VersionIdentifier

    anchor_similarity: SimilarityScore
    proposed_weight: ConceptWeight
    proposal_rank: PositiveRank

    @model_validator(mode="after")
    def validate_concept(self) -> Self:
        """Restrict proposals to Phase 1 concepts."""

        if self.concept_slug not in PHASE1_CONCEPTS:
            raise ValueError("Chunk concept proposal is outside Phase 1")

        return self


class ReviewedConceptWeight(APIModel):
    """Reviewer-approved concept relevance for one chunk."""

    concept_id: UUID
    concept_slug: ConceptSlug

    proposed_weight: ConceptWeight | None = None
    approved_weight: ConceptWeight

    reviewer_notes: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_concept(self) -> Self:
        """Restrict reviewed weights to Phase 1 concepts."""

        if self.concept_slug not in PHASE1_CONCEPTS:
            raise ValueError("Reviewed concept weight is outside Phase 1")

        return self


class ChunkReviewDecision(APIModel):
    """Human review outcome for one chunk."""

    chunk_id: NonEmptyString
    decision: ReviewDecisionType

    concept_weights: tuple[ReviewedConceptWeight, ...] = ()

    reviewer: NonEmptyString
    reviewed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    rejection_reason: RejectionReason | None = None
    notes: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        """Ensure evidence and rejection details match the decision."""

        approved_decisions = {
            ReviewDecisionType.APPROVE,
            ReviewDecisionType.APPROVE_WITH_EDITS,
        }

        if self.decision in approved_decisions:
            if not self.concept_weights:
                raise ValueError("Approved chunks require reviewed concept weights")

            if self.rejection_reason is not None:
                raise ValueError("Approved chunks cannot have a rejection reason")

        if self.decision is ReviewDecisionType.REJECT and self.rejection_reason is None:
            raise ValueError("Rejected chunks require a rejection reason")

        concept_ids = [weight.concept_id for weight in self.concept_weights]
        concept_slugs = [weight.concept_slug for weight in self.concept_weights]

        if len(concept_ids) != len(set(concept_ids)):
            raise ValueError("Review contains duplicate concept IDs")

        if len(concept_slugs) != len(set(concept_slugs)):
            raise ValueError("Review contains duplicate concept slugs")

        return self


class IngestionManifest(APIModel):
    """Audit summary for one Phase 1 ingestion run."""

    run_id: UUID
    corpus_version: VersionIdentifier

    status: IngestionRunStatus

    source_ids: tuple[SourceIdentifier, ...]
    parser_versions: dict[NonEmptyString, VersionIdentifier]
    chunker_versions: dict[NonEmptyString, VersionIdentifier]

    acquired_source_count: NonNegativeInteger
    parsed_document_count: NonNegativeInteger
    draft_chunk_count: NonNegativeInteger
    embedded_chunk_count: NonNegativeInteger
    reviewed_chunk_count: NonNegativeInteger
    active_chunk_count: NonNegativeInteger
    failed_item_count: NonNegativeInteger

    started_at: datetime
    completed_at: datetime | None = None

    errors: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        """Validate run completion and count relationships."""

        if not self.source_ids:
            raise ValueError("Ingestion manifest requires at least one source")

        if len(self.source_ids) != len(set(self.source_ids)):
            raise ValueError("Ingestion manifest contains duplicate source IDs")

        if self.status is IngestionRunStatus.STARTED:
            if self.completed_at is not None:
                raise ValueError("A started run cannot have completed_at")
        elif self.completed_at is None:
            raise ValueError("A finished ingestion run requires completed_at")

        if self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("completed_at cannot precede started_at")

        if self.active_chunk_count > self.reviewed_chunk_count:
            raise ValueError("Active chunk count cannot exceed reviewed count")

        if self.reviewed_chunk_count > self.embedded_chunk_count:
            raise ValueError("Reviewed chunk count cannot exceed embedded count")

        if self.embedded_chunk_count > self.draft_chunk_count:
            raise ValueError("Embedded chunk count cannot exceed draft count")

        if self.status is IngestionRunStatus.COMPLETED and self.failed_item_count != 0:
            raise ValueError("Completed runs cannot contain failed items")

        return self


class ActivationFailure(APIModel):
    """Reason one reviewed chunk could not be activated."""

    chunk_id: NonEmptyString
    reason: NonEmptyString


class ActivationManifest(APIModel):
    """Audit record for reviewed-chunk activation."""

    corpus_version: VersionIdentifier

    requested_chunk_ids: tuple[NonEmptyString, ...]
    activated_chunk_ids: tuple[NonEmptyString, ...]
    failures: tuple[ActivationFailure, ...] = ()

    activated_by: NonEmptyString
    activated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_activation(self) -> Self:
        """Validate activation results against requested chunks."""

        requested = set(self.requested_chunk_ids)
        activated = set(self.activated_chunk_ids)
        failed = {failure.chunk_id for failure in self.failures}

        if not requested:
            raise ValueError("Activation must request at least one chunk")

        if len(requested) != len(self.requested_chunk_ids):
            raise ValueError("requested_chunk_ids must not contain duplicates")

        if len(activated) != len(self.activated_chunk_ids):
            raise ValueError("activated_chunk_ids must not contain duplicates")

        if len(failed) != len(self.failures):
            raise ValueError("Activation failures contain duplicate chunk IDs")

        if not activated.issubset(requested):
            raise ValueError("Activated chunks must be part of the request")

        if not failed.issubset(requested):
            raise ValueError("Failed chunks must be part of the request")

        if activated & failed:
            raise ValueError("A chunk cannot be both activated and failed")

        if activated | failed != requested:
            raise ValueError("Every requested chunk needs an activation result")

        return self


class ActiveChunkBundle(APIModel):
    """Complete validated bundle required for chunk activation."""

    source: SourceCatalogueEntry
    rights_review: SourceRightsReview

    chunk: ChunkDraft
    embedding: ChunkEmbeddingRecord
    review: ChunkReviewDecision

    corpus_version: VersionIdentifier

    @model_validator(mode="after")
    def validate_exit_gate(self) -> Self:
        """Enforce the Phase 1 active-chunk exit gate."""

        if not self.source.enabled:
            raise ValueError("Disabled catalogue sources cannot produce active chunks")

        if self.source.checksum is None:
            raise ValueError("Active chunks require a catalogue checksum")

        if self.source.inclusion_status in {
            SourceInclusionStatus.RESTRICTED,
            SourceInclusionStatus.REJECTED,
        }:
            raise ValueError("Restricted or rejected sources cannot be activated")

        if self.rights_review.status not in {
            RightsStatus.ELIGIBLE,
            RightsStatus.ELIGIBLE_WITH_CONDITIONS,
        }:
            raise ValueError("Active chunks require an eligible rights review")

        if self.source.source_id != self.chunk.source_id:
            raise ValueError("Chunk source_id does not match catalogue source")

        if self.source.source_id != self.rights_review.source_id:
            raise ValueError("Rights-review source_id does not match catalogue source")

        if self.source.domain is not self.chunk.domain:
            raise ValueError("Chunk domain does not match catalogue source")

        if self.source.source_type is not self.chunk.source_type:
            raise ValueError("Chunk source type does not match catalogue source")

        if self.source.checksum != self.chunk.source_checksum:
            raise ValueError("Chunk source checksum does not match catalogue checksum")

        if self.chunk.chunk_id != self.embedding.chunk_id:
            raise ValueError("Embedding chunk_id does not match chunk")

        if self.chunk.chunk_id != self.review.chunk_id:
            raise ValueError("Review chunk_id does not match chunk")

        if self.chunk.text_checksum != self.embedding.text_checksum:
            raise ValueError("Embedding text checksum does not match chunk text")

        if self.review.decision not in {
            ReviewDecisionType.APPROVE,
            ReviewDecisionType.APPROVE_WITH_EDITS,
        }:
            raise ValueError("Active chunks require an approval decision")

        if not self.review.concept_weights:
            raise ValueError("Active chunks require reviewed concept weights")

        if self.chunk.review_status not in {
            ReviewStatus.REVIEWED,
            ReviewStatus.ACTIVE,
        }:
            raise ValueError("Active bundles require a reviewed chunk")

        return self
