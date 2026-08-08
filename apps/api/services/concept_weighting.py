"""Concept-weight proposal service for Phase 1 corpus chunks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from apps.api.models.corpus import (
    PHASE1_CONCEPTS,
    ChunkConceptProposal,
    ChunkEmbeddingRecord,
    ConceptAnchorEmbeddingRecord,
)
from apps.api.models.enums import ConceptSlug, MappingMethod


DEFAULT_SIMILARITY_FLOOR: Final = 0.20
DEFAULT_SIMILARITY_CEILING: Final = 0.80
DEFAULT_WEIGHT_PRECISION: Final = 6
DEFAULT_NORMALIZATION_TOLERANCE: Final = 1e-5
PHASE1_EMBEDDING_DIMENSIONS: Final = 768


class ConceptWeightingError(RuntimeError):
    """Base exception for concept-weighting failures."""


class ConceptWeightingConfigurationError(ConceptWeightingError):
    """Raised when concept-weighting configuration is invalid."""


class ConceptWeightingInputError(ConceptWeightingError):
    """Raised when chunk or anchor inputs are invalid."""


class EmbeddingCompatibilityError(ConceptWeightingError):
    """Raised when chunk and anchor embeddings are incompatible."""


@dataclass(frozen=True, slots=True)
class ConceptWeightingConfig:
    """Configuration for converting anchor similarity into weights.

    Similarity is retained separately in every proposal. The proposed
    weight is an independently calibrated 0-1 value and is not
    normalized across concepts.

    The floor and ceiling are provisional Phase 1 values. They may be
    tuned on the development evaluation set but must be frozen before
    held-out baseline evaluation.
    """

    similarity_floor: float = DEFAULT_SIMILARITY_FLOOR
    similarity_ceiling: float = DEFAULT_SIMILARITY_CEILING

    weight_precision: int = DEFAULT_WEIGHT_PRECISION
    normalization_tolerance: float = DEFAULT_NORMALIZATION_TOLERANCE

    expected_dimensions: int = PHASE1_EMBEDDING_DIMENSIONS

    def __post_init__(self) -> None:
        """Validate weighting and embedding constraints."""

        if not -1.0 <= self.similarity_floor <= 1.0:
            raise ConceptWeightingConfigurationError("similarity_floor must be between -1 and 1")

        if not -1.0 <= self.similarity_ceiling <= 1.0:
            raise ConceptWeightingConfigurationError("similarity_ceiling must be between -1 and 1")

        if self.similarity_ceiling <= self.similarity_floor:
            raise ConceptWeightingConfigurationError(
                "similarity_ceiling must exceed similarity_floor"
            )

        if not 0 <= self.weight_precision <= 12:
            raise ConceptWeightingConfigurationError("weight_precision must be between 0 and 12")

        if self.normalization_tolerance <= 0:
            raise ConceptWeightingConfigurationError("normalization_tolerance must be positive")

        if self.expected_dimensions != PHASE1_EMBEDDING_DIMENSIONS:
            raise ConceptWeightingConfigurationError(
                "Phase 1 concept weighting requires 768-dimensional embeddings"
            )


@dataclass(frozen=True, slots=True)
class ChunkConceptWeightingResult:
    """Concept-weight proposals produced for a chunk collection."""

    proposals: tuple[ChunkConceptProposal, ...]

    chunk_count: int
    concept_count: int
    anchor_version: str

    def __post_init__(self) -> None:
        """Validate result counts and per-chunk proposal ranks."""

        if self.chunk_count < 1:
            raise ValueError("chunk_count must be at least 1")

        if self.concept_count < 1:
            raise ValueError("concept_count must be at least 1")

        expected_proposal_count = self.chunk_count * self.concept_count

        if len(self.proposals) != expected_proposal_count:
            raise ValueError("Proposal count must equal chunk_count multiplied by concept_count")

        proposals_by_chunk: dict[
            str,
            list[ChunkConceptProposal],
        ] = {}

        for proposal in self.proposals:
            proposals_by_chunk.setdefault(
                proposal.chunk_id,
                [],
            ).append(proposal)

        if len(proposals_by_chunk) != self.chunk_count:
            raise ValueError("Proposal chunk count does not match chunk_count")

        for chunk_proposals in proposals_by_chunk.values():
            if len(chunk_proposals) != self.concept_count:
                raise ValueError("Every chunk must have one proposal for every concept anchor")

            ranks = sorted(proposal.proposal_rank for proposal in chunk_proposals)

            expected_ranks = list(
                range(
                    1,
                    self.concept_count + 1,
                )
            )

            if ranks != expected_ranks:
                raise ValueError("Proposal ranks must be contiguous and start at 1 for each chunk")

    def for_chunk(
        self,
        chunk_id: str,
    ) -> tuple[ChunkConceptProposal, ...]:
        """Return rank-ordered proposals for one chunk."""

        return tuple(
            sorted(
                (proposal for proposal in self.proposals if proposal.chunk_id == chunk_id),
                key=lambda proposal: proposal.proposal_rank,
            )
        )


@dataclass(frozen=True, slots=True)
class _AnchorScore:
    """Internal similarity result for one concept anchor."""

    anchor: ConceptAnchorEmbeddingRecord
    similarity: float
    proposed_weight: float


class ConceptWeightingService:
    """Propose plural concept weights using anchor-vector alignment.

    Every chunk is compared against all three Phase 1 concept anchors.
    The service emits one ranked proposal per chunk and concept.

    Proposed weights remain independent. They are not softmax values
    and are not forced to sum to one, because a chunk may be strongly
    relevant to multiple concepts.
    """

    def __init__(
        self,
        *,
        config: ConceptWeightingConfig | None = None,
    ) -> None:
        """Initialize the concept-weighting service."""

        self._config = config if config is not None else ConceptWeightingConfig()

    @property
    def config(self) -> ConceptWeightingConfig:
        """Return the current weighting configuration."""

        return self._config

    def propose_weights(
        self,
        *,
        chunk_embeddings: tuple[
            ChunkEmbeddingRecord,
            ...,
        ],
        anchor_embeddings: tuple[
            ConceptAnchorEmbeddingRecord,
            ...,
        ],
    ) -> ChunkConceptWeightingResult:
        """Generate concept-weight proposals for all supplied chunks."""

        self._validate_chunk_embeddings(chunk_embeddings)
        anchor_version = self._validate_anchor_embeddings(anchor_embeddings)
        self._validate_embedding_compatibility(
            chunk_embeddings=chunk_embeddings,
            anchor_embeddings=anchor_embeddings,
        )

        proposals: list[ChunkConceptProposal] = []

        for chunk_embedding in chunk_embeddings:
            anchor_scores = tuple(
                self._score_anchor(
                    chunk_embedding=chunk_embedding,
                    anchor_embedding=anchor_embedding,
                )
                for anchor_embedding in anchor_embeddings
            )

            ranked_scores = sorted(
                anchor_scores,
                key=lambda score: (
                    -score.similarity,
                    score.anchor.concept_slug.value,
                ),
            )

            proposals.extend(
                ChunkConceptProposal(
                    chunk_id=chunk_embedding.chunk_id,
                    concept_id=score.anchor.concept_id,
                    concept_slug=score.anchor.concept_slug,
                    mapping_method=MappingMethod.ANCHOR_VECTOR,
                    anchor_version=score.anchor.anchor_version,
                    anchor_similarity=score.similarity,
                    proposed_weight=score.proposed_weight,
                    proposal_rank=rank,
                )
                for rank, score in enumerate(
                    ranked_scores,
                    start=1,
                )
            )

        return ChunkConceptWeightingResult(
            proposals=tuple(proposals),
            chunk_count=len(chunk_embeddings),
            concept_count=len(anchor_embeddings),
            anchor_version=anchor_version,
        )

    def _score_anchor(
        self,
        *,
        chunk_embedding: ChunkEmbeddingRecord,
        anchor_embedding: ConceptAnchorEmbeddingRecord,
    ) -> _AnchorScore:
        """Calculate cosine similarity and independent proposal weight."""

        similarity = _cosine_similarity(
            chunk_embedding.embedding,
            anchor_embedding.embedding,
        )

        proposed_weight = self.similarity_to_weight(similarity)

        return _AnchorScore(
            anchor=anchor_embedding,
            similarity=similarity,
            proposed_weight=proposed_weight,
        )

    def similarity_to_weight(
        self,
        similarity: float,
    ) -> float:
        """Convert cosine similarity into an independent 0-1 weight."""

        if not math.isfinite(similarity):
            raise ConceptWeightingInputError("Similarity must be finite")

        if similarity <= self._config.similarity_floor:
            return 0.0

        if similarity >= self._config.similarity_ceiling:
            return 1.0

        span = self._config.similarity_ceiling - self._config.similarity_floor

        weight = (similarity - self._config.similarity_floor) / span

        return round(
            min(
                1.0,
                max(
                    0.0,
                    weight,
                ),
            ),
            self._config.weight_precision,
        )

    def _validate_chunk_embeddings(
        self,
        records: tuple[
            ChunkEmbeddingRecord,
            ...,
        ],
    ) -> None:
        """Validate chunk embedding identity and normalization."""

        if not records:
            raise ConceptWeightingInputError("At least one chunk embedding is required")

        chunk_ids = [record.chunk_id for record in records]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ConceptWeightingInputError("Chunk embeddings contain duplicate chunk IDs")

        for record in records:
            if record.dimensions != self._config.expected_dimensions:
                raise EmbeddingCompatibilityError(
                    f"Chunk {record.chunk_id} does not use 768-dimensional embeddings"
                )

            if not record.is_l2_normalized:
                raise EmbeddingCompatibilityError(
                    f"Chunk {record.chunk_id} is not marked as L2 normalized"
                )

            _validate_unit_vector(
                record.embedding,
                expected_dimensions=(self._config.expected_dimensions),
                tolerance=(self._config.normalization_tolerance),
                description=(f"chunk {record.chunk_id}"),
            )

    def _validate_anchor_embeddings(
        self,
        records: tuple[
            ConceptAnchorEmbeddingRecord,
            ...,
        ],
    ) -> str:
        """Validate Phase 1 anchor completeness and normalization."""

        if not records:
            raise ConceptWeightingInputError("Concept anchor embeddings are required")

        concept_ids = [record.concept_id for record in records]
        concept_slugs = [record.concept_slug for record in records]

        if len(concept_ids) != len(set(concept_ids)):
            raise ConceptWeightingInputError("Anchor embeddings contain duplicate concept IDs")

        if len(concept_slugs) != len(set(concept_slugs)):
            raise ConceptWeightingInputError("Anchor embeddings contain duplicate concept slugs")

        actual_concepts = set(concept_slugs)

        if actual_concepts != PHASE1_CONCEPTS:
            missing = PHASE1_CONCEPTS - actual_concepts
            unexpected = actual_concepts - PHASE1_CONCEPTS

            details: list[str] = []

            if missing:
                details.append("missing=" + ",".join(sorted(concept.value for concept in missing)))

            if unexpected:
                details.append(
                    "unexpected=" + ",".join(sorted(concept.value for concept in unexpected))
                )

            raise ConceptWeightingInputError(
                "Anchor embeddings must contain exactly "
                "the three Phase 1 concepts: " + "; ".join(details)
            )

        anchor_versions = {record.anchor_version for record in records}

        if len(anchor_versions) != 1:
            raise EmbeddingCompatibilityError(
                "All concept anchors must use the same anchor version"
            )

        for record in records:
            if record.dimensions != self._config.expected_dimensions:
                raise EmbeddingCompatibilityError(
                    "Concept anchor "
                    f"{record.concept_slug.value} does not "
                    "use 768-dimensional embeddings"
                )

            _validate_unit_vector(
                record.embedding,
                expected_dimensions=(self._config.expected_dimensions),
                tolerance=(self._config.normalization_tolerance),
                description=(f"concept anchor {record.concept_slug.value}"),
            )

        return next(iter(anchor_versions))

    @staticmethod
    def _validate_embedding_compatibility(
        *,
        chunk_embeddings: tuple[
            ChunkEmbeddingRecord,
            ...,
        ],
        anchor_embeddings: tuple[
            ConceptAnchorEmbeddingRecord,
            ...,
        ],
    ) -> None:
        """Require identical embedding configuration for both sets."""

        first_chunk = chunk_embeddings[0]
        first_anchor = anchor_embeddings[0]

        expected_configuration = (
            first_anchor.provider,
            first_anchor.model,
            first_anchor.dimensions,
            first_anchor.task_type,
        )

        for anchor in anchor_embeddings:
            configuration = (
                anchor.provider,
                anchor.model,
                anchor.dimensions,
                anchor.task_type,
            )

            if configuration != expected_configuration:
                raise EmbeddingCompatibilityError(
                    "Concept anchors do not share one embedding configuration"
                )

        for chunk in chunk_embeddings:
            configuration = (
                chunk.provider,
                chunk.model,
                chunk.dimensions,
                chunk.task_type,
            )

            if configuration != expected_configuration:
                raise EmbeddingCompatibilityError(
                    "Chunk and concept-anchor embeddings "
                    "must use the same provider, model, "
                    "dimensions, and task type. "
                    f"Incompatible chunk: {chunk.chunk_id}"
                )

        if first_chunk.model != first_anchor.model:
            raise EmbeddingCompatibilityError("Chunk and anchor models do not match")


def _cosine_similarity(
    first: tuple[float, ...],
    second: tuple[float, ...],
) -> float:
    """Calculate cosine similarity for validated unit vectors."""

    if len(first) != len(second):
        raise EmbeddingCompatibilityError("Cannot compare vectors with different dimensions")

    similarity = math.fsum(
        left * right
        for left, right in zip(
            first,
            second,
            strict=True,
        )
    )

    return min(
        1.0,
        max(
            -1.0,
            float(similarity),
        ),
    )


def _validate_unit_vector(
    vector: tuple[float, ...],
    *,
    expected_dimensions: int,
    tolerance: float,
    description: str,
) -> None:
    """Ensure a vector is finite and approximately unit length."""

    if len(vector) != expected_dimensions:
        raise EmbeddingCompatibilityError(
            f"{description} has {len(vector)} dimensions; expected {expected_dimensions}"
        )

    if not all(math.isfinite(component) for component in vector):
        raise EmbeddingCompatibilityError(f"{description} contains non-finite values")

    magnitude_squared = math.fsum(component * component for component in vector)

    if not math.isclose(
        magnitude_squared,
        1.0,
        rel_tol=tolerance,
        abs_tol=tolerance,
    ):
        raise EmbeddingCompatibilityError(
            f"{description} is not an L2-normalized vector; squared magnitude={magnitude_squared}"
        )
