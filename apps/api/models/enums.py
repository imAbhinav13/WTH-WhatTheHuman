"""Shared enumerations used across API schemas and provider contracts."""

from enum import StrEnum


class Domain(StrEnum):
    """Knowledge traditions available in WTH."""

    SCIENCE = "science"
    ADVAITA = "advaita"
    SAMKHYA = "samkhya"


class ConceptSlug(StrEnum):
    """Canonical concept identifiers seeded in the database."""

    SELF_IDENTITY = "self_identity"
    CONSCIOUSNESS = "consciousness"
    REALITY_APPEARANCE = "reality_appearance"
    MATTER_MIND = "matter_mind"
    COSMOLOGY_ORIGINS = "cosmology_origins"
    AGENCY_FREE_WILL = "agency_free_will"
    CAUSATION_KARMA = "causation_karma"
    MORAL_RESPONSIBILITY_SUFFERING = (
        "moral_responsibility_suffering"
    )


class ClaimType(StrEnum):
    """Type of claim represented by a source passage or question."""

    EMPIRICAL = "empirical"
    METAPHYSICAL = "metaphysical"
    NORMATIVE = "normative"
    MIXED = "mixed"


class SourceType(StrEnum):
    """Supported corpus source categories."""

    PAPER = "paper"
    PRIMARY_TEXT = "primary_text"
    COMMENTARY = "commentary"


class ReviewStatus(StrEnum):
    """Lifecycle state of an ingested corpus chunk."""

    DRAFT = "draft"
    REVIEWED = "reviewed"
    ACTIVE = "active"
    ARCHIVED = "archived"


class MappingMethod(StrEnum):
    """Method used by the Semantic Mapper."""

    ANCHOR_VECTOR = "anchor_vector"
    LLM_FALLBACK = "llm_fallback"


class CoverageStatus(StrEnum):
    """Overall or domain-level response coverage."""

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    OUT_OF_CORPUS = "out_of_corpus"


class ConceptCoverageStatus(StrEnum):
    """Coverage state for an individual activated concept."""

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"


class MatchStrength(StrEnum):
    """Human-readable retrieval strength exposed by API panels."""

    STRONG = "strong"
    WEAK = "weak"
    NONE = "none"


class RelationshipType(StrEnum):
    """Relationship between generated domain perspectives."""

    GENUINE_DISAGREEMENT = "genuine_disagreement"
    SURFACE_SIMILARITY_DEEP_DIFFERENCE = (
        "surface_similarity_deep_difference"
    )
    NOT_COMPARABLE = "not_comparable"
    NO_TENSION = "no_tension"


class ProviderStatus(StrEnum):
    """Operational state returned by readiness checks."""

    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    SKIPPED = "skipped"


class HealthStatus(StrEnum):
    """Top-level API health state."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ProviderKind(StrEnum):
    """External provider categories used by readiness reporting."""

    DATABASE = "database"
    EMBEDDING = "embedding"
    GENERATION = "generation"


class StreamEventType(StrEnum):
    """Server-sent event types used by the query pipeline."""

    QUERY_ACCEPTED = "query_accepted"
    CONCEPTS_MAPPED = "concepts_mapped"
    RETRIEVAL_COMPLETED = "retrieval_completed"
    DOMAIN_COMPLETED = "domain_completed"
    SYNTHESIS_COMPLETED = "synthesis_completed"
    QUERY_COMPLETED = "query_completed"
    ERROR = "error"