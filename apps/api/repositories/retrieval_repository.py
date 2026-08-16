"""Supabase repository for frozen Phase 1 runtime retrieval candidates.

This repository is intentionally a data-access boundary only.

It owns:
- backend-only Supabase RPC invocation;
- strict validation of the pgvector candidate rows returned by
  ``match_phase1_active_chunks``;
- reconstruction of the frozen Phase 1 concept-relation records;
- preservation of the complete active domain candidate set.

It does NOT own:
- concept activation;
- concept eligibility filtering;
- Phase 14 weighted ranking;
- candidate-pool truncation;
- citation-quality scoring;
- source diversity;
- deduplication;
- token/context budgets.

Those behaviors remain in the Phase 14 runtime retrieval service so the
validated Python ranking logic continues to be the source of truth.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias, cast

from supabase import Client


class RetrievalRepositoryError(RuntimeError):
    """Raised when production retrieval candidates cannot be read safely."""


Phase1Domain: TypeAlias = Literal[
    "science",
    "advaita",
    "samkhya",
]

DOMAINS: Final[tuple[Phase1Domain, ...]] = (
    "science",
    "advaita",
    "samkhya",
)

CONCEPTS: Final[tuple[str, ...]] = (
    "consciousness",
    "self_identity",
    "reality_appearance",
)

RPC_NAME: Final = "match_phase1_active_chunks"

FROZEN_CORPUS_VERSION: Final = "phase1_active_corpus_v1"
FROZEN_ACTIVE_CHUNK_COUNT: Final = 318
FROZEN_EMBEDDING_DIMENSION: Final = 768

# The Stage 3.5A RPC intentionally receives 318 rather than the historical
# candidate_pool_per_domain=30. Historical Phase 14 computes its weighted
# base score BEFORE truncating to the top 30. Pre-limiting by vector score in
# SQL would therefore change the validated algorithm.
RPC_CANDIDATE_LIMIT: Final = FROZEN_ACTIVE_CHUNK_COUNT

# Frozen Phase 1 citation invariants from the final 318 active bundles.
#
# All 318 final active chunks had:
#   citation text present
#   citation_verified == "yes"
#   structural_locator == ""
#
# The production relational schema stores the canonical citation text but not
# these two invariant fields. They are reconstructed here so the later runtime
# Phase 14 service can call the unchanged historical citation_quality_score()
# logic and reproduce the frozen score of 0.85 for every active chunk.
FROZEN_CITATION_VERIFIED: Final = True
FROZEN_STRUCTURAL_LOCATOR: Final = ""


@dataclass(frozen=True, slots=True)
class RetrievalConceptRelationRecord:
    """Frozen reviewed concept metadata for one chunk/concept relationship."""

    concept_id: str
    human_label: str
    production_active: bool
    calibrated_weight: float
    human_override: bool


@dataclass(frozen=True, slots=True)
class RetrievalCandidateRecord:
    """Database-backed Phase 14 candidate before Python ranking."""

    chunk_id: str
    source_id: str
    domain: Phase1Domain
    citation: str
    reviewed_text: str
    corpus_version: str
    vector_similarity: float
    concept_relations: dict[str, RetrievalConceptRelationRecord]
    source_title: str
    translator: str
    citation_verified: bool
    structural_locator: str


class RetrievalRepository:
    """Read-only access to pgvector candidates in the frozen active corpus."""

    def __init__(self, client: Client) -> None:
        self._client = client

    def get_domain_candidates(
        self,
        *,
        query_embedding: Sequence[float],
        domain: Phase1Domain,
        corpus_version: str = FROZEN_CORPUS_VERSION,
    ) -> list[RetrievalCandidateRecord]:
        """Return the complete active candidate set for one Phase 1 domain.

        The RPC orders rows by pgvector cosine similarity for transport only.
        This method deliberately does not truncate to the historical Phase 14
        candidate pool of 30. The runtime retrieval service must first compute
        the full historical weighted base score and only then take that pool.
        """

        embedding = _validated_embedding(query_embedding)

        validated_domain = _validated_domain(domain)

        validated_corpus_version = _validated_corpus_version(corpus_version)

        params: dict[str, object] = {
            "p_query_embedding": embedding,
            "p_domain": validated_domain,
            "p_corpus_version": (validated_corpus_version),
            "p_candidate_limit": (RPC_CANDIDATE_LIMIT),
        }

        try:
            response = self._client.rpc(
                RPC_NAME,
                params,
            ).execute()
        except Exception as exc:
            raise RetrievalRepositoryError("Phase 1 retrieval candidate lookup failed.") from exc

        if response is None:
            raise RetrievalRepositoryError(
                "Phase 1 retrieval candidate query returned no API response."
            )

        rows = response.data

        if not isinstance(rows, list):
            raise RetrievalRepositoryError(
                "Phase 1 retrieval candidate query returned an invalid payload."
            )

        candidates: list[RetrievalCandidateRecord] = []

        seen_chunk_ids: set[str] = set()

        for index, raw in enumerate(
            rows,
            start=1,
        ):
            row = _required_mapping(
                raw,
                f"retrieval candidate row {index}",
            )

            candidate = _parse_candidate(
                row,
                expected_domain=validated_domain,
                expected_corpus_version=(validated_corpus_version),
            )

            if candidate.chunk_id in seen_chunk_ids:
                raise RetrievalRepositoryError(
                    "Phase 1 retrieval candidate query returned "
                    f"duplicate chunk {candidate.chunk_id!r}."
                )

            seen_chunk_ids.add(candidate.chunk_id)

            candidates.append(candidate)

        return candidates

    def get_all_domain_candidates(
        self,
        *,
        query_embedding: Sequence[float],
        corpus_version: str = FROZEN_CORPUS_VERSION,
    ) -> dict[
        Phase1Domain,
        list[RetrievalCandidateRecord],
    ]:
        """Return complete active candidates independently for all domains."""

        # Validate once before any network request. Each domain call validates
        # again at its public boundary, intentionally keeping that method safe
        # when called directly.
        _validated_embedding(query_embedding)

        validated_corpus_version = _validated_corpus_version(corpus_version)

        return {
            domain: self.get_domain_candidates(
                query_embedding=query_embedding,
                domain=domain,
                corpus_version=(validated_corpus_version),
            )
            for domain in DOMAINS
        }


def _parse_candidate(
    row: Mapping[str, object],
    *,
    expected_domain: Phase1Domain,
    expected_corpus_version: str,
) -> RetrievalCandidateRecord:
    chunk_id = _required_string(
        row,
        "chunk_id",
    )

    source_id = _required_string(
        row,
        "source_id",
    )

    domain_text = _required_string(
        row,
        "domain",
    ).casefold()

    domain = _validated_domain(domain_text)

    if domain != expected_domain:
        raise RetrievalRepositoryError(
            f"Chunk {chunk_id!r} crossed the requested "
            f"domain boundary: expected {expected_domain!r}, "
            f"received {domain!r}."
        )

    corpus_version = _required_string(
        row,
        "corpus_version",
    )

    if corpus_version != expected_corpus_version:
        raise RetrievalRepositoryError(
            f"Chunk {chunk_id!r} crossed the requested corpus-version boundary."
        )

    relations = _parse_concept_relations(
        row.get("concept_relations"),
        chunk_id=chunk_id,
    )

    return RetrievalCandidateRecord(
        chunk_id=chunk_id,
        source_id=source_id,
        domain=domain,
        citation=_required_string(
            row,
            "citation",
        ),
        reviewed_text=_required_string(
            row,
            "reviewed_text",
        ),
        corpus_version=corpus_version,
        vector_similarity=_required_similarity(
            row,
            "vector_similarity",
        ),
        concept_relations=relations,
        source_title=_optional_string(row.get("source_title")),
        translator=_optional_string(row.get("translator")),
        citation_verified=(FROZEN_CITATION_VERIFIED),
        structural_locator=(FROZEN_STRUCTURAL_LOCATOR),
    )


def _parse_concept_relations(
    value: object,
    *,
    chunk_id: str,
) -> dict[
    str,
    RetrievalConceptRelationRecord,
]:
    if not isinstance(value, list):
        raise RetrievalRepositoryError(f"Chunk {chunk_id!r} concept_relations must be a list.")

    relations: dict[
        str,
        RetrievalConceptRelationRecord,
    ] = {}

    for index, raw in enumerate(
        value,
        start=1,
    ):
        item = _required_mapping(
            raw,
            (f"chunk {chunk_id!r} concept relation {index}"),
        )

        concept_id = _required_string(
            item,
            "concept_id",
        )

        if concept_id not in CONCEPTS:
            raise RetrievalRepositoryError(
                f"Chunk {chunk_id!r} has unexpected Phase 1 concept {concept_id!r}."
            )

        if concept_id in relations:
            raise RetrievalRepositoryError(
                f"Chunk {chunk_id!r} has duplicate concept relation {concept_id!r}."
            )

        human_label = _required_string(
            item,
            "human_label",
        ).casefold()

        if human_label not in {
            "positive",
            "partial",
            "negative",
        }:
            raise RetrievalRepositoryError(
                f"Chunk {chunk_id!r}/{concept_id} has invalid human_label {human_label!r}."
            )

        production_active = _required_bool(
            item,
            "production_active",
        )

        human_override = _required_bool(
            item,
            "human_override",
        )

        calibrated_weight = _required_unit_interval_float(
            item,
            "calibrated_weight",
        )

        relations[concept_id] = RetrievalConceptRelationRecord(
            concept_id=concept_id,
            human_label=human_label,
            production_active=(production_active),
            calibrated_weight=(calibrated_weight),
            human_override=(human_override),
        )

    if set(relations) != set(CONCEPTS):
        missing = sorted(set(CONCEPTS) - set(relations))

        extra = sorted(set(relations) - set(CONCEPTS))

        raise RetrievalRepositoryError(
            f"Chunk {chunk_id!r} does not have the exact "
            "three frozen Phase 1 concept relations; "
            f"missing={missing}, extra={extra}."
        )

    return relations


def _validated_embedding(
    values: Sequence[float],
) -> list[float]:
    if isinstance(
        values,
        str | bytes | bytearray,
    ):
        raise RetrievalRepositoryError("Query embedding must be a numeric sequence.")

    if len(values) != FROZEN_EMBEDDING_DIMENSION:
        raise RetrievalRepositoryError(
            "Query embedding dimension mismatch: "
            f"expected {FROZEN_EMBEDDING_DIMENSION}, "
            f"received {len(values)}."
        )

    embedding: list[float] = []

    for index, raw_value in enumerate(values):
        if isinstance(
            raw_value,
            bool,
        ):
            raise RetrievalRepositoryError(f"Query embedding contains a boolean at index {index}.")

        try:
            value = float(raw_value)
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise RetrievalRepositoryError(
                f"Query embedding contains a non-numeric value at index {index}."
            ) from exc

        if not math.isfinite(value):
            raise RetrievalRepositoryError(
                f"Query embedding contains a non-finite value at index {index}."
            )

        embedding.append(value)

    return embedding


def _validated_domain(
    value: str,
) -> Phase1Domain:
    normalized = value.strip().casefold()

    if normalized == "science":
        return "science"

    if normalized == "advaita":
        return "advaita"

    if normalized == "samkhya":
        return "samkhya"

    raise RetrievalRepositoryError(f"Unsupported Phase 1 domain {value!r}.")


def _validated_corpus_version(
    value: str,
) -> str:
    normalized = value.strip()

    if not normalized:
        raise RetrievalRepositoryError("Corpus version must be non-empty.")

    if normalized != FROZEN_CORPUS_VERSION:
        raise RetrievalRepositoryError(
            "Runtime Phase 14 retrieval is locked to "
            f"{FROZEN_CORPUS_VERSION!r}; received "
            f"{normalized!r}."
        )

    return normalized


def _required_mapping(
    value: object,
    description: str,
) -> dict[str, object]:
    if not isinstance(
        value,
        Mapping,
    ):
        raise RetrievalRepositoryError(f"{description} must be an object.")

    result: dict[
        str,
        object,
    ] = {}

    for key, nested in value.items():
        if not isinstance(
            key,
            str,
        ):
            raise RetrievalRepositoryError(f"{description} contains a non-string key.")

        result[key] = nested

    return result


def _required_string(
    record: Mapping[str, object],
    field: str,
) -> str:
    value = record.get(field)

    if (
        not isinstance(
            value,
            str,
        )
        or not value.strip()
    ):
        raise RetrievalRepositoryError(f"Database field {field!r} is missing or invalid.")

    return value.strip()


def _optional_string(
    value: object,
) -> str:
    if isinstance(
        value,
        str,
    ):
        return value.strip()

    return ""


def _required_bool(
    record: Mapping[str, object],
    field: str,
) -> bool:
    value = record.get(field)

    if not isinstance(
        value,
        bool,
    ):
        raise RetrievalRepositoryError(f"Database field {field!r} is missing or invalid.")

    return value


def _required_float(
    record: Mapping[str, object],
    field: str,
) -> float:
    raw_value = record.get(field)

    if isinstance(raw_value, bool):
        raise RetrievalRepositoryError(f"Database field {field!r} is missing or invalid.")

    try:
        # cast tells mypy raw_value is compatible with float()
        value = float(cast(str | int | float | bytes, raw_value))
    except (TypeError, ValueError) as exc:
        raise RetrievalRepositoryError(f"Database field {field!r} is missing or invalid.") from exc

    if not math.isfinite(value):
        raise RetrievalRepositoryError(f"Database field {field!r} is non-finite.")

    return value


def _required_unit_interval_float(
    record: Mapping[str, object],
    field: str,
) -> float:
    value = _required_float(
        record,
        field,
    )

    if not 0.0 <= value <= 1.0:
        raise RetrievalRepositoryError(f"Database field {field!r} must be in [0, 1].")

    return value


def _required_similarity(
    record: Mapping[str, object],
    field: str,
) -> float:
    value = _required_float(
        record,
        field,
    )

    # pgvector cosine similarity is theoretically in [-1, 1]. A tiny tolerance
    # is allowed for floating-point roundoff, but the original value is kept so
    # the later Phase 14 score calculation does not silently change it.
    tolerance = 1e-9

    if value < -1.0 - tolerance or value > 1.0 + tolerance:
        raise RetrievalRepositoryError(
            f"Database field {field!r} is outside the cosine-similarity range."
        )

    return value


__all__ = [
    "CONCEPTS",
    "DOMAINS",
    "FROZEN_ACTIVE_CHUNK_COUNT",
    "FROZEN_CORPUS_VERSION",
    "FROZEN_EMBEDDING_DIMENSION",
    "RPC_CANDIDATE_LIMIT",
    "RPC_NAME",
    "Phase1Domain",
    "RetrievalCandidateRecord",
    "RetrievalConceptRelationRecord",
    "RetrievalRepository",
    "RetrievalRepositoryError",
]
