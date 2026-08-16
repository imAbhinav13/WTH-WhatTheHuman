from __future__ import annotations

import math
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, Protocol, TypeAlias

import httpx
import numpy as np

from apps.api.models.runtime_contracts import (
    EvidencePackage,
    RetrievalManifest,
)
from apps.api.repositories.concept_repository import (
    ConceptRepository,
    ConceptRepositoryError,
)
from apps.api.repositories.retrieval_repository import (
    FROZEN_ACTIVE_CHUNK_COUNT,
    FROZEN_CORPUS_VERSION,
    Phase1Domain,
    RetrievalCandidateRecord,
    RetrievalRepository,
    RetrievalRepositoryError,
)
from apps.api.services.concept_activation import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    MODEL_VERSION,
    PROTOTYPE_VERSION,
    ConceptActivationError,
    ConceptActivationService,
    QueryActivation,
)

SCRIPT_VERSION: Final = "1.0.0"
RETRIEVAL_VERSION: Final = (
    "phase1-concept-domain-retrieval-v1"
)

DOMAINS: Final[
    tuple[Phase1Domain, ...]
] = (
    "science",
    "advaita",
    "samkhya",
)

FROZEN_DOMAIN_COUNTS: Final[
    dict[Phase1Domain, int]
] = {
    "science": 90,
    "advaita": 120,
    "samkhya": 108,
}

DEFAULT_TOP_K_PER_DOMAIN: Final = 3
DEFAULT_CANDIDATE_POOL_PER_DOMAIN: Final = 30
DEFAULT_TOKEN_BUDGET_PER_DOMAIN: Final = 900
DEFAULT_MAX_CHUNKS_PER_SOURCE: Final = 2
DEFAULT_MIN_VECTOR_SIMILARITY: Final = 0.20

VECTOR_WEIGHT: Final = 0.55
CONCEPT_ALIGNMENT_WEIGHT: Final = 0.25
HUMAN_RELEVANCE_WEIGHT: Final = 0.15
CITATION_QUALITY_WEIGHT: Final = 0.05

SOURCE_REPEAT_PENALTY: Final = 0.08
SAME_SOURCE_OVERLAP_JACCARD: Final = 0.82
CROSS_SOURCE_DUPLICATE_JACCARD: Final = 0.96
MIN_DEDUP_TOKEN_COUNT: Final = 12

# Frozen before Phase 14 retrieval evaluation.
MAX_ACCEPTABLE_RECALL_RELATIVE_LOSS: Final = 0.10
MIN_PRECISION_DELTA: Final = 0.0
MIN_CONCEPT_COVERAGE_DELTA: Final = 0.0

# The frozen Phase 14 evaluation retained the concept-aware configuration.
# Runtime requests reuse that locked version/config instead of rerunning the
# 20-question evaluation on every query.
FROZEN_RETRIEVAL_EVALUATION_COMPLETE: Final = True
FROZEN_CONCEPT_AWARE_RETAINED: Final = True

GEMINI_API_BASE: Final = (
    "https://generativelanguage.googleapis.com/v1beta"
)
QUERY_PREFIX: Final = (
    "task: search result | query: {question}"
)

TOKEN_RE: Final = re.compile(
    r"[A-Za-z0-9]+(?:[''-][A-Za-z0-9]+)?"
)
SPACE_RE: Final = re.compile(
    r"\s+"
)

DEFAULT_RETRIEVAL_CONFIG_OUTPUT: Final = (
    "artifacts/phase1/retrieval/retrieval_config.json"
)
DEFAULT_EVIDENCE_PACKAGE_OUTPUT: Final = (
    "artifacts/phase1/retrieval/evidence_package.json"
)
DEFAULT_RETRIEVAL_EVALUATION_OUTPUT: Final = (
    "artifacts/phase1/retrieval/"
    "retrieval_evaluation_results.json"
)
DEFAULT_RETRIEVAL_REPORT_OUTPUT: Final = (
    "docs/evaluation/phase1_retrieval_report.md"
)


class RetrievalError(
    RuntimeError
):
    """Raised when runtime Phase 14 retrieval cannot safely complete."""


QueryEmbedding: TypeAlias = list[float]


@dataclass(frozen=True)
class QueryEmbeddingConfig:
    """Frozen Phase 14 Gemini query-embedding configuration."""

    api_key: str
    model: str = EMBEDDING_MODEL
    dimensions: int = EMBEDDING_DIMENSIONS
    timeout_seconds: float = 45.0
    max_attempts: int = 4


@dataclass(frozen=True)
class RuntimeChunk:
    """Repository candidate enriched with frozen Phase 14 local features."""

    candidate: RetrievalCandidateRecord
    citation_quality: float
    normalized_text: str
    token_set: frozenset[str]
    estimated_tokens: int


@dataclass(frozen=True)
class RankedChunk:
    """Frozen Phase 14 ranking state."""

    chunk: RuntimeChunk
    vector_similarity: float
    concept_alignment: float
    human_relevance: float
    citation_quality: float
    base_score: float
    diversity_adjusted_score: float


@dataclass(frozen=True)
class RetrievalConfig:
    """Frozen Phase 14 retrieval selection configuration."""

    top_k_per_domain: int = (
        DEFAULT_TOP_K_PER_DOMAIN
    )
    candidate_pool_per_domain: int = (
        DEFAULT_CANDIDATE_POOL_PER_DOMAIN
    )
    token_budget_per_domain: int = (
        DEFAULT_TOKEN_BUDGET_PER_DOMAIN
    )
    max_chunks_per_source: int = (
        DEFAULT_MAX_CHUNKS_PER_SOURCE
    )
    min_vector_similarity: float = (
        DEFAULT_MIN_VECTOR_SIMILARITY
    )

    def as_dict(
        self,
    ) -> dict[str, object]:
        return {
            "top_k_per_domain": (
                self.top_k_per_domain
            ),
            "candidate_pool_per_domain": (
                self.candidate_pool_per_domain
            ),
            "token_budget_per_domain": (
                self.token_budget_per_domain
            ),
            "max_chunks_per_source": (
                self.max_chunks_per_source
            ),
            "min_vector_similarity": (
                self.min_vector_similarity
            ),
        }


@dataclass(frozen=True)
class RetrievalOutputPaths:
    """Manifest provenance fields; the runtime service performs no file I/O."""

    retrieval_config: str = (
        DEFAULT_RETRIEVAL_CONFIG_OUTPUT
    )
    evidence_package: str = (
        DEFAULT_EVIDENCE_PACKAGE_OUTPUT
    )
    retrieval_evaluation_results: str = (
        DEFAULT_RETRIEVAL_EVALUATION_OUTPUT
    )
    retrieval_report: str = (
        DEFAULT_RETRIEVAL_REPORT_OUTPUT
    )


@dataclass(frozen=True)
class RetrievalServiceResult:
    """Runtime Phase 14 outputs."""

    evidence_package: EvidencePackage
    manifest: RetrievalManifest


class QueryEmbeddingRunner(
    Protocol
):
    """Injectable boundary for exact frozen query embedding."""

    def __call__(
        self,
        *,
        question: str,
        config: QueryEmbeddingConfig,
    ) -> QueryEmbedding:
        ...


def utc_now() -> str:
    return datetime.now(
        UTC
    ).isoformat()


def normalize_text(
    text: str,
) -> str:
    """Exact historical Phase 14 normalization."""

    return SPACE_RE.sub(
        " ",
        text.casefold(),
    ).strip()


def token_set(
    text: str,
) -> frozenset[str]:
    """Exact historical Phase 14 dedup tokenization."""

    return frozenset(
        token.casefold()
        for token
        in TOKEN_RE.findall(
            text
        )
    )


def estimated_tokens(
    text: str,
) -> int:
    """Exact historical Phase 14 context-budget estimate."""

    words = max(
        1,
        len(
            text.split()
        ),
    )

    return max(
        1,
        math.ceil(
            words * 1.30
        ),
    )


def jaccard(
    left: frozenset[str],
    right: frozenset[str],
) -> float:
    """Exact historical Phase 14 token-set Jaccard."""

    if not left or not right:
        return 0.0

    union = left | right

    if not union:
        return 0.0

    return (
        len(
            left & right
        )
        / len(union)
    )


def l2_normalize(
    values: Sequence[float],
) -> QueryEmbedding:
    """Exact historical Phase 14 NumPy L2 normalization."""

    if isinstance(
        values,
        str | bytes | bytearray,
    ):
        raise RetrievalError(
            "Embedding vector must be numeric."
        )

    if (
        len(values)
        != EMBEDDING_DIMENSIONS
    ):
        raise RetrievalError(
            "Embedding dimension mismatch: "
            f"{len(values)} != {EMBEDDING_DIMENSIONS}."
        )

    try:
        vector = np.asarray(
            values,
            dtype=np.float64,
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise RetrievalError(
            "Embedding contains non-numeric values."
        ) from exc

    if not np.all(
        np.isfinite(vector)
    ):
        raise RetrievalError(
            "Embedding contains a non-finite value."
        )

    norm = float(
        np.linalg.norm(vector)
    )

    if norm <= 0.0:
        raise RetrievalError(
            "Embedding vector has zero norm."
        )

    normalized = (
        vector / norm
    )

    return [
        float(value)
        for value
        in normalized
    ]


def validate_embedding_config(
    config: QueryEmbeddingConfig,
) -> None:
    """Lock runtime query embedding to the frozen Phase 14 identity."""

    if not config.api_key.strip():
        raise RetrievalError(
            "Gemini API key must be non-empty."
        )

    if (
        config.model
        != EMBEDDING_MODEL
    ):
        raise RetrievalError(
            "Runtime Phase 14 query embedding is locked to "
            f"{EMBEDDING_MODEL!r}; received {config.model!r}."
        )

    if (
        config.dimensions
        != EMBEDDING_DIMENSIONS
    ):
        raise RetrievalError(
            "Runtime Phase 14 query embedding is locked to "
            f"{EMBEDDING_DIMENSIONS} dimensions."
        )

    if (
        config.timeout_seconds
        <= 0.0
    ):
        raise RetrievalError(
            "Embedding timeout_seconds must be positive."
        )

    if (
        config.max_attempts
        <= 0
    ):
        raise RetrievalError(
            "Embedding max_attempts must be positive."
        )


def embed_question(
    *,
    question: str,
    config: QueryEmbeddingConfig,
) -> QueryEmbedding:
    """Exact live Gemini request used by the validated Phase 14 script."""

    validate_embedding_config(
        config
    )

    normalized_question = (
        question.strip()
    )

    if not normalized_question:
        raise RetrievalError(
            "Question must be non-empty."
        )

    prepared = QUERY_PREFIX.format(
        question=normalized_question
    )

    url = (
        f"{GEMINI_API_BASE}/"
        f"models/{config.model}:embedContent"
    )

    payload = {
        "model": (
            f"models/{config.model}"
        ),
        "content": {
            "parts": [
                {
                    "text": prepared,
                }
            ]
        },
        "embedContentConfig": {
            "outputDimensionality": (
                config.dimensions
            ),
        },
    }

    for attempt in range(
        1,
        config.max_attempts + 1,
    ):
        try:
            with httpx.Client(
                timeout=(
                    config.timeout_seconds
                )
            ) as client:
                response = client.post(
                    url,
                    headers={
                        "Content-Type": (
                            "application/json"
                        ),
                        "x-goog-api-key": (
                            config.api_key
                        ),
                    },
                    json=payload,
                )

        except httpx.HTTPError as exc:
            if (
                attempt
                == config.max_attempts
            ):
                raise RetrievalError(
                    "Gemini embedding request failed: "
                    f"{exc}"
                ) from exc

            time.sleep(
                float(
                    attempt * 2
                )
            )
            continue

        if (
            response.status_code
            == 200
        ):
            try:
                raw_response = (
                    response.json()
                )
            except ValueError as exc:
                raise RetrievalError(
                    "Gemini embedding response "
                    "was not valid JSON."
                ) from exc

            if not isinstance(
                raw_response,
                Mapping,
            ):
                raise RetrievalError(
                    "Gemini embedding response "
                    "must be an object."
                )

            embedding = (
                raw_response.get(
                    "embedding"
                )
            )

            if not isinstance(
                embedding,
                Mapping,
            ):
                raise RetrievalError(
                    "Gemini embedding response "
                    "has no embedding object."
                )

            values = embedding.get(
                "values"
            )

            if not isinstance(
                values,
                list,
            ):
                raise RetrievalError(
                    "Gemini embedding response "
                    "has no values list."
                )

            if (
                len(values)
                != config.dimensions
            ):
                raise RetrievalError(
                    "Gemini query embedding dimension "
                    f"mismatch: {len(values)} "
                    f"!= {config.dimensions}."
                )

            return l2_normalize(
                values
            )

        if (
            response.status_code
            in {
                429,
                500,
                502,
                503,
                504,
            }
            and attempt
            < config.max_attempts
        ):
            time.sleep(
                float(
                    attempt * 3
                )
            )
            continue

        raise RetrievalError(
            "Gemini embedding request failed "
            f"status={response.status_code}: "
            f"{response.text[:500]}"
        )

    raise RetrievalError(
        "Gemini embedding request exhausted retries."
    )


def human_label_value(
    label: str,
) -> float:
    """Exact historical Phase 14 reviewed-label ranking value."""

    if label == "positive":
        return 1.0

    if label == "partial":
        return 0.65

    return 0.0


def citation_quality_score(
    candidate: RetrievalCandidateRecord,
) -> float:
    """Reproduce historical Phase 14 citation-quality scoring."""

    if not candidate.citation.strip():
        return 0.0

    score = 0.70

    if (
        candidate.structural_locator.strip()
    ):
        score += 0.15

    if candidate.citation_verified:
        score += 0.15

    return min(
        score,
        1.0,
    )


def runtime_chunk(
    candidate: RetrievalCandidateRecord,
) -> RuntimeChunk:
    """Enrich one DB candidate with Phase 14 deterministic local features."""

    reviewed_text = (
        candidate.reviewed_text
    )

    return RuntimeChunk(
        candidate=candidate,
        citation_quality=(
            citation_quality_score(
                candidate
            )
        ),
        normalized_text=(
            normalize_text(
                reviewed_text
            )
        ),
        token_set=token_set(
            reviewed_text
        ),
        estimated_tokens=(
            estimated_tokens(
                reviewed_text
            )
        ),
    )


def alignment_components(
    chunk: RuntimeChunk,
    activation: QueryActivation,
) -> tuple[
    float,
    float,
]:
    """Exact historical Phase 14 concept/human alignment calculation."""

    if not activation.active_concepts:
        return (
            0.0,
            0.0,
        )

    total_query_weight = sum(
        activation.calibrated_weights[
            concept
        ]
        for concept
        in activation.active_concepts
    )

    if (
        total_query_weight
        <= 0.0
    ):
        return (
            0.0,
            0.0,
        )

    alignment = 0.0
    human_relevance = 0.0

    relations = (
        chunk.candidate.concept_relations
    )

    for concept in (
        activation.active_concepts
    ):
        query_weight = (
            activation.calibrated_weights[
                concept
            ]
        )

        relation = relations[
            concept
        ]

        if relation.production_active:
            alignment += (
                query_weight
                * relation.calibrated_weight
            )

            human_relevance += (
                query_weight
                * human_label_value(
                    relation.human_label
                )
            )

    return (
        alignment
        / total_query_weight,
        human_relevance
        / total_query_weight,
    )


def score_candidate(
    *,
    chunk: RuntimeChunk,
    activation: QueryActivation,
    concept_aware: bool,
) -> RankedChunk:
    """Exact historical Phase 14 score, using pgvector cosine similarity."""

    vector_similarity = (
        chunk.candidate.vector_similarity
    )

    concept_alignment = 0.0
    human_relevance = 0.0

    if concept_aware:
        (
            concept_alignment,
            human_relevance,
        ) = alignment_components(
            chunk,
            activation,
        )

    if concept_aware:
        base_score = (
            VECTOR_WEIGHT
            * vector_similarity
            + CONCEPT_ALIGNMENT_WEIGHT
            * concept_alignment
            + HUMAN_RELEVANCE_WEIGHT
            * human_relevance
            + CITATION_QUALITY_WEIGHT
            * chunk.citation_quality
        )

    else:
        base_score = (
            vector_similarity
        )

    return RankedChunk(
        chunk=chunk,
        vector_similarity=(
            vector_similarity
        ),
        concept_alignment=(
            concept_alignment
        ),
        human_relevance=(
            human_relevance
        ),
        citation_quality=(
            chunk.citation_quality
        ),
        base_score=base_score,
        diversity_adjusted_score=(
            base_score
        ),
    )


def chunk_is_concept_eligible(
    chunk: RuntimeChunk,
    activation: QueryActivation,
) -> bool:
    """Preserve human-reviewed production_active as eligibility authority."""

    return any(
        chunk.candidate.concept_relations[
            concept
        ].production_active
        for concept
        in activation.active_concepts
    )


def is_duplicate(
    candidate: RuntimeChunk,
    selected: Sequence[RankedChunk],
) -> bool:
    """Exact historical Phase 14 deduplication policy."""

    for existing_ranked in (
        selected
    ):
        existing = (
            existing_ranked.chunk
        )

        if (
            candidate.candidate.chunk_id
            == existing.candidate.chunk_id
        ):
            return True

        if (
            candidate.normalized_text
            == existing.normalized_text
        ):
            return True

        if (
            len(candidate.token_set)
            < MIN_DEDUP_TOKEN_COUNT
            or len(existing.token_set)
            < MIN_DEDUP_TOKEN_COUNT
        ):
            continue

        overlap = jaccard(
            candidate.token_set,
            existing.token_set,
        )

        if (
            candidate.candidate.source_id
            == existing.candidate.source_id
            and overlap
            >= SAME_SOURCE_OVERLAP_JACCARD
        ):
            return True

        if (
            overlap
            >= CROSS_SOURCE_DUPLICATE_JACCARD
        ):
            return True

    return False


def select_with_diversity_and_budget(
    candidates: Sequence[RankedChunk],
    *,
    config: RetrievalConfig,
) -> list[RankedChunk]:
    """Exact historical Phase 14 iterative diversity/budget selection."""

    selected: list[
        RankedChunk
    ] = []

    source_counts: Counter[
        str
    ] = Counter()

    used_tokens = 0

    remaining = list(
        candidates
    )

    while (
        remaining
        and len(selected)
        < config.top_k_per_domain
    ):
        adjusted: list[
            RankedChunk
        ] = []

        for item in remaining:
            repeats = source_counts[
                item.chunk.candidate.source_id
            ]

            diversity_adjusted = (
                item.base_score
                - SOURCE_REPEAT_PENALTY
                * repeats
            )

            adjusted.append(
                RankedChunk(
                    chunk=item.chunk,
                    vector_similarity=(
                        item.vector_similarity
                    ),
                    concept_alignment=(
                        item.concept_alignment
                    ),
                    human_relevance=(
                        item.human_relevance
                    ),
                    citation_quality=(
                        item.citation_quality
                    ),
                    base_score=(
                        item.base_score
                    ),
                    diversity_adjusted_score=(
                        diversity_adjusted
                    ),
                )
            )

        adjusted.sort(
            key=lambda item: (
                -item.diversity_adjusted_score,
                -item.vector_similarity,
                item.chunk.candidate.source_id,
                item.chunk.candidate.chunk_id,
            )
        )

        chosen: (
            RankedChunk | None
        ) = None

        for item in adjusted:
            source_id = (
                item.chunk.candidate.source_id
            )

            if (
                source_counts[
                    source_id
                ]
                >= config.max_chunks_per_source
            ):
                continue

            if is_duplicate(
                item.chunk,
                selected,
            ):
                continue

            next_tokens = (
                used_tokens
                + item.chunk.estimated_tokens
            )

            if (
                next_tokens
                > config.token_budget_per_domain
            ):
                continue

            chosen = item
            break

        if chosen is None:
            break

        selected.append(
            chosen
        )

        chosen_source = (
            chosen.chunk.candidate.source_id
        )

        source_counts[
            chosen_source
        ] += 1

        used_tokens += (
            chosen.chunk.estimated_tokens
        )

        chosen_chunk_id = (
            chosen.chunk.candidate.chunk_id
        )

        remaining = [
            item
            for item
            in remaining
            if (
                item.chunk.candidate.chunk_id
                != chosen_chunk_id
            )
        ]

    return selected


def retrieve_domain(
    *,
    domain: Phase1Domain,
    chunks: Sequence[RuntimeChunk],
    activation: QueryActivation,
    concept_aware: bool,
    config: RetrievalConfig,
) -> list[RankedChunk]:
    """Exact historical Phase 14 domain ranking and pool truncation."""

    candidates: list[
        RankedChunk
    ] = []

    if (
        concept_aware
        and activation.unsupported
    ):
        return []

    for chunk in chunks:
        if (
            chunk.candidate.domain
            != domain
        ):
            continue

        if (
            concept_aware
            and not chunk_is_concept_eligible(
                chunk,
                activation,
            )
        ):
            continue

        scored = score_candidate(
            chunk=chunk,
            activation=activation,
            concept_aware=concept_aware,
        )

        if (
            scored.vector_similarity
            < config.min_vector_similarity
        ):
            continue

        candidates.append(
            scored
        )

    # CRITICAL PARITY POINT:
    # weighted base score is calculated over the complete active domain
    # candidate set BEFORE applying the frozen pool size of 30.
    candidates.sort(
        key=lambda item: (
            -item.base_score,
            -item.vector_similarity,
            item.chunk.candidate.source_id,
            item.chunk.candidate.chunk_id,
        )
    )

    pool = candidates[
        : config.candidate_pool_per_domain
    ]

    return (
        select_with_diversity_and_budget(
            pool,
            config=config,
        )
    )


def retrieve_all_domains(
    *,
    candidates_by_domain: Mapping[
        Phase1Domain,
        Sequence[RetrievalCandidateRecord],
    ],
    activation: QueryActivation,
    concept_aware: bool,
    config: RetrievalConfig,
) -> dict[
    Phase1Domain,
    list[RankedChunk],
]:
    """Apply the frozen ranking independently inside each domain."""

    return {
        domain: retrieve_domain(
            domain=domain,
            chunks=[
                runtime_chunk(
                    candidate
                )
                for candidate
                in candidates_by_domain[
                    domain
                ]
            ],
            activation=activation,
            concept_aware=concept_aware,
            config=config,
        )
        for domain in DOMAINS
    }


def ranked_chunk_payload(
    item: RankedChunk,
    *,
    rank: int,
) -> dict[str, object]:
    """Exact historical Phase 14 evidence-item schema."""

    chunk = item.chunk
    candidate = (
        chunk.candidate
    )

    return {
        "rank": rank,
        "chunk_id": (
            candidate.chunk_id
        ),
        "source_id": (
            candidate.source_id
        ),
        "domain": candidate.domain,
        "citation": (
            candidate.citation
        ),
        "reviewed_text": (
            candidate.reviewed_text
        ),
        "corpus_version": (
            candidate.corpus_version
        ),
        "estimated_tokens": (
            chunk.estimated_tokens
        ),
        # Keep the validated historical field name. Phase 17's frozen
        # derived-proxy behavior depends on the artifact carrying "scores".
        "scores": {
            "vector_similarity": (
                item.vector_similarity
            ),
            "concept_alignment": (
                item.concept_alignment
            ),
            "human_relevance": (
                item.human_relevance
            ),
            "citation_quality": (
                item.citation_quality
            ),
            "base_score": (
                item.base_score
            ),
            "diversity_adjusted_score": (
                item.diversity_adjusted_score
            ),
        },
        "concepts": {
            concept: {
                "human_label": (
                    relation.human_label
                ),
                "production_active": (
                    relation.production_active
                ),
                "calibrated_weight": (
                    relation.calibrated_weight
                ),
                "human_override": (
                    relation.human_override
                ),
            }
            for concept, relation
            in sorted(
                candidate.concept_relations.items()
            )
        },
    }


def build_evidence_package_payload(
    *,
    question: str,
    activation: QueryActivation,
    retrieval: Mapping[
        Phase1Domain,
        Sequence[RankedChunk],
    ],
    config: RetrievalConfig,
    generated_at: str,
) -> dict[str, object]:
    """Exact historical concept-aware EvidencePackage payload."""

    domains: dict[
        str,
        object,
    ] = {}

    for domain in DOMAINS:
        items = retrieval[
            domain
        ]

        domains[
            domain
        ] = {
            "status": (
                "evidence_found"
                if items
                else "no_strong_match"
            ),
            "evidence_count": (
                len(items)
            ),
            "estimated_tokens": sum(
                item.chunk.estimated_tokens
                for item in items
            ),
            "unique_source_count": len(
                {
                    item.chunk.candidate.source_id
                    for item in items
                }
            ),
            "evidence": [
                ranked_chunk_payload(
                    item,
                    rank=index,
                )
                for index, item
                in enumerate(
                    items,
                    start=1,
                )
            ],
        }

    return {
        "retrieval_version": (
            RETRIEVAL_VERSION
        ),
        "retrieval_mode": (
            "concept_aware"
        ),
        "generated_at": (
            generated_at
        ),
        "question": question,
        "query_activation": (
            activation.evidence_payload()
        ),
        "config": config.as_dict(),
        "scoring": {
            "vector_similarity_weight": (
                VECTOR_WEIGHT
            ),
            "concept_alignment_weight": (
                CONCEPT_ALIGNMENT_WEIGHT
            ),
            "human_relevance_weight": (
                HUMAN_RELEVANCE_WEIGHT
            ),
            "citation_quality_weight": (
                CITATION_QUALITY_WEIGHT
            ),
            "source_repeat_penalty": (
                SOURCE_REPEAT_PENALTY
            ),
        },
        "corpus_version": (
            FROZEN_CORPUS_VERSION
        ),
        "model_version": (
            MODEL_VERSION
        ),
        "prototype_version": (
            PROTOTYPE_VERSION
        ),
        "domains": domains,
    }


def build_retrieval_manifest_payload(
    *,
    config: RetrievalConfig,
    generated_at: str,
    output_paths: RetrievalOutputPaths,
) -> dict[str, object]:
    """Build runtime manifest using the already-frozen evaluation gate.

    ``retrieval_evaluation_complete`` and ``concept_aware_retained`` refer to
    the frozen Phase 14 retrieval version/config evaluation, not to a new
    evaluation run for this individual user question.
    """

    return {
        "phase": (
            "phase_14_build_retrieval_by_concept_and_domain"
        ),
        "status": (
            "evaluation_complete"
        ),
        "script_version": (
            SCRIPT_VERSION
        ),
        "retrieval_version": (
            RETRIEVAL_VERSION
        ),
        "generated_at": (
            generated_at
        ),
        "corpus_version": (
            FROZEN_CORPUS_VERSION
        ),
        "active_chunk_count": (
            FROZEN_ACTIVE_CHUNK_COUNT
        ),
        "retrieval_config": (
            config.as_dict()
        ),
        "outputs": {
            "retrieval_config": (
                output_paths.retrieval_config
            ),
            "evidence_package": (
                output_paths.evidence_package
            ),
            "retrieval_evaluation_results": (
                output_paths.retrieval_evaluation_results
            ),
            "retrieval_report": (
                output_paths.retrieval_report
            ),
        },
        "exit_gate": {
            "question_embedding_uses_frozen_model": True,
            (
                "weighted_concept_activation_"
                "uses_frozen_phase10"
            ): True,
            "only_active_chunks_retrieved": True,
            "domain_separation_enforced": True,
            "source_diversity_enforced": True,
            "deduplication_enforced": True,
            (
                "per_domain_context_budgets_"
                "enforced"
            ): True,
            "retrieval_evaluation_complete": (
                FROZEN_RETRIEVAL_EVALUATION_COMPLETE
            ),
            "concept_aware_retained": (
                FROZEN_CONCEPT_AWARE_RETAINED
            ),
        },
        "next_step": (
            "If retrieval evaluation passes, freeze this "
            "retrieval configuration and begin Phase 15 "
            "domain-specific generation. If the dedicated "
            "frozen retrieval question file does not yet "
            "exist, freeze it before evaluating; do not use "
            "Phase 11 Held-out outcomes to tune retrieval."
        ),
    }


def validate_config(
    config: RetrievalConfig,
) -> None:
    """Historical validation plus Stage 3 lock to the evaluated config."""

    if (
        config.top_k_per_domain
        <= 0
    ):
        raise RetrievalError(
            "top_k_per_domain must be positive."
        )

    if (
        config.candidate_pool_per_domain
        < config.top_k_per_domain
    ):
        raise RetrievalError(
            "candidate_pool_per_domain must be "
            ">= top_k_per_domain."
        )

    if (
        config.token_budget_per_domain
        <= 0
    ):
        raise RetrievalError(
            "token_budget_per_domain must be positive."
        )

    if (
        config.max_chunks_per_source
        <= 0
    ):
        raise RetrievalError(
            "max_chunks_per_source must be positive."
        )

    if not (
        -1.0
        <= config.min_vector_similarity
        <= 1.0
    ):
        raise RetrievalError(
            "min_vector_similarity must be between -1 and 1."
        )

    # Stage 3 production uses the exact configuration whose retention gate
    # passed. Runtime tuning would invalidate that proof.
    frozen = RetrievalConfig()

    if config != frozen:
        raise RetrievalError(
            "Runtime Phase 14 retrieval configuration is frozen. "
            "Do not retune top-k, candidate pool, token budget, "
            "source cap, or vector threshold in Stage 3."
        )


def validate_complete_candidate_set(
    candidates_by_domain: Mapping[
        Phase1Domain,
        Sequence[RetrievalCandidateRecord],
    ],
) -> None:
    """Prove the RPC returned the complete frozen 318-chunk representation."""

    if (
        set(candidates_by_domain)
        != set(DOMAINS)
    ):
        raise RetrievalError(
            "Runtime candidate map does not contain "
            "exactly the three Phase 1 domains."
        )

    seen: set[
        str
    ] = set()

    total = 0

    for domain in DOMAINS:
        rows = candidates_by_domain[
            domain
        ]

        expected_count = (
            FROZEN_DOMAIN_COUNTS[
                domain
            ]
        )

        if (
            len(rows)
            != expected_count
        ):
            raise RetrievalError(
                f"{domain} active candidate count changed: "
                f"expected {expected_count}, "
                f"received {len(rows)}."
            )

        for row in rows:
            if row.domain != domain:
                raise RetrievalError(
                    "Runtime retrieval candidate crossed "
                    "its domain boundary."
                )

            if (
                row.corpus_version
                != FROZEN_CORPUS_VERSION
            ):
                raise RetrievalError(
                    "Runtime retrieval candidate crossed "
                    "its corpus-version boundary."
                )

            if (
                row.chunk_id
                in seen
            ):
                raise RetrievalError(
                    "Runtime retrieval candidate appeared "
                    "in more than one domain."
                )

            seen.add(
                row.chunk_id
            )

        total += len(rows)

    if (
        total
        != FROZEN_ACTIVE_CHUNK_COUNT
    ):
        raise RetrievalError(
            "Runtime active candidate total changed: "
            f"expected {FROZEN_ACTIVE_CHUNK_COUNT}, "
            f"received {total}."
        )


class RetrievalService:
    """Execute production Phase 14 without local artifact dependencies."""

    def __init__(
        self,
        *,
        retrieval_repository: RetrievalRepository,
        concept_repository: ConceptRepository,
        embedding_config: QueryEmbeddingConfig,
        embedding_runner: QueryEmbeddingRunner = (
            embed_question
        ),
    ) -> None:
        self._retrieval_repository = (
            retrieval_repository
        )
        self._concept_repository = (
            concept_repository
        )
        self._embedding_config = (
            embedding_config
        )
        self._embedding_runner = (
            embedding_runner
        )

        validate_embedding_config(
            embedding_config
        )

    def retrieve(
        self,
        *,
        question: str,
        config: RetrievalConfig | None = None,
        generated_at: str | None = None,
        manifest_generated_at: str | None = None,
        output_paths: RetrievalOutputPaths | None = None,
    ) -> RetrievalServiceResult:
        """Run the frozen production Phase 14 query path.

        The Gemini embedding is generated exactly once and reused by both
        concept activation and Supabase pgvector candidate retrieval.
        """

        normalized_question = (
            question.strip()
        )

        if not normalized_question:
            raise RetrievalError(
                "Question must be non-empty."
            )

        retrieval_config = (
            config
            if config is not None
            else RetrievalConfig()
        )

        validate_config(
            retrieval_config
        )

        paths = (
            output_paths
            if output_paths is not None
            else RetrievalOutputPaths()
        )

        try:
            query_embedding = (
                self._embedding_runner(
                    question=normalized_question,
                    config=(
                        self._embedding_config
                    ),
                )
            )

            # Enforce the frozen dimensions/normalization even for an injected
            # test runner before the vector reaches either downstream path.
            query_embedding = l2_normalize(
                query_embedding
            )

            prototype_bank = (
                self._concept_repository
                .get_prototype_bank()
            )

            activation = (
                ConceptActivationService(
                    prototype_bank
                ).activate(
                    question=(
                        normalized_question
                    ),
                    query_embedding=(
                        query_embedding
                    ),
                )
            )

            if activation.unsupported:
                candidates_by_domain: dict[
                    Phase1Domain,
                    list[
                        RetrievalCandidateRecord
                    ],
                ] = {
                    domain: []
                    for domain
                    in DOMAINS
                }

                retrieval: dict[
                    Phase1Domain,
                    list[
                        RankedChunk
                    ],
                ] = {
                    domain: []
                    for domain
                    in DOMAINS
                }

            else:
                candidates_by_domain = (
                    self._retrieval_repository
                    .get_all_domain_candidates(
                        query_embedding=(
                            query_embedding
                        ),
                        corpus_version=(
                            FROZEN_CORPUS_VERSION
                        ),
                    )
                )

                validate_complete_candidate_set(
                    candidates_by_domain
                )

                retrieval = retrieve_all_domains(
                    candidates_by_domain=(
                        candidates_by_domain
                    ),
                    activation=activation,
                    concept_aware=True,
                    config=retrieval_config,
                )

        except (
            ConceptRepositoryError,
            ConceptActivationError,
            RetrievalRepositoryError,
        ) as exc:
            raise RetrievalError(
                "Runtime Phase 14 retrieval dependency failed."
            ) from exc

        evidence_timestamp = (
            generated_at
            or utc_now()
        )

        manifest_timestamp = (
            manifest_generated_at
            or utc_now()
        )

        evidence_payload = (
            build_evidence_package_payload(
                question=(
                    normalized_question
                ),
                activation=activation,
                retrieval=retrieval,
                config=(
                    retrieval_config
                ),
                generated_at=(
                    evidence_timestamp
                ),
            )
        )

        manifest_payload = (
            build_retrieval_manifest_payload(
                config=(
                    retrieval_config
                ),
                generated_at=(
                    manifest_timestamp
                ),
                output_paths=paths,
            )
        )

        try:
            evidence_model = (
                EvidencePackage.model_validate(
                    evidence_payload
                )
            )

            manifest_model = (
                RetrievalManifest.model_validate(
                    manifest_payload
                )
            )

        except Exception as exc:
            raise RetrievalError(
                "Runtime Phase 14 output does not match "
                "the frozen Stage 3.0 contract."
            ) from exc

        return RetrievalServiceResult(
            evidence_package=(
                evidence_model
            ),
            manifest=manifest_model,
        )


__all__ = [
    "CITATION_QUALITY_WEIGHT",
    "CONCEPT_ALIGNMENT_WEIGHT",
    "CROSS_SOURCE_DUPLICATE_JACCARD",
    "DEFAULT_CANDIDATE_POOL_PER_DOMAIN",
    "DEFAULT_MAX_CHUNKS_PER_SOURCE",
    "DEFAULT_MIN_VECTOR_SIMILARITY",
    "DEFAULT_TOKEN_BUDGET_PER_DOMAIN",
    "DEFAULT_TOP_K_PER_DOMAIN",
    "DOMAINS",
    "FROZEN_CONCEPT_AWARE_RETAINED",
    "FROZEN_DOMAIN_COUNTS",
    "FROZEN_RETRIEVAL_EVALUATION_COMPLETE",
    "HUMAN_RELEVANCE_WEIGHT",
    "MIN_DEDUP_TOKEN_COUNT",
    "QUERY_PREFIX",
    "RETRIEVAL_VERSION",
    "SAME_SOURCE_OVERLAP_JACCARD",
    "SOURCE_REPEAT_PENALTY",
    "VECTOR_WEIGHT",
    "QueryEmbeddingConfig",
    "QueryEmbeddingRunner",
    "RankedChunk",
    "RetrievalConfig",
    "RetrievalError",
    "RetrievalOutputPaths",
    "RetrievalService",
    "RetrievalServiceResult",
    "RuntimeChunk",
    "alignment_components",
    "build_evidence_package_payload",
    "build_retrieval_manifest_payload",
    "citation_quality_score",
    "embed_question",
    "estimated_tokens",
    "human_label_value",
    "is_duplicate",
    "jaccard",
    "normalize_text",
    "ranked_chunk_payload",
    "retrieve_all_domains",
    "retrieve_domain",
    "runtime_chunk",
    "score_candidate",
    "select_with_diversity_and_budget",
    "token_set",
    "validate_complete_candidate_set",
    "validate_config",
]
