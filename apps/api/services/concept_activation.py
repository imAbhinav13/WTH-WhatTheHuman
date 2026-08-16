from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray


class ConceptActivationError(RuntimeError):
    """Raised when frozen Phase 10 activation cannot safely complete."""


FloatArray: TypeAlias = NDArray[np.float64]

CONCEPTS: Final[tuple[str, ...]] = (
    "consciousness",
    "self_identity",
    "reality_appearance",
)

EMBEDDING_PROVIDER: Final = "Google Gemini API"
EMBEDDING_MODEL: Final = "gemini-embedding-2"
EMBEDDING_MODEL_REVISION: Final = "2"
EMBEDDING_DIMENSIONS: Final = 768
EMBEDDING_NORMALIZATION: Final = "provider_auto_l2"

PROTOTYPE_VERSION: Final = "phase1-prototype-v2"
MAPPING_METHOD: Final = "hybrid:question:centroid"
MODEL_VERSION: Final = "gemini-embedding-2@2"

EPSILON: Final = 1e-12

# Frozen Phase 10 selected development result.
FROZEN_EMBEDDING_WEIGHT: Final = 0.9
FROZEN_LEXICAL_WEIGHT: Final = 0.2
FROZEN_NEGATIVE_PENALTY: Final = 0.5
FROZEN_PROTOTYPE_SOURCE: Final = "question"
FROZEN_PROTOTYPE_AGGREGATION: Final = "centroid"
FROZEN_METHOD: Final = "hybrid"

FROZEN_ACTIVATION_THRESHOLDS: Final[dict[str, float]] = {
    "consciousness": 0.4,
    "reality_appearance": 0.75,
    "self_identity": 0.7,
}

FROZEN_AMBIGUITY_MARGIN: Final = 0.1
FROZEN_MAX_ACTIVE_CONCEPTS: Final = 3

FROZEN_CALIBRATION: Final[dict[str, tuple[float, float]]] = {
    # concept: (slope, intercept)
    "consciousness": (
        1.1810531849391321,
        -0.8799501273554392,
    ),
    "reality_appearance": (
        1.7509856440379357,
        0.47630047840111434,
    ),
    "self_identity": (
        1.4953476724245536,
        0.4463238424461934,
    ),
}

# Copied from the validated Phase 10 implementation.
LEXICAL_INDICATORS: Final[dict[str, tuple[str, ...]]] = {
    "consciousness": (
        "conscious",
        "consciousness",
        "awareness",
        "aware",
        "experience",
        "experiential",
        "subjective",
        "qualia",
        "witness",
        "sentience",
    ),
    "self_identity": (
        "self",
        "identity",
        "ego",
        "subject",
        "atman",
        "purusha",
        "ahamkara",
        "jiva",
        "first person",
    ),
    "reality_appearance": (
        "reality",
        "real",
        "appearance",
        "illusion",
        "illusory",
        "maya",
        "perception",
        "representation",
        "superimposition",
        "phenomenal",
    ),
}

# Copied from the validated Phase 10 implementation.
LEXICAL_EXCLUSIONS: Final[dict[str, tuple[str, ...]]] = {
    "consciousness": (
        "attention",
        "cognition",
        "cognitive",
        "information processing",
    ),
    "self_identity": (
        "personality",
        "autobiographical",
        "body representation",
    ),
    "reality_appearance": (
        "cosmology",
        "cosmological",
        "creation sequence",
        "evolution of elements",
    ),
}


@dataclass(frozen=True)
class Calibration:
    """Frozen concept-specific sigmoid calibration."""

    slope: float
    intercept: float

    def apply(
        self,
        raw: float,
    ) -> float:
        return sigmoid_scalar(self.slope * raw + self.intercept)


@dataclass(frozen=True)
class Candidate:
    """Frozen Phase 10 selected concept-mapping method."""

    method: str
    source: str
    aggregation: str
    negative_penalty: float = 0.0
    lexical_weight: float = 0.0
    embed_weight: float = 1.0


@dataclass(frozen=True)
class Activation:
    """Frozen Phase 10 activation policy."""

    thresholds: dict[str, float]
    ambiguity_margin: float
    max_active: int


@dataclass(frozen=True)
class LabeledRecord:
    """Phase 10 scoring record shape.

    Runtime activation uses a synthetic record whose labels are irrelevant to
    scoring. The shape is retained because ``raw_scores`` is extracted from
    the validated Phase 10 implementation.
    """

    chunk_id: str
    split: str
    text: str
    labels: dict[str, str]
    hard_negative_for: tuple[str, ...]


@dataclass(frozen=True)
class PrototypeBank:
    """Normalized frozen Phase 1 concept prototype vectors."""

    question: dict[
        str,
        tuple[FloatArray, ...],
    ]
    passage: dict[
        str,
        tuple[FloatArray, ...],
    ]
    hard_negative: dict[
        str,
        tuple[FloatArray, ...],
    ]


@dataclass(frozen=True)
class QueryActivation:
    """Phase 14-compatible query activation result."""

    question: str
    raw_scores: dict[str, float]
    calibrated_weights: dict[str, float]
    active_concepts: tuple[str, ...]
    ambiguous: bool
    unsupported: bool

    def evidence_payload(
        self,
    ) -> dict[str, object]:
        """Return the exact query_activation shape embedded in Phase 14."""

        return {
            "raw_scores": dict(self.raw_scores),
            "calibrated_weights": dict(self.calibrated_weights),
            "active_concepts": list(self.active_concepts),
            "ambiguous": self.ambiguous,
            "unsupported": self.unsupported,
        }


FROZEN_CANDIDATE: Final = Candidate(
    method=FROZEN_METHOD,
    source=FROZEN_PROTOTYPE_SOURCE,
    aggregation=(FROZEN_PROTOTYPE_AGGREGATION),
    negative_penalty=(FROZEN_NEGATIVE_PENALTY),
    lexical_weight=(FROZEN_LEXICAL_WEIGHT),
    embed_weight=(FROZEN_EMBEDDING_WEIGHT),
)

FROZEN_ACTIVATION: Final = Activation(
    thresholds=dict(FROZEN_ACTIVATION_THRESHOLDS),
    ambiguity_margin=(FROZEN_AMBIGUITY_MARGIN),
    max_active=(FROZEN_MAX_ACTIVE_CONCEPTS),
)

FROZEN_CALIBRATIONS: Final[dict[str, Calibration]] = {
    concept: Calibration(
        slope=values[0],
        intercept=values[1],
    )
    for concept, values in FROZEN_CALIBRATION.items()
}


def normalize_text(
    value: str,
) -> str:
    """Validated Phase 10 lexical normalization."""

    value = unicodedata.normalize(
        "NFKD",
        value,
    )

    value = "".join(character for character in value if not unicodedata.combining(character))

    value = re.sub(
        r"[''`']",
        "",
        value.casefold(),
    )

    value = re.sub(
        r"[^\w-]+",
        " ",
        value,
        flags=re.UNICODE,
    )

    return " ".join(value.split())


def sigmoid_scalar(
    value: float,
) -> float:
    """Validated Phase 10 bounded sigmoid."""

    value = max(
        -40.0,
        min(
            40.0,
            value,
        ),
    )

    return 1.0 / (1.0 + math.exp(-value))


def centroid(
    vectors: Sequence[FloatArray],
) -> FloatArray:
    """Validated Phase 10 normalized centroid."""

    mean = np.mean(
        np.stack(
            vectors,
            axis=0,
        ),
        axis=0,
    )

    norm = float(np.linalg.norm(mean))

    if norm <= EPSILON:
        raise ConceptActivationError("Prototype centroid has zero norm.")

    return cast(
        FloatArray,
        mean / norm,
    )


def positive_vectors(
    bank: PrototypeBank,
    concept: str,
    source: str,
) -> tuple[FloatArray, ...]:
    """Validated Phase 10 positive-prototype selector."""

    if source == "question":
        return bank.question[concept]

    if source == "passage":
        return bank.passage[concept]

    if source == "combined":
        return (
            *bank.question[concept],
            *bank.passage[concept],
        )

    raise ConceptActivationError(f"Unknown prototype source {source!r}.")


def positive_similarity(
    vector: FloatArray,
    bank: PrototypeBank,
    concept: str,
    source: str,
    aggregation: str,
) -> float:
    """Validated Phase 10 positive prototype similarity."""

    vectors = positive_vectors(
        bank,
        concept,
        source,
    )

    if aggregation == "centroid":
        return float(vector @ centroid(vectors))

    if aggregation == "maximum":
        return max(float(vector @ item) for item in vectors)

    raise ConceptActivationError(f"Unknown aggregation {aggregation!r}.")


def negative_similarity(
    vector: FloatArray,
    bank: PrototypeBank,
    concept: str,
) -> float:
    """Validated Phase 10 maximum hard-negative similarity."""

    return max(float(vector @ item) for item in bank.hard_negative[concept])


def lexical_score(
    text: str,
    concept: str,
) -> float:
    """Validated Phase 10 lexical contribution."""

    normalized = normalize_text(text)

    positive_hits = sum(normalize_text(term) in normalized for term in LEXICAL_INDICATORS[concept])

    negative_hits = sum(normalize_text(term) in normalized for term in LEXICAL_EXCLUSIONS[concept])

    positive = min(
        1.0,
        positive_hits / 3.0,
    )

    negative = min(
        1.0,
        negative_hits / 2.0,
    )

    return max(
        0.0,
        min(
            1.0,
            positive - 0.5 * negative,
        ),
    )


def raw_scores(
    candidate: Candidate,
    record: LabeledRecord,
    vector: FloatArray,
    bank: PrototypeBank,
    classifiers: Mapping[
        str,
        tuple[FloatArray, float],
    ],
) -> dict[str, float]:
    """Validated Phase 10 raw concept-score function.

    The frozen production candidate is ``hybrid``. Other branches are retained
    where they do not require training state so this function remains faithful
    to the original implementation. Classifier mode is intentionally rejected
    because Stage 3 runtime uses the frozen hybrid candidate and carries no
    trained classifier weights.
    """

    scores: dict[
        str,
        float,
    ] = {}

    for concept in CONCEPTS:
        if candidate.method == "classifier":
            classifier = classifiers.get(concept)

            if classifier is None:
                raise ConceptActivationError(
                    "Classifier scoring requested without frozen classifier state."
                )

            weights, bias = classifier

            scores[concept] = sigmoid_scalar(float(vector @ weights + bias))

            continue

        positive = positive_similarity(
            vector,
            bank,
            concept,
            candidate.source,
            candidate.aggregation,
        )

        if candidate.method in {
            "centroid",
            "maximum_example",
        }:
            scores[concept] = positive
            continue

        negative = negative_similarity(
            vector,
            bank,
            concept,
        )

        if candidate.method == "positive_minus_negative":
            scores[concept] = positive - candidate.negative_penalty * negative

            continue

        if candidate.method == "hybrid":
            scores[concept] = candidate.embed_weight * (
                positive - candidate.negative_penalty * negative
            ) + candidate.lexical_weight * lexical_score(
                record.text,
                concept,
            )

            continue

        raise ConceptActivationError(f"Unknown method {candidate.method!r}.")

    return scores


def apply_calibration(
    raw: Mapping[str, float],
    calibrations: Mapping[
        str,
        Calibration,
    ],
) -> dict[str, float]:
    """Validated Phase 10 concept-specific calibration."""

    return {concept: calibrations[concept].apply(raw[concept]) for concept in CONCEPTS}


def predicted_active(
    weights: Mapping[str, float],
    activation: Activation,
) -> list[str]:
    """Validated Phase 10 active-concept selection."""

    eligible = [
        concept for concept in CONCEPTS if weights[concept] >= activation.thresholds[concept]
    ]

    eligible.sort(
        key=lambda concept: (
            -weights[concept],
            CONCEPTS.index(concept),
        )
    )

    return eligible[: activation.max_active]


def predicted_ambiguous(
    weights: Mapping[str, float],
    activation: Activation,
) -> bool:
    """Validated Phase 10 ambiguity classifier."""

    eligible = [
        concept for concept in CONCEPTS if weights[concept] >= activation.thresholds[concept]
    ]

    if len(eligible) < 2:
        return False

    ranked = sorted(
        (weights[concept] for concept in eligible),
        reverse=True,
    )

    return ranked[0] - ranked[1] <= activation.ambiguity_margin


def build_prototype_bank(
    *,
    question: Mapping[
        str,
        Sequence[Sequence[float]],
    ],
    passage: Mapping[
        str,
        Sequence[Sequence[float]],
    ],
    hard_negative: Mapping[
        str,
        Sequence[Sequence[float]],
    ],
) -> PrototypeBank:
    """Build the Phase 10 bank from runtime numeric prototype records.

    This is the runtime equivalent of Phase 10 ``load_prototype_bank`` without
    local JSONL I/O. Every supplied vector is normalized using the same logic
    as Phase 10 ``vector_from_record``.
    """

    return PrototypeBank(
        question=_validated_vector_bank(
            question,
            description="query prototype",
        ),
        passage=_validated_vector_bank(
            passage,
            description="positive passage prototype",
        ),
        hard_negative=_validated_vector_bank(
            hard_negative,
            description="hard-negative prototype",
        ),
    )


def _validated_vector_bank(
    values: Mapping[
        str,
        Sequence[Sequence[float]],
    ],
    *,
    description: str,
) -> dict[
    str,
    tuple[FloatArray, ...],
]:
    result: dict[
        str,
        tuple[FloatArray, ...],
    ] = {}

    extra = sorted(set(values) - set(CONCEPTS))

    if extra:
        raise ConceptActivationError(
            f"{description} bank has unknown concepts: " + ", ".join(extra)
        )

    for concept in CONCEPTS:
        raw_vectors = values.get(concept)

        if raw_vectors is None or len(raw_vectors) == 0:
            raise ConceptActivationError(f"{concept} has no {description} vectors.")

        normalized: list[FloatArray] = []

        for index, raw in enumerate(
            raw_vectors,
            start=1,
        ):
            normalized.append(
                _normalize_vector(
                    raw,
                    description=(f"{description} {concept} #{index}"),
                )
            )

        result[concept] = tuple(normalized)

    return result


def _normalize_vector(
    values: Sequence[float],
    *,
    description: str,
) -> FloatArray:
    if isinstance(
        values,
        str | bytes | bytearray,
    ):
        raise ConceptActivationError(f"{description} must be a numeric vector.")

    if len(values) != EMBEDDING_DIMENSIONS:
        raise ConceptActivationError(
            f"{description} dimension mismatch: {len(values)} != {EMBEDDING_DIMENSIONS}."
        )

    numbers: list[float] = []

    for index, raw in enumerate(values):
        if isinstance(
            raw,
            bool,
        ):
            raise ConceptActivationError(f"{description} contains a boolean at index {index}.")

        try:
            number = float(raw)
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ConceptActivationError(
                f"{description} contains a non-numeric value at index {index}."
            ) from exc

        if not math.isfinite(number):
            raise ConceptActivationError(
                f"{description} contains a non-finite value at index {index}."
            )

        numbers.append(number)

    vector = np.asarray(
        numbers,
        dtype=np.float64,
    )

    norm = float(np.linalg.norm(vector))

    if norm <= EPSILON:
        raise ConceptActivationError(f"{description} vector is zero.")

    return vector / norm


class ConceptActivationService:
    """Run frozen Phase 10 activation from in-memory runtime inputs."""

    def __init__(
        self,
        bank: PrototypeBank,
    ) -> None:
        self._bank = bank
        self._validate_bank()

    def activate(
        self,
        *,
        question: str,
        query_embedding: Sequence[float],
    ) -> QueryActivation:
        """Return the exact Phase 14 query-activation semantics."""

        normalized_question = question.strip()

        if not normalized_question:
            raise ConceptActivationError("Question must be non-empty.")

        # Historical Phase 14 l2-normalized the Gemini query embedding before
        # invoking query_activation(). Do the same at this runtime boundary.
        query_vector = _normalize_vector(
            query_embedding,
            description="runtime query embedding",
        )

        record = LabeledRecord(
            chunk_id="runtime_query",
            split="runtime",
            text=normalized_question,
            labels=dict.fromkeys(
                CONCEPTS,
                "negative",
            ),
            hard_negative_for=(),
        )

        raw = raw_scores(
            FROZEN_CANDIDATE,
            record,
            query_vector,
            self._bank,
            {},
        )

        calibrated = apply_calibration(
            raw,
            FROZEN_CALIBRATIONS,
        )

        active = tuple(
            predicted_active(
                calibrated,
                FROZEN_ACTIVATION,
            )
        )

        # This second ordering step is copied from Phase 14 query_activation().
        # It is intentionally alphabetical for exact-weight ties rather than
        # using the Phase 10 CONCEPTS-index tie-breaker above.
        eligible = sorted(
            (
                (
                    concept,
                    calibrated[concept],
                )
                for concept in active
            ),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )

        ambiguous = (
            len(eligible) >= 2
            and abs(eligible[0][1] - eligible[1][1]) <= FROZEN_ACTIVATION.ambiguity_margin
        )

        return QueryActivation(
            question=normalized_question,
            raw_scores={concept: float(raw[concept]) for concept in CONCEPTS},
            calibrated_weights={concept: float(calibrated[concept]) for concept in CONCEPTS},
            active_concepts=tuple(concept for concept, _ in eligible),
            ambiguous=ambiguous,
            unsupported=not bool(active),
        )

    def _validate_bank(
        self,
    ) -> None:
        for name, bank_part in (
            (
                "question",
                self._bank.question,
            ),
            (
                "passage",
                self._bank.passage,
            ),
            (
                "hard_negative",
                self._bank.hard_negative,
            ),
        ):
            if set(bank_part) != set(CONCEPTS):
                raise ConceptActivationError(
                    f"Prototype bank {name!r} does not contain exactly the frozen Phase 1 concepts."
                )

            for concept in CONCEPTS:
                vectors = bank_part[concept]

                if not vectors:
                    raise ConceptActivationError(f"{concept} has no {name} prototypes.")

                for index, vector in enumerate(
                    vectors,
                    start=1,
                ):
                    if vector.shape != (EMBEDDING_DIMENSIONS,):
                        raise ConceptActivationError(
                            f"{name} prototype {concept} #{index} has the wrong dimension."
                        )

                    if not np.all(np.isfinite(vector)):
                        raise ConceptActivationError(
                            f"{name} prototype {concept} #{index} contains a non-finite value."
                        )

                    norm = float(np.linalg.norm(vector))

                    if abs(norm - 1.0) > 1e-9:
                        raise ConceptActivationError(
                            f"{name} prototype {concept} #{index} is not L2 normalized."
                        )


__all__ = [
    "CONCEPTS",
    "EMBEDDING_DIMENSIONS",
    "EMBEDDING_MODEL",
    "EMBEDDING_MODEL_REVISION",
    "EMBEDDING_NORMALIZATION",
    "EMBEDDING_PROVIDER",
    "FROZEN_ACTIVATION",
    "FROZEN_CALIBRATIONS",
    "FROZEN_CANDIDATE",
    "MAPPING_METHOD",
    "MODEL_VERSION",
    "PROTOTYPE_VERSION",
    "Activation",
    "Calibration",
    "Candidate",
    "ConceptActivationError",
    "ConceptActivationService",
    "FloatArray",
    "LabeledRecord",
    "PrototypeBank",
    "QueryActivation",
    "apply_calibration",
    "build_prototype_bank",
    "centroid",
    "lexical_score",
    "negative_similarity",
    "normalize_text",
    "positive_similarity",
    "positive_vectors",
    "predicted_active",
    "predicted_ambiguous",
    "raw_scores",
    "sigmoid_scalar",
]
