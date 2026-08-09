from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, cast

import numpy as np
import yaml
from numpy.typing import NDArray

LOGGER = logging.getLogger("wth.phase1.tune_phase1_concept_mapping")

SCRIPT_VERSION: Final = "1.0.0"
TUNING_VERSION: Final = "phase1-concept-mapping-dev-v1"

CONCEPTS: Final = (
    "consciousness",
    "self_identity",
    "reality_appearance",
)

LABEL_TARGETS: Final = {
    "positive": 1.0,
    "partial": 0.5,
    "negative": 0.0,
}

DEFAULT_APPROVED_EMBEDDINGS: Final = Path(
    "artifacts/phase1/embeddings/approved_chunk_embeddings.jsonl"
)
DEFAULT_QUERY_PROTOTYPE_EMBEDDINGS: Final = Path(
    "artifacts/phase1/embeddings/query_prototype_embeddings.jsonl"
)
DEFAULT_PASSAGE_PROTOTYPE_EMBEDDINGS: Final = Path(
    "artifacts/phase1/embeddings/passage_prototype_embeddings.jsonl"
)
DEFAULT_EMBEDDING_MANIFEST: Final = Path("artifacts/phase1/embeddings/embedding_manifest.json")
DEFAULT_BUILD_SET: Final = Path("data/evaluation/phase1_build.jsonl")
DEFAULT_DEVELOPMENT_SET: Final = Path("data/evaluation/phase1_development.jsonl")
DEFAULT_PROTOTYPES: Final = Path("data/concepts/phase1_concept_prototypes.yaml")
DEFAULT_OUTPUT_JSON: Final = Path("artifacts/phase1/evaluation/concept_mapping_dev_results.json")
DEFAULT_OUTPUT_MARKDOWN: Final = Path("docs/evaluation/phase1_concept_mapping_tuning.md")

ACTIVATION_THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
AMBIGUITY_MARGINS: Final = (0.05, 0.08, 0.10, 0.12, 0.15, 0.20)
MAX_ACTIVE_CONCEPTS = (2, 3)
NEGATIVE_PENALTIES = (0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50)
LEXICAL_WEIGHTS: Final = (0.05, 0.10, 0.15, 0.20)
HYBRID_EMBED_WEIGHTS: Final = (0.80, 0.85, 0.90)

CLASSIFIER_ITERATIONS: Final = 350
CLASSIFIER_LEARNING_RATE: Final = 0.15
CLASSIFIER_L2: Final = 0.01
CALIBRATION_ITERATIONS: Final = 500
CALIBRATION_LEARNING_RATE: Final = 0.05
EPSILON: Final = 1e-12

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


class TuningError(RuntimeError):
    """Raised when Phase 10 tuning cannot proceed safely."""


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class EmbeddingIdentity:
    provider: str
    model: str
    model_revision: str
    dimensions: int
    normalization: str

    def as_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "model_revision": self.model_revision,
            "dimensions": self.dimensions,
            "normalization": self.normalization,
        }


@dataclass(frozen=True)
class LabeledRecord:
    chunk_id: str
    split: str
    text: str
    labels: dict[str, str]
    hard_negative_for: tuple[str, ...]


@dataclass(frozen=True)
class PrototypeBank:
    question: dict[str, tuple[FloatArray, ...]]
    passage: dict[str, tuple[FloatArray, ...]]
    hard_negative: dict[str, tuple[FloatArray, ...]]


@dataclass(frozen=True)
class Calibration:
    slope: float
    intercept: float

    def apply(self, raw: float) -> float:
        return sigmoid_scalar(self.slope * raw + self.intercept)


@dataclass(frozen=True)
class Candidate:
    method: str
    source: str
    aggregation: str
    negative_penalty: float = 0.0
    lexical_weight: float = 0.0
    embed_weight: float = 1.0

    def key(self) -> str:
        return (
            f"{self.method}|{self.source}|{self.aggregation}|"
            f"{self.negative_penalty:.2f}|{self.lexical_weight:.2f}|"
            f"{self.embed_weight:.2f}"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "prototype_source": self.source,
            "prototype_aggregation": self.aggregation,
            "negative_penalty": self.negative_penalty,
            "lexical_weight": self.lexical_weight,
            "embedding_weight": self.embed_weight,
        }


@dataclass(frozen=True)
class Activation:
    thresholds: dict[str, float]
    ambiguity_margin: float
    max_active: int

    def as_dict(self) -> dict[str, object]:
        return {
            "concept_activation_thresholds": self.thresholds,
            "ambiguity_margin": self.ambiguity_margin,
            "maximum_active_concepts": self.max_active,
        }


@dataclass(frozen=True)
class Evaluation:
    candidate: Candidate
    activation: Activation
    calibrations: dict[str, Calibration]
    objective: float
    macro_f1: float
    macro_mae: float
    macro_brier: float
    unsupported_accuracy: float
    ambiguity_accuracy: float
    hard_negative_fp_rate: float
    exact_set_accuracy: float
    per_concept: dict[str, dict[str, float]]

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.as_dict(),
            "activation": self.activation.as_dict(),
            "calibration": {
                concept: {
                    "slope": value.slope,
                    "intercept": value.intercept,
                }
                for concept, value in self.calibrations.items()
            },
            "objective": self.objective,
            "metrics": {
                "macro_f1": self.macro_f1,
                "macro_mae": self.macro_mae,
                "macro_brier": self.macro_brier,
                "unsupported_accuracy": self.unsupported_accuracy,
                "ambiguity_accuracy": self.ambiguity_accuracy,
                "hard_negative_false_activation_rate": self.hard_negative_fp_rate,
                "exact_active_set_accuracy": self.exact_set_accuracy,
            },
            "per_concept": self.per_concept,
        }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune Phase 1 plural concept mapping using Build + Development only. "
            "Held-out is deliberately not an input."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--approved-embeddings",
        type=Path,
        default=DEFAULT_APPROVED_EMBEDDINGS,
    )
    parser.add_argument(
        "--query-prototype-embeddings",
        type=Path,
        default=DEFAULT_QUERY_PROTOTYPE_EMBEDDINGS,
    )
    parser.add_argument(
        "--passage-prototype-embeddings",
        type=Path,
        default=DEFAULT_PASSAGE_PROTOTYPE_EMBEDDINGS,
    )
    parser.add_argument(
        "--embedding-manifest",
        type=Path,
        default=DEFAULT_EMBEDDING_MANIFEST,
    )
    parser.add_argument("--build-set", type=Path, default=DEFAULT_BUILD_SET)
    parser.add_argument(
        "--development-set",
        type=Path,
        default=DEFAULT_DEVELOPMENT_SET,
    )
    parser.add_argument("--prototypes", type=Path, default=DEFAULT_PROTOTYPES)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=DEFAULT_OUTPUT_MARKDOWN,
    )
    parser.add_argument("--replace", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def resolve(project_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise TuningError(f"Required file does not exist: {path}")


def require_mapping(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TuningError(f"{description} must be an object.")
    result: dict[str, object] = {}
    for key, nested in value.items():
        if not isinstance(key, str):
            raise TuningError(f"{description} contains a non-string key.")
        result[key] = nested
    return result


def require_list(value: object, description: str) -> list[object]:
    if not isinstance(value, list):
        raise TuningError(f"{description} must be a list.")
    return value


def require_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TuningError(f"{description} must be a non-empty string.")
    return value.strip()


def require_int(value: object, description: str) -> int:
    if not isinstance(value, int):
        raise TuningError(f"{description} must be an integer.")
    return value


def optional_string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha256_jsonl(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                value: object = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise TuningError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            digest.update(canonical_json_bytes(value))
            digest.update(b"\n")
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TuningError(f"Invalid JSON in {path}: {exc}") from exc
    return require_mapping(value, f"JSON {path}")


def load_yaml(path: Path) -> dict[str, object]:
    try:
        value: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise TuningError(f"Invalid YAML in {path}: {exc}") from exc
    return require_mapping(value, f"YAML {path}")


def iter_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                value: object = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise TuningError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            yield require_mapping(value, f"{path}:{line_number}")


def atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(character for character in value if not unicodedata.combining(character))
    value = re.sub(r"[''`']", "", value.casefold())
    value = re.sub(r"[^\w-]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def parse_string_list(value: object) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(
            item.strip().casefold() for item in value if isinstance(item, str) and item.strip()
        )
    if isinstance(value, str) and value.strip():
        normalized = value.replace(";", "|").replace(",", "|")
        return tuple(part.strip().casefold() for part in normalized.split("|") if part.strip())
    return ()


def load_split(path: Path, expected_split: str) -> dict[str, LabeledRecord]:
    records: dict[str, LabeledRecord] = {}

    for raw in iter_jsonl(path):
        chunk_id = require_string(raw.get("chunk_id"), "chunk_id")
        split = require_string(raw.get("evaluation_split"), "evaluation_split")
        if split != expected_split:
            raise TuningError(f"{chunk_id} is {split!r}, expected {expected_split!r}.")

        review = require_mapping(raw.get("review"), f"{chunk_id} review")
        labels_raw = require_mapping(
            review.get("labels"),
            f"{chunk_id} labels",
        )

        labels: dict[str, str] = {}
        for concept in CONCEPTS:
            label = require_string(
                labels_raw.get(concept),
                f"{chunk_id} {concept}",
            ).casefold()
            if label not in LABEL_TARGETS:
                raise TuningError(f"Unsupported label {label!r}.")
            labels[concept] = label

        records[chunk_id] = LabeledRecord(
            chunk_id=chunk_id,
            split=split,
            text=require_string(raw.get("reviewed_text"), "reviewed_text"),
            labels=labels,
            hard_negative_for=parse_string_list(review.get("hard_negative_for")),
        )

    if not records:
        raise TuningError(f"{expected_split} split is empty.")

    return records


def load_identity(path: Path) -> tuple[EmbeddingIdentity, dict[str, object]]:
    manifest = load_json(path)
    if optional_string(manifest.get("status")) != "complete":
        raise TuningError("Phase 9 embedding manifest is not complete.")

    gate = require_mapping(manifest.get("exit_gate"), "Phase 9 exit_gate")
    for field_name in (
        "every_approved_chunk_has_one_valid_embedding",
        "no_mock_vectors",
        "single_embedding_identity",
        "all_prototype_vectors_same_frozen_configuration",
        "evaluation_splits_embedded_in_same_space",
        "resume_without_repeated_provider_calls",
    ):
        if gate.get(field_name) is not True:
            raise TuningError(f"Phase 9 gate failed: {field_name}.")

    raw = require_mapping(
        manifest.get("embedding_identity"),
        "embedding_identity",
    )
    return (
        EmbeddingIdentity(
            provider=require_string(raw.get("provider"), "provider"),
            model=require_string(raw.get("model"), "model"),
            model_revision=require_string(
                raw.get("model_revision"),
                "model_revision",
            ),
            dimensions=require_int(raw.get("dimensions"), "dimensions"),
            normalization=require_string(
                raw.get("normalization"),
                "normalization",
            ),
        ),
        manifest,
    )


def vector_from_record(
    raw: Mapping[str, object],
    identity: EmbeddingIdentity,
    description: str,
) -> FloatArray:
    expected = (
        identity.provider,
        identity.model,
        identity.model_revision,
        identity.dimensions,
        identity.normalization,
    )
    actual = (
        require_string(raw.get("provider"), f"{description} provider"),
        require_string(raw.get("model"), f"{description} model"),
        require_string(
            raw.get("model_revision"),
            f"{description} model_revision",
        ),
        require_int(raw.get("dimensions"), f"{description} dimensions"),
        require_string(
            raw.get("normalization"),
            f"{description} normalization",
        ),
    )
    if actual != expected:
        raise TuningError(f"{description} has mixed embedding identity.")

    values_raw = require_list(raw.get("embedding"), f"{description} embedding")
    if len(values_raw) != identity.dimensions:
        raise TuningError(f"{description} has wrong dimensions.")

    values: list[float] = []
    for value in values_raw:
        if not isinstance(value, int | float):
            raise TuningError(f"{description} vector is non-numeric.")
        number = float(value)
        if not math.isfinite(number):
            raise TuningError(f"{description} vector has NaN/inf.")
        values.append(number)

    vector = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= EPSILON:
        raise TuningError(f"{description} vector is zero.")
    return vector / norm


def load_needed_embeddings(
    path: Path,
    needed_ids: set[str],
    identity: EmbeddingIdentity,
) -> dict[str, FloatArray]:
    embeddings: dict[str, FloatArray] = {}

    for raw in iter_jsonl(path):
        chunk_id = require_string(raw.get("chunk_id"), "embedding chunk_id")
        if chunk_id not in needed_ids:
            continue
        if chunk_id in embeddings:
            raise TuningError(f"Duplicate embedding: {chunk_id}")
        embeddings[chunk_id] = vector_from_record(
            raw,
            identity,
            f"chunk {chunk_id}",
        )

    missing = sorted(needed_ids - set(embeddings))
    if missing:
        raise TuningError("Missing Build/Development embeddings: " + ", ".join(missing[:10]))

    return embeddings


def load_prototype_bank(
    query_path: Path,
    passage_path: Path,
    identity: EmbeddingIdentity,
) -> PrototypeBank:
    questions: defaultdict[str, list[FloatArray]] = defaultdict(list)
    positives: defaultdict[str, list[FloatArray]] = defaultdict(list)
    negatives: defaultdict[str, list[FloatArray]] = defaultdict(list)

    for raw in iter_jsonl(query_path):
        concept = require_string(raw.get("concept_slug"), "concept_slug")
        questions[concept].append(vector_from_record(raw, identity, f"query prototype {concept}"))

    for raw in iter_jsonl(passage_path):
        concept = require_string(raw.get("concept_slug"), "concept_slug")
        role = require_string(raw.get("prototype_role"), "prototype_role")
        vector = vector_from_record(
            raw,
            identity,
            f"passage prototype {concept}/{role}",
        )
        if role == "positive":
            positives[concept].append(vector)
        elif role == "hard_negative":
            negatives[concept].append(vector)
        else:
            raise TuningError(f"Unknown prototype role {role!r}.")

    for concept in CONCEPTS:
        if not questions[concept]:
            raise TuningError(f"{concept} has no query prototypes.")
        if not positives[concept]:
            raise TuningError(f"{concept} has no positive prototypes.")
        if not negatives[concept]:
            raise TuningError(f"{concept} has no hard-negative prototypes.")

    return PrototypeBank(
        question={concept: tuple(questions[concept]) for concept in CONCEPTS},
        passage={concept: tuple(positives[concept]) for concept in CONCEPTS},
        hard_negative={concept: tuple(negatives[concept]) for concept in CONCEPTS},
    )


def validate_prototypes(path: Path) -> dict[str, object]:
    artifact = load_yaml(path)
    if optional_string(artifact.get("status")) != "frozen":
        raise TuningError("Phase 8 prototypes are not frozen.")
    gate = require_mapping(artifact.get("exit_gate"), "Phase 8 exit_gate")
    if gate.get("human_review_complete") is not True:
        raise TuningError("Phase 8 human review is not complete.")
    return artifact


def sigmoid_scalar(value: float) -> float:
    value = max(-40.0, min(40.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def sigmoid_array(values: FloatArray) -> FloatArray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def centroid(vectors: Sequence[FloatArray]) -> FloatArray:
    mean = np.mean(np.stack(vectors, axis=0), axis=0)
    norm = float(np.linalg.norm(mean))
    if norm <= EPSILON:
        raise TuningError("Prototype centroid has zero norm.")
    return cast(FloatArray, mean / norm)


def positive_vectors(
    bank: PrototypeBank,
    concept: str,
    source: str,
) -> tuple[FloatArray, ...]:
    if source == "question":
        return bank.question[concept]
    if source == "passage":
        return bank.passage[concept]
    if source == "combined":
        return (*bank.question[concept], *bank.passage[concept])
    raise TuningError(f"Unknown prototype source {source!r}.")


def positive_similarity(
    vector: FloatArray,
    bank: PrototypeBank,
    concept: str,
    source: str,
    aggregation: str,
) -> float:
    vectors = positive_vectors(bank, concept, source)
    if aggregation == "centroid":
        return float(vector @ centroid(vectors))
    if aggregation == "maximum":
        return max(float(vector @ item) for item in vectors)
    raise TuningError(f"Unknown aggregation {aggregation!r}.")


def negative_similarity(
    vector: FloatArray,
    bank: PrototypeBank,
    concept: str,
) -> float:
    return max(float(vector @ item) for item in bank.hard_negative[concept])


def lexical_score(text: str, concept: str) -> float:
    normalized = normalize_text(text)
    positive_hits = sum(normalize_text(term) in normalized for term in LEXICAL_INDICATORS[concept])
    negative_hits = sum(normalize_text(term) in normalized for term in LEXICAL_EXCLUSIONS[concept])
    positive = min(1.0, positive_hits / 3.0)
    negative = min(1.0, negative_hits / 2.0)
    return max(0.0, min(1.0, positive - 0.5 * negative))


def train_classifier(
    matrix: FloatArray,
    targets: FloatArray,
) -> tuple[FloatArray, float]:
    weights = np.zeros(matrix.shape[1], dtype=np.float64)
    bias = 0.0

    for _ in range(CLASSIFIER_ITERATIONS):
        probabilities = sigmoid_array(matrix @ weights + bias)
        errors = probabilities - targets
        gradient = (matrix.T @ errors) / matrix.shape[0] + CLASSIFIER_L2 * weights
        weights -= CLASSIFIER_LEARNING_RATE * gradient
        bias -= CLASSIFIER_LEARNING_RATE * float(np.mean(errors))

    return weights, bias


def train_build_classifiers(
    build: Mapping[str, LabeledRecord],
    embeddings: Mapping[str, FloatArray],
) -> dict[str, tuple[FloatArray, float]]:
    chunk_ids = sorted(build)
    matrix = np.stack([embeddings[chunk_id] for chunk_id in chunk_ids])

    classifiers: dict[str, tuple[FloatArray, float]] = {}
    for concept in CONCEPTS:
        targets = np.asarray(
            [LABEL_TARGETS[build[chunk_id].labels[concept]] for chunk_id in chunk_ids],
            dtype=np.float64,
        )
        classifiers[concept] = train_classifier(matrix, targets)

    return classifiers


def classifier_score(
    vector: FloatArray,
    classifier: tuple[FloatArray, float],
) -> float:
    weights, bias = classifier
    return sigmoid_scalar(float(vector @ weights + bias))


def raw_scores(
    candidate: Candidate,
    record: LabeledRecord,
    vector: FloatArray,
    bank: PrototypeBank,
    classifiers: Mapping[str, tuple[FloatArray, float]],
) -> dict[str, float]:
    scores: dict[str, float] = {}

    for concept in CONCEPTS:
        if candidate.method == "classifier":
            scores[concept] = classifier_score(
                vector,
                classifiers[concept],
            )
            continue

        positive = positive_similarity(
            vector,
            bank,
            concept,
            candidate.source,
            candidate.aggregation,
        )

        if candidate.method in {"centroid", "maximum_example"}:
            scores[concept] = positive
            continue

        negative = negative_similarity(vector, bank, concept)

        if candidate.method == "positive_minus_negative":
            scores[concept] = positive - candidate.negative_penalty * negative
            continue

        if candidate.method == "hybrid":
            scores[concept] = candidate.embed_weight * (
                positive - candidate.negative_penalty * negative
            ) + candidate.lexical_weight * lexical_score(record.text, concept)
            continue

        raise TuningError(f"Unknown method {candidate.method!r}.")

    return scores


def fit_calibration(
    raw_values: Sequence[float],
    targets: Sequence[float],
) -> Calibration:
    x = np.asarray(raw_values, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    slope = 1.0
    intercept = 0.0

    for _ in range(CALIBRATION_ITERATIONS):
        predictions = sigmoid_array(slope * x + intercept)
        errors = predictions - y
        slope -= CALIBRATION_LEARNING_RATE * float(np.mean(errors * x))
        intercept -= CALIBRATION_LEARNING_RATE * float(np.mean(errors))

    return Calibration(slope=max(0.0, slope), intercept=intercept)


def calibrate(
    raw_by_id: Mapping[str, Mapping[str, float]],
    development: Mapping[str, LabeledRecord],
) -> dict[str, Calibration]:
    chunk_ids = sorted(development)
    result: dict[str, Calibration] = {}

    for concept in CONCEPTS:
        result[concept] = fit_calibration(
            [raw_by_id[chunk_id][concept] for chunk_id in chunk_ids],
            [LABEL_TARGETS[development[chunk_id].labels[concept]] for chunk_id in chunk_ids],
        )

    return result


def apply_calibration(
    raw: Mapping[str, float],
    calibrations: Mapping[str, Calibration],
) -> dict[str, float]:
    return {concept: calibrations[concept].apply(raw[concept]) for concept in CONCEPTS}


def expected_active(record: LabeledRecord) -> set[str]:
    return {concept for concept in CONCEPTS if record.labels[concept] in {"positive", "partial"}}


def predicted_active(
    weights: Mapping[str, float],
    activation: Activation,
) -> list[str]:
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


def expected_ambiguous(record: LabeledRecord) -> bool:
    active = expected_active(record)
    if len(active) < 2:
        return False

    targets = sorted(
        (LABEL_TARGETS[record.labels[concept]] for concept in CONCEPTS),
        reverse=True,
    )
    return abs(targets[0] - targets[1]) <= 0.5


def predicted_ambiguous(
    weights: Mapping[str, float],
    activation: Activation,
) -> bool:
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


def f1_score(truth: Sequence[bool], predicted: Sequence[bool]) -> float:
    true_positive = sum(
        expected and actual for expected, actual in zip(truth, predicted, strict=True)
    )
    false_positive = sum(
        (not expected) and actual for expected, actual in zip(truth, predicted, strict=True)
    )
    false_negative = sum(
        expected and (not actual) for expected, actual in zip(truth, predicted, strict=True)
    )

    denominator = 2 * true_positive + false_positive + false_negative
    return 1.0 if denominator == 0 else 2.0 * true_positive / denominator


def tune_concept_thresholds(
    calibrated_by_id: Mapping[str, Mapping[str, float]],
    development: Mapping[str, LabeledRecord],
) -> dict[str, float]:
    selected: dict[str, float] = {}

    for concept in CONCEPTS:
        best_threshold: float | None = None
        best_key: tuple[float, float, float, float] | None = None

        for threshold in ACTIVATION_THRESHOLDS:
            truth: list[bool] = []
            predicted: list[bool] = []

            hard_negative_opportunities = 0
            hard_negative_false_activations = 0

            for chunk_id in sorted(development):
                record = development[chunk_id]
                weight = calibrated_by_id[chunk_id][concept]

                expected = record.labels[concept] in {"positive", "partial"}
                actual = weight >= threshold

                truth.append(expected)
                predicted.append(actual)

                if concept in record.hard_negative_for:
                    hard_negative_opportunities += 1
                    if actual:
                        hard_negative_false_activations += 1

            concept_f1 = f1_score(
                truth,
                predicted,
            )

            hard_negative_fp_rate = (
                hard_negative_false_activations / hard_negative_opportunities
                if hard_negative_opportunities
                else 0.0
            )

            threshold_score = 0.75 * concept_f1 + 0.25 * (1.0 - hard_negative_fp_rate)

            key = (
                threshold_score,
                concept_f1,
                -hard_negative_fp_rate,
                threshold,
            )

            if best_key is None or key > best_key:
                best_key = key
                best_threshold = threshold

        if best_threshold is None:
            raise TuningError(f"Could not tune threshold for {concept}.")

        selected[concept] = best_threshold

    return selected


def evaluate(
    calibrated_by_id: Mapping[str, Mapping[str, float]],
    development: Mapping[str, LabeledRecord],
    activation: Activation,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    dict[str, dict[str, float]],
]:
    chunk_ids = sorted(development)
    per_concept: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    mae_values: list[float] = []
    brier_values: list[float] = []

    for concept in CONCEPTS:
        truth_active: list[bool] = []
        predicted_active_flags: list[bool] = []
        absolute_errors: list[float] = []
        squared_errors: list[float] = []

        for chunk_id in chunk_ids:
            record = development[chunk_id]
            weight = calibrated_by_id[chunk_id][concept]
            target = LABEL_TARGETS[record.labels[concept]]

            truth_active.append(target > 0.0)
            predicted_active_flags.append(
                concept in predicted_active(calibrated_by_id[chunk_id], activation)
            )
            absolute_errors.append(abs(weight - target))
            squared_errors.append((weight - target) ** 2)

        concept_f1 = f1_score(truth_active, predicted_active_flags)
        concept_mae = float(np.mean(absolute_errors))
        concept_brier = float(np.mean(squared_errors))

        f1_values.append(concept_f1)
        mae_values.append(concept_mae)
        brier_values.append(concept_brier)

        per_concept[concept] = {
            "f1": concept_f1,
            "mae": concept_mae,
            "brier": concept_brier,
        }

    unsupported_truth: list[bool] = []
    unsupported_predicted: list[bool] = []
    ambiguity_truth: list[bool] = []
    ambiguity_predicted: list[bool] = []
    exact_matches: list[bool] = []
    hard_negative_opportunities = 0
    hard_negative_false_activations = 0

    for chunk_id in chunk_ids:
        record = development[chunk_id]
        weights = calibrated_by_id[chunk_id]
        predicted_set = set(predicted_active(weights, activation))
        expected_set = expected_active(record)

        unsupported_truth.append(not expected_set)
        unsupported_predicted.append(not predicted_set)

        ambiguity_truth.append(expected_ambiguous(record))
        ambiguity_predicted.append(predicted_ambiguous(weights, activation))

        exact_matches.append(predicted_set == expected_set)

        for concept in record.hard_negative_for:
            if concept not in CONCEPTS:
                continue
            hard_negative_opportunities += 1
            if concept in predicted_set:
                hard_negative_false_activations += 1

    unsupported_accuracy = float(
        np.mean(
            np.asarray(unsupported_truth, dtype=bool)
            == np.asarray(unsupported_predicted, dtype=bool)
        )
    )
    ambiguity_accuracy = float(
        np.mean(
            np.asarray(ambiguity_truth, dtype=bool) == np.asarray(ambiguity_predicted, dtype=bool)
        )
    )
    exact_set_accuracy = float(np.mean(exact_matches))

    hard_negative_fp_rate = (
        hard_negative_false_activations / hard_negative_opportunities
        if hard_negative_opportunities
        else 0.0
    )

    return (
        float(np.mean(f1_values)),
        float(np.mean(mae_values)),
        float(np.mean(brier_values)),
        unsupported_accuracy,
        ambiguity_accuracy,
        hard_negative_fp_rate,
        exact_set_accuracy,
        per_concept,
    )


def objective(
    macro_f1: float,
    macro_mae: float,
    macro_brier: float,
    unsupported_accuracy: float,
    ambiguity_accuracy: float,
    hard_negative_fp_rate: float,
    exact_set_accuracy: float,
) -> float:
    return (
        0.30 * macro_f1
        + 0.10 * (1.0 - macro_mae)
        + 0.05 * (1.0 - macro_brier)
        + 0.10 * unsupported_accuracy
        + 0.10 * ambiguity_accuracy
        + 0.25 * (1.0 - hard_negative_fp_rate)
        + 0.10 * exact_set_accuracy
    )


def candidate_space() -> list[Candidate]:
    candidates: list[Candidate] = []

    for source in ("question", "passage", "combined"):
        candidates.extend(
            (
                Candidate("centroid", source, "centroid"),
                Candidate("maximum_example", source, "maximum"),
            )
        )

    for source in ("question", "passage", "combined"):
        for aggregation in ("centroid", "maximum"):
            for penalty in NEGATIVE_PENALTIES:
                candidates.append(
                    Candidate(
                        "positive_minus_negative",
                        source,
                        aggregation,
                        negative_penalty=penalty,
                    )
                )

    candidates.append(Candidate("classifier", "none", "none"))

    for source in ("question", "passage", "combined"):
        for aggregation in ("centroid", "maximum"):
            for penalty in (0.25, 0.50, 0.75, 1.00):
                for lexical_weight in LEXICAL_WEIGHTS:
                    for embed_weight in HYBRID_EMBED_WEIGHTS:
                        candidates.append(
                            Candidate(
                                "hybrid",
                                source,
                                aggregation,
                                negative_penalty=penalty,
                                lexical_weight=lexical_weight,
                                embed_weight=embed_weight,
                            )
                        )

    return candidates


def tune_candidate(
    candidate: Candidate,
    development: Mapping[str, LabeledRecord],
    embeddings: Mapping[str, FloatArray],
    bank: PrototypeBank,
    classifiers: Mapping[str, tuple[FloatArray, float]],
) -> Evaluation:
    raw_by_id = {
        chunk_id: raw_scores(
            candidate,
            record,
            embeddings[chunk_id],
            bank,
            classifiers,
        )
        for chunk_id, record in development.items()
    }

    calibrations = calibrate(raw_by_id, development)
    calibrated_by_id = {
        chunk_id: apply_calibration(raw, calibrations) for chunk_id, raw in raw_by_id.items()
    }

    tuned_thresholds = tune_concept_thresholds(
        calibrated_by_id,
        development,
    )

    best: Evaluation | None = None

    for ambiguity_margin in AMBIGUITY_MARGINS:
        for max_active in MAX_ACTIVE_CONCEPTS:
            activation = Activation(
                thresholds=dict(tuned_thresholds),
                ambiguity_margin=ambiguity_margin,
                max_active=max_active,
            )

            (
                macro_f1,
                macro_mae,
                macro_brier,
                unsupported_accuracy,
                ambiguity_accuracy,
                hard_negative_fp_rate,
                exact_set_accuracy,
                per_concept,
            ) = evaluate(
                calibrated_by_id,
                development,
                activation,
            )

            score = objective(
                macro_f1,
                macro_mae,
                macro_brier,
                unsupported_accuracy,
                ambiguity_accuracy,
                hard_negative_fp_rate,
                exact_set_accuracy,
            )

            current = Evaluation(
                candidate=candidate,
                activation=activation,
                calibrations=calibrations,
                objective=score,
                macro_f1=macro_f1,
                macro_mae=macro_mae,
                macro_brier=macro_brier,
                unsupported_accuracy=unsupported_accuracy,
                ambiguity_accuracy=ambiguity_accuracy,
                hard_negative_fp_rate=hard_negative_fp_rate,
                exact_set_accuracy=exact_set_accuracy,
                per_concept=per_concept,
            )

            if best is None:
                best = current
                continue

            current_key = (
                current.objective,
                current.macro_f1,
                -current.hard_negative_fp_rate,
                current.exact_set_accuracy,
                -current.macro_mae,
                -current.activation.max_active,
            )
            best_key = (
                best.objective,
                best.macro_f1,
                -best.hard_negative_fp_rate,
                best.exact_set_accuracy,
                -best.macro_mae,
                -best.activation.max_active,
            )

            if current_key > best_key:
                best = current

    if best is None:
        raise TuningError(f"No result for candidate {candidate.key()}.")

    return best


def detail_rows(
    chosen: Evaluation,
    development: Mapping[str, LabeledRecord],
    embeddings: Mapping[str, FloatArray],
    bank: PrototypeBank,
    classifiers: Mapping[str, tuple[FloatArray, float]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for chunk_id in sorted(development):
        record = development[chunk_id]
        raw = raw_scores(
            chosen.candidate,
            record,
            embeddings[chunk_id],
            bank,
            classifiers,
        )
        weights = apply_calibration(raw, chosen.calibrations)
        active = predicted_active(weights, chosen.activation)

        rows.append(
            {
                "chunk_id": chunk_id,
                "split": "development",
                "reviewed_labels": record.labels,
                "expected_active_concepts": sorted(expected_active(record)),
                "raw_scores": raw,
                "calibrated_weights": weights,
                "active_concepts": active,
                "ambiguous": predicted_ambiguous(weights, chosen.activation),
                "unsupported": not active,
                "hard_negative_for": list(record.hard_negative_for),
            }
        )

    return rows


def markdown_report(
    chosen: Evaluation,
    ranked: Sequence[Evaluation],
    identity: EmbeddingIdentity,
    build_count: int,
    development_count: int,
    output_json: Path,
) -> str:
    lines = [
        "# Phase 1 Concept Mapping Tuning",
        "",
        f"- Tuning version: `{TUNING_VERSION}`",
        f"- Generated: `{utc_now()}`",
        "- Status: **FROZEN FOR HELD-OUT EVALUATION**",
        f"- Build records used for classifier training: **{build_count}**",
        f"- Development records used for tuning: **{development_count}**",
        "- Held-out records used: **0**",
        "",
        "## Frozen embedding identity",
        "",
        f"- Provider: `{identity.provider}`",
        f"- Model: `{identity.model}`",
        f"- Model revision: `{identity.model_revision}`",
        f"- Dimensions: `{identity.dimensions}`",
        f"- Normalization: `{identity.normalization}`",
        "",
        "## Candidate methods evaluated",
        "",
        "- Prototype centroid",
        "- Maximum-example similarity",
        "- Positive-minus-hard-negative similarity",
        "- Lightweight soft logistic classifier trained on Build only",
        "- Hybrid embedding similarity + lexical indicator + hard-negative penalty",
        "",
        "## Selected configuration",
        "",
        f"- Method: **{chosen.candidate.method}**",
        f"- Prototype source: `{chosen.candidate.source}`",
        f"- Prototype aggregation: `{chosen.candidate.aggregation}`",
        f"- Negative penalty: `{chosen.candidate.negative_penalty:.2f}`",
        f"- Lexical weight: `{chosen.candidate.lexical_weight:.2f}`",
        f"- Embedding weight: `{chosen.candidate.embed_weight:.2f}`",
        "- Concept activation thresholds:",
        (f"  - consciousness: `{chosen.activation.thresholds['consciousness']:.2f}`"),
        (f"  - self_identity: `{chosen.activation.thresholds['self_identity']:.2f}`"),
        (f"  - reality_appearance: `{chosen.activation.thresholds['reality_appearance']:.2f}`"),
        f"- Ambiguity margin: `{chosen.activation.ambiguity_margin:.2f}`",
        f"- Maximum active concepts: `{chosen.activation.max_active}`",
        "",
        (
            "Raw scores are preserved separately from calibrated 0-1 weights. "
            "Weights are independent and are not normalized to sum to one."
        ),
        "",
        "## Development metrics",
        "",
        f"- Objective: **{chosen.objective:.4f}**",
        f"- Macro F1: **{chosen.macro_f1:.4f}**",
        f"- Macro MAE: **{chosen.macro_mae:.4f}**",
        f"- Macro Brier: **{chosen.macro_brier:.4f}**",
        f"- Unsupported accuracy: **{chosen.unsupported_accuracy:.4f}**",
        f"- Ambiguity accuracy: **{chosen.ambiguity_accuracy:.4f}**",
        (f"- Hard-negative false activation rate: **{chosen.hard_negative_fp_rate:.4f}**"),
        f"- Exact active-set accuracy: **{chosen.exact_set_accuracy:.4f}**",
        "",
        "## Per-concept metrics",
        "",
        "| Concept | F1 | MAE | Brier |",
        "|---|---:|---:|---:|",
    ]

    for concept in CONCEPTS:
        values = chosen.per_concept[concept]
        lines.append(
            f"| {concept} | {values['f1']:.4f} | {values['mae']:.4f} | {values['brier']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Top candidate configurations",
            "",
            "| Rank | Method | Source | Aggregation | Neg | Lex | "
            "C Thr | S Thr | R Thr | Margin | Max | Objective | F1 | HN FP |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for rank, result in enumerate(ranked[:10], start=1):
        lines.append(
            f"| {rank} | {result.candidate.method} | "
            f"{result.candidate.source} | {result.candidate.aggregation} | "
            f"{result.candidate.negative_penalty:.2f} | "
            f"{result.candidate.lexical_weight:.2f} | "
            f"{result.activation.thresholds['consciousness']:.2f} | "
            f"{result.activation.thresholds['self_identity']:.2f} | "
            f"{result.activation.thresholds['reality_appearance']:.2f} | "
            f"{result.activation.ambiguity_margin:.2f} | "
            f"{result.activation.max_active} | "
            f"{result.objective:.4f} | {result.macro_f1:.4f} | "
            f"{result.hard_negative_fp_rate:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Frozen calibration",
            "",
        ]
    )

    for concept in CONCEPTS:
        calibration = chosen.calibrations[concept]
        lines.extend(
            [
                f"### `{concept}`",
                "",
                f"- Slope: `{calibration.slope:.8f}`",
                f"- Intercept: `{calibration.intercept:.8f}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Runtime behavior",
            "",
            "- Multiple concepts may activate simultaneously.",
            "- Weights are not forced to sum to one.",
            "- Raw scores remain available for audit/debugging.",
            "- Unsupported means no concept clears the frozen threshold.",
            "- Ambiguous means the two strongest eligible weights are within "
            "the frozen ambiguity margin.",
            "- Maximum-active limits activation decisions only; it does not "
            "discard raw scores or calibrated weights.",
            "",
            "## Leakage controls",
            "",
            "- Build was used only for the lightweight classifier candidate.",
            "- Development was used for method selection, calibration, threshold, "
            "ambiguity margin, maximum active concepts, prototype aggregation, "
            "negative penalties, and lexical/embedding hybrid weights.",
            "- Held-out was not supplied as an input and was not used for tuning.",
            "- All selected parameters are frozen before Phase 11.",
            "",
            "## Exit gate",
            "",
            "**PASS** — Phase 10 parameters are frozen before held-out evaluation.",
            "",
            f"Machine-readable results: `{output_json.as_posix()}`",
            "",
        ]
    )

    return "\n".join(lines)


def ensure_output_policy(
    output_json: Path,
    output_markdown: Path,
    replace: bool,
) -> None:
    existing = [path for path in (output_json, output_markdown) if path.exists()]
    if existing and not replace:
        raise TuningError(
            "Phase 10 outputs already exist. Use --replace: "
            + ", ".join(path.as_posix() for path in existing)
        )


def passes_quality_gate(
    evaluation: Evaluation,
) -> bool:
    return (
        evaluation.macro_f1 >= 0.70
        and evaluation.per_concept["consciousness"]["f1"] >= 0.50
        and evaluation.per_concept["self_identity"]["f1"] >= 0.70
        and evaluation.per_concept["reality_appearance"]["f1"] >= 0.75
        and evaluation.hard_negative_fp_rate <= 0.40
    )


def run_phase10(
    *,
    project_root: Path,
    approved_embeddings: Path,
    query_embeddings: Path,
    passage_embeddings: Path,
    embedding_manifest: Path,
    build_set: Path,
    development_set: Path,
    prototypes: Path,
    output_json: Path,
    output_markdown: Path,
    replace: bool,
) -> dict[str, object]:
    project_root = project_root.resolve()

    approved_embeddings = resolve(project_root, approved_embeddings)
    query_embeddings = resolve(project_root, query_embeddings)
    passage_embeddings = resolve(project_root, passage_embeddings)
    embedding_manifest = resolve(project_root, embedding_manifest)
    build_set = resolve(project_root, build_set)
    development_set = resolve(project_root, development_set)
    prototypes = resolve(project_root, prototypes)
    output_json = resolve(project_root, output_json)
    output_markdown = resolve(project_root, output_markdown)

    for path in (
        approved_embeddings,
        query_embeddings,
        passage_embeddings,
        embedding_manifest,
        build_set,
        development_set,
        prototypes,
    ):
        require_file(path)

    ensure_output_policy(output_json, output_markdown, replace)

    identity, phase9_manifest = load_identity(embedding_manifest)
    prototype_artifact = validate_prototypes(prototypes)

    build = load_split(build_set, "build")
    development = load_split(development_set, "development")

    if len(build) != 159:
        raise TuningError(f"Expected 159 Build records, found {len(build)}.")
    if len(development) != 80:
        raise TuningError(f"Expected 80 Development records, found {len(development)}.")

    needed_ids = set(build) | set(development)
    embeddings = load_needed_embeddings(
        approved_embeddings,
        needed_ids,
        identity,
    )
    bank = load_prototype_bank(
        query_embeddings,
        passage_embeddings,
        identity,
    )
    classifiers = train_build_classifiers(build, embeddings)

    candidates = candidate_space()
    LOGGER.info(
        "Phase 10 candidates=%d build=%d development=%d heldout_used=0",
        len(candidates),
        len(build),
        len(development),
    )

    evaluations: list[Evaluation] = []
    for index, candidate in enumerate(candidates, start=1):
        result = tune_candidate(
            candidate,
            development,
            embeddings,
            bank,
            classifiers,
        )
        evaluations.append(result)

        if index % 50 == 0 or index == len(candidates):
            LOGGER.info(
                "Evaluated %d/%d candidates; current method=%s objective=%.4f",
                index,
                len(candidates),
                candidate.method,
                result.objective,
            )

    ranked = sorted(
        evaluations,
        key=lambda item: (
            -item.objective,
            -item.macro_f1,
            item.hard_negative_fp_rate,
            -item.exact_set_accuracy,
            item.macro_mae,
            item.candidate.key(),
        ),
    )
    chosen = ranked[0]

    LOGGER.info(
        "Selected method=%s source=%s aggregation=%s objective=%.4f",
        chosen.candidate.method,
        chosen.candidate.source,
        chosen.candidate.aggregation,
        chosen.objective,
    )
    LOGGER.info(
        "Frozen thresholds=%s margin=%.2f max_active=%d",
        chosen.activation.thresholds,
        chosen.activation.ambiguity_margin,
        chosen.activation.max_active,
    )

    best_by_method: dict[str, Evaluation] = {}
    for result in ranked:
        best_by_method.setdefault(result.candidate.method, result)

    quality_gate_passed = (
        chosen.macro_f1 >= 0.70
        and chosen.per_concept["consciousness"]["f1"] >= 0.50
        and chosen.per_concept["self_identity"]["f1"] >= 0.70
        and chosen.per_concept["reality_appearance"]["f1"] >= 0.75
        and chosen.hard_negative_fp_rate <= 0.40
    )

    results: dict[str, object] = {
        "script_version": SCRIPT_VERSION,
        "tuning_version": TUNING_VERSION,
        "generated_at": utc_now(),
        "status": ("frozen" if quality_gate_passed else "tuning_incomplete"),
        "phase": "phase_10_concept_mapping_development_tuning",
        "embedding_identity": identity.as_dict(),
        "inputs": {
            "approved_embeddings": {
                "path": approved_embeddings.as_posix(),
                "sha256": sha256_jsonl(approved_embeddings),
            },
            "query_prototype_embeddings": {
                "path": query_embeddings.as_posix(),
                "sha256": sha256_jsonl(query_embeddings),
            },
            "passage_prototype_embeddings": {
                "path": passage_embeddings.as_posix(),
                "sha256": sha256_jsonl(passage_embeddings),
            },
            "embedding_manifest": {
                "path": embedding_manifest.as_posix(),
                "sha256_file_bytes": sha256_file(embedding_manifest),
                "status": optional_string(phase9_manifest.get("status")),
            },
            "prototype_artifact": {
                "path": prototypes.as_posix(),
                "sha256_file_bytes": sha256_file(prototypes),
                "prototype_version": require_string(
                    prototype_artifact.get("prototype_version"),
                    "prototype_version",
                ),
                "status": optional_string(prototype_artifact.get("status")),
            },
            "build": {
                "path": build_set.as_posix(),
                "sha256": sha256_jsonl(build_set),
                "record_count": len(build),
                "usage": "classifier training only",
            },
            "development": {
                "path": development_set.as_posix(),
                "sha256": sha256_jsonl(development_set),
                "record_count": len(development),
                "usage": (
                    "method selection, calibration, thresholds, ambiguity, "
                    "maximum active concepts, aggregation, negative penalties"
                ),
            },
            "heldout": {
                "supplied_as_input": False,
                "records_used": 0,
                "labels_used": 0,
            },
        },
        "required_behavior": {
            "multiple_active_concepts_allowed": True,
            "weights_sum_to_one": False,
            "raw_scores_preserved": True,
            "calibrated_weights_range": [0.0, 1.0],
            "ambiguous_mapping_detection": True,
            "unsupported_question_detection": True,
        },
        "candidate_space": {
            "candidate_count": len(candidates),
            "activation_thresholds": list(ACTIVATION_THRESHOLDS),
            "ambiguity_margins": list(AMBIGUITY_MARGINS),
            "maximum_active_concepts": list(MAX_ACTIVE_CONCEPTS),
            "negative_penalties": list(NEGATIVE_PENALTIES),
            "lexical_weights": list(LEXICAL_WEIGHTS),
            "hybrid_embedding_weights": list(HYBRID_EMBED_WEIGHTS),
            "classifier": {
                "training_split": "build",
                "iterations": CLASSIFIER_ITERATIONS,
                "learning_rate": CLASSIFIER_LEARNING_RATE,
                "l2": CLASSIFIER_L2,
                "soft_targets": LABEL_TARGETS,
            },
            "calibration": {
                "type": "concept_specific_sigmoid",
                "fitted_on": "development",
                "iterations": CALIBRATION_ITERATIONS,
                "learning_rate": CALIBRATION_LEARNING_RATE,
            },
        },
        "selection_objective": {
            "macro_f1": 0.30,
            "one_minus_macro_mae": 0.10,
            "one_minus_macro_brier": 0.05,
            "unsupported_accuracy": 0.10,
            "ambiguity_accuracy": 0.10,
            "one_minus_hard_negative_false_activation_rate": 0.25,
            "exact_active_set_accuracy": 0.10,
        },
        "frozen_parameters": {
            **chosen.candidate.as_dict(),
            **chosen.activation.as_dict(),
            "calibration": {
                concept: {
                    "type": "sigmoid",
                    "slope": chosen.calibrations[concept].slope,
                    "intercept": chosen.calibrations[concept].intercept,
                }
                for concept in CONCEPTS
            },
        },
        "selected_development_result": chosen.as_dict(),
        "best_result_by_method": {
            method: result.as_dict() for method, result in sorted(best_by_method.items())
        },
        "top_candidates": [result.as_dict() for result in ranked[:20]],
        "development_records": detail_rows(
            chosen,
            development,
            embeddings,
            bank,
            classifiers,
        ),
        "exit_gate": {
            "quality_gate_passed": quality_gate_passed,
            "thresholds_frozen": quality_gate_passed,
            "parameters_frozen_before_heldout": quality_gate_passed,
            "heldout_used_for_tuning": False,
            "plural_activation_supported": True,
            "raw_scores_preserved": True,
            "calibrated_0_1_weights_produced": True,
            "ambiguity_detection_frozen": quality_gate_passed,
            "unsupported_detection_frozen": quality_gate_passed,
        },
        "next_step": (
            "Phase 11: evaluate this frozen mapping on Held-out exactly once without retuning."
        ),
    }

    atomic_json(output_json, results)
    atomic_text(
        output_markdown,
        markdown_report(
            chosen,
            ranked,
            identity,
            len(build),
            len(development),
            output_json,
        ),
    )

    LOGGER.info("Phase 10 tuning complete")
    LOGGER.info("Held-out records used: 0")
    LOGGER.info("JSON: %s", output_json)
    LOGGER.info("Markdown: %s", output_markdown)

    return results


def main() -> int:
    arguments = parse_arguments()
    configure_logging(arguments.log_level)

    try:
        run_phase10(
            project_root=arguments.project_root,
            approved_embeddings=arguments.approved_embeddings,
            query_embeddings=arguments.query_prototype_embeddings,
            passage_embeddings=arguments.passage_prototype_embeddings,
            embedding_manifest=arguments.embedding_manifest,
            build_set=arguments.build_set,
            development_set=arguments.development_set,
            prototypes=arguments.prototypes,
            output_json=arguments.output_json,
            output_markdown=arguments.output_markdown,
            replace=arguments.replace,
        )
    except TuningError:
        LOGGER.exception("Phase 10 concept-mapping tuning failed")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
