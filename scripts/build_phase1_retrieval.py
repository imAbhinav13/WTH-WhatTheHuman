from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import httpx
import numpy as np
from scripts import tune_phase1_concept_mapping as phase10

LOGGER = logging.getLogger("wth.phase1.build_phase1_retrieval")

SCRIPT_VERSION: Final = "1.0.0"
RETRIEVAL_VERSION: Final = "phase1-concept-domain-retrieval-v1"

DOMAINS: Final = ("science", "advaita", "samkhya")
CONCEPTS: Final = (
    "consciousness",
    "self_identity",
    "reality_appearance",
)

DEFAULT_ACTIVE_BUNDLES: Final = Path("artifacts/phase1/active/active_chunk_bundles.jsonl")
DEFAULT_ACTIVATION_MANIFEST: Final = Path("artifacts/phase1/active/activation_manifest.json")
DEFAULT_PHASE10_RESULTS: Final = Path(
    "artifacts/phase1/evaluation/concept_mapping_dev_results.json"
)
DEFAULT_QUERY_PROTOTYPES: Final = Path(
    "artifacts/phase1/embeddings/query_prototype_embeddings.jsonl"
)
DEFAULT_PASSAGE_PROTOTYPES: Final = Path(
    "artifacts/phase1/embeddings/passage_prototype_embeddings.jsonl"
)
DEFAULT_EMBEDDING_MANIFEST: Final = Path("artifacts/phase1/embeddings/embedding_manifest.json")
DEFAULT_EVAL_QUESTIONS: Final = Path("packages/eval/eval_questions.jsonl")
DEFAULT_OUTPUT_DIRECTORY: Final = Path("artifacts/phase1/retrieval")
DEFAULT_REPORT_PATH: Final = Path("docs/evaluation/phase1_retrieval_report.md")

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

# Exit gate frozen before retrieval evaluation.
MAX_ACCEPTABLE_RECALL_RELATIVE_LOSS: Final = 0.10
MIN_PRECISION_DELTA: Final = 0.0
MIN_CONCEPT_COVERAGE_DELTA: Final = 0.0

GEMINI_API_BASE: Final = "https://generativelanguage.googleapis.com/v1beta"
QUERY_PREFIX: Final = "task: search result | query: {question}"

TOKEN_RE: Final = re.compile(r"[A-Za-z0-9]+(?:[''-][A-Za-z0-9]+)?")
SPACE_RE: Final = re.compile(r"\s+")


class RetrievalError(RuntimeError):
    """Raised when Phase 14 retrieval cannot safely complete."""


@dataclass(frozen=True)
class FrozenMapping:
    candidate: phase10.Candidate
    activation: phase10.Activation
    calibrations: dict[str, phase10.Calibration]
    prototype_version: str
    model_version: str


@dataclass(frozen=True)
class ConceptRelation:
    concept_id: str
    human_label: str
    production_active: bool
    calibrated_weight: float
    human_override: bool


@dataclass(frozen=True)
class ActiveChunk:
    chunk_id: str
    source_id: str
    domain: str
    citation: str
    reviewed_text: str
    corpus_version: str
    vector: phase10.FloatArray
    concept_relations: dict[str, ConceptRelation]
    citation_quality: float
    normalized_text: str
    token_set: frozenset[str]
    estimated_tokens: int
    structural_locator: str
    source_title: str
    translator: str


@dataclass(frozen=True)
class QueryActivation:
    question: str
    raw_scores: dict[str, float]
    calibrated_weights: dict[str, float]
    active_concepts: tuple[str, ...]
    ambiguous: bool
    unsupported: bool


@dataclass(frozen=True)
class RankedChunk:
    chunk: ActiveChunk
    vector_similarity: float
    concept_alignment: float
    human_relevance: float
    citation_quality: float
    base_score: float
    diversity_adjusted_score: float


@dataclass(frozen=True)
class RetrievalConfig:
    top_k_per_domain: int
    candidate_pool_per_domain: int
    token_budget_per_domain: int
    max_chunks_per_source: int
    min_vector_similarity: float

    def as_dict(self) -> dict[str, object]:
        return {
            "top_k_per_domain": self.top_k_per_domain,
            "candidate_pool_per_domain": self.candidate_pool_per_domain,
            "token_budget_per_domain": self.token_budget_per_domain,
            "max_chunks_per_source": self.max_chunks_per_source,
            "min_vector_similarity": self.min_vector_similarity,
        }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 14: build concept-aware, domain-separated retrieval from "
            "the frozen Phase 1 active corpus and optionally evaluate it "
            "against the frozen retrieval-question set."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--active-bundles",
        type=Path,
        default=DEFAULT_ACTIVE_BUNDLES,
    )
    parser.add_argument(
        "--activation-manifest",
        type=Path,
        default=DEFAULT_ACTIVATION_MANIFEST,
    )
    parser.add_argument(
        "--phase10-results",
        type=Path,
        default=DEFAULT_PHASE10_RESULTS,
    )
    parser.add_argument(
        "--query-prototypes",
        type=Path,
        default=DEFAULT_QUERY_PROTOTYPES,
    )
    parser.add_argument(
        "--passage-prototypes",
        type=Path,
        default=DEFAULT_PASSAGE_PROTOTYPES,
    )
    parser.add_argument(
        "--embedding-manifest",
        type=Path,
        default=DEFAULT_EMBEDDING_MANIFEST,
    )
    parser.add_argument(
        "--evaluation-questions",
        type=Path,
        default=DEFAULT_EVAL_QUESTIONS,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
    )
    parser.add_argument("--question")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run frozen-question retrieval comparison.",
    )
    parser.add_argument(
        "--top-k-per-domain",
        type=int,
        default=DEFAULT_TOP_K_PER_DOMAIN,
    )
    parser.add_argument(
        "--candidate-pool-per-domain",
        type=int,
        default=DEFAULT_CANDIDATE_POOL_PER_DOMAIN,
    )
    parser.add_argument(
        "--token-budget-per-domain",
        type=int,
        default=DEFAULT_TOKEN_BUDGET_PER_DOMAIN,
    )
    parser.add_argument(
        "--max-chunks-per-source",
        type=int,
        default=DEFAULT_MAX_CHUNKS_PER_SOURCE,
    )
    parser.add_argument(
        "--min-vector-similarity",
        type=float,
        default=DEFAULT_MIN_VECTOR_SIMILARITY,
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace derived Phase 14 outputs.",
    )
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
        raise RetrievalError(f"Required file does not exist: {path}")


def require_mapping(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RetrievalError(f"{description} must be an object.")

    result: dict[str, object] = {}
    for key, nested in value.items():
        if not isinstance(key, str):
            raise RetrievalError(f"{description} contains a non-string key.")
        result[key] = nested
    return result


def require_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RetrievalError(f"{description} must be a non-empty string.")
    return value.strip()


def optional_string(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, int | float):
        return str(value)
    return ""


def require_float(value: object, description: str) -> float:
    if not isinstance(value, int | float):
        raise RetrievalError(f"{description} must be numeric.")
    return float(value)


def require_int(value: object, description: str) -> int:
    if not isinstance(value, int):
        raise RetrievalError(f"{description} must be an integer.")
    return value


def load_json(path: Path) -> dict[str, object]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RetrievalError(f"Invalid JSON in {path}: {exc}") from exc
    return require_mapping(raw, f"JSON document {path}")


def iter_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                raw: object = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RetrievalError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc

            yield require_mapping(
                raw,
                f"JSONL record {path}:{line_number}",
            )


def atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f"{path.suffix}.tmp")
    temp.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def atomic_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f"{path.suffix}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
            handle.write("\n")
    temp.replace(path)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def normalize_text(text: str) -> str:
    return SPACE_RE.sub(" ", text.casefold()).strip()


def token_set(text: str) -> frozenset[str]:
    return frozenset(token.casefold() for token in TOKEN_RE.findall(text))


def estimated_tokens(text: str) -> int:
    words = max(1, len(text.split()))
    return max(1, math.ceil(words * 1.30))


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def l2_normalize(vector: phase10.FloatArray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise RetrievalError("Embedding vector has zero norm.")
    return vector / norm


def cosine_similarity(left: phase10.FloatArray, right: phase10.FloatArray) -> float:
    return float(np.dot(left, right))


def candidate_from_mapping(
    raw: Mapping[str, object],
) -> phase10.Candidate:
    return phase10.Candidate(
        method=require_string(raw.get("method"), "mapping method"),
        source=require_string(
            raw.get("prototype_source"),
            "prototype_source",
        ),
        aggregation=require_string(
            raw.get("prototype_aggregation"),
            "prototype_aggregation",
        ),
        negative_penalty=require_float(
            raw.get("negative_penalty"),
            "negative_penalty",
        ),
        lexical_weight=require_float(
            raw.get("lexical_weight"),
            "lexical_weight",
        ),
        embed_weight=require_float(
            raw.get("embedding_weight"),
            "embedding_weight",
        ),
    )


def activation_from_mapping(
    raw: Mapping[str, object],
) -> phase10.Activation:
    threshold_raw = require_mapping(
        raw.get("concept_activation_thresholds"),
        "concept_activation_thresholds",
    )
    return phase10.Activation(
        thresholds={
            concept: require_float(
                threshold_raw.get(concept),
                f"{concept} threshold",
            )
            for concept in CONCEPTS
        },
        ambiguity_margin=require_float(
            raw.get("ambiguity_margin"),
            "ambiguity_margin",
        ),
        max_active=require_int(
            raw.get("maximum_active_concepts"),
            "maximum_active_concepts",
        ),
    )


def calibrations_from_mapping(
    raw: Mapping[str, object],
) -> dict[str, phase10.Calibration]:
    result: dict[str, phase10.Calibration] = {}
    for concept in CONCEPTS:
        item = require_mapping(
            raw.get(concept),
            f"{concept} calibration",
        )
        result[concept] = phase10.Calibration(
            slope=require_float(
                item.get("slope"),
                f"{concept} calibration slope",
            ),
            intercept=require_float(
                item.get("intercept"),
                f"{concept} calibration intercept",
            ),
        )
    return result


def validate_frozen_mapping(path: Path) -> FrozenMapping:
    result = load_json(path)

    if optional_string(result.get("status")) != "frozen":
        raise RetrievalError("Phase 10 mapping is not frozen.")

    gate = require_mapping(result.get("exit_gate"), "Phase 10 exit_gate")
    for field_name in (
        "quality_gate_passed",
        "thresholds_frozen",
        "parameters_frozen_before_heldout",
        "plural_activation_supported",
        "raw_scores_preserved",
        "calibrated_0_1_weights_produced",
    ):
        if gate.get(field_name) is not True:
            raise RetrievalError(f"Phase 10 exit gate failed: {field_name}.")

    frozen = require_mapping(
        result.get("frozen_parameters"),
        "Phase 10 frozen_parameters",
    )
    calibration_raw = require_mapping(
        frozen.get("calibration"),
        "Phase 10 calibration",
    )
    inputs = require_mapping(result.get("inputs"), "Phase 10 inputs")
    prototype_artifact = require_mapping(
        inputs.get("prototype_artifact"),
        "Phase 10 prototype_artifact",
    )

    prototype_version = require_string(
        prototype_artifact.get("prototype_version"),
        "prototype_version",
    )

    embedding_identity = require_mapping(
        result.get("embedding_identity"),
        "Phase 10 embedding_identity",
    )
    model = require_string(
        embedding_identity.get("model"),
        "embedding model",
    )
    revision = require_string(
        embedding_identity.get("model_revision"),
        "embedding revision",
    )

    return FrozenMapping(
        candidate=candidate_from_mapping(frozen),
        activation=activation_from_mapping(frozen),
        calibrations=calibrations_from_mapping(calibration_raw),
        prototype_version=prototype_version,
        model_version=f"{model}@{revision}",
    )


def validate_activation_manifest(
    path: Path,
) -> tuple[dict[str, object], str, int]:
    manifest = load_json(path)
    if optional_string(manifest.get("status")) != "activation_artifacts_complete":
        raise RetrievalError("Phase 13 activation artifacts are not complete.")

    gate = require_mapping(
        manifest.get("exit_gate"),
        "Phase 13 exit_gate",
    )
    required_true = (
        "active_chunk_count_within_250_350",
        "approved_source_present_for_every_chunk",
        "source_rights_status_present_for_every_chunk",
        "source_checksum_present_for_every_chunk",
        "citation_present_for_every_chunk",
        "reviewed_text_present_for_every_chunk",
        "selected_embedding_present_for_every_chunk",
        "embedding_metadata_present_for_every_chunk",
        "reviewed_concept_labels_present_for_every_chunk",
        "calibrated_weights_present_for_every_chunk",
        "corpus_version_present_for_every_chunk",
        "review_decision_present_for_every_chunk",
    )
    for field_name in required_true:
        if gate.get(field_name) is not True:
            raise RetrievalError(f"Phase 13 exit gate failed: {field_name}.")

    counts = require_mapping(
        manifest.get("counts"),
        "Phase 13 counts",
    )
    active_count = require_int(
        counts.get("active_chunk_count"),
        "active_chunk_count",
    )
    corpus_version = require_string(
        manifest.get("corpus_version"),
        "corpus_version",
    )
    return manifest, corpus_version, active_count


def citation_quality_score(citation: Mapping[str, object]) -> float:
    citation_text = optional_string(citation.get("citation"))
    locator = optional_string(citation.get("structural_locator"))
    verified = optional_string(citation.get("citation_verified")).casefold()

    if not citation_text:
        return 0.0

    score = 0.70
    if locator:
        score += 0.15
    if verified in {"true", "yes", "verified", "reviewed", "1"}:
        score += 0.15
    return min(score, 1.0)


def parse_active_chunks(
    path: Path,
    *,
    expected_count: int,
    corpus_version: str,
    dimensions: int,
) -> list[ActiveChunk]:
    chunks: list[ActiveChunk] = []
    seen: set[str] = set()

    for raw in iter_jsonl(path):
        chunk_id = require_string(
            raw.get("chunk_id"),
            "active chunk_id",
        )
        if chunk_id in seen:
            raise RetrievalError(f"Duplicate active chunk bundle: {chunk_id}")
        seen.add(chunk_id)

        if raw.get("queryable") is not True:
            raise RetrievalError(f"{chunk_id} is not queryable.")
        if optional_string(raw.get("lifecycle_status")) != "active":
            raise RetrievalError(f"{chunk_id} is not active.")
        if optional_string(raw.get("corpus_version")) != corpus_version:
            raise RetrievalError(f"{chunk_id} corpus version mismatch.")

        source = require_mapping(
            raw.get("source"),
            f"{chunk_id} source",
        )
        content = require_mapping(
            raw.get("content"),
            f"{chunk_id} content",
        )
        embedding = require_mapping(
            raw.get("embedding"),
            f"{chunk_id} embedding",
        )
        citation = require_mapping(
            raw.get("citation"),
            f"{chunk_id} citation",
        )

        vector_raw = embedding.get("vector")
        if not isinstance(vector_raw, list):
            raise RetrievalError(f"{chunk_id} embedding vector must be a list.")
        if len(vector_raw) != dimensions:
            raise RetrievalError(f"{chunk_id} embedding dimensions mismatch.")

        try:
            vector = np.asarray(vector_raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise RetrievalError(f"{chunk_id} embedding contains non-numeric values.") from exc
        vector = l2_normalize(vector)

        concepts_raw = raw.get("concepts")
        if not isinstance(concepts_raw, list):
            raise RetrievalError(f"{chunk_id} concepts must be a list.")

        relations: dict[str, ConceptRelation] = {}
        for item_raw in concepts_raw:
            item = require_mapping(
                item_raw,
                f"{chunk_id} concept relation",
            )
            concept_id = require_string(
                item.get("concept_id"),
                f"{chunk_id} concept_id",
            )
            if concept_id not in CONCEPTS:
                raise RetrievalError(f"{chunk_id} has unexpected concept {concept_id!r}.")
            production_active = item.get("production_active")
            human_override = item.get("human_override")
            if not isinstance(production_active, bool):
                raise RetrievalError(f"{chunk_id}/{concept_id} production_active invalid.")
            if not isinstance(human_override, bool):
                raise RetrievalError(f"{chunk_id}/{concept_id} human_override invalid.")

            relations[concept_id] = ConceptRelation(
                concept_id=concept_id,
                human_label=require_string(
                    item.get("human_label"),
                    f"{chunk_id}/{concept_id} human_label",
                ).casefold(),
                production_active=production_active,
                calibrated_weight=require_float(
                    item.get("calibrated_weight"),
                    f"{chunk_id}/{concept_id} calibrated_weight",
                ),
                human_override=human_override,
            )

        if set(relations) != set(CONCEPTS):
            raise RetrievalError(f"{chunk_id} does not have all Phase 1 concept relations.")

        reviewed_text = require_string(
            content.get("reviewed_text"),
            f"{chunk_id} reviewed_text",
        )
        domain = require_string(
            source.get("domain"),
            f"{chunk_id} domain",
        ).casefold()
        if domain not in DOMAINS:
            raise RetrievalError(f"{chunk_id} has invalid domain {domain!r}.")

        chunks.append(
            ActiveChunk(
                chunk_id=chunk_id,
                source_id=require_string(
                    source.get("source_id"),
                    f"{chunk_id} source_id",
                ),
                domain=domain,
                citation=require_string(
                    citation.get("citation"),
                    f"{chunk_id} citation",
                ),
                reviewed_text=reviewed_text,
                corpus_version=corpus_version,
                vector=vector,
                concept_relations=relations,
                citation_quality=citation_quality_score(citation),
                normalized_text=normalize_text(reviewed_text),
                token_set=token_set(reviewed_text),
                estimated_tokens=estimated_tokens(reviewed_text),
                structural_locator=optional_string(citation.get("structural_locator")),
                source_title=optional_string(source.get("source_title")),
                translator=optional_string(source.get("translator")),
            )
        )

    if len(chunks) != expected_count:
        raise RetrievalError(f"Expected {expected_count} active chunks, found {len(chunks)}.")

    return chunks


def api_key_from_env(project_root: Path) -> str:
    for key_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.getenv(key_name)
        if value and value.strip():
            return value.strip()

    candidates = (
        project_root / ".env",
        project_root / ".env.local",
        project_root / "apps" / "api" / ".env",
        project_root / "apps" / "api" / ".env.local",
    )

    for path in candidates:
        if not path.is_file():
            continue
        for line in path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, raw_value = stripped.split("=", 1)
            if key.strip() not in {"GEMINI_API_KEY", "GOOGLE_API_KEY"}:
                continue
            value = raw_value.strip().strip('"').strip("'")
            if value:
                return value

    raise RetrievalError(
        "Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY "
        "in the environment or project .env file."
    )


def embedding_identity(
    manifest_path: Path,
) -> tuple[dict[str, object], int, str]:
    identity, _manifest = phase10.load_identity(manifest_path)
    as_dict = getattr(identity, "as_dict", None)
    if not callable(as_dict):
        raise RetrievalError("Embedding identity does not expose as_dict().")
    identity_dict = require_mapping(
        as_dict(),
        "embedding identity",
    )

    dimensions = require_int(
        identity_dict.get("dimensions"),
        "embedding dimensions",
    )
    model = require_string(
        identity_dict.get("model"),
        "embedding model",
    )

    if model != "gemini-embedding-2":
        raise RetrievalError(
            "Phase 14 query embedder currently supports the frozen "
            f"gemini-embedding-2 configuration, found {model!r}."
        )

    return identity_dict, dimensions, model


def embed_question(
    *,
    question: str,
    model: str,
    dimensions: int,
    api_key: str,
) -> phase10.FloatArray:
    prepared = QUERY_PREFIX.format(question=question.strip())
    url = f"{GEMINI_API_BASE}/models/{model}:embedContent"
    payload = {
        "model": f"models/{model}",
        "content": {
            "parts": [
                {
                    "text": prepared,
                }
            ]
        },
        "embedContentConfig": {
            "outputDimensionality": dimensions,
        },
    }

    attempts = 4
    for attempt in range(1, attempts + 1):
        try:
            with httpx.Client(timeout=45.0) as client:
                response = client.post(
                    url,
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": api_key,
                    },
                    json=payload,
                )
        except httpx.HTTPError as exc:
            if attempt == attempts:
                raise RetrievalError(f"Gemini embedding request failed: {exc}") from exc
            time.sleep(float(attempt * 2))
            continue

        if response.status_code == 200:
            raw = require_mapping(
                response.json(),
                "Gemini embedding response",
            )
            embedding = require_mapping(
                raw.get("embedding"),
                "Gemini embedding",
            )
            values = embedding.get("values")
            if not isinstance(values, list):
                raise RetrievalError("Gemini embedding response has no values list.")
            if len(values) != dimensions:
                raise RetrievalError(
                    f"Gemini query embedding dimension mismatch: {len(values)} != {dimensions}."
                )
            try:
                vector = np.asarray(values, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise RetrievalError("Gemini query embedding contains non-numeric values.") from exc
            return l2_normalize(vector)

        if response.status_code in {429, 500, 502, 503, 504} and attempt < attempts:
            LOGGER.warning(
                "Query embedding retry %d/%d status=%d",
                attempt,
                attempts,
                response.status_code,
            )
            time.sleep(float(attempt * 3))
            continue

        raise RetrievalError(
            f"Gemini embedding request failed status={response.status_code}: {response.text[:500]}"
        )

    raise RetrievalError("Gemini embedding request exhausted retries.")


def query_activation(
    *,
    question: str,
    query_vector: phase10.FloatArray,
    mapping: FrozenMapping,
    bank: phase10.PrototypeBank,
) -> QueryActivation:
    record = phase10.LabeledRecord(
        chunk_id="runtime_query",
        split="runtime",
        text=question,
        labels=dict.fromkeys(CONCEPTS, "negative"),
        hard_negative_for=(),
    )

    raw = phase10.raw_scores(
        mapping.candidate,
        record,
        query_vector,
        bank,
        {},
    )
    calibrated = phase10.apply_calibration(
        raw,
        mapping.calibrations,
    )
    active = tuple(
        phase10.predicted_active(
            calibrated,
            mapping.activation,
        )
    )

    eligible = sorted(
        ((concept, calibrated[concept]) for concept in active),
        key=lambda item: (-item[1], item[0]),
    )
    ambiguous = (
        len(eligible) >= 2
        and abs(eligible[0][1] - eligible[1][1]) <= mapping.activation.ambiguity_margin
    )

    return QueryActivation(
        question=question,
        raw_scores={concept: float(raw[concept]) for concept in CONCEPTS},
        calibrated_weights={concept: float(calibrated[concept]) for concept in CONCEPTS},
        active_concepts=tuple(concept for concept, _ in eligible),
        ambiguous=ambiguous,
        unsupported=not bool(active),
    )


def human_label_value(label: str) -> float:
    if label == "positive":
        return 1.0
    if label == "partial":
        return 0.65
    return 0.0


def alignment_components(
    chunk: ActiveChunk,
    activation: QueryActivation,
) -> tuple[float, float]:
    if not activation.active_concepts:
        return 0.0, 0.0

    total_query_weight = sum(
        activation.calibrated_weights[concept] for concept in activation.active_concepts
    )
    if total_query_weight <= 0.0:
        return 0.0, 0.0

    alignment = 0.0
    human_relevance = 0.0

    for concept in activation.active_concepts:
        query_weight = activation.calibrated_weights[concept]
        relation = chunk.concept_relations[concept]

        if relation.production_active:
            alignment += query_weight * relation.calibrated_weight
            human_relevance += query_weight * human_label_value(relation.human_label)

    return (
        alignment / total_query_weight,
        human_relevance / total_query_weight,
    )


def score_candidate(
    *,
    chunk: ActiveChunk,
    query_vector: phase10.FloatArray,
    activation: QueryActivation,
    concept_aware: bool,
) -> RankedChunk:
    vector_similarity = cosine_similarity(query_vector, chunk.vector)

    concept_alignment = 0.0
    human_relevance = 0.0

    if concept_aware:
        concept_alignment, human_relevance = alignment_components(
            chunk,
            activation,
        )

    if concept_aware:
        base_score = (
            VECTOR_WEIGHT * vector_similarity
            + CONCEPT_ALIGNMENT_WEIGHT * concept_alignment
            + HUMAN_RELEVANCE_WEIGHT * human_relevance
            + CITATION_QUALITY_WEIGHT * chunk.citation_quality
        )
    else:
        base_score = vector_similarity

    return RankedChunk(
        chunk=chunk,
        vector_similarity=vector_similarity,
        concept_alignment=concept_alignment,
        human_relevance=human_relevance,
        citation_quality=chunk.citation_quality,
        base_score=base_score,
        diversity_adjusted_score=base_score,
    )


def chunk_is_concept_eligible(
    chunk: ActiveChunk,
    activation: QueryActivation,
) -> bool:
    return any(
        chunk.concept_relations[concept].production_active for concept in activation.active_concepts
    )


def is_duplicate(
    candidate: ActiveChunk,
    selected: Sequence[RankedChunk],
) -> bool:
    for existing_ranked in selected:
        existing = existing_ranked.chunk

        if candidate.chunk_id == existing.chunk_id:
            return True

        if candidate.normalized_text == existing.normalized_text:
            return True

        if (
            len(candidate.token_set) < MIN_DEDUP_TOKEN_COUNT
            or len(existing.token_set) < MIN_DEDUP_TOKEN_COUNT
        ):
            continue

        overlap = jaccard(candidate.token_set, existing.token_set)

        if candidate.source_id == existing.source_id and overlap >= SAME_SOURCE_OVERLAP_JACCARD:
            return True

        if overlap >= CROSS_SOURCE_DUPLICATE_JACCARD:
            return True

    return False


def select_with_diversity_and_budget(
    candidates: Sequence[RankedChunk],
    *,
    config: RetrievalConfig,
) -> list[RankedChunk]:
    selected: list[RankedChunk] = []
    source_counts: Counter[str] = Counter()
    used_tokens = 0

    remaining = list(candidates)

    while remaining and len(selected) < config.top_k_per_domain:
        adjusted: list[RankedChunk] = []

        for item in remaining:
            repeats = source_counts[item.chunk.source_id]
            diversity_adjusted = item.base_score - SOURCE_REPEAT_PENALTY * repeats
            adjusted.append(
                RankedChunk(
                    chunk=item.chunk,
                    vector_similarity=item.vector_similarity,
                    concept_alignment=item.concept_alignment,
                    human_relevance=item.human_relevance,
                    citation_quality=item.citation_quality,
                    base_score=item.base_score,
                    diversity_adjusted_score=diversity_adjusted,
                )
            )

        adjusted.sort(
            key=lambda item: (
                -item.diversity_adjusted_score,
                -item.vector_similarity,
                item.chunk.source_id,
                item.chunk.chunk_id,
            )
        )

        chosen: RankedChunk | None = None

        for item in adjusted:
            if source_counts[item.chunk.source_id] >= config.max_chunks_per_source:
                continue

            if is_duplicate(item.chunk, selected):
                continue

            next_tokens = used_tokens + item.chunk.estimated_tokens
            if next_tokens > config.token_budget_per_domain:
                continue

            chosen = item
            break

        if chosen is None:
            break

        selected.append(chosen)
        source_counts[chosen.chunk.source_id] += 1
        used_tokens += chosen.chunk.estimated_tokens
        remaining = [item for item in remaining if item.chunk.chunk_id != chosen.chunk.chunk_id]

    return selected


def retrieve_domain(
    *,
    domain: str,
    chunks: Sequence[ActiveChunk],
    query_vector: phase10.FloatArray,
    activation: QueryActivation,
    concept_aware: bool,
    config: RetrievalConfig,
) -> list[RankedChunk]:
    candidates: list[RankedChunk] = []

    if concept_aware and activation.unsupported:
        return []

    for chunk in chunks:
        if chunk.domain != domain:
            continue

        if concept_aware and not chunk_is_concept_eligible(
            chunk,
            activation,
        ):
            continue

        scored = score_candidate(
            chunk=chunk,
            query_vector=query_vector,
            activation=activation,
            concept_aware=concept_aware,
        )

        if scored.vector_similarity < config.min_vector_similarity:
            continue

        candidates.append(scored)

    candidates.sort(
        key=lambda item: (
            -item.base_score,
            -item.vector_similarity,
            item.chunk.source_id,
            item.chunk.chunk_id,
        )
    )

    pool = candidates[: config.candidate_pool_per_domain]
    return select_with_diversity_and_budget(
        pool,
        config=config,
    )


def retrieve_all_domains(
    *,
    chunks: Sequence[ActiveChunk],
    query_vector: phase10.FloatArray,
    activation: QueryActivation,
    concept_aware: bool,
    config: RetrievalConfig,
) -> dict[str, list[RankedChunk]]:
    return {
        domain: retrieve_domain(
            domain=domain,
            chunks=chunks,
            query_vector=query_vector,
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
    chunk = item.chunk
    return {
        "rank": rank,
        "chunk_id": chunk.chunk_id,
        "source_id": chunk.source_id,
        "domain": chunk.domain,
        "citation": chunk.citation,
        "reviewed_text": chunk.reviewed_text,
        "corpus_version": chunk.corpus_version,
        "estimated_tokens": chunk.estimated_tokens,
        "scores": {
            "vector_similarity": item.vector_similarity,
            "concept_alignment": item.concept_alignment,
            "human_relevance": item.human_relevance,
            "citation_quality": item.citation_quality,
            "base_score": item.base_score,
            "diversity_adjusted_score": (item.diversity_adjusted_score),
        },
        "concepts": {
            concept: {
                "human_label": relation.human_label,
                "production_active": relation.production_active,
                "calibrated_weight": relation.calibrated_weight,
                "human_override": relation.human_override,
            }
            for concept, relation in sorted(chunk.concept_relations.items())
        },
    }


def evidence_package(
    *,
    question: str,
    activation: QueryActivation,
    retrieval: Mapping[str, Sequence[RankedChunk]],
    config: RetrievalConfig,
    corpus_version: str,
    retrieval_mode: str,
    model_version: str,
    prototype_version: str,
) -> dict[str, object]:
    domains: dict[str, object] = {}

    for domain in DOMAINS:
        items = retrieval[domain]
        domains[domain] = {
            "status": "evidence_found" if items else "no_strong_match",
            "evidence_count": len(items),
            "estimated_tokens": sum(item.chunk.estimated_tokens for item in items),
            "unique_source_count": len({item.chunk.source_id for item in items}),
            "evidence": [
                ranked_chunk_payload(item, rank=index) for index, item in enumerate(items, start=1)
            ],
        }

    return {
        "retrieval_version": RETRIEVAL_VERSION,
        "retrieval_mode": retrieval_mode,
        "generated_at": utc_now(),
        "question": question,
        "query_activation": {
            "raw_scores": activation.raw_scores,
            "calibrated_weights": activation.calibrated_weights,
            "active_concepts": list(activation.active_concepts),
            "ambiguous": activation.ambiguous,
            "unsupported": activation.unsupported,
        },
        "config": config.as_dict(),
        "scoring": {
            "vector_similarity_weight": (
                VECTOR_WEIGHT if retrieval_mode == "concept_aware" else 1.0
            ),
            "concept_alignment_weight": (
                CONCEPT_ALIGNMENT_WEIGHT if retrieval_mode == "concept_aware" else 0.0
            ),
            "human_relevance_weight": (
                HUMAN_RELEVANCE_WEIGHT if retrieval_mode == "concept_aware" else 0.0
            ),
            "citation_quality_weight": (
                CITATION_QUALITY_WEIGHT if retrieval_mode == "concept_aware" else 0.0
            ),
            "source_repeat_penalty": SOURCE_REPEAT_PENALTY,
        },
        "corpus_version": corpus_version,
        "model_version": model_version,
        "prototype_version": prototype_version,
        "domains": domains,
    }


def parse_expected_concepts(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise RetrievalError("Evaluation expected_concepts must be a list.")

    concepts: list[str] = []
    aliases = {
        "self": "self_identity",
        "reality": "reality_appearance",
        "appearance": "reality_appearance",
    }

    for item in value:
        if not isinstance(item, str):
            raise RetrievalError("Evaluation expected_concepts contains non-string value.")
        concept = aliases.get(item.strip().casefold(), item.strip().casefold())
        if concept not in CONCEPTS:
            raise RetrievalError(f"Evaluation question has unsupported concept {concept!r}.")
        concepts.append(concept)

    return tuple(dict.fromkeys(concepts))


def parse_expected_domains(value: object) -> tuple[str, ...]:
    if value is None:
        return DOMAINS

    if not isinstance(value, list | tuple):
        raise RetrievalError("Evaluation expected_domains must be a list when present.")

    domains: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise RetrievalError("Evaluation expected_domains contains non-string value.")
        domain = item.strip().casefold()
        if domain not in DOMAINS:
            raise RetrievalError(f"Evaluation question has unsupported domain {domain!r}.")
        domains.append(domain)

    return tuple(dict.fromkeys(domains))


def load_evaluation_questions(
    path: Path,
) -> list[dict[str, object]]:
    questions: list[dict[str, object]] = []

    for index, raw in enumerate(iter_jsonl(path), start=1):
        question = require_string(
            raw.get("question_text"),
            f"evaluation question {index} question_text",
        )
        expected_concepts = parse_expected_concepts(raw.get("expected_concepts"))
        expected_domains = parse_expected_domains(raw.get("expected_domains"))

        if not expected_concepts:
            raise RetrievalError(f"Evaluation question {index} has no expected concepts.")

        question_id = optional_string(raw.get("id")) or f"q{index:03d}"

        questions.append(
            {
                "id": question_id,
                "question_text": question,
                "expected_concepts": list(expected_concepts),
                "expected_domains": list(expected_domains),
                "reviewer_notes": optional_string(raw.get("reviewer_notes")),
            }
        )

    if not questions:
        raise RetrievalError("Frozen retrieval evaluation question file is empty.")

    return questions


def is_relevant_to_expected(
    chunk: ActiveChunk,
    expected_concepts: Sequence[str],
) -> bool:
    return any(chunk.concept_relations[concept].production_active for concept in expected_concepts)


def domain_metrics(
    *,
    retrieved: Sequence[RankedChunk],
    all_domain_chunks: Sequence[ActiveChunk],
    expected_concepts: Sequence[str],
) -> dict[str, float | int]:
    relevant_total = sum(
        1 for chunk in all_domain_chunks if is_relevant_to_expected(chunk, expected_concepts)
    )
    relevant_retrieved = sum(
        1
        for item in retrieved
        if is_relevant_to_expected(
            item.chunk,
            expected_concepts,
        )
    )

    retrieved_count = len(retrieved)
    precision = relevant_retrieved / retrieved_count if retrieved_count else 0.0
    recall = relevant_retrieved / relevant_total if relevant_total else 1.0

    covered_concepts = {
        concept
        for concept in expected_concepts
        if any(item.chunk.concept_relations[concept].production_active for item in retrieved)
    }
    concept_coverage = len(covered_concepts) / len(expected_concepts) if expected_concepts else 1.0

    reciprocal_rank = 0.0
    for rank, item in enumerate(retrieved, start=1):
        if is_relevant_to_expected(
            item.chunk,
            expected_concepts,
        ):
            reciprocal_rank = 1.0 / rank
            break

    source_diversity = (
        len({item.chunk.source_id for item in retrieved}) / retrieved_count
        if retrieved_count
        else 0.0
    )

    return {
        "retrieved_count": retrieved_count,
        "relevant_retrieved": relevant_retrieved,
        "relevant_total": relevant_total,
        "precision_at_k": precision,
        "recall_at_k": recall,
        "concept_coverage": concept_coverage,
        "reciprocal_rank": reciprocal_rank,
        "source_diversity": source_diversity,
    }


def aggregate_metrics(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    if not rows:
        return {
            "macro_precision_at_k": 0.0,
            "macro_recall_at_k": 0.0,
            "macro_concept_coverage": 0.0,
            "mrr": 0.0,
            "mean_source_diversity": 0.0,
        }

    def mean(key: str) -> float:
        values = [require_float(row.get(key), key) for row in rows]
        return sum(values) / len(values)

    return {
        "macro_precision_at_k": mean("precision_at_k"),
        "macro_recall_at_k": mean("recall_at_k"),
        "macro_concept_coverage": mean("concept_coverage"),
        "mrr": mean("reciprocal_rank"),
        "mean_source_diversity": mean("source_diversity"),
    }


def evaluate_retrieval(
    *,
    questions: Sequence[Mapping[str, object]],
    chunks: Sequence[ActiveChunk],
    mapping: FrozenMapping,
    bank: phase10.PrototypeBank,
    model: str,
    dimensions: int,
    api_key: str,
    config: RetrievalConfig,
    corpus_version: str,
) -> dict[str, object]:
    per_question: list[dict[str, object]] = []
    plain_metric_rows: list[dict[str, object]] = []
    concept_metric_rows: list[dict[str, object]] = []

    by_domain = {
        domain: [chunk for chunk in chunks if chunk.domain == domain] for domain in DOMAINS
    }

    for index, raw in enumerate(questions, start=1):
        question_id = require_string(
            raw.get("id"),
            f"evaluation question {index} id",
        )
        question = require_string(
            raw.get("question_text"),
            f"{question_id} question_text",
        )
        expected_concepts_raw = raw.get("expected_concepts")
        expected_domains_raw = raw.get("expected_domains")
        if not isinstance(expected_concepts_raw, list):
            raise RetrievalError(f"{question_id} expected_concepts invalid.")
        if not isinstance(expected_domains_raw, list):
            raise RetrievalError(f"{question_id} expected_domains invalid.")
        expected_concepts = tuple(
            require_string(item, f"{question_id} expected concept")
            for item in expected_concepts_raw
        )
        expected_domains = tuple(
            require_string(item, f"{question_id} expected domain") for item in expected_domains_raw
        )

        LOGGER.info(
            "Retrieval eval %d/%d id=%s",
            index,
            len(questions),
            question_id,
        )

        query_vector = embed_question(
            question=question,
            model=model,
            dimensions=dimensions,
            api_key=api_key,
        )
        activation = query_activation(
            question=question,
            query_vector=query_vector,
            mapping=mapping,
            bank=bank,
        )

        plain = retrieve_all_domains(
            chunks=chunks,
            query_vector=query_vector,
            activation=activation,
            concept_aware=False,
            config=config,
        )
        concept = retrieve_all_domains(
            chunks=chunks,
            query_vector=query_vector,
            activation=activation,
            concept_aware=True,
            config=config,
        )

        plain_domains: dict[str, object] = {}
        concept_domains: dict[str, object] = {}

        for domain in expected_domains:
            plain_metrics = domain_metrics(
                retrieved=plain[domain],
                all_domain_chunks=by_domain[domain],
                expected_concepts=expected_concepts,
            )
            concept_metrics = domain_metrics(
                retrieved=concept[domain],
                all_domain_chunks=by_domain[domain],
                expected_concepts=expected_concepts,
            )

            plain_metric_rows.append(dict[str, object](plain_metrics))
            concept_metric_rows.append(dict[str, object](concept_metrics))

            plain_domains[domain] = {
                "metrics": plain_metrics,
                "chunk_ids": [item.chunk.chunk_id for item in plain[domain]],
            }
            concept_domains[domain] = {
                "metrics": concept_metrics,
                "chunk_ids": [item.chunk.chunk_id for item in concept[domain]],
            }

        per_question.append(
            {
                "id": question_id,
                "question_text": question,
                "expected_concepts": list(expected_concepts),
                "expected_domains": list(expected_domains),
                "activated_concepts": list(activation.active_concepts),
                "unsupported": activation.unsupported,
                "plain_vector": plain_domains,
                "concept_aware": concept_domains,
            }
        )

    plain_overall = aggregate_metrics(plain_metric_rows)
    concept_overall = aggregate_metrics(concept_metric_rows)

    precision_delta = (
        concept_overall["macro_precision_at_k"] - plain_overall["macro_precision_at_k"]
    )
    coverage_delta = (
        concept_overall["macro_concept_coverage"] - plain_overall["macro_concept_coverage"]
    )
    plain_recall = plain_overall["macro_recall_at_k"]
    concept_recall = concept_overall["macro_recall_at_k"]

    if plain_recall <= 0.0:
        recall_ratio = 1.0 if concept_recall >= plain_recall else 0.0
    else:
        recall_ratio = concept_recall / plain_recall

    recall_gate = recall_ratio >= (1.0 - MAX_ACCEPTABLE_RECALL_RELATIVE_LOSS)
    precision_gate = precision_delta >= MIN_PRECISION_DELTA
    coverage_gate = coverage_delta >= MIN_CONCEPT_COVERAGE_DELTA
    improvement_gate = (
        precision_delta > 0.0
        or coverage_delta > 0.0
        or concept_overall["mrr"] > plain_overall["mrr"]
    )

    retained = recall_gate and precision_gate and coverage_gate and improvement_gate

    return {
        "evaluation_version": "phase1-retrieval-evaluation-v1",
        "question_count": len(questions),
        "metric_definition": {
            "relevance": (
                "A retrieved chunk is label-relevant when its authoritative "
                "production_active relation overlaps at least one frozen "
                "expected concept for the evaluation question."
            ),
            "recall": (
                "Retrieved authoritative concept-relevant chunks divided by "
                "all authoritative concept-relevant active chunks in that domain."
            ),
            "concept_coverage": (
                "Fraction of frozen expected concepts represented by at least "
                "one retrieved production-active chunk in the domain."
            ),
        },
        "plain_vector": plain_overall,
        "concept_aware": concept_overall,
        "deltas": {
            "macro_precision_at_k": precision_delta,
            "macro_concept_coverage": coverage_delta,
            "macro_recall_at_k": (concept_recall - plain_recall),
            "mrr": (concept_overall["mrr"] - plain_overall["mrr"]),
            "mean_source_diversity": (
                concept_overall["mean_source_diversity"] - plain_overall["mean_source_diversity"]
            ),
            "recall_ratio_vs_baseline": recall_ratio,
        },
        "retention_gate": {
            "maximum_acceptable_recall_relative_loss": (MAX_ACCEPTABLE_RECALL_RELATIVE_LOSS),
            "minimum_precision_delta": MIN_PRECISION_DELTA,
            "minimum_concept_coverage_delta": (MIN_CONCEPT_COVERAGE_DELTA),
            "precision_not_worse": precision_gate,
            "concept_coverage_not_worse": coverage_gate,
            "recall_loss_acceptable": recall_gate,
            "at_least_one_quality_metric_improved": improvement_gate,
            "concept_aware_retained": retained,
        },
        "per_question": per_question,
    }


def markdown_report(
    *,
    evaluation: Mapping[str, object],
    config: RetrievalConfig,
    corpus_version: str,
) -> str:
    plain = require_mapping(
        evaluation.get("plain_vector"),
        "plain_vector metrics",
    )
    concept = require_mapping(
        evaluation.get("concept_aware"),
        "concept_aware metrics",
    )
    deltas = require_mapping(
        evaluation.get("deltas"),
        "retrieval deltas",
    )
    gate = require_mapping(
        evaluation.get("retention_gate"),
        "retention_gate",
    )

    retained = gate.get("concept_aware_retained") is True

    lines = [
        "# Phase 1 Retrieval Evaluation",
        "",
        f"- Retrieval version: `{RETRIEVAL_VERSION}`",
        f"- Corpus version: `{corpus_version}`",
        f"- Questions: {require_int(evaluation.get('question_count'), 'question_count')}",
        f"- Top-k per domain: {config.top_k_per_domain}",
        f"- Token budget per domain: {config.token_budget_per_domain}",
        f"- Decision: **{'RETAIN CONCEPT-AWARE' if retained else 'DO NOT RETAIN YET'}**",
        "",
        "## Overall metrics",
        "",
        "| Metric | Plain vector | Concept-aware | Delta |",
        "|---|---:|---:|---:|",
        (
            "| Macro precision@k | "
            f"{require_float(plain.get('macro_precision_at_k'), 'plain precision'):.4f} | "
            f"{require_float(concept.get('macro_precision_at_k'), 'concept precision'):.4f} | "
            f"{require_float(deltas.get('macro_precision_at_k'), 'precision delta'):+.4f} |"
        ),
        (
            "| Macro recall@k | "
            f"{require_float(plain.get('macro_recall_at_k'), 'plain recall'):.4f} | "
            f"{require_float(concept.get('macro_recall_at_k'), 'concept recall'):.4f} | "
            f"{require_float(deltas.get('macro_recall_at_k'), 'recall delta'):+.4f} |"
        ),
        (
            "| Concept coverage | "
            f"{require_float(plain.get('macro_concept_coverage'), 'plain coverage'):.4f} | "
            f"{require_float(concept.get('macro_concept_coverage'), 'concept coverage'):.4f} | "
            f"{require_float(deltas.get('macro_concept_coverage'), 'coverage delta'):+.4f} |"
        ),
        (
            "| MRR | "
            f"{require_float(plain.get('mrr'), 'plain mrr'):.4f} | "
            f"{require_float(concept.get('mrr'), 'concept mrr'):.4f} | "
            f"{require_float(deltas.get('mrr'), 'mrr delta'):+.4f} |"
        ),
        (
            "| Source diversity | "
            f"{require_float(plain.get('mean_source_diversity'), 'plain diversity'):.4f} | "
            f"{require_float(concept.get('mean_source_diversity'), 'concept diversity'):.4f} | "
            f"{require_float(deltas.get('mean_source_diversity'), 'diversity delta'):+.4f} |"
        ),
        "",
        "## Exit gate",
        "",
        f"- Precision not worse: `{gate.get('precision_not_worse')}`",
        f"- Concept coverage not worse: `{gate.get('concept_coverage_not_worse')}`",
        f"- Recall loss acceptable: `{gate.get('recall_loss_acceptable')}`",
        f"- At least one quality metric improved: `{gate.get('at_least_one_quality_metric_improved')}`",
        f"- Concept-aware retained: `{gate.get('concept_aware_retained')}`",
        "",
        "## Important limitation",
        "",
        (
            "This automatic retrieval evaluation uses the frozen question's "
            "expected concepts plus authoritative reviewed chunk labels as "
            "the relevance signal. It measures concept relevance and coverage; "
            "it is not a substitute for human judgment of question-specific "
            "passage usefulness."
        ),
        "",
    ]
    return "\n".join(lines)


def validate_config(config: RetrievalConfig) -> None:
    if config.top_k_per_domain <= 0:
        raise RetrievalError("top_k_per_domain must be positive.")
    if config.candidate_pool_per_domain < config.top_k_per_domain:
        raise RetrievalError("candidate_pool_per_domain must be >= top_k_per_domain.")
    if config.token_budget_per_domain <= 0:
        raise RetrievalError("token_budget_per_domain must be positive.")
    if config.max_chunks_per_source <= 0:
        raise RetrievalError("max_chunks_per_source must be positive.")
    if not -1.0 <= config.min_vector_similarity <= 1.0:
        raise RetrievalError("min_vector_similarity must be between -1 and 1.")


def run_phase14(
    *,
    project_root: Path,
    active_bundles_path: Path,
    activation_manifest_path: Path,
    phase10_results_path: Path,
    query_prototypes_path: Path,
    passage_prototypes_path: Path,
    embedding_manifest_path: Path,
    evaluation_questions_path: Path,
    output_directory: Path,
    report_path: Path,
    question: str | None,
    evaluate: bool,
    config: RetrievalConfig,
    replace: bool,
) -> dict[str, object]:
    validate_config(config)

    project_root = project_root.resolve()
    active_bundles_path = resolve(
        project_root,
        active_bundles_path,
    )
    activation_manifest_path = resolve(
        project_root,
        activation_manifest_path,
    )
    phase10_results_path = resolve(
        project_root,
        phase10_results_path,
    )
    query_prototypes_path = resolve(
        project_root,
        query_prototypes_path,
    )
    passage_prototypes_path = resolve(
        project_root,
        passage_prototypes_path,
    )
    embedding_manifest_path = resolve(
        project_root,
        embedding_manifest_path,
    )
    evaluation_questions_path = resolve(
        project_root,
        evaluation_questions_path,
    )
    output_directory = resolve(project_root, output_directory)
    report_path = resolve(project_root, report_path)

    for path in (
        active_bundles_path,
        activation_manifest_path,
        phase10_results_path,
        query_prototypes_path,
        passage_prototypes_path,
        embedding_manifest_path,
    ):
        require_file(path)

    if not question and not evaluate:
        raise RetrievalError(
            "Provide --question for a retrieval smoke test, --evaluate for "
            "frozen-question evaluation, or both."
        )

    if evaluate:
        require_file(evaluation_questions_path)

    LOGGER.info("Phase 14 retrieval starting")

    _activation_manifest, corpus_version, active_count = validate_activation_manifest(
        activation_manifest_path
    )
    mapping = validate_frozen_mapping(phase10_results_path)
    identity, dimensions, model = embedding_identity(embedding_manifest_path)

    chunks = parse_active_chunks(
        active_bundles_path,
        expected_count=active_count,
        corpus_version=corpus_version,
        dimensions=dimensions,
    )

    bank = phase10.load_prototype_bank(
        query_prototypes_path,
        passage_prototypes_path,
        phase10.load_identity(embedding_manifest_path)[0],
    )

    api_key = api_key_from_env(project_root)
    output_directory.mkdir(parents=True, exist_ok=True)

    retrieval_config_path = output_directory / "retrieval_config.json"
    evaluation_output_path = output_directory / "retrieval_evaluation_results.json"
    smoke_output_path = output_directory / "evidence_package.json"

    if not replace:
        requested_outputs = [retrieval_config_path]
        if question:
            requested_outputs.append(smoke_output_path)
        if evaluate:
            requested_outputs.extend([evaluation_output_path, report_path])
        existing = [path for path in requested_outputs if path.exists()]
        if existing:
            raise RetrievalError(
                "Phase 14 derived output already exists. Use --replace: "
                + ", ".join(path.as_posix() for path in existing)
            )

    config_artifact: dict[str, object] = {
        "script_version": SCRIPT_VERSION,
        "retrieval_version": RETRIEVAL_VERSION,
        "generated_at": utc_now(),
        "status": (
            "retrieval_built_evaluation_pending"
            if not evaluate
            else "retrieval_built_and_evaluated"
        ),
        "corpus_version": corpus_version,
        "active_chunk_count": active_count,
        "domains": list(DOMAINS),
        "concepts": list(CONCEPTS),
        "embedding_identity": identity,
        "query_embedding_format": QUERY_PREFIX,
        "mapping": {
            "method": mapping.candidate.method,
            "prototype_source": mapping.candidate.source,
            "prototype_aggregation": mapping.candidate.aggregation,
            "prototype_version": mapping.prototype_version,
            "model_version": mapping.model_version,
            "activation_thresholds": mapping.activation.thresholds,
            "ambiguity_margin": mapping.activation.ambiguity_margin,
            "maximum_active_concepts": mapping.activation.max_active,
        },
        "retrieval_config": config.as_dict(),
        "concept_aware_scoring": {
            "vector_similarity": VECTOR_WEIGHT,
            "concept_alignment": CONCEPT_ALIGNMENT_WEIGHT,
            "human_relevance_label": HUMAN_RELEVANCE_WEIGHT,
            "citation_quality": CITATION_QUALITY_WEIGHT,
            "source_repeat_penalty_during_selection": (SOURCE_REPEAT_PENALTY),
        },
        "deduplication": {
            "same_chunk_once": True,
            "exact_normalized_text": True,
            "same_source_overlap_jaccard": (SAME_SOURCE_OVERLAP_JACCARD),
            "cross_source_duplicate_jaccard": (CROSS_SOURCE_DUPLICATE_JACCARD),
            "repeated_translation_policy": (
                "Remove only exact/near-duplicate cross-source text in "
                "Phase 1; preserve materially different translations."
            ),
        },
        "context_budget_policy": {
            "per_domain_token_budget": (config.token_budget_per_domain),
            "max_chunks_per_source_per_domain": (config.max_chunks_per_source),
            "one_domain_cannot_consume_another_domain_budget": True,
        },
        "evaluation_gate_frozen_before_results": {
            "max_recall_relative_loss": (MAX_ACCEPTABLE_RECALL_RELATIVE_LOSS),
            "precision_delta_min": MIN_PRECISION_DELTA,
            "concept_coverage_delta_min": (MIN_CONCEPT_COVERAGE_DELTA),
            "requires_at_least_one_quality_improvement": True,
        },
    }

    atomic_json(retrieval_config_path, config_artifact)

    if question:
        query_vector = embed_question(
            question=question,
            model=model,
            dimensions=dimensions,
            api_key=api_key,
        )
        activation = query_activation(
            question=question,
            query_vector=query_vector,
            mapping=mapping,
            bank=bank,
        )
        concept_retrieval = retrieve_all_domains(
            chunks=chunks,
            query_vector=query_vector,
            activation=activation,
            concept_aware=True,
            config=config,
        )
        package = evidence_package(
            question=question,
            activation=activation,
            retrieval=concept_retrieval,
            config=config,
            corpus_version=corpus_version,
            retrieval_mode="concept_aware",
            model_version=mapping.model_version,
            prototype_version=mapping.prototype_version,
        )
        atomic_json(smoke_output_path, package)

        LOGGER.info(
            "Question active concepts=%s unsupported=%s",
            list(activation.active_concepts),
            activation.unsupported,
        )
        for domain in DOMAINS:
            LOGGER.info(
                "%s evidence=%d sources=%d tokens=%d",
                domain,
                len(concept_retrieval[domain]),
                len({item.chunk.source_id for item in concept_retrieval[domain]}),
                sum(item.chunk.estimated_tokens for item in concept_retrieval[domain]),
            )
        LOGGER.info("Evidence package: %s", smoke_output_path)

    evaluation: dict[str, object] | None = None

    if evaluate:
        questions = load_evaluation_questions(evaluation_questions_path)
        evaluation = evaluate_retrieval(
            questions=questions,
            chunks=chunks,
            mapping=mapping,
            bank=bank,
            model=model,
            dimensions=dimensions,
            api_key=api_key,
            config=config,
            corpus_version=corpus_version,
        )
        evaluation["generated_at"] = utc_now()
        evaluation["retrieval_version"] = RETRIEVAL_VERSION
        evaluation["corpus_version"] = corpus_version
        evaluation["config"] = config.as_dict()
        evaluation["evaluation_questions_path"] = evaluation_questions_path.as_posix()
        atomic_json(evaluation_output_path, evaluation)

        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            markdown_report(
                evaluation=evaluation,
                config=config,
                corpus_version=corpus_version,
            ),
            encoding="utf-8",
        )

        gate = require_mapping(
            evaluation.get("retention_gate"),
            "retention gate",
        )
        LOGGER.info(
            "Retrieval evaluation complete concept_aware_retained=%s",
            gate.get("concept_aware_retained"),
        )
        LOGGER.info(
            "Evaluation JSON: %s",
            evaluation_output_path,
        )
        LOGGER.info("Evaluation report: %s", report_path)

    manifest: dict[str, object] = {
        "phase": "phase_14_build_retrieval_by_concept_and_domain",
        "status": (
            "evaluation_complete"
            if evaluation is not None
            else "retrieval_runtime_ready_evaluation_pending"
        ),
        "script_version": SCRIPT_VERSION,
        "retrieval_version": RETRIEVAL_VERSION,
        "generated_at": utc_now(),
        "corpus_version": corpus_version,
        "active_chunk_count": active_count,
        "retrieval_config": config.as_dict(),
        "outputs": {
            "retrieval_config": retrieval_config_path.as_posix(),
            "evidence_package": (smoke_output_path.as_posix() if question else None),
            "retrieval_evaluation_results": (
                evaluation_output_path.as_posix() if evaluate else None
            ),
            "retrieval_report": (report_path.as_posix() if evaluate else None),
        },
        "exit_gate": {
            "question_embedding_uses_frozen_model": True,
            "weighted_concept_activation_uses_frozen_phase10": True,
            "only_active_chunks_retrieved": True,
            "domain_separation_enforced": True,
            "source_diversity_enforced": True,
            "deduplication_enforced": True,
            "per_domain_context_budgets_enforced": True,
            "retrieval_evaluation_complete": evaluation is not None,
            "concept_aware_retained": (
                require_mapping(
                    evaluation.get("retention_gate"),
                    "retention gate",
                ).get("concept_aware_retained")
                if evaluation is not None
                else None
            ),
        },
        "next_step": (
            "If retrieval evaluation passes, freeze this retrieval configuration "
            "and begin Phase 15 domain-specific generation. If the dedicated "
            "frozen retrieval question file does not yet exist, freeze it before "
            "evaluating; do not use Phase 11 Held-out outcomes to tune retrieval."
        ),
    }

    manifest_path = output_directory / "retrieval_manifest.json"
    atomic_json(manifest_path, manifest)

    LOGGER.info("Phase 14 retrieval build complete")
    LOGGER.info("Active chunks loaded: %d", active_count)
    LOGGER.info("Retrieval manifest: %s", manifest_path)
    return manifest


def main() -> int:
    arguments = parse_arguments()
    configure_logging(arguments.log_level)

    config = RetrievalConfig(
        top_k_per_domain=arguments.top_k_per_domain,
        candidate_pool_per_domain=(arguments.candidate_pool_per_domain),
        token_budget_per_domain=(arguments.token_budget_per_domain),
        max_chunks_per_source=arguments.max_chunks_per_source,
        min_vector_similarity=arguments.min_vector_similarity,
    )

    try:
        run_phase14(
            project_root=arguments.project_root,
            active_bundles_path=arguments.active_bundles,
            activation_manifest_path=arguments.activation_manifest,
            phase10_results_path=arguments.phase10_results,
            query_prototypes_path=arguments.query_prototypes,
            passage_prototypes_path=arguments.passage_prototypes,
            embedding_manifest_path=arguments.embedding_manifest,
            evaluation_questions_path=arguments.evaluation_questions,
            output_directory=arguments.output_directory,
            report_path=arguments.report_path,
            question=arguments.question,
            evaluate=arguments.evaluate,
            config=config,
            replace=arguments.replace,
        )
    except RetrievalError:
        LOGGER.exception("Phase 14 retrieval failed")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
