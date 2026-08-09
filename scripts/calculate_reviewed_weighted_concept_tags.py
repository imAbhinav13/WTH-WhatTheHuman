from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from scripts import tune_phase1_concept_mapping as phase10

LOGGER = logging.getLogger("wth.phase1.calculate_reviewed_weighted_concept_tags")

SCRIPT_VERSION: Final = "1.0.0"
TAGGING_VERSION: Final = "phase1-reviewed-weighted-concept-tags-v1"

CONCEPTS: Final = (
    "consciousness",
    "self_identity",
    "reality_appearance",
)

VALID_LABELS: Final = {"positive", "partial", "negative"}
HUMAN_ACTIVE_LABELS: Final = {"positive", "partial"}
EXPECTED_APPROVED_CHUNKS: Final = 318

DEFAULT_GOLD_CORPUS: Final = Path("artifacts/phase1/reviewed/phase1_reviewed_gold_corpus.jsonl")
DEFAULT_REVIEW_MANIFEST: Final = Path("artifacts/phase1/reviewed/phase1_human_review_manifest.json")
DEFAULT_PHASE10_RESULTS: Final = Path(
    "artifacts/phase1/evaluation/concept_mapping_dev_results.json"
)
DEFAULT_PHASE11_RESULTS: Final = Path("artifacts/phase1/evaluation/heldout_results.json")
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
DEFAULT_OUTPUT_JSONL: Final = Path(
    "artifacts/phase1/concepts/phase1_reviewed_weighted_concept_tags.jsonl"
)
DEFAULT_OUTPUT_MANIFEST: Final = Path(
    "artifacts/phase1/concepts/phase1_reviewed_weighted_concept_tags_manifest.json"
)


class ProductionTagError(RuntimeError):
    """Raised when Phase 12 production tagging cannot safely complete."""


@dataclass(frozen=True)
class GoldChunk:
    chunk_id: str
    source_id: str
    domain: str
    reviewed_text: str
    labels: dict[str, str]
    hard_negative_for: tuple[str, ...]
    hard_negative_category: str
    primary_concept: str
    corpus_status: str

    @property
    def valid_roles(self) -> tuple[str, ...]:
        roles: list[str] = []

        for concept in CONCEPTS:
            if self.labels[concept] in HUMAN_ACTIVE_LABELS:
                roles.append(f"evidence:{concept}")

        for concept in self.hard_negative_for:
            if concept in CONCEPTS:
                roles.append(f"hard_negative:{concept}")

        return tuple(dict.fromkeys(roles))


@dataclass(frozen=True)
class FrozenMapping:
    candidate: phase10.Candidate
    activation: phase10.Activation
    calibrations: dict[str, phase10.Calibration]
    prototype_version: str
    mapping_method: str


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 12: calculate production concept weights for every approved "
            "Phase 1 chunk while keeping human-reviewed concept labels authoritative."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--gold-corpus", type=Path, default=DEFAULT_GOLD_CORPUS)
    parser.add_argument(
        "--review-manifest",
        type=Path,
        default=DEFAULT_REVIEW_MANIFEST,
    )
    parser.add_argument(
        "--phase10-results",
        type=Path,
        default=DEFAULT_PHASE10_RESULTS,
    )
    parser.add_argument(
        "--phase11-results",
        type=Path,
        default=DEFAULT_PHASE11_RESULTS,
    )
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
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=DEFAULT_OUTPUT_JSONL,
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=DEFAULT_OUTPUT_MANIFEST,
    )
    parser.add_argument(
        "--expected-chunks",
        type=int,
        default=EXPECTED_APPROVED_CHUNKS,
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing Phase 12 derived outputs.",
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
        raise ProductionTagError(f"Required file does not exist: {path}")


def require_mapping(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ProductionTagError(f"{description} must be an object.")

    result: dict[str, object] = {}
    for key, nested in value.items():
        if not isinstance(key, str):
            raise ProductionTagError(f"{description} contains a non-string key.")
        result[key] = nested

    return result


def require_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProductionTagError(f"{description} must be a non-empty string.")
    return value.strip()


def optional_string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def require_float(value: object, description: str) -> float:
    if not isinstance(value, int | float):
        raise ProductionTagError(f"{description} must be numeric.")
    return float(value)


def require_int(value: object, description: str) -> int:
    if not isinstance(value, int):
        raise ProductionTagError(f"{description} must be an integer.")
    return value


def parse_string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, list | tuple):
        return tuple(
            item.strip().casefold() for item in value if isinstance(item, str) and item.strip()
        )

    if isinstance(value, str) and value.strip():
        normalized = value.replace(";", "|").replace(",", "|")
        return tuple(item.strip().casefold() for item in normalized.split("|") if item.strip())

    return ()


def load_json(path: Path) -> dict[str, object]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProductionTagError(f"Invalid JSON in {path}: {exc}") from exc

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
                raise ProductionTagError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc

            yield require_mapping(
                raw,
                f"JSONL record {path}:{line_number}",
            )


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


def atomic_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")

    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
            handle.write("\n")

    temporary.replace(path)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def validate_review_manifest(
    path: Path,
    *,
    expected_chunks: int,
) -> dict[str, object]:
    manifest = load_json(path)

    if manifest.get("strict_gate_passed") is not True:
        raise ProductionTagError("Phase 5 human-review manifest did not pass its strict gate.")

    status = optional_string(manifest.get("status"))
    if status != "phase1_human_review_complete":
        raise ProductionTagError(f"Unexpected Phase 5 review status: {status!r}.")

    summary = require_mapping(
        manifest.get("summary"),
        "Phase 5 review summary",
    )
    approved_rows = require_int(
        summary.get("approved_rows"),
        "Phase 5 approved_rows",
    )

    if approved_rows != expected_chunks:
        raise ProductionTagError(
            "Human-review approved count does not match expected Phase 12 "
            f"chunk count: {approved_rows} != {expected_chunks}."
        )

    return manifest


def validate_phase11_complete(path: Path) -> dict[str, object]:
    result = load_json(path)

    if optional_string(result.get("status")) != "evaluation_complete":
        raise ProductionTagError("Phase 11 Held-out evaluation is not complete.")

    gate = require_mapping(
        result.get("exit_gate"),
        "Phase 11 exit_gate",
    )

    required_true = (
        "heldout_results_recorded",
        "no_post_hoc_threshold_changes",
        "no_post_hoc_calibration_changes",
        "no_post_hoc_model_reselection",
        "failures_and_limitations_documented",
        "selected_method_justification_recorded",
    )

    for field_name in required_true:
        if gate.get(field_name) is not True:
            raise ProductionTagError(f"Phase 11 exit gate failed: {field_name}.")

    return result


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

    thresholds = {
        concept: require_float(
            threshold_raw.get(concept),
            f"{concept} activation threshold",
        )
        for concept in CONCEPTS
    }

    return phase10.Activation(
        thresholds=thresholds,
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
        concept_raw = require_mapping(
            raw.get(concept),
            f"{concept} calibration",
        )
        result[concept] = phase10.Calibration(
            slope=require_float(
                concept_raw.get("slope"),
                f"{concept} slope",
            ),
            intercept=require_float(
                concept_raw.get("intercept"),
                f"{concept} intercept",
            ),
        )

    return result


def validate_phase10_frozen(
    path: Path,
) -> tuple[dict[str, object], FrozenMapping]:
    result = load_json(path)

    if optional_string(result.get("status")) != "frozen":
        raise ProductionTagError("Phase 10 concept mapping is not frozen.")

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
            raise ProductionTagError(f"Phase 10 exit gate failed: {field_name}.")

    frozen = require_mapping(
        result.get("frozen_parameters"),
        "Phase 10 frozen_parameters",
    )
    calibration_raw = require_mapping(
        frozen.get("calibration"),
        "Phase 10 calibration",
    )

    inputs = require_mapping(result.get("inputs"), "Phase 10 inputs")
    prototype_meta = require_mapping(
        inputs.get("prototype_artifact"),
        "Phase 10 prototype_artifact",
    )

    prototype_version = require_string(
        prototype_meta.get("prototype_version"),
        "prototype_version",
    )
    if optional_string(prototype_meta.get("status")) != "frozen":
        raise ProductionTagError("Phase 10 does not reference a frozen prototype artifact.")

    candidate = candidate_from_mapping(frozen)
    activation = activation_from_mapping(frozen)
    calibrations = calibrations_from_mapping(calibration_raw)

    mapping_method = f"{candidate.method}:{candidate.source}:{candidate.aggregation}"

    return (
        result,
        FrozenMapping(
            candidate=candidate,
            activation=activation,
            calibrations=calibrations,
            prototype_version=prototype_version,
            mapping_method=mapping_method,
        ),
    )


def load_gold_chunks(
    path: Path,
    *,
    expected_chunks: int,
) -> list[GoldChunk]:
    chunks: list[GoldChunk] = []

    for index, raw in enumerate(iter_jsonl(path), start=1):
        chunk_id = require_string(
            raw.get("chunk_id"),
            f"gold record {index} chunk_id",
        )
        source_id = require_string(
            raw.get("source_id"),
            f"{chunk_id} source_id",
        )
        domain = require_string(
            raw.get("domain"),
            f"{chunk_id} domain",
        ).casefold()
        reviewed_text = require_string(
            raw.get("reviewed_text"),
            f"{chunk_id} reviewed_text",
        )

        review = require_mapping(
            raw.get("review"),
            f"{chunk_id} review",
        )
        labels_raw = require_mapping(
            review.get("labels"),
            f"{chunk_id} review.labels",
        )

        labels: dict[str, str] = {}
        for concept in CONCEPTS:
            label = require_string(
                labels_raw.get(concept),
                f"{chunk_id} label {concept}",
            ).casefold()

            if label not in VALID_LABELS:
                raise ProductionTagError(
                    f"{chunk_id} has non-production label {concept}={label!r}."
                )
            labels[concept] = label

        chunk = GoldChunk(
            chunk_id=chunk_id,
            source_id=source_id,
            domain=domain,
            reviewed_text=reviewed_text,
            labels=labels,
            hard_negative_for=parse_string_tuple(review.get("hard_negative_for")),
            hard_negative_category=optional_string(review.get("hard_negative_category")).casefold(),
            primary_concept=optional_string(review.get("primary_concept")).casefold(),
            corpus_status=optional_string(raw.get("corpus_status")),
        )

        if not chunk.valid_roles:
            raise ProductionTagError(
                f"{chunk_id} has no valid Phase 1 role. An approved chunk must "
                "have at least one positive/partial reviewed concept label or "
                "an explicit reviewed hard-negative role."
            )

        chunks.append(chunk)

    if len(chunks) != expected_chunks:
        raise ProductionTagError(
            f"Expected {expected_chunks} approved chunks, found {len(chunks)}."
        )

    chunk_ids = [chunk.chunk_id for chunk in chunks]
    duplicate_ids = sorted(chunk_id for chunk_id, count in Counter(chunk_ids).items() if count > 1)
    if duplicate_ids:
        raise ProductionTagError("Duplicate approved chunk IDs: " + ", ".join(duplicate_ids))

    return chunks


def similarity_candidate_for(
    selected: phase10.Candidate,
) -> phase10.Candidate:
    if selected.aggregation == "centroid":
        method = "centroid"
    elif selected.aggregation == "maximum":
        method = "maximum_example"
    else:
        raise ProductionTagError(
            f"Cannot derive raw prototype similarity for aggregation {selected.aggregation!r}."
        )

    return phase10.Candidate(
        method=method,
        source=selected.source,
        aggregation=selected.aggregation,
        negative_penalty=0.0,
        lexical_weight=0.0,
        embed_weight=1.0,
    )


def model_version_string(identity: object) -> str:
    identity_mapping = require_mapping(
        phase10_identity_as_dict(identity),
        "embedding identity",
    )
    model = require_string(identity_mapping.get("model"), "embedding model")
    revision = require_string(
        identity_mapping.get("model_revision"),
        "embedding model_revision",
    )
    return f"{model}@{revision}"


def phase10_identity_as_dict(identity: object) -> dict[str, object]:
    as_dict = getattr(identity, "as_dict", None)
    if not callable(as_dict):
        raise ProductionTagError("Embedding identity object does not expose as_dict().")
    raw = as_dict()
    return require_mapping(raw, "embedding identity")


def human_label_is_active(label: str) -> bool:
    return label in HUMAN_ACTIVE_LABELS


def review_status_for_label(label: str) -> str:
    if label in VALID_LABELS:
        return "human_reviewed_authoritative"
    raise ProductionTagError(f"Unexpected human label {label!r}.")


def make_phase10_record(chunk: GoldChunk) -> phase10.LabeledRecord:
    return phase10.LabeledRecord(
        chunk_id=chunk.chunk_id,
        split="production",
        text=chunk.reviewed_text,
        labels=chunk.labels,
        hard_negative_for=chunk.hard_negative_for,
    )


def calculate_tags(
    *,
    chunks: Sequence[GoldChunk],
    mapping: FrozenMapping,
    embeddings: Mapping[str, phase10.FloatArray],
    bank: phase10.PrototypeBank,
    model_version: str,
    embedding_identity: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    similarity_candidate = similarity_candidate_for(mapping.candidate)

    rows: list[dict[str, object]] = []
    override_counts: Counter[str] = Counter()
    label_counts: dict[str, Counter[str]] = {concept: Counter() for concept in CONCEPTS}
    role_counts: Counter[str] = Counter()

    for chunk in sorted(chunks, key=lambda item: item.chunk_id):
        record = make_phase10_record(chunk)
        vector = embeddings[chunk.chunk_id]

        raw_similarity = phase10.raw_scores(
            similarity_candidate,
            record,
            vector,
            bank,
            {},
        )
        raw_mapping_score = phase10.raw_scores(
            mapping.candidate,
            record,
            vector,
            bank,
            {},
        )
        calibrated_weights = phase10.apply_calibration(
            raw_mapping_score,
            mapping.calibrations,
        )

        automated_active = set(
            phase10.predicted_active(
                calibrated_weights,
                mapping.activation,
            )
        )

        for role in chunk.valid_roles:
            role_counts[role] += 1

        for concept in CONCEPTS:
            human_label = chunk.labels[concept]
            human_active = human_label_is_active(human_label)
            model_active = concept in automated_active
            human_override = human_active != model_active

            if human_override:
                override_counts[concept] += 1

            label_counts[concept][human_label] += 1

            rows.append(
                {
                    # Required Phase 12 contract.
                    "chunk_id": chunk.chunk_id,
                    "concept_id": concept,
                    "raw_similarity": float(raw_similarity[concept]),
                    "calibrated_weight": float(calibrated_weights[concept]),
                    "mapping_method": mapping.mapping_method,
                    "prototype_version": mapping.prototype_version,
                    "model_version": model_version,
                    "human_label": human_label,
                    "human_override": human_override,
                    "review_status": review_status_for_label(human_label),
                    # Explicit production semantics and provenance.
                    "production_active": human_active,
                    "automated_active": model_active,
                    "activation_threshold": float(mapping.activation.thresholds[concept]),
                    "raw_mapping_score": float(raw_mapping_score[concept]),
                    "source_id": chunk.source_id,
                    "domain": chunk.domain,
                    "primary_concept": chunk.primary_concept,
                    "hard_negative_for": list(chunk.hard_negative_for),
                    "hard_negative_category": (chunk.hard_negative_category),
                    "phase1_roles": list(chunk.valid_roles),
                    "corpus_status": chunk.corpus_status,
                    "embedding_provider": optional_string(embedding_identity.get("provider")),
                    "embedding_dimensions": require_int(
                        embedding_identity.get("dimensions"),
                        "embedding dimensions",
                    ),
                    "embedding_normalization": optional_string(
                        embedding_identity.get("normalization")
                    ),
                }
            )

    diagnostics: dict[str, object] = {
        "tag_count": len(rows),
        "expected_tag_count": len(chunks) * len(CONCEPTS),
        "human_override_count": sum(override_counts.values()),
        "human_override_count_by_concept": dict(sorted(override_counts.items())),
        "label_distribution": {
            concept: dict(sorted(counts.items())) for concept, counts in label_counts.items()
        },
        "role_distribution": dict(sorted(role_counts.items())),
    }

    return rows, diagnostics


def validate_tag_output(
    *,
    chunks: Sequence[GoldChunk],
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    expected_count = len(chunks) * len(CONCEPTS)
    if len(rows) != expected_count:
        raise ProductionTagError(
            f"Expected {expected_count} chunk-concept tags, found {len(rows)}."
        )

    by_chunk: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        chunk_id = require_string(row.get("chunk_id"), "output chunk_id")
        by_chunk.setdefault(chunk_id, []).append(row)

    missing_chunks = sorted(chunk.chunk_id for chunk in chunks if chunk.chunk_id not in by_chunk)
    if missing_chunks:
        raise ProductionTagError("Missing production tags for chunks: " + ", ".join(missing_chunks))

    for chunk in chunks:
        chunk_rows = by_chunk[chunk.chunk_id]

        if len(chunk_rows) != len(CONCEPTS):
            raise ProductionTagError(
                f"{chunk.chunk_id} does not have exactly {len(CONCEPTS)} concept tag rows."
            )

        concepts = {
            require_string(row.get("concept_id"), "output concept_id") for row in chunk_rows
        }
        if concepts != set(CONCEPTS):
            raise ProductionTagError(
                f"{chunk.chunk_id} output concept set is invalid: {sorted(concepts)}."
            )

        if not chunk.valid_roles:
            raise ProductionTagError(f"{chunk.chunk_id} has no valid Phase 1 role.")

        for row in chunk_rows:
            weight = require_float(
                row.get("calibrated_weight"),
                "calibrated_weight",
            )
            if not 0.0 <= weight <= 1.0:
                raise ProductionTagError(f"{chunk.chunk_id} has calibrated_weight outside [0, 1].")

            if optional_string(row.get("review_status")) != "human_reviewed_authoritative":
                raise ProductionTagError(f"{chunk.chunk_id} has invalid review_status.")

            human_label = require_string(
                row.get("human_label"),
                "human_label",
            )
            production_active = row.get("production_active")
            expected_active = human_label_is_active(human_label)
            if production_active is not expected_active:
                raise ProductionTagError(
                    f"{chunk.chunk_id}/{row.get('concept_id')} silently "
                    "overrode the authoritative human label."
                )

    return {
        "all_approved_chunks_present": True,
        "exactly_three_concept_rows_per_chunk": True,
        "all_chunks_have_valid_phase1_role": True,
        "all_reviewed_labels_present": True,
        "all_calibrated_weights_in_0_1": True,
        "human_labels_authoritative": True,
        "full_model_provenance_present": True,
        "full_prototype_provenance_present": True,
    }


def ensure_output_policy(
    paths: Sequence[Path],
    *,
    replace: bool,
) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not replace:
        raise ProductionTagError(
            "Phase 12 outputs already exist. Use --replace to regenerate the "
            "derived tags from the same frozen inputs: "
            + ", ".join(path.as_posix() for path in existing)
        )


def run_phase12(
    *,
    project_root: Path,
    gold_corpus_path: Path,
    review_manifest_path: Path,
    phase10_results_path: Path,
    phase11_results_path: Path,
    approved_embeddings_path: Path,
    query_prototype_embeddings_path: Path,
    passage_prototype_embeddings_path: Path,
    embedding_manifest_path: Path,
    output_jsonl_path: Path,
    output_manifest_path: Path,
    expected_chunks: int,
    replace: bool,
) -> dict[str, object]:
    project_root = project_root.resolve()

    gold_corpus_path = resolve(project_root, gold_corpus_path)
    review_manifest_path = resolve(project_root, review_manifest_path)
    phase10_results_path = resolve(project_root, phase10_results_path)
    phase11_results_path = resolve(project_root, phase11_results_path)
    approved_embeddings_path = resolve(
        project_root,
        approved_embeddings_path,
    )
    query_prototype_embeddings_path = resolve(
        project_root,
        query_prototype_embeddings_path,
    )
    passage_prototype_embeddings_path = resolve(
        project_root,
        passage_prototype_embeddings_path,
    )
    embedding_manifest_path = resolve(
        project_root,
        embedding_manifest_path,
    )
    output_jsonl_path = resolve(project_root, output_jsonl_path)
    output_manifest_path = resolve(project_root, output_manifest_path)

    for path in (
        gold_corpus_path,
        review_manifest_path,
        phase10_results_path,
        phase11_results_path,
        approved_embeddings_path,
        query_prototype_embeddings_path,
        passage_prototype_embeddings_path,
        embedding_manifest_path,
    ):
        require_file(path)

    ensure_output_policy(
        (output_jsonl_path, output_manifest_path),
        replace=replace,
    )

    LOGGER.info(
        "Phase 12 starting: expected approved chunks=%d",
        expected_chunks,
    )

    review_manifest = validate_review_manifest(
        review_manifest_path,
        expected_chunks=expected_chunks,
    )
    phase11_result = validate_phase11_complete(phase11_results_path)
    phase10_result, mapping = validate_phase10_frozen(phase10_results_path)

    chunks = load_gold_chunks(
        gold_corpus_path,
        expected_chunks=expected_chunks,
    )

    identity, embedding_manifest = phase10.load_identity(embedding_manifest_path)
    identity_dict = phase10_identity_as_dict(identity)
    model_version = model_version_string(identity)

    embeddings = phase10.load_needed_embeddings(
        approved_embeddings_path,
        {chunk.chunk_id for chunk in chunks},
        identity,
    )
    bank = phase10.load_prototype_bank(
        query_prototype_embeddings_path,
        passage_prototype_embeddings_path,
        identity,
    )

    rows, diagnostics = calculate_tags(
        chunks=chunks,
        mapping=mapping,
        embeddings=embeddings,
        bank=bank,
        model_version=model_version,
        embedding_identity=identity_dict,
    )

    exit_gate = validate_tag_output(
        chunks=chunks,
        rows=rows,
    )

    atomic_jsonl(output_jsonl_path, rows)

    manifest: dict[str, object] = {
        "script_version": SCRIPT_VERSION,
        "tagging_version": TAGGING_VERSION,
        "generated_at": utc_now(),
        "status": "production_concept_weights_complete",
        "phase": "phase_12_reviewed_weighted_concept_tags",
        "policy": {
            "human_reviewed_labels_authoritative": True,
            "automated_weights_support_ranking": True,
            "automated_activation_may_not_overwrite_gold_label": True,
            "production_active_derived_from_human_label": True,
            "human_override_definition": (
                "True when frozen automated threshold activation disagrees "
                "with the authoritative reviewed positive/partial versus "
                "negative label."
            ),
            "raw_similarity_definition": (
                "Embedding-only similarity using the selected Phase 10 "
                "prototype source and aggregation, before lexical or "
                "hard-negative additions."
            ),
            "raw_mapping_score_definition": (
                "Frozen Phase 10 pre-calibration hybrid score used to produce calibrated_weight."
            ),
        },
        "inputs": {
            "gold_corpus": {
                "path": gold_corpus_path.as_posix(),
                "sha256": phase10.sha256_jsonl(gold_corpus_path),
                "chunk_count": len(chunks),
            },
            "review_manifest": {
                "path": review_manifest_path.as_posix(),
                "sha256_file_bytes": phase10.sha256_file(review_manifest_path),
                "status": optional_string(review_manifest.get("status")),
            },
            "phase10_results": {
                "path": phase10_results_path.as_posix(),
                "sha256_file_bytes": phase10.sha256_file(phase10_results_path),
                "status": optional_string(phase10_result.get("status")),
            },
            "phase11_results": {
                "path": phase11_results_path.as_posix(),
                "sha256_file_bytes": phase10.sha256_file(phase11_results_path),
                "status": optional_string(phase11_result.get("status")),
            },
            "approved_embeddings": {
                "path": approved_embeddings_path.as_posix(),
                "sha256": phase10.sha256_jsonl(approved_embeddings_path),
            },
            "query_prototype_embeddings": {
                "path": query_prototype_embeddings_path.as_posix(),
                "sha256": phase10.sha256_jsonl(query_prototype_embeddings_path),
            },
            "passage_prototype_embeddings": {
                "path": passage_prototype_embeddings_path.as_posix(),
                "sha256": phase10.sha256_jsonl(passage_prototype_embeddings_path),
            },
            "embedding_manifest": {
                "path": embedding_manifest_path.as_posix(),
                "sha256_file_bytes": phase10.sha256_file(embedding_manifest_path),
                "status": optional_string(embedding_manifest.get("status")),
            },
        },
        "mapping_provenance": {
            "mapping_method": mapping.mapping_method,
            "candidate": mapping.candidate.as_dict(),
            "activation": mapping.activation.as_dict(),
            "calibration": {
                concept: {
                    "slope": calibration.slope,
                    "intercept": calibration.intercept,
                    "type": "sigmoid",
                }
                for concept, calibration in mapping.calibrations.items()
            },
            "prototype_version": mapping.prototype_version,
            "model_version": model_version,
            "embedding_identity": identity_dict,
        },
        "counts": {
            "approved_chunks": len(chunks),
            **diagnostics,
        },
        "output": {
            "path": output_jsonl_path.as_posix(),
            "sha256": phase10.sha256_jsonl(output_jsonl_path),
        },
        "exit_gate": exit_gate,
        "next_step": (
            "Phase 13: activate only the reviewed approved Phase 1 corpus. "
            "Use production_active as the authoritative concept relation and "
            "calibrated_weight as a ranking signal; preserve human_override "
            "and all model/prototype provenance."
        ),
    }

    atomic_json(output_manifest_path, manifest)

    LOGGER.info("Phase 12 production tagging complete")
    LOGGER.info("Approved chunks: %d", len(chunks))
    LOGGER.info("Concept tag rows: %d", len(rows))
    LOGGER.info(
        "Human overrides recorded: %d",
        require_int(
            diagnostics.get("human_override_count"),
            "human_override_count",
        ),
    )
    LOGGER.info("Model version: %s", model_version)
    LOGGER.info("Prototype version: %s", mapping.prototype_version)
    LOGGER.info("JSONL: %s", output_jsonl_path)
    LOGGER.info("Manifest: %s", output_manifest_path)

    return manifest


def main() -> int:
    arguments = parse_arguments()
    configure_logging(arguments.log_level)

    try:
        run_phase12(
            project_root=arguments.project_root,
            gold_corpus_path=arguments.gold_corpus,
            review_manifest_path=arguments.review_manifest,
            phase10_results_path=arguments.phase10_results,
            phase11_results_path=arguments.phase11_results,
            approved_embeddings_path=arguments.approved_embeddings,
            query_prototype_embeddings_path=(arguments.query_prototype_embeddings),
            passage_prototype_embeddings_path=(arguments.passage_prototype_embeddings),
            embedding_manifest_path=arguments.embedding_manifest,
            output_jsonl_path=arguments.output_jsonl,
            output_manifest_path=arguments.output_manifest,
            expected_chunks=arguments.expected_chunks,
            replace=arguments.replace,
        )
    except ProductionTagError:
        LOGGER.exception("Phase 12 production tagging failed")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
