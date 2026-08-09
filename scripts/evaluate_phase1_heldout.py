from __future__ import annotations

import argparse
import csv
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
from scripts import tune_phase1_concept_mapping as phase10

LOGGER = logging.getLogger("wth.phase1.evaluate_phase1_heldout")

SCRIPT_VERSION: Final = "1.0.0"
EVALUATION_VERSION: Final = "phase1-heldout-evaluation-v1"

DEFAULT_PHASE10_RESULTS: Final = Path(
    "artifacts/phase1/evaluation/concept_mapping_dev_results.json"
)
DEFAULT_SPLIT_MANIFEST: Final = Path("data/evaluation/phase1_split_manifest.json")
DEFAULT_BUILD_SET: Final = Path("data/evaluation/phase1_build.jsonl")
DEFAULT_HELDOUT_SET: Final = Path("data/evaluation/phase1_heldout.jsonl")
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
DEFAULT_OUTPUT_JSON: Final = Path("artifacts/phase1/evaluation/heldout_results.json")
DEFAULT_ERROR_CSV: Final = Path("artifacts/phase1/evaluation/error_analysis.csv")
DEFAULT_REPORT: Final = Path("docs/evaluation/phase1_heldout_report.md")

EXPECTED_BUILD_COUNT: Final = 159
EXPECTED_HELDOUT_COUNT: Final = 79
ECE_BINS: Final = 10

DOMAIN_ORDER: Final = ("science", "advaita", "samkhya")

HARD_NEGATIVE_REPORT_GROUPS: Final[dict[str, dict[str, object]]] = {
    "attention_mistaken_for_consciousness": {
        "target_concept": "consciousness",
        "categories": ("consciousness_vs_attention",),
    },
    "ego_mistaken_for_self": {
        "target_concept": "self_identity",
        "categories": (
            "self_vs_ego",
            "self_identity_vs_ego",
        ),
    },
    "cosmology_mistaken_for_reality_appearance": {
        "target_concept": "reality_appearance",
        "categories": ("reality_appearance_vs_cosmology",),
    },
    "purusha_collapsed_into_atman": {
        "target_concept": "self_identity",
        "categories": ("advaita_atman_vs_samkhya_purusha",),
    },
    "perception_description_mistaken_for_metaphysical_appearance": {
        "target_concept": "reality_appearance",
        "categories": (
            "reality_appearance_vs_description",
            "reality_appearance_vs_perceptual_description",
        ),
    },
}

EXTRA_HARD_NEGATIVE_GROUPS: Final[dict[str, dict[str, object]]] = {
    "cognition_mistaken_for_consciousness": {
        "target_concept": "consciousness",
        "categories": ("consciousness_vs_cognition",),
    },
    "personality_mistaken_for_self": {
        "target_concept": "self_identity",
        "categories": (
            "self_vs_personality",
            "self_identity_vs_personality",
        ),
    },
}

CONFUSION_PAIRS: Final = (
    ("consciousness", "self_identity"),
    ("self_identity", "reality_appearance"),
    ("consciousness", "reality_appearance"),
)


class HeldoutEvaluationError(RuntimeError):
    """Raised when Phase 11 cannot proceed without violating the freeze."""


@dataclass(frozen=True)
class HeldoutRecord:
    chunk_id: str
    source_id: str
    domain: str
    text: str
    labels: dict[str, str]
    hard_negative_for: tuple[str, ...]
    hard_negative_category: str


@dataclass(frozen=True)
class FrozenSystem:
    system_id: str
    display_name: str
    description: str
    candidate: phase10.Candidate
    activation: phase10.Activation
    calibrations: dict[str, phase10.Calibration]
    selected_method: bool


@dataclass(frozen=True)
class ScoredRecord:
    record: HeldoutRecord
    raw_scores: dict[str, float]
    calibrated_weights: dict[str, float]
    active_concepts: tuple[str, ...]
    ambiguous: bool
    unsupported: bool


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the frozen Phase 10 concept mapper exactly once on the "
            "untouched Phase 1 Held-out split. No tuning is performed."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--phase10-results",
        type=Path,
        default=DEFAULT_PHASE10_RESULTS,
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST,
    )
    parser.add_argument("--build-set", type=Path, default=DEFAULT_BUILD_SET)
    parser.add_argument("--heldout-set", type=Path, default=DEFAULT_HELDOUT_SET)
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
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--error-csv", type=Path, default=DEFAULT_ERROR_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing Phase 11 outputs. This never changes parameters.",
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
        raise HeldoutEvaluationError(f"Required file does not exist: {path}")


def require_mapping(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise HeldoutEvaluationError(f"{description} must be an object.")

    result: dict[str, object] = {}
    for key, nested in value.items():
        if not isinstance(key, str):
            raise HeldoutEvaluationError(f"{description} contains a non-string key.")
        result[key] = nested
    return result


def require_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HeldoutEvaluationError(f"{description} must be a non-empty string.")
    return value.strip()


def require_float(value: object, description: str) -> float:
    if not isinstance(value, int | float):
        raise HeldoutEvaluationError(f"{description} must be numeric.")
    return float(value)


def require_int(value: object, description: str) -> int:
    if not isinstance(value, int):
        raise HeldoutEvaluationError(f"{description} must be an integer.")
    return value


def optional_string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def calibration_ece(
    metrics: Mapping[str, object],
    description: str,
) -> float | None:
    raw = metrics.get("calibration_error")

    if raw is None:
        return None

    if isinstance(raw, int | float):
        return float(raw)

    if isinstance(raw, Mapping):
        calibration = require_mapping(
            raw,
            description,
        )

        value = calibration.get("macro_soft_ece_10_bins")

        if value is None:
            return None

        return require_float(
            value,
            f"{description} macro_soft_ece_10_bins",
        )

    raise HeldoutEvaluationError(f"{description} has unsupported calibration_error format.")


def parse_string_list(value: object) -> tuple[str, ...]:
    if isinstance(value, list):
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
        raise HeldoutEvaluationError(f"Invalid JSON in {path}: {exc}") from exc

    return require_mapping(raw, f"JSON {path}")


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


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def validate_phase10_freeze(
    path: Path,
) -> dict[str, object]:
    result = load_json(path)

    if optional_string(result.get("status")) != "frozen":
        raise HeldoutEvaluationError(
            "Phase 10 result is not frozen. Held-out evaluation is prohibited."
        )

    gate = require_mapping(result.get("exit_gate"), "Phase 10 exit_gate")

    required_true = (
        "quality_gate_passed",
        "thresholds_frozen",
        "parameters_frozen_before_heldout",
        "plural_activation_supported",
        "raw_scores_preserved",
        "calibrated_0_1_weights_produced",
    )
    for field_name in required_true:
        if gate.get(field_name) is not True:
            raise HeldoutEvaluationError(f"Phase 10 freeze gate failed: {field_name}.")

    if gate.get("heldout_used_for_tuning") is not False:
        raise HeldoutEvaluationError("Phase 10 indicates Held-out may have influenced tuning.")

    return result


def validate_split_manifest(
    path: Path,
    heldout_path: Path,
) -> dict[str, object]:
    manifest = load_json(path)

    if optional_string(manifest.get("status")) != "frozen":
        raise HeldoutEvaluationError("Evaluation split manifest is not frozen.")

    gate = require_mapping(
        manifest.get("exit_gate"),
        "split manifest exit_gate",
    )

    for field_name in (
        "splits_checksummed",
        "heldout_marked_read_only",
        "distribution_report_generated",
    ):
        if gate.get(field_name) is not True:
            raise HeldoutEvaluationError(f"Split manifest gate failed: {field_name}.")

    outputs = require_mapping(manifest.get("outputs"), "split outputs")
    heldout_meta = require_mapping(outputs.get("heldout"), "heldout output")

    expected_count = require_int(
        heldout_meta.get("record_count"),
        "heldout record_count",
    )
    if expected_count != EXPECTED_HELDOUT_COUNT:
        raise HeldoutEvaluationError(
            f"Expected manifest Held-out count {EXPECTED_HELDOUT_COUNT}, found {expected_count}."
        )

    if heldout_meta.get("read_only") is not True:
        raise HeldoutEvaluationError("Split manifest does not mark Held-out read-only.")

    expected_sha = require_string(
        heldout_meta.get("sha256"),
        "heldout sha256",
    )
    actual_sha = phase10.sha256_jsonl(heldout_path)

    if actual_sha != expected_sha:
        raise HeldoutEvaluationError(
            "Held-out checksum mismatch. Refusing evaluation because the "
            "frozen test set has changed."
        )

    return manifest


def load_heldout_records(path: Path) -> dict[str, HeldoutRecord]:
    records: dict[str, HeldoutRecord] = {}

    for raw in phase10.iter_jsonl(path):
        chunk_id = require_string(raw.get("chunk_id"), "chunk_id")
        split = require_string(
            raw.get("evaluation_split"),
            f"{chunk_id} evaluation_split",
        )
        if split != "heldout":
            raise HeldoutEvaluationError(f"{chunk_id} belongs to {split!r}, expected 'heldout'.")

        source_id = require_string(raw.get("source_id"), f"{chunk_id} source_id")
        domain = require_string(raw.get("domain"), f"{chunk_id} domain").casefold()
        if domain not in DOMAIN_ORDER:
            raise HeldoutEvaluationError(f"{chunk_id} has unsupported domain {domain!r}.")

        review = require_mapping(raw.get("review"), f"{chunk_id} review")
        labels_raw = require_mapping(
            review.get("labels"),
            f"{chunk_id} review.labels",
        )

        labels: dict[str, str] = {}
        for concept in phase10.CONCEPTS:
            label = require_string(
                labels_raw.get(concept),
                f"{chunk_id} label {concept}",
            ).casefold()
            if label not in phase10.LABEL_TARGETS:
                raise HeldoutEvaluationError(
                    f"{chunk_id} has unsupported {concept} label {label!r}."
                )
            labels[concept] = label

        records[chunk_id] = HeldoutRecord(
            chunk_id=chunk_id,
            source_id=source_id,
            domain=domain,
            text=require_string(
                raw.get("reviewed_text"),
                f"{chunk_id} reviewed_text",
            ),
            labels=labels,
            hard_negative_for=parse_string_list(review.get("hard_negative_for")),
            hard_negative_category=optional_string(review.get("hard_negative_category")).casefold(),
        )

    if len(records) != EXPECTED_HELDOUT_COUNT:
        raise HeldoutEvaluationError(
            f"Expected {EXPECTED_HELDOUT_COUNT} Held-out records, found {len(records)}."
        )

    return records


def candidate_from_mapping(
    raw: Mapping[str, object],
) -> phase10.Candidate:
    return phase10.Candidate(
        method=require_string(raw.get("method"), "candidate method"),
        source=require_string(
            raw.get("prototype_source"),
            "candidate prototype_source",
        ),
        aggregation=require_string(
            raw.get("prototype_aggregation"),
            "candidate prototype_aggregation",
        ),
        negative_penalty=require_float(
            raw.get("negative_penalty", 0.0),
            "candidate negative_penalty",
        ),
        lexical_weight=require_float(
            raw.get("lexical_weight", 0.0),
            "candidate lexical_weight",
        ),
        embed_weight=require_float(
            raw.get("embedding_weight", 1.0),
            "candidate embedding_weight",
        ),
    )


def activation_from_mapping(
    raw: Mapping[str, object],
) -> phase10.Activation:
    thresholds_raw = require_mapping(
        raw.get("concept_activation_thresholds"),
        "concept activation thresholds",
    )

    thresholds = {
        concept: require_float(
            thresholds_raw.get(concept),
            f"threshold {concept}",
        )
        for concept in phase10.CONCEPTS
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

    for concept in phase10.CONCEPTS:
        concept_raw = require_mapping(
            raw.get(concept),
            f"calibration {concept}",
        )
        result[concept] = phase10.Calibration(
            slope=require_float(
                concept_raw.get("slope"),
                f"{concept} calibration slope",
            ),
            intercept=require_float(
                concept_raw.get("intercept"),
                f"{concept} calibration intercept",
            ),
        )

    return result


def system_from_result(
    *,
    system_id: str,
    display_name: str,
    description: str,
    raw: Mapping[str, object],
    selected: bool,
) -> FrozenSystem:
    if selected:
        candidate_raw = raw
        activation_raw = raw
        calibration_raw = require_mapping(
            raw.get("calibration"),
            f"{system_id} calibration",
        )
    else:
        candidate_raw = require_mapping(
            raw.get("candidate"),
            f"{system_id} candidate",
        )
        activation_raw = require_mapping(
            raw.get("activation"),
            f"{system_id} activation",
        )
        calibration_raw = require_mapping(
            raw.get("calibration"),
            f"{system_id} calibration",
        )

    return FrozenSystem(
        system_id=system_id,
        display_name=display_name,
        description=description,
        candidate=candidate_from_mapping(candidate_raw),
        activation=activation_from_mapping(activation_raw),
        calibrations=calibrations_from_mapping(calibration_raw),
        selected_method=selected,
    )


def build_frozen_systems(
    phase10_result: Mapping[str, object],
) -> list[FrozenSystem]:
    frozen = require_mapping(
        phase10_result.get("frozen_parameters"),
        "Phase 10 frozen_parameters",
    )
    best_by_method = require_mapping(
        phase10_result.get("best_result_by_method"),
        "Phase 10 best_result_by_method",
    )

    systems = [
        system_from_result(
            system_id="selected_hybrid",
            display_name="Selected — Frozen Hybrid",
            description=(
                "Phase 10 selected hybrid: embedding similarity + transparent "
                "lexical indicators + hard-negative penalty."
            ),
            raw=frozen,
            selected=True,
        ),
        system_from_result(
            system_id="baseline_a_plain_embedding_similarity",
            display_name="Baseline A — Plain embedding similarity",
            description=(
                "Operationalized as the Phase 10 maximum-example method: "
                "strongest direct cosine similarity to a frozen concept example, "
                "without lexical or hard-negative terms."
            ),
            raw=require_mapping(
                best_by_method.get("maximum_example"),
                "maximum_example baseline",
            ),
            selected=False,
        ),
        system_from_result(
            system_id="baseline_b_prototype_mapping",
            display_name="Baseline B — Prototype centroid",
            description=(
                "Prototype-based concept mapping using frozen query prototype "
                "centroids without hard-negative penalties."
            ),
            raw=require_mapping(
                best_by_method.get("centroid"),
                "centroid baseline",
            ),
            selected=False,
        ),
        system_from_result(
            system_id="baseline_c_prototype_plus_hard_negative",
            display_name="Baseline C — Prototype + hard-negative",
            description=(
                "Prototype mapping with a frozen positive-minus-hard-negative "
                "score and no lexical feature."
            ),
            raw=require_mapping(
                best_by_method.get("positive_minus_negative"),
                "positive_minus_negative baseline",
            ),
            selected=False,
        ),
        system_from_result(
            system_id="baseline_d_supervised_classifier",
            display_name="Baseline D — Build-trained classifier",
            description=(
                "Lightweight soft logistic classifier trained only on Build, "
                "using its Phase 10 Development-frozen calibration and thresholds."
            ),
            raw=require_mapping(
                best_by_method.get("classifier"),
                "classifier baseline",
            ),
            selected=False,
        ),
    ]

    return systems


def label_is_active(label: str) -> bool:
    return label in {"positive", "partial"}


def soft_target(label: str) -> float:
    return float(phase10.LABEL_TARGETS[label])


def precision_recall_f1(
    truth: Sequence[bool],
    predicted: Sequence[bool],
) -> tuple[float, float, float, int, int, int]:
    true_positive = sum(
        expected and actual for expected, actual in zip(truth, predicted, strict=True)
    )
    false_positive = sum(
        (not expected) and actual for expected, actual in zip(truth, predicted, strict=True)
    )
    false_negative = sum(
        expected and (not actual) for expected, actual in zip(truth, predicted, strict=True)
    )

    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0
    )
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = 2 * true_positive / denominator if denominator else 1.0

    return (
        precision,
        recall,
        f1,
        true_positive,
        false_positive,
        false_negative,
    )


def average_precision(
    truth: Sequence[bool],
    scores: Sequence[float],
) -> float | None:
    positives = sum(truth)
    if positives == 0:
        return None

    ranked = sorted(
        zip(scores, truth, strict=True),
        key=lambda pair: pair[0],
        reverse=True,
    )

    true_positive = 0
    precision_sum = 0.0

    for rank, (_, is_positive) in enumerate(ranked, start=1):
        if not is_positive:
            continue
        true_positive += 1
        precision_sum += true_positive / rank

    return precision_sum / positives


def expected_calibration_error(
    scores: Sequence[float],
    targets: Sequence[float],
    *,
    bins: int = ECE_BINS,
) -> float:
    if not scores:
        return 0.0

    score_array = np.asarray(scores, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)

    total = len(scores)
    ece = 0.0

    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins

        if bin_index == bins - 1:
            mask = (score_array >= lower) & (score_array <= upper)
        else:
            mask = (score_array >= lower) & (score_array < upper)

        count = int(np.sum(mask))
        if count == 0:
            continue

        mean_score = float(np.mean(score_array[mask]))
        mean_target = float(np.mean(target_array[mask]))
        ece += (count / total) * abs(mean_score - mean_target)

    return ece


def score_records(
    *,
    system: FrozenSystem,
    records: Mapping[str, HeldoutRecord],
    embeddings: Mapping[str, phase10.FloatArray],
    bank: phase10.PrototypeBank,
    classifiers: Mapping[str, tuple[phase10.FloatArray, float]],
) -> dict[str, ScoredRecord]:
    scored: dict[str, ScoredRecord] = {}

    for chunk_id in sorted(records):
        record = records[chunk_id]

        adapter = phase10.LabeledRecord(
            chunk_id=record.chunk_id,
            split="heldout",
            text=record.text,
            labels=record.labels,
            hard_negative_for=record.hard_negative_for,
        )

        raw = phase10.raw_scores(
            system.candidate,
            adapter,
            embeddings[chunk_id],
            bank,
            classifiers,
        )
        weights = phase10.apply_calibration(
            raw,
            system.calibrations,
        )
        active = tuple(
            phase10.predicted_active(
                weights,
                system.activation,
            )
        )

        scored[chunk_id] = ScoredRecord(
            record=record,
            raw_scores=raw,
            calibrated_weights=weights,
            active_concepts=active,
            ambiguous=phase10.predicted_ambiguous(
                weights,
                system.activation,
            ),
            unsupported=not active,
        )

    return scored


def metrics_for_subset(
    scored: Sequence[ScoredRecord],
) -> dict[str, object]:
    if not scored:
        return {
            "record_count": 0,
            "macro_f1": None,
            "micro_f1": None,
            "macro_average_precision": None,
            "calibration_error": None,
            "per_concept": {},
        }

    per_concept: dict[str, dict[str, object]] = {}
    concept_f1_values: list[float] = []
    ap_values: list[float] = []
    ece_values: list[float] = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for concept in phase10.CONCEPTS:
        truth = [label_is_active(item.record.labels[concept]) for item in scored]
        predicted = [concept in item.active_concepts for item in scored]
        weights = [item.calibrated_weights[concept] for item in scored]
        soft_targets = [soft_target(item.record.labels[concept]) for item in scored]

        precision, recall, f1, tp, fp, fn = precision_recall_f1(
            truth,
            predicted,
        )
        ap = average_precision(truth, weights)
        ece = expected_calibration_error(weights, soft_targets)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        concept_f1_values.append(f1)
        ece_values.append(ece)
        if ap is not None:
            ap_values.append(ap)

        per_concept[concept] = {
            "support_active": sum(truth),
            "support_inactive": len(truth) - sum(truth),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "average_precision": ap,
            "calibration_error_soft_ece_10_bins": ece,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
        }

    micro_denominator = 2 * total_tp + total_fp + total_fn
    micro_f1 = 2 * total_tp / micro_denominator if micro_denominator else 1.0

    exact_matches = [
        set(item.active_concepts)
        == {concept for concept in phase10.CONCEPTS if label_is_active(item.record.labels[concept])}
        for item in scored
    ]

    unsupported_truth = [
        not any(label_is_active(item.record.labels[concept]) for concept in phase10.CONCEPTS)
        for item in scored
    ]
    unsupported_pred = [item.unsupported for item in scored]

    unsupported_accuracy = float(
        np.mean(
            np.asarray(unsupported_truth, dtype=bool) == np.asarray(unsupported_pred, dtype=bool)
        )
    )

    ambiguity_count = sum(item.ambiguous for item in scored)

    return {
        "record_count": len(scored),
        "macro_f1": float(np.mean(concept_f1_values)),
        "micro_f1": micro_f1,
        "macro_average_precision": (float(np.mean(ap_values)) if ap_values else None),
        "calibration_error": {
            "definition": (
                "Macro mean of per-concept 10-bin expected calibration error "
                "against Phase 10 soft targets: negative=0, partial=0.5, positive=1."
            ),
            "macro_soft_ece_10_bins": float(np.mean(ece_values)),
        },
        "exact_active_set_accuracy": float(np.mean(exact_matches)),
        "unsupported_accuracy": unsupported_accuracy,
        "predicted_ambiguous_count": ambiguity_count,
        "per_concept": per_concept,
    }


def hard_negative_evaluation(
    scored: Sequence[ScoredRecord],
) -> dict[str, object]:
    all_groups = {
        **HARD_NEGATIVE_REPORT_GROUPS,
        **EXTRA_HARD_NEGATIVE_GROUPS,
    }

    results: dict[str, object] = {}

    for group_name, group in all_groups.items():
        target_concept = require_string(
            group.get("target_concept"),
            f"{group_name} target_concept",
        )
        raw_categories = group.get("categories")
        if not isinstance(raw_categories, tuple):
            raise HeldoutEvaluationError(f"{group_name} categories must be a tuple.")
        categories = {str(category).casefold() for category in raw_categories}

        matching = [item for item in scored if item.record.hard_negative_category in categories]

        false_positives = [item for item in matching if target_concept in item.active_concepts]

        results[group_name] = {
            "required_phase11_metric": group_name in HARD_NEGATIVE_REPORT_GROUPS,
            "target_concept": target_concept,
            "accepted_category_labels": sorted(categories),
            "sample_count": len(matching),
            "false_positive_count": len(false_positives),
            "false_positive_rate": (len(false_positives) / len(matching) if matching else None),
            "status": ("measured" if matching else "not_available_in_heldout"),
            "chunk_ids": [item.record.chunk_id for item in matching],
        }

    observed_categories = sorted(
        {
            item.record.hard_negative_category
            for item in scored
            if item.record.hard_negative_category
        }
    )

    return {
        "groups": results,
        "observed_heldout_categories": observed_categories,
        "note": (
            "A required category with zero samples is reported as unavailable; "
            "it is not silently substituted with a broader hard-negative class."
        ),
    }


def confusion_analysis(
    scored: Sequence[ScoredRecord],
) -> dict[str, object]:
    result: dict[str, object] = {}

    for left, right in CONFUSION_PAIRS:
        left_only_opportunities = 0
        left_to_right_errors = 0
        right_only_opportunities = 0
        right_to_left_errors = 0

        left_to_right_chunks: list[str] = []
        right_to_left_chunks: list[str] = []

        for item in scored:
            truth_left = label_is_active(item.record.labels[left])
            truth_right = label_is_active(item.record.labels[right])
            pred_left = left in item.active_concepts
            pred_right = right in item.active_concepts

            if truth_left and not truth_right:
                left_only_opportunities += 1
                if pred_right:
                    left_to_right_errors += 1
                    left_to_right_chunks.append(item.record.chunk_id)

            if truth_right and not truth_left:
                right_only_opportunities += 1
                if pred_left:
                    right_to_left_errors += 1
                    right_to_left_chunks.append(item.record.chunk_id)

        pair_name = f"{left}__{right}"
        result[pair_name] = {
            f"{left}_mistaken_as_{right}": {
                "opportunities": left_only_opportunities,
                "error_count": left_to_right_errors,
                "error_rate": (
                    left_to_right_errors / left_only_opportunities
                    if left_only_opportunities
                    else None
                ),
                "chunk_ids": left_to_right_chunks,
            },
            f"{right}_mistaken_as_{left}": {
                "opportunities": right_only_opportunities,
                "error_count": right_to_left_errors,
                "error_rate": (
                    right_to_left_errors / right_only_opportunities
                    if right_only_opportunities
                    else None
                ),
                "chunk_ids": right_to_left_chunks,
            },
        }

    return result


def serialize_record(item: ScoredRecord) -> dict[str, object]:
    expected = sorted(
        concept for concept in phase10.CONCEPTS if label_is_active(item.record.labels[concept])
    )

    predicted = list(item.active_concepts)
    false_positive = sorted(set(predicted) - set(expected))
    false_negative = sorted(set(expected) - set(predicted))

    return {
        "chunk_id": item.record.chunk_id,
        "source_id": item.record.source_id,
        "domain": item.record.domain,
        "reviewed_labels": item.record.labels,
        "expected_active_concepts": expected,
        "raw_scores": item.raw_scores,
        "calibrated_weights": item.calibrated_weights,
        "active_concepts": predicted,
        "false_positive_concepts": false_positive,
        "false_negative_concepts": false_negative,
        "ambiguous": item.ambiguous,
        "unsupported": item.unsupported,
        "hard_negative_category": item.record.hard_negative_category,
        "hard_negative_for": list(item.record.hard_negative_for),
    }


def write_error_csv(
    path: Path,
    scored: Sequence[ScoredRecord],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")

    fieldnames = [
        "chunk_id",
        "source_id",
        "domain",
        "hard_negative_category",
        "hard_negative_for",
        "expected_active_concepts",
        "predicted_active_concepts",
        "false_positive_concepts",
        "false_negative_concepts",
        "ambiguous",
        "unsupported",
        "consciousness_label",
        "consciousness_weight",
        "self_identity_label",
        "self_identity_weight",
        "reality_appearance_label",
        "reality_appearance_weight",
    ]

    rows_written = 0

    with temporary.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for item in scored:
            expected = {
                concept
                for concept in phase10.CONCEPTS
                if label_is_active(item.record.labels[concept])
            }
            predicted = set(item.active_concepts)

            false_positive = sorted(predicted - expected)
            false_negative = sorted(expected - predicted)

            if not false_positive and not false_negative and not item.record.hard_negative_category:
                continue

            writer.writerow(
                {
                    "chunk_id": item.record.chunk_id,
                    "source_id": item.record.source_id,
                    "domain": item.record.domain,
                    "hard_negative_category": (item.record.hard_negative_category),
                    "hard_negative_for": "|".join(item.record.hard_negative_for),
                    "expected_active_concepts": "|".join(sorted(expected)),
                    "predicted_active_concepts": "|".join(item.active_concepts),
                    "false_positive_concepts": "|".join(false_positive),
                    "false_negative_concepts": "|".join(false_negative),
                    "ambiguous": item.ambiguous,
                    "unsupported": item.unsupported,
                    "consciousness_label": (item.record.labels["consciousness"]),
                    "consciousness_weight": (f"{item.calibrated_weights['consciousness']:.8f}"),
                    "self_identity_label": (item.record.labels["self_identity"]),
                    "self_identity_weight": (f"{item.calibrated_weights['self_identity']:.8f}"),
                    "reality_appearance_label": (item.record.labels["reality_appearance"]),
                    "reality_appearance_weight": (
                        f"{item.calibrated_weights['reality_appearance']:.8f}"
                    ),
                }
            )
            rows_written += 1

    temporary.replace(path)
    return rows_written


def metric_value(
    metrics: Mapping[str, object],
    key: str,
) -> float | None:
    value = metrics.get(key)
    if isinstance(value, int | float):
        return float(value)
    return None


def markdown_report(
    *,
    phase10_result: Mapping[str, object],
    systems: Sequence[FrozenSystem],
    system_results: Mapping[str, Mapping[str, object]],
    selected_scored: Sequence[ScoredRecord],
    output_json: Path,
    error_csv: Path,
) -> str:
    selected = system_results["selected_hybrid"]
    selected_overall = require_mapping(
        selected.get("overall"),
        "selected overall metrics",
    )
    selected_domains = require_mapping(
        selected.get("domains"),
        "selected domains",
    )
    selected_hn = require_mapping(
        selected.get("hard_negative_evaluation"),
        "selected hard negative evaluation",
    )
    hn_groups = require_mapping(
        selected_hn.get("groups"),
        "selected hard negative groups",
    )
    selected_confusions = require_mapping(
        selected.get("confusion_analysis"),
        "selected confusion analysis",
    )

    dev_selected = require_mapping(
        phase10_result.get("selected_development_result"),
        "Phase 10 selected development result",
    )
    dev_metrics = require_mapping(
        dev_selected.get("metrics"),
        "Phase 10 selected development metrics",
    )

    heldout_macro_f1 = metric_value(selected_overall, "macro_f1")
    dev_macro_f1 = metric_value(dev_metrics, "macro_f1")
    macro_f1_gap = (
        heldout_macro_f1 - dev_macro_f1
        if heldout_macro_f1 is not None and dev_macro_f1 is not None
        else None
    )

    lines = [
        "# Phase 1 Held-out Concept Mapping Evaluation",
        "",
        f"- Evaluation version: `{EVALUATION_VERSION}`",
        f"- Generated: `{utc_now()}`",
        "- Status: **HELD-OUT EVALUATION RECORDED**",
        "- Held-out records: **79**",
        "- Post-hoc threshold changes: **NONE**",
        "- Post-hoc calibration changes: **NONE**",
        "- Selected method changed after Held-out: **NO**",
        "",
        "## Evaluation rule",
        "",
        (
            "The Phase 10 hybrid method, concept-specific thresholds, calibration "
            "parameters, ambiguity margin, prototype aggregation, lexical weight, "
            "embedding weight, negative penalty and maximum-active setting were "
            "frozen before this evaluation. Held-out is used only to measure "
            "generalization; it does not select or retune the model."
        ),
        "",
        "## Selected frozen method",
        "",
    ]

    frozen = require_mapping(
        phase10_result.get("frozen_parameters"),
        "frozen_parameters",
    )
    thresholds = require_mapping(
        frozen.get("concept_activation_thresholds"),
        "frozen thresholds",
    )

    overall_ece = calibration_ece(
        selected_overall,
        "selected overall calibration",
    )
    overall_ece_text = f"{overall_ece:.4f}" if overall_ece is not None else "N/A"

    lines.extend(
        [
            f"- Method: `{require_string(frozen.get('method'), 'method')}`",
            (
                "- Prototype source / aggregation: "
                f"`{require_string(frozen.get('prototype_source'), 'source')}` / "
                f"`{require_string(frozen.get('prototype_aggregation'), 'aggregation')}`"
            ),
            (
                "- Embedding / lexical weight: "
                f"`{require_float(frozen.get('embedding_weight'), 'embedding_weight'):.2f}` / "
                f"`{require_float(frozen.get('lexical_weight'), 'lexical_weight'):.2f}`"
            ),
            (
                "- Hard-negative penalty: "
                f"`{require_float(frozen.get('negative_penalty'), 'negative_penalty'):.2f}`"
            ),
            (
                "- Thresholds: "
                f"consciousness `{require_float(thresholds.get('consciousness'), 'consciousness threshold'):.2f}`, "
                f"self `{require_float(thresholds.get('self_identity'), 'self threshold'):.2f}`, "
                f"reality/appearance `{require_float(thresholds.get('reality_appearance'), 'reality threshold'):.2f}`"
            ),
            "",
            "## Overall Held-out metrics",
            "",
            f"- Macro F1: **{require_float(selected_overall['macro_f1'], 'macro_f1'):.4f}**",
            f"- Micro F1: **{require_float(selected_overall['micro_f1'], 'micro_f1'):.4f}**",
            (
                "- Macro average precision: "
                f"**{require_float(selected_overall['macro_average_precision'], 'macro_average_precision'):.4f}**"
            ),
            (f"- Calibration error (macro soft ECE): **{overall_ece_text}**"),
            (
                "- Exact active-set accuracy: "
                f"**{require_float(selected_overall['exact_active_set_accuracy'], 'exact_active_set_accuracy'):.4f}**"
            ),
            (
                "- Unsupported accuracy: "
                f"**{require_float(selected_overall['unsupported_accuracy'], 'unsupported_accuracy'):.4f}**"
            ),
            "",
            "## Development → Held-out generalization",
            "",
            f"- Development macro F1: `{dev_macro_f1:.4f}`"
            if dev_macro_f1 is not None
            else "- Development macro F1: unavailable",
            f"- Held-out macro F1: `{heldout_macro_f1:.4f}`"
            if heldout_macro_f1 is not None
            else "- Held-out macro F1: unavailable",
            f"- Macro F1 delta: `{macro_f1_gap:+.4f}`"
            if macro_f1_gap is not None
            else "- Macro F1 delta: unavailable",
            "",
            "## Per-concept Held-out performance",
            "",
            "| Concept | Precision | Recall | F1 | Avg precision | Calibration error |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    per_concept = require_mapping(
        selected_overall.get("per_concept"),
        "selected per_concept",
    )
    for concept in phase10.CONCEPTS:
        values = require_mapping(
            per_concept.get(concept),
            f"{concept} metrics",
        )
        ap = values.get("average_precision")
        ap_text = f"{float(ap):.4f}" if isinstance(ap, int | float) else "N/A"
        lines.append(
            f"| {concept} | "
            f"{require_float(values['precision'], f'{concept} precision'):.4f} | "
            f"{require_float(values['recall'], f'{concept} recall'):.4f} | "
            f"{require_float(values['f1'], f'{concept} f1'):.4f} | "
            f"{ap_text} | "
            f"{require_float(values['calibration_error_soft_ece_10_bins'], f'{concept} ECE'):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Baseline comparison",
            "",
            "| System | Macro F1 | Micro F1 | Avg precision | Calibration error | Exact set |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    for system in systems:
        result = system_results[system.system_id]
        metrics = require_mapping(result.get("overall"), "system overall")
        sys_ece = calibration_ece(
            metrics,
            f"{system.system_id} calibration",
        )
        sys_ece_text = f"{sys_ece:.4f}" if sys_ece is not None else "N/A"
        lines.append(
            f"| {system.display_name} | "
            f"{require_float(metrics['macro_f1'], 'macro_f1'):.4f} | "
            f"{require_float(metrics['micro_f1'], 'micro_f1'):.4f} | "
            f"{require_float(metrics['macro_average_precision'], 'macro_average_precision'):.4f} | "
            f"{sys_ece_text} | "
            f"{require_float(metrics['exact_active_set_accuracy'], 'exact_active_set_accuracy'):.4f} |"
        )

    lines.extend(
        [
            "",
            "The baseline table is descriptive. It does **not** trigger model "
            "reselection after Held-out.",
            "",
            "## Domain-level results — selected method",
            "",
            "| Domain | Records | Macro F1 | Micro F1 | Avg precision | Calibration error |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    for domain in DOMAIN_ORDER:
        values = require_mapping(
            selected_domains.get(domain),
            f"{domain} domain metrics",
        )

        rec_count = require_int(values.get("record_count", 0), f"{domain} record_count")

        macro_f1 = values.get("macro_f1")
        f1_str = f"{float(macro_f1):.4f}" if isinstance(macro_f1, (int, float)) else "N/A"

        micro_f1 = values.get("micro_f1")
        micro_str = f"{float(micro_f1):.4f}" if isinstance(micro_f1, (int, float)) else "N/A"

        macro_ap = values.get("macro_average_precision")
        ap_str = f"{float(macro_ap):.4f}" if isinstance(macro_ap, (int, float)) else "N/A"

        ece = calibration_ece(
            values,
            f"{domain} calibration",
        )
        ece_str = f"{ece:.4f}" if ece is not None else "N/A"

        lines.append(
            f"| {domain} | {rec_count} | "
            f"{f1_str} | "
            f"{micro_str} | "
            f"{ap_str} | "
            f"{ece_str} |"
        )

    lines.extend(
        [
            "",
            "## Required hard-negative evaluation — selected method",
            "",
            "| Error type | Samples | False positives | FP rate | Status |",
            "|---|---:|---:|---:|---|",
        ]
    )

    for group_name in HARD_NEGATIVE_REPORT_GROUPS:
        values = require_mapping(
            hn_groups.get(group_name),
            f"{group_name} hard-negative result",
        )
        rate = values.get("false_positive_rate")
        rate_text = f"{float(rate):.4f}" if isinstance(rate, int | float) else "N/A"
        lines.append(
            f"| {group_name} | {require_int(values['sample_count'], f'{group_name} sample_count')} | "
            f"{require_int(values['false_positive_count'], f'{group_name} false_positive_count')} | "
            f"{rate_text} | {values['status']} |"
        )

    lines.extend(
        [
            "",
            "A required hard-negative category with zero Held-out examples is "
            "reported as unavailable rather than inferred from another category.",
            "",
            "## Adjacent-concept confusion — selected method",
            "",
        ]
    )

    for pair_name, pair_raw in selected_confusions.items():
        pair = require_mapping(pair_raw, f"confusion {pair_name}")
        lines.append(f"### `{pair_name}`")
        lines.append("")
        for direction_name, direction_raw in pair.items():
            direction = require_mapping(
                direction_raw,
                f"confusion direction {direction_name}",
            )
            rate = direction.get("error_rate")
            rate_text = f"{float(rate):.4f}" if isinstance(rate, int | float) else "N/A"
            lines.append(
                f"- `{direction_name}`: {require_int(direction['error_count'], f'{direction_name} error_count')}/"
                f"{require_int(direction['opportunities'], f'{direction_name} opportunities')} = **{rate_text}**"
            )
        lines.append("")

    error_count = sum(
        1
        for item in selected_scored
        if (
            set(item.active_concepts)
            != {
                concept
                for concept in phase10.CONCEPTS
                if label_is_active(item.record.labels[concept])
            }
            or item.record.hard_negative_category
        )
    )

    lines.extend(
        [
            "## Failure and limitation record",
            "",
            (
                f"- `{error_count}` Held-out records are included in the error-analysis "
                "CSV because they contain an active-set mismatch and/or a reviewed "
                "hard-negative category."
            ),
            "- `partial` labels are treated as active for multi-label precision/recall/F1.",
            (
                "- Calibration error uses the original soft calibration targets "
                "negative=0, partial=0.5, positive=1 because Phase 10 calibrated "
                "weights against those targets."
            ),
            (
                "- Hard-negative categories are evaluated only when the exact reviewed "
                "category is present in Held-out; zero-sample requirements remain explicit."
            ),
            (
                "- The supervised baseline is deterministically reconstructed from Build "
                "using the frozen Phase 10 training algorithm. Held-out labels do not "
                "influence its training."
            ),
            (
                "- If a baseline outperforms the selected method on Held-out, that is "
                "documented as a limitation; Phase 10 parameters are not changed."
            ),
            "",
            "## Selected-method justification",
            "",
            (
                "The hybrid remains the selected method because it was chosen and frozen "
                "on Development before Held-out was opened. Phase 11 reports whether that "
                "choice generalizes and compares it with frozen baselines; it does not use "
                "Held-out to make a new model-selection decision."
            ),
            "",
            "## Exit gate",
            "",
            "- Held-out results recorded: **PASS**",
            "- Post-hoc threshold changes: **NONE**",
            "- Post-hoc calibration changes: **NONE**",
            "- Failures and limitations explicitly documented: **PASS**",
            "- Frozen selected method retained without Held-out reselection: **PASS**",
            "",
            f"Machine-readable results: `{output_json.as_posix()}`",
            f"Error analysis: `{error_csv.as_posix()}`",
            "",
        ]
    )

    return "\n".join(lines)


def ensure_output_policy(
    paths: Sequence[Path],
    *,
    replace: bool,
) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not replace:
        raise HeldoutEvaluationError(
            "Phase 11 outputs already exist. Use --replace to reproduce the "
            "same frozen evaluation without changing parameters: "
            + ", ".join(path.as_posix() for path in existing)
        )


def run_phase11(
    *,
    project_root: Path,
    phase10_results_path: Path,
    split_manifest_path: Path,
    build_set_path: Path,
    heldout_set_path: Path,
    approved_embeddings_path: Path,
    query_embeddings_path: Path,
    passage_embeddings_path: Path,
    embedding_manifest_path: Path,
    output_json_path: Path,
    error_csv_path: Path,
    report_path: Path,
    replace: bool,
) -> dict[str, object]:
    project_root = project_root.resolve()

    phase10_results_path = resolve(project_root, phase10_results_path)
    split_manifest_path = resolve(project_root, split_manifest_path)
    build_set_path = resolve(project_root, build_set_path)
    heldout_set_path = resolve(project_root, heldout_set_path)
    approved_embeddings_path = resolve(
        project_root,
        approved_embeddings_path,
    )
    query_embeddings_path = resolve(project_root, query_embeddings_path)
    passage_embeddings_path = resolve(project_root, passage_embeddings_path)
    embedding_manifest_path = resolve(
        project_root,
        embedding_manifest_path,
    )
    output_json_path = resolve(project_root, output_json_path)
    error_csv_path = resolve(project_root, error_csv_path)
    report_path = resolve(project_root, report_path)

    for path in (
        phase10_results_path,
        split_manifest_path,
        build_set_path,
        heldout_set_path,
        approved_embeddings_path,
        query_embeddings_path,
        passage_embeddings_path,
        embedding_manifest_path,
    ):
        require_file(path)

    ensure_output_policy(
        (output_json_path, error_csv_path, report_path),
        replace=replace,
    )

    LOGGER.info("Phase 11 starting: opening frozen Held-out for final evaluation only")

    phase10_result = validate_phase10_freeze(phase10_results_path)
    split_manifest = validate_split_manifest(
        split_manifest_path,
        heldout_set_path,
    )

    heldout = load_heldout_records(heldout_set_path)
    build = phase10.load_split(build_set_path, "build")

    if len(build) != EXPECTED_BUILD_COUNT:
        raise HeldoutEvaluationError(
            f"Expected {EXPECTED_BUILD_COUNT} Build records, found {len(build)}."
        )

    identity, embedding_manifest = phase10.load_identity(embedding_manifest_path)

    needed_ids = set(build) | set(heldout)
    embeddings = phase10.load_needed_embeddings(
        approved_embeddings_path,
        needed_ids,
        identity,
    )
    bank = phase10.load_prototype_bank(
        query_embeddings_path,
        passage_embeddings_path,
        identity,
    )

    # Deterministic reconstruction of optional Baseline D only.
    # This is Build-only training with the already-frozen Phase 10 algorithm.
    classifiers = phase10.train_build_classifiers(build, embeddings)

    systems = build_frozen_systems(phase10_result)

    LOGGER.info(
        "Held-out records=%d systems=%d thresholds/calibration frozen",
        len(heldout),
        len(systems),
    )

    scored_by_system: dict[str, dict[str, ScoredRecord]] = {}
    system_results: dict[str, dict[str, object]] = {}

    for system in systems:
        scored = score_records(
            system=system,
            records=heldout,
            embeddings=embeddings,
            bank=bank,
            classifiers=classifiers,
        )
        scored_by_system[system.system_id] = scored

        all_scored = list(scored.values())
        domains = {
            domain: metrics_for_subset(
                [item for item in all_scored if item.record.domain == domain]
            )
            for domain in DOMAIN_ORDER
        }

        system_results[system.system_id] = {
            "display_name": system.display_name,
            "description": system.description,
            "selected_method": system.selected_method,
            "frozen_candidate": system.candidate.as_dict(),
            "frozen_activation": system.activation.as_dict(),
            "overall": metrics_for_subset(all_scored),
            "domains": domains,
            "hard_negative_evaluation": hard_negative_evaluation(all_scored),
            "confusion_analysis": confusion_analysis(all_scored),
        }

        overall = require_mapping(
            system_results[system.system_id]["overall"],
            f"{system.system_id} overall",
        )
        LOGGER.info(
            "%s macro_f1=%.4f micro_f1=%.4f exact_set=%.4f",
            system.system_id,
            require_float(overall["macro_f1"], "overall macro_f1"),
            require_float(overall["micro_f1"], "overall micro_f1"),
            require_float(
                overall["exact_active_set_accuracy"],
                "overall exact_active_set_accuracy",
            ),
        )

    selected_scored = list(scored_by_system["selected_hybrid"].values())

    error_rows = write_error_csv(
        error_csv_path,
        selected_scored,
    )

    dev_selected = require_mapping(
        phase10_result.get("selected_development_result"),
        "Phase 10 selected development result",
    )
    dev_metrics = require_mapping(
        dev_selected.get("metrics"),
        "Phase 10 selected development metrics",
    )
    heldout_selected_metrics = require_mapping(
        system_results["selected_hybrid"]["overall"],
        "selected Held-out overall",
    )

    dev_macro_f1 = require_float(
        dev_metrics.get("macro_f1"),
        "Development macro_f1",
    )
    heldout_macro_f1 = require_float(
        heldout_selected_metrics.get("macro_f1"),
        "Held-out macro_f1",
    )

    results: dict[str, object] = {
        "script_version": SCRIPT_VERSION,
        "evaluation_version": EVALUATION_VERSION,
        "generated_at": utc_now(),
        "status": "evaluation_complete",
        "phase": "phase_11_untouched_heldout_evaluation",
        "evaluation_policy": {
            "heldout_opened_for_final_evaluation": True,
            "post_hoc_threshold_changes": False,
            "post_hoc_calibration_changes": False,
            "post_hoc_feature_changes": False,
            "post_hoc_model_reselection": False,
            "selected_method_origin": "Phase 10 Development-only tuning",
            "baseline_comparison_is_descriptive_only": True,
        },
        "inputs": {
            "phase10_results": {
                "path": phase10_results_path.as_posix(),
                "sha256_file_bytes": phase10.sha256_file(phase10_results_path),
                "status": optional_string(phase10_result.get("status")),
            },
            "split_manifest": {
                "path": split_manifest_path.as_posix(),
                "sha256_file_bytes": phase10.sha256_file(split_manifest_path),
                "status": optional_string(split_manifest.get("status")),
            },
            "heldout": {
                "path": heldout_set_path.as_posix(),
                "sha256": phase10.sha256_jsonl(heldout_set_path),
                "record_count": len(heldout),
                "use": "final evaluation only",
            },
            "build": {
                "path": build_set_path.as_posix(),
                "sha256": phase10.sha256_jsonl(build_set_path),
                "record_count": len(build),
                "use": "reconstruct optional supervised baseline only",
            },
            "approved_embeddings": {
                "path": approved_embeddings_path.as_posix(),
                "sha256": phase10.sha256_jsonl(approved_embeddings_path),
            },
            "query_prototype_embeddings": {
                "path": query_embeddings_path.as_posix(),
                "sha256": phase10.sha256_jsonl(query_embeddings_path),
            },
            "passage_prototype_embeddings": {
                "path": passage_embeddings_path.as_posix(),
                "sha256": phase10.sha256_jsonl(passage_embeddings_path),
            },
            "embedding_manifest": {
                "path": embedding_manifest_path.as_posix(),
                "sha256_file_bytes": phase10.sha256_file(embedding_manifest_path),
                "status": optional_string(embedding_manifest.get("status")),
            },
        },
        "embedding_identity": identity.as_dict(),
        "frozen_phase10_parameters": require_mapping(
            phase10_result.get("frozen_parameters"),
            "frozen Phase 10 parameters",
        ),
        "systems": system_results,
        "selected_method_generalization": {
            "development_macro_f1": dev_macro_f1,
            "heldout_macro_f1": heldout_macro_f1,
            "macro_f1_delta": heldout_macro_f1 - dev_macro_f1,
        },
        "selected_method_records": [serialize_record(item) for item in selected_scored],
        "error_analysis": {
            "path": error_csv_path.as_posix(),
            "rows_written": error_rows,
            "policy": (
                "Selected-method active-set mismatches plus all reviewed "
                "hard-negative Held-out records."
            ),
        },
        "limitations_policy": {
            "zero_sample_required_hard_negative_categories_are_explicit": True,
            "partial_labels_count_as_active_for_multilabel_metrics": True,
            "calibration_error_uses_soft_review_targets": True,
            "heldout_does_not_trigger_reselection": True,
        },
        "exit_gate": {
            "heldout_results_recorded": True,
            "no_post_hoc_threshold_changes": True,
            "no_post_hoc_calibration_changes": True,
            "no_post_hoc_model_reselection": True,
            "failures_and_limitations_documented": True,
            "selected_method_justification_recorded": True,
        },
        "next_step": (
            "Phase 12: calculate reviewed weighted concept tags using the "
            "frozen Phase 10 mapping. Do not alter Phase 10 parameters based "
            "on Held-out outcomes."
        ),
    }

    atomic_json(output_json_path, results)
    atomic_text(
        report_path,
        markdown_report(
            phase10_result=phase10_result,
            systems=systems,
            system_results=system_results,
            selected_scored=selected_scored,
            output_json=output_json_path,
            error_csv=error_csv_path,
        ),
    )

    LOGGER.info("Phase 11 Held-out evaluation complete")
    LOGGER.info("Post-hoc threshold changes: NONE")
    LOGGER.info("Post-hoc calibration changes: NONE")
    LOGGER.info("Post-hoc model reselection: NONE")
    LOGGER.info("Error-analysis rows: %d", error_rows)
    LOGGER.info("JSON: %s", output_json_path)
    LOGGER.info("CSV: %s", error_csv_path)
    LOGGER.info("Markdown: %s", report_path)

    return results


def main() -> int:
    arguments = parse_arguments()
    configure_logging(arguments.log_level)

    try:
        run_phase11(
            project_root=arguments.project_root,
            phase10_results_path=arguments.phase10_results,
            split_manifest_path=arguments.split_manifest,
            build_set_path=arguments.build_set,
            heldout_set_path=arguments.heldout_set,
            approved_embeddings_path=arguments.approved_embeddings,
            query_embeddings_path=arguments.query_prototype_embeddings,
            passage_embeddings_path=arguments.passage_prototype_embeddings,
            embedding_manifest_path=arguments.embedding_manifest,
            output_json_path=arguments.output_json,
            error_csv_path=arguments.error_csv,
            report_path=arguments.report,
            replace=arguments.replace,
        )
    except HeldoutEvaluationError:
        LOGGER.exception("Phase 11 Held-out evaluation failed")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
