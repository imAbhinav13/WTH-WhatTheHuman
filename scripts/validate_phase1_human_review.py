from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import logging
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Final

LOGGER = logging.getLogger("wth.phase1.validate_phase1_human_review")

VALIDATOR_VERSION: Final = "1.0.0"

DEFAULT_REVIEW_CSV: Final = Path("artifacts/review/phase1_selection_review_packet.csv")
DEFAULT_SELECTION_MANIFEST: Final = Path(
    "artifacts/phase1/selection/phase1_selection_manifest.json"
)
DEFAULT_OUTPUT_DIRECTORY: Final = Path("artifacts/phase1/reviewed")

GOLD_CORPUS_FILENAME: Final = "phase1_reviewed_gold_corpus.jsonl"
MANIFEST_FILENAME: Final = "phase1_human_review_manifest.json"
REPORT_FILENAME: Final = "phase1_human_review_report.html"

EXPECTED_REVIEW_ROWS: Final = 424

CONCEPTS: Final = (
    "consciousness",
    "self_identity",
    "reality_appearance",
)

REVIEW_DECISIONS: Final = {
    "include",
    "include_with_edits",
    "exclude",
    "needs_source_review",
}

CONCEPT_LABELS: Final = {
    "positive",
    "partial",
    "negative",
    "uncertain",
    "not_reviewed",
}

FINAL_CONCEPT_LABELS: Final = {
    "positive",
    "partial",
    "negative",
    "uncertain",
}

CITATION_VALUES: Final = {
    "yes",
    "no",
    "uncertain",
}

TEXT_QUALITY_VALUES: Final = {
    "acceptable",
    "needs_edit",
    "unacceptable",
    "uncertain",
}

OCR_REVIEW_VALUES: Final = {
    "clean",
    "acceptable",
    "corrected",
    "unacceptable",
    "uncertain",
}

APPROVED_DECISIONS: Final = {
    "include",
    "include_with_edits",
}

APPROVED_TEXT_QUALITY: Final = {
    "acceptable",
    "needs_edit",
}

APPROVED_OCR_STATUS: Final = {
    "clean",
    "acceptable",
    "corrected",
}

DOMAIN_TARGETS: Final = {
    "science": (70, 100),
    "advaita": (80, 120),
    "samkhya": (80, 120),
}

APPROVED_COUNT_MINIMUM: Final = 250
APPROVED_COUNT_MAXIMUM: Final = 350

HARD_NEGATIVE_CATEGORIES: Final = {
    "consciousness_vs_attention",
    "consciousness_vs_cognition",
    "self_vs_ego",
    "self_vs_personality",
    "reality_appearance_vs_cosmology",
    "reality_appearance_vs_perceptual_description",
    "advaita_atman_vs_samkhya_purusha",
}

REQUIRED_COLUMNS: Final = {
    "chunk_id",
    "source_id",
    "domain",
    "source_title",
    "chunk_text",
    "token_count",
    "text_checksum",
    "review_decision",
    "reviewer",
    "reviewed_at",
    "review_notes",
    "consciousness_label",
    "self_identity_label",
    "reality_appearance_label",
    "primary_concept",
    "secondary_concepts",
    "hard_negative_for",
    "citation_verified",
    "text_quality_status",
    "ocr_review_status",
}

HARD_NEGATIVE_REVIEW_COLUMNS: Final = (
    "hard_negative_category_reviewed",
    "hard_negative_category",
)


class ReviewValidationError(RuntimeError):
    """Raised when Phase 5 review artifacts cannot be validated safely."""


@dataclass(frozen=True)
class ReviewIssue:
    severity: str
    chunk_id: str
    field: str
    message: str


@dataclass
class ValidationState:
    issues: list[ReviewIssue] = field(default_factory=list)

    def error(self, chunk_id: str, field_name: str, message: str) -> None:
        self.issues.append(
            ReviewIssue(
                severity="error",
                chunk_id=chunk_id,
                field=field_name,
                message=message,
            )
        )

    def warning(self, chunk_id: str, field_name: str, message: str) -> None:
        self.issues.append(
            ReviewIssue(
                severity="warning",
                chunk_id=chunk_id,
                field=field_name,
                message=message,
            )
        )

    @property
    def errors(self) -> list[ReviewIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ReviewIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]


@dataclass(frozen=True)
class ReviewSummary:
    total_rows: int
    reviewed_rows: int
    approved_rows: int
    excluded_rows: int
    needs_source_review_rows: int
    by_domain: dict[str, int]
    by_source: dict[str, int]
    by_primary_concept: dict[str, int]
    by_label: dict[str, dict[str, int]]
    hard_negative_rows: int
    hard_negative_categories: dict[str, int]
    multi_concept_rows: int
    edited_rows: int


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Phase 5 human review and emit the authoritative "
            "Phase 1 reviewed gold corpus when the exit gate passes."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=DEFAULT_SELECTION_MANIFEST,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--expected-review-rows",
        type=int,
        default=EXPECTED_REVIEW_ROWS,
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Validation-progress mode. Generate the report and manifest "
            "even while review fields or exit-gate targets remain incomplete. "
            "The gold corpus will not be emitted unless the strict gate passes."
        ),
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing validator outputs.",
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


def resolve_from_project(project_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def normalize(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def normalize_lower(value: object) -> str:
    return normalize(value).casefold()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise ReviewValidationError(f"Required file does not exist: {path}")


def load_json_mapping(path: Path) -> dict[str, object]:
    try:
        loaded: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReviewValidationError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise ReviewValidationError(f"Expected a JSON object in {path}")
    result: dict[str, object] = {}
    for raw_key, value in loaded.items():
        if not isinstance(raw_key, str):
            raise ReviewValidationError(f"Non-string key found in {path}")
        result[raw_key] = value
    return result


def parse_timestamp(value: str) -> bool:
    if not value:
        return False
    candidate = value
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        datetime.fromisoformat(candidate)
    except ValueError:
        return False
    return True


def parse_multi_value(value: str) -> tuple[str, ...]:
    if not value.strip():
        return ()
    parts = re.split(r"[|;,]", value)
    return tuple(dict.fromkeys(part.strip().casefold() for part in parts if part.strip()))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_review_rows(
    path: Path,
) -> tuple[list[dict[str, str]], str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ReviewValidationError(f"Review CSV has no header: {path}")

        fieldnames = tuple(reader.fieldnames)
        missing = REQUIRED_COLUMNS - set(fieldnames)
        if missing:
            raise ReviewValidationError(
                "Review CSV is missing required columns: " + ", ".join(sorted(missing))
            )

        hard_negative_column = ""
        for candidate in HARD_NEGATIVE_REVIEW_COLUMNS:
            if candidate in fieldnames:
                hard_negative_column = candidate
                break
        if not hard_negative_column:
            raise ReviewValidationError(
                "Review CSV is missing a reviewed hard-negative category column. "
                "Expected one of: " + ", ".join(HARD_NEGATIVE_REVIEW_COLUMNS)
            )

        rows: list[dict[str, str]] = []
        for raw_row in reader:
            normalized_row = {
                key: normalize(value) for key, value in raw_row.items() if key is not None
            }
            rows.append(normalized_row)

    return rows, hard_negative_column


def validate_unique_chunk_ids(
    rows: Sequence[Mapping[str, str]],
    state: ValidationState,
) -> None:
    seen: set[str] = set()
    for row in rows:
        chunk_id = normalize(row.get("chunk_id"))
        if not chunk_id:
            state.error("", "chunk_id", "chunk_id is required.")
            continue
        if chunk_id in seen:
            state.error(
                chunk_id,
                "chunk_id",
                "Duplicate chunk_id appears in the review packet.",
            )
        seen.add(chunk_id)


def validate_review_row(
    row: Mapping[str, str],
    *,
    hard_negative_column: str,
    state: ValidationState,
    allow_incomplete: bool,
) -> None:
    chunk_id = normalize(row.get("chunk_id"))
    decision = normalize_lower(row.get("review_decision"))
    reviewer = normalize(row.get("reviewer"))
    reviewed_at = normalize(row.get("reviewed_at"))

    if not decision:
        state.error(chunk_id, "review_decision", "Review decision is blank.")
        return

    if decision not in REVIEW_DECISIONS:
        state.error(
            chunk_id,
            "review_decision",
            f"Invalid review decision: {decision!r}.",
        )

    if not reviewer:
        state.error(chunk_id, "reviewer", "Reviewer is required.")

    if not reviewed_at:
        state.error(chunk_id, "reviewed_at", "reviewed_at is required.")
    elif not parse_timestamp(reviewed_at):
        state.error(
            chunk_id,
            "reviewed_at",
            "reviewed_at must be an ISO-8601 timestamp.",
        )

    labels: dict[str, str] = {}
    for concept in CONCEPTS:
        field_name = f"{concept}_label"
        label = normalize_lower(row.get(field_name))
        labels[concept] = label

        if not label:
            state.error(chunk_id, field_name, "Concept label is blank.")
        elif label not in CONCEPT_LABELS:
            state.error(
                chunk_id,
                field_name,
                f"Invalid concept label: {label!r}.",
            )
        elif not allow_incomplete and label == "not_reviewed":
            state.error(
                chunk_id,
                field_name,
                "not_reviewed is not allowed at the Phase 5 exit gate.",
            )

    primary_concept = normalize_lower(row.get("primary_concept"))
    secondary_concepts = parse_multi_value(normalize(row.get("secondary_concepts")))

    if primary_concept and primary_concept not in CONCEPTS:
        state.error(
            chunk_id,
            "primary_concept",
            f"Unknown primary concept: {primary_concept!r}.",
        )

    invalid_secondary = [concept for concept in secondary_concepts if concept not in CONCEPTS]
    if invalid_secondary:
        state.error(
            chunk_id,
            "secondary_concepts",
            "Unknown secondary concept(s): " + ", ".join(sorted(invalid_secondary)),
        )

    if primary_concept and primary_concept in secondary_concepts:
        state.error(
            chunk_id,
            "secondary_concepts",
            "primary_concept must not also appear in secondary_concepts.",
        )

    hard_negative_for = parse_multi_value(normalize(row.get("hard_negative_for")))
    invalid_hard_negative_targets = [
        concept for concept in hard_negative_for if concept not in CONCEPTS
    ]
    if invalid_hard_negative_targets:
        state.error(
            chunk_id,
            "hard_negative_for",
            "Unknown hard-negative target concept(s): "
            + ", ".join(sorted(invalid_hard_negative_targets)),
        )

    hard_negative_category = normalize_lower(row.get(hard_negative_column))
    if hard_negative_category and hard_negative_category not in HARD_NEGATIVE_CATEGORIES:
        state.error(
            chunk_id,
            hard_negative_column,
            f"Unknown hard-negative category: {hard_negative_category!r}.",
        )

    if bool(hard_negative_for) != bool(hard_negative_category):
        state.error(
            chunk_id,
            hard_negative_column,
            "hard_negative_for and reviewed hard-negative category "
            "must either both be populated or both be blank.",
        )

    citation_verified = normalize_lower(row.get("citation_verified"))
    text_quality_status = normalize_lower(row.get("text_quality_status"))
    ocr_review_status = normalize_lower(row.get("ocr_review_status"))
    edited_text = normalize(row.get("edited_text"))

    if citation_verified not in CITATION_VALUES:
        state.error(
            chunk_id,
            "citation_verified",
            "citation_verified must be yes, no, or uncertain.",
        )

    if text_quality_status not in TEXT_QUALITY_VALUES:
        state.error(
            chunk_id,
            "text_quality_status",
            "text_quality_status must be acceptable, needs_edit, unacceptable, or uncertain.",
        )

    if ocr_review_status not in OCR_REVIEW_VALUES:
        state.error(
            chunk_id,
            "ocr_review_status",
            "ocr_review_status must be clean, acceptable, corrected, unacceptable, or uncertain.",
        )

    chunk_text = normalize(row.get("chunk_text"))
    text_checksum = normalize_lower(row.get("text_checksum"))
    if chunk_text and text_checksum:
        calculated = sha256_text(chunk_text)
        if calculated != text_checksum:
            state.warning(
                chunk_id,
                "text_checksum",
                "Current chunk_text checksum does not match text_checksum. "
                "Confirm whether the packet stores a transformed review view.",
            )

    if decision in APPROVED_DECISIONS:
        relevant_labels = {
            concept for concept, label in labels.items() if label in {"positive", "partial"}
        }
        if not relevant_labels and not hard_negative_for:
            state.error(
                chunk_id,
                "review_decision",
                "Approved chunk must be positive/partial for at least one "
                "Phase 1 concept or explicitly serve as a hard negative.",
            )

        if citation_verified != "yes":
            state.error(
                chunk_id,
                "citation_verified",
                "Approved chunks require citation_verified=yes.",
            )

        if text_quality_status not in APPROVED_TEXT_QUALITY:
            state.error(
                chunk_id,
                "text_quality_status",
                "Approved chunks require acceptable or needs_edit text quality.",
            )

        if ocr_review_status not in APPROVED_OCR_STATUS:
            state.error(
                chunk_id,
                "ocr_review_status",
                "Approved chunks require clean, acceptable, or corrected OCR status.",
            )

        if decision == "include_with_edits" and not edited_text:
            state.error(
                chunk_id,
                "edited_text",
                "include_with_edits requires an auditable edited_text value.",
            )

        if decision == "include" and edited_text:
            state.warning(
                chunk_id,
                "edited_text",
                "edited_text is populated while review_decision=include. "
                "Consider include_with_edits.",
            )

        if relevant_labels:
            if not primary_concept:
                state.error(
                    chunk_id,
                    "primary_concept",
                    "Approved relevant chunk requires primary_concept.",
                )
            elif primary_concept not in relevant_labels:
                state.error(
                    chunk_id,
                    "primary_concept",
                    "primary_concept should be labelled positive or partial.",
                )

            invalid_secondaries = [
                concept for concept in secondary_concepts if concept not in relevant_labels
            ]
            if invalid_secondaries:
                state.error(
                    chunk_id,
                    "secondary_concepts",
                    "Secondary concept(s) are not labelled positive/partial: "
                    + ", ".join(sorted(invalid_secondaries)),
                )

    elif decision == "exclude":
        if citation_verified == "yes":
            state.warning(
                chunk_id,
                "citation_verified",
                "Excluded chunk has a verified citation; this is allowed but "
                "confirm exclusion rationale in review_notes.",
            )

    elif decision == "needs_source_review":
        if not normalize(row.get("review_notes")):
            state.error(
                chunk_id,
                "review_notes",
                "needs_source_review requires review_notes describing the issue.",
            )


def approved_rows(
    rows: Sequence[Mapping[str, str]],
) -> list[Mapping[str, str]]:
    return [
        row for row in rows if normalize_lower(row.get("review_decision")) in APPROVED_DECISIONS
    ]


def build_summary(
    rows: Sequence[Mapping[str, str]],
    *,
    hard_negative_column: str,
) -> ReviewSummary:
    approved = approved_rows(rows)
    by_domain = Counter(normalize_lower(row.get("domain")) for row in approved)
    by_source = Counter(normalize(row.get("source_id")) for row in approved)
    by_primary = Counter(
        normalize_lower(row.get("primary_concept"))
        for row in approved
        if normalize(row.get("primary_concept"))
    )

    by_label: dict[str, dict[str, int]] = {}
    for concept in CONCEPTS:
        counter = Counter(normalize_lower(row.get(f"{concept}_label")) for row in approved)
        by_label[concept] = dict(sorted(counter.items()))

    hard_negative_categories = Counter(
        normalize_lower(row.get(hard_negative_column))
        for row in approved
        if normalize(row.get(hard_negative_column))
    )

    hard_negative_rows = sum(
        bool(parse_multi_value(normalize(row.get("hard_negative_for")))) for row in approved
    )

    multi_concept_rows = 0
    for row in approved:
        relevant_count = sum(
            normalize_lower(row.get(f"{concept}_label")) in {"positive", "partial"}
            for concept in CONCEPTS
        )
        if relevant_count >= 2:
            multi_concept_rows += 1

    return ReviewSummary(
        total_rows=len(rows),
        reviewed_rows=sum(
            normalize_lower(row.get("review_decision")) in REVIEW_DECISIONS for row in rows
        ),
        approved_rows=len(approved),
        excluded_rows=sum(normalize_lower(row.get("review_decision")) == "exclude" for row in rows),
        needs_source_review_rows=sum(
            normalize_lower(row.get("review_decision")) == "needs_source_review" for row in rows
        ),
        by_domain=dict(sorted(by_domain.items())),
        by_source=dict(sorted(by_source.items())),
        by_primary_concept=dict(sorted(by_primary.items())),
        by_label=by_label,
        hard_negative_rows=hard_negative_rows,
        hard_negative_categories=dict(sorted(hard_negative_categories.items())),
        multi_concept_rows=multi_concept_rows,
        edited_rows=sum(
            normalize_lower(row.get("review_decision")) == "include_with_edits" for row in approved
        ),
    )


def validate_exit_gate(
    rows: Sequence[Mapping[str, str]],
    *,
    summary: ReviewSummary,
    expected_review_rows: int,
    state: ValidationState,
    allow_incomplete: bool,
) -> None:
    if summary.total_rows != expected_review_rows:
        state.error(
            "",
            "review_packet",
            f"Expected {expected_review_rows} review rows, found {summary.total_rows}.",
        )

    if allow_incomplete:
        return

    if summary.reviewed_rows != summary.total_rows:
        state.error(
            "",
            "review_packet",
            "Every candidate must have a valid review decision.",
        )

    if not (APPROVED_COUNT_MINIMUM <= summary.approved_rows <= APPROVED_COUNT_MAXIMUM):
        state.error(
            "",
            "approved_count",
            f"Approved corpus must contain {APPROVED_COUNT_MINIMUM}-"
            f"{APPROVED_COUNT_MAXIMUM} chunks; found {summary.approved_rows}.",
        )

    for domain, (minimum, maximum) in DOMAIN_TARGETS.items():
        count = summary.by_domain.get(domain, 0)
        if not minimum <= count <= maximum:
            state.error(
                "",
                f"domain:{domain}",
                f"{domain} approved count must be {minimum}-{maximum}; found {count}.",
            )

    if summary.needs_source_review_rows > 0:
        state.error(
            "",
            "needs_source_review",
            f"{summary.needs_source_review_rows} rows still require source review.",
        )

    for concept in CONCEPTS:
        label_counts = summary.by_label.get(concept, {})
        positive_or_partial = label_counts.get("positive", 0) + label_counts.get("partial", 0)
        if positive_or_partial < 20:
            state.error(
                "",
                f"concept:{concept}",
                f"{concept} has only {positive_or_partial} approved "
                "positive/partial examples; at least 20 are required.",
            )

    if summary.hard_negative_rows < 30:
        state.error(
            "",
            "hard_negatives",
            f"Only {summary.hard_negative_rows} approved hard-negative chunks "
            "remain; at least 30 are required.",
        )

    if summary.multi_concept_rows < 15:
        state.warning(
            "",
            "multi_concept",
            f"Only {summary.multi_concept_rows} approved multi-concept chunks "
            "remain. Review whether adjacent-concept ambiguity is rich enough.",
        )

    for source_id, count in summary.by_source.items():
        domain = ""
        for row in rows:
            if (
                normalize(row.get("source_id")) == source_id
                and normalize_lower(row.get("review_decision")) in APPROVED_DECISIONS
            ):
                domain = normalize_lower(row.get("domain"))
                break
        domain_count = summary.by_domain.get(domain, 0)
        if domain_count > 0 and count / domain_count > 0.60:
            state.warning(
                "",
                f"source:{source_id}",
                f"{source_id} contributes {count}/{domain_count} "
                f"({count / domain_count:.1%}) approved {domain} chunks. "
                "Review source concentration.",
            )


def row_to_gold_record(
    row: Mapping[str, str],
    *,
    hard_negative_column: str,
) -> dict[str, object]:
    decision = normalize_lower(row.get("review_decision"))
    original_text = normalize(row.get("chunk_text"))
    edited_text = normalize(row.get("edited_text"))
    reviewed_text = edited_text if decision == "include_with_edits" else original_text

    return {
        "chunk_id": normalize(row.get("chunk_id")),
        "source_id": normalize(row.get("source_id")),
        "domain": normalize_lower(row.get("domain")),
        "source_title": normalize(row.get("source_title")),
        "author": normalize(row.get("author")),
        "translator": normalize(row.get("translator")),
        "publication_year": normalize(row.get("publication_year")),
        "citation": normalize(row.get("citation")),
        "section_title": normalize(row.get("section_title")),
        "structural_locator": normalize(row.get("structural_locator")),
        "token_count": normalize(row.get("token_count")),
        "source_checksum": normalize(row.get("source_checksum")),
        "text_checksum": normalize(row.get("text_checksum")),
        "original_text": original_text,
        "edited_text": edited_text,
        "reviewed_text": reviewed_text,
        "reviewed_text_checksum": sha256_text(reviewed_text),
        "selection_rank": normalize(row.get("selection_rank")),
        "selection_class": normalize(row.get("selection_class")),
        "selection_score": normalize(row.get("selection_score")),
        "selection_rule_ids": normalize(row.get("selection_rule_ids")),
        "matched_terms": normalize(row.get("matched_terms")),
        "eligible_concepts": normalize(row.get("eligible_concepts")),
        "proposed_hard_negative_category": normalize(row.get("hard_negative_category")),
        "selection_rationale": normalize(row.get("selection_rationale")),
        "parser_warnings": normalize(row.get("parser_warnings")),
        "review": {
            "decision": decision,
            "reviewer": normalize(row.get("reviewer")),
            "reviewed_at": normalize(row.get("reviewed_at")),
            "notes": normalize(row.get("review_notes")),
            "labels": {
                concept: normalize_lower(row.get(f"{concept}_label")) for concept in CONCEPTS
            },
            "primary_concept": normalize_lower(row.get("primary_concept")),
            "secondary_concepts": list(parse_multi_value(normalize(row.get("secondary_concepts")))),
            "hard_negative_for": list(parse_multi_value(normalize(row.get("hard_negative_for")))),
            "hard_negative_category": normalize_lower(row.get(hard_negative_column)),
            "citation_verified": normalize_lower(row.get("citation_verified")),
            "text_quality_status": normalize_lower(row.get("text_quality_status")),
            "ocr_review_status": normalize_lower(row.get("ocr_review_status")),
        },
        "corpus_status": "phase1_reviewed_gold_candidate",
        "validator_version": VALIDATOR_VERSION,
    }


def atomic_write_jsonl(
    path: Path,
    records: Iterable[Mapping[str, object]],
) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            handle.write("\n")
    temporary.replace(path)


def atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def summary_to_dict(summary: ReviewSummary) -> dict[str, object]:
    return {
        "total_rows": summary.total_rows,
        "reviewed_rows": summary.reviewed_rows,
        "approved_rows": summary.approved_rows,
        "excluded_rows": summary.excluded_rows,
        "needs_source_review_rows": summary.needs_source_review_rows,
        "by_domain": summary.by_domain,
        "by_source": summary.by_source,
        "by_primary_concept": summary.by_primary_concept,
        "by_label": summary.by_label,
        "hard_negative_rows": summary.hard_negative_rows,
        "hard_negative_categories": summary.hard_negative_categories,
        "multi_concept_rows": summary.multi_concept_rows,
        "edited_rows": summary.edited_rows,
    }


def render_html_report(
    *,
    summary: ReviewSummary,
    state: ValidationState,
    strict_gate_passed: bool,
    review_csv: Path,
) -> str:
    def esc(value: object) -> str:
        return html.escape(str(value), quote=True)

    status_label = "PASS" if strict_gate_passed else "NOT READY"

    issue_rows = "\n".join(
        "<tr>"
        f"<td>{esc(issue.severity)}</td>"
        f"<td>{esc(issue.chunk_id)}</td>"
        f"<td>{esc(issue.field)}</td>"
        f"<td>{esc(issue.message)}</td>"
        "</tr>"
        for issue in state.issues
    )
    if not issue_rows:
        issue_rows = '<tr><td colspan="4">No validation issues found.</td></tr>'

    domain_rows = "\n".join(
        "<tr>"
        f"<td>{esc(domain)}</td>"
        f"<td>{esc(summary.by_domain.get(domain, 0))}</td>"
        f"<td>{esc(minimum)}-{esc(maximum)}</td>"
        "</tr>"
        for domain, (minimum, maximum) in DOMAIN_TARGETS.items()
    )

    source_rows = "\n".join(
        f"<tr><td>{esc(source_id)}</td><td>{esc(count)}</td></tr>"
        for source_id, count in sorted(
            summary.by_source.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )

    concept_rows = "\n".join(
        "<tr>"
        f"<td>{esc(concept)}</td>"
        f"<td>{esc(summary.by_label.get(concept, {}).get('positive', 0))}</td>"
        f"<td>{esc(summary.by_label.get(concept, {}).get('partial', 0))}</td>"
        f"<td>{esc(summary.by_label.get(concept, {}).get('negative', 0))}</td>"
        f"<td>{esc(summary.by_label.get(concept, {}).get('uncertain', 0))}</td>"
        f"<td>{esc(summary.by_label.get(concept, {}).get('not_reviewed', 0))}</td>"
        "</tr>"
        for concept in CONCEPTS
    )

    hard_negative_rows = "\n".join(
        f"<tr><td>{esc(category)}</td><td>{esc(count)}</td></tr>"
        for category, count in sorted(summary.hard_negative_categories.items())
    )
    if not hard_negative_rows:
        hard_negative_rows = '<tr><td colspan="2">No reviewed hard negatives yet.</td></tr>'

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>WTH Phase 1 Human Review Validation</title>
<style>
body {{
  font-family: Arial, sans-serif;
  margin: 32px;
  color: #1f2937;
}}
h1, h2 {{ color: #111827; }}
.status {{
  display: inline-block;
  padding: 8px 14px;
  border-radius: 8px;
  font-weight: 700;
  background: {"#dcfce7" if strict_gate_passed else "#fef3c7"};
}}
.cards {{
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin: 20px 0;
}}
.card {{
  border: 1px solid #d1d5db;
  border-radius: 8px;
  padding: 12px 16px;
  min-width: 150px;
}}
.card .value {{
  font-size: 24px;
  font-weight: 700;
}}
table {{
  border-collapse: collapse;
  width: 100%;
  margin-bottom: 28px;
}}
th, td {{
  border: 1px solid #d1d5db;
  padding: 8px;
  vertical-align: top;
  text-align: left;
}}
th {{ background: #f3f4f6; }}
.error {{ color: #b91c1c; }}
.warning {{ color: #92400e; }}
small {{ color: #6b7280; }}
</style>
</head>
<body>
<h1>WTH Phase 1 Human Review Validation</h1>
<p class="status">{esc(status_label)}</p>
<p><small>Review CSV: {esc(review_csv)}</small></p>

<div class="cards">
  <div class="card">
    <div>Review rows</div>
    <div class="value">{summary.total_rows}</div>
  </div>
  <div class="card">
    <div>Reviewed</div>
    <div class="value">{summary.reviewed_rows}</div>
  </div>
  <div class="card">
    <div>Approved</div>
    <div class="value">{summary.approved_rows}</div>
  </div>
  <div class="card">
    <div>Excluded</div>
    <div class="value">{summary.excluded_rows}</div>
  </div>
  <div class="card">
    <div>Needs source review</div>
    <div class="value">{summary.needs_source_review_rows}</div>
  </div>
  <div class="card">
    <div>Hard negatives</div>
    <div class="value">{summary.hard_negative_rows}</div>
  </div>
  <div class="card">
    <div>Multi-concept</div>
    <div class="value">{summary.multi_concept_rows}</div>
  </div>
</div>

<h2>Approved corpus by domain</h2>
<table>
<thead><tr><th>Domain</th><th>Approved</th><th>Target</th></tr></thead>
<tbody>{domain_rows}</tbody>
</table>

<h2>Concept-label coverage in approved corpus</h2>
<table>
<thead>
<tr>
<th>Concept</th><th>Positive</th><th>Partial</th>
<th>Negative</th><th>Uncertain</th><th>Not reviewed</th>
</tr>
</thead>
<tbody>{concept_rows}</tbody>
</table>

<h2>Reviewed hard negatives</h2>
<table>
<thead><tr><th>Category</th><th>Approved chunks</th></tr></thead>
<tbody>{hard_negative_rows}</tbody>
</table>

<h2>Approved corpus by source</h2>
<table>
<thead><tr><th>Source</th><th>Approved chunks</th></tr></thead>
<tbody>{source_rows}</tbody>
</table>

<h2>Validation issues</h2>
<table>
<thead><tr><th>Severity</th><th>Chunk</th><th>Field</th><th>Message</th></tr></thead>
<tbody>{issue_rows}</tbody>
</table>
</body>
</html>
"""


def prepare_output_paths(
    output_directory: Path,
    *,
    replace: bool,
) -> tuple[Path, Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    gold_path = output_directory / GOLD_CORPUS_FILENAME
    manifest_path = output_directory / MANIFEST_FILENAME
    report_path = output_directory / REPORT_FILENAME

    existing = [path for path in (gold_path, manifest_path, report_path) if path.exists()]
    if existing and not replace:
        raise ReviewValidationError(
            "Validator outputs already exist. Use --replace: "
            + ", ".join(path.as_posix() for path in existing)
        )
    return gold_path, manifest_path, report_path


def validate_phase1_human_review(
    *,
    project_root: Path,
    review_csv: Path,
    selection_manifest_path: Path,
    output_directory: Path,
    expected_review_rows: int,
    allow_incomplete: bool,
    replace: bool,
) -> bool:
    project_root = project_root.resolve()
    review_csv = resolve_from_project(project_root, review_csv)
    selection_manifest_path = resolve_from_project(
        project_root,
        selection_manifest_path,
    )
    output_directory = resolve_from_project(
        project_root,
        output_directory,
    )

    require_file(review_csv)
    require_file(selection_manifest_path)

    selection_manifest = load_json_mapping(selection_manifest_path)
    counts_value = selection_manifest.get("counts")
    if not isinstance(counts_value, Mapping):
        raise ReviewValidationError("Selection manifest has no counts object.")

    selected_count_value = counts_value.get("selected_candidates")
    if not isinstance(selected_count_value, int):
        raise ReviewValidationError("Selection manifest selected_candidates is missing or invalid.")

    if selected_count_value != expected_review_rows:
        raise ReviewValidationError(
            "Selection manifest candidate count does not match expected "
            f"review rows: {selected_count_value} != {expected_review_rows}"
        )

    rows, hard_negative_column = load_review_rows(review_csv)
    state = ValidationState()
    validate_unique_chunk_ids(rows, state)

    for row in rows:
        validate_review_row(
            row,
            hard_negative_column=hard_negative_column,
            state=state,
            allow_incomplete=allow_incomplete,
        )

    summary = build_summary(
        rows,
        hard_negative_column=hard_negative_column,
    )
    validate_exit_gate(
        rows,
        summary=summary,
        expected_review_rows=expected_review_rows,
        state=state,
        allow_incomplete=allow_incomplete,
    )

    strict_state = ValidationState()
    validate_unique_chunk_ids(rows, strict_state)
    for row in rows:
        validate_review_row(
            row,
            hard_negative_column=hard_negative_column,
            state=strict_state,
            allow_incomplete=False,
        )
    validate_exit_gate(
        rows,
        summary=summary,
        expected_review_rows=expected_review_rows,
        state=strict_state,
        allow_incomplete=False,
    )
    strict_gate_passed = not strict_state.errors

    gold_path, manifest_path, report_path = prepare_output_paths(
        output_directory,
        replace=replace,
    )

    report_path.write_text(
        render_html_report(
            summary=summary,
            state=state if allow_incomplete else strict_state,
            strict_gate_passed=strict_gate_passed,
            review_csv=review_csv,
        ),
        encoding="utf-8",
    )

    if strict_gate_passed:
        gold_records = [
            row_to_gold_record(
                row,
                hard_negative_column=hard_negative_column,
            )
            for row in rows
            if normalize_lower(row.get("review_decision")) in APPROVED_DECISIONS
        ]
        atomic_write_jsonl(gold_path, gold_records)
    elif gold_path.exists() and replace:
        gold_path.unlink()

    manifest: dict[str, object] = {
        "validator_version": VALIDATOR_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": (
            "phase1_human_review_complete"
            if strict_gate_passed
            else "phase1_human_review_in_progress"
        ),
        "strict_gate_passed": strict_gate_passed,
        "allow_incomplete": allow_incomplete,
        "inputs": {
            "review_csv": {
                "path": review_csv.as_posix(),
                "sha256": sha256_file(review_csv),
            },
            "selection_manifest": {
                "path": selection_manifest_path.as_posix(),
                "sha256": sha256_file(selection_manifest_path),
            },
        },
        "outputs": {
            "gold_corpus": (
                {
                    "path": gold_path.as_posix(),
                    "sha256": sha256_file(gold_path),
                }
                if strict_gate_passed
                else None
            ),
            "report": {
                "path": report_path.as_posix(),
                "sha256": sha256_file(report_path),
            },
        },
        "exit_gate": {
            "approved_count_range": [
                APPROVED_COUNT_MINIMUM,
                APPROVED_COUNT_MAXIMUM,
            ],
            "domain_targets": {
                domain: {
                    "minimum": minimum,
                    "maximum": maximum,
                }
                for domain, (minimum, maximum) in DOMAIN_TARGETS.items()
            },
            "minimum_positive_or_partial_per_concept": 20,
            "minimum_hard_negative_chunks": 30,
            "requires_all_rows_reviewed": True,
            "requires_no_needs_source_review": True,
            "requires_verified_citations_for_approved": True,
            "requires_acceptable_text_and_ocr_for_approved": True,
        },
        "summary": summary_to_dict(summary),
        "errors": [
            {
                "chunk_id": issue.chunk_id,
                "field": issue.field,
                "message": issue.message,
            }
            for issue in strict_state.errors
        ],
        "warnings": [
            {
                "chunk_id": issue.chunk_id,
                "field": issue.field,
                "message": issue.message,
            }
            for issue in strict_state.warnings
        ],
        "next_step": (
            "Freeze build/development/held-out evaluation sets."
            if strict_gate_passed
            else "Continue human review and rerun validation."
        ),
    }
    atomic_write_json(manifest_path, manifest)

    LOGGER.info("Phase 1 human-review validation completed")
    LOGGER.info("Review rows: %d", summary.total_rows)
    LOGGER.info("Approved rows: %d", summary.approved_rows)
    LOGGER.info("Errors: %d", len(strict_state.errors))
    LOGGER.info("Warnings: %d", len(strict_state.warnings))
    LOGGER.info("Strict exit gate passed: %s", strict_gate_passed)
    LOGGER.info("Report: %s", report_path)
    LOGGER.info("Manifest: %s", manifest_path)
    if strict_gate_passed:
        LOGGER.info("Gold corpus: %s", gold_path)

    if strict_gate_passed:
        return True

    if allow_incomplete:
        return False

    raise ReviewValidationError(
        "Phase 5 exit gate did not pass. "
        "Open the validation report for row-level issues, or use "
        "--allow-incomplete during review."
    )


def main() -> int:
    arguments = parse_arguments()
    configure_logging(arguments.log_level)
    try:
        passed = validate_phase1_human_review(
            project_root=arguments.project_root,
            review_csv=arguments.review_csv,
            selection_manifest_path=arguments.selection_manifest,
            output_directory=arguments.output_directory,
            expected_review_rows=arguments.expected_review_rows,
            allow_incomplete=arguments.allow_incomplete,
            replace=arguments.replace,
        )
    except ReviewValidationError:
        LOGGER.exception("Phase 1 human review validation failed")
        return 1

    if not passed and arguments.allow_incomplete:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
