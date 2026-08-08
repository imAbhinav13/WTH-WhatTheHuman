from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import logging
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import yaml

LOGGER = logging.getLogger("wth.phase1.generate_phase1_selection_review_packet")

SCRIPT_VERSION: Final = "1.0.0"
DEFAULT_SELECTION_CANDIDATES: Final = Path(
    "artifacts/phase1/selection/phase1_selection_candidates.jsonl"
)
DEFAULT_SELECTION_MANIFEST: Final = Path(
    "artifacts/phase1/selection/phase1_selection_manifest.json"
)
DEFAULT_CATALOGUE: Final = Path("docs/catalogues/phase1_sources.yaml")
DEFAULT_ACQUISITION_MANIFEST: Final = Path("artifacts/phase1/acquisition_manifest.json")
DEFAULT_OUTPUT_CSV: Final = Path("artifacts/review/phase1_selection_review_packet.csv")
DEFAULT_OUTPUT_HTML: Final = Path("artifacts/review/phase1_selection_review_packet.html")
DEFAULT_EXPECTED_STATUS: Final = "pre_review_candidates"

CONCEPTS: Final = (
    "consciousness",
    "self_identity",
    "reality_appearance",
)
CONCEPT_LABEL_VALUES: Final = (
    "positive",
    "partial",
    "negative",
    "uncertain",
    "not_reviewed",
)
REVIEW_DECISION_VALUES: Final = (
    "include",
    "include_with_edits",
    "exclude",
    "needs_source_review",
)
CITATION_VERIFIED_VALUES: Final = (
    "yes",
    "no",
    "uncertain",
    "not_reviewed",
)
TEXT_QUALITY_VALUES: Final = (
    "acceptable",
    "needs_edit",
    "unusable",
    "uncertain",
    "not_reviewed",
)
OCR_REVIEW_VALUES: Final = (
    "not_needed",
    "acceptable",
    "needs_correction",
    "needs_source_review",
    "unusable",
    "not_reviewed",
)

CSV_FIELDS: Final = (
    # Provenance
    "chunk_id",
    "source_id",
    "domain",
    "source_title",
    "author",
    "translator",
    "publication_year",
    "citation",
    "section_title",
    "structural_locator",
    "chunk_text",
    "token_count",
    "source_checksum",
    "text_checksum",
    # Selection evidence
    "selection_rank",
    "selection_class",
    "selection_score",
    "selection_rule_ids",
    "matched_terms",
    "eligible_concepts",
    "hard_negative_category",
    "selection_rationale",
    "parser_warnings",
    "review_flags",
    "structure_report_action",
    "ocr_noise_score",
    "severe_ocr_noise",
    # Human review
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
    "hard_negative_category_reviewed",
    "citation_verified",
    "text_quality_status",
    "ocr_review_status",
    "edited_text",
)


class ReviewPacketError(RuntimeError):
    """Raised when the Phase 1 review packet cannot be generated safely."""


@dataclass(frozen=True)
class SourceMetadata:
    source_id: str
    title: str
    author: str
    translator: str
    publication_year: str
    citation: str
    checksum: str


@dataclass(frozen=True)
class ReviewRow:
    values: dict[str, str]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the Phase 1 selection-review packet from deterministic pre-review candidates."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--selection-candidates",
        type=Path,
        default=DEFAULT_SELECTION_CANDIDATES,
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=DEFAULT_SELECTION_MANIFEST,
    )
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    parser.add_argument(
        "--acquisition-manifest",
        type=Path,
        default=DEFAULT_ACQUISITION_MANIFEST,
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-html", type=Path, default=DEFAULT_OUTPUT_HTML)
    parser.add_argument(
        "--expected-status",
        default=DEFAULT_EXPECTED_STATUS,
        help="Expected status in the selection manifest.",
    )
    parser.add_argument(
        "--allow-missing-source-checksum",
        action="store_true",
        help=(
            "Allow review-packet generation when one or more source checksums "
            "cannot be recovered from catalogue/acquisition metadata."
        ),
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


def resolve_from_project(project_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise ReviewPacketError(f"Required file does not exist: {path}")


def require_mapping(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ReviewPacketError(f"{description} must be an object")
    result: dict[str, object] = {}
    for raw_key, nested_value in value.items():
        if not isinstance(raw_key, str):
            raise ReviewPacketError(f"{description} contains a non-string key")
        result[raw_key] = nested_value
    return result


def optional_string(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, int | float) and not isinstance(value, bool):
        return str(value)
    return ""


def optional_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def optional_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def require_string(value: object, description: str) -> str:
    normalized = optional_string(value)
    if not normalized:
        raise ReviewPacketError(f"{description} must be a non-empty string")
    return normalized


def normalize_string_list(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        normalized = value.strip()
        return (normalized,) if normalized else ()
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        result: list[str] = []
        for item in value:
            normalized = optional_string(item)
            if normalized:
                result.append(normalized)
        return tuple(dict.fromkeys(result))
    return ()


def stringify_list(value: object, delimiter: str = " | ") -> str:
    return delimiter.join(normalize_string_list(value))


def load_json(path: Path) -> object:
    try:
        loaded: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReviewPacketError(f"Invalid JSON in {path}: {exc}") from exc
    return loaded


def load_yaml(path: Path) -> object:
    try:
        loaded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ReviewPacketError(f"Invalid YAML in {path}: {exc}") from exc
    return loaded


def iter_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                loaded: object = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ReviewPacketError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            yield require_mapping(loaded, f"candidate record at line {line_number}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def find_first_key(value: object, target_key: str) -> object | None:
    if isinstance(value, Mapping):
        mapping = require_mapping(value, "recursive mapping")
        for key, nested_value in mapping.items():
            if key.casefold() == target_key.casefold():
                return nested_value
        for nested_value in mapping.values():
            result = find_first_key(nested_value, target_key)
            if result is not None:
                return result
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for nested_value in value:
            result = find_first_key(nested_value, target_key)
            if result is not None:
                return result
    return None


def metadata_string(value: object) -> str:
    direct = optional_string(value)
    if direct:
        return direct
    values = normalize_string_list(value)
    return "; ".join(values)


def first_non_empty(mapping: Mapping[str, object], keys: Sequence[str]) -> str:
    for key in keys:
        value = metadata_string(mapping.get(key))
        if value:
            return value
    return ""


def extract_source_records(raw_catalogue: object) -> list[dict[str, object]]:
    if isinstance(raw_catalogue, list):
        return [
            require_mapping(item, f"catalogue source {index}")
            for index, item in enumerate(raw_catalogue, start=1)
        ]
    catalogue = require_mapping(raw_catalogue, "catalogue")
    for key in ("sources", "source_catalogue", "records"):
        raw_sources = catalogue.get(key)
        if isinstance(raw_sources, list):
            return [
                require_mapping(item, f"catalogue {key} source {index}")
                for index, item in enumerate(raw_sources, start=1)
            ]
    raise ReviewPacketError(
        "Source catalogue must be a YAML list or contain a list under "
        "'sources', 'source_catalogue', or 'records'."
    )


def build_fallback_citation(
    *,
    author: str,
    title: str,
    publication_year: str,
    translator: str,
) -> str:
    parts: list[str] = []
    if author:
        parts.append(author)
    if title:
        parts.append(title)
    if translator:
        parts.append(f"trans. {translator}")
    if publication_year:
        parts.append(publication_year)
    return ". ".join(parts)


def load_catalogue_metadata(path: Path) -> dict[str, SourceMetadata]:
    records = extract_source_records(load_yaml(path))
    result: dict[str, SourceMetadata] = {}

    for index, record in enumerate(records, start=1):
        source_id = require_string(
            record.get("source_id"),
            f"catalogue record {index}.source_id",
        )
        if source_id in result:
            raise ReviewPacketError(f"Duplicate source_id in catalogue: {source_id}")

        title = first_non_empty(record, ("title", "source_title", "name"))
        author = first_non_empty(record, ("author", "authors", "creator"))
        translator = first_non_empty(record, ("translator", "translated_by"))
        publication_year = first_non_empty(
            record,
            ("publication_year", "year", "published_year"),
        )
        citation = first_non_empty(
            record,
            ("citation", "preferred_citation", "bibliographic_citation"),
        )
        checksum = first_non_empty(
            record,
            (
                "source_checksum",
                "checksum",
                "sha256",
                "content_checksum",
            ),
        )

        if not citation:
            citation = build_fallback_citation(
                author=author,
                title=title,
                publication_year=publication_year,
                translator=translator,
            )

        result[source_id] = SourceMetadata(
            source_id=source_id,
            title=title,
            author=author,
            translator=translator,
            publication_year=publication_year,
            citation=citation,
            checksum=checksum,
        )

    return result


def extract_acquisition_records(raw: object) -> list[dict[str, object]]:
    if isinstance(raw, list):
        return [
            require_mapping(item, f"acquisition record {index}")
            for index, item in enumerate(raw, start=1)
        ]
    mapping = require_mapping(raw, "acquisition manifest")
    for key in ("sources", "artifacts", "records", "acquisitions"):
        nested = mapping.get(key)
        if isinstance(nested, list):
            return [
                require_mapping(item, f"acquisition {key} record {index}")
                for index, item in enumerate(nested, start=1)
            ]
    return [mapping]


def load_acquisition_checksums(path: Path) -> dict[str, str]:
    records = extract_acquisition_records(load_json(path))
    checksums: dict[str, str] = {}
    for record in records:
        source_id = first_non_empty(record, ("source_id", "id"))
        if not source_id:
            nested_source_id = optional_string(find_first_key(record, "source_id"))
            source_id = nested_source_id
        if not source_id:
            continue

        checksum = first_non_empty(
            record,
            (
                "source_checksum",
                "checksum",
                "sha256",
                "content_checksum",
                "artifact_checksum",
            ),
        )
        if not checksum:
            for key in (
                "source_checksum",
                "checksum",
                "sha256",
                "content_checksum",
                "artifact_checksum",
            ):
                checksum = optional_string(find_first_key(record, key))
                if checksum:
                    break
        if checksum:
            checksums[source_id] = checksum
    return checksums


def validate_selection_manifest(
    path: Path,
    *,
    expected_status: str,
) -> dict[str, object]:
    manifest = require_mapping(load_json(path), "selection manifest")
    status = require_string(manifest.get("status"), "selection manifest status")
    if status != expected_status:
        raise ReviewPacketError(
            f"Selection manifest status must be {expected_status!r}; found {status!r}"
        )

    counts = require_mapping(manifest.get("counts"), "selection manifest counts")
    selected_count = optional_int(counts.get("selected_candidates"), -1)
    if selected_count <= 0:
        raise ReviewPacketError(
            "Selection manifest does not contain a positive selected_candidates count"
        )

    embedding_independent = manifest.get("embedding_independent")
    if embedding_independent is not True:
        raise ReviewPacketError("Selection manifest does not assert embedding_independent=true")
    return manifest


def flatten_hit_mapping(value: object) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        return normalize_string_list(value)
    mapping = require_mapping(value, "selection hit mapping")
    result: list[str] = []
    for hits in mapping.values():
        result.extend(normalize_string_list(hits))
    return tuple(dict.fromkeys(result))


def build_selection_rule_ids(candidate: Mapping[str, object]) -> tuple[str, ...]:
    result: list[str] = []
    result.extend(normalize_string_list(candidate.get("source_include_rule_ids")))
    result.extend(
        f"concept_lexical:{concept_id}"
        for concept_id in normalize_string_list(candidate.get("active_concepts"))
    )
    result.extend(
        f"hard_negative:{category}"
        for category in normalize_string_list(candidate.get("hard_negative_categories"))
    )
    return tuple(dict.fromkeys(result))


def build_matched_terms(candidate: Mapping[str, object]) -> tuple[str, ...]:
    result: list[str] = []
    for field_name in (
        "concept_positive_hits",
        "concept_context_hits",
        "source_include_pattern_hits",
        "source_exclude_pattern_hits",
    ):
        result.extend(flatten_hit_mapping(candidate.get(field_name)))
    return tuple(dict.fromkeys(result))


def build_selection_rationale(candidate: Mapping[str, object]) -> str:
    selection_class = optional_string(candidate.get("selection_class"))
    eligible_concepts = normalize_string_list(candidate.get("active_concepts"))
    hard_negatives = normalize_string_list(candidate.get("hard_negative_categories"))
    source_rules = normalize_string_list(candidate.get("source_include_rule_ids"))
    review_flags = normalize_string_list(candidate.get("review_flags"))

    parts: list[str] = []
    if selection_class:
        parts.append(f"class={selection_class}")
    if eligible_concepts:
        parts.append("eligible_concepts=" + ", ".join(eligible_concepts))
    if hard_negatives:
        parts.append("hard_negative=" + ", ".join(hard_negatives))
    if source_rules:
        parts.append("source_rules=" + ", ".join(source_rules))
    if review_flags:
        parts.append("review_flags=" + ", ".join(review_flags))
    if not parts:
        return "Selected by deterministic Phase 1 structural and lexical policy."
    return "; ".join(parts)


def checksum_from_candidate(candidate: Mapping[str, object], text: str) -> str:
    for key in ("text_checksum", "content_checksum", "checksum"):
        checksum = optional_string(candidate.get(key))
        if checksum:
            return checksum
    return sha256_text(text)


def quality_value(candidate: Mapping[str, object], key: str) -> object | None:
    raw_quality = candidate.get("quality")
    if isinstance(raw_quality, Mapping):
        quality = require_mapping(raw_quality, "candidate quality")
        return quality.get(key)
    return None


def candidate_sort_key(record: Mapping[str, object]) -> tuple[int, str]:
    return (
        optional_int(record.get("selection_rank"), 2**31 - 1),
        optional_string(record.get("chunk_id")),
    )


def create_review_row(
    candidate: Mapping[str, object],
    *,
    catalogue_metadata: Mapping[str, SourceMetadata],
    acquisition_checksums: Mapping[str, str],
) -> ReviewRow:
    chunk_id = require_string(candidate.get("chunk_id"), "candidate.chunk_id")
    source_id = require_string(candidate.get("source_id"), f"{chunk_id}.source_id")
    text = require_string(candidate.get("text"), f"{chunk_id}.text")

    source_metadata = catalogue_metadata.get(
        source_id,
        SourceMetadata(
            source_id=source_id,
            title="",
            author="",
            translator="",
            publication_year="",
            citation="",
            checksum="",
        ),
    )

    source_title = optional_string(candidate.get("source_title")) or source_metadata.title
    source_checksum = source_metadata.checksum or acquisition_checksums.get(source_id, "")
    citation = optional_string(candidate.get("citation")) or source_metadata.citation

    active_concepts = normalize_string_list(candidate.get("active_concepts"))
    hard_negative_categories = normalize_string_list(candidate.get("hard_negative_categories"))
    hard_negative_targets = normalize_string_list(candidate.get("hard_negative_targets"))

    values: dict[str, str] = {
        # Provenance
        "chunk_id": chunk_id,
        "source_id": source_id,
        "domain": optional_string(candidate.get("domain")),
        "source_title": source_title,
        "author": source_metadata.author,
        "translator": source_metadata.translator,
        "publication_year": source_metadata.publication_year,
        "citation": citation,
        "section_title": optional_string(candidate.get("section_title")),
        "structural_locator": optional_string(candidate.get("structural_locator")),
        "chunk_text": text,
        "token_count": str(optional_int(candidate.get("token_count"), 0)),
        "source_checksum": source_checksum,
        "text_checksum": checksum_from_candidate(candidate, text),
        # Selection evidence
        "selection_rank": str(optional_int(candidate.get("selection_rank"), 0)),
        "selection_class": optional_string(candidate.get("selection_class")),
        "selection_score": str(optional_float(candidate.get("selection_score"), 0.0)),
        "selection_rule_ids": " | ".join(build_selection_rule_ids(candidate)),
        "matched_terms": " | ".join(build_matched_terms(candidate)),
        "eligible_concepts": " | ".join(active_concepts),
        "hard_negative_category": " | ".join(hard_negative_categories),
        "selection_rationale": build_selection_rationale(candidate),
        "parser_warnings": stringify_list(candidate.get("parser_warnings")),
        "review_flags": stringify_list(candidate.get("review_flags")),
        "structure_report_action": optional_string(candidate.get("structure_report_action")),
        "ocr_noise_score": str(optional_float(quality_value(candidate, "ocr_noise_score"), 0.0)),
        "severe_ocr_noise": str(quality_value(candidate, "severe_ocr_noise") is True).lower(),
        # Human review - deliberately blank/defaulted
        "review_decision": "",
        "reviewer": "",
        "reviewed_at": "",
        "review_notes": "",
        "consciousness_label": "not_reviewed",
        "self_identity_label": "not_reviewed",
        "reality_appearance_label": "not_reviewed",
        "primary_concept": "",
        "secondary_concepts": "",
        "hard_negative_for": " | ".join(hard_negative_targets),
        "hard_negative_category_reviewed": "",
        "citation_verified": "not_reviewed",
        "text_quality_status": "not_reviewed",
        "ocr_review_status": "not_reviewed",
        "edited_text": "",
    }
    return ReviewRow(values=values)


def validate_rows(
    rows: Sequence[ReviewRow],
    *,
    expected_count: int,
    allow_missing_source_checksum: bool,
) -> list[str]:
    if len(rows) != expected_count:
        raise ReviewPacketError(
            f"Review row count mismatch: expected {expected_count}, generated {len(rows)}"
        )

    chunk_ids: set[str] = set()
    warnings: list[str] = []
    missing_source_checksums: list[str] = []
    missing_citations: list[str] = []

    for row in rows:
        values = row.values
        chunk_id = values["chunk_id"]
        if chunk_id in chunk_ids:
            raise ReviewPacketError(f"Duplicate chunk_id in review packet: {chunk_id}")
        chunk_ids.add(chunk_id)

        if not values["source_checksum"]:
            missing_source_checksums.append(values["source_id"])
        if not values["citation"]:
            missing_citations.append(chunk_id)
        if not values["text_checksum"]:
            raise ReviewPacketError(f"Missing text checksum for {chunk_id}")

    unique_missing_sources = sorted(set(missing_source_checksums))
    if unique_missing_sources:
        message = "Source checksum missing for: " + ", ".join(unique_missing_sources)
        if not allow_missing_source_checksum:
            raise ReviewPacketError(
                message + ". Use --allow-missing-source-checksum only if this is an "
                "accepted Phase 4 review limitation."
            )
        warnings.append(message)

    if missing_citations:
        warnings.append(f"Citation metadata is missing for {len(missing_citations)} review rows.")

    return warnings


def prepare_output(path: Path, *, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise ReviewPacketError(f"Output exists; rerun with --replace: {path}")


def atomic_write_csv(path: Path, rows: Sequence[ReviewRow]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.values)
    temporary_path.replace(path)


def html_badge(text: str, class_name: str = "") -> str:
    class_attribute = f" {class_name}" if class_name else ""
    return f'<span class="badge{class_attribute}">{html.escape(text)}</span>'


def render_multivalue_badges(value: str) -> str:
    values = [item.strip() for item in value.split("|") if item.strip()]
    if not values:
        return '<span class="muted">—</span>'
    return " ".join(html_badge(item) for item in values)


def render_source_summary(rows: Sequence[ReviewRow]) -> str:
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        values = row.values
        counts[(values["domain"], values["source_title"] or values["source_id"])] += 1
    table_rows = []
    for (domain, source_title), count in sorted(counts.items()):
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(domain)}</td>"
            f"<td>{html.escape(source_title)}</td>"
            f'<td class="number">{count}</td>'
            "</tr>"
        )
    return "\n".join(table_rows)


def render_domain_summary(rows: Sequence[ReviewRow]) -> str:
    counts = Counter(row.values["domain"] for row in rows)
    return "\n".join(
        f'<tr><td>{html.escape(domain)}</td><td class="number">{count}</td></tr>'
        for domain, count in sorted(counts.items())
    )


def render_class_summary(rows: Sequence[ReviewRow]) -> str:
    counts = Counter(row.values["selection_class"] for row in rows)
    return "\n".join(
        f'<tr><td>{html.escape(selection_class)}</td><td class="number">{count}</td></tr>'
        for selection_class, count in sorted(counts.items())
    )


def render_review_cards(rows: Sequence[ReviewRow]) -> str:
    cards: list[str] = []
    for row in rows:
        values = row.values
        cards.append(
            f"""
<section class="review-card" data-domain="{html.escape(values["domain"])}"
         data-source="{html.escape(values["source_id"])}"
         data-class="{html.escape(values["selection_class"])}">
  <div class="card-header">
    <div>
      <span class="rank">#{html.escape(values["selection_rank"])}</span>
      <strong>{html.escape(values["source_title"] or values["source_id"])}</strong>
    </div>
    <div>
      {html_badge(values["domain"], "domain")}
      {html_badge(values["selection_class"], "selection-class")}
    </div>
  </div>

  <div class="meta-grid">
    <div><span>Chunk</span><code>{html.escape(values["chunk_id"])}</code></div>
    <div><span>Section</span>{html.escape(values["section_title"]) or "—"}</div>
    <div><span>Locator</span>{html.escape(values["structural_locator"]) or "—"}</div>
    <div><span>Tokens</span>{html.escape(values["token_count"])}</div>
    <div><span>Author</span>{html.escape(values["author"]) or "—"}</div>
    <div><span>Translator</span>{html.escape(values["translator"]) or "—"}</div>
    <div><span>Year</span>{html.escape(values["publication_year"]) or "—"}</div>
    <div><span>OCR score</span>{html.escape(values["ocr_noise_score"])}</div>
  </div>

  <div class="citation"><strong>Citation:</strong> {html.escape(values["citation"]) or "—"}</div>

  <div class="evidence-grid">
    <div>
      <strong>Eligible concepts</strong><br>
      {render_multivalue_badges(values['eligible_concepts'])}
    </div>
    <div>
      <strong>Hard negative</strong><br>
      {render_multivalue_badges(values['hard_negative_category'])}
    </div>
    <div>
      <strong>Rules</strong><br>
      {render_multivalue_badges(values['selection_rule_ids'])}
    </div>
    <div>
      <strong>Matched terms</strong><br>
      {render_multivalue_badges(values['matched_terms'])}
    </div>
  </div>

  <div class="rationale">
    <strong>Selection rationale:</strong> {html.escape(values['selection_rationale'])}
  </div>
  <div class="warnings">
    <strong>Parser warnings:</strong> {html.escape(values['parser_warnings']) or '—'}
  </div>
  <div class="review-flags">
    <strong>Review flags:</strong> {html.escape(values['review_flags']) or '—'}
  </div>

  <div class="rationale">
    <strong>Selection rationale:</strong>
    {html.escape(values['selection_rationale'])}
  </div>
  <div class="warnings">
    <strong>Parser warnings:</strong>
    {html.escape(values['parser_warnings']) or '—'}
  </div>
  <div class="review-flags">
    <strong>Review flags:</strong>
    {html.escape(values['review_flags']) or '—'}
  </div>

  <blockquote>{html.escape(values["chunk_text"])}</blockquote>

  <div class="review-template">
    <strong>Human review</strong>
    <div class="review-grid">
      <div>Decision: ____________________</div>
      <div>Reviewer: ____________________</div>
      <div>Consciousness: ____________________</div>
      <div>Self / identity: ____________________</div>
      <div>Reality / appearance: ____________________</div>
      <div>Primary concept: ____________________</div>
      <div>Citation verified: ____________________</div>
      <div>Text quality: ____________________</div>
      <div>OCR review: ____________________</div>
      <div>Hard negative: ____________________</div>
    </div>
    <div class="notes">Notes: ________________________________________________________________</div>
  </div>
</section>
"""
        )
    return "\n".join(cards)


def atomic_write_html(
    path: Path,
    *,
    rows: Sequence[ReviewRow],
    generated_at: str,
    selection_manifest_sha256: str,
    warnings: Sequence[str],
) -> None:
    warning_html = ""
    if warnings:
        items = "".join(f"<li>{html.escape(item)}</li>" for item in warnings)
        warning_html = (
            f'<div class="notice warning"><strong>Warnings</strong><ul>{items}</ul></div>'
        )

    content = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WTH Phase 1 Selection Review Packet</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f5f7fa;
  --panel: #ffffff;
  --ink: #17202a;
  --muted: #667085;
  --line: #d0d5dd;
  --soft: #eef2f6;
  --accent: #344054;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: Inter, Segoe UI, Arial, sans-serif;
  background: var(--bg);
  color: var(--ink);
}}
main {{ max-width: 1480px; margin: 0 auto; padding: 32px; }}
h1 {{ margin-bottom: 8px; }}
h2 {{ margin-top: 36px; }}
.subtitle {{ color: var(--muted); max-width: 980px; line-height: 1.55; }}
.notice {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 14px 18px;
  margin: 18px 0;
}}
.warning {{ border-left: 5px solid #667085; }}
.summary-grid {{
  display: grid;
  grid-template-columns: repeat(3, minmax(240px, 1fr));
  gap: 16px;
}}
.summary-card {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 16px;
}}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
th, td {{
  border-bottom: 1px solid var(--line);
  padding: 7px 8px;
  text-align: left;
  vertical-align: top;
}}
th {{ background: var(--soft); }}
.number {{ text-align: right; font-variant-numeric: tabular-nums; }}
.filters {{
  position: sticky;
  top: 0;
  z-index: 20;
  background: rgba(245,247,250,.96);
  padding: 12px 0;
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}}
.filters input, .filters select {{
  padding: 8px 10px;
  border: 1px solid var(--line);
  border-radius: 7px;
  background: white;
}}
.review-card {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 20px;
  margin: 18px 0;
  box-shadow: 0 1px 2px rgba(16,24,40,.04);
}}
.card-header {{
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: center;
  border-bottom: 1px solid var(--line);
  padding-bottom: 12px;
}}
.rank {{ font-variant-numeric: tabular-nums; color: var(--muted); margin-right: 8px; }}
.badge {{
  display: inline-block;
  padding: 3px 8px;
  border-radius: 999px;
  background: var(--soft);
  border: 1px solid var(--line);
  margin: 2px;
  font-size: 12px;
}}
.domain {{ font-weight: 600; }}
.meta-grid, .evidence-grid, .review-grid {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px 18px;
  margin: 14px 0;
}}
.meta-grid div {{ display: grid; grid-template-columns: 90px 1fr; gap: 6px; }}
.meta-grid span {{ color: var(--muted); }}
code {{ overflow-wrap: anywhere; }}
.citation, .rationale, .warnings, .review-flags {{ margin: 10px 0; line-height: 1.45; }}
blockquote {{
  white-space: pre-wrap;
  background: #fafafa;
  border-left: 4px solid #98a2b3;
  margin: 18px 0;
  padding: 16px;
  line-height: 1.6;
}}
.review-template {{
  background: #fcfcfd;
  border: 1px dashed #98a2b3;
  border-radius: 8px;
  padding: 14px;
}}
.notes {{ margin-top: 10px; }}
.muted {{ color: var(--muted); }}
.footer {{ color: var(--muted); margin: 36px 0 12px; font-size: 13px; }}
@media (max-width: 900px) {{
  main {{ padding: 18px; }}
  .summary-grid, .meta-grid, .evidence-grid, .review-grid {{ grid-template-columns: 1fr; }}
  .card-header {{ align-items: flex-start; flex-direction: column; }}
}}
@media print {{
  body {{ background: white; }}
  main {{ max-width: none; padding: 0; }}
  .filters {{ display: none; }}
  .review-card {{ break-inside: avoid; box-shadow: none; }}
}}
</style>
</head>
<body>
<main>
  <h1>WTH Phase 1 Selection Review Packet</h1>
  <p class="subtitle">
    Human-review packet for the deterministic, embedding-independent Phase 1
    three-concept vertical slice: consciousness, self/identity, and
    reality/appearance across Science, Advaita Vedanta, and Samkhya.
    Automated evidence is advisory only; human review is authoritative.
  </p>

  <div class="notice">
    <strong>Review vocabulary</strong><br>
    Concept labels: {html.escape(", ".join(CONCEPT_LABEL_VALUES))}<br>
    Review decisions: {html.escape(", ".join(REVIEW_DECISION_VALUES))}<br>
    Citation status: {html.escape(", ".join(CITATION_VERIFIED_VALUES))}<br>
    Text quality: {html.escape(", ".join(TEXT_QUALITY_VALUES))}<br>
    OCR review: {html.escape(", ".join(OCR_REVIEW_VALUES))}
  </div>
  {warning_html}

  <div class="summary-grid">
    <div class="summary-card">
      <h3>By domain</h3>
      <table><thead><tr><th>Domain</th><th>Chunks</th></tr></thead><tbody>
      {render_domain_summary(rows)}
      </tbody></table>
    </div>
    <div class="summary-card">
      <h3>By selection class</h3>
      <table><thead><tr><th>Class</th><th>Chunks</th></tr></thead><tbody>
      {render_class_summary(rows)}
      </tbody></table>
    </div>
    <div class="summary-card">
      <h3>Packet metadata</h3>
      <table><tbody>
        <tr><td>Rows</td><td class="number">{len(rows)}</td></tr>
        <tr><td>Generated</td><td>{html.escape(generated_at)}</td></tr>
        <tr><td>Script</td><td>{html.escape(SCRIPT_VERSION)}</td></tr>
        <tr>
          <td>Selection manifest SHA-256</td>
          <td><code>{html.escape(selection_manifest_sha256)}</code></td>
        </tr>
      </tbody></table>
    </div>
  </div>

  <h2>Source representation</h2>
  <table>
    <thead><tr><th>Domain</th><th>Source</th><th>Chunks</th></tr></thead>
    <tbody>{render_source_summary(rows)}</tbody>
  </table>

  <h2>Candidate review</h2>
  <div class="filters">
    <input id="searchBox" type="search" placeholder="Search source, chunk, text...">
    <select id="domainFilter">
      <option value="">All domains</option>
      <option value="science">Science</option>
      <option value="advaita">Advaita</option>
      <option value="samkhya">Samkhya</option>
    </select>
    <select id="classFilter">
      <option value="">All classes</option>
      <option value="positive">Positive</option>
      <option value="mixed_positive_hard_negative">Mixed positive/hard negative</option>
      <option value="hard_negative">Hard negative</option>
      <option value="ambiguous_review">Ambiguous review</option>
    </select>
    <span id="visibleCount" class="muted"></span>
  </div>

  <div id="reviewCards">
    {render_review_cards(rows)}
  </div>

  <div class="footer">
    This HTML is a review aid. Persist authoritative review decisions in the CSV
    or the subsequent review-ingestion artifact, not by editing this HTML.
  </div>
</main>
<script>
const cards = Array.from(document.querySelectorAll('.review-card'));
const searchBox = document.getElementById('searchBox');
const domainFilter = document.getElementById('domainFilter');
const classFilter = document.getElementById('classFilter');
const visibleCount = document.getElementById('visibleCount');
function updateFilters() {{
  const query = searchBox.value.trim().toLowerCase();
  const domain = domainFilter.value;
  const cls = classFilter.value;
  let visible = 0;
  for (const card of cards) {{
    const matchesQuery = !query || card.textContent.toLowerCase().includes(query);
    const matchesDomain = !domain || card.dataset.domain === domain;
    const matchesClass = !cls || card.dataset.class === cls;
    const show = matchesQuery && matchesDomain && matchesClass;
    card.style.display = show ? '' : 'none';
    if (show) visible += 1;
  }}
  visibleCount.textContent = `${{visible}} of ${{cards.length}} shown`;
}}
searchBox.addEventListener('input', updateFilters);
domainFilter.addEventListener('change', updateFilters);
classFilter.addEventListener('change', updateFilters);
updateFilters();
</script>
</body>
</html>
"""

    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def generate_review_packet(
    *,
    project_root: Path,
    selection_candidates_path: Path,
    selection_manifest_path: Path,
    catalogue_path: Path,
    acquisition_manifest_path: Path,
    output_csv_path: Path,
    output_html_path: Path,
    expected_status: str,
    allow_missing_source_checksum: bool,
    replace: bool,
) -> tuple[int, list[str]]:
    project_root = project_root.resolve()
    selection_candidates_path = resolve_from_project(
        project_root,
        selection_candidates_path,
    )
    selection_manifest_path = resolve_from_project(
        project_root,
        selection_manifest_path,
    )
    catalogue_path = resolve_from_project(project_root, catalogue_path)
    acquisition_manifest_path = resolve_from_project(
        project_root,
        acquisition_manifest_path,
    )
    output_csv_path = resolve_from_project(project_root, output_csv_path)
    output_html_path = resolve_from_project(project_root, output_html_path)

    for required_path in (
        selection_candidates_path,
        selection_manifest_path,
        catalogue_path,
        acquisition_manifest_path,
    ):
        require_file(required_path)

    manifest = validate_selection_manifest(
        selection_manifest_path,
        expected_status=expected_status,
    )
    manifest_counts = require_mapping(manifest.get("counts"), "selection manifest counts")
    expected_count = optional_int(manifest_counts.get("selected_candidates"), -1)
    if expected_count <= 0:
        raise ReviewPacketError("Invalid selected candidate count in selection manifest")

    catalogue_metadata = load_catalogue_metadata(catalogue_path)
    acquisition_checksums = load_acquisition_checksums(acquisition_manifest_path)

    candidate_records = list(iter_jsonl(selection_candidates_path))
    candidate_records.sort(key=candidate_sort_key)

    rows = [
        create_review_row(
            candidate,
            catalogue_metadata=catalogue_metadata,
            acquisition_checksums=acquisition_checksums,
        )
        for candidate in candidate_records
    ]
    warnings = validate_rows(
        rows,
        expected_count=expected_count,
        allow_missing_source_checksum=allow_missing_source_checksum,
    )

    prepare_output(output_csv_path, replace=replace)
    prepare_output(output_html_path, replace=replace)

    atomic_write_csv(output_csv_path, rows)
    generated_at = datetime.now(UTC).isoformat()
    atomic_write_html(
        output_html_path,
        rows=rows,
        generated_at=generated_at,
        selection_manifest_sha256=sha256_file(selection_manifest_path),
        warnings=warnings,
    )

    LOGGER.info("Phase 1 selection-review packet generated successfully")
    LOGGER.info("Review rows: %d", len(rows))
    LOGGER.info("CSV: %s", output_csv_path)
    LOGGER.info("HTML: %s", output_html_path)
    for warning in warnings:
        LOGGER.warning("%s", warning)

    return len(rows), warnings


def main() -> int:
    arguments = parse_arguments()
    configure_logging(arguments.log_level)
    try:
        generate_review_packet(
            project_root=arguments.project_root,
            selection_candidates_path=arguments.selection_candidates,
            selection_manifest_path=arguments.selection_manifest,
            catalogue_path=arguments.catalogue,
            acquisition_manifest_path=arguments.acquisition_manifest,
            output_csv_path=arguments.output_csv,
            output_html_path=arguments.output_html,
            expected_status=arguments.expected_status,
            allow_missing_source_checksum=arguments.allow_missing_source_checksum,
            replace=arguments.replace,
        )
    except ReviewPacketError:
        LOGGER.exception("Phase 1 selection-review packet generation failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
