"""Generate the Phase 1 candidate-corpus structure report.

This script examines the broad candidate corpus before any Phase 1
concept selection or embedding work. It does not use embeddings,
concept anchors, LLM classification, or concept-weight proposals.

Outputs:

- artifacts/phase1/scope/source_structure_report.csv
- artifacts/phase1/scope/source_structure_report.html

The report supports the human-authored source-section scope rules for
consciousness, self/identity, and reality/appearance.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import yaml

LOGGER = logging.getLogger("wth.phase1.report_candidate_corpus_structure")

DEFAULT_PROJECT_ROOT: Final = Path()
DEFAULT_CHUNKS_ROOT: Final = Path("artifacts/phase1/chunks")
DEFAULT_CANDIDATE_MANIFEST: Final = Path(
    "artifacts/phase1/candidate/candidate_corpus_manifest.json"
)
DEFAULT_ACQUISITION_MANIFEST: Final = Path("artifacts/phase1/acquisition_manifest.json")
DEFAULT_OUTPUT_CSV: Final = Path("artifacts/phase1/scope/source_structure_report.csv")
DEFAULT_OUTPUT_HTML: Final = Path("artifacts/phase1/scope/source_structure_report.html")

DEFAULT_EXPECTED_CHUNK_COUNT: Final = 7_469
DEFAULT_CORPUS_VERSION: Final = "phase1_candidate_corpus_v1"
DEFAULT_SAMPLE_LENGTH: Final = 400
DEFAULT_OCR_HEAVY_THRESHOLD: Final = 0.03
DEFAULT_LARGE_SECTION_CHUNKS: Final = 15
DEFAULT_LARGE_SECTION_TOKENS: Final = 3_000
DEFAULT_CATALOGUE: Final = Path("docs/catalogues/phase1_sources.yaml")

GENERIC_HEADINGS: Final = frozenset(
    {
        "",
        "body",
        "chapter",
        "content",
        "document",
        "main text",
        "none",
        "n/a",
        "section",
        "source",
        "text",
        "unknown",
        "untitled",
    }
)

FRONT_MATTER_PATTERNS: Final = (
    re.compile(r"\btitle\s+page\b", re.IGNORECASE),
    re.compile(r"\bcopyright\b", re.IGNORECASE),
    re.compile(r"\bpreface\b", re.IGNORECASE),
    re.compile(r"\bforeword\b", re.IGNORECASE),
    re.compile(r"\bdedication\b", re.IGNORECASE),
    re.compile(r"\backnowledg(e)?ments?\b", re.IGNORECASE),
    re.compile(r"\btable\s+of\s+contents\b", re.IGNORECASE),
    re.compile(r"^contents$", re.IGNORECASE),
    re.compile(r"\bpublisher\b", re.IGNORECASE),
)

INDEX_PATTERNS: Final = (
    re.compile(r"^index$", re.IGNORECASE),
    re.compile(r"\bsubject\s+index\b", re.IGNORECASE),
    re.compile(r"\bauthor\s+index\b", re.IGNORECASE),
    re.compile(r"\bglossary\b", re.IGNORECASE),
    re.compile(r"\bconcordance\b", re.IGNORECASE),
)

REFERENCE_PATTERNS: Final = (
    re.compile(r"^references?$", re.IGNORECASE),
    re.compile(r"\bbibliograph(y|ies)\b", re.IGNORECASE),
    re.compile(r"\bworks?\s+cited\b", re.IGNORECASE),
    re.compile(r"\bliterature\s+cited\b", re.IGNORECASE),
)

APPENDIX_PATTERNS: Final = (
    re.compile(r"\bappendix\b", re.IGNORECASE),
    re.compile(r"\bsupplement(ary)?\b", re.IGNORECASE),
    re.compile(r"\bannex\b", re.IGNORECASE),
)

VERSE_UNIT_TERMS: Final = frozenset(
    {
        "aphorism",
        "karika",
        "mantra",
        "sloka",
        "stanza",
        "sutra",
        "verse",
    }
)

WARNING_KEYS: Final = frozenset(
    {
        "warning",
        "warnings",
        "parser_warning",
        "parser_warnings",
        "chunking_warning",
        "chunking_warnings",
        "ocr_warning",
        "ocr_warnings",
    }
)

REPORT_FIELDS: Final = (
    "source_id",
    "domain",
    "source_title",
    "section_id",
    "section_title",
    "parent_section",
    "unit_type",
    "structural_locator",
    "chunk_count",
    "token_count",
    "sample_text",
    "parser_warning_count",
    "ocr_noise_score",
    "unusually_high_chunk_count",
    "missing_heading",
    "generic_heading",
    "ocr_heavy",
    "front_matter",
    "index_section",
    "references_section",
    "appendix_section",
    "repeated_header_or_footer",
    "empty_heading",
    "large_unstructured_section",
    "verse_range_rule_candidate",
    "proposed_structure_action",
    "manual_scope_decision",
    "manual_relevant_concepts",
    "manual_structure_action",
    "manual_review_notes",
)


class StructureReportError(RuntimeError):
    """Raised when the structure report cannot be generated safely."""


@dataclass(frozen=True, slots=True)
class CandidateChunk:
    """Normalized structural metadata for one candidate chunk."""

    chunk_id: str
    source_id: str
    domain: str
    source_title: str
    section_id: str
    section_title: str
    parent_section: str
    unit_type: str
    structural_locator: str
    token_count: int
    text: str
    parser_warning_count: int
    ocr_noise_score: float
    first_line: str
    last_line: str


@dataclass(slots=True)
class SectionAccumulator:
    """Mutable aggregation state for one structural section."""

    source_id: str
    domain: str
    source_title: str
    section_id: str
    section_title: str
    parent_section: str
    unit_type: str
    structural_locator: str

    chunk_count: int = 0
    token_count: int = 0
    parser_warning_count: int = 0
    ocr_noise_total: float = 0.0
    sample_text: str = ""
    repeated_header_footer_count: int = 0


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=("Inspect the Phase 1 candidate corpus and generate source-structure reports.")
    )

    parser.add_argument(
        "--catalogue",
        type=Path,
        default=DEFAULT_CATALOGUE,
        help="Path to the phase1 sources YAML catalogue",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
    )
    parser.add_argument(
        "--chunks-root",
        type=Path,
        default=DEFAULT_CHUNKS_ROOT,
    )
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=DEFAULT_CANDIDATE_MANIFEST,
    )
    parser.add_argument(
        "--acquisition-manifest",
        type=Path,
        default=DEFAULT_ACQUISITION_MANIFEST,
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=DEFAULT_OUTPUT_HTML,
    )
    parser.add_argument(
        "--corpus-version",
        default=DEFAULT_CORPUS_VERSION,
    )
    parser.add_argument(
        "--expected-chunk-count",
        type=int,
        default=DEFAULT_EXPECTED_CHUNK_COUNT,
    )
    parser.add_argument(
        "--ocr-heavy-threshold",
        type=float,
        default=DEFAULT_OCR_HEAVY_THRESHOLD,
    )
    parser.add_argument(
        "--large-section-chunks",
        type=int,
        default=DEFAULT_LARGE_SECTION_CHUNKS,
    )
    parser.add_argument(
        "--large-section-tokens",
        type=int,
        default=DEFAULT_LARGE_SECTION_TOKENS,
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing CSV and HTML reports.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )

    return parser.parse_args()


def main() -> None:
    """Generate the candidate-corpus structure reports."""

    args = parse_arguments()
    configure_logging(args.log_level)

    try:
        rows = generate_structure_report(
            project_root=args.project_root,
            chunks_root=args.chunks_root,
            candidate_manifest_path=args.candidate_manifest,
            acquisition_manifest_path=args.acquisition_manifest,
            catalogue_path=args.catalogue,
            output_csv=args.output_csv,
            output_html=args.output_html,
            corpus_version=args.corpus_version,
            expected_chunk_count=args.expected_chunk_count,
            ocr_heavy_threshold=args.ocr_heavy_threshold,
            large_section_chunks=args.large_section_chunks,
            large_section_tokens=args.large_section_tokens,
            replace=args.replace,
        )
    except Exception:
        LOGGER.exception("Candidate-corpus structure report failed")
        raise SystemExit(1) from None

    LOGGER.info("Structure report generated successfully")
    LOGGER.info(
        "Section rows: %d",
        len(rows),
    )


def generate_structure_report(
    *,
    project_root: Path,
    chunks_root: Path,
    candidate_manifest_path: Path,
    acquisition_manifest_path: Path,
    catalogue_path: Path,
    output_csv: Path,
    output_html: Path,
    corpus_version: str,
    expected_chunk_count: int,
    ocr_heavy_threshold: float,
    large_section_chunks: int,
    large_section_tokens: int,
    replace: bool,
) -> list[dict[str, object]]:
    """Generate CSV and HTML source-structure reports."""

    validate_numeric_arguments(
        expected_chunk_count=expected_chunk_count,
        ocr_heavy_threshold=ocr_heavy_threshold,
        large_section_chunks=large_section_chunks,
        large_section_tokens=large_section_tokens,
    )

    project_root = project_root.resolve()

    chunks_root = resolve_from_project(
        project_root,
        chunks_root,
    )
    candidate_manifest_path = resolve_from_project(
        project_root,
        candidate_manifest_path,
    )
    acquisition_manifest_path = resolve_from_project(
        project_root,
        acquisition_manifest_path,
    )
    catalogue_path = resolve_from_project(
        project_root,
        catalogue_path,
    )
    output_csv = resolve_from_project(
        project_root,
        output_csv,
    )
    output_html = resolve_from_project(
        project_root,
        output_html,
    )

    require_directory(chunks_root)
    require_file(candidate_manifest_path)
    require_file(acquisition_manifest_path)
    require_file(catalogue_path)

    validate_output_paths(
        output_csv=output_csv,
        output_html=output_html,
        replace=replace,
    )

    candidate_manifest = load_json_object(candidate_manifest_path)

    validate_candidate_manifest(
        candidate_manifest,
        corpus_version=corpus_version,
        expected_chunk_count=expected_chunk_count,
    )

    source_titles = load_source_titles_from_catalogue(catalogue_path)

    chunks = load_candidate_chunks(
        chunks_root=chunks_root,
        source_titles=source_titles,
    )

    if len(chunks) != expected_chunk_count:
        raise StructureReportError(
            "Loaded candidate chunk count does not match "
            f"expectation: {len(chunks)} != "
            f"{expected_chunk_count}"
        )

    repeated_lines = identify_repeated_boundary_lines(chunks)

    section_accumulators = aggregate_sections(
        chunks=chunks,
        repeated_lines=repeated_lines,
    )

    high_count_thresholds = calculate_high_count_thresholds(section_accumulators.values())

    rows = build_report_rows(
        accumulators=section_accumulators.values(),
        high_count_thresholds=high_count_thresholds,
        ocr_heavy_threshold=ocr_heavy_threshold,
        large_section_chunks=large_section_chunks,
        large_section_tokens=large_section_tokens,
    )

    rows.sort(
        key=lambda row: (
            str(row["domain"]),
            str(row["source_id"]),
            str(row["structural_locator"]),
            str(row["section_id"]),
        )
    )

    write_csv_report(
        output_csv,
        rows,
    )

    write_html_report(
        output_html=output_html,
        rows=rows,
        corpus_version=corpus_version,
        chunk_count=len(chunks),
        source_count=len({chunk.source_id for chunk in chunks}),
        generated_at=datetime.now(UTC).isoformat(),
    )

    return rows


def load_candidate_chunks(
    *,
    chunks_root: Path,
    source_titles: dict[str, str],
) -> list[CandidateChunk]:
    """Load and normalize all candidate chunks."""

    artifact_paths = tuple(sorted(chunks_root.glob("*.json")))

    if not artifact_paths:
        raise StructureReportError(f"No chunk JSON files found in {chunks_root}")

    chunks: list[CandidateChunk] = []
    seen_chunk_ids: set[str] = set()

    for artifact_path in artifact_paths:
        raw = load_json_value(artifact_path)

        if not isinstance(raw, list):
            raise StructureReportError(f"Chunk artifact must contain a JSON list: {artifact_path}")

        for record_number, raw_record in enumerate(
            raw,
            start=1,
        ):
            record = normalize_object_mapping(
                raw_record,
                (f"chunk record at {artifact_path}:{record_number}"),
            )

            chunk = normalize_chunk(
                record=record,
                source_titles=source_titles,
            )

            if chunk.chunk_id in seen_chunk_ids:
                raise StructureReportError(f"Duplicate chunk ID: {chunk.chunk_id}")

            seen_chunk_ids.add(chunk.chunk_id)
            chunks.append(chunk)

    return chunks


def normalize_chunk(
    *,
    record: dict[str, object],
    source_titles: dict[str, str],
) -> CandidateChunk:
    """Normalize one chunk record."""

    chunk_id = require_non_empty_string(
        get_first_path_value(
            record,
            (
                ("chunk_id",),
                ("id",),
            ),
        ),
        "chunk_id",
    )

    source_id = require_non_empty_string(
        get_first_path_value(
            record,
            (
                ("source_id",),
                ("metadata", "source_id"),
                ("provenance", "source_id"),
            ),
        ),
        f"source_id for {chunk_id}",
    )

    domain = normalize_domain(
        get_first_path_value(
            record,
            (
                ("domain",),
                ("metadata", "domain"),
                ("provenance", "domain"),
            ),
        ),
        chunk_id=chunk_id,
    )

    source_title = first_non_empty_string(
        get_first_path_value(
            record,
            (
                ("source_title",),
                ("metadata", "source_title"),
                ("provenance", "source_title"),
            ),
        ),
        source_titles.get(
            source_id,
            source_id,
        ),
    )

    section_id = first_non_empty_string(
        get_first_path_value(
            record,
            (
                ("section_id",),
                ("metadata", "section_id"),
                ("structure", "section_id"),
                ("provenance", "section_id"),
            ),
        ),
        "",
    )

    section_title = first_non_empty_string(
        get_first_path_value(
            record,
            (
                ("section_title",),
                ("heading",),
                ("heading_text",),
                ("metadata", "section_title"),
                ("metadata", "heading"),
                ("structure", "section_title"),
                ("structure", "heading"),
                ("provenance", "section_title"),
            ),
        ),
        "",
    )

    parent_section = first_non_empty_string(
        get_first_path_value(
            record,
            (
                ("parent_section",),
                ("parent_section_title",),
                ("metadata", "parent_section"),
                ("structure", "parent_section"),
                ("provenance", "parent_section"),
            ),
        ),
        "",
    )

    unit_type = first_non_empty_string(
        get_first_path_value(
            record,
            (
                ("unit_type",),
                ("chunk_type",),
                ("metadata", "unit_type"),
                ("structure", "unit_type"),
                ("provenance", "unit_type"),
            ),
        ),
        "unknown",
    )

    structural_locator = first_non_empty_string(
        get_first_path_value(
            record,
            (
                ("structural_locator",),
                ("citation_locator",),
                ("locator",),
                ("metadata", "structural_locator"),
                ("metadata", "citation_locator"),
                ("structure", "locator"),
                ("provenance", "structural_locator"),
            ),
        ),
        "",
    )

    text = first_non_empty_string(
        get_first_path_value(
            record,
            (
                ("text",),
                ("chunk_text",),
                ("content",),
                ("metadata", "text"),
            ),
        ),
        "",
    )

    if not text:
        raise StructureReportError(f"Chunk {chunk_id} has no text")

    token_count = normalize_token_count(
        get_first_path_value(
            record,
            (
                ("token_count",),
                ("metadata", "token_count"),
            ),
        ),
        text=text,
    )

    parser_warning_count = count_warning_values(record)

    first_line, last_line = extract_boundary_lines(text)

    return CandidateChunk(
        chunk_id=chunk_id,
        source_id=source_id,
        domain=domain,
        source_title=source_title,
        section_id=section_id,
        section_title=section_title,
        parent_section=parent_section,
        unit_type=unit_type,
        structural_locator=structural_locator,
        token_count=token_count,
        text=text,
        parser_warning_count=parser_warning_count,
        ocr_noise_score=calculate_ocr_noise_score(text),
        first_line=first_line,
        last_line=last_line,
    )


def aggregate_sections(
    *,
    chunks: Sequence[CandidateChunk],
    repeated_lines: dict[str, set[str]],
) -> dict[tuple[str, str], SectionAccumulator]:
    """Aggregate chunks into structural sections."""

    accumulators: dict[
        tuple[str, str],
        SectionAccumulator,
    ] = {}

    for chunk in chunks:
        section_key = build_section_key(chunk)
        key = (
            chunk.source_id,
            section_key,
        )

        accumulator = accumulators.get(key)

        if accumulator is None:
            accumulator = SectionAccumulator(
                source_id=chunk.source_id,
                domain=chunk.domain,
                source_title=chunk.source_title,
                section_id=chunk.section_id,
                section_title=chunk.section_title,
                parent_section=chunk.parent_section,
                unit_type=chunk.unit_type,
                structural_locator=chunk.structural_locator,
            )
            accumulators[key] = accumulator

        accumulator.chunk_count += 1
        accumulator.token_count += chunk.token_count
        accumulator.parser_warning_count += chunk.parser_warning_count
        accumulator.ocr_noise_total += chunk.ocr_noise_score

        if not accumulator.sample_text:
            accumulator.sample_text = normalize_sample_text(chunk.text)

        source_repeated_lines = repeated_lines.get(
            chunk.source_id,
            set(),
        )

        if chunk.first_line in source_repeated_lines or chunk.last_line in source_repeated_lines:
            accumulator.repeated_header_footer_count += 1

    return accumulators


def build_section_key(
    chunk: CandidateChunk,
) -> str:
    """Build a stable section-grouping key."""

    if chunk.section_id:
        return f"id:{chunk.section_id}"

    if chunk.section_title:
        return "title:" + normalize_heading(chunk.section_title)

    if chunk.structural_locator:
        return "locator:" + normalize_heading(chunk.structural_locator)

    return "unstructured:missing"


def identify_repeated_boundary_lines(
    chunks: Sequence[CandidateChunk],
) -> dict[str, set[str]]:
    """Identify repeated first or last lines by source."""

    counts_by_source: dict[
        str,
        Counter[str],
    ] = defaultdict(Counter)

    chunk_counts_by_source: Counter[str] = Counter()

    for chunk in chunks:
        chunk_counts_by_source[chunk.source_id] += 1

        for boundary_line in {
            chunk.first_line,
            chunk.last_line,
        }:
            if boundary_line:
                counts_by_source[chunk.source_id][boundary_line] += 1

    repeated: dict[str, set[str]] = {}

    for source_id, line_counts in counts_by_source.items():
        source_chunk_count = chunk_counts_by_source[source_id]
        minimum_count = max(
            3,
            math.ceil(source_chunk_count * 0.05),
        )

        repeated[source_id] = {
            line for line, count in line_counts.items() if count >= minimum_count
        }

    return repeated


def calculate_high_count_thresholds(
    accumulators: Iterable[SectionAccumulator],
) -> dict[str, int]:
    """Calculate a robust high-section-size threshold per source."""

    counts_by_source: dict[
        str,
        list[int],
    ] = defaultdict(list)

    for accumulator in accumulators:
        counts_by_source[accumulator.source_id].append(accumulator.chunk_count)

    thresholds: dict[str, int] = {}

    for source_id, counts in counts_by_source.items():
        if len(counts) < 4:
            thresholds[source_id] = max(
                10,
                max(counts),
            )
            continue

        sorted_counts = sorted(counts)

        q1 = percentile(
            sorted_counts,
            0.25,
        )
        q3 = percentile(
            sorted_counts,
            0.75,
        )
        p90 = percentile(
            sorted_counts,
            0.90,
        )

        iqr = q3 - q1

        threshold = math.ceil(
            max(
                10.0,
                p90,
                q3 + (1.5 * iqr),
            )
        )

        thresholds[source_id] = threshold

    return thresholds


def build_report_rows(
    *,
    accumulators: Iterable[SectionAccumulator],
    high_count_thresholds: dict[str, int],
    ocr_heavy_threshold: float,
    large_section_chunks: int,
    large_section_tokens: int,
) -> list[dict[str, object]]:
    """Build report rows with structural-quality flags."""

    rows: list[dict[str, object]] = []

    for accumulator in accumulators:
        heading = accumulator.section_title.strip()
        normalized_heading = normalize_heading(heading)

        empty_heading = not heading
        missing_heading = (
            empty_heading and not accumulator.section_id and not accumulator.structural_locator
        )

        generic_heading = is_generic_heading(normalized_heading)

        average_ocr_noise = accumulator.ocr_noise_total / accumulator.chunk_count

        ocr_heavy = accumulator.parser_warning_count > 0 or average_ocr_noise >= ocr_heavy_threshold

        heading_and_sample = f"{heading}\n{accumulator.sample_text}"

        front_matter = matches_any(
            heading_and_sample,
            FRONT_MATTER_PATTERNS,
        )
        index_section = matches_any(
            heading_and_sample,
            INDEX_PATTERNS,
        )
        references_section = matches_any(
            heading_and_sample,
            REFERENCE_PATTERNS,
        )
        appendix_section = matches_any(
            heading_and_sample,
            APPENDIX_PATTERNS,
        )

        repeated_header_or_footer = accumulator.repeated_header_footer_count > 0

        unusually_high_chunk_count = accumulator.chunk_count >= high_count_thresholds.get(
            accumulator.source_id,
            10,
        )

        large_unstructured_section = (empty_heading or generic_heading) and (
            accumulator.chunk_count >= large_section_chunks
            or accumulator.token_count >= large_section_tokens
        )

        verse_range_rule_candidate = requires_verse_range_rule(
            unit_type=accumulator.unit_type,
            locator=(accumulator.structural_locator),
            missing_or_generic=(empty_heading or generic_heading),
        )

        proposed_action = determine_proposed_action(
            front_matter=front_matter,
            index_section=index_section,
            references_section=references_section,
            appendix_section=appendix_section,
            ocr_heavy=ocr_heavy,
            repeated_header_or_footer=(repeated_header_or_footer),
            verse_range_rule_candidate=(verse_range_rule_candidate),
            missing_heading=missing_heading,
            generic_heading=generic_heading,
            large_unstructured_section=(large_unstructured_section),
        )

        rows.append(
            {
                "source_id": (accumulator.source_id),
                "domain": accumulator.domain,
                "source_title": (accumulator.source_title),
                "section_id": (accumulator.section_id),
                "section_title": (accumulator.section_title),
                "parent_section": (accumulator.parent_section),
                "unit_type": (accumulator.unit_type),
                "structural_locator": (accumulator.structural_locator),
                "chunk_count": (accumulator.chunk_count),
                "token_count": (accumulator.token_count),
                "sample_text": (accumulator.sample_text),
                "parser_warning_count": (accumulator.parser_warning_count),
                "ocr_noise_score": round(
                    average_ocr_noise,
                    6,
                ),
                "unusually_high_chunk_count": (unusually_high_chunk_count),
                "missing_heading": (missing_heading),
                "generic_heading": (generic_heading),
                "ocr_heavy": ocr_heavy,
                "front_matter": front_matter,
                "index_section": index_section,
                "references_section": (references_section),
                "appendix_section": (appendix_section),
                "repeated_header_or_footer": (repeated_header_or_footer),
                "empty_heading": empty_heading,
                "large_unstructured_section": (large_unstructured_section),
                "verse_range_rule_candidate": (verse_range_rule_candidate),
                "proposed_structure_action": (proposed_action),
                "manual_scope_decision": "",
                "manual_relevant_concepts": "",
                "manual_structure_action": "",
                "manual_review_notes": "",
            }
        )

    return rows


def determine_proposed_action(
    *,
    front_matter: bool,
    index_section: bool,
    references_section: bool,
    appendix_section: bool,
    ocr_heavy: bool,
    repeated_header_or_footer: bool,
    verse_range_rule_candidate: bool,
    missing_heading: bool,
    generic_heading: bool,
    large_unstructured_section: bool,
) -> str:
    """Propose a structural action for human review."""

    if front_matter or index_section or references_section or appendix_section:
        return "exclude_candidate"

    if ocr_heavy or repeated_header_or_footer:
        return "source_specific_preprocessing"

    if verse_range_rule_candidate:
        return "manual_range_annotation"

    if large_unstructured_section or missing_heading or generic_heading:
        return "manual_section_review"

    return "structure_usable"


def write_csv_report(
    path: Path,
    rows: Sequence[dict[str, object]],
) -> None:
    """Write the section-level CSV report."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(f".{path.name}.tmp")

    with temporary.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(REPORT_FIELDS),
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    temporary.replace(path)


def write_html_report(
    *,
    output_html: Path,
    rows: Sequence[dict[str, object]],
    corpus_version: str,
    chunk_count: int,
    source_count: int,
    generated_at: str,
) -> None:
    """Write the human-readable HTML report."""

    output_html.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_summary = summarize_by_source(rows)

    action_summary = Counter(str(row["proposed_structure_action"]) for row in rows)

    summary_sections = (
        (
            "Chunks by source",
            source_summary,
        ),
        (
            "Headings with unusually high chunk counts",
            filter_rows(
                rows,
                "unusually_high_chunk_count",
            ),
        ),
        (
            "Missing or generic headings",
            [row for row in rows if bool(row["missing_heading"]) or bool(row["generic_heading"])],
        ),
        (
            "OCR-heavy sections",
            filter_rows(
                rows,
                "ocr_heavy",
            ),
        ),
        (
            "Front matter",
            filter_rows(
                rows,
                "front_matter",
            ),
        ),
        (
            "Indexes",
            filter_rows(
                rows,
                "index_section",
            ),
        ),
        (
            "References and bibliographies",
            filter_rows(
                rows,
                "references_section",
            ),
        ),
        (
            "Appendices and supplements",
            filter_rows(
                rows,
                "appendix_section",
            ),
        ),
        (
            "Repeated headers or footers",
            filter_rows(
                rows,
                "repeated_header_or_footer",
            ),
        ),
        (
            "Empty headings",
            filter_rows(
                rows,
                "empty_heading",
            ),
        ),
        (
            "Large unstructured sections",
            filter_rows(
                rows,
                "large_unstructured_section",
            ),
        ),
        (
            "Verse-range rule candidates",
            filter_rows(
                rows,
                "verse_range_rule_candidate",
            ),
        ),
    )

    summary_html = "\n".join(
        render_summary_section(
            title,
            section_rows,
        )
        for title, section_rows in summary_sections
    )

    action_rows = [
        {
            "proposed_structure_action": action,
            "section_count": count,
        }
        for action, count in sorted(action_summary.items())
    ]

    full_table_fields = (
        "source_id",
        "domain",
        "section_id",
        "section_title",
        "parent_section",
        "unit_type",
        "structural_locator",
        "chunk_count",
        "token_count",
        "parser_warning_count",
        "ocr_noise_score",
        "proposed_structure_action",
        "sample_text",
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Phase 1 Candidate Corpus Structure Report</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 2rem;
    color: #1f2937;
}}
h1, h2, h3 {{
    color: #111827;
}}
.summary {{
    display: grid;
    grid-template-columns: repeat(4, minmax(150px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
}}
.card {{
    border: 1px solid #d1d5db;
    border-radius: 6px;
    padding: 1rem;
    background: #f9fafb;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 2rem;
    font-size: 0.88rem;
}}
th, td {{
    border: 1px solid #d1d5db;
    padding: 0.45rem;
    text-align: left;
    vertical-align: top;
}}
th {{
    background: #f3f4f6;
    position: sticky;
    top: 0;
}}
tr:nth-child(even) {{
    background: #fafafa;
}}
code {{
    background: #f3f4f6;
    padding: 0.1rem 0.25rem;
}}
.note {{
    border-left: 4px solid #6b7280;
    padding: 0.75rem 1rem;
    background: #f9fafb;
    margin-bottom: 1rem;
}}
.small {{
    font-size: 0.8rem;
    color: #4b5563;
}}
</style>
</head>
<body>

<h1>Phase 1 Candidate Corpus Structure Report</h1>

<div class="summary">
  <div class="card">
    <strong>Corpus version</strong><br>
    {escape(corpus_version)}
  </div>
  <div class="card">
    <strong>Sources</strong><br>
    {source_count:,}
  </div>
  <div class="card">
    <strong>Chunks</strong><br>
    {chunk_count:,}
  </div>
  <div class="card">
    <strong>Structural sections</strong><br>
    {len(rows):,}
  </div>
</div>

<p class="small">
Generated at {escape(generated_at)}.
This report uses structural and lexical-quality signals only.
It does not use embeddings, concept anchors, or LLM classification.
</p>

<div class="note">
<strong>Manual-review objective:</strong>
For every source, identify relevant, irrelevant, and ambiguous
sections; source-specific OCR problems; and sections requiring
verse-range rules rather than heading rules.
</div>

<h2>Proposed structural actions</h2>
{
        render_html_table(
            action_rows,
            (
                "proposed_structure_action",
                "section_count",
            ),
        )
    }

{summary_html}

<h2>Complete section-level report</h2>
{
        render_html_table(
            rows,
            full_table_fields,
        )
    }

</body>
</html>
"""

    temporary = output_html.with_name(f".{output_html.name}.tmp")

    temporary.write_text(
        document,
        encoding="utf-8",
        newline="\n",
    )

    temporary.replace(output_html)


def summarize_by_source(
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Summarize chunks and sections by source."""

    summaries: dict[
        str,
        dict[str, object],
    ] = {}

    for row in rows:
        source_id = str(row["source_id"])

        summary = summaries.setdefault(
            source_id,
            {
                "source_id": source_id,
                "domain": row["domain"],
                "source_title": (row["source_title"]),
                "section_count": 0,
                "chunk_count": 0,
                "token_count": 0,
                "flagged_section_count": 0,
            },
        )

        summary["section_count"] = (
            object_to_int(summary["section_count"], description="section_count") + 1
        )
        summary["chunk_count"] = object_to_int(
            summary["chunk_count"], description="chunk_count"
        ) + object_to_int(row["chunk_count"], description="row chunk_count")
        summary["token_count"] = object_to_int(
            summary["token_count"], description="token_count"
        ) + object_to_int(row["token_count"], description="row token_count")

        if str(row["proposed_structure_action"]) != "structure_usable":
            summary["flagged_section_count"] = (
                object_to_int(summary["flagged_section_count"], description="flagged_section_count")
                + 1
            )

    return [summaries[source_id] for source_id in sorted(summaries)]


def render_summary_section(
    title: str,
    rows: Sequence[dict[str, object]],
) -> str:
    """Render one summary section."""

    if not rows:
        return f"<h2>{escape(title)}</h2><p>No matching sections.</p>"

    preferred_fields = (
        "source_id",
        "domain",
        "source_title",
        "section_id",
        "section_title",
        "structural_locator",
        "chunk_count",
        "token_count",
        "parser_warning_count",
        "ocr_noise_score",
        "proposed_structure_action",
    )

    available_fields = tuple(
        field for field in preferred_fields if any(field in row for row in rows)
    )

    return f"<h2>{escape(title)}</h2>" + render_html_table(
        rows,
        available_fields,
    )


def render_html_table(
    rows: Sequence[dict[str, object]],
    fields: Sequence[str],
) -> str:
    """Render a basic escaped HTML table."""

    if not rows:
        return "<p>No records.</p>"

    header = "".join(f"<th>{escape(field)}</th>" for field in fields)

    body_rows: list[str] = []

    for row in rows:
        cells = "".join(
            ("<td>" + escape(format_cell(row.get(field))) + "</td>") for field in fields
        )

        body_rows.append(f"<tr>{cells}</tr>")

    return (
        f"<table><thead><tr>{header}</tr></thead><tbody>" + "".join(body_rows) + "</tbody></table>"
    )


def filter_rows(
    rows: Sequence[dict[str, object]],
    field: str,
) -> list[dict[str, object]]:
    """Return rows where a boolean field is true."""

    return [row for row in rows if bool(row.get(field))]


def calculate_ocr_noise_score(
    text: str,
) -> float:
    """Estimate OCR noise using transparent text-quality indicators."""

    if not text:
        return 1.0

    character_count = len(text)
    score = 0.0

    replacement_characters = text.count("\ufffd")
    score += (replacement_characters / character_count) * 8.0

    unusual_sequences = re.findall(
        r"[^A-Za-z0-9\s.,;:'\"!?()\-\u0900-\u097F]{3,}",
        text,
    )
    score += min(
        len(unusual_sequences) * 0.003,
        0.03,
    )

    split_hyphenations = len(
        re.findall(
            r"[A-Za-z]-\s*\n\s*[A-Za-z]",
            text,
        )
    )
    score += min(
        split_hyphenations * 0.002,
        0.02,
    )

    repeated_punctuation = len(
        re.findall(
            r"([^\w\s])\1{3,}",
            text,
        )
    )
    score += min(
        repeated_punctuation * 0.004,
        0.02,
    )

    printable_characters = sum(
        1 for character in text if character.isprintable() or character in "\n\t"
    )

    printable_ratio = printable_characters / character_count

    if printable_ratio < 0.98:
        score += 0.98 - printable_ratio

    alphabetic_characters = sum(character.isalpha() for character in text)

    non_space_characters = sum(not character.isspace() for character in text)

    if non_space_characters:
        alphabetic_ratio = alphabetic_characters / non_space_characters

        if alphabetic_ratio < 0.45:
            score += (0.45 - alphabetic_ratio) * 0.10

    return min(
        max(score, 0.0),
        1.0,
    )


def count_warning_values(
    value: object,
) -> int:
    """Count structured parser and OCR warning values recursively."""

    if isinstance(value, dict):
        count = 0

        for raw_key, raw_nested in value.items():
            key = str(raw_key).casefold()
            nested: object = raw_nested

            if key in WARNING_KEYS:
                count += warning_value_size(nested)
            else:
                count += count_warning_values(nested)

        return count

    if isinstance(value, list | tuple):
        return sum(count_warning_values(item) for item in value)

    return 0


def warning_value_size(
    value: object,
) -> int:
    """Return the number of warnings represented by a value."""

    if value is None:
        return 0

    if isinstance(value, str):
        return 1 if value.strip() else 0

    if isinstance(value, list | tuple):
        return sum(
            max(
                warning_value_size(item),
                1,
            )
            for item in value
        )

    if isinstance(value, dict):
        return 1 if value else 0

    if isinstance(value, bool):
        return 1 if value else 0

    return 1


def requires_verse_range_rule(
    *,
    unit_type: str,
    locator: str,
    missing_or_generic: bool,
) -> bool:
    """Detect sections better scoped by verse or numbered ranges."""

    normalized_unit_type = unit_type.casefold()

    if any(term in normalized_unit_type for term in VERSE_UNIT_TERMS):
        return missing_or_generic

    locator_has_range = bool(
        re.search(
            r"\b\d+\s*[--:]\s*\d+\b",
            locator,
        )
    )

    return missing_or_generic and locator_has_range


def is_generic_heading(
    normalized_heading: str,
) -> bool:
    """Return whether a heading is too generic for transparent scope."""

    if normalized_heading in GENERIC_HEADINGS:
        return True

    return bool(
        re.fullmatch(
            r"(chapter|section|part|book|text)"
            r"\s*[\divxlcdm]*",
            normalized_heading,
            flags=re.IGNORECASE,
        )
    )


def matches_any(
    text: str,
    patterns: Sequence[re.Pattern[str]],
) -> bool:
    """Return whether text matches any supplied pattern."""

    return any(pattern.search(text) is not None for pattern in patterns)


def extract_boundary_lines(
    text: str,
) -> tuple[str, str]:
    """Extract normalized first and last meaningful lines."""

    normalized_lines = [normalize_boundary_line(line) for line in text.splitlines()]

    meaningful_lines = [line for line in normalized_lines if line]

    if not meaningful_lines:
        return "", ""

    return (
        meaningful_lines[0],
        meaningful_lines[-1],
    )


def normalize_boundary_line(
    line: str,
) -> str:
    """Normalize a possible repeated header or footer line."""

    normalized = " ".join(line.split()).strip()

    if not (8 <= len(normalized) <= 140):
        return ""

    if re.fullmatch(
        r"\d+",
        normalized,
    ):
        return ""

    return normalized.casefold()


def normalize_sample_text(
    text: str,
) -> str:
    """Normalize and truncate sample text."""

    normalized = " ".join(text.split())

    if len(normalized) <= DEFAULT_SAMPLE_LENGTH:
        return normalized

    return normalized[: DEFAULT_SAMPLE_LENGTH - 1].rstrip() + "…"


def normalize_heading(
    value: str,
) -> str:
    """Normalize a heading for comparison."""

    return " ".join(value.casefold().split())


def percentile(
    sorted_values: Sequence[int],
    fraction: float,
) -> float:
    """Calculate a linearly interpolated percentile."""

    if not sorted_values:
        return 0.0

    if len(sorted_values) == 1:
        return float(sorted_values[0])

    position = (len(sorted_values) - 1) * fraction

    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return float(sorted_values[lower_index])

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]

    weight = position - lower_index

    return lower_value + (upper_value - lower_value) * weight


def load_source_titles_from_catalogue(
    path: Path,
) -> dict[str, str]:
    """Load human-readable source titles from the YAML catalogue."""

    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise StructureReportError(f"Invalid catalogue YAML: {path}") from exc

    if not isinstance(raw, list):
        raise StructureReportError("Source catalogue must contain a YAML list")

    titles: dict[str, str] = {}

    for index, item in enumerate(raw, start=1):
        record = normalize_object_mapping(
            item,
            f"catalogue record {index}",
        )

        source_id = require_non_empty_string(
            record.get("source_id"),
            f"source_id in catalogue record {index}",
        )

        title = first_non_empty_string(
            record.get("title"),
            source_id,
        )

        titles[source_id] = title

    return titles


def load_source_titles(
    path: Path,
) -> dict[str, str]:
    """Load source titles from the acquisition manifest."""

    raw = load_json_value(path)

    records: list[object]

    if isinstance(raw, list):
        records = list(raw)
    elif isinstance(raw, dict):
        possible_sources = raw.get("sources")
        records = list(possible_sources) if isinstance(possible_sources, list) else [raw]
    else:
        raise StructureReportError("Acquisition manifest must be a list or object")

    titles: dict[str, str] = {}

    for raw_record in records:
        if not isinstance(
            raw_record,
            dict,
        ):
            continue

        record = normalize_object_mapping(
            raw_record,
            "acquisition source record",
        )

        source_id = first_non_empty_string(
            get_first_path_value(
                record,
                (
                    ("source_id",),
                    ("id",),
                ),
            ),
            "",
        )

        if not source_id:
            continue

        source_title = first_non_empty_string(
            get_first_path_value(
                record,
                (
                    ("source_title",),
                    ("title",),
                    ("metadata", "title"),
                ),
            ),
            source_id,
        )

        titles[source_id] = source_title

    return titles


def validate_candidate_manifest(
    manifest: dict[str, object],
    *,
    corpus_version: str,
    expected_chunk_count: int,
) -> None:
    """Validate candidate-only corpus metadata."""

    if manifest.get("status") != "candidate_only":
        raise StructureReportError("Candidate manifest status must be candidate_only")

    if manifest.get("corpus_version") != corpus_version:
        raise StructureReportError("Candidate manifest corpus version mismatch")

    inventory = normalize_object_mapping(
        manifest.get("inventory"),
        "candidate manifest inventory",
    )

    if inventory.get("chunk_count") != expected_chunk_count:
        raise StructureReportError("Candidate manifest chunk count mismatch")

    lifecycle = normalize_object_mapping(
        manifest.get("lifecycle"),
        "candidate manifest lifecycle",
    )

    if lifecycle.get("direct_activation_eligible") is not False:
        raise StructureReportError("Candidate corpus must block direct activation")


def get_first_path_value(
    value: dict[str, object],
    paths: Sequence[tuple[str, ...]],
) -> object | None:
    """Return the first value found at one of the supplied paths."""

    for path in paths:
        current: object = value
        found = True

        for component in path:
            if not isinstance(
                current,
                dict,
            ):
                found = False
                break

            normalized_current = normalize_object_mapping(
                current,
                "nested metadata",
            )

            if component not in normalized_current:
                found = False
                break

            current = normalized_current[component]

        if found:
            return current

    return None


def normalize_object_mapping(
    value: object,
    description: str,
) -> dict[str, object]:
    """Normalize a JSON object to dict[str, object]."""

    if not isinstance(value, dict):
        raise StructureReportError(f"{description} must be a JSON object")

    normalized: dict[str, object] = {}

    for raw_key, raw_value in value.items():
        key = str(raw_key)
        nested_value: object = raw_value
        normalized[key] = nested_value

    return normalized


def normalize_domain(
    value: object,
    *,
    chunk_id: str,
) -> str:
    """Normalize a chunk domain."""

    if isinstance(value, str):
        normalized = value.strip()

        if normalized:
            return normalized

    if isinstance(value, dict):
        mapping = normalize_object_mapping(
            value,
            f"domain for {chunk_id}",
        )

        nested = mapping.get("value")

        if (
            isinstance(
                nested,
                str,
            )
            and nested.strip()
        ):
            return nested.strip()

    raise StructureReportError(f"Chunk {chunk_id} has no valid domain")


def object_to_int(
    value: object,
    *,
    description: str,
) -> int:
    """Convert a verified numeric report value to an integer."""

    if isinstance(value, bool):
        raise StructureReportError(f"{description} must not be a boolean")

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if value.is_integer():
            return int(value)

        raise StructureReportError(f"{description} must be a whole number")

    if isinstance(value, str):
        normalized = value.strip()

        try:
            return int(normalized)
        except ValueError as exc:
            raise StructureReportError(f"{description} must be an integer") from exc

    raise StructureReportError(f"{description} must be an integer")


def normalize_token_count(
    value: object,
    *,
    text: str,
) -> int:
    """Normalize token count with a conservative fallback."""

    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value

    return len(text.split())


def first_non_empty_string(
    value: object,
    fallback: str,
) -> str:
    """Return a normalized non-empty string or a fallback."""

    if isinstance(value, str):
        normalized = value.strip()

        if normalized:
            return normalized

    return fallback


def require_non_empty_string(
    value: object,
    description: str,
) -> str:
    """Require a non-empty string."""

    result = first_non_empty_string(
        value,
        "",
    )

    if not result:
        raise StructureReportError(f"{description} must be a non-empty string")

    return result


def load_json_object(
    path: Path,
) -> dict[str, object]:
    """Load a JSON object."""

    raw = load_json_value(path)

    return normalize_object_mapping(
        raw,
        str(path),
    )


def load_json_value(
    path: Path,
) -> object:
    """Load any JSON value."""

    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StructureReportError(f"Invalid JSON file: {path}") from exc

    return raw


def validate_numeric_arguments(
    *,
    expected_chunk_count: int,
    ocr_heavy_threshold: float,
    large_section_chunks: int,
    large_section_tokens: int,
) -> None:
    """Validate numeric configuration."""

    if expected_chunk_count < 1:
        raise StructureReportError("expected_chunk_count must be at least 1")

    if not 0.0 <= ocr_heavy_threshold <= 1.0:
        raise StructureReportError("ocr_heavy_threshold must be between 0 and 1")

    if large_section_chunks < 1:
        raise StructureReportError("large_section_chunks must be at least 1")

    if large_section_tokens < 1:
        raise StructureReportError("large_section_tokens must be at least 1")


def validate_output_paths(
    *,
    output_csv: Path,
    output_html: Path,
    replace: bool,
) -> None:
    """Protect existing reports."""

    existing = [
        path
        for path in (
            output_csv,
            output_html,
        )
        if path.exists()
    ]

    if existing and not replace:
        raise StructureReportError(
            "Structure report output already exists. "
            "Use --replace only for intentional regeneration: "
            + ", ".join(str(path) for path in existing)
        )


def resolve_from_project(
    project_root: Path,
    path: Path,
) -> Path:
    """Resolve a path relative to the project root."""

    if path.is_absolute():
        return path.resolve()

    return (project_root / path).resolve()


def require_file(path: Path) -> None:
    """Require an existing file."""

    if not path.is_file():
        raise StructureReportError(f"Required file does not exist: {path}")


def require_directory(path: Path) -> None:
    """Require an existing directory."""

    if not path.is_dir():
        raise StructureReportError(f"Required directory does not exist: {path}")


def format_cell(value: object) -> str:
    """Format a report cell."""

    if value is None:
        return ""

    if isinstance(value, bool):
        return "Yes" if value else "No"

    if isinstance(value, float):
        return f"{value:.6f}"

    return str(value)


def escape(value: object) -> str:
    """HTML-escape a report value."""

    return html.escape(
        format_cell(value),
        quote=True,
    )


def configure_logging(level: str) -> None:
    """Configure console logging."""

    logging.basicConfig(
        level=getattr(
            logging,
            level,
        ),
        format=("%(asctime)s %(levelname)s %(name)s %(message)s"),
    )


if __name__ == "__main__":
    main()
