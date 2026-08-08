"""Generate CSV and HTML review packets for Phase 1 corpus chunks."""

from __future__ import annotations

import argparse
import csv
import html
import json
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import yaml

from apps.api.models.corpus import (
    ChunkConceptProposal,
    ChunkDraft,
    ChunkEmbeddingRecord,
    ParsedDocument,
    SourceCatalogueEntry,
)
from apps.api.models.enums import ConceptSlug

LOGGER = logging.getLogger("wth.phase1.review_packet")

DEFAULT_CATALOGUE_PATH: Final = Path("docs/catalogues/phase1_sources.yaml")
DEFAULT_CHUNKS_ROOT: Final = Path("artifacts/phase1/chunks")
DEFAULT_PARSED_ROOT: Final = Path("artifacts/phase1/parsed")
DEFAULT_EMBEDDINGS_PATH: Final = Path("artifacts/phase1/embeddings/chunk_embeddings.jsonl")
DEFAULT_PROPOSALS_PATH: Final = Path("artifacts/phase1/concepts/chunk_concept_proposals.jsonl")
DEFAULT_CSV_OUTPUT: Final = Path("artifacts/review/phase1_review_packet.csv")
DEFAULT_HTML_OUTPUT: Final = Path("artifacts/review/phase1_review_packet.html")

PHASE1_CONCEPTS: Final = (
    ConceptSlug.CONSCIOUSNESS,
    ConceptSlug.SELF_IDENTITY,
    ConceptSlug.REALITY_APPEARANCE,
)

BASE_FIELD_NAMES: Final = (
    "source_id",
    "domain",
    "source_title",
    "author",
    "translator",
    "editor",
    "publication_year",
    "source_type",
    "inclusion_status",
    "authority_notes",
    "license_name",
    "license_url",
    "rights_statement",
    "rights_jurisdiction",
    "canonical_url",
    "source_checksum",
    "chunk_id",
    "section_id",
    "unit_type",
    "structural_locator",
    "citation",
    "token_count",
    "parser_name",
    "parser_version",
    "chunker_name",
    "chunker_version",
    "text_checksum",
    "chunk_text",
    "parser_warning_count",
    "parser_warnings",
    "embedding_present",
    "embedding_provider",
    "embedding_model",
    "embedding_dimensions",
    "embedding_task_type",
)

REVIEW_FIELD_NAMES: Final = (
    "rights_review_status",
    "rights_reviewed_by",
    "rights_reviewed_at",
    "rights_conditions",
    "review_decision",
    "reviewer",
    "reviewed_at",
    "rejection_reason",
    "review_notes",
)


class ReviewPacketError(RuntimeError):
    """Raised when review-packet inputs are missing or inconsistent."""


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=("Generate Phase 1 corpus-review packets from ingestion artifacts.")
    )

    parser.add_argument(
        "--catalogue",
        type=Path,
        default=DEFAULT_CATALOGUE_PATH,
    )
    parser.add_argument(
        "--chunks-root",
        type=Path,
        default=DEFAULT_CHUNKS_ROOT,
    )
    parser.add_argument(
        "--parsed-root",
        type=Path,
        default=DEFAULT_PARSED_ROOT,
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=DEFAULT_EMBEDDINGS_PATH,
    )
    parser.add_argument(
        "--proposals",
        type=Path,
        default=DEFAULT_PROPOSALS_PATH,
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_CSV_OUTPUT,
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=DEFAULT_HTML_OUTPUT,
    )
    parser.add_argument(
        "--log-level",
        choices=(
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
        ),
        default="INFO",
    )

    return parser.parse_args()


def main() -> None:
    """Generate both review packet formats."""

    args = parse_arguments()
    configure_logging(args.log_level)

    try:
        sources = load_catalogue(args.catalogue)
        chunks = load_chunks(args.chunks_root)
        embeddings = load_embeddings(args.embeddings)
        proposals = load_proposals(args.proposals)
        warnings = load_parser_warnings(args.parsed_root)

        rows = build_review_rows(
            sources=sources,
            chunks=chunks,
            embeddings=embeddings,
            proposals=proposals,
            warnings=warnings,
        )

        field_names = review_field_names()

        write_csv(
            path=args.csv_output,
            field_names=field_names,
            rows=rows,
        )
        write_html(
            path=args.html_output,
            rows=rows,
        )

        LOGGER.info(
            "Generated review packet with %d chunks",
            len(rows),
        )
        LOGGER.info(
            "CSV packet: %s",
            args.csv_output,
        )
        LOGGER.info(
            "HTML packet: %s",
            args.html_output,
        )

    except Exception:
        LOGGER.exception("Review-packet generation failed")
        raise SystemExit(1) from None


def review_field_names() -> tuple[str, ...]:
    """Return stable CSV column ordering."""

    concept_fields: list[str] = []

    for concept in PHASE1_CONCEPTS:
        concept_fields.extend(
            (
                f"{concept.value}_proposal_rank",
                f"{concept.value}_similarity",
                f"{concept.value}_proposed_weight",
                f"{concept.value}_approved_weight",
            )
        )

    return (
        *BASE_FIELD_NAMES,
        *concept_fields,
        *REVIEW_FIELD_NAMES,
    )


def load_catalogue(
    path: Path,
) -> dict[str, SourceCatalogueEntry]:
    """Load and index source-catalogue entries."""

    raw = load_yaml(path)

    if not isinstance(raw, list):
        raise ReviewPacketError("Source catalogue must be a YAML list")

    sources: dict[
        str,
        SourceCatalogueEntry,
    ] = {}

    for index, item in enumerate(
        raw,
        start=1,
    ):
        try:
            source = SourceCatalogueEntry.model_validate(item)
        except Exception as exc:
            raise ReviewPacketError(f"Invalid catalogue entry {index}: {exc}") from exc

        if source.source_id in sources:
            raise ReviewPacketError(f"Duplicate source ID in catalogue: {source.source_id}")

        sources[source.source_id] = source

    return sources


def load_chunks(
    root: Path,
) -> tuple[ChunkDraft, ...]:
    """Load all per-source chunk JSON artifacts."""

    if not root.exists():
        raise ReviewPacketError(f"Chunk directory does not exist: {root}")

    chunks: list[ChunkDraft] = []

    for path in sorted(root.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ReviewPacketError(f"Invalid chunk JSON: {path}") from exc

        if not isinstance(raw, list):
            raise ReviewPacketError(f"Chunk artifact must contain a list: {path}")

        for index, item in enumerate(
            raw,
            start=1,
        ):
            try:
                chunks.append(ChunkDraft.model_validate(item))
            except Exception as exc:
                raise ReviewPacketError(f"Invalid chunk {path}:{index}: {exc}") from exc

    if not chunks:
        raise ReviewPacketError("No chunk artifacts were found")

    chunk_ids = [chunk.chunk_id for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ReviewPacketError("Chunk artifacts contain duplicate chunk IDs")

    return tuple(chunks)


def load_embeddings(
    path: Path,
) -> dict[str, ChunkEmbeddingRecord]:
    """Load and index chunk-embedding JSONL records."""

    if not path.exists():
        raise ReviewPacketError(f"Chunk embedding artifact is missing: {path}")

    records: dict[
        str,
        ChunkEmbeddingRecord,
    ] = {}

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            record = ChunkEmbeddingRecord.model_validate_json(line)
        except Exception as exc:
            raise ReviewPacketError(
                f"Invalid embedding record at {path}:{line_number}: {exc}"
            ) from exc

        if record.chunk_id in records:
            raise ReviewPacketError(f"Duplicate chunk embedding: {record.chunk_id}")

        records[record.chunk_id] = record

    return records


def load_proposals(
    path: Path,
) -> dict[
    str,
    dict[ConceptSlug, ChunkConceptProposal],
]:
    """Load and index concept proposals by chunk and concept."""

    if not path.exists():
        raise ReviewPacketError(f"Concept-proposal artifact is missing: {path}")

    proposals: dict[
        str,
        dict[
            ConceptSlug,
            ChunkConceptProposal,
        ],
    ] = {}

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            proposal = ChunkConceptProposal.model_validate_json(line)
        except Exception as exc:
            raise ReviewPacketError(
                f"Invalid concept proposal at {path}:{line_number}: {exc}"
            ) from exc

        chunk_proposals = proposals.setdefault(
            proposal.chunk_id,
            {},
        )

        if proposal.concept_slug in chunk_proposals:
            raise ReviewPacketError(
                "Duplicate concept proposal for chunk "
                f"{proposal.chunk_id}: "
                f"{proposal.concept_slug.value}"
            )

        chunk_proposals[proposal.concept_slug] = proposal

    return proposals


def load_parser_warnings(
    root: Path,
) -> dict[str, tuple[str, ...]]:
    """Load parser warnings grouped by source ID."""

    if not root.exists():
        return {}

    warnings: dict[
        str,
        tuple[str, ...],
    ] = {}

    for path in sorted(root.glob("*.json")):
        try:
            document = ParsedDocument.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ReviewPacketError(f"Invalid parsed document {path}: {exc}") from exc

        warnings[document.source_id] = tuple(
            (f"{warning.severity.value}:{warning.code}: {warning.message}")
            for warning in document.warnings
        )

    return warnings


def build_review_rows(
    *,
    sources: Mapping[str, SourceCatalogueEntry],
    chunks: Sequence[ChunkDraft],
    embeddings: Mapping[
        str,
        ChunkEmbeddingRecord,
    ],
    proposals: Mapping[
        str,
        Mapping[
            ConceptSlug,
            ChunkConceptProposal,
        ],
    ],
    warnings: Mapping[
        str,
        tuple[str, ...],
    ],
) -> tuple[dict[str, str], ...]:
    """Join all ingestion artifacts into reviewable rows."""

    rows: list[dict[str, str]] = []

    ordered_chunks = sorted(
        chunks,
        key=lambda chunk: (
            chunk.domain.value,
            chunk.source_id,
            chunk.section_id,
            chunk.chunk_id,
        ),
    )

    for chunk in ordered_chunks:
        source = sources.get(chunk.source_id)

        if source is None:
            raise ReviewPacketError(
                f"Chunk references a source missing from the catalogue: {chunk.source_id}"
            )

        embedding = embeddings.get(chunk.chunk_id)
        chunk_proposals = proposals.get(
            chunk.chunk_id,
            {},
        )
        source_warnings = warnings.get(
            chunk.source_id,
            (),
        )

        row = {
            "source_id": source.source_id,
            "domain": source.domain.value,
            "source_title": source.title,
            "author": source.author,
            "translator": source.translator or "",
            "editor": source.editor or "",
            "publication_year": str(source.publication_year),
            "source_type": source.source_type.value,
            "inclusion_status": (source.inclusion_status.value),
            "authority_notes": source.authority_notes,
            "license_name": source.license_name,
            "license_url": str(source.license_url),
            "rights_statement": (source.rights_statement),
            "rights_jurisdiction": (source.rights_jurisdiction),
            "canonical_url": str(source.canonical_url),
            "source_checksum": source.checksum or "",
            "chunk_id": chunk.chunk_id,
            "section_id": chunk.section_id,
            "unit_type": chunk.unit_type.value,
            "structural_locator": (chunk.citation.structural_locator),
            "citation": (chunk.citation.display_text),
            "token_count": str(chunk.token_count),
            "parser_name": chunk.parser_name,
            "parser_version": chunk.parser_version,
            "chunker_name": chunk.chunker_name,
            "chunker_version": chunk.chunker_version,
            "text_checksum": chunk.text_checksum,
            "chunk_text": chunk.text,
            "parser_warning_count": str(len(source_warnings)),
            "parser_warnings": " | ".join(source_warnings),
            "embedding_present": ("true" if embedding is not None else "false"),
            "embedding_provider": (embedding.provider if embedding is not None else ""),
            "embedding_model": (embedding.model if embedding is not None else ""),
            "embedding_dimensions": (str(embedding.dimensions) if embedding is not None else ""),
            "embedding_task_type": (embedding.task_type if embedding is not None else ""),
            "rights_review_status": "",
            "rights_reviewed_by": "",
            "rights_reviewed_at": "",
            "rights_conditions": "",
            "review_decision": "",
            "reviewer": "",
            "reviewed_at": "",
            "rejection_reason": "",
            "review_notes": "",
        }

        for concept in PHASE1_CONCEPTS:
            proposal = chunk_proposals.get(concept)

            row[f"{concept.value}_proposal_rank"] = (
                str(proposal.proposal_rank) if proposal is not None else ""
            )
            row[f"{concept.value}_similarity"] = (
                format_float(proposal.anchor_similarity) if proposal is not None else ""
            )
            row[f"{concept.value}_proposed_weight"] = (
                format_float(proposal.proposed_weight) if proposal is not None else ""
            )
            row[f"{concept.value}_approved_weight"] = ""

        rows.append(row)

    return tuple(rows)


def write_csv(
    *,
    path: Path,
    field_names: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> None:
    """Write the editable CSV review packet."""

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
            fieldnames=list(field_names),
            extrasaction="raise",
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(rows)

    temporary.replace(path)


def write_html(
    *,
    path: Path,
    rows: Sequence[Mapping[str, str]],
) -> None:
    """Write a read-only HTML companion packet."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated_at = datetime.now(UTC).isoformat()

    body_rows = "\n".join(
        html_review_row(
            row=row,
            number=index,
        )
        for index, row in enumerate(
            rows,
            start=1,
        )
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WTH Phase 1 Corpus Review Packet</title>
<style>
body {{
  font-family: Arial, sans-serif;
  margin: 0;
  background: #f5f5f5;
  color: #161616;
}}
header {{
  position: sticky;
  top: 0;
  z-index: 10;
  padding: 18px 24px;
  background: #161616;
  color: white;
}}
main {{
  max-width: 1500px;
  margin: 0 auto;
  padding: 24px;
}}
.card {{
  background: white;
  border: 1px solid #d8d8d8;
  border-radius: 8px;
  margin-bottom: 24px;
  padding: 20px;
}}
.meta {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 10px;
}}
.label {{
  color: #525252;
  font-size: 12px;
  text-transform: uppercase;
}}
.value {{
  margin-top: 3px;
  overflow-wrap: anywhere;
}}
.chunk {{
  white-space: pre-wrap;
  line-height: 1.55;
  border-left: 4px solid #0f62fe;
  background: #f4f4f4;
  padding: 16px;
  margin: 16px 0;
}}
.concepts {{
  width: 100%;
  border-collapse: collapse;
  margin-top: 14px;
}}
.concepts th,
.concepts td {{
  border: 1px solid #d8d8d8;
  padding: 8px;
  text-align: left;
}}
.warning {{
  background: #fff1f1;
  border: 1px solid #da1e28;
  padding: 10px;
  margin-top: 12px;
}}
.review-note {{
  background: #edf5ff;
  border: 1px solid #0f62fe;
  padding: 12px;
  margin-top: 14px;
}}
</style>
</head>
<body>
<header>
  <strong>WTH Phase 1 Corpus Review Packet</strong><br>
  <span>{len(rows)} chunks — generated {html.escape(generated_at)}</span>
</header>
<main>
  <div class="review-note">
    Enter review decisions and approved concept weights in the CSV packet.
    The HTML packet is a read-only review companion.
  </div>
  {body_rows}
</main>
</body>
</html>
"""

    atomic_write_text(
        path,
        document,
    )


def html_review_row(
    *,
    row: Mapping[str, str],
    number: int,
) -> str:
    """Render one chunk as an HTML review card."""

    concept_rows = "\n".join(
        (
            "<tr>"
            f"<td>{html.escape(concept.value)}</td>"
            f"<td>{html.escape(str(row.get(f'{concept.value}_proposal_rank', '')))}</td>"
            f"<td>{html.escape(str(row.get(f'{concept.value}_similarity', '')))}</td>"
            f"<td>{html.escape(str(row.get(f'{concept.value}_proposed_weight', '')))}</td>"
            "</tr>"
        )
        for concept in PHASE1_CONCEPTS
    )

    warnings = row.get(
        "parser_warnings",
        "",
    )

    warning_block = ""

    if warnings:
        warning_block = (
            f'<div class="warning"><strong>Parser warnings:</strong> {html.escape(warnings)}</div>'
        )

    return f"""
<section class="card">
  <h2>{number}. {html.escape(row["source_title"])}</h2>
  <div class="meta">
    {_html_meta("Domain", row["domain"])}
    {_html_meta("Source ID", row["source_id"])}
    {_html_meta("Chunk ID", row["chunk_id"])}
    {_html_meta("Source type", row["source_type"])}
    {_html_meta("Author", row["author"])}
    {_html_meta("Translator", row["translator"])}
    {_html_meta("Licence", row["license_name"])}
    {_html_meta("Inclusion status", row["inclusion_status"])}
    {_html_meta("Tokens", row["token_count"])}
    {_html_meta("Embedding model", row["embedding_model"])}
    {_html_meta("Dimensions", row["embedding_dimensions"])}
    {_html_meta("Unit type", row["unit_type"])}
  </div>
  <p><strong>Citation:</strong> {html.escape(row["citation"])}</p>
  <p><strong>Locator:</strong>
     {html.escape(row["structural_locator"])}</p>
  <div class="chunk">{html.escape(row["chunk_text"])}</div>
  {warning_block}
  <table class="concepts">
    <thead>
      <tr>
        <th>Concept</th>
        <th>Rank</th>
        <th>Similarity</th>
        <th>Proposed weight</th>
      </tr>
    </thead>
    <tbody>
      {concept_rows}
    </tbody>
  </table>
</section>
"""


def _html_meta(
    label: str,
    value: str,
) -> str:
    """Render one metadata value."""

    return (
        "<div>"
        f'<div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value or "—")}</div>'
        "</div>"
    )


def format_float(value: float) -> str:
    """Format a review score without unnecessary trailing zeros."""

    return f"{value:.6f}".rstrip("0").rstrip(".")


def load_yaml(path: Path) -> object:
    """Load a UTF-8 YAML document."""

    if not path.exists():
        raise ReviewPacketError(f"Required YAML file does not exist: {path}")

    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ReviewPacketError(f"Invalid YAML file {path}: {exc}") from exc


def atomic_write_text(
    path: Path,
    value: str,
) -> None:
    """Write text using an atomic replacement."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(f".{path.name}.tmp")

    temporary.write_text(
        value,
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


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
