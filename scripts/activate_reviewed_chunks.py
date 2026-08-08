"""Validate reviewed Phase 1 chunks and create active-corpus artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Final

import yaml

from apps.api.models.corpus import (
    ActiveChunkBundle,
    ActivationFailure,
    ActivationManifest,
    ChunkDraft,
    ChunkEmbeddingRecord,
    ChunkReviewDecision,
    ReviewedConceptWeight,
    ReviewDecisionType,
    RejectionReason,
    RightsStatus,
    SourceCatalogueEntry,
    SourceRightsReview,
)
from apps.api.models.enums import ConceptSlug, ReviewStatus


LOGGER = logging.getLogger("wth.phase1.activation")

DEFAULT_REVIEW_PACKET: Final = Path(
    "artifacts/review/phase1_review_packet.csv"
)
DEFAULT_CATALOGUE_PATH: Final = Path(
    "docs/catalogues/phase1_sources.yaml"
)
DEFAULT_CHUNKS_ROOT: Final = Path(
    "artifacts/phase1/chunks"
)
DEFAULT_EMBEDDINGS_PATH: Final = Path(
    "artifacts/phase1/embeddings/chunk_embeddings.jsonl"
)
DEFAULT_OUTPUT_ROOT: Final = Path(
    "artifacts/phase1/active"
)
DEFAULT_CORPUS_VERSION: Final = "phase1-three-concept-v1"

PHASE1_CONCEPTS: Final = (
    ConceptSlug.CONSCIOUSNESS,
    ConceptSlug.SELF_IDENTITY,
    ConceptSlug.REALITY_APPEARANCE,
)

APPROVED_DECISIONS: Final = frozenset(
    {
        ReviewDecisionType.APPROVE,
        ReviewDecisionType.APPROVE_WITH_EDITS,
    }
)


class ActivationError(RuntimeError):
    """Raised when reviewed chunks cannot be activated."""


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate reviewed Phase 1 chunks and create "
            "active-corpus artifacts."
        )
    )

    parser.add_argument(
        "--review-packet",
        type=Path,
        default=DEFAULT_REVIEW_PACKET,
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
        "--embeddings",
        type=Path,
        default=DEFAULT_EMBEDDINGS_PATH,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--corpus-version",
        default=DEFAULT_CORPUS_VERSION,
    )
    parser.add_argument(
        "--activated-by",
        required=True,
    )
    parser.add_argument(
        "--accept-proposed-weights",
        action="store_true",
        help=(
            "Use proposed weights where approved-weight cells "
            "are blank. Keep disabled for strict human review."
        ),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Write successfully validated active chunks even when "
            "other approved rows fail validation."
        ),
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
    """Validate review decisions and produce active artifacts."""

    args = parse_arguments()
    configure_logging(args.log_level)

    try:
        sources = load_catalogue(args.catalogue)
        chunks = load_chunks(args.chunks_root)
        embeddings = load_embeddings(args.embeddings)
        review_rows = load_review_rows(
            args.review_packet
        )

        requested_ids: list[str] = []
        active_records: list[dict[str, object]] = []
        concept_records: list[dict[str, object]] = []
        review_records: list[dict[str, object]] = []
        failures: list[ActivationFailure] = []

        seen_chunk_ids: set[str] = set()

        for row in review_rows:
            decision = parse_review_decision(
                row.get(
                    "review_decision",
                    "",
                )
            )

            if decision not in APPROVED_DECISIONS:
                continue

            chunk_id = required_cell(
                row,
                "chunk_id",
            )

            if chunk_id in seen_chunk_ids:
                raise ActivationError(
                    "Review packet contains duplicate approved "
                    f"chunk rows: {chunk_id}"
                )

            seen_chunk_ids.add(chunk_id)
            requested_ids.append(chunk_id)

            try:
                bundle = build_active_bundle(
                    row=row,
                    sources=sources,
                    chunks=chunks,
                    embeddings=embeddings,
                    corpus_version=args.corpus_version,
                    accept_proposed_weights=(
                        args.accept_proposed_weights
                    ),
                )
            except Exception as exc:
                failures.append(
                    ActivationFailure(
                        chunk_id=chunk_id,
                        reason=(
                            f"{type(exc).__name__}: {exc}"
                        ),
                    )
                )
                continue

            active_chunk = bundle.chunk.model_copy(
                update={
                    "review_status": ReviewStatus.ACTIVE,
                }
            )

            active_records.append(
                {
                    "source": bundle.source.model_dump(
                        mode="json"
                    ),
                    "rights_review": (
                        bundle.rights_review.model_dump(
                            mode="json"
                        )
                    ),
                    "chunk": active_chunk.model_dump(
                        mode="json"
                    ),
                    "embedding": (
                        bundle.embedding.model_dump(
                            mode="json"
                        )
                    ),
                    "review": bundle.review.model_dump(
                        mode="json"
                    ),
                    "corpus_version": (
                        bundle.corpus_version
                    ),
                }
            )

            for weight in bundle.review.concept_weights:
                concept_records.append(
                    {
                        "chunk_id": chunk_id,
                        "concept_id": str(
                            weight.concept_id
                        ),
                        "concept_slug": (
                            weight.concept_slug.value
                        ),
                        "weight": (
                            weight.approved_weight
                        ),
                        "reviewed": True,
                        "corpus_version": (
                            bundle.corpus_version
                        ),
                    }
                )

            review_records.append(
                bundle.review.model_dump(
                    mode="json"
                )
            )

        if not requested_ids:
            raise ActivationError(
                "The review packet contains no approved chunks"
            )

        activated_ids = tuple(
            str(
                record["chunk"]["chunk_id"]
            )
            for record in active_records
        )

        manifest = ActivationManifest(
            corpus_version=args.corpus_version,
            requested_chunk_ids=tuple(
                requested_ids
            ),
            activated_chunk_ids=activated_ids,
            failures=tuple(failures),
            activated_by=args.activated_by,
        )

        if failures and not args.allow_partial:
            write_activation_manifest(
                output_root=args.output_root,
                manifest=manifest,
            )

            raise ActivationError(
                f"{len(failures)} approved chunk(s) failed "
                "activation validation. Rerun with "
                "--allow-partial only after reviewing failures."
            )

        write_activation_outputs(
            output_root=args.output_root,
            active_records=active_records,
            concept_records=concept_records,
            review_records=review_records,
            manifest=manifest,
        )

        LOGGER.info(
            "Activated %d of %d approved chunks",
            len(active_records),
            len(requested_ids),
        )

        if failures:
            LOGGER.warning(
                "%d approved chunks failed validation",
                len(failures),
            )

    except Exception:
        LOGGER.exception(
            "Reviewed-chunk activation failed"
        )
        raise SystemExit(1) from None


def build_active_bundle(
    *,
    row: Mapping[str, str],
    sources: Mapping[str, SourceCatalogueEntry],
    chunks: Mapping[str, ChunkDraft],
    embeddings: Mapping[
        str,
        ChunkEmbeddingRecord,
    ],
    corpus_version: str,
    accept_proposed_weights: bool,
) -> ActiveChunkBundle:
    """Build and validate one complete active-chunk bundle."""

    chunk_id = required_cell(
        row,
        "chunk_id",
    )
    source_id = required_cell(
        row,
        "source_id",
    )

    source = sources.get(source_id)

    if source is None:
        raise ActivationError(
            f"Unknown source ID: {source_id}"
        )

    chunk = chunks.get(chunk_id)

    if chunk is None:
        raise ActivationError(
            f"Unknown chunk ID: {chunk_id}"
        )

    embedding = embeddings.get(chunk_id)

    if embedding is None:
        raise ActivationError(
            f"Chunk has no embedding: {chunk_id}"
        )

    rights_review = SourceRightsReview(
        source_id=source_id,
        status=RightsStatus(
            required_cell(
                row,
                "rights_review_status",
            )
        ),
        license_name=source.license_name,
        license_url=source.license_url,
        rights_statement=source.rights_statement,
        rights_jurisdiction=(
            source.rights_jurisdiction
        ),
        conditions=parse_conditions(
            row.get(
                "rights_conditions",
                "",
            )
        ),
        reviewed_by=required_cell(
            row,
            "rights_reviewed_by",
        ),
        reviewed_at=parse_datetime(
            required_cell(
                row,
                "rights_reviewed_at",
            ),
            field_name="rights_reviewed_at",
        ),
    )

    review_decision = ReviewDecisionType(
        required_cell(
            row,
            "review_decision",
        )
    )

    concept_weights = parse_concept_weights(
        row=row,
        accept_proposed_weights=(
            accept_proposed_weights
        ),
    )

    review = ChunkReviewDecision(
        chunk_id=chunk_id,
        decision=review_decision,
        concept_weights=concept_weights,
        reviewer=required_cell(
            row,
            "reviewer",
        ),
        reviewed_at=parse_datetime(
            required_cell(
                row,
                "reviewed_at",
            ),
            field_name="reviewed_at",
        ),
        rejection_reason=parse_rejection_reason(
            row.get(
                "rejection_reason",
                "",
            )
        ),
        notes=optional_cell(
            row,
            "review_notes",
        ),
    )

    reviewed_chunk = chunk.model_copy(
        update={
            "review_status": ReviewStatus.REVIEWED,
        }
    )

    return ActiveChunkBundle(
        source=source,
        rights_review=rights_review,
        chunk=reviewed_chunk,
        embedding=embedding,
        review=review,
        corpus_version=corpus_version,
    )


def parse_concept_weights(
    *,
    row: Mapping[str, str],
    accept_proposed_weights: bool,
) -> tuple[ReviewedConceptWeight, ...]:
    """Parse reviewed concept weights from one CSV row."""

    weights: list[ReviewedConceptWeight] = []

    for concept in PHASE1_CONCEPTS:
        approved_key = (
            f"{concept.value}_approved_weight"
        )
        proposed_key = (
            f"{concept.value}_proposed_weight"
        )

        approved_value = row.get(
            approved_key,
            "",
        ).strip()

        proposed_value = row.get(
            proposed_key,
            "",
        ).strip()

        if (
            not approved_value
            and accept_proposed_weights
        ):
            approved_value = proposed_value

        if not approved_value:
            continue

        approved_weight = parse_weight(
            approved_value,
            field_name=approved_key,
        )

        proposed_weight = (
            parse_weight(
                proposed_value,
                field_name=proposed_key,
            )
            if proposed_value
            else None
        )

        concept_id = concept_id_from_row(
            concept
        )

        weights.append(
            ReviewedConceptWeight(
                concept_id=concept_id,
                concept_slug=concept,
                proposed_weight=proposed_weight,
                approved_weight=approved_weight,
            )
        )

    if not weights:
        raise ActivationError(
            "Approved chunk requires at least one "
            "approved concept weight"
        )

    if not any(
        weight.approved_weight > 0
        for weight in weights
    ):
        raise ActivationError(
            "At least one approved concept weight "
            "must be greater than zero"
        )

    return tuple(weights)


def concept_id_from_row(
    concept: ConceptSlug,
):
    """Return the fixed UUID assigned to a Phase 1 concept."""

    from uuid import UUID

    identifiers = {
        ConceptSlug.SELF_IDENTITY: UUID(
            "10000000-0000-4000-8000-000000000001"
        ),
        ConceptSlug.CONSCIOUSNESS: UUID(
            "10000000-0000-4000-8000-000000000002"
        ),
        ConceptSlug.REALITY_APPEARANCE: UUID(
            "10000000-0000-4000-8000-000000000003"
        ),
    }

    return identifiers[concept]


def load_review_rows(
    path: Path,
) -> tuple[dict[str, str], ...]:
    """Load the edited CSV review packet."""

    if not path.exists():
        raise ActivationError(
            f"Review packet does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ActivationError(
                "Review packet contains no header"
            )

        return tuple(
            {
                str(key): value or ""
                for key, value in row.items()
                if key is not None
            }
            for row in reader
        )


def load_catalogue(
    path: Path,
) -> dict[str, SourceCatalogueEntry]:
    """Load source catalogue records."""

    raw = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(raw, list):
        raise ActivationError(
            "Source catalogue must be a YAML list"
        )

    sources: dict[
        str,
        SourceCatalogueEntry,
    ] = {}

    for item in raw:
        source = SourceCatalogueEntry.model_validate(
            item
        )
        sources[source.source_id] = source

    return sources


def load_chunks(
    root: Path,
) -> dict[str, ChunkDraft]:
    """Load and index draft chunks."""

    chunks: dict[str, ChunkDraft] = {}

    for path in sorted(root.glob("*.json")):
        raw = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(raw, list):
            raise ActivationError(
                f"Chunk artifact must be a list: {path}"
            )

        for item in raw:
            chunk = ChunkDraft.model_validate(
                item
            )

            if chunk.chunk_id in chunks:
                raise ActivationError(
                    "Duplicate chunk ID: "
                    f"{chunk.chunk_id}"
                )

            chunks[chunk.chunk_id] = chunk

    return chunks


def load_embeddings(
    path: Path,
) -> dict[str, ChunkEmbeddingRecord]:
    """Load and index chunk embeddings."""

    records: dict[
        str,
        ChunkEmbeddingRecord,
    ] = {}

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue

        record = (
            ChunkEmbeddingRecord.model_validate_json(
                line
            )
        )

        if record.chunk_id in records:
            raise ActivationError(
                "Duplicate embedding for chunk: "
                f"{record.chunk_id}"
            )

        records[record.chunk_id] = record

    return records


def parse_review_decision(
    value: str,
) -> ReviewDecisionType | None:
    """Parse an optional review decision."""

    normalized = value.strip()

    if not normalized:
        return None

    try:
        return ReviewDecisionType(normalized)
    except ValueError as exc:
        raise ActivationError(
            f"Invalid review decision: {normalized}"
        ) from exc


def parse_rejection_reason(
    value: str,
) -> RejectionReason | None:
    """Parse an optional rejection reason."""

    normalized = value.strip()

    if not normalized:
        return None

    return RejectionReason(normalized)


def parse_datetime(
    value: str,
    *,
    field_name: str,
) -> datetime:
    """Parse and require a timezone-aware ISO timestamp."""

    try:
        result = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ActivationError(
            f"{field_name} must be an ISO timestamp"
        ) from exc

    if result.tzinfo is None:
        raise ActivationError(
            f"{field_name} must include a timezone"
        )

    return result


def parse_weight(
    value: str,
    *,
    field_name: str,
) -> float:
    """Parse a 0–1 concept weight."""

    try:
        weight = float(value)
    except ValueError as exc:
        raise ActivationError(
            f"{field_name} must be numeric"
        ) from exc

    if not 0.0 <= weight <= 1.0:
        raise ActivationError(
            f"{field_name} must be between 0 and 1"
        )

    return weight


def parse_conditions(
    value: str,
) -> tuple[str, ...]:
    """Parse pipe-separated rights conditions."""

    return tuple(
        condition.strip()
        for condition in value.split("|")
        if condition.strip()
    )


def required_cell(
    row: Mapping[str, str],
    field_name: str,
) -> str:
    """Return a non-empty CSV value."""

    value = row.get(
        field_name,
        "",
    ).strip()

    if not value:
        raise ActivationError(
            f"Required review field is empty: {field_name}"
        )

    return value


def optional_cell(
    row: Mapping[str, str],
    field_name: str,
) -> str | None:
    """Return an optional normalized CSV value."""

    value = row.get(
        field_name,
        "",
    ).strip()

    return value or None


def write_activation_outputs(
    *,
    output_root: Path,
    active_records: Iterable[
        Mapping[str, object]
    ],
    concept_records: Iterable[
        Mapping[str, object]
    ],
    review_records: Iterable[
        Mapping[str, object]
    ],
    manifest: ActivationManifest,
) -> None:
    """Write validated active-corpus artifacts."""

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_json_lines(
        output_root / "active_chunk_bundles.jsonl",
        active_records,
    )
    write_json_lines(
        output_root / "reviewed_chunk_concepts.jsonl",
        concept_records,
    )
    write_json_lines(
        output_root / "chunk_review_decisions.jsonl",
        review_records,
    )
    write_activation_manifest(
        output_root=output_root,
        manifest=manifest,
    )


def write_activation_manifest(
    *,
    output_root: Path,
    manifest: ActivationManifest,
) -> None:
    """Write the activation audit manifest."""

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    atomic_write_text(
        output_root / "activation_manifest.json",
        manifest.model_dump_json(
            indent=2,
        )
        + "\n",
    )


def write_json_lines(
    path: Path,
    values: Iterable[
        Mapping[str, object]
    ],
) -> None:
    """Write JSONL records atomically."""

    content = "\n".join(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for value in values
    )

    if content:
        content += "\n"

    atomic_write_text(
        path,
        content,
    )


def atomic_write_text(
    path: Path,
    value: str,
) -> None:
    """Write text with atomic replacement."""

    temporary = path.with_name(
        f".{path.name}.tmp"
    )

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
        format=(
            "%(asctime)s %(levelname)s "
            "%(name)s %(message)s"
        ),
    )


if __name__ == "__main__":
    main()
