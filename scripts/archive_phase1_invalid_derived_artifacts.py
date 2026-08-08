"""Archive and remove invalid Phase 1 derived artifacts.

This action preserves artifacts created before the Phase 1 scope
correction, including mock or superseded embeddings, concept-weight
proposals, broad-corpus review packets, and failed activation outputs.

The script:

1. Verifies that the candidate corpus has already been frozen.
2. Verifies that the broad corpus has been classified as candidate-only.
3. Copies invalid derived artifacts into a checksummed archive.
4. Verifies every archived copy.
5. Removes the working copies only after successful verification.

The source corpus, parsed documents, chunks, acquisition manifests, and
candidate-corpus classification records are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final
from uuid import uuid4

LOGGER = logging.getLogger("wth.phase1.archive_invalid_derived_artifacts")

DEFAULT_PROJECT_ROOT: Final = Path()
DEFAULT_ARCHIVE_ROOT: Final = Path("artifacts/archive/pre_scope_correction")

DEFAULT_FREEZE_MANIFEST: Final = Path(
    "artifacts/archive/phase1_candidate_corpus_v1/freeze_manifest.json"
)
DEFAULT_CANDIDATE_MANIFEST: Final = Path(
    "artifacts/phase1/candidate/candidate_corpus_manifest.json"
)

DEFAULT_EXPECTED_CHUNK_COUNT: Final = 7_469
DEFAULT_CORPUS_VERSION: Final = "phase1_candidate_corpus_v1"

ARCHIVE_MANIFEST_NAME: Final = "archive_manifest.json"
STATUS_DOCUMENT_NAME: Final = "STATUS.md"
BUFFER_SIZE: Final = 1024 * 1024

INVALID_DERIVED_TARGETS: Final = (
    Path("artifacts/phase1/embeddings"),
    Path("artifacts/phase1/concepts/chunk_concept_proposals.jsonl"),
    Path("artifacts/review/phase1_review_packet.csv"),
    Path("artifacts/review/phase1_review_packet.html"),
    Path("artifacts/phase1/active"),
)

PROTECTED_PATHS: Final = (
    Path("data/raw"),
    Path("artifacts/phase1/parsed"),
    Path("artifacts/phase1/chunks"),
    Path("artifacts/phase1/acquisition_manifest.json"),
    Path("artifacts/phase1/manifests/ingestion_manifest.json"),
    Path("artifacts/phase1/candidate"),
    Path("artifacts/archive/phase1_candidate_corpus_v1"),
)


class ArchiveError(RuntimeError):
    """Raised when invalid artifacts cannot be archived safely."""


@dataclass(frozen=True, slots=True)
class ArchivedFileRecord:
    """Manifest record for one archived file."""

    source_relative_path: str
    archive_relative_path: str
    reason: str
    size_bytes: int
    sha256: str
    source_modified_at: str


@dataclass(frozen=True, slots=True)
class ArchiveManifest:
    """Audit manifest for the pre-scope-correction archive."""

    manifest_version: str
    archive_status: str
    created_at: str
    completed_at: str | None

    corpus_version: str
    candidate_chunk_count: int

    project_root: str
    archive_root: str

    freeze_manifest_path: str
    freeze_manifest_sha256: str

    candidate_manifest_path: str
    candidate_manifest_sha256: str

    target_paths: tuple[str, ...]
    protected_paths: tuple[str, ...]

    archived_file_count: int
    archived_total_size_bytes: int
    archived_files: tuple[ArchivedFileRecord, ...]

    originals_removed: bool
    removed_source_files: tuple[str, ...]
    removed_empty_directories: tuple[str, ...]

    diagnostic_evidence: dict[str, object]

    classification: tuple[str, ...]
    notes: tuple[str, ...]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Archive invalid Phase 1 derived artifacts created before the corpus-scope correction."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=DEFAULT_ARCHIVE_ROOT,
    )
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=DEFAULT_FREEZE_MANIFEST,
    )
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=DEFAULT_CANDIDATE_MANIFEST,
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
        "--keep-originals",
        action="store_true",
        help=("Archive and verify the files but do not remove the working copies."),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=("Validate prerequisites and list artifacts without copying or removing anything."),
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
    """Archive invalid derived artifacts."""

    args = parse_arguments()
    configure_logging(args.log_level)

    try:
        manifest = archive_invalid_artifacts(
            project_root=args.project_root,
            archive_root=args.archive_root,
            freeze_manifest_path=args.freeze_manifest,
            candidate_manifest_path=args.candidate_manifest,
            corpus_version=args.corpus_version,
            expected_chunk_count=args.expected_chunk_count,
            keep_originals=args.keep_originals,
            dry_run=args.dry_run,
        )
    except Exception:
        LOGGER.exception("Invalid-artifact archival failed")
        raise SystemExit(1) from None

    if manifest is None:
        LOGGER.info("Dry run completed successfully")
        return

    LOGGER.info(
        "Archive completed: %s",
        manifest.archive_root,
    )
    LOGGER.info(
        "Archived files: %d",
        manifest.archived_file_count,
    )
    LOGGER.info(
        "Archived bytes: %d",
        manifest.archived_total_size_bytes,
    )
    LOGGER.info(
        "Originals removed: %s",
        manifest.originals_removed,
    )


def archive_invalid_artifacts(
    *,
    project_root: Path,
    archive_root: Path,
    freeze_manifest_path: Path,
    candidate_manifest_path: Path,
    corpus_version: str,
    expected_chunk_count: int,
    keep_originals: bool,
    dry_run: bool,
) -> ArchiveManifest | None:
    """Archive invalid artifacts and optionally remove originals."""

    if expected_chunk_count < 1:
        raise ArchiveError("expected_chunk_count must be at least 1")

    project_root = project_root.resolve()

    archive_root = resolve_from_project(
        project_root=project_root,
        path=archive_root,
    )
    freeze_manifest_path = resolve_from_project(
        project_root=project_root,
        path=freeze_manifest_path,
    )
    candidate_manifest_path = resolve_from_project(
        project_root=project_root,
        path=candidate_manifest_path,
    )

    validate_prerequisites(
        freeze_manifest_path=freeze_manifest_path,
        candidate_manifest_path=candidate_manifest_path,
        corpus_version=corpus_version,
        expected_chunk_count=expected_chunk_count,
    )

    validate_archive_destination(
        project_root=project_root,
        archive_root=archive_root,
    )

    target_paths = tuple(project_root / path for path in INVALID_DERIVED_TARGETS)

    source_files = discover_source_files(
        project_root=project_root,
        target_paths=target_paths,
    )

    if not source_files:
        LOGGER.info( "No invalid derived artifacts were found. " "The working tree is already clean for Action 4.3." ) 
        return None

    diagnostic_evidence = collect_diagnostic_evidence(project_root)

    LOGGER.info(
        "Found %d invalid derived files",
        len(source_files),
    )

    for source_file in source_files:
        LOGGER.info(
            "Will archive: %s",
            source_file.relative_to(project_root).as_posix(),
        )

    if dry_run:
        log_diagnostic_summary(diagnostic_evidence)
        return None

    temporary_root = archive_root.with_name(f".{archive_root.name}.tmp-{uuid4().hex}")

    if temporary_root.exists():
        raise ArchiveError(f"Temporary archive path already exists: {temporary_root}")

    temporary_root.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    records: list[ArchivedFileRecord] = []

    try:
        temporary_root.mkdir(
            parents=True,
            exist_ok=False,
        )

        for source_file in source_files:
            records.append(
                copy_and_verify_file(
                    project_root=project_root,
                    archive_root=temporary_root,
                    source_file=source_file,
                )
            )

        preliminary_manifest = build_manifest(
            project_root=project_root,
            archive_root=archive_root,
            freeze_manifest_path=freeze_manifest_path,
            candidate_manifest_path=candidate_manifest_path,
            corpus_version=corpus_version,
            expected_chunk_count=expected_chunk_count,
            records=tuple(records),
            originals_removed=False,
            removed_source_files=(),
            removed_empty_directories=(),
            diagnostic_evidence=diagnostic_evidence,
            archive_status=("copied_verified_pending_source_removal"),
            completed_at=None,
        )

        write_archive_documents(
            archive_root=temporary_root,
            manifest=preliminary_manifest,
        )

        verify_archive(
            archive_root=temporary_root,
            records=tuple(records),
        )

        temporary_root.rename(archive_root)

    except Exception:
        shutil.rmtree(
            temporary_root,
            ignore_errors=True,
        )
        raise

    removed_files: tuple[str, ...] = ()
    removed_directories: tuple[str, ...] = ()

    if not keep_originals:
        removed_files = remove_source_files(
            project_root=project_root,
            source_files=source_files,
        )
        removed_directories = remove_empty_target_directories(
            project_root=project_root,
            target_paths=target_paths,
        )

    final_manifest = build_manifest(
        project_root=project_root,
        archive_root=archive_root,
        freeze_manifest_path=freeze_manifest_path,
        candidate_manifest_path=candidate_manifest_path,
        corpus_version=corpus_version,
        expected_chunk_count=expected_chunk_count,
        records=tuple(records),
        originals_removed=not keep_originals,
        removed_source_files=removed_files,
        removed_empty_directories=removed_directories,
        diagnostic_evidence=diagnostic_evidence,
        archive_status=(
            "archived_verified_originals_removed"
            if not keep_originals
            else "archived_verified_originals_retained"
        ),
        completed_at=datetime.now(UTC).isoformat(),
    )

    write_archive_documents(
        archive_root=archive_root,
        manifest=final_manifest,
    )

    verify_archive(
        archive_root=archive_root,
        records=final_manifest.archived_files,
    )

    if not keep_originals:
        verify_originals_removed(source_files=source_files)

    return final_manifest


def validate_prerequisites(
    *,
    freeze_manifest_path: Path,
    candidate_manifest_path: Path,
    corpus_version: str,
    expected_chunk_count: int,
) -> None:
    """Require Actions 4.1 and 4.2 to be complete."""

    freeze_manifest = load_json_object(freeze_manifest_path)
    candidate_manifest = load_json_object(candidate_manifest_path)

    if freeze_manifest.get("corpus_status") != "candidate_only_frozen":
        raise ArchiveError("Freeze manifest does not identify a candidate-only frozen corpus")

    if freeze_manifest.get("corpus_version") != corpus_version:
        raise ArchiveError("Freeze-manifest corpus version does not match the requested version")

    lifecycle = candidate_manifest.get("lifecycle")

    if not isinstance(lifecycle, dict):
        raise ArchiveError("Candidate manifest has no valid lifecycle")

    if lifecycle.get("corpus_classification") != "candidate_corpus":
        raise ArchiveError("Corpus has not been classified as a candidate corpus")

    if lifecycle.get("review_status") != "not_reviewed":
        raise ArchiveError("Candidate corpus must be classified as not reviewed")

    if lifecycle.get("phase1_activation_status") != "not_active":
        raise ArchiveError("Candidate corpus must be classified as not active")

    if lifecycle.get("direct_activation_eligible") is not False:
        raise ArchiveError("Candidate corpus must explicitly block direct activation")

    verified_count = candidate_manifest.get("verified_chunk_count")

    if verified_count != expected_chunk_count:
        raise ArchiveError(
            "Candidate corpus chunk count does not match "
            f"the expected count: {verified_count!r} != "
            f"{expected_chunk_count}"
        )


def validate_archive_destination(
    *,
    project_root: Path,
    archive_root: Path,
) -> None:
    """Validate the archive destination."""

    if archive_root.exists():
        raise ArchiveError(
            "Archive destination already exists. "
            "Existing archives are never overwritten: "
            f"{archive_root}"
        )

    for protected_path in PROTECTED_PATHS:
        protected = (project_root / protected_path).resolve()

        if archive_root == protected:
            raise ArchiveError("Archive destination cannot equal a protected corpus path")

        if is_relative_to(
            archive_root,
            protected,
        ):
            raise ArchiveError(
                f"Archive destination cannot be inside a protected corpus path: {protected}"
            )

    for target in INVALID_DERIVED_TARGETS:
        target_path = (project_root / target).resolve()

        if archive_root == target_path:
            raise ArchiveError("Archive destination cannot equal an invalid-artifact source path")

        if is_relative_to(
            archive_root,
            target_path,
        ):
            raise ArchiveError(
                "Archive destination cannot be inside an "
                f"invalid-artifact source path: {target_path}"
            )


def discover_source_files(
    *,
    project_root: Path,
    target_paths: Iterable[Path],
) -> tuple[Path, ...]:
    """Discover regular files under all invalid targets."""

    files_by_relative_path: dict[str, Path] = {}

    for target_path in target_paths:
        if not target_path.exists():
            LOGGER.warning(
                "Invalid-artifact target is absent: %s",
                target_path,
            )
            continue

        if target_path.is_symlink():
            raise ArchiveError(f"Archive targets must not be symlinks: {target_path}")

        if target_path.is_file():
            add_source_file(
                project_root=project_root,
                source_file=target_path,
                files_by_relative_path=files_by_relative_path,
            )
            continue

        for candidate in sorted(target_path.rglob("*")):
            if candidate.is_symlink():
                raise ArchiveError(f"Symlinks are not allowed inside archive targets: {candidate}")

            if not candidate.is_file():
                continue

            add_source_file(
                project_root=project_root,
                source_file=candidate,
                files_by_relative_path=files_by_relative_path,
            )

    return tuple(files_by_relative_path[key] for key in sorted(files_by_relative_path))


def add_source_file(
    *,
    project_root: Path,
    source_file: Path,
    files_by_relative_path: dict[str, Path],
) -> None:
    """Add one source file to the archive collection."""

    resolved_source = source_file.resolve()

    if not is_relative_to(
        resolved_source,
        project_root,
    ):
        raise ArchiveError(f"Artifact path is outside the project root: {resolved_source}")

    for protected_path in PROTECTED_PATHS:
        protected = (project_root / protected_path).resolve()

        if resolved_source == protected or is_relative_to(
            resolved_source,
            protected,
        ):
            raise ArchiveError(
                f"Refusing to archive a protected corpus artifact: {resolved_source}"
            )

    relative_path = resolved_source.relative_to(project_root).as_posix()

    files_by_relative_path[relative_path] = resolved_source


def copy_and_verify_file(
    *,
    project_root: Path,
    archive_root: Path,
    source_file: Path,
) -> ArchivedFileRecord:
    """Copy one artifact and verify its archived checksum."""

    relative_path = source_file.relative_to(project_root)
    destination = archive_root / relative_path

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_checksum = sha256_file(source_file)

    shutil.copy2(
        source_file,
        destination,
    )

    archived_checksum = sha256_file(destination)

    if archived_checksum != source_checksum:
        raise ArchiveError(f"Checksum mismatch after archiving: {relative_path.as_posix()}")

    source_stat = source_file.stat()

    return ArchivedFileRecord(
        source_relative_path=relative_path.as_posix(),
        archive_relative_path=relative_path.as_posix(),
        reason=classification_reason(relative_path),
        size_bytes=source_stat.st_size,
        sha256=source_checksum,
        source_modified_at=(
            datetime.fromtimestamp(
                source_stat.st_mtime,
                tz=UTC,
            ).isoformat()
        ),
    )


def classification_reason(
    relative_path: Path,
) -> str:
    """Return why an artifact is being archived."""

    normalized = relative_path.as_posix()

    if normalized.startswith("artifacts/phase1/embeddings/"):
        return (
            "Embedding artifact created for the broad "
            "pre-scope-correction corpus. It may contain "
            "mock, incomplete, or superseded vectors."
        )

    if normalized.endswith("chunk_concept_proposals.jsonl"):
        return (
            "Concept-weight proposals depend on mock or "
            "superseded embeddings and the unfiltered "
            "7,469-chunk candidate corpus."
        )

    if normalized.startswith("artifacts/review/"):
        return (
            "Review packet was generated from the broad "
            "candidate corpus and may contain mock or zero "
            "concept weights. It is not the Phase 1 "
            "selection-review packet."
        )

    if normalized.startswith("artifacts/phase1/active/"):
        return (
            "Activation artifact was produced by a failed "
            "bulk activation attempt against the unreviewed "
            "candidate corpus."
        )

    return (
        "Derived artifact was created before the Phase 1 "
        "scope correction and is not valid for continued use."
    )


def collect_diagnostic_evidence(
    project_root: Path,
) -> dict[str, object]:
    """Collect evidence supporting artifact invalidation."""

    evidence: dict[str, object] = {}

    chunk_embeddings = project_root / "artifacts/phase1/embeddings/chunk_embeddings.jsonl"
    anchor_embeddings = project_root / "artifacts/phase1/embeddings/concept_anchor_embeddings.json"
    proposals = project_root / "artifacts/phase1/concepts/chunk_concept_proposals.jsonl"
    review_packet = project_root / "artifacts/review/phase1_review_packet.csv"
    activation_manifest = project_root / "artifacts/phase1/active/activation_manifest.json"

    if chunk_embeddings.exists():
        evidence["chunk_embeddings"] = inspect_jsonl_first_record(
            chunk_embeddings,
            keys=(
                "provider",
                "model",
                "dimensions",
                "task_type",
            ),
        )

    if anchor_embeddings.exists():
        evidence["concept_anchor_embeddings"] = inspect_anchor_embeddings(anchor_embeddings)

    if proposals.exists():
        evidence["concept_proposals"] = inspect_concept_proposals(proposals)

    if review_packet.exists():
        evidence["review_packet"] = inspect_review_packet(review_packet)

    if activation_manifest.exists():
        evidence["activation_manifest"] = inspect_activation_manifest(activation_manifest)

    return evidence


def inspect_jsonl_first_record(
    path: Path,
    *,
    keys: Iterable[str],
) -> dict[str, object]:
    """Inspect metadata from the first JSONL record."""

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        raw = json.loads(line)

        if not isinstance(raw, dict):
            raise ArchiveError(f"Expected JSON object in {path}")

        return {key: raw.get(key) for key in keys}

    return {
        "status": "empty_file",
    }


def inspect_anchor_embeddings(
    path: Path,
) -> dict[str, object]:
    """Inspect concept-anchor embedding metadata."""

    raw = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(raw, list):
        return {
            "status": "unexpected_format",
        }

    if not raw:
        return {
            "record_count": 0,
        }

    first = raw[0]

    if not isinstance(first, dict):
        return {
            "status": "unexpected_record_format",
            "record_count": len(raw),
        }

    return {
        "record_count": len(raw),
        "provider": first.get("provider"),
        "model": first.get("model"),
        "dimensions": first.get("dimensions"),
        "task_type": first.get("task_type"),
        "anchor_version": first.get("anchor_version"),
    }


def inspect_concept_proposals(
    path: Path,
) -> dict[str, object]:
    """Inspect concept-proposal counts and weight distribution."""

    record_count = 0
    zero_weight_count = 0
    positive_weight_count = 0
    similarities: list[float] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ArchiveError(f"Invalid concept proposal at {path}:{line_number}") from exc

            if not isinstance(record, dict):
                raise ArchiveError(f"Concept proposal must be an object at {path}:{line_number}")

            record_count += 1

            weight = record.get("proposed_weight")
            similarity = record.get("anchor_similarity")

            if isinstance(weight, int | float):
                if float(weight) > 0:
                    positive_weight_count += 1
                else:
                    zero_weight_count += 1

            if isinstance(
                similarity,
                int | float,
            ):
                similarities.append(float(similarity))

    result: dict[str, object] = {
        "record_count": record_count,
        "zero_weight_count": zero_weight_count,
        "positive_weight_count": positive_weight_count,
    }

    if similarities:
        result.update(
            {
                "minimum_similarity": min(similarities),
                "maximum_similarity": max(similarities),
            }
        )

    return result


def inspect_review_packet(
    path: Path,
) -> dict[str, object]:
    """Inspect broad-corpus review-packet evidence."""

    row_count = 0
    all_zero_weight_rows = 0
    embedding_providers: Counter[str] = Counter()

    weight_columns = (
        "consciousness_proposed_weight",
        "self_identity_proposed_weight",
        "reality_appearance_proposed_weight",
    )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            return {
                "status": "missing_header",
            }

        for row in reader:
            row_count += 1

            provider = (
                row.get(
                    "embedding_provider",
                    "",
                )
                or ""
            ).strip()

            if provider:
                embedding_providers[provider] += 1

            weights = tuple(parse_optional_float(row.get(column)) for column in weight_columns)

            if all(
                weight
                in {
                    None,
                    0.0,
                }
                for weight in weights
            ):
                all_zero_weight_rows += 1

    return {
        "row_count": row_count,
        "all_zero_weight_rows": all_zero_weight_rows,
        "embedding_providers": dict(sorted(embedding_providers.items())),
    }


def inspect_activation_manifest(
    path: Path,
) -> dict[str, object]:
    """Inspect failed activation evidence."""

    raw = load_json_object(path)
    failures = raw.get("failures")

    if not isinstance(failures, list):
        return {
            "status": "no_failure_list",
        }

    reason_counts: Counter[str] = Counter()

    for failure in failures:
        if not isinstance(failure, dict):
            continue

        reason = failure.get("reason")

        if isinstance(reason, str):
            reason_counts[reason] += 1

    return {
        "failure_count": len(failures),
        "top_failure_reasons": dict(reason_counts.most_common(10)),
    }


def parse_optional_float(
    value: str | None,
) -> float | None:
    """Parse an optional numeric CSV value."""

    if value is None:
        return None

    normalized = value.strip()

    if not normalized:
        return None

    try:
        return float(normalized)
    except ValueError:
        return None


def build_manifest(
    *,
    project_root: Path,
    archive_root: Path,
    freeze_manifest_path: Path,
    candidate_manifest_path: Path,
    corpus_version: str,
    expected_chunk_count: int,
    records: tuple[ArchivedFileRecord, ...],
    originals_removed: bool,
    removed_source_files: tuple[str, ...],
    removed_empty_directories: tuple[str, ...],
    diagnostic_evidence: dict[str, object],
    archive_status: str,
    completed_at: str | None,
) -> ArchiveManifest:
    """Build the archive audit manifest."""

    return ArchiveManifest(
        manifest_version="1.0",
        archive_status=archive_status,
        created_at=datetime.now(UTC).isoformat(),
        completed_at=completed_at,
        corpus_version=corpus_version,
        candidate_chunk_count=expected_chunk_count,
        project_root=str(project_root),
        archive_root=str(archive_root),
        freeze_manifest_path=str(freeze_manifest_path),
        freeze_manifest_sha256=sha256_file(freeze_manifest_path),
        candidate_manifest_path=str(candidate_manifest_path),
        candidate_manifest_sha256=sha256_file(candidate_manifest_path),
        target_paths=tuple(path.as_posix() for path in INVALID_DERIVED_TARGETS),
        protected_paths=tuple(path.as_posix() for path in PROTECTED_PATHS),
        archived_file_count=len(records),
        archived_total_size_bytes=sum(record.size_bytes for record in records),
        archived_files=records,
        originals_removed=originals_removed,
        removed_source_files=removed_source_files,
        removed_empty_directories=removed_empty_directories,
        diagnostic_evidence=diagnostic_evidence,
        classification=(
            "invalid_derived_artifacts",
            "pre_scope_correction",
            "not_for_phase1_activation",
            "not_for_production_retrieval",
            "retained_for_audit_only",
        ),
        notes=(
            (
                "The archived artifacts were generated "
                "before the three-concept Phase 1 corpus "
                "scope was corrected."
            ),
            ("The 7,469 source-derived chunks remain preserved as the candidate corpus."),
            (
                "This archive does not contain the raw "
                "sources, parsed documents, chunk corpus, "
                "or candidate-corpus classification."
            ),
            (
                "New embeddings, review packets, and "
                "activation artifacts must be generated "
                "only for the approved Phase 1 vertical "
                "slice."
            ),
        ),
    )


def remove_source_files(
    *,
    project_root: Path,
    source_files: Iterable[Path],
) -> tuple[str, ...]:
    """Remove archived working files."""

    removed: list[str] = []

    for source_file in source_files:
        if not source_file.exists():
            continue

        source_file.unlink()

        removed.append(source_file.relative_to(project_root).as_posix())

    return tuple(sorted(removed))


def remove_empty_target_directories(
    *,
    project_root: Path,
    target_paths: Iterable[Path],
) -> tuple[str, ...]:
    """Remove directories left empty after archival."""

    directories: set[Path] = set()

    for target_path in target_paths:
        if not target_path.exists():
            continue

        if target_path.is_file():
            continue

        directories.update(path for path in target_path.rglob("*") if path.is_dir())
        directories.add(target_path)

    removed: list[str] = []

    for directory in sorted(
        directories,
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            continue

        removed.append(directory.relative_to(project_root).as_posix())

    return tuple(sorted(removed))


def verify_archive(
    *,
    archive_root: Path,
    records: Iterable[ArchivedFileRecord],
) -> None:
    """Verify every archived file against the manifest."""

    for record in records:
        archived_path = archive_root / record.archive_relative_path

        if not archived_path.is_file():
            raise ArchiveError(f"Archived file is missing: {record.archive_relative_path}")

        if archived_path.stat().st_size != record.size_bytes:
            raise ArchiveError(
                f"Archived file size differs from manifest: {record.archive_relative_path}"
            )

        if sha256_file(archived_path) != record.sha256:
            raise ArchiveError(
                f"Archived checksum differs from manifest: {record.archive_relative_path}"
            )


def verify_originals_removed(
    *,
    source_files: Iterable[Path],
) -> None:
    """Require every archived working file to be absent."""

    remaining = tuple(source_file for source_file in source_files if source_file.exists())

    if remaining:
        raise ArchiveError(
            "Some archived working files were not removed: "
            + ", ".join(str(path) for path in remaining)
        )


def write_archive_documents(
    *,
    archive_root: Path,
    manifest: ArchiveManifest,
) -> None:
    """Write machine-readable and human-readable records."""

    archive_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    atomic_write_text(
        archive_root / ARCHIVE_MANIFEST_NAME,
        json.dumps(
            asdict(manifest),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )

    atomic_write_text(
        archive_root / STATUS_DOCUMENT_NAME,
        render_status_document(manifest),
    )


def render_status_document(
    manifest: ArchiveManifest,
) -> str:
    """Render the human-readable archive status."""

    archived_files = "\n".join(
        (
            f"- `{record.source_relative_path}`\n"
            f"  - Reason: {record.reason}\n"
            f"  - SHA-256: `{record.sha256}`"
        )
        for record in manifest.archived_files
    )

    evidence = json.dumps(
        manifest.diagnostic_evidence,
        ensure_ascii=False,
        indent=2,
    )

    return f"""# Pre-Scope-Correction Artifact Archive

## Status

- **Archive status:** `{manifest.archive_status}`
- **Corpus version:** `{manifest.corpus_version}`
- **Candidate chunk count:** {manifest.candidate_chunk_count:,}
- **Archived files:** {manifest.archived_file_count}
- **Archived bytes:** {manifest.archived_total_size_bytes:,}
- **Original working files removed:** {manifest.originals_removed}
- **Completed at:** `{manifest.completed_at or "pending"}`

## Classification

The files in this archive are invalid or superseded derived artifacts
created before correction of the Phase 1 three-concept corpus scope.

They are retained for audit and debugging only.

They must not be used for:

- Phase 1 production retrieval;
- final concept-weight evaluation;
- human-review decisions;
- active corpus generation;
- final claim-cited answers.

## Preserved candidate corpus

The following remain outside this archive and are unchanged:

- `data/raw/`
- `artifacts/phase1/parsed/`
- `artifacts/phase1/chunks/`
- `artifacts/phase1/acquisition_manifest.json`
- `artifacts/phase1/manifests/ingestion_manifest.json`
- `artifacts/phase1/candidate/`
- `artifacts/archive/phase1_candidate_corpus_v1/`

## Archived files

{archived_files}

## Diagnostic evidence

```json
{evidence}
```

## Next valid derived artifacts

New artifacts must be generated only after:

1. source-structure analysis;
2. source-section scope approval;
3. rule-based Phase 1 selection;
4. human review;
5. build/development/held-out split freezing;
6. embedding-model selection.

The next review packet must describe the balanced consciousness,
self/identity, and reality/appearance vertical slice—not the complete
7,469-chunk candidate corpus.
"""


def load_json_object(
    path: Path,
) -> dict[str, object]:
    """Load a JSON object from a required file."""

    if not path.is_file():
        raise ArchiveError(f"Required JSON file does not exist: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArchiveError(f"Invalid JSON file: {path}") from exc

    if not isinstance(raw, dict):
        raise ArchiveError(f"JSON file must contain an object: {path}")

    return {str(key): value for key, value in raw.items()}


def resolve_from_project(
    *,
    project_root: Path,
    path: Path,
) -> Path:
    """Resolve a path relative to the project root."""

    if path.is_absolute():
        return path.resolve()

    return (project_root / path).resolve()


def sha256_file(path: Path) -> str:
    """Calculate a file SHA-256 checksum."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while block := file.read(BUFFER_SIZE):
            digest.update(block)

    return digest.hexdigest()


def atomic_write_text(
    path: Path,
    content: str,
) -> None:
    """Write text using atomic replacement."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(f".{path.name}.tmp")

    temporary.write_text(
        content,
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def is_relative_to(
    path: Path,
    possible_parent: Path,
) -> bool:
    """Return whether a path is below another path."""

    try:
        path.relative_to(possible_parent)
    except ValueError:
        return False

    return True


def log_diagnostic_summary(
    evidence: Mapping[str, object],
) -> None:
    """Log diagnostic evidence during a dry run."""

    LOGGER.info(
        "Diagnostic evidence: %s",
        json.dumps(
            evidence,
            ensure_ascii=False,
            sort_keys=True,
        ),
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
