"""Freeze the existing Phase 1 ingestion artifacts as a candidate corpus.

The snapshot preserves the broad source-derived corpus before Phase 1
relevance filtering begins. It copies the selected artifacts, verifies
every copy by SHA-256, and writes an auditable freeze manifest.

The script never modifies or deletes the original artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final
from uuid import uuid4

LOGGER = logging.getLogger("wth.phase1.freeze_candidate_corpus")

DEFAULT_PROJECT_ROOT: Final = Path()
DEFAULT_SNAPSHOT_ROOT: Final = Path("artifacts/archive/phase1_candidate_corpus_v1")
MANIFEST_NAME: Final = "freeze_manifest.json"
BUFFER_SIZE: Final = 1024 * 1024

PRESERVED_PATHS: Final = (
    Path("data/raw"),
    Path("artifacts/phase1/parsed"),
    Path("artifacts/phase1/chunks"),
    Path("artifacts/phase1/acquisition_manifest.json"),
    Path("artifacts/phase1/manifests/ingestion_manifest.json"),
)


class FreezeError(RuntimeError):
    """Raised when the candidate-corpus snapshot cannot be created."""


@dataclass(frozen=True, slots=True)
class FrozenFileRecord:
    """Manifest record for one frozen file."""

    relative_path: str
    size_bytes: int
    sha256: str
    source_modified_at: str


@dataclass(frozen=True, slots=True)
class FreezeManifest:
    """Audit manifest for the frozen candidate corpus."""

    snapshot_name: str
    corpus_status: str
    corpus_version: str
    created_at: str
    project_root: str
    snapshot_root: str
    preserved_paths: tuple[str, ...]
    file_count: int
    total_size_bytes: int
    files: tuple[FrozenFileRecord, ...]
    notes: tuple[str, ...]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Freeze the current Phase 1 ingestion artifacts as a "
            "versioned candidate-corpus snapshot."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Repository root. Default: current directory.",
    )
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        default=DEFAULT_SNAPSHOT_ROOT,
        help=(
            "Snapshot location relative to the project root. "
            "The destination must not already exist."
        ),
    )
    parser.add_argument(
        "--corpus-version",
        default="phase1_candidate_corpus_v1",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )

    return parser.parse_args()


def main() -> None:
    """Create and verify the frozen candidate-corpus snapshot."""

    args = parse_arguments()
    configure_logging(args.log_level)

    try:
        manifest = freeze_candidate_corpus(
            project_root=args.project_root,
            snapshot_root=args.snapshot_root,
            corpus_version=args.corpus_version,
        )
    except Exception:
        LOGGER.exception("Candidate-corpus snapshot failed")
        raise SystemExit(1) from None

    LOGGER.info(
        "Snapshot created: %s",
        manifest.snapshot_root,
    )
    LOGGER.info(
        "Files frozen: %d",
        manifest.file_count,
    )
    LOGGER.info(
        "Total size: %d bytes",
        manifest.total_size_bytes,
    )


def freeze_candidate_corpus(
    *,
    project_root: Path,
    snapshot_root: Path,
    corpus_version: str,
) -> FreezeManifest:
    """Copy, verify, and freeze the selected ingestion artifacts."""

    resolved_project_root = project_root.resolve()
    resolved_snapshot_root = resolve_snapshot_root(
        project_root=resolved_project_root,
        snapshot_root=snapshot_root,
    )

    validate_snapshot_location(
        project_root=resolved_project_root,
        snapshot_root=resolved_snapshot_root,
    )

    source_paths = tuple(resolved_project_root / relative_path for relative_path in PRESERVED_PATHS)

    validate_source_paths(source_paths)

    source_files = collect_source_files(
        project_root=resolved_project_root,
        source_paths=source_paths,
    )

    if not source_files:
        raise FreezeError("The preserved paths contain no files")

    temporary_root = resolved_snapshot_root.with_name(
        f".{resolved_snapshot_root.name}.tmp-{uuid4().hex}"
    )

    temporary_root.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if temporary_root.exists():
        raise FreezeError(f"Temporary snapshot path already exists: {temporary_root}")

    LOGGER.info(
        "Freezing %d files into %s",
        len(source_files),
        resolved_snapshot_root,
    )

    records: list[FrozenFileRecord] = []

    try:
        temporary_root.mkdir(
            parents=True,
            exist_ok=False,
        )

        create_preserved_directories(
            project_root=resolved_project_root,
            snapshot_root=temporary_root,
            source_paths=source_paths,
        )

        for source_file in source_files:
            record = copy_and_verify_file(
                project_root=resolved_project_root,
                snapshot_root=temporary_root,
                source_file=source_file,
            )
            records.append(record)

        manifest = build_manifest(
            project_root=resolved_project_root,
            final_snapshot_root=resolved_snapshot_root,
            corpus_version=corpus_version,
            records=tuple(records),
        )

        write_manifest(
            snapshot_root=temporary_root,
            manifest=manifest,
        )

        verify_snapshot(
            snapshot_root=temporary_root,
            records=manifest.files,
        )

        temporary_root.rename(resolved_snapshot_root)

    except Exception:
        shutil.rmtree(
            temporary_root,
            ignore_errors=True,
        )
        raise

    return manifest


def resolve_snapshot_root(
    *,
    project_root: Path,
    snapshot_root: Path,
) -> Path:
    """Resolve an absolute snapshot destination."""

    if snapshot_root.is_absolute():
        return snapshot_root.resolve()

    return (project_root / snapshot_root).resolve()


def validate_snapshot_location(
    *,
    project_root: Path,
    snapshot_root: Path,
) -> None:
    """Validate that the snapshot destination is safe."""

    if snapshot_root.exists():
        raise FreezeError(
            "Snapshot destination already exists. "
            "Frozen snapshots are never overwritten: "
            f"{snapshot_root}"
        )

    for preserved_path in PRESERVED_PATHS:
        source = (project_root / preserved_path).resolve()

        if snapshot_root == source:
            raise FreezeError("Snapshot destination cannot equal a preserved path")

        if is_relative_to(
            snapshot_root,
            source,
        ):
            raise FreezeError(
                f"Snapshot destination cannot be inside a preserved source path: {source}"
            )


def validate_source_paths(
    source_paths: Iterable[Path],
) -> None:
    """Require every preserved path to exist and reject symlinks."""

    for source_path in source_paths:
        if not source_path.exists():
            raise FreezeError(f"Required preserved path does not exist: {source_path}")

        if source_path.is_symlink():
            raise FreezeError(f"Preserved top-level paths must not be symlinks: {source_path}")


def collect_source_files(
    *,
    project_root: Path,
    source_paths: Iterable[Path],
) -> tuple[Path, ...]:
    """Collect unique regular files from all preserved paths."""

    files_by_relative_path: dict[
        str,
        Path,
    ] = {}

    for source_path in source_paths:
        if source_path.is_file():
            add_source_file(
                project_root=project_root,
                source_file=source_path,
                files_by_relative_path=files_by_relative_path,
            )
            continue

        for candidate in sorted(source_path.rglob("*")):
            if candidate.is_symlink():
                raise FreezeError(f"Symlinks are not allowed in frozen artifacts: {candidate}")

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
    """Add one regular source file to the deduplicated collection."""

    relative_path = source_file.relative_to(project_root).as_posix()

    files_by_relative_path[relative_path] = source_file


def create_preserved_directories(
    *,
    project_root: Path,
    snapshot_root: Path,
    source_paths: Iterable[Path],
) -> None:
    """Recreate preserved directory structure, including empty folders."""

    for source_path in source_paths:
        if source_path.is_file():
            destination_parent = snapshot_root / source_path.relative_to(project_root).parent
            destination_parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            continue

        destination_root = snapshot_root / source_path.relative_to(project_root)
        destination_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        for directory in sorted(path for path in source_path.rglob("*") if path.is_dir()):
            destination = snapshot_root / directory.relative_to(project_root)
            destination.mkdir(
                parents=True,
                exist_ok=True,
            )


def copy_and_verify_file(
    *,
    project_root: Path,
    snapshot_root: Path,
    source_file: Path,
) -> FrozenFileRecord:
    """Copy one file and verify that its checksum is unchanged."""

    relative_path = source_file.relative_to(project_root)
    destination = snapshot_root / relative_path

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_checksum = sha256_file(source_file)

    shutil.copy2(
        source_file,
        destination,
    )

    destination_checksum = sha256_file(destination)

    if destination_checksum != source_checksum:
        raise FreezeError(f"Checksum mismatch after copy: {relative_path.as_posix()}")

    source_stat = source_file.stat()

    return FrozenFileRecord(
        relative_path=relative_path.as_posix(),
        size_bytes=source_stat.st_size,
        sha256=source_checksum,
        source_modified_at=datetime.fromtimestamp(
            source_stat.st_mtime,
            tz=UTC,
        ).isoformat(),
    )


def build_manifest(
    *,
    project_root: Path,
    final_snapshot_root: Path,
    corpus_version: str,
    records: tuple[FrozenFileRecord, ...],
) -> FreezeManifest:
    """Build the candidate-corpus freeze manifest."""

    total_size_bytes = sum(record.size_bytes for record in records)

    return FreezeManifest(
        snapshot_name=final_snapshot_root.name,
        corpus_status="candidate_only_frozen",
        corpus_version=corpus_version,
        created_at=datetime.now(UTC).isoformat(),
        project_root=str(project_root),
        snapshot_root=str(final_snapshot_root),
        preserved_paths=tuple(path.as_posix() for path in PRESERVED_PATHS),
        file_count=len(records),
        total_size_bytes=total_size_bytes,
        files=records,
        notes=(
            (
                "This snapshot preserves the broad source-derived "
                "candidate corpus before Phase 1 relevance filtering."
            ),
            ("The frozen chunks are not reviewed, active, or approved for Phase 1 retrieval."),
            ("The snapshot must not be overwritten. Create a new version for subsequent freezes."),
        ),
    )


def write_manifest(
    *,
    snapshot_root: Path,
    manifest: FreezeManifest,
) -> None:
    """Write the JSON freeze manifest."""

    manifest_path = snapshot_root / MANIFEST_NAME

    manifest_path.write_text(
        json.dumps(
            asdict(manifest),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def verify_snapshot(
    *,
    snapshot_root: Path,
    records: Iterable[FrozenFileRecord],
) -> None:
    """Verify every frozen file against the completed manifest."""

    for record in records:
        frozen_path = snapshot_root / record.relative_path

        if not frozen_path.exists():
            raise FreezeError(
                f"Frozen file is missing during final verification: {record.relative_path}"
            )

        if not frozen_path.is_file():
            raise FreezeError(f"Frozen artifact is not a regular file: {record.relative_path}")

        if frozen_path.stat().st_size != record.size_bytes:
            raise FreezeError(f"Frozen file size does not match manifest: {record.relative_path}")

        checksum = sha256_file(frozen_path)

        if checksum != record.sha256:
            raise FreezeError(
                f"Frozen file checksum does not match manifest: {record.relative_path}"
            )


def sha256_file(path: Path) -> str:
    """Calculate a file SHA-256 checksum."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while block := file.read(BUFFER_SIZE):
            digest.update(block)

    return digest.hexdigest()


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
