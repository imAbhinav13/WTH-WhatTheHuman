"""Enrich the Phase 1 candidate-corpus manifest with corpus metadata.

This script updates the candidate-corpus manifest created during
Action 4.2. It preserves the existing lifecycle classification and adds:

- corpus version;
- verified source and chunk counts;
- counts by source and domain;
- parser versions;
- chunker versions;
- original corpus creation time;
- metadata update time;
- known warnings;
- status: candidate_only.

The script does not modify source files, parsed documents, chunks,
embeddings, review records, or database state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

LOGGER = logging.getLogger("wth.phase1.enrich_candidate_corpus_manifest")

DEFAULT_PROJECT_ROOT: Final = Path()
DEFAULT_CHUNKS_ROOT: Final = Path("artifacts/phase1/chunks")
DEFAULT_PARSED_ROOT: Final = Path("artifacts/phase1/parsed")
DEFAULT_INGESTION_MANIFEST: Final = Path("artifacts/phase1/manifests/ingestion_manifest.json")
DEFAULT_ACQUISITION_MANIFEST: Final = Path("artifacts/phase1/acquisition_manifest.json")
DEFAULT_CANDIDATE_MANIFEST: Final = Path(
    "artifacts/phase1/candidate/candidate_corpus_manifest.json"
)

DEFAULT_EXPECTED_CHUNK_COUNT: Final = 7_469
DEFAULT_CORPUS_VERSION: Final = "phase1_candidate_corpus_v1"

BACKUP_SUFFIX: Final = ".pre_metadata_enrichment.json"
MANIFEST_SCHEMA_VERSION: Final = "1.1"
BUFFER_SIZE: Final = 1024 * 1024

PARSER_VERSION_KEYS: Final = (
    "parser_version",
    "parser_versions",
    "parser_name",
    "parser",
)

CHUNKER_VERSION_KEYS: Final = (
    "chunker_version",
    "chunker_versions",
    "chunker_name",
    "chunker",
)

WARNING_KEYS: Final = (
    "warning",
    "warnings",
    "parser_warning",
    "parser_warnings",
    "chunking_warning",
    "chunking_warnings",
    "known_warning",
    "known_warnings",
)


class MetadataError(RuntimeError):
    """Raised when corpus metadata cannot be verified safely."""


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=("Enrich the Phase 1 candidate-corpus manifest with verified corpus metadata.")
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
        "--parsed-root",
        type=Path,
        default=DEFAULT_PARSED_ROOT,
    )
    parser.add_argument(
        "--ingestion-manifest",
        type=Path,
        default=DEFAULT_INGESTION_MANIFEST,
    )
    parser.add_argument(
        "--acquisition-manifest",
        type=Path,
        default=DEFAULT_ACQUISITION_MANIFEST,
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
        "--replace-backup",
        action="store_true",
        help=(
            "Replace an existing pre-enrichment backup. "
            "The candidate manifest itself is updated atomically."
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
    """Enrich the candidate-corpus manifest."""

    args = parse_arguments()
    configure_logging(args.log_level)

    try:
        manifest = enrich_candidate_manifest(
            project_root=args.project_root,
            chunks_root=args.chunks_root,
            parsed_root=args.parsed_root,
            ingestion_manifest_path=args.ingestion_manifest,
            acquisition_manifest_path=args.acquisition_manifest,
            candidate_manifest_path=args.candidate_manifest,
            corpus_version=args.corpus_version,
            expected_chunk_count=args.expected_chunk_count,
            replace_backup=args.replace_backup,
        )
    except Exception:
        LOGGER.exception("Candidate-corpus metadata enrichment failed")
        raise SystemExit(1) from None

    inventory = require_mapping(
        manifest.get("inventory"),
        "inventory",
    )

    LOGGER.info("Candidate-corpus manifest enriched successfully")
    LOGGER.info(
        "Status: %s",
        manifest.get("status"),
    )
    LOGGER.info(
        "Sources: %s",
        inventory.get("source_count"),
    )
    LOGGER.info(
        "Chunks: %s",
        inventory.get("chunk_count"),
    )


def enrich_candidate_manifest(
    *,
    project_root: Path,
    chunks_root: Path,
    parsed_root: Path,
    ingestion_manifest_path: Path,
    acquisition_manifest_path: Path,
    candidate_manifest_path: Path,
    corpus_version: str,
    expected_chunk_count: int,
    replace_backup: bool,
) -> dict[str, object]:
    """Verify corpus metadata and enrich the existing manifest."""

    if expected_chunk_count < 1:
        raise MetadataError("expected_chunk_count must be at least 1")

    project_root = project_root.resolve()

    chunks_root = resolve_from_project(
        project_root,
        chunks_root,
    )
    parsed_root = resolve_from_project(
        project_root,
        parsed_root,
    )
    ingestion_manifest_path = resolve_from_project(
        project_root,
        ingestion_manifest_path,
    )
    acquisition_manifest_path = resolve_from_project(
        project_root,
        acquisition_manifest_path,
    )
    candidate_manifest_path = resolve_from_project(
        project_root,
        candidate_manifest_path,
    )

    require_directory(chunks_root)
    require_directory(parsed_root)
    require_file(ingestion_manifest_path)
    require_file(acquisition_manifest_path)
    require_file(candidate_manifest_path)

    existing_manifest = load_json_object(candidate_manifest_path)

    validate_existing_classification(
        manifest=existing_manifest,
        corpus_version=corpus_version,
        expected_chunk_count=expected_chunk_count,
    )

    chunk_inventory = inspect_chunks(chunks_root)

    verified_chunk_count = require_int(
        chunk_inventory.get("chunk_count"),
        "chunk inventory chunk_count",
    )

    if verified_chunk_count != expected_chunk_count:
        raise MetadataError(
            "Verified chunk count does not match the expected "
            f"candidate corpus size: {verified_chunk_count} != "
            f"{expected_chunk_count}"
        )

    ingestion_manifest = load_json_object(ingestion_manifest_path)
    acquisition_manifest = load_json_value(acquisition_manifest_path)

    parsed_metadata = inspect_json_artifacts(parsed_root)

    parser_versions = discover_implementation_versions(
        values=(
            ingestion_manifest,
            parsed_metadata,
        ),
        keys=PARSER_VERSION_KEYS,
    )

    chunker_versions = discover_implementation_versions(
        values=(
            ingestion_manifest,
            chunk_inventory,
        ),
        keys=CHUNKER_VERSION_KEYS,
    )

    known_warnings = collect_known_warnings(
        values=(
            ingestion_manifest,
            acquisition_manifest,
            parsed_metadata,
            chunk_inventory,
        )
    )

    corpus_created_at = determine_corpus_creation_time(
        existing_manifest=existing_manifest,
        ingestion_manifest=ingestion_manifest,
        chunks_root=chunks_root,
    )

    source_count = require_int(
        chunk_inventory.get("source_count"),
        "chunk inventory source_count",
    )

    now = datetime.now(UTC).isoformat()

    enriched_manifest = dict(existing_manifest)

    enriched_manifest.update(
        {
            "manifest_schema_version": (MANIFEST_SCHEMA_VERSION),
            "status": "candidate_only",
            "corpus_version": corpus_version,
            "corpus_created_at": corpus_created_at,
            "metadata_updated_at": now,
            "inventory": {
                "source_count": source_count,
                "chunk_count": verified_chunk_count,
                "artifact_file_count": (chunk_inventory["artifact_file_count"]),
                "total_token_count": (chunk_inventory["total_token_count"]),
                "counts_by_domain": (chunk_inventory["counts_by_domain"]),
                "counts_by_source": (chunk_inventory["counts_by_source"]),
            },
            "implementation_versions": {
                "parsers": parser_versions,
                "chunkers": chunker_versions,
            },
            "known_warnings": known_warnings,
            "metadata_provenance": {
                "chunks_root": str(chunks_root),
                "parsed_root": str(parsed_root),
                "ingestion_manifest_path": str(ingestion_manifest_path),
                "ingestion_manifest_sha256": sha256_file(ingestion_manifest_path),
                "acquisition_manifest_path": str(acquisition_manifest_path),
                "acquisition_manifest_sha256": sha256_file(acquisition_manifest_path),
                "metadata_generated_by": ("scripts.enrich_phase1_candidate_corpus_manifest"),
            },
        }
    )

    preserve_lifecycle_guards(enriched_manifest)

    backup_path = candidate_manifest_path.with_name(candidate_manifest_path.stem + BACKUP_SUFFIX)

    create_backup(
        source=candidate_manifest_path,
        destination=backup_path,
        replace=replace_backup,
    )

    atomic_write_json(
        candidate_manifest_path,
        enriched_manifest,
    )

    verify_written_manifest(
        path=candidate_manifest_path,
        expected_chunk_count=expected_chunk_count,
        expected_source_count=source_count,
        corpus_version=corpus_version,
    )

    return enriched_manifest


def inspect_chunks(
    chunks_root: Path,
) -> dict[str, object]:
    """Inspect all chunk artifacts and produce corpus inventory."""

    artifact_paths = tuple(sorted(chunks_root.glob("*.json")))

    if not artifact_paths:
        raise MetadataError(f"No chunk JSON files found in {chunks_root}")

    chunk_ids: set[str] = set()
    source_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    total_token_count = 0

    chunker_values: set[str] = set()
    warnings: Counter[str] = Counter()

    for artifact_path in artifact_paths:
        raw = load_json_value(artifact_path)

        if not isinstance(raw, list):
            raise MetadataError(f"Chunk artifact must contain a JSON list: {artifact_path}")

        for index, record in enumerate(
            raw,
            start=1,
        ):
            if not isinstance(record, dict):
                raise MetadataError(
                    f"Chunk record must be a JSON object at {artifact_path}:{index}"
                )

            chunk_id = require_non_empty_string(
                record.get("chunk_id"),
                (f"chunk_id at {artifact_path}:{index}"),
            )

            if chunk_id in chunk_ids:
                raise MetadataError(f"Duplicate chunk ID: {chunk_id}")

            chunk_ids.add(chunk_id)

            source_id = require_non_empty_string(
                record.get("source_id"),
                (f"source_id for chunk {chunk_id}"),
            )
            source_counts[source_id] += 1

            domain = normalize_domain(
                record.get("domain"),
                chunk_id=chunk_id,
            )
            domain_counts[domain] += 1

            token_count = record.get(
                "token_count",
                0,
            )

            if isinstance(token_count, bool):
                token_count = 0

            if isinstance(token_count, int):
                total_token_count += max(
                    token_count,
                    0,
                )

            collect_version_values(
                record,
                keys=CHUNKER_VERSION_KEYS,
                destination=chunker_values,
            )

            collect_warning_values(
                record,
                destination=warnings,
            )

    return {
        "root": str(chunks_root),
        "artifact_file_count": len(artifact_paths),
        "chunk_count": len(chunk_ids),
        "source_count": len(source_counts),
        "total_token_count": total_token_count,
        "counts_by_domain": dict(sorted(domain_counts.items())),
        "counts_by_source": dict(sorted(source_counts.items())),
        "chunker_versions": sorted(chunker_values),
        "warnings": [
            {
                "message": message,
                "count": count,
            }
            for message, count in sorted(warnings.items())
        ],
    }


def inspect_json_artifacts(
    root: Path,
) -> dict[str, object]:
    """Summarize metadata available in parsed JSON artifacts."""

    artifact_paths = tuple(sorted(root.rglob("*.json")))

    parser_values: set[str] = set()
    warnings: Counter[str] = Counter()

    for artifact_path in artifact_paths:
        raw = load_json_value(artifact_path)

        collect_version_values(
            raw,
            keys=PARSER_VERSION_KEYS,
            destination=parser_values,
        )

        collect_warning_values(
            raw,
            destination=warnings,
        )

    return {
        "root": str(root),
        "artifact_file_count": len(artifact_paths),
        "parser_versions": sorted(parser_values),
        "warnings": [
            {
                "message": message,
                "count": count,
            }
            for message, count in sorted(warnings.items())
        ],
    }


def discover_implementation_versions(
    *,
    values: Iterable[object],
    keys: Sequence[str],
) -> list[str]:
    """Discover unique parser or chunker implementation values."""

    discovered: set[str] = set()

    for value in values:
        collect_version_values(
            value,
            keys=keys,
            destination=discovered,
        )

    if not discovered:
        return ["not_recorded_in_existing_artifacts"]

    return sorted(discovered)


def collect_version_values(
    value: object,
    *,
    keys: Sequence[str],
    destination: set[str],
) -> None:
    """Recursively collect implementation/version metadata."""

    normalized_keys = {key.casefold() for key in keys}

    if isinstance(value, Mapping):
        for raw_key, nested_value in value.items():
            key = str(raw_key).casefold()

            if key in normalized_keys:
                add_scalar_values(
                    nested_value,
                    destination,
                )

            collect_version_values(
                nested_value,
                keys=keys,
                destination=destination,
            )

        return

    if isinstance(value, list | tuple):
        for nested_value in value:
            collect_version_values(
                nested_value,
                keys=keys,
                destination=destination,
            )


def collect_known_warnings(
    *,
    values: Iterable[object],
) -> list[dict[str, object]]:
    """Collect and aggregate warnings from current artifacts."""

    warnings: Counter[str] = Counter()

    for value in values:
        collect_warning_values(
            value,
            destination=warnings,
        )

    if not warnings:
        return [
            {
                "code": "warnings_not_structurally_recorded",
                "message": (
                    "No structured warning records were found "
                    "in the current manifests or JSON artifacts. "
                    "Previously observed OCR warnings remain a "
                    "known corpus-quality consideration."
                ),
                "count": 1,
                "severity": "informational",
            }
        ]

    return [
        {
            "code": warning_code(message),
            "message": message,
            "count": count,
            "severity": infer_warning_severity(message),
        }
        for message, count in sorted(warnings.items())
    ]


def collect_warning_values(
    value: object,
    *,
    destination: Counter[str],
) -> None:
    """Recursively collect structured warning values."""

    normalized_warning_keys = {key.casefold() for key in WARNING_KEYS}

    if isinstance(value, Mapping):
        for raw_key, nested_value in value.items():
            key = str(raw_key).casefold()

            if key in normalized_warning_keys:
                for warning in scalar_strings(nested_value):
                    destination[warning] += 1

            collect_warning_values(
                nested_value,
                destination=destination,
            )

        return

    if isinstance(value, list | tuple):
        for nested_value in value:
            collect_warning_values(
                nested_value,
                destination=destination,
            )


def scalar_strings(
    value: object,
) -> tuple[str, ...]:
    """Convert scalar or structured warning values to strings."""

    if value is None:
        return ()

    if isinstance(value, str):
        normalized = " ".join(value.split())
        return (normalized,) if normalized else ()

    if isinstance(value, bool | int | float):
        return (str(value),)

    if isinstance(value, Mapping):
        message = value.get("message")

        if isinstance(message, str):
            normalized = " ".join(message.split())

            if normalized:
                return (normalized,)

        return (
            json.dumps(
                dict(value),
                ensure_ascii=False,
                sort_keys=True,
            ),
        )

    if isinstance(value, list | tuple):
        results: list[str] = []

        for item in value:
            results.extend(scalar_strings(item))

        return tuple(results)

    return (str(value),)


def add_scalar_values(
    value: object,
    destination: set[str],
) -> None:
    """Add normalized implementation metadata to a set."""

    for item in scalar_strings(value):
        if len(item) <= 500:
            destination.add(item)


def determine_corpus_creation_time(
    *,
    existing_manifest: Mapping[str, object],
    ingestion_manifest: Mapping[str, object],
    chunks_root: Path,
) -> str:
    """Determine the best available original corpus creation time."""

    candidate_keys = (
        "corpus_created_at",
        "created_at",
        "completed_at",
        "generated_at",
        "ingested_at",
    )

    for key in candidate_keys:
        value = existing_manifest.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    for key in candidate_keys:
        value = find_first_key(
            ingestion_manifest,
            key,
        )

        if isinstance(value, str) and value.strip():
            return value.strip()

    chunk_files = tuple(chunks_root.glob("*.json"))

    if not chunk_files:
        raise MetadataError("Cannot determine corpus creation time because no chunk files exist")

    earliest_modified = min(path.stat().st_mtime for path in chunk_files)

    return datetime.fromtimestamp(
        earliest_modified,
        tz=UTC,
    ).isoformat()


def find_first_key(
    value: object,
    target_key: str,
) -> object | None:
    """Find the first matching key recursively."""

    if isinstance(value, Mapping):
        mapping = require_mapping(
            value,
            "recursive mapping",
        )

        for raw_key, nested_value in mapping.items():
            if raw_key.casefold() == target_key.casefold():
                return nested_value

        for nested_value in mapping.values():
            result = find_first_key(
                nested_value,
                target_key,
            )

            if result is not None:
                return result

    if isinstance(value, list | tuple):
        for nested_value in value:
            result = find_first_key(
                nested_value,
                target_key,
            )

            if result is not None:
                return result

    return None


def validate_existing_classification(
    *,
    manifest: Mapping[str, object],
    corpus_version: str,
    expected_chunk_count: int,
) -> None:
    """Validate the lifecycle record written in Action 4.2."""

    if manifest.get("corpus_version") != corpus_version:
        raise MetadataError(
            "Existing candidate manifest corpus version does not match the requested version"
        )

    lifecycle = require_mapping(
        manifest.get("lifecycle"),
        "lifecycle",
    )

    if lifecycle.get("corpus_classification") != "candidate_corpus":
        raise MetadataError("Existing manifest does not classify this as a candidate corpus")

    if lifecycle.get("review_status") != "not_reviewed":
        raise MetadataError("Existing manifest must classify the corpus as not reviewed")

    if lifecycle.get("phase1_activation_status") != "not_active":
        raise MetadataError("Existing manifest must classify the corpus as not active")

    if lifecycle.get("direct_activation_eligible") is not False:
        raise MetadataError("Existing manifest must prohibit direct activation")

    verified_count = manifest.get("verified_chunk_count")

    if verified_count != expected_chunk_count:
        raise MetadataError(
            f"Existing manifest verified_chunk_count does not match expectation: {verified_count!r}"
        )


def preserve_lifecycle_guards(
    manifest: dict[str, object],
) -> None:
    """Reassert non-negotiable candidate-corpus guards."""

    lifecycle = require_mapping(
        manifest.get("lifecycle"),
        "lifecycle",
    )

    lifecycle_copy = dict(lifecycle)
    lifecycle_copy.update(
        {
            "corpus_classification": ("candidate_corpus"),
            "review_status": "not_reviewed",
            "phase1_activation_status": ("not_active"),
            "production_retrieval_status": ("not_eligible"),
            "evaluation_gold_status": ("not_gold_labeled"),
            "direct_activation_eligible": False,
            "database_activation_permitted": False,
        }
    )

    manifest["lifecycle"] = lifecycle_copy
    manifest["status"] = "candidate_only"


def verify_written_manifest(
    *,
    path: Path,
    expected_chunk_count: int,
    expected_source_count: int,
    corpus_version: str,
) -> None:
    """Reload and verify the enriched manifest."""

    written = load_json_object(path)

    if written.get("status") != "candidate_only":
        raise MetadataError("Written manifest has an invalid status")

    if written.get("corpus_version") != corpus_version:
        raise MetadataError("Written manifest corpus version is invalid")

    inventory = require_mapping(
        written.get("inventory"),
        "written inventory",
    )

    if inventory.get("chunk_count") != expected_chunk_count:
        raise MetadataError("Written manifest chunk count is invalid")

    if inventory.get("source_count") != expected_source_count:
        raise MetadataError("Written manifest source count is invalid")

    lifecycle = require_mapping(
        written.get("lifecycle"),
        "written lifecycle",
    )

    if lifecycle.get("direct_activation_eligible") is not False:
        raise MetadataError("Written manifest does not block direct activation")


def create_backup(
    *,
    source: Path,
    destination: Path,
    replace: bool,
) -> None:
    """Create a backup of the Action 4.2 manifest."""

    if destination.exists():
        if not replace:
            raise MetadataError(
                "Pre-enrichment manifest backup already exists. "
                "Use --replace-backup only when intentionally "
                f"regenerating Action 4.4: {destination}"
            )

        destination.unlink()

    shutil.copy2(
        source,
        destination,
    )

    if sha256_file(source) != sha256_file(destination):
        raise MetadataError("Candidate-manifest backup checksum mismatch")


def normalize_domain(
    value: object,
    *,
    chunk_id: str,
) -> str:
    """Normalize a chunk domain value."""

    if isinstance(value, str):
        normalized = value.strip()

        if normalized:
            return normalized

    if isinstance(value, Mapping):
        nested = value.get("value")

        if isinstance(nested, str) and nested.strip():
            return nested.strip()

    raise MetadataError(f"Chunk {chunk_id} has no valid domain")


def warning_code(message: str) -> str:
    """Create a stable warning code from warning text."""

    normalized = "".join(character.lower() if character.isalnum() else "_" for character in message)

    normalized = "_".join(part for part in normalized.split("_") if part)

    return normalized[:80] or "unspecified_warning"


def infer_warning_severity(
    message: str,
) -> str:
    """Infer a conservative warning severity."""

    lowered = message.casefold()

    if any(
        term in lowered
        for term in (
            "corrupt",
            "missing",
            "failed",
            "invalid",
            "unreadable",
        )
    ):
        return "warning"

    return "informational"


def load_json_object(
    path: Path,
) -> dict[str, object]:
    """Load a required JSON object."""

    raw = load_json_value(path)

    if not isinstance(raw, dict):
        raise MetadataError(f"Expected a JSON object: {path}")

    return {str(key): value for key, value in raw.items()}


def load_json_value(
    path: Path,
) -> object:
    """Load any JSON value from a required file."""

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MetadataError(f"Invalid JSON file: {path}") from exc


def atomic_write_json(
    path: Path,
    value: Mapping[str, object],
) -> None:
    """Write JSON through an atomic replacement."""

    temporary = path.with_name(f".{path.name}.tmp")

    temporary.write_text(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    temporary.replace(path)


def sha256_file(path: Path) -> str:
    """Calculate a file SHA-256 checksum."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while block := file.read(BUFFER_SIZE):
            digest.update(block)

    return digest.hexdigest()


def resolve_from_project(
    project_root: Path,
    path: Path,
) -> Path:
    """Resolve a path relative to the project root."""

    if path.is_absolute():
        return path.resolve()

    return (project_root / path).resolve()


def require_file(path: Path) -> None:
    """Require an existing regular file."""

    if not path.is_file():
        raise MetadataError(f"Required file does not exist: {path}")


def require_directory(path: Path) -> None:
    """Require an existing directory."""

    if not path.is_dir():
        raise MetadataError(f"Required directory does not exist: {path}")


def require_mapping(
    value: object,
    description: str,
) -> dict[str, object]:
    """Require and normalize a mapping value."""

    if not isinstance(value, Mapping):
        raise MetadataError(f"{description} must be an object")

    return {str(key): nested_value for key, nested_value in value.items()}


def require_int(
    value: object,
    description: str,
) -> int:
    """Require a non-boolean integer value."""

    if isinstance(value, bool) or not isinstance(
        value,
        int,
    ):
        raise MetadataError(f"{description} must be an integer")

    return value


def require_non_empty_string(
    value: object,
    description: str,
) -> str:
    """Require a non-empty string."""

    if not isinstance(value, str):
        raise MetadataError(f"{description} must be a string")

    normalized = value.strip()

    if not normalized:
        raise MetadataError(f"{description} must not be empty")

    return normalized


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
