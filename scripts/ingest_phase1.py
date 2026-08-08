"""Orchestrate Phase 1 corpus acquisition and preprocessing.

Pipeline stages:

catalogue
    -> source acquisition
    -> checksum verification
    -> source-specific parsing
    -> domain-aware chunking
    -> 768-dimensional embeddings
    -> concept-anchor embeddings
    -> weighted concept proposals

This script creates reviewable artifacts. It does not activate chunks or
write reviewed chunk_concepts to the production database.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import IntEnum
from pathlib import Path, PurePosixPath
from typing import Any, Final
from urllib.parse import quote, urlparse
from uuid import uuid4

import httpx
import yaml  # type: ignore[import-untyped]
from pydantic import AnyHttpUrl, SecretStr

from apps.api.clients.base import EmbeddingProvider
from apps.api.clients.embedding import (
    GeminiEmbeddingProvider,
    MockEmbeddingProvider,
)
from apps.api.core.config import ProviderMode, Settings, get_settings
from apps.api.ingestion.chunkers.base import ChunkerRegistry
from apps.api.ingestion.chunkers.classical_text import (
    ClassicalTextChunker,
)
from apps.api.ingestion.chunkers.scientific import ScientificChunker
from apps.api.ingestion.parsers.base import ParserRegistry
from apps.api.ingestion.parsers.gutenberg_html import (
    GutenbergHTMLParser,
)
from apps.api.ingestion.parsers.pmc_jats import PMCJATSParser
from apps.api.ingestion.parsers.structured_text import (
    StructuredTextParser,
)
from apps.api.models.corpus import (
    AcquiredSourceArtifact,
    AcquisitionMethod,
    ChunkConceptProposal,
    ChunkDraft,
    ChunkEmbeddingRecord,
    ConceptAnchorDefinition,
    ConceptAnchorEmbeddingRecord,
    IngestionManifest,
    IngestionRunStatus,
    ParsedDocument,
    SourceCatalogueEntry,
    SourceFormat,
    SourceInclusionStatus,
)
from apps.api.services.concept_weighting import (
    ConceptWeightingService,
)
from apps.api.services.corpus_embeddings import (
    CorpusEmbeddingConfig,
    CorpusEmbeddingService,
)

LOGGER = logging.getLogger("wth.phase1.ingestion")

DEFAULT_CATALOGUE_PATH: Final = Path("docs/catalogues/phase1_sources.yaml")
DEFAULT_ANCHOR_PATH: Final = Path("data/concepts/phase1_concept_anchors.yaml")
DEFAULT_RAW_ROOT: Final = Path("data/raw")
DEFAULT_ARTIFACT_ROOT: Final = Path("artifacts/phase1")

DEFAULT_CORPUS_VERSION: Final = "phase1-three-concept-v1"
DEFAULT_EMBEDDING_TASK_TYPE: Final = "SEMANTIC_SIMILARITY"

PHASE1_EMBEDDING_DIMENSIONS: Final = 768
MAXIMUM_DOWNLOAD_BYTES: Final = 100 * 1024 * 1024

_AUTO_ACQUISITION_STATUSES: Final = frozenset(
    {
        SourceInclusionStatus.APPROVED_FOR_ACQUISITION,
    }
)

_CANDIDATE_ACQUISITION_STATUSES: Final = frozenset(
    {
        SourceInclusionStatus.CANDIDATE_PENDING_JURISDICTION_REVIEW,
        (SourceInclusionStatus.CANDIDATE_PENDING_RIGHTS_AND_TEXT_QUALITY_REVIEW),
        SourceInclusionStatus.PENDING_REVIEW,
    }
)


class PipelineExecutionError(RuntimeError):
    """Raised when the Phase 1 pipeline cannot continue."""


class SourceAcquisitionError(PipelineExecutionError):
    """Raised when a source artifact cannot be acquired safely."""


class CatalogueValidationError(PipelineExecutionError):
    """Raised when catalogue data is invalid or inconsistent."""


class AnchorValidationError(PipelineExecutionError):
    """Raised when concept anchors are not approved or valid."""


class PipelineStage(IntEnum):
    """Ordered Phase 1 ingestion stages."""

    ACQUIRE = 1
    PARSE = 2
    CHUNK = 3
    EMBED = 4
    WEIGHT = 5


_STAGE_NAMES: Final = {
    "acquire": PipelineStage.ACQUIRE,
    "parse": PipelineStage.PARSE,
    "chunk": PipelineStage.CHUNK,
    "embed": PipelineStage.EMBED,
    "weight": PipelineStage.WEIGHT,
}


@dataclass(frozen=True, slots=True)
class ResolvedDownload:
    """Resolved URL and acquisition method for one source."""

    url: str
    method: AcquisitionMethod


@dataclass(frozen=True, slots=True)
class PipelinePaths:
    """Filesystem locations used by one ingestion run."""

    catalogue: Path
    anchors: Path
    raw_root: Path
    artifact_root: Path

    acquisition_manifest: Path
    parsed_root: Path
    chunks_root: Path
    embeddings_root: Path
    concepts_root: Path
    manifests_root: Path

    chunk_embeddings: Path
    anchor_embeddings: Path
    concept_proposals: Path
    ingestion_manifest: Path

    @classmethod
    def create(
        cls,
        *,
        catalogue: Path,
        anchors: Path,
        raw_root: Path,
        artifact_root: Path,
    ) -> PipelinePaths:
        """Create all derived Phase 1 artifact paths."""

        return cls(
            catalogue=catalogue,
            anchors=anchors,
            raw_root=raw_root,
            artifact_root=artifact_root,
            acquisition_manifest=(artifact_root / "acquisition_manifest.json"),
            parsed_root=artifact_root / "parsed",
            chunks_root=artifact_root / "chunks",
            embeddings_root=artifact_root / "embeddings",
            concepts_root=artifact_root / "concepts",
            manifests_root=artifact_root / "manifests",
            chunk_embeddings=(artifact_root / "embeddings" / "chunk_embeddings.jsonl"),
            anchor_embeddings=(artifact_root / "embeddings" / "concept_anchor_embeddings.json"),
            concept_proposals=(artifact_root / "concepts" / "chunk_concept_proposals.jsonl"),
            ingestion_manifest=(artifact_root / "manifests" / "ingestion_manifest.json"),
        )

    def create_directories(self) -> None:
        """Create the artifact directories required by the pipeline."""

        for directory in (
            self.raw_root,
            self.artifact_root,
            self.parsed_root,
            self.chunks_root,
            self.embeddings_root,
            self.concepts_root,
            self.manifests_root,
        ):
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )


@dataclass(slots=True)
class PipelineState:
    """Mutable state collected across one pipeline execution."""

    sources: tuple[SourceCatalogueEntry, ...]
    artifacts: dict[str, AcquiredSourceArtifact]
    documents: dict[str, ParsedDocument]
    chunks: dict[str, tuple[ChunkDraft, ...]]

    chunk_embeddings: tuple[ChunkEmbeddingRecord, ...] = ()
    anchor_embeddings: tuple[
        ConceptAnchorEmbeddingRecord,
        ...,
    ] = ()
    proposals: tuple[ChunkConceptProposal, ...] = ()

    errors: list[str] | None = None

    def __post_init__(self) -> None:
        """Initialize the mutable error collection."""

        if self.errors is None:
            self.errors = []

    @property
    def all_chunks(self) -> tuple[ChunkDraft, ...]:
        """Return chunks in catalogue and document order."""

        return tuple(
            chunk
            for source in self.sources
            for chunk in self.chunks.get(
                source.source_id,
                (),
            )
        )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=("Acquire, parse, chunk, embed, and concept-map the WTH Phase 1 corpus.")
    )

    parser.add_argument(
        "--catalogue",
        type=Path,
        default=DEFAULT_CATALOGUE_PATH,
    )
    parser.add_argument(
        "--anchors",
        type=Path,
        default=DEFAULT_ANCHOR_PATH,
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
    )
    parser.add_argument(
        "--corpus-version",
        default=DEFAULT_CORPUS_VERSION,
    )
    parser.add_argument(
        "--through",
        choices=tuple(_STAGE_NAMES),
        default="weight",
        help=("Run through the selected pipeline stage. Default: weight."),
    )
    parser.add_argument(
        "--source-id",
        action="append",
        default=[],
        help=("Restrict execution to a source ID. May be specified more than once."),
    )
    parser.add_argument(
        "--include-candidates",
        action="store_true",
        help=(
            "Include enabled candidate sources whose rights or "
            "text-quality review is still pending. Such sources "
            "must not be activated until separately approved."
        ),
    )
    parser.add_argument(
        "--overwrite-downloads",
        action="store_true",
        help="Download sources even when a local raw file exists.",
    )
    parser.add_argument(
        "--accept-source-change",
        action="store_true",
        help=(
            "Allow a newly downloaded file to replace a catalogue "
            "checksum. Use only after reviewing the source change."
        ),
    )
    parser.add_argument(
        "--mock-embeddings",
        action="store_true",
        help=(
            "Use deterministic mock embeddings instead of Gemini. "
            "Mock artifacts must not be used for the Phase 1 experiment."
        ),
    )
    parser.add_argument(
        "--embedding-task-type",
        default=DEFAULT_EMBEDDING_TASK_TYPE,
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--download-attempts",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--download-timeout-seconds",
        type=float,
        default=60.0,
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


async def execute_pipeline(
    *,
    args: argparse.Namespace,
) -> int:
    """Execute the requested Phase 1 ingestion stages."""

    stage = _STAGE_NAMES[args.through]

    paths = PipelinePaths.create(
        catalogue=args.catalogue,
        anchors=args.anchors,
        raw_root=args.raw_root,
        artifact_root=args.artifact_root,
    )
    paths.create_directories()

    raw_catalogue, catalogue = load_catalogue(paths.catalogue)

    selected_sources = select_sources(
        catalogue=catalogue,
        requested_source_ids=tuple(args.source_id),
        include_candidates=args.include_candidates,
    )

    state = PipelineState(
        sources=selected_sources,
        artifacts={},
        documents={},
        chunks={},
    )

    started_at = datetime.now(UTC)

    await run_acquisition_stage(
        state=state,
        paths=paths,
        raw_catalogue=raw_catalogue,
        overwrite_downloads=args.overwrite_downloads,
        accept_source_change=args.accept_source_change,
        attempts=args.download_attempts,
        timeout_seconds=args.download_timeout_seconds,
    )

    if stage >= PipelineStage.PARSE:
        run_parsing_stage(
            state=state,
            paths=paths,
        )

    if stage >= PipelineStage.CHUNK:
        run_chunking_stage(
            state=state,
            paths=paths,
        )

    if stage >= PipelineStage.EMBED:
        await run_embedding_stage(
            state=state,
            paths=paths,
            settings=get_settings(),
            use_mock=args.mock_embeddings,
            embedding_task_type=args.embedding_task_type,
            batch_size=args.embedding_batch_size,
        )

    if stage >= PipelineStage.WEIGHT:
        run_weighting_stage(
            state=state,
            paths=paths,
        )

    manifest = build_ingestion_manifest(
        state=state,
        corpus_version=args.corpus_version,
        started_at=started_at,
        completed_at=datetime.now(UTC),
        include_parser_versions=(stage >= PipelineStage.PARSE),
        include_chunker_versions=(stage >= PipelineStage.CHUNK),
    )

    write_json(
        paths.ingestion_manifest,
        manifest.model_dump(
            mode="json",
        ),
    )

    print_summary(
        stage=stage,
        state=state,
        manifest=manifest,
        paths=paths,
    )

    return 1 if state.errors else 0


def load_catalogue(
    path: Path,
) -> tuple[
    list[dict[str, Any]],
    tuple[SourceCatalogueEntry, ...],
]:
    """Load and validate the Phase 1 YAML source catalogue."""

    raw = load_yaml(path)

    if not isinstance(raw, list):
        raise CatalogueValidationError("The source catalogue must be a YAML list")

    raw_records: list[dict[str, Any]] = []
    entries: list[SourceCatalogueEntry] = []

    for index, value in enumerate(raw):
        if not isinstance(value, dict):
            raise CatalogueValidationError(f"Catalogue entry {index + 1} must be a mapping")

        record = dict(value)
        raw_records.append(record)

        try:
            entries.append(SourceCatalogueEntry.model_validate(record))
        except Exception as exc:
            raise CatalogueValidationError(f"Invalid catalogue entry {index + 1}: {exc}") from exc

    if not entries:
        raise CatalogueValidationError("The source catalogue contains no entries")

    source_ids = [entry.source_id for entry in entries]

    if len(source_ids) != len(set(source_ids)):
        raise CatalogueValidationError("The source catalogue contains duplicate source IDs")

    return raw_records, tuple(entries)


def select_sources(
    *,
    catalogue: Sequence[SourceCatalogueEntry],
    requested_source_ids: tuple[str, ...],
    include_candidates: bool,
) -> tuple[SourceCatalogueEntry, ...]:
    """Select enabled and acquisition-eligible sources."""

    requested = set(requested_source_ids)
    known_source_ids = {source.source_id for source in catalogue}

    unknown = requested - known_source_ids

    if unknown:
        raise CatalogueValidationError(
            "Unknown requested source IDs: " + ", ".join(sorted(unknown))
        )

    allowed_statuses = set(_AUTO_ACQUISITION_STATUSES)

    if include_candidates:
        allowed_statuses.update(_CANDIDATE_ACQUISITION_STATUSES)

    selected: list[SourceCatalogueEntry] = []

    for source in catalogue:
        if requested and source.source_id not in requested:
            continue

        if not source.enabled:
            continue

        if source.inclusion_status not in allowed_statuses:
            if source.source_id in requested:
                raise CatalogueValidationError(
                    f"Source {source.source_id} has status "
                    f"{source.inclusion_status.value}. "
                    "Use --include-candidates only for pending "
                    "candidate sources. Restricted and rejected "
                    "sources cannot be acquired."
                )

            continue

        selected.append(source)

    if not selected:
        raise CatalogueValidationError("No enabled sources are eligible for this run")

    return tuple(selected)


async def run_acquisition_stage(
    *,
    state: PipelineState,
    paths: PipelinePaths,
    raw_catalogue: list[dict[str, Any]],
    overwrite_downloads: bool,
    accept_source_change: bool,
    attempts: int,
    timeout_seconds: float,
) -> None:
    """Acquire selected source files and populate checksums."""

    if attempts < 1:
        raise SourceAcquisitionError("download_attempts must be at least 1")

    if timeout_seconds <= 0:
        raise SourceAcquisitionError("download_timeout_seconds must be positive")

    updated_sources: list[SourceCatalogueEntry] = []
    artifacts: list[AcquiredSourceArtifact] = []

    timeout = httpx.Timeout(timeout_seconds)

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=timeout,
        headers={"User-Agent": ("WTH-WhatTheHuman/0.1 (Phase 1 corpus research ingestion)")},
    ) as client:
        for source in state.sources:
            LOGGER.info(
                "Acquiring source %s",
                source.source_id,
            )

            try:
                updated_source, artifact = await acquire_source(
                    client=client,
                    source=source,
                    raw_root=paths.raw_root,
                    overwrite_downloads=overwrite_downloads,
                    accept_source_change=accept_source_change,
                    attempts=attempts,
                )
            except Exception as exc:
                record_error(
                    state,
                    stage="acquire",
                    source_id=source.source_id,
                    exc=exc,
                )
                continue

            updated_sources.append(updated_source)
            artifacts.append(artifact)
            state.artifacts[source.source_id] = artifact

    if not artifacts:
        raise SourceAcquisitionError("No source artifacts were acquired successfully")

    updated_by_id = {source.source_id: source for source in updated_sources}

    state.sources = tuple(
        updated_by_id.get(
            source.source_id,
            source,
        )
        for source in state.sources
        if source.source_id in state.artifacts
    )

    update_catalogue_checksums(
        path=paths.catalogue,
        raw_records=raw_catalogue,
        updated_sources=updated_sources,
    )

    write_json(
        paths.acquisition_manifest,
        [
            artifact.model_dump(
                mode="json",
            )
            for artifact in artifacts
        ],
    )


async def acquire_source(
    *,
    client: httpx.AsyncClient,
    source: SourceCatalogueEntry,
    raw_root: Path,
    overwrite_downloads: bool,
    accept_source_change: bool,
    attempts: int,
) -> tuple[
    SourceCatalogueEntry,
    AcquiredSourceArtifact,
]:
    """Acquire or reuse one source artifact."""

    extension = extension_for_format(source.format)
    target = raw_root / source.domain.value / f"{source.source_id}{extension}"
    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    resolved = await resolve_download(
        client=client,
        source=source,
        attempts=attempts,
    )

    media_type = media_type_for_format(source.format)

    if target.exists() and not overwrite_downloads:
        content = target.read_bytes()
        acquisition_method = AcquisitionMethod.MANUAL_DOWNLOAD
    else:
        content, response_media_type = await download_bytes(
            client=client,
            url=resolved.url,
            attempts=attempts,
        )
        acquisition_method = resolved.method

        if response_media_type:
            media_type = response_media_type

        validate_downloaded_content(
            source=source,
            content=content,
        )

        atomic_write_bytes(
            target,
            content,
        )

    if not content:
        raise SourceAcquisitionError(f"Downloaded source {source.source_id} is empty")

    checksum = hashlib.sha256(content).hexdigest()

    if source.checksum is not None and source.checksum != checksum and not accept_source_change:
        raise SourceAcquisitionError(
            f"Checksum change detected for {source.source_id}. "
            f"Catalogue={source.checksum}, downloaded={checksum}. "
            "Review the upstream change and rerun with "
            "--accept-source-change only when intentional."
        )

    updated_source = source.model_copy(
        update={
            "checksum": checksum,
            "accessed_at": datetime.now(UTC).date(),
        }
    )

    artifact = AcquiredSourceArtifact(
        source_id=source.source_id,
        source_url=AnyHttpUrl(resolved.url),
        local_path=target,
        media_type=media_type,
        file_size_bytes=len(content),
        checksum=checksum,
        acquisition_method=acquisition_method,
    )

    return updated_source, artifact


async def resolve_download(
    *,
    client: httpx.AsyncClient,
    source: SourceCatalogueEntry,
    attempts: int,
) -> ResolvedDownload:
    """Resolve direct URLs and Internet Archive item directories."""

    download_url = str(source.download_url)
    identifier = internet_archive_identifier(download_url)

    if identifier is not None and is_archive_directory_url(download_url):
        resolved_url = await resolve_archive_ocr_file(
            client=client,
            identifier=identifier,
            attempts=attempts,
        )

        return ResolvedDownload(
            url=resolved_url,
            method=AcquisitionMethod.REPOSITORY_EXPORT,
        )

    return ResolvedDownload(
        url=download_url,
        method=AcquisitionMethod.DIRECT_HTTP,
    )


async def resolve_archive_ocr_file(
    *,
    client: httpx.AsyncClient,
    identifier: str,
    attempts: int,
) -> str:
    """Resolve a text export from Internet Archive item metadata."""

    metadata_url = f"https://archive.org/metadata/{quote(identifier, safe='')}"

    content, _ = await download_bytes(
        client=client,
        url=metadata_url,
        attempts=attempts,
    )

    try:
        metadata = json.loads(content)
    except json.JSONDecodeError as exc:
        raise SourceAcquisitionError(
            f"Internet Archive metadata was not valid JSON for item {identifier}"
        ) from exc

    files = metadata.get("files")

    if not isinstance(files, list):
        raise SourceAcquisitionError(
            f"Internet Archive metadata contains no file list for item {identifier}"
        )

    candidates: list[tuple[int, str]] = []

    for file_record in files:
        if not isinstance(file_record, dict):
            continue

        name = file_record.get("name")

        if not isinstance(name, str):
            continue

        score = archive_text_file_score(name)

        if score > 0:
            candidates.append(
                (
                    score,
                    name,
                )
            )

    if not candidates:
        raise SourceAcquisitionError(
            f"No suitable OCR text export was found for Internet Archive item {identifier}"
        )

    candidates.sort(
        key=lambda candidate: (
            -candidate[0],
            len(candidate[1]),
            candidate[1],
        )
    )

    filename = candidates[0][1]

    return f"https://archive.org/download/{quote(identifier, safe='')}/{quote(filename, safe='/')}"


def internet_archive_identifier(
    url: str,
) -> str | None:
    """Extract an Internet Archive item identifier from a URL."""

    parsed = urlparse(url)

    if parsed.netloc.casefold() not in {
        "archive.org",
        "www.archive.org",
    }:
        return None

    parts = PurePosixPath(parsed.path).parts

    for marker in (
        "details",
        "download",
    ):
        try:
            marker_index = parts.index(marker)
        except ValueError:
            continue

        identifier_index = marker_index + 1

        if identifier_index < len(parts):
            return parts[identifier_index]

    return None


def is_archive_directory_url(url: str) -> bool:
    """Return whether an Archive download URL names only an item."""

    parsed = urlparse(url)
    parts = PurePosixPath(parsed.path).parts

    try:
        download_index = parts.index("download")
    except ValueError:
        return False

    return len(parts) == download_index + 2


def archive_text_file_score(
    filename: str,
) -> int:
    """Rank Internet Archive files suitable for text parsing."""

    normalized = filename.casefold()

    excluded_tokens = (
        "_meta.",
        "_files.xml",
        "_scandata.xml",
        "_reviews.xml",
        "_djvu.xml",
        "__ia_thumb",
    )

    if any(token in normalized for token in excluded_tokens):
        return 0

    if normalized.endswith("_djvu.txt"):
        return 100

    if normalized.endswith("_text.pdf"):
        return 0

    if normalized.endswith(".txt"):
        return 60

    return 0


async def download_bytes(
    *,
    client: httpx.AsyncClient,
    url: str,
    attempts: int,
) -> tuple[bytes, str | None]:
    """Download bytes with bounded exponential retry."""

    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            response = await client.get(url)
            response.raise_for_status()
        except Exception as exc:
            last_error = exc

            if attempt >= attempts:
                break

            await asyncio.sleep(min(2 ** (attempt - 1), 8))
        else:
            # Executes ONLY if response.raise_for_status() succeeded without raising
            content_length = response.headers.get("content-length")
            declared_size: int | None = None

            if content_length is not None:
                try:
                    declared_size = int(content_length)
                except ValueError:
                    declared_size = None

            content = response.content
            validate_download_size(actual_size=len(content), declared_size=declared_size)

            media_type = response.headers.get("content-type")
            if media_type is not None:
                media_type = media_type.split(";", maxsplit=1)[0].strip()

            return content, media_type

    # If all attempts fail, raise or handle last_error
    if last_error:
        raise last_error
    raise RuntimeError(f"Failed to download from {url}")


def validate_downloaded_content(
    *,
    source: SourceCatalogueEntry,
    content: bytes,
) -> None:
    """Reject obvious directory pages and format mismatches."""

    sample = content[:65_536].lower()

    if (
        source.format is SourceFormat.JATS_XML
        and b"<article" not in sample
        and b":article" not in sample
    ):
        raise SourceAcquisitionError(f"{source.source_id} does not appear to be JATS XML")

    if (
        source.format is SourceFormat.GUTENBERG_HTML
        and b"<html" not in sample
        and b"<body" not in sample
    ):
        raise SourceAcquisitionError(f"{source.source_id} does not appear to be HTML")

    if source.format in {
        SourceFormat.STRUCTURED_TEXT,
        SourceFormat.OCR_TEXT,
        SourceFormat.SCANNED_BOOK_WITH_OCR,
    } and (b"<html" in sample or b"<!doctype html" in sample):
        raise SourceAcquisitionError(
            f"{source.source_id} returned HTML instead of an OCR or structured-text file"
        )


def update_catalogue_checksums(
    *,
    path: Path,
    raw_records: list[dict[str, Any]],
    updated_sources: Sequence[SourceCatalogueEntry],
) -> None:
    """Write acquired checksums back into the source catalogue."""

    updated_by_id = {source.source_id: source for source in updated_sources}

    for record in raw_records:
        source_id = record.get("source_id")

        if not isinstance(source_id, str):
            continue

        updated = updated_by_id.get(source_id)

        if updated is None:
            continue

        record["checksum"] = updated.checksum
        record["accessed_at"] = updated.accessed_at.isoformat()

    serialized = yaml.safe_dump(
        raw_records,
        allow_unicode=True,
        sort_keys=False,
        width=100,
    )

    atomic_write_text(
        path,
        serialized,
    )


def run_parsing_stage(
    *,
    state: PipelineState,
    paths: PipelinePaths,
) -> None:
    """Parse acquired artifacts through source-format routing."""

    parsers = (
        PMCJATSParser(),
        GutenbergHTMLParser(),
        StructuredTextParser(),
    )

    registry = ParserRegistry(parsers)

    for source in state.sources:
        artifact = state.artifacts.get(source.source_id)

        if artifact is None:
            continue

        LOGGER.info(
            "Parsing source %s",
            source.source_id,
        )

        try:
            document = registry.parse(
                source=source,
                artifact=artifact,
            )
        except Exception as exc:
            record_error(
                state,
                stage="parse",
                source_id=source.source_id,
                exc=exc,
            )
            continue

        state.documents[source.source_id] = document

        atomic_write_text(
            paths.parsed_root / f"{source.source_id}.json",
            document.model_dump_json(
                indent=2,
            ),
        )

    if not state.documents:
        raise PipelineExecutionError("Parsing produced no valid documents")


def run_chunking_stage(
    *,
    state: PipelineState,
    paths: PipelinePaths,
) -> None:
    """Create draft chunks through domain-aware routing."""

    chunkers = (
        ScientificChunker(),
        ClassicalTextChunker(),
    )

    registry = ChunkerRegistry(chunkers)

    for source in state.sources:
        document = state.documents.get(source.source_id)

        if document is None:
            continue

        LOGGER.info(
            "Chunking source %s",
            source.source_id,
        )

        try:
            chunks = registry.chunk(
                source=source,
                document=document,
            )
        except Exception as exc:
            record_error(
                state,
                stage="chunk",
                source_id=source.source_id,
                exc=exc,
            )
            continue

        state.chunks[source.source_id] = chunks

        write_json(
            paths.chunks_root / f"{source.source_id}.json",
            [
                chunk.model_dump(
                    mode="json",
                )
                for chunk in chunks
            ],
        )

    if not state.all_chunks:
        raise PipelineExecutionError("Chunking produced no valid draft chunks")


async def run_embedding_stage(
    *,
    state: PipelineState,
    paths: PipelinePaths,
    settings: Settings,
    use_mock: bool,
    embedding_task_type: str,
    batch_size: int,
) -> None:
    """Generate or reuse chunk and concept-anchor embeddings."""

    chunks = state.all_chunks

    if not chunks:
        raise PipelineExecutionError("Chunk embeddings require draft chunks")

    anchors = load_approved_anchors(paths.anchors)

    provider = create_embedding_provider(
        settings=settings,
        force_mock=use_mock,
    )

    config = CorpusEmbeddingConfig(
        model=settings.embedding_model,
        dimensions=PHASE1_EMBEDDING_DIMENSIONS,
        task_type=embedding_task_type,
        batch_size=batch_size,
        maximum_attempts=max(
            1,
            settings.embedding_max_retries + 1,
        ),
    )

    service = CorpusEmbeddingService(
        provider=provider,
        config=config,
    )

    try:
        existing_chunks = load_chunk_embedding_records(paths.chunk_embeddings)
        existing_anchors = load_anchor_embedding_records(paths.anchor_embeddings)

        chunk_result = await service.embed_chunks(
            chunks=chunks,
            existing_records=existing_chunks,
        )

        anchor_result = await service.embed_concept_anchors(
            anchors=anchors,
            existing_records=existing_anchors,
        )

        state.chunk_embeddings = chunk_result.records
        state.anchor_embeddings = anchor_result.records

        write_json_lines(
            paths.chunk_embeddings,
            (
                record.model_dump(
                    mode="json",
                )
                for record in chunk_result.records
            ),
        )

        write_json(
            paths.anchor_embeddings,
            [
                record.model_dump(
                    mode="json",
                )
                for record in anchor_result.records
            ],
        )

        LOGGER.info(
            "Chunk embeddings: generated=%d reused=%d",
            chunk_result.generated_count,
            chunk_result.reused_count,
        )
        LOGGER.info(
            "Anchor embeddings: generated=%d reused=%d",
            anchor_result.generated_count,
            anchor_result.reused_count,
        )

    finally:
        await provider.close()


def load_approved_anchors(
    path: Path,
) -> tuple[ConceptAnchorDefinition, ...]:
    """Load anchors and reject draft or pending definitions."""

    raw = load_yaml(path)

    if not isinstance(raw, list):
        raise AnchorValidationError("Concept anchors must be a YAML list")

    anchors: list[ConceptAnchorDefinition] = []

    for index, value in enumerate(raw):
        try:
            anchor = ConceptAnchorDefinition.model_validate(value)
        except Exception as exc:
            raise AnchorValidationError(f"Invalid concept anchor {index + 1}: {exc}") from exc

        if anchor.reviewed_by.casefold() == "pending_human_review":
            raise AnchorValidationError(
                "Concept anchor "
                f"{anchor.concept_slug.value} still has "
                "reviewed_by=pending_human_review"
            )

        if anchor.anchor_version.casefold().endswith("-draft"):
            raise AnchorValidationError(
                "Concept anchor "
                f"{anchor.concept_slug.value} still uses "
                f"draft version {anchor.anchor_version}"
            )

        anchors.append(anchor)

    if len(anchors) != 3:
        raise AnchorValidationError("Phase 1 requires exactly three approved anchors")

    return tuple(anchors)


def create_embedding_provider(
    *,
    settings: Settings,
    force_mock: bool,
) -> EmbeddingProvider:
    """Create the configured Phase 1 embedding provider."""

    if settings.embedding_dimension != 768:
        raise PipelineExecutionError("EMBEDDING_DIMENSION must equal 768 for Phase 1")

    if force_mock or settings.provider_mode is ProviderMode.MOCK:
        LOGGER.warning(
            "Using mock embeddings. These vectors must not be "
            "used in the Phase 1 baseline experiment."
        )

        return MockEmbeddingProvider(
            model=settings.embedding_model,
            dimensions=768,
        )

    return GeminiEmbeddingProvider(
        api_key=secret_value(
            settings.google_api_key,
            field_name="GOOGLE_API_KEY",
        ),
        model=settings.embedding_model,
        dimensions=768,
    )


def run_weighting_stage(
    *,
    state: PipelineState,
    paths: PipelinePaths,
) -> None:
    """Generate plural weighted concept proposals."""

    if not state.chunk_embeddings:
        state.chunk_embeddings = load_chunk_embedding_records(paths.chunk_embeddings)

    if not state.anchor_embeddings:
        state.anchor_embeddings = load_anchor_embedding_records(paths.anchor_embeddings)

    if not state.chunk_embeddings:
        raise PipelineExecutionError("Concept weighting requires chunk embeddings")

    if not state.anchor_embeddings:
        raise PipelineExecutionError("Concept weighting requires anchor embeddings")

    service = ConceptWeightingService()

    result = service.propose_weights(
        chunk_embeddings=state.chunk_embeddings,
        anchor_embeddings=state.anchor_embeddings,
    )

    state.proposals = result.proposals

    write_json_lines(
        paths.concept_proposals,
        (
            proposal.model_dump(
                mode="json",
            )
            for proposal in result.proposals
        ),
    )

    LOGGER.info(
        "Created %d concept proposals across %d chunks",
        len(result.proposals),
        result.chunk_count,
    )


def load_chunk_embedding_records(
    path: Path,
) -> tuple[ChunkEmbeddingRecord, ...]:
    """Load existing JSONL chunk-embedding records."""

    if not path.exists():
        return ()

    records: list[ChunkEmbeddingRecord] = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            records.append(ChunkEmbeddingRecord.model_validate_json(line))
        except Exception as exc:
            raise PipelineExecutionError(
                f"Invalid chunk embedding record at {path}:{line_number}: {exc}"
            ) from exc

    return tuple(records)


def load_anchor_embedding_records(
    path: Path,
) -> tuple[
    ConceptAnchorEmbeddingRecord,
    ...,
]:
    """Load existing concept-anchor embedding records."""

    if not path.exists():
        return ()

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PipelineExecutionError(f"Invalid anchor embedding JSON: {path}") from exc

    if not isinstance(raw, list):
        raise PipelineExecutionError("Anchor embedding artifact must be a JSON list")

    return tuple(ConceptAnchorEmbeddingRecord.model_validate(value) for value in raw)


def build_ingestion_manifest(
    *,
    state: PipelineState,
    corpus_version: str,
    started_at: datetime,
    completed_at: datetime,
    include_parser_versions: bool,
    include_chunker_versions: bool,
) -> IngestionManifest:
    """Create the typed audit manifest for the run."""

    errors = tuple(state.errors or ())

    return IngestionManifest(
        run_id=uuid4(),
        corpus_version=corpus_version,
        status=(
            IngestionRunStatus.COMPLETED if not errors else IngestionRunStatus.COMPLETED_WITH_ERRORS
        ),
        source_ids=tuple(source.source_id for source in state.sources),
        parser_versions=(
            {
                "pmc_jats": "1.0.0",
                "gutenberg_html": "1.0.0",
                "structured_text": "1.0.0",
            }
            if include_parser_versions
            else {}
        ),
        chunker_versions=(
            {
                "scientific": "1.0.0",
                "classical_text": "1.0.0",
            }
            if include_chunker_versions
            else {}
        ),
        acquired_source_count=len(state.artifacts),
        parsed_document_count=len(state.documents),
        draft_chunk_count=len(state.all_chunks),
        embedded_chunk_count=len(state.chunk_embeddings),
        reviewed_chunk_count=0,
        active_chunk_count=0,
        failed_item_count=len(errors),
        started_at=started_at,
        completed_at=completed_at,
        errors=errors,
    )


def record_error(
    state: PipelineState,
    *,
    stage: str,
    source_id: str,
    exc: Exception,
) -> None:
    """Record and log one source-specific pipeline error."""

    message = f"{stage}:{source_id}:{type(exc).__name__}: {exc}"

    if state.errors is None:
        state.errors = []

    state.errors.append(message)
    LOGGER.error(message)


def load_yaml(path: Path) -> object:
    """Load one UTF-8 YAML document."""

    if not path.exists():
        raise PipelineExecutionError(f"Required YAML file does not exist: {path}")

    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PipelineExecutionError(f"Invalid YAML file {path}: {exc}") from exc


def write_json(
    path: Path,
    value: object,
) -> None:
    """Write formatted JSON atomically."""

    atomic_write_text(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )


def write_json_lines(
    path: Path,
    values: Iterable[object],
) -> None:
    """Write newline-delimited JSON atomically."""

    serialized = "\n".join(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for value in values
    )

    if serialized:
        serialized += "\n"

    atomic_write_text(
        path,
        serialized,
    )


def atomic_write_text(
    path: Path,
    text: str,
) -> None:
    """Write UTF-8 text using an atomic replacement."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(f".{path.name}.tmp")

    temporary.write_text(
        text,
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def atomic_write_bytes(
    path: Path,
    content: bytes,
) -> None:
    """Write source bytes using an atomic replacement."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(f".{path.name}.tmp")

    temporary.write_bytes(content)
    temporary.replace(path)


def extension_for_format(
    source_format: SourceFormat,
) -> str:
    """Return the local extension for a source format."""

    return {
        SourceFormat.JATS_XML: ".xml",
        SourceFormat.GUTENBERG_HTML: ".html",
        SourceFormat.STRUCTURED_TEXT: ".txt",
        SourceFormat.OCR_TEXT: ".txt",
        SourceFormat.SCANNED_BOOK_WITH_OCR: ".txt",
    }[source_format]


def media_type_for_format(
    source_format: SourceFormat,
) -> str:
    """Return the expected media type for a source format."""

    return {
        SourceFormat.JATS_XML: "application/xml",
        SourceFormat.GUTENBERG_HTML: "text/html",
        SourceFormat.STRUCTURED_TEXT: "text/plain",
        SourceFormat.OCR_TEXT: "text/plain",
        SourceFormat.SCANNED_BOOK_WITH_OCR: "text/plain",
    }[source_format]


def secret_value(
    value: SecretStr | None,
    *,
    field_name: str,
) -> str:
    """Return a non-empty configured secret."""

    if value is None:
        raise PipelineExecutionError(f"{field_name} is required for live embeddings")

    secret = value.get_secret_value().strip()

    if not secret:
        raise PipelineExecutionError(f"{field_name} must not be empty")

    return secret


def print_summary(
    *,
    stage: PipelineStage,
    state: PipelineState,
    manifest: IngestionManifest,
    paths: PipelinePaths,
) -> None:
    """Print a concise execution summary."""

    print()
    print("Phase 1 ingestion summary")
    print("-------------------------")
    print(f"Completed through: {stage.name.lower()}")
    print(f"Sources acquired: {manifest.acquired_source_count}")
    print(f"Documents parsed: {manifest.parsed_document_count}")
    print(f"Draft chunks: {manifest.draft_chunk_count}")
    print(f"Chunk embeddings: {manifest.embedded_chunk_count}")
    print(f"Anchor embeddings: {len(state.anchor_embeddings)}")
    print(f"Concept proposals: {len(state.proposals)}")
    print(f"Errors: {manifest.failed_item_count}")
    print(f"Manifest: {paths.ingestion_manifest}")

    if state.errors:
        print()
        print("Errors:")
        for error in state.errors:
            print(f"- {error}")


def configure_logging(
    level: str,
) -> None:
    """Configure console logging."""

    logging.basicConfig(
        level=getattr(
            logging,
            level,
        ),
        format=("%(asctime)s %(levelname)s %(name)s %(message)s"),
    )


def validate_download_size(
    *,
    actual_size: int,
    declared_size: int | None = None,
) -> None:
    """Reject source artifacts exceeding the download limit."""

    if declared_size is not None and declared_size > MAXIMUM_DOWNLOAD_BYTES:
        raise SourceAcquisitionError(
            f"Source exceeds the maximum declared download size: {declared_size} bytes"
        )

    if actual_size > MAXIMUM_DOWNLOAD_BYTES:
        raise SourceAcquisitionError(
            f"Source exceeds the maximum download size: {actual_size} bytes"
        )


def main() -> None:
    """Run the Phase 1 ingestion command."""

    args = parse_arguments()
    configure_logging(args.log_level)

    try:
        exit_code = asyncio.run(
            execute_pipeline(
                args=args,
            )
        )
    except KeyboardInterrupt:
        LOGGER.info("Phase 1 ingestion was interrupted")
        raise SystemExit(130) from None
    except Exception as exc:
        LOGGER.exception("Phase 1 ingestion failed")
        raise SystemExit(1) from exc

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
