"""Base contracts and shared utilities for corpus chunkers."""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final, Protocol, final

from apps.api.models.corpus import (
    ChunkCitation,
    ChunkDraft,
    DocumentSection,
    DocumentUnit,
    DocumentUnitType,
    ParsedDocument,
    SourceCatalogueEntry,
)
from apps.api.models.enums import Domain, ReviewStatus, SourceType

_TOKEN_PATTERN: Final = re.compile(
    r"\w+|[^\w\s]",
    flags=re.UNICODE,
)
_WHITESPACE_PATTERN: Final = re.compile(r"[ \t\f\v]+")
_MULTIPLE_NEWLINES_PATTERN: Final = re.compile(r"\n{3,}")


class CorpusChunkerError(RuntimeError):
    """Base exception for corpus chunking failures."""


class ChunkerConfigurationError(CorpusChunkerError):
    """Raised when a chunker declares invalid configuration."""


class UnsupportedDocumentError(CorpusChunkerError):
    """Raised when a chunker does not support a source or document."""


class ChunkConstructionError(CorpusChunkerError):
    """Raised when a valid chunk cannot be constructed."""


class ChunkValidationError(CorpusChunkerError):
    """Raised when generated chunks fail provenance validation."""


class TokenCounter(Protocol):
    """Contract used to count tokens for chunk-size decisions."""

    def count(self, text: str) -> int:
        """Return the token count for text."""


class HeuristicTokenCounter:
    """Dependency-free token counter for deterministic chunk sizing.

    This counter approximates model tokens by counting words, numbers,
    and punctuation marks. It is intended for chunk-boundary decisions,
    not provider billing or exact model-token accounting.
    """

    def count(self, text: str) -> int:
        """Return a deterministic approximate token count."""

        return len(_TOKEN_PATTERN.findall(text))


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Token-size limits shared by a chunker implementation."""

    minimum_tokens: int
    target_tokens: int
    maximum_tokens: int
    overlap_tokens: int = 0

    def __post_init__(self) -> None:
        """Validate token limits and overlap."""

        if self.minimum_tokens < 1:
            raise ChunkerConfigurationError("minimum_tokens must be at least 1")

        if self.target_tokens < self.minimum_tokens:
            raise ChunkerConfigurationError("target_tokens cannot be below minimum_tokens")

        if self.maximum_tokens < self.target_tokens:
            raise ChunkerConfigurationError("maximum_tokens cannot be below target_tokens")

        if self.overlap_tokens < 0:
            raise ChunkerConfigurationError("overlap_tokens cannot be negative")

        if self.overlap_tokens >= self.target_tokens:
            raise ChunkerConfigurationError("overlap_tokens must be below target_tokens")


class SourceChunker(ABC):
    """Template contract implemented by domain-aware chunkers.

    The public ``chunk`` method validates source provenance, delegates
    grouping to ``_chunk_document``, and then validates every produced
    chunk against the source and parsed document.
    """

    def __init__(
        self,
        *,
        token_counter: TokenCounter | None = None,
    ) -> None:
        """Initialize the chunker with a deterministic token counter."""

        self._token_counter = (
            token_counter if token_counter is not None else HeuristicTokenCounter()
        )

    @property
    @abstractmethod
    def chunker_name(self) -> str:
        """Return the stable chunker implementation name."""

    @property
    @abstractmethod
    def chunker_version(self) -> str:
        """Return the version used for reproducibility."""

    @property
    @abstractmethod
    def config(self) -> ChunkingConfig:
        """Return token-size configuration for this chunker."""

    @property
    @abstractmethod
    def supported_domains(self) -> frozenset[Domain]:
        """Return domains accepted by this chunker."""

    @property
    @abstractmethod
    def supported_source_types(self) -> frozenset[SourceType]:
        """Return source types accepted by this chunker."""

    @final
    def chunk(
        self,
        *,
        source: SourceCatalogueEntry,
        document: ParsedDocument,
    ) -> tuple[ChunkDraft, ...]:
        """Validate and chunk one normalized parsed document."""

        self._validate_chunker_metadata()
        self._validate_input(
            source=source,
            document=document,
        )

        chunks = self._chunk_document(
            source=source,
            document=document,
        )

        self._validate_output(
            source=source,
            document=document,
            chunks=chunks,
        )

        return chunks

    @abstractmethod
    def _chunk_document(
        self,
        *,
        source: SourceCatalogueEntry,
        document: ParsedDocument,
    ) -> tuple[ChunkDraft, ...]:
        """Create draft chunks from a validated parsed document."""

    def supports(
        self,
        *,
        domain: Domain,
        source_type: SourceType,
    ) -> bool:
        """Return whether the chunker supports the source."""

        return domain in self.supported_domains and source_type in self.supported_source_types

    def count_tokens(self, text: str) -> int:
        """Return the configured token count for text."""

        return self._token_counter.count(text)

    def normalize_chunk_text(self, text: str) -> str:
        """Normalize layout without altering substantive wording."""

        normalized_lines = [
            _WHITESPACE_PATTERN.sub(
                " ",
                line,
            ).strip()
            for line in text.replace(
                "\r\n",
                "\n",
            )
            .replace(
                "\r",
                "\n",
            )
            .splitlines()
        ]

        normalized = "\n".join(normalized_lines)
        normalized = _MULTIPLE_NEWLINES_PATTERN.sub(
            "\n\n",
            normalized,
        )

        return normalized.strip()

    def merge_unit_text(
        self,
        units: Sequence[DocumentUnit],
        *,
        separator: str = "\n\n",
    ) -> str:
        """Join ordered document units into normalized chunk text."""

        if not units:
            raise ChunkConstructionError("Cannot merge an empty sequence of document units")

        return self.normalize_chunk_text(separator.join(unit.text for unit in units))

    def build_chunk(
        self,
        *,
        source: SourceCatalogueEntry,
        document: ParsedDocument,
        section: DocumentSection,
        units: Sequence[DocumentUnit],
        text: str | None = None,
        unit_type: DocumentUnitType | None = None,
        citation_text: str | None = None,
        structural_locator: str | None = None,
    ) -> ChunkDraft:
        """Build one deterministic draft chunk from source units."""

        if not units:
            raise ChunkConstructionError("A chunk must contain at least one document unit")

        section_unit_ids = {unit.unit_id for unit in section.units}

        supplied_unit_ids = [unit.unit_id for unit in units]

        if len(supplied_unit_ids) != len(set(supplied_unit_ids)):
            raise ChunkConstructionError("A chunk cannot contain duplicate source units")

        if not set(supplied_unit_ids).issubset(section_unit_ids):
            raise ChunkConstructionError("All chunk units must belong to the supplied section")

        chunk_text = self.normalize_chunk_text(
            text if text is not None else self.merge_unit_text(units)
        )

        if not chunk_text:
            raise ChunkConstructionError("Chunk text must not be empty")

        token_count = self.count_tokens(chunk_text)

        if token_count < self.config.minimum_tokens:
            raise ChunkConstructionError(
                "Chunk is below the configured minimum: "
                f"{token_count} < {self.config.minimum_tokens}"
            )

        if token_count > self.config.maximum_tokens:
            raise ChunkConstructionError(
                "Chunk exceeds the configured maximum: "
                f"{token_count} > {self.config.maximum_tokens}"
            )

        resolved_unit_type = unit_type if unit_type is not None else self.resolve_unit_type(units)

        locator = (
            structural_locator
            if structural_locator is not None
            else self.combine_locators(
                section=section,
                units=units,
            )
        )

        text_checksum = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()

        chunk_id = self.create_chunk_id(
            source=source,
            document=document,
            section=section,
            units=units,
            text_checksum=text_checksum,
        )

        citation = ChunkCitation(
            display_text=(
                citation_text
                if citation_text is not None
                else self.create_citation_text(
                    source=source,
                    structural_locator=locator,
                )
            ),
            structural_locator=locator,
            canonical_url=source.canonical_url,
            external_identifier=None,
        )

        return ChunkDraft(
            chunk_id=chunk_id,
            source_id=source.source_id,
            source_checksum=document.source_checksum,
            domain=source.domain,
            source_type=source.source_type,
            unit_type=resolved_unit_type,
            section_id=section.section_id,
            source_unit_ids=tuple(supplied_unit_ids),
            text=chunk_text,
            text_checksum=text_checksum,
            token_count=token_count,
            citation=citation,
            parser_name=document.parser_name,
            parser_version=document.parser_version,
            chunker_name=self.chunker_name,
            chunker_version=self.chunker_version,
            review_status=ReviewStatus.DRAFT,
        )

    def resolve_unit_type(
        self,
        units: Sequence[DocumentUnit],
    ) -> DocumentUnitType:
        """Return the common type for units in one chunk.

        Mixing root text, commentary, translator notes, or other unit
        categories must be explicitly handled by a specialized chunker.
        """

        unit_types = {unit.unit_type for unit in units}

        if len(unit_types) != 1:
            values = ", ".join(sorted(unit_type.value for unit_type in unit_types))

            raise ChunkConstructionError(f"Chunk units have mixed semantic types: {values}")

        return next(iter(unit_types))

    def combine_locators(
        self,
        *,
        section: DocumentSection,
        units: Sequence[DocumentUnit],
    ) -> str:
        """Create a stable locator covering one or more source units."""

        if not units:
            raise ChunkConstructionError("Cannot create a locator without source units")

        if len(units) == 1:
            return units[0].structural_locator

        return (
            f"{section.structural_locator}; "
            f"{units[0].structural_locator}.."
            f"{units[-1].structural_locator}"
        )

    def create_citation_text(
        self,
        *,
        source: SourceCatalogueEntry,
        structural_locator: str,
    ) -> str:
        """Create a generic human-readable citation."""

        contributors: list[str] = [
            source.author,
        ]

        if source.translator is not None:
            contributors.append(f"trans. {source.translator}")

        if source.editor is not None:
            contributors.append(f"ed. {source.editor}")

        contributor_text = "; ".join(contributors)

        return (
            f"{contributor_text} "
            f"({source.publication_year}), "
            f"“{source.title},” "
            f"{structural_locator}."
        )

    def create_chunk_id(
        self,
        *,
        source: SourceCatalogueEntry,
        document: ParsedDocument,
        section: DocumentSection,
        units: Sequence[DocumentUnit],
        text_checksum: str,
    ) -> str:
        """Create a deterministic chunk ID from frozen provenance."""

        material = "\n".join(
            (
                source.source_id,
                document.source_checksum,
                section.section_id,
                *(unit.unit_id for unit in units),
                text_checksum,
                self.chunker_name,
                self.chunker_version,
            )
        )

        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]

        return f"{source.source_id}:chunk:{digest}"

    def _validate_chunker_metadata(self) -> None:
        """Validate implementation identity and configuration."""

        if not self.chunker_name.strip():
            raise ChunkerConfigurationError("Chunker name must not be empty")

        if not self.chunker_version.strip():
            raise ChunkerConfigurationError("Chunker version must not be empty")

        if not self.supported_domains:
            raise ChunkerConfigurationError("Chunker must support at least one domain")

        if not self.supported_source_types:
            raise ChunkerConfigurationError("Chunker must support at least one source type")

        _ = self.config

    def _validate_input(
        self,
        *,
        source: SourceCatalogueEntry,
        document: ParsedDocument,
    ) -> None:
        """Ensure source metadata matches the parsed document."""

        if not source.enabled:
            raise UnsupportedDocumentError(f"Source {source.source_id} is disabled")

        if not self.supports(
            domain=source.domain,
            source_type=source.source_type,
        ):
            raise UnsupportedDocumentError(
                f"{self.chunker_name} does not support "
                f"domain={source.domain.value}, "
                f"source_type={source.source_type.value}"
            )

        if source.source_id != document.source_id:
            raise UnsupportedDocumentError("Catalogue source_id does not match parsed document")

        if source.domain is not document.domain:
            raise UnsupportedDocumentError("Catalogue domain does not match parsed document")

        if source.checksum is None:
            raise UnsupportedDocumentError(
                "The catalogue checksum must be populated before chunking"
            )

        if source.checksum != document.source_checksum:
            raise UnsupportedDocumentError("Catalogue checksum does not match parsed document")

    def _validate_output(
        self,
        *,
        source: SourceCatalogueEntry,
        document: ParsedDocument,
        chunks: tuple[ChunkDraft, ...],
    ) -> None:
        """Verify generated chunk identity, provenance, and sizing."""

        if not chunks:
            raise ChunkValidationError("Chunking produced no draft chunks")

        chunk_ids = [chunk.chunk_id for chunk in chunks]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ChunkValidationError("Chunking produced duplicate chunk IDs")

        document_sections = {section.section_id: section for section in document.sections}

        document_unit_ids = {
            unit.unit_id for section in document.sections for unit in section.units
        }

        for chunk in chunks:
            if chunk.source_id != source.source_id:
                raise ChunkValidationError("Chunk source_id does not match source")

            if chunk.source_checksum != document.source_checksum:
                raise ChunkValidationError("Chunk source checksum does not match document")

            if chunk.domain is not source.domain:
                raise ChunkValidationError("Chunk domain does not match source")

            if chunk.source_type is not source.source_type:
                raise ChunkValidationError("Chunk source type does not match source")

            if chunk.parser_name != document.parser_name:
                raise ChunkValidationError("Chunk parser name does not match document")

            if chunk.parser_version != document.parser_version:
                raise ChunkValidationError("Chunk parser version does not match document")

            if chunk.chunker_name != self.chunker_name:
                raise ChunkValidationError("Chunk chunker name does not match implementation")

            if chunk.chunker_version != self.chunker_version:
                raise ChunkValidationError("Chunk chunker version does not match implementation")

            if chunk.section_id not in document_sections:
                raise ChunkValidationError("Chunk references an unknown document section")

            if not set(chunk.source_unit_ids).issubset(document_unit_ids):
                raise ChunkValidationError("Chunk references unknown document units")

            expected_checksum = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()

            if chunk.text_checksum != expected_checksum:
                raise ChunkValidationError("Chunk text checksum is invalid")

            expected_token_count = self.count_tokens(chunk.text)

            if chunk.token_count != expected_token_count:
                raise ChunkValidationError("Chunk token count does not match chunk text")

            if not (self.config.minimum_tokens <= chunk.token_count <= self.config.maximum_tokens):
                raise ChunkValidationError("Chunk token count is outside configured limits")

            if chunk.citation.canonical_url != source.canonical_url:
                raise ChunkValidationError("Chunk citation URL does not match source")

            if chunk.review_status is not ReviewStatus.DRAFT:
                raise ChunkValidationError("Newly generated chunks must remain in draft status")


class ChunkerRegistry:
    """Resolve chunkers from domain and source type."""

    def __init__(
        self,
        chunkers: Iterable[SourceChunker] = (),
    ) -> None:
        """Initialize the registry with optional chunkers."""

        self._chunkers: dict[
            tuple[Domain, SourceType],
            SourceChunker,
        ] = {}

        for chunker in chunkers:
            self.register(chunker)

    def register(
        self,
        chunker: SourceChunker,
    ) -> None:
        """Register one chunker for its supported source combinations."""

        chunker._validate_chunker_metadata()

        combinations = (
            (
                domain,
                source_type,
            )
            for domain in chunker.supported_domains
            for source_type in chunker.supported_source_types
        )

        for combination in combinations:
            existing = self._chunkers.get(combination)

            if existing is not None:
                domain, source_type = combination

                raise ChunkerConfigurationError(
                    "A chunker is already registered for "
                    f"domain={domain.value}, "
                    f"source_type={source_type.value}: "
                    f"{existing.chunker_name}"
                )

            self._chunkers[combination] = chunker

    def resolve(
        self,
        *,
        domain: Domain,
        source_type: SourceType,
    ) -> SourceChunker:
        """Return the chunker registered for a source."""

        chunker = self._chunkers.get(
            (
                domain,
                source_type,
            )
        )

        if chunker is None:
            raise UnsupportedDocumentError(
                "No chunker is registered for "
                f"domain={domain.value}, "
                f"source_type={source_type.value}"
            )

        return chunker

    def chunk(
        self,
        *,
        source: SourceCatalogueEntry,
        document: ParsedDocument,
    ) -> tuple[ChunkDraft, ...]:
        """Resolve and execute the appropriate chunker."""

        chunker = self.resolve(
            domain=source.domain,
            source_type=source.source_type,
        )

        return chunker.chunk(
            source=source,
            document=document,
        )
