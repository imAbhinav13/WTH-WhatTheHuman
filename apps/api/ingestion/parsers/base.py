"""Base contracts, validation, and routing for corpus source parsers."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import final

from apps.api.models.corpus import (
    AcquiredSourceArtifact,
    ParsedDocument,
    SourceCatalogueEntry,
    SourceFormat,
)


class CorpusParserError(RuntimeError):
    """Base exception for corpus parsing failures."""


class ParserRegistrationError(CorpusParserError):
    """Raised when a parser cannot be registered safely."""


class UnsupportedSourceFormatError(CorpusParserError):
    """Raised when no parser supports a source format."""


class ArtifactValidationError(CorpusParserError):
    """Raised when an acquired artifact fails validation."""


class ArtifactDecodingError(CorpusParserError):
    """Raised when source bytes cannot be decoded reliably."""


class ParsedDocumentValidationError(CorpusParserError):
    """Raised when parser output conflicts with source metadata."""


class SourceParser(ABC):
    """Template contract implemented by source-specific parsers.

    The public ``parse`` method performs common validation before and
    after delegating source-specific extraction to ``_parse``.
    """

    @property
    @abstractmethod
    def parser_name(self) -> str:
        """Return the stable parser implementation name."""

    @property
    @abstractmethod
    def parser_version(self) -> str:
        """Return the parser version used for reproducibility."""

    @property
    @abstractmethod
    def supported_formats(self) -> frozenset[SourceFormat]:
        """Return all catalogue formats accepted by this parser."""

    @final
    def parse(
        self,
        *,
        source: SourceCatalogueEntry,
        artifact: AcquiredSourceArtifact,
    ) -> ParsedDocument:
        """Validate, parse, and verify one acquired source artifact."""

        self._validate_parser_metadata()

        content = self._load_and_validate_artifact(
            source=source,
            artifact=artifact,
        )

        document = self._parse(
            source=source,
            artifact=artifact,
            content=content,
        )

        self._validate_parsed_document(
            source=source,
            artifact=artifact,
            document=document,
        )

        return document

    @abstractmethod
    def _parse(
        self,
        *,
        source: SourceCatalogueEntry,
        artifact: AcquiredSourceArtifact,
        content: bytes,
    ) -> ParsedDocument:
        """Convert validated source bytes into a normalized document."""

    def supports(self, source_format: SourceFormat) -> bool:
        """Return whether this parser accepts the supplied format."""

        return source_format in self.supported_formats

    @staticmethod
    def decode_text(
        content: bytes,
        *,
        encodings: tuple[str, ...] = (
            "utf-8-sig",
            "utf-8",
            "cp1252",
        ),
    ) -> str:
        """Decode bytes using an explicit ordered encoding strategy."""

        if not encodings:
            raise ArtifactDecodingError("At least one text encoding must be provided")

        attempted: list[str] = []

        for encoding in encodings:
            attempted.append(encoding)

            try:
                return content.decode(encoding)
            except (LookupError, UnicodeDecodeError):
                continue

        attempted_text = ", ".join(attempted)

        raise ArtifactDecodingError(
            f"Unable to decode source artifact using encodings: {attempted_text}"
        )

    @staticmethod
    def normalize_line_endings(text: str) -> str:
        """Normalize source line endings without changing wording."""

        return text.replace("\r\n", "\n").replace("\r", "\n")

    def _validate_parser_metadata(self) -> None:
        """Validate the parser's declared identity and capabilities."""

        if not self.parser_name.strip():
            raise ParserRegistrationError("Parser name must not be empty")

        if not self.parser_version.strip():
            raise ParserRegistrationError("Parser version must not be empty")

        if not self.supported_formats:
            raise ParserRegistrationError(f"{self.parser_name} must support at least one format")

    def _load_and_validate_artifact(
        self,
        *,
        source: SourceCatalogueEntry,
        artifact: AcquiredSourceArtifact,
    ) -> bytes:
        """Load and verify the exact raw file before parsing."""

        if not self.supports(source.format):
            raise UnsupportedSourceFormatError(
                f"{self.parser_name} does not support format {source.format.value}"
            )

        if source.source_id != artifact.source_id:
            raise ArtifactValidationError(
                "Catalogue source_id does not match acquired artifact: "
                f"{source.source_id!r} != {artifact.source_id!r}"
            )

        path = artifact.local_path

        if not path.exists():
            raise ArtifactValidationError(f"Acquired source file does not exist: {path}")

        if not path.is_file():
            raise ArtifactValidationError(f"Acquired source path is not a file: {path}")

        try:
            content = path.read_bytes()
        except OSError as exc:
            raise ArtifactValidationError(
                f"Unable to read acquired source file {path}: {exc}"
            ) from exc

        actual_size = len(content)

        if actual_size != artifact.file_size_bytes:
            raise ArtifactValidationError(
                "Acquired source size mismatch: "
                f"expected {artifact.file_size_bytes}, "
                f"found {actual_size}"
            )

        actual_checksum = hashlib.sha256(content).hexdigest()

        if actual_checksum != artifact.checksum:
            raise ArtifactValidationError(
                "Acquired source checksum mismatch: "
                f"expected {artifact.checksum}, "
                f"found {actual_checksum}"
            )

        if source.checksum is not None and source.checksum != actual_checksum:
            raise ArtifactValidationError(
                "Catalogue checksum does not match acquired source: "
                f"expected {source.checksum}, "
                f"found {actual_checksum}"
            )

        return content

    def _validate_parsed_document(
        self,
        *,
        source: SourceCatalogueEntry,
        artifact: AcquiredSourceArtifact,
        document: ParsedDocument,
    ) -> None:
        """Ensure parser output remains tied to its source artifact."""

        if document.source_id != source.source_id:
            raise ParsedDocumentValidationError(
                "Parsed document source_id does not match catalogue"
            )

        if document.source_checksum != artifact.checksum:
            raise ParsedDocumentValidationError("Parsed document checksum does not match artifact")

        if document.domain is not source.domain:
            raise ParsedDocumentValidationError("Parsed document domain does not match catalogue")

        if document.parser_name != self.parser_name:
            raise ParsedDocumentValidationError("Parsed document parser_name does not match parser")

        if document.parser_version != self.parser_version:
            raise ParsedDocumentValidationError(
                "Parsed document parser_version does not match parser"
            )


class ParserRegistry:
    """Map catalogue source formats to parser implementations."""

    def __init__(
        self,
        parsers: Iterable[SourceParser] = (),
    ) -> None:
        """Initialize the registry with optional parser instances."""

        self._parsers: dict[SourceFormat, SourceParser] = {}

        for parser in parsers:
            self.register(parser)

    @property
    def registered_formats(self) -> tuple[SourceFormat, ...]:
        """Return registered formats in stable lexical order."""

        return tuple(
            sorted(
                self._parsers,
                key=lambda source_format: source_format.value,
            )
        )

    def register(self, parser: SourceParser) -> None:
        """Register one parser for each of its supported formats."""

        parser._validate_parser_metadata()

        for source_format in parser.supported_formats:
            existing = self._parsers.get(source_format)

            if existing is not None:
                raise ParserRegistrationError(
                    "A parser is already registered for "
                    f"{source_format.value}: "
                    f"{existing.parser_name}"
                )

        for source_format in parser.supported_formats:
            self._parsers[source_format] = parser

    def resolve(
        self,
        source_format: SourceFormat,
    ) -> SourceParser:
        """Return the parser registered for a source format."""

        parser = self._parsers.get(source_format)

        if parser is None:
            raise UnsupportedSourceFormatError(
                f"No parser is registered for source format {source_format.value}"
            )

        return parser

    def parse(
        self,
        *,
        source: SourceCatalogueEntry,
        artifact: AcquiredSourceArtifact,
    ) -> ParsedDocument:
        """Resolve and execute the parser selected by catalogue format."""

        parser = self.resolve(source.format)

        return parser.parse(
            source=source,
            artifact=artifact,
        )
