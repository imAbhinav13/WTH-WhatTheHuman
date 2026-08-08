"""Structured-text and OCR-text parsers for corpus sources."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Final

from apps.api.ingestion.parsers.base import (
    CorpusParserError,
    SourceParser,
)
from apps.api.models.corpus import (
    AcquiredSourceArtifact,
    DocumentSection,
    DocumentUnit,
    DocumentUnitType,
    ParsedDocument,
    ParserWarning,
    ParserWarningSeverity,
    SourceCatalogueEntry,
    SourceFormat,
)
from apps.api.models.enums import SourceType

PARSER_NAME: Final = "structured_text"
PARSER_VERSION: Final = "1.0.0"

_MARKDOWN_HEADING_PATTERN: Final = re.compile(r"^(?P<marks>#{1,6})\s+(?P<title>.+?)\s*#*\s*$")

_SETEXT_HEADING_PATTERN: Final = re.compile(r"^\s*(?P<underline>=+|-+)\s*$")

_EXPLICIT_HEADING_PATTERN: Final = re.compile(
    r"""(?ix)
    ^
    (?:
        part
        | book
        | chapter
        | section
        | lecture
        | discourse
        | appendix
        | introduction
        | preface
        | foreword
        | contents
        | translator(?:'s)?\s+note
        | editor(?:'s)?\s+note
        | notes
        | karika
        | sutra
    )
    \b
    .*$
    """
)

_PAGE_MARKER_PATTERN: Final = re.compile(
    r"""(?ix)
    ^
    [\[\(\--—_\s]*
    (?:page|pg\.?|p\.)
    \s*
    (?P<label>[0-9]+|[ivxlcdm]+)
    [\]\)\--—_\s]*
    $
    """
)

_WHITESPACE_PATTERN: Final = re.compile(r"\s+")
_HORIZONTAL_WHITESPACE_PATTERN: Final = re.compile(r"[ \t\v\f]+")
_SAFE_LOCATOR_PATTERN: Final = re.compile(r"[^A-Za-z0-9._-]+")

_TRANSLATOR_NOTE_PATTERN: Final = re.compile(r"(?i)^\s*translator(?:'s)?\s+note\b")

_EDITOR_NOTE_PATTERN: Final = re.compile(r"(?i)^\s*editor(?:'s)?\s+note\b")

_FOOTNOTE_SECTION_PATTERN: Final = re.compile(r"(?i)^\s*(?:footnotes?|notes)\b")

_APPENDIX_PATTERN: Final = re.compile(r"(?i)^\s*appendix\b")

_HTML_MARKUP_PATTERN: Final = re.compile(r"(?is)<\s*(?:html|body|head|div|p|article)\b")


class StructuredTextParseError(CorpusParserError):
    """Raised when structured text cannot be parsed safely."""


@dataclass(frozen=True, slots=True)
class _TextBlock:
    """Heading or paragraph extracted from a text artifact."""

    text: str
    locator: str
    heading_level: int | None


@dataclass(frozen=True, slots=True)
class _HeadingMatch:
    """Detected heading and number of consumed source lines."""

    title: str
    level: int
    consumed_lines: int


class StructuredTextParser(SourceParser):
    """Parse structured plain text and repository OCR exports."""

    @property
    def parser_name(self) -> str:
        """Return the stable parser implementation name."""

        return PARSER_NAME

    @property
    def parser_version(self) -> str:
        """Return the parser version."""

        return PARSER_VERSION

    @property
    def supported_formats(self) -> frozenset[SourceFormat]:
        """Return the source formats supported by this parser."""

        return frozenset(
            {
                SourceFormat.STRUCTURED_TEXT,
                SourceFormat.OCR_TEXT,
                SourceFormat.SCANNED_BOOK_WITH_OCR,
            }
        )

    def _parse(
        self,
        *,
        source: SourceCatalogueEntry,
        artifact: AcquiredSourceArtifact,
        content: bytes,
    ) -> ParsedDocument:
        """Convert structured or OCR text into normalized sections."""

        warnings: list[ParserWarning] = []

        text = self._decode_source_text(content)
        text = self.normalize_line_endings(text)
        text = text.replace("\f", "\n[[PAGE_BREAK]]\n")

        if _HTML_MARKUP_PATTERN.search(text[:8_192]):
            raise StructuredTextParseError(
                "The acquired artifact appears to contain HTML markup. "
                "Use the Gutenberg HTML parser or another HTML parser."
            )

        if not text.strip():
            raise StructuredTextParseError(f"Source {source.source_id} contains no usable text")

        if source.format in {
            SourceFormat.OCR_TEXT,
            SourceFormat.SCANNED_BOOK_WITH_OCR,
        }:
            warnings.append(
                ParserWarning(
                    code="ocr_source_requires_visual_review",
                    message=(
                        "This source is an OCR-derived text artifact. "
                        "Chunks must be checked against page images "
                        "before activation."
                    ),
                    severity=ParserWarningSeverity.WARNING,
                    structural_locator="text",
                )
            )

        blocks, page_marker_count = _extract_blocks(
            text=text,
        )

        if not blocks:
            raise StructuredTextParseError(
                f"Structured-text parsing produced no usable blocks for source {source.source_id}"
            )

        _append_text_quality_warnings(
            text=text,
            blocks=blocks,
            source_format=source.format,
            page_marker_count=page_marker_count,
            warnings=warnings,
        )

        sections = _build_sections(
            source=source,
            blocks=blocks,
            warnings=warnings,
        )

        if not sections:
            raise StructuredTextParseError(
                f"Structured-text parsing produced no usable sections for source {source.source_id}"
            )

        return ParsedDocument(
            source_id=source.source_id,
            source_checksum=artifact.checksum,
            domain=source.domain,
            title=source.title,
            parser_name=self.parser_name,
            parser_version=self.parser_version,
            sections=sections,
            warnings=tuple(warnings),
        )

    def _decode_source_text(
        self,
        content: bytes,
    ) -> str:
        """Decode a text artifact using conservative fallbacks."""

        if content.startswith(
            (
                b"\xff\xfe",
                b"\xfe\xff",
            )
        ):
            encodings = (
                "utf-16",
                "utf-8-sig",
                "utf-8",
                "cp1252",
            )
        else:
            encodings = (
                "utf-8-sig",
                "utf-8",
                "cp1252",
                "iso-8859-1",
            )

        return self.decode_text(
            content,
            encodings=encodings,
        )


def _extract_blocks(
    *,
    text: str,
) -> tuple[tuple[_TextBlock, ...], int]:
    """Extract headings and paragraphs in source order."""

    lines = text.splitlines()

    blocks: list[_TextBlock] = []
    paragraph_lines: list[str] = []

    paragraph_start_line = 0
    paragraph_page = "unpaginated"

    current_page = "unpaginated"
    page_break_number = 0
    page_marker_count = 0

    def flush_paragraph(
        *,
        end_line: int,
    ) -> None:
        nonlocal paragraph_lines
        nonlocal paragraph_start_line
        nonlocal paragraph_page

        if not paragraph_lines:
            return

        paragraph = _normalize_paragraph(paragraph_lines)

        if paragraph:
            blocks.append(
                _TextBlock(
                    text=paragraph,
                    locator=_line_locator(
                        page=paragraph_page,
                        start_line=paragraph_start_line,
                        end_line=end_line,
                    ),
                    heading_level=None,
                )
            )

        paragraph_lines = []
        paragraph_start_line = 0
        paragraph_page = current_page

    index = 0

    while index < len(lines):
        raw_line = lines[index]
        line_number = index + 1
        stripped = raw_line.strip()

        if stripped == "[[PAGE_BREAK]]":
            flush_paragraph(
                end_line=max(
                    1,
                    line_number - 1,
                )
            )

            page_break_number += 1
            page_marker_count += 1
            current_page = f"break-{page_break_number}"

            index += 1
            continue

        page_match = _PAGE_MARKER_PATTERN.fullmatch(stripped)

        if page_match is not None:
            flush_paragraph(
                end_line=max(
                    1,
                    line_number - 1,
                )
            )

            current_page = page_match.group("label").lower()
            page_marker_count += 1

            index += 1
            continue

        if not stripped:
            flush_paragraph(
                end_line=max(
                    1,
                    line_number - 1,
                )
            )

            index += 1
            continue

        heading = _detect_heading(
            lines=lines,
            index=index,
        )

        if heading is not None:
            flush_paragraph(
                end_line=max(
                    1,
                    line_number - 1,
                )
            )

            blocks.append(
                _TextBlock(
                    text=heading.title,
                    locator=_line_locator(
                        page=current_page,
                        start_line=line_number,
                        end_line=(line_number + heading.consumed_lines - 1),
                    ),
                    heading_level=heading.level,
                )
            )

            index += heading.consumed_lines
            continue

        if not paragraph_lines:
            paragraph_start_line = line_number
            paragraph_page = current_page

        paragraph_lines.append(stripped)
        index += 1

    flush_paragraph(
        end_line=max(
            1,
            len(lines),
        )
    )

    return tuple(blocks), page_marker_count


def _detect_heading(
    *,
    lines: list[str],
    index: int,
) -> _HeadingMatch | None:
    """Detect explicit, Markdown, Setext, or conservative headings."""

    line = lines[index].strip()

    markdown_match = _MARKDOWN_HEADING_PATTERN.fullmatch(line)

    if markdown_match is not None:
        title = markdown_match.group("title").strip()

        if title:
            return _HeadingMatch(
                title=title,
                level=len(markdown_match.group("marks")),
                consumed_lines=1,
            )

    if index + 1 < len(lines):
        next_line = lines[index + 1].strip()
        setext_match = _SETEXT_HEADING_PATTERN.fullmatch(next_line)

        if setext_match is not None and _is_heading_length(line):
            level = 1 if setext_match.group("underline").startswith("=") else 2

            return _HeadingMatch(
                title=line,
                level=level,
                consumed_lines=2,
            )

    if _EXPLICIT_HEADING_PATTERN.fullmatch(line) is not None and _is_heading_length(line):
        return _HeadingMatch(
            title=line,
            level=_explicit_heading_level(line),
            consumed_lines=1,
        )

    if _looks_like_uppercase_heading(line):
        return _HeadingMatch(
            title=line,
            level=2,
            consumed_lines=1,
        )

    return None


def _is_heading_length(text: str) -> bool:
    """Return whether text is reasonably sized for a heading."""

    word_count = len(text.split())

    return 1 <= word_count <= 18 and 1 <= len(text) <= 120


def _looks_like_uppercase_heading(text: str) -> bool:
    """Conservatively identify short all-uppercase headings."""

    if not _is_heading_length(text):
        return False

    if text.endswith(
        (
            ".",
            "?",
            "!",
            ";",
        )
    ):
        return False

    letters = [character for character in text if character.isalpha()]

    if len(letters) < 3:
        return False

    return all(character.isupper() for character in letters)


def _explicit_heading_level(text: str) -> int:
    """Assign a structural level to a recognized heading."""

    normalized = text.casefold()

    if normalized.startswith(
        (
            "part",
            "book",
        )
    ):
        return 1

    if normalized.startswith(
        (
            "chapter",
            "lecture",
            "discourse",
        )
    ):
        return 2

    if normalized.startswith(
        (
            "section",
            "karika",
            "sutra",
        )
    ):
        return 3

    return 1


def _normalize_paragraph(
    lines: list[str],
) -> str:
    """Join source lines without silently correcting their text."""

    normalized_lines = [
        _HORIZONTAL_WHITESPACE_PATTERN.sub(
            " ",
            line,
        ).strip()
        for line in lines
        if line.strip()
    ]

    return _WHITESPACE_PATTERN.sub(
        " ",
        " ".join(normalized_lines),
    ).strip()


def _line_locator(
    *,
    page: str,
    start_line: int,
    end_line: int,
) -> str:
    """Create a stable text locator from page and line positions."""

    safe_page = _SAFE_LOCATOR_PATTERN.sub(
        "-",
        page,
    ).strip("-")

    if not safe_page:
        safe_page = "unpaginated"

    if start_line == end_line:
        line_part = f"line[{start_line}]"
    else:
        line_part = f"lines[{start_line}-{end_line}]"

    return f"text/page[{safe_page}]/{line_part}"


def _build_sections(
    *,
    source: SourceCatalogueEntry,
    blocks: tuple[_TextBlock, ...],
    warnings: list[ParserWarning],
) -> tuple[DocumentSection, ...]:
    """Convert ordered text blocks into normalized sections."""

    sections: list[DocumentSection] = []

    current_title: str | None = None
    current_level = 0
    current_locator = "text/front-matter"
    current_blocks: list[_TextBlock] = []
    heading_count = 0

    def flush_section() -> None:
        nonlocal current_blocks

        if not current_blocks:
            return

        section_title = current_title if current_title is not None else "Front Matter"

        section_id = _stable_identifier(
            kind="section",
            source_id=source.source_id,
            locator=current_locator,
            text=section_title,
        )

        default_unit_type = _section_unit_type(
            source_type=source.source_type,
            section_title=section_title,
        )

        units = tuple(
            DocumentUnit(
                unit_id=_stable_identifier(
                    kind="unit",
                    source_id=source.source_id,
                    locator=block.locator,
                    text=block.text,
                ),
                order=order,
                unit_type=_paragraph_unit_type(
                    default_unit_type=default_unit_type,
                    text=block.text,
                ),
                heading=None,
                structural_locator=block.locator,
                text=block.text,
                parent_section_id=section_id,
            )
            for order, block in enumerate(current_blocks)
        )

        sections.append(
            DocumentSection(
                section_id=section_id,
                order=len(sections),
                level=current_level,
                title=section_title,
                structural_locator=current_locator,
                units=units,
            )
        )

        current_blocks = []

    for block in blocks:
        if block.heading_level is not None:
            if current_title is not None and not current_blocks:
                warnings.append(
                    ParserWarning(
                        code="heading_without_content",
                        message=("A detected heading contained no text before the next heading"),
                        severity=(ParserWarningSeverity.INFO),
                        structural_locator=current_locator,
                    )
                )

            flush_section()

            current_title = block.text
            current_level = min(
                12,
                max(
                    0,
                    block.heading_level - 1,
                ),
            )
            current_locator = block.locator
            heading_count += 1
            continue

        current_blocks.append(block)

    flush_section()

    if heading_count == 0:
        warnings.append(
            ParserWarning(
                code="no_explicit_headings",
                message=(
                    "No reliable headings were detected. "
                    "The source was represented as one "
                    "front-matter section."
                ),
                severity=ParserWarningSeverity.WARNING,
                structural_locator="text",
            )
        )

    return tuple(sections)


def _section_unit_type(
    *,
    source_type: SourceType,
    section_title: str,
) -> DocumentUnitType:
    """Determine the default unit type for a section."""

    if _TRANSLATOR_NOTE_PATTERN.match(section_title):
        return DocumentUnitType.TRANSLATOR_NOTE

    if _EDITOR_NOTE_PATTERN.match(section_title):
        return DocumentUnitType.EDITOR_NOTE

    if _FOOTNOTE_SECTION_PATTERN.match(section_title):
        return DocumentUnitType.FOOTNOTE

    if _APPENDIX_PATTERN.match(section_title):
        return DocumentUnitType.APPENDIX

    if source_type is SourceType.PRIMARY_TEXT:
        return DocumentUnitType.ROOT_TEXT

    if source_type is SourceType.COMMENTARY:
        return DocumentUnitType.COMMENTARY

    return DocumentUnitType.BODY


def _paragraph_unit_type(
    *,
    default_unit_type: DocumentUnitType,
    text: str,
) -> DocumentUnitType:
    """Classify clearly labelled note paragraphs."""

    if _TRANSLATOR_NOTE_PATTERN.match(text):
        return DocumentUnitType.TRANSLATOR_NOTE

    if _EDITOR_NOTE_PATTERN.match(text):
        return DocumentUnitType.EDITOR_NOTE

    return default_unit_type


def _append_text_quality_warnings(
    *,
    text: str,
    blocks: tuple[_TextBlock, ...],
    source_format: SourceFormat,
    page_marker_count: int,
    warnings: list[ParserWarning],
) -> None:
    """Add document-level warnings without modifying source wording."""

    replacement_character_count = text.count("\ufffd")

    if replacement_character_count:
        warnings.append(
            ParserWarning(
                code="replacement_characters_detected",
                message=(
                    f"Detected {replacement_character_count} "
                    "Unicode replacement characters. "
                    "The source encoding or OCR must be reviewed."
                ),
                severity=ParserWarningSeverity.ERROR,
                structural_locator="text",
            )
        )

    line_end_hyphen_count = sum(
        1 for line in text.splitlines() if line.rstrip().endswith("-") and len(line.strip()) > 2
    )

    if line_end_hyphen_count >= 3:
        warnings.append(
            ParserWarning(
                code="possible_ocr_line_hyphenation",
                message=(
                    f"Detected {line_end_hyphen_count} lines ending "
                    "in hyphens. No automatic dehyphenation was "
                    "performed."
                ),
                severity=ParserWarningSeverity.WARNING,
                structural_locator="text",
            )
        )

    suspicious_blocks = sum(
        1
        for block in blocks
        if block.heading_level is None and _looks_textually_suspicious(block.text)
    )

    if suspicious_blocks:
        warnings.append(
            ParserWarning(
                code="possible_ocr_corruption",
                message=(
                    f"Detected {suspicious_blocks} text blocks "
                    "with unusually low alphabetic content or "
                    "excessive OCR-like symbols."
                ),
                severity=ParserWarningSeverity.WARNING,
                structural_locator="text",
            )
        )

    if source_format is SourceFormat.SCANNED_BOOK_WITH_OCR and page_marker_count == 0:
        warnings.append(
            ParserWarning(
                code="missing_page_markers",
                message=(
                    "The scanned-book OCR text contains no "
                    "recognized page markers. Citations may require "
                    "manual page-image reconciliation."
                ),
                severity=ParserWarningSeverity.WARNING,
                structural_locator="text",
            )
        )


def _looks_textually_suspicious(text: str) -> bool:
    """Return whether a block has signs of possible OCR corruption."""

    if len(text) < 40:
        return False

    alphabetic_count = sum(character.isalpha() for character in text)
    visible_count = sum(not character.isspace() for character in text)

    if visible_count == 0:
        return True

    alphabetic_ratio = alphabetic_count / visible_count

    unusual_symbol_count = sum(
        character
        in {
            "|",
            "¬",
            "¦",
            "§",
            "¤",
        }
        for character in text
    )

    return alphabetic_ratio < 0.45 or unusual_symbol_count >= 4


def _stable_identifier(
    *,
    kind: str,
    source_id: str,
    locator: str,
    text: str | None = None,
) -> str:
    """Create a stable identifier from source provenance."""

    material = "\n".join(
        value
        for value in (
            kind,
            source_id,
            locator,
            text,
        )
        if value is not None
    )

    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]

    return f"{source_id}:{kind}:{digest}"
