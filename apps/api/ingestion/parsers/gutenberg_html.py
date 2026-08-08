"""Project Gutenberg HTML parser for classical-text corpus sources."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from html.parser import HTMLParser
from typing import Final

from apps.api.ingestion.parsers.base import (
    ArtifactDecodingError,
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

PARSER_NAME: Final = "gutenberg_html"
PARSER_VERSION: Final = "1.0.0"

_HEADING_TAGS: Final = frozenset(
    {
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    }
)

_TEXT_BLOCK_TAGS: Final = frozenset(
    {
        "p",
        "pre",
        "li",
    }
)

_VOID_TAGS: Final = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)

_SKIPPED_TAGS: Final = frozenset(
    {
        "script",
        "style",
        "nav",
        "noscript",
        "svg",
        "canvas",
        "form",
    }
)

_SKIPPED_METADATA_TOKENS: Final = frozenset(
    {
        "pgheader",
        "pgfooter",
        "pg-header",
        "pg-footer",
        "pagenum",
        "page-number",
        "page_number",
        "toc",
        "table-of-contents",
        "table_of_contents",
    }
)

_START_MARKER_PATTERN: Final = re.compile(
    r"(?im)^[ \t]*\*{0,3}[ \t]*"
    r"START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK"
    r".*?$"
)

_END_MARKER_PATTERN: Final = re.compile(
    r"(?im)^[ \t]*\*{0,3}[ \t]*"
    r"END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK"
    r".*?$"
)

_META_CHARSET_PATTERN: Final = re.compile(
    rb"""(?ix)
    <meta[^>]+charset\s*=\s*
    ["']?\s*([a-z0-9._-]+)
    """
)

_CONTENT_TYPE_CHARSET_PATTERN: Final = re.compile(
    rb"""(?ix)
    <meta[^>]+content\s*=\s*
    ["'][^"']*charset\s*=\s*([a-z0-9._-]+)
    """
)

_WHITESPACE_PATTERN: Final = re.compile(r"\s+")
_HORIZONTAL_WHITESPACE_PATTERN: Final = re.compile(r"[ \t\f\v]+")
_METADATA_TOKEN_PATTERN: Final = re.compile(r"[^a-z0-9_-]+")


class GutenbergHTMLParseError(CorpusParserError):
    """Raised when Gutenberg HTML cannot be parsed safely."""


@dataclass(slots=True)
class _HTMLNode:
    """Minimal HTML tree node used by the Gutenberg parser."""

    tag: str
    attributes: dict[str, str]
    children: list[_HTMLNode | str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _ContentBlock:
    """Normalized heading or textual block extracted from HTML."""

    text: str
    locator: str
    heading_level: int | None
    unit_type: DocumentUnitType | None


class _HTMLTreeBuilder(HTMLParser):
    """Build a minimal tolerant HTML tree using the standard library."""

    def __init__(self) -> None:
        """Initialize the parser and synthetic document root."""

        super().__init__(convert_charrefs=True)

        self.root = _HTMLNode(
            tag="document",
            attributes={},
        )
        self._stack: list[_HTMLNode] = [self.root]

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        """Add a normal start tag to the tree."""

        normalized_tag = tag.lower()

        node = _HTMLNode(
            tag=normalized_tag,
            attributes={name.lower(): value or "" for name, value in attrs},
        )

        self._stack[-1].children.append(node)

        if normalized_tag not in _VOID_TAGS:
            self._stack.append(node)

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        """Add a self-closing tag to the tree."""

        node = _HTMLNode(
            tag=tag.lower(),
            attributes={name.lower(): value or "" for name, value in attrs},
        )

        self._stack[-1].children.append(node)

    def handle_endtag(self, tag: str) -> None:
        """Close the nearest matching open tag."""

        normalized_tag = tag.lower()

        matching_index: int | None = None

        for index in range(
            len(self._stack) - 1,
            0,
            -1,
        ):
            if self._stack[index].tag == normalized_tag:
                matching_index = index
                break

        if matching_index is not None:
            del self._stack[matching_index:]

    def handle_data(self, data: str) -> None:
        """Add literal text to the current tree node."""

        if data:
            self._stack[-1].children.append(data)


class GutenbergHTMLParser(SourceParser):
    """Parse Gutenberg HTML into normalized classical-text sections."""

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
                SourceFormat.GUTENBERG_HTML,
            }
        )

    def _parse(
        self,
        *,
        source: SourceCatalogueEntry,
        artifact: AcquiredSourceArtifact,
        content: bytes,
    ) -> ParsedDocument:
        """Convert validated Gutenberg HTML into a parsed document."""

        warnings: list[ParserWarning] = []

        html = self._decode_html(
            content=content,
            warnings=warnings,
        )
        html = self.normalize_line_endings(html)

        tree_builder = _HTMLTreeBuilder()

        try:
            tree_builder.feed(html)
            tree_builder.close()
        except Exception as exc:
            raise GutenbergHTMLParseError(
                f"Unable to parse Gutenberg HTML for {source.source_id}: {exc}"
            ) from exc

        body = _find_first_node(
            tree_builder.root,
            tag="body",
        )

        if body is None:
            warnings.append(
                ParserWarning(
                    code="missing_html_body",
                    message=("No HTML body element was found; the complete document tree was used"),
                    severity=ParserWarningSeverity.WARNING,
                    structural_locator="document",
                )
            )
            content_root = tree_builder.root
            root_locator = "document"
        else:
            content_root = body
            root_locator = "body"

        embedded_title = _extract_embedded_title(tree_builder.root)

        if embedded_title is None:
            warnings.append(
                ParserWarning(
                    code="missing_html_title",
                    message=("No usable HTML title was found; the catalogue title was retained"),
                    severity=ParserWarningSeverity.INFO,
                    structural_locator="document/head/title",
                )
            )

        blocks = tuple(
            _extract_content_blocks(
                node=content_root,
                path=root_locator,
                source_type=source.source_type,
                inherited_tokens=frozenset(),
            )
        )

        trimmed_blocks = _trim_gutenberg_boilerplate(
            blocks=blocks,
            warnings=warnings,
        )

        if not trimmed_blocks:
            raise GutenbergHTMLParseError(
                f"Gutenberg HTML parsing produced no usable content for source {source.source_id}"
            )

        sections = _build_sections(
            source=source,
            blocks=trimmed_blocks,
            warnings=warnings,
        )

        if not sections:
            raise GutenbergHTMLParseError(
                f"Gutenberg HTML parsing produced no usable sections for source {source.source_id}"
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

    def _decode_html(
        self,
        *,
        content: bytes,
        warnings: list[ParserWarning],
    ) -> str:
        """Decode HTML using declared charset and safe fallbacks."""

        declared_encoding = _detect_declared_encoding(content)

        encodings: list[str] = []

        if declared_encoding is not None:
            encodings.append(declared_encoding)

        for encoding in (
            "utf-8-sig",
            "utf-8",
            "cp1252",
            "iso-8859-1",
        ):
            if encoding not in encodings:
                encodings.append(encoding)

        try:
            return self.decode_text(
                content,
                encodings=tuple(encodings),
            )
        except ArtifactDecodingError:
            raise
        except Exception as exc:
            raise GutenbergHTMLParseError(f"Unexpected HTML decoding failure: {exc}") from exc

        finally:
            if declared_encoding is None:
                warnings.append(
                    ParserWarning(
                        code="missing_declared_charset",
                        message=(
                            "The HTML did not declare a recognizable "
                            "character encoding; fallback decoding "
                            "was used"
                        ),
                        severity=ParserWarningSeverity.INFO,
                        structural_locator="document/head/meta",
                    )
                )


def _detect_declared_encoding(
    content: bytes,
) -> str | None:
    """Return a character encoding declared in early HTML metadata."""

    header = content[:16_384]

    match = _META_CHARSET_PATTERN.search(header)

    if match is None:
        match = _CONTENT_TYPE_CHARSET_PATTERN.search(header)

    if match is None:
        return None

    try:
        return match.group(1).decode("ascii").lower()
    except UnicodeDecodeError:
        return None


def _find_first_node(
    node: _HTMLNode,
    *,
    tag: str,
) -> _HTMLNode | None:
    """Return the first descendant with the requested tag."""

    if node.tag == tag:
        return node

    for child in node.children:
        if not isinstance(child, _HTMLNode):
            continue

        match = _find_first_node(
            child,
            tag=tag,
        )

        if match is not None:
            return match

    return None


def _extract_embedded_title(
    root: _HTMLNode,
) -> str | None:
    """Extract a normalized HTML title when present."""

    title_node = _find_first_node(
        root,
        tag="title",
    )

    if title_node is None:
        return None

    title = _node_text(
        title_node,
        preserve_lines=False,
    )

    return title or None


def _extract_content_blocks(
    *,
    node: _HTMLNode,
    path: str,
    source_type: SourceType,
    inherited_tokens: frozenset[str],
) -> Iterator[_ContentBlock]:
    """Yield headings and text blocks in document order."""

    own_tokens = _metadata_tokens(node)
    combined_tokens = inherited_tokens | own_tokens

    if _should_skip_node(
        node=node,
        metadata_tokens=combined_tokens,
    ):
        return

    sibling_counts: dict[str, int] = {}

    for child in node.children:
        if not isinstance(child, _HTMLNode):
            continue

        sibling_counts[child.tag] = sibling_counts.get(child.tag, 0) + 1

        child_path = f"{path}/{child.tag}[{sibling_counts[child.tag]}]"

        child_tokens = combined_tokens | _metadata_tokens(child)

        if _should_skip_node(
            node=child,
            metadata_tokens=child_tokens,
        ):
            continue

        if child.tag in _HEADING_TAGS:
            heading = _node_text(
                child,
                preserve_lines=False,
            )

            if heading:
                yield _ContentBlock(
                    text=heading,
                    locator=child_path,
                    heading_level=int(child.tag[1]),
                    unit_type=None,
                )

            continue

        if child.tag in _TEXT_BLOCK_TAGS:
            text = _node_text(
                child,
                preserve_lines=child.tag == "pre",
            )

            if text:
                yield _ContentBlock(
                    text=text,
                    locator=child_path,
                    heading_level=None,
                    unit_type=_classify_unit(
                        source_type=source_type,
                        metadata_tokens=child_tokens,
                    ),
                )

            continue

        if child.tag == "blockquote":
            if _contains_text_block(child):
                yield from _extract_content_blocks(
                    node=child,
                    path=child_path,
                    source_type=source_type,
                    inherited_tokens=child_tokens,
                )
            else:
                text = _node_text(
                    child,
                    preserve_lines=False,
                )

                if text:
                    yield _ContentBlock(
                        text=text,
                        locator=child_path,
                        heading_level=None,
                        unit_type=_classify_unit(
                            source_type=source_type,
                            metadata_tokens=child_tokens,
                        ),
                    )

            continue

        yield from _extract_content_blocks(
            node=child,
            path=child_path,
            source_type=source_type,
            inherited_tokens=child_tokens,
        )


def _should_skip_node(
    *,
    node: _HTMLNode,
    metadata_tokens: frozenset[str],
) -> bool:
    """Return whether an HTML subtree should be excluded."""

    if node.tag in _SKIPPED_TAGS:
        return True

    return bool(metadata_tokens & _SKIPPED_METADATA_TOKENS)


def _metadata_tokens(
    node: _HTMLNode,
) -> frozenset[str]:
    """Extract normalized class and ID metadata tokens."""

    values = (
        node.attributes.get("class", ""),
        node.attributes.get("id", ""),
        node.attributes.get("role", ""),
    )

    tokens: set[str] = set()

    for value in values:
        normalized = value.lower().strip()

        if not normalized:
            continue

        tokens.update(token for token in _METADATA_TOKEN_PATTERN.split(normalized) if token)

    return frozenset(tokens)


def _contains_text_block(node: _HTMLNode) -> bool:
    """Return whether a node contains a nested block element."""

    for child in node.children:
        if not isinstance(child, _HTMLNode):
            continue

        if child.tag in _TEXT_BLOCK_TAGS:
            return True

        if _contains_text_block(child):
            return True

    return False


def _classify_unit(
    *,
    source_type: SourceType,
    metadata_tokens: frozenset[str],
) -> DocumentUnitType:
    """Classify a text block using structural metadata."""

    if _contains_note_tokens(
        metadata_tokens,
        prefix="translator",
    ):
        return DocumentUnitType.TRANSLATOR_NOTE

    if _contains_note_tokens(
        metadata_tokens,
        prefix="editor",
    ):
        return DocumentUnitType.EDITOR_NOTE

    if any(
        token.startswith("footnote") or token in {"fn", "foot-note"} for token in metadata_tokens
    ):
        return DocumentUnitType.FOOTNOTE

    if source_type is SourceType.PRIMARY_TEXT:
        return DocumentUnitType.ROOT_TEXT

    if source_type is SourceType.COMMENTARY:
        return DocumentUnitType.COMMENTARY

    return DocumentUnitType.BODY


def _contains_note_tokens(
    metadata_tokens: frozenset[str],
    *,
    prefix: str,
) -> bool:
    """Return whether metadata identifies a specialized note."""

    compact = {token.replace("-", "").replace("_", "") for token in metadata_tokens}

    return f"{prefix}note" in compact or (prefix in compact and "note" in compact)


def _node_text(
    node: _HTMLNode,
    *,
    preserve_lines: bool,
) -> str:
    """Extract normalized textual content from one HTML node."""

    parts: list[str] = []

    _collect_text(
        node=node,
        parts=parts,
    )

    raw_text = "".join(parts)

    if preserve_lines:
        lines = [
            _HORIZONTAL_WHITESPACE_PATTERN.sub(
                " ",
                line,
            ).strip()
            for line in raw_text.splitlines()
        ]

        normalized_lines: list[str] = []

        for line in lines:
            if line:
                normalized_lines.append(line)
            elif normalized_lines and normalized_lines[-1] != "":
                normalized_lines.append("")

        return "\n".join(normalized_lines).strip()

    return _WHITESPACE_PATTERN.sub(
        " ",
        raw_text,
    ).strip()


def _collect_text(
    *,
    node: _HTMLNode,
    parts: list[str],
) -> None:
    """Collect visible text while preserving explicit line breaks."""

    for child in node.children:
        if isinstance(child, str):
            parts.append(child)
            continue

        if child.tag in _SKIPPED_TAGS:
            continue

        if child.tag == "br":
            parts.append("\n")
            continue

        _collect_text(
            node=child,
            parts=parts,
        )

        if child.tag in {
            "p",
            "pre",
            "li",
            "div",
            "blockquote",
        }:
            parts.append("\n")


def _trim_gutenberg_boilerplate(
    *,
    blocks: tuple[_ContentBlock, ...],
    warnings: list[ParserWarning],
) -> tuple[_ContentBlock, ...]:
    """Remove content before and after Gutenberg boundary markers."""

    contains_start_marker = any(_START_MARKER_PATTERN.search(block.text) for block in blocks)

    trimmed: list[_ContentBlock] = []
    started = not contains_start_marker
    found_start = False
    found_end = False

    for block in blocks:
        if not started:
            start_match = _START_MARKER_PATTERN.search(block.text)

            if start_match is None:
                continue

            found_start = True
            started = True

            suffix = block.text[start_match.end() :].strip()

            if suffix:
                trimmed.append(
                    replace(
                        block,
                        text=suffix,
                    )
                )

            continue

        end_match = _END_MARKER_PATTERN.search(block.text)

        if end_match is not None:
            found_end = True

            prefix = block.text[: end_match.start()].strip()

            if prefix:
                trimmed.append(
                    replace(
                        block,
                        text=prefix,
                    )
                )

            break

        trimmed.append(block)

    if not found_start:
        warnings.append(
            ParserWarning(
                code="missing_gutenberg_start_marker",
                message=(
                    "No Project Gutenberg start marker was found; "
                    "content extraction began at the HTML body"
                ),
                severity=ParserWarningSeverity.WARNING,
                structural_locator="document",
            )
        )

    if not found_end:
        warnings.append(
            ParserWarning(
                code="missing_gutenberg_end_marker",
                message=(
                    "No Project Gutenberg end marker was found; "
                    "content extraction continued to the end "
                    "of the HTML body"
                ),
                severity=ParserWarningSeverity.WARNING,
                structural_locator="document",
            )
        )

    return tuple(block for block in trimmed if block.text.strip())


def _build_sections(
    *,
    source: SourceCatalogueEntry,
    blocks: tuple[_ContentBlock, ...],
    warnings: list[ParserWarning],
) -> tuple[DocumentSection, ...]:
    """Convert ordered content blocks into normalized sections."""

    sections: list[DocumentSection] = []

    current_title: str | None = None
    current_level = 0
    current_locator = "document/front-matter"
    current_units: list[_ContentBlock] = []

    def flush_section() -> None:
        if not current_units:
            return

        section_title = current_title if current_title is not None else "Front Matter"

        section_id = _stable_identifier(
            kind="section",
            source_id=source.source_id,
            locator=current_locator,
            text=section_title,
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
                unit_type=(
                    block.unit_type if block.unit_type is not None else DocumentUnitType.OTHER
                ),
                heading=None,
                structural_locator=block.locator,
                text=block.text,
                parent_section_id=section_id,
            )
            for order, block in enumerate(current_units)
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

    for block in blocks:
        if block.heading_level is not None:
            flush_section()

            current_units = []
            current_title = block.text
            current_level = min(
                12,
                max(
                    0,
                    block.heading_level - 1,
                ),
            )
            current_locator = block.locator
            continue

        current_units.append(block)

    flush_section()

    if not sections:
        warnings.append(
            ParserWarning(
                code="no_usable_sections",
                message=("No sections containing usable text were produced"),
                severity=ParserWarningSeverity.ERROR,
                structural_locator="document",
            )
        )

    return tuple(sections)


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
