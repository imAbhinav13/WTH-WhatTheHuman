"""PMC JATS XML parser for scientific corpus sources."""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
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

PARSER_NAME: Final = "pmc_jats"
PARSER_VERSION: Final = "1.0.0"

_WHITESPACE_PATTERN: Final = re.compile(r"\s+")
_IGNORED_ELEMENT_NAMES: Final = frozenset(
    {
        "article-id",
        "label",
        "permissions",
        "ref-list",
        "title",
    }
)
_SKIPPED_CONTAINER_NAMES: Final = frozenset(
    {
        "ack",
        "app-group",
        "back",
        "funding-group",
        "notes",
        "permissions",
        "ref-list",
    }
)

ExtractedUnit = tuple[
    DocumentUnitType,
    str,
    str,
]


class PMCJATSParseError(CorpusParserError):
    """Raised when a PMC JATS article cannot be parsed safely."""


class PMCJATSParser(SourceParser):
    """Parse PubMed Central JATS XML into normalized document sections."""

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
        """Return the catalogue formats supported by this parser."""

        return frozenset({SourceFormat.JATS_XML})

    def _parse(
        self,
        *,
        source: SourceCatalogueEntry,
        artifact: AcquiredSourceArtifact,
        content: bytes,
    ) -> ParsedDocument:
        """Convert PMC JATS XML into a normalized parsed document."""

        try:
            root = ET.fromstring(content)
        except ET.ParseError as exc:
            raise PMCJATSParseError(
                f"Invalid JATS XML for source {source.source_id}: {exc}"
            ) from exc

        if _local_name(root.tag) != "article":
            raise PMCJATSParseError(
                f"Expected a JATS <article> root element, found <{_local_name(root.tag)}>"
            )

        warnings: list[ParserWarning] = []
        sections: list[DocumentSection] = []

        article_metadata = _first_descendant(
            root,
            "article-meta",
        )

        title = self._extract_title(
            article_metadata=article_metadata,
            catalogue_title=source.title,
            warnings=warnings,
        )

        self._extract_abstracts(
            source=source,
            article_metadata=article_metadata,
            sections=sections,
            warnings=warnings,
        )

        body = _first_descendant(root, "body")

        if body is None:
            warnings.append(
                ParserWarning(
                    code="missing_body",
                    message=("The JATS article does not contain a body element"),
                    severity=ParserWarningSeverity.ERROR,
                    structural_locator="body",
                )
            )
        else:
            self._extract_body(
                source=source,
                body=body,
                sections=sections,
                warnings=warnings,
            )

        if not sections:
            raise PMCJATSParseError(
                f"JATS parsing produced no usable document sections for source {source.source_id}"
            )

        return ParsedDocument(
            source_id=source.source_id,
            source_checksum=artifact.checksum,
            domain=source.domain,
            title=title,
            parser_name=self.parser_name,
            parser_version=self.parser_version,
            sections=tuple(sections),
            warnings=tuple(warnings),
        )

    def _extract_title(
        self,
        *,
        article_metadata: ET.Element | None,
        catalogue_title: str,
        warnings: list[ParserWarning],
    ) -> str:
        """Extract the article title or fall back to catalogue metadata."""

        article_title = (
            _first_descendant(
                article_metadata,
                "article-title",
            )
            if article_metadata is not None
            else None
        )

        title = _element_text(article_title)

        if title:
            return title

        warnings.append(
            ParserWarning(
                code="missing_article_title",
                message=(
                    "No article-title was found in JATS metadata; the catalogue title was used"
                ),
                severity=ParserWarningSeverity.WARNING,
                structural_locator="front/article-meta/title-group",
            )
        )

        return catalogue_title

    def _extract_abstracts(
        self,
        *,
        source: SourceCatalogueEntry,
        article_metadata: ET.Element | None,
        sections: list[DocumentSection],
        warnings: list[ParserWarning],
    ) -> None:
        """Extract ordered abstract sections from article metadata."""

        if article_metadata is None:
            warnings.append(
                ParserWarning(
                    code="missing_article_metadata",
                    message=("The JATS article does not contain article-meta"),
                    severity=ParserWarningSeverity.WARNING,
                    structural_locator="front/article-meta",
                )
            )
            return

        abstracts = _direct_children(
            article_metadata,
            "abstract",
        )

        if not abstracts:
            warnings.append(
                ParserWarning(
                    code="missing_abstract",
                    message=("The JATS article does not contain an abstract"),
                    severity=ParserWarningSeverity.INFO,
                    structural_locator="front/article-meta/abstract",
                )
            )
            return

        for index, abstract in enumerate(
            abstracts,
            start=1,
        ):
            locator = f"front/article-meta/abstract[{index}]"

            abstract_title = _element_text(_direct_child(abstract, "title"))

            if not abstract_title:
                abstract_title = "Abstract" if len(abstracts) == 1 else f"Abstract {index}"

            self._append_section(
                source=source,
                container=abstract,
                title=abstract_title,
                locator=locator,
                level=0,
                default_unit_type=DocumentUnitType.ABSTRACT,
                sections=sections,
                warnings=warnings,
            )

    def _extract_body(
        self,
        *,
        source: SourceCatalogueEntry,
        body: ET.Element,
        sections: list[DocumentSection],
        warnings: list[ParserWarning],
    ) -> None:
        """Extract body-level content and recursively process sections."""

        body_has_direct_content = any(
            _local_name(child.tag) != "sec"
            and _local_name(child.tag) not in _IGNORED_ELEMENT_NAMES
            and _local_name(child.tag) not in _SKIPPED_CONTAINER_NAMES
            for child in list(body)
        )

        if body_has_direct_content:
            self._append_section(
                source=source,
                container=body,
                title="Body",
                locator="body",
                level=0,
                default_unit_type=DocumentUnitType.BODY,
                sections=sections,
                warnings=warnings,
            )

        top_level_sections = _direct_children(body, "sec")

        if not top_level_sections and not body_has_direct_content:
            warnings.append(
                ParserWarning(
                    code="empty_body",
                    message=("The JATS body contains no usable sections or paragraph content"),
                    severity=ParserWarningSeverity.ERROR,
                    structural_locator="body",
                )
            )
            return

        for index, section in enumerate(
            top_level_sections,
            start=1,
        ):
            self._extract_section_tree(
                source=source,
                section=section,
                locator=f"body/sec[{index}]",
                level=0,
                sections=sections,
                warnings=warnings,
            )

    def _extract_section_tree(
        self,
        *,
        source: SourceCatalogueEntry,
        section: ET.Element,
        locator: str,
        level: int,
        sections: list[DocumentSection],
        warnings: list[ParserWarning],
    ) -> None:
        """Flatten a nested JATS section hierarchy in source order."""

        if level > 12:
            warnings.append(
                ParserWarning(
                    code="section_depth_exceeded",
                    message=("A section deeper than the supported level was skipped"),
                    severity=ParserWarningSeverity.ERROR,
                    structural_locator=locator,
                )
            )
            return

        section_title = _element_text(_direct_child(section, "title"))

        direct_nested_sections = _direct_children(
            section,
            "sec",
        )

        units = self._build_units(
            source=source,
            container=section,
            section_locator=locator,
            default_unit_type=DocumentUnitType.BODY,
        )

        if units:
            section_id = _stable_identifier(
                kind="section",
                source_id=source.source_id,
                locator=locator,
            )

            units = tuple(
                unit.model_copy(
                    update={
                        "parent_section_id": section_id,
                    }
                )
                for unit in units
            )

            sections.append(
                DocumentSection(
                    section_id=section_id,
                    order=len(sections),
                    level=level,
                    title=section_title or None,
                    structural_locator=locator,
                    units=units,
                )
            )
        elif not direct_nested_sections:
            warnings.append(
                ParserWarning(
                    code="empty_section",
                    message=("The JATS section contained no usable text"),
                    severity=ParserWarningSeverity.WARNING,
                    structural_locator=locator,
                )
            )

        for index, nested_section in enumerate(
            direct_nested_sections,
            start=1,
        ):
            self._extract_section_tree(
                source=source,
                section=nested_section,
                locator=f"{locator}/sec[{index}]",
                level=level + 1,
                sections=sections,
                warnings=warnings,
            )

    def _append_section(
        self,
        *,
        source: SourceCatalogueEntry,
        container: ET.Element,
        title: str | None,
        locator: str,
        level: int,
        default_unit_type: DocumentUnitType,
        sections: list[DocumentSection],
        warnings: list[ParserWarning],
    ) -> None:
        """Create and append one normalized section when text exists."""

        units = self._build_units(
            source=source,
            container=container,
            section_locator=locator,
            default_unit_type=default_unit_type,
        )

        if not units:
            warnings.append(
                ParserWarning(
                    code="empty_section",
                    message=("The JATS container contained no usable text"),
                    severity=ParserWarningSeverity.WARNING,
                    structural_locator=locator,
                )
            )
            return

        section_id = _stable_identifier(
            kind="section",
            source_id=source.source_id,
            locator=locator,
        )

        units_with_parent = tuple(
            unit.model_copy(
                update={
                    "parent_section_id": section_id,
                }
            )
            for unit in units
        )

        sections.append(
            DocumentSection(
                section_id=section_id,
                order=len(sections),
                level=level,
                title=title,
                structural_locator=locator,
                units=units_with_parent,
            )
        )

    def _build_units(
        self,
        *,
        source: SourceCatalogueEntry,
        container: ET.Element,
        section_locator: str,
        default_unit_type: DocumentUnitType,
    ) -> tuple[DocumentUnit, ...]:
        """Create ordered normalized units from one JATS container."""

        extracted = tuple(
            self._walk_content(
                element=container,
                base_locator=section_locator,
                default_unit_type=default_unit_type,
            )
        )

        units: list[DocumentUnit] = []

        for order, (
            unit_type,
            text,
            locator,
        ) in enumerate(extracted):
            unit_id = _stable_identifier(
                kind="unit",
                source_id=source.source_id,
                locator=locator,
                text=text,
            )

            units.append(
                DocumentUnit(
                    unit_id=unit_id,
                    order=order,
                    unit_type=unit_type,
                    heading=None,
                    structural_locator=locator,
                    text=text,
                    parent_section_id=None,
                )
            )

        return tuple(units)

    def _walk_content(
        self,
        *,
        element: ET.Element,
        base_locator: str,
        default_unit_type: DocumentUnitType,
    ) -> Iterator[ExtractedUnit]:
        """Yield normalized block-level units in source order."""

        sibling_counts: dict[str, int] = {}

        for child in list(element):
            name = _local_name(child.tag)
            sibling_counts[name] = sibling_counts.get(name, 0) + 1

            locator = f"{base_locator}/{name}[{sibling_counts[name]}]"

            if name == "sec":
                continue

            if name in _IGNORED_ELEMENT_NAMES:
                continue

            if name in _SKIPPED_CONTAINER_NAMES:
                continue

            if name == "p":
                text = _element_text(child)

                if text:
                    yield (
                        default_unit_type,
                        text,
                        locator,
                    )

                continue

            if name == "fig":
                caption = _direct_child(child, "caption")
                caption_text = _element_text(caption)

                if caption_text:
                    yield (
                        DocumentUnitType.CAPTION,
                        caption_text,
                        f"{locator}/caption",
                    )

                continue

            if name == "table-wrap":
                caption = _direct_child(child, "caption")
                caption_text = _element_text(caption)

                if caption_text:
                    yield (
                        DocumentUnitType.CAPTION,
                        caption_text,
                        f"{locator}/caption",
                    )

                table = _first_descendant(child, "table")
                table_text = _element_text(table)

                if table_text:
                    yield (
                        DocumentUnitType.TABLE,
                        table_text,
                        f"{locator}/table",
                    )

                continue

            if name == "fn":
                footnote_text = _element_text(child)

                if footnote_text:
                    yield (
                        DocumentUnitType.FOOTNOTE,
                        footnote_text,
                        locator,
                    )

                continue

            if name == "caption":
                caption_text = _element_text(child)

                if caption_text:
                    yield (
                        DocumentUnitType.CAPTION,
                        caption_text,
                        locator,
                    )

                continue

            yield from self._walk_content(
                element=child,
                base_locator=locator,
                default_unit_type=default_unit_type,
            )


def _local_name(tag: str) -> str:
    """Return an XML tag without its namespace."""

    return tag.rsplit("}", maxsplit=1)[-1]


def _direct_child(
    element: ET.Element,
    name: str,
) -> ET.Element | None:
    """Return the first direct child with a local tag name."""

    for child in list(element):
        if _local_name(child.tag) == name:
            return child

    return None


def _direct_children(
    element: ET.Element,
    name: str,
) -> tuple[ET.Element, ...]:
    """Return direct children matching a local tag name."""

    return tuple(child for child in list(element) if _local_name(child.tag) == name)


def _first_descendant(
    element: ET.Element | None,
    name: str,
) -> ET.Element | None:
    """Return the first descendant with a local tag name."""

    if element is None:
        return None

    for descendant in element.iter():
        if _local_name(descendant.tag) == name:
            return descendant

    return None


def _element_text(
    element: ET.Element | None,
) -> str:
    """Extract and normalize all textual content from an element."""

    if element is None:
        return ""

    text = " ".join(element.itertext())
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


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
