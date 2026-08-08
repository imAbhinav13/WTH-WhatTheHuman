"""Structure-preserving chunker for Advaita and Samkhya texts."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Final

from apps.api.ingestion.chunkers.base import ChunkConstructionError, ChunkingConfig, SourceChunker
from apps.api.models.corpus import (
    ChunkDraft,
    DocumentSection,
    DocumentUnit,
    DocumentUnitType,
    ParsedDocument,
    SourceCatalogueEntry,
)
from apps.api.models.enums import Domain, SourceType

CHUNKER_NAME: Final = "classical_text"
CHUNKER_VERSION: Final = "1.0.0"

DEFAULT_CONFIG: Final = ChunkingConfig(
    minimum_tokens=1,
    target_tokens=180,
    maximum_tokens=300,
    overlap_tokens=0,
)

_INCLUDED_UNIT_TYPES: Final = frozenset(
    {
        DocumentUnitType.ROOT_TEXT,
        DocumentUnitType.COMMENTARY,
        DocumentUnitType.TRANSLATOR_NOTE,
        DocumentUnitType.EDITOR_NOTE,
        DocumentUnitType.FOOTNOTE,
        DocumentUnitType.APPENDIX,
        DocumentUnitType.BODY,
    }
)

_SENTENCE_BOUNDARY_PATTERN: Final = re.compile(
    r"""(?x)
    (?<=[.!?।॥])
    \s+
    (?=
        ["'\u201c\u2018(\[]*
        [A-Z0-9Ā-ž]
    )
    """
)

_PARAGRAPH_BOUNDARY_PATTERN: Final = re.compile(r"\n\s*\n+")

_VERSE_NUMBER_PATTERN: Final = re.compile(
    r"""(?ix)
    \b
    (?:
        karika
        | verse
        | sutra
        | mantra
        | section
        | chapter
    )
    \s*
    (?P<number>[0-9]+(?:[.-][0-9]+)*)
    \b
    """
)


@dataclass(frozen=True, slots=True)
class _Segment:
    """Complete or partial classical-text unit prepared for packing."""

    unit: DocumentUnit
    unit_type: DocumentUnitType
    text: str
    locator: str
    token_count: int


@dataclass(frozen=True, slots=True)
class _ChunkPlan:
    """Internal classical-text chunk plan."""

    segments: tuple[_Segment, ...]
    unit_type: DocumentUnitType
    text: str
    locator: str
    token_count: int


class ClassicalTextChunker(SourceChunker):
    """Chunk Advaita and Samkhya sources along textual boundaries.

    Root text, commentary, translator notes, editor notes, and
    footnotes are never mixed in one chunk. Short canonical units may
    remain below the nominal target size when combining them would
    weaken citation precision or merge distinct textual roles.
    """

    def __init__(
        self,
        *,
        config: ChunkingConfig = DEFAULT_CONFIG,
    ) -> None:
        """Initialize the classical-text chunker."""

        super().__init__()
        self._config = config

    @property
    def chunker_name(self) -> str:
        """Return the stable chunker implementation name."""

        return CHUNKER_NAME

    @property
    def chunker_version(self) -> str:
        """Return the chunker version."""

        return CHUNKER_VERSION

    @property
    def config(self) -> ChunkingConfig:
        """Return classical-text chunk-size configuration."""

        return self._config

    @property
    def supported_domains(self) -> frozenset[Domain]:
        """Return the supported knowledge domains."""

        return frozenset(
            {
                Domain.ADVAITA,
                Domain.SAMKHYA,
            }
        )

    @property
    def supported_source_types(
        self,
    ) -> frozenset[SourceType]:
        """Return supported classical source types."""

        return frozenset(
            {
                SourceType.PRIMARY_TEXT,
                SourceType.COMMENTARY,
            }
        )

    def _chunk_document(
        self,
        *,
        source: SourceCatalogueEntry,
        document: ParsedDocument,
    ) -> tuple[ChunkDraft, ...]:
        """Create section-bounded classical-text chunks."""

        chunks: list[ChunkDraft] = []

        for section in document.sections:
            chunks.extend(
                self._chunk_section(
                    source=source,
                    document=document,
                    section=section,
                )
            )

        return tuple(chunks)

    def _chunk_section(
        self,
        *,
        source: SourceCatalogueEntry,
        document: ParsedDocument,
        section: DocumentSection,
    ) -> tuple[ChunkDraft, ...]:
        """Chunk eligible units within one textual section."""

        eligible_units = tuple(
            unit for unit in section.units if unit.unit_type in _INCLUDED_UNIT_TYPES
        )

        if not eligible_units:
            return ()

        segments: list[_Segment] = []

        for unit in eligible_units:
            segments.extend(
                self._segment_unit(
                    source=source,
                    unit=unit,
                )
            )

        plans = self._pack_segments(
            section=section,
            segments=tuple(segments),
        )

        chunks: list[ChunkDraft] = []

        for plan in plans:
            units = _unique_units(plan.segments)

            chunks.append(
                self.build_chunk(
                    source=source,
                    document=document,
                    section=section,
                    units=units,
                    text=plan.text,
                    unit_type=plan.unit_type,
                    structural_locator=plan.locator,
                    citation_text=self.create_citation_text(
                        source=source,
                        structural_locator=plan.locator,
                    ),
                )
            )

        return tuple(chunks)

    def _segment_unit(
        self,
        *,
        source: SourceCatalogueEntry,
        unit: DocumentUnit,
    ) -> tuple[_Segment, ...]:
        """Preserve a unit unless it exceeds the maximum chunk size."""

        normalized_type = _normalize_unit_type(
            source_type=source.source_type,
            unit_type=unit.unit_type,
        )

        text = self.normalize_chunk_text(unit.text)
        token_count = self.count_tokens(text)

        if token_count <= self.config.maximum_tokens:
            return (
                _Segment(
                    unit=unit,
                    unit_type=normalized_type,
                    text=text,
                    locator=unit.structural_locator,
                    token_count=token_count,
                ),
            )

        parts = self._split_oversized_text(text)

        return tuple(
            _Segment(
                unit=unit,
                unit_type=normalized_type,
                text=part,
                locator=(f"{unit.structural_locator}#part-{index}"),
                token_count=self.count_tokens(part),
            )
            for index, part in enumerate(
                parts,
                start=1,
            )
        )

    def _split_oversized_text(
        self,
        text: str,
    ) -> tuple[str, ...]:
        """Split oversized classical text without adding overlap."""

        paragraphs = tuple(
            self.normalize_chunk_text(paragraph)
            for paragraph in _PARAGRAPH_BOUNDARY_PATTERN.split(text)
            if paragraph.strip()
        )

        if len(paragraphs) > 1:
            parts = self._pack_text_fragments(paragraphs)

            if all(self.count_tokens(part) <= self.config.maximum_tokens for part in parts):
                return parts

        sentences = tuple(
            sentence.strip()
            for sentence in _SENTENCE_BOUNDARY_PATTERN.split(text)
            if sentence.strip()
        )

        if len(sentences) > 1:
            parts = self._pack_text_fragments(sentences)

            if all(self.count_tokens(part) <= self.config.maximum_tokens for part in parts):
                return parts

        return self._split_by_words(text)

    def _pack_text_fragments(
        self,
        fragments: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Pack textual fragments without exceeding the maximum."""

        parts: list[str] = []
        current: list[str] = []

        for fragment in fragments:
            if self.count_tokens(fragment) > self.config.maximum_tokens:
                if current:
                    parts.append(self.normalize_chunk_text("\n\n".join(current)))
                    current = []

                parts.extend(self._split_by_words(fragment))
                continue

            candidate = self.normalize_chunk_text(
                "\n\n".join(
                    (
                        *current,
                        fragment,
                    )
                )
            )

            if current and self.count_tokens(candidate) > self.config.maximum_tokens:
                parts.append(self.normalize_chunk_text("\n\n".join(current)))
                current = [fragment]
            else:
                current.append(fragment)

        if current:
            parts.append(self.normalize_chunk_text("\n\n".join(current)))

        return tuple(part for part in parts if part)

    def _split_by_words(
        self,
        text: str,
    ) -> tuple[str, ...]:
        """Apply a final hard split when no structural boundary exists."""

        words = text.split()

        if not words:
            raise ChunkConstructionError("Cannot split empty classical text")

        parts: list[str] = []
        current: list[str] = []

        for word in words:
            candidate = " ".join(
                (
                    *current,
                    word,
                )
            )

            if current and self.count_tokens(candidate) > self.config.maximum_tokens:
                parts.append(self.normalize_chunk_text(" ".join(current)))
                current = [word]
            else:
                current.append(word)

        if current:
            parts.append(self.normalize_chunk_text(" ".join(current)))

        for part in parts:
            if self.count_tokens(part) > self.config.maximum_tokens:
                raise ChunkConstructionError(
                    "A classical-text token could not be split within the configured maximum"
                )

        return tuple(parts)

    def _pack_segments(
        self,
        *,
        section: DocumentSection,
        segments: tuple[_Segment, ...],
    ) -> tuple[_ChunkPlan, ...]:
        """Combine adjacent compatible units toward the target size."""

        if not segments:
            return ()

        plans: list[_ChunkPlan] = []
        current: list[_Segment] = []

        for segment in segments:
            if current and not _segments_are_compatible(
                current[-1],
                segment,
            ):
                plans.append(
                    self._create_plan(
                        section=section,
                        segments=tuple(current),
                    )
                )
                current = []

            candidate_segments = (
                *current,
                segment,
            )
            candidate_text = _join_segment_text(candidate_segments)
            candidate_tokens = self.count_tokens(candidate_text)

            if current and candidate_tokens > self.config.maximum_tokens:
                plans.append(
                    self._create_plan(
                        section=section,
                        segments=tuple(current),
                    )
                )
                current = [segment]
            else:
                current.append(segment)

            if (
                current
                and self.count_tokens(_join_segment_text(tuple(current)))
                >= self.config.target_tokens
            ):
                plans.append(
                    self._create_plan(
                        section=section,
                        segments=tuple(current),
                    )
                )
                current = []

        if current:
            plans.append(
                self._create_plan(
                    section=section,
                    segments=tuple(current),
                )
            )

        return self._rebalance_short_plans(
            section=section,
            plans=tuple(plans),
        )

    def _rebalance_short_plans(
        self,
        *,
        section: DocumentSection,
        plans: tuple[_ChunkPlan, ...],
    ) -> tuple[_ChunkPlan, ...]:
        """Merge short neighboring plans only when semantically safe."""

        mutable_plans = list(plans)
        index = 0

        while index < len(mutable_plans):
            plan = mutable_plans[index]

            if plan.token_count >= self.config.target_tokens // 2:
                index += 1
                continue

            if index > 0:
                previous = mutable_plans[index - 1]

                if self._plans_can_merge(
                    previous,
                    plan,
                ):
                    mutable_plans[index - 1] = self._create_plan(
                        section=section,
                        segments=(
                            *previous.segments,
                            *plan.segments,
                        ),
                    )
                    mutable_plans.pop(index)
                    continue

            if index + 1 < len(mutable_plans):
                following = mutable_plans[index + 1]

                if self._plans_can_merge(
                    plan,
                    following,
                ):
                    mutable_plans[index] = self._create_plan(
                        section=section,
                        segments=(
                            *plan.segments,
                            *following.segments,
                        ),
                    )
                    mutable_plans.pop(index + 1)
                    continue

            # Preserve the short passage as an independently citable
            # unit rather than joining unlike textual roles.
            index += 1

        return tuple(mutable_plans)

    def _plans_can_merge(
        self,
        first: _ChunkPlan,
        second: _ChunkPlan,
    ) -> bool:
        """Return whether neighboring plans can be combined safely."""

        if first.unit_type is not second.unit_type:
            return False

        combined_segments = (
            *first.segments,
            *second.segments,
        )

        if not all(
            _segments_are_compatible(
                left,
                right,
            )
            for left, right in pairwise(combined_segments)
        ):
            return False

        combined_text = _join_segment_text(combined_segments)

        return self.count_tokens(combined_text) <= self.config.maximum_tokens

    def _create_plan(
        self,
        *,
        section: DocumentSection,
        segments: tuple[_Segment, ...],
    ) -> _ChunkPlan:
        """Create one validated internal chunk plan."""

        if not segments:
            raise ChunkConstructionError("Cannot create an empty classical-text plan")

        unit_types = {segment.unit_type for segment in segments}

        if len(unit_types) != 1:
            raise ChunkConstructionError(
                "Classical-text chunks cannot mix root text, commentary, or notes"
            )

        text = _join_segment_text(segments)
        token_count = self.count_tokens(text)

        if token_count > self.config.maximum_tokens:
            raise ChunkConstructionError("Classical-text chunk plan exceeds maximum size")

        if len(segments) == 1:
            locator = segments[0].locator
        else:
            locator = f"{section.structural_locator}; {segments[0].locator}..{segments[-1].locator}"

        return _ChunkPlan(
            segments=segments,
            unit_type=next(iter(unit_types)),
            text=text,
            locator=locator,
            token_count=token_count,
        )

    def create_citation_text(
        self,
        *,
        source: SourceCatalogueEntry,
        structural_locator: str,
    ) -> str:
        """Create a classical-text citation with edition roles."""

        contributors: list[str] = []

        if source.author:
            contributors.append(source.author)

        if source.translator is not None:
            contributors.append(f"trans. {source.translator}")

        if source.editor is not None:
            contributors.append(f"ed. {source.editor}")

        contributor_text = "; ".join(contributors)

        citation = (
            f"{source.title}, {contributor_text} ({source.publication_year}), {structural_locator}."
        )

        return re.sub(
            r"\s+",
            " ",
            citation,
        ).strip()

    def combine_locators(
        self,
        *,
        section: DocumentSection,
        units: Sequence[DocumentUnit],
    ) -> str:
        """Create a classical locator and retain visible verse labels."""

        base_locator = super().combine_locators(
            section=section,
            units=units,
        )

        labels = _extract_visible_labels(
            section=section,
            units=tuple(units),
        )

        if not labels:
            return base_locator

        return f"{base_locator}; {', '.join(labels)}"


def _normalize_unit_type(
    *,
    source_type: SourceType,
    unit_type: DocumentUnitType,
) -> DocumentUnitType:
    """Normalize generic body units using catalogue source type."""

    if unit_type is not DocumentUnitType.BODY:
        return unit_type

    if source_type is SourceType.PRIMARY_TEXT:
        return DocumentUnitType.ROOT_TEXT

    if source_type is SourceType.COMMENTARY:
        return DocumentUnitType.COMMENTARY

    return unit_type


def _segments_are_compatible(
    first: _Segment,
    second: _Segment,
) -> bool:
    """Return whether two adjacent segments may share a chunk."""

    return first.unit_type is second.unit_type


def _join_segment_text(
    segments: tuple[_Segment, ...],
) -> str:
    """Join classical segments while preserving textual separation."""

    return "\n\n".join(segment.text for segment in segments).strip()


def _unique_units(
    segments: tuple[_Segment, ...],
) -> tuple[DocumentUnit, ...]:
    """Return referenced source units in first-occurrence order."""

    units: list[DocumentUnit] = []
    seen_unit_ids: set[str] = set()

    for segment in segments:
        if segment.unit.unit_id in seen_unit_ids:
            continue

        seen_unit_ids.add(segment.unit.unit_id)
        units.append(segment.unit)

    return tuple(units)


def _extract_visible_labels(
    *,
    section: DocumentSection,
    units: tuple[DocumentUnit, ...],
) -> tuple[str, ...]:
    """Extract explicit verse, sutra, or karika labels for citations."""

    candidate_texts = [
        section.title or "",
        *(unit.heading or "" for unit in units),
    ]

    labels: list[str] = []
    seen: set[str] = set()

    for text in candidate_texts:
        for match in _VERSE_NUMBER_PATTERN.finditer(text):
            label = match.group(0).strip()

            normalized = label.casefold()

            if normalized in seen:
                continue

            seen.add(normalized)
            labels.append(label)

    return tuple(labels)
