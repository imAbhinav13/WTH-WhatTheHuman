"""Section-aware chunker for scientific corpus sources."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from apps.api.ingestion.chunkers.base import (
    ChunkConstructionError,
    ChunkingConfig,
    SourceChunker,
)
from apps.api.models.corpus import (
    ChunkDraft,
    DocumentSection,
    DocumentUnit,
    DocumentUnitType,
    ParsedDocument,
    SourceCatalogueEntry,
)
from apps.api.models.enums import Domain, SourceType

CHUNKER_NAME: Final = "scientific"
CHUNKER_VERSION: Final = "1.0.0"

DEFAULT_CONFIG: Final = ChunkingConfig(
    minimum_tokens=120,
    target_tokens=320,
    maximum_tokens=450,
    overlap_tokens=40,
)

_INCLUDED_UNIT_TYPES: Final = frozenset(
    {
        DocumentUnitType.ABSTRACT,
        DocumentUnitType.BODY,
        DocumentUnitType.CAPTION,
        DocumentUnitType.TABLE,
        DocumentUnitType.APPENDIX,
    }
)

_SENTENCE_BOUNDARY_PATTERN: Final = re.compile(
    r"""(?x)
    (?<=[.!?])
    \s+
    (?=
        ["'"\u201c\u2018(\[]*
        [A-Z0-9]
    )
    """
)


@dataclass(frozen=True, slots=True)
class _Segment:
    """A complete or partial source unit prepared for packing."""

    unit: DocumentUnit
    text: str
    locator: str
    token_count: int


@dataclass(frozen=True, slots=True)
class _ChunkPlan:
    """Internal plan used before constructing a validated chunk."""

    segments: tuple[_Segment, ...]
    unit_type: DocumentUnitType
    text: str
    locator: str
    token_count: int


class ScientificChunker(SourceChunker):
    """Chunk scientific articles without crossing section boundaries.

    Paragraphs are combined until the target size is reached. Oversized
    paragraphs are split conservatively at sentence boundaries, with a
    limited overlap used only for those oversized source units.
    """

    def __init__(
        self,
        *,
        config: ChunkingConfig = DEFAULT_CONFIG,
    ) -> None:
        """Initialize the scientific chunker."""

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
        """Return scientific chunk-size configuration."""

        return self._config

    @property
    def supported_domains(self) -> frozenset[Domain]:
        """Return supported knowledge domains."""

        return frozenset(
            {
                Domain.SCIENCE,
            }
        )

    @property
    def supported_source_types(
        self,
    ) -> frozenset[SourceType]:
        """Return supported scientific source types."""

        return frozenset(
            {
                SourceType.PAPER,
            }
        )

    def _chunk_document(
        self,
        *,
        source: SourceCatalogueEntry,
        document: ParsedDocument,
    ) -> tuple[ChunkDraft, ...]:
        """Create section-bounded scientific chunks."""

        chunks: list[ChunkDraft] = []

        for section in document.sections:
            section_chunks = self._chunk_section(
                source=source,
                document=document,
                section=section,
            )
            chunks.extend(section_chunks)

        return tuple(chunks)

    def _chunk_section(
        self,
        *,
        source: SourceCatalogueEntry,
        document: ParsedDocument,
        section: DocumentSection,
    ) -> tuple[ChunkDraft, ...]:
        """Create chunks from eligible units in one section."""

        eligible_units = tuple(
            unit for unit in section.units if unit.unit_type in _INCLUDED_UNIT_TYPES
        )

        if not eligible_units:
            return ()

        segments: list[_Segment] = []

        for unit in eligible_units:
            segments.extend(self._segment_unit(unit))

        plans = self._pack_segments(
            segments=tuple(segments),
            section=section,
        )

        chunks: list[ChunkDraft] = []

        for plan in plans:
            source_units = _unique_units(plan.segments)

            chunks.append(
                self.build_chunk(
                    source=source,
                    document=document,
                    section=section,
                    units=source_units,
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
        unit: DocumentUnit,
    ) -> tuple[_Segment, ...]:
        """Split only source units exceeding the maximum size."""

        text = self.normalize_chunk_text(unit.text)
        token_count = self.count_tokens(text)

        if token_count <= self.config.maximum_tokens:
            return (
                _Segment(
                    unit=unit,
                    text=text,
                    locator=unit.structural_locator,
                    token_count=token_count,
                ),
            )

        parts = self._split_oversized_text(text)

        return tuple(
            _Segment(
                unit=unit,
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
        """Split oversized text at sentence boundaries when possible."""

        sentences = tuple(
            sentence.strip()
            for sentence in _SENTENCE_BOUNDARY_PATTERN.split(text)
            if sentence.strip()
        )

        if len(sentences) <= 1:
            return self._split_by_words(text)

        parts: list[str] = []
        current_sentences: list[str] = []

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            if sentence_tokens > self.config.maximum_tokens:
                if current_sentences:
                    parts.append(self.normalize_chunk_text(" ".join(current_sentences)))
                    current_sentences = []

                parts.extend(self._split_by_words(sentence))
                continue

            candidate = self.normalize_chunk_text(
                " ".join(
                    (
                        *current_sentences,
                        sentence,
                    )
                )
            )

            if current_sentences and self.count_tokens(candidate) > self.config.maximum_tokens:
                completed_part = self.normalize_chunk_text(" ".join(current_sentences))
                parts.append(completed_part)

                current_sentences = self._overlap_sentences(current_sentences)

                candidate = self.normalize_chunk_text(
                    " ".join(
                        (
                            *current_sentences,
                            sentence,
                        )
                    )
                )

                if self.count_tokens(candidate) > self.config.maximum_tokens:
                    current_sentences = [sentence]
                else:
                    current_sentences.append(sentence)
            else:
                current_sentences.append(sentence)

        if current_sentences:
            parts.append(self.normalize_chunk_text(" ".join(current_sentences)))

        normalized_parts = tuple(part for part in parts if part)

        if not normalized_parts:
            raise ChunkConstructionError("Unable to split oversized scientific text")

        for part in normalized_parts:
            if self.count_tokens(part) > self.config.maximum_tokens:
                raise ChunkConstructionError("Scientific text splitting produced an oversized part")

        return normalized_parts

    def _split_by_words(
        self,
        text: str,
    ) -> tuple[str, ...]:
        """Split a sentence that exceeds the maximum token limit."""

        words = text.split()

        if not words:
            raise ChunkConstructionError("Cannot split empty scientific text")

        parts: list[str] = []
        start_index = 0

        while start_index < len(words):
            end_index = start_index + 1

            while end_index <= len(words):
                candidate = " ".join(words[start_index:end_index])

                if self.count_tokens(candidate) > self.config.maximum_tokens:
                    end_index -= 1
                    break

                end_index += 1

            if end_index > len(words):
                end_index = len(words)

            if end_index <= start_index:
                raise ChunkConstructionError(
                    "Unable to split a scientific sentence within the configured maximum"
                )

            part = self.normalize_chunk_text(" ".join(words[start_index:end_index]))
            parts.append(part)

            if end_index == len(words):
                break

            next_start = self._word_overlap_start(
                words=words,
                chunk_start=start_index,
                chunk_end=end_index,
            )

            if next_start <= start_index:
                next_start = end_index

            start_index = next_start

        return tuple(parts)

    def _word_overlap_start(
        self,
        *,
        words: list[str],
        chunk_start: int,
        chunk_end: int,
    ) -> int:
        """Find a safe trailing word overlap for a split chunk."""

        if self.config.overlap_tokens == 0:
            return chunk_end

        overlap_start = chunk_end

        while overlap_start > chunk_start:
            candidate_start = overlap_start - 1
            overlap_text = " ".join(words[candidate_start:chunk_end])

            if self.count_tokens(overlap_text) > self.config.overlap_tokens:
                break

            overlap_start = candidate_start

        if overlap_start == chunk_start:
            return chunk_end

        return overlap_start

    def _overlap_sentences(
        self,
        sentences: list[str],
    ) -> list[str]:
        """Return trailing sentences within the overlap budget."""

        if self.config.overlap_tokens == 0:
            return []

        selected: list[str] = []

        for sentence in reversed(sentences):
            candidate = [
                sentence,
                *selected,
            ]

            candidate_text = " ".join(candidate)

            if self.count_tokens(candidate_text) > self.config.overlap_tokens:
                break

            selected = candidate

        return selected

    def _pack_segments(
        self,
        *,
        segments: tuple[_Segment, ...],
        section: DocumentSection,
    ) -> tuple[_ChunkPlan, ...]:
        """Pack adjacent same-type segments into target-sized chunks."""

        if not segments:
            return ()

        plans: list[_ChunkPlan] = []
        current: list[_Segment] = []

        for segment in segments:
            if current and current[0].unit.unit_type is not segment.unit.unit_type:
                plans.append(
                    self._create_plan(
                        segments=tuple(current),
                        section=section,
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
                        segments=tuple(current),
                        section=section,
                    )
                )
                current = [segment]
            else:
                current.append(segment)

            if current:
                current_text = _join_segment_text(tuple(current))

                if self.count_tokens(current_text) >= self.config.target_tokens:
                    plans.append(
                        self._create_plan(
                            segments=tuple(current),
                            section=section,
                        )
                    )
                    current = []

        if current:
            plans.append(
                self._create_plan(
                    segments=tuple(current),
                    section=section,
                )
            )

        return self._rebalance_short_plans(
            plans=tuple(plans),
            section=section,
        )

    def _rebalance_short_plans(
        self,
        *,
        plans: tuple[_ChunkPlan, ...],
        section: DocumentSection,
    ) -> tuple[_ChunkPlan, ...]:
        """Merge undersized neighboring plans when safely possible."""

        mutable_plans = list(plans)
        index = 0

        while index < len(mutable_plans):
            plan = mutable_plans[index]

            if plan.token_count >= self.config.minimum_tokens:
                index += 1
                continue

            if index > 0:
                previous = mutable_plans[index - 1]

                if self._plans_can_merge(
                    previous,
                    plan,
                ):
                    mutable_plans[index - 1] = self._create_plan(
                        segments=(
                            *previous.segments,
                            *plan.segments,
                        ),
                        section=section,
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
                        segments=(
                            *plan.segments,
                            *following.segments,
                        ),
                        section=section,
                    )
                    mutable_plans.pop(index + 1)
                    continue

            # A short scientific fragment that cannot be merged safely
            # is omitted rather than crossing a section or semantic-type
            # boundary merely to satisfy the minimum size.
            mutable_plans.pop(index)

        return tuple(mutable_plans)

    def _plans_can_merge(
        self,
        first: _ChunkPlan,
        second: _ChunkPlan,
    ) -> bool:
        """Return whether two neighboring plans can be merged."""

        if first.unit_type is not second.unit_type:
            return False

        combined_text = _join_segment_text(
            (
                *first.segments,
                *second.segments,
            )
        )

        return self.count_tokens(combined_text) <= self.config.maximum_tokens

    def _create_plan(
        self,
        *,
        segments: tuple[_Segment, ...],
        section: DocumentSection,
    ) -> _ChunkPlan:
        """Create one internal chunk plan."""

        if not segments:
            raise ChunkConstructionError("Cannot create an empty scientific chunk plan")

        unit_types = {segment.unit.unit_type for segment in segments}

        if len(unit_types) != 1:
            raise ChunkConstructionError("Scientific chunk plans cannot mix unit types")

        text = _join_segment_text(segments)
        token_count = self.count_tokens(text)

        if token_count > self.config.maximum_tokens:
            raise ChunkConstructionError("Scientific chunk plan exceeds maximum size")

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
        """Create a scientific source citation."""

        return (
            f"{source.author} ({source.publication_year}), “{source.title},” {structural_locator}."
        )


def _join_segment_text(
    segments: tuple[_Segment, ...],
) -> str:
    """Join scientific segments while preserving paragraph breaks."""

    return "\n\n".join(segment.text for segment in segments).strip()


def _unique_units(
    segments: tuple[_Segment, ...],
) -> tuple[DocumentUnit, ...]:
    """Return source units in first-occurrence order."""

    units: list[DocumentUnit] = []
    seen_unit_ids: set[str] = set()

    for segment in segments:
        unit = segment.unit

        if unit.unit_id in seen_unit_ids:
            continue

        seen_unit_ids.add(unit.unit_id)
        units.append(unit)

    return tuple(units)
