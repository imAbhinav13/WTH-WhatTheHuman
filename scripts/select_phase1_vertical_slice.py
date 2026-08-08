from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import yaml

LOGGER = logging.getLogger("wth.phase1.select_phase1_vertical_slice")

SELECTOR_VERSION: Final = "1.1.0"
DEFAULT_CHUNKS_ROOT: Final = Path("artifacts/phase1/chunks")
DEFAULT_CANDIDATE_MANIFEST: Final = Path(
    "artifacts/phase1/candidate/candidate_corpus_manifest.json"
)
DEFAULT_SCOPE_FILE: Final = Path("docs/corpus/phase1_section_scope.yaml")
DEFAULT_STRUCTURE_REPORT: Final = Path("artifacts/phase1/scope/source_structure_report.csv")
DEFAULT_OUTPUT_DIRECTORY: Final = Path("artifacts/phase1/selection")
DEFAULT_CORPUS_VERSION: Final = "phase1_candidate_corpus_v1"
DEFAULT_EXPECTED_CHUNK_COUNT: Final = 7469

CANDIDATES_FILENAME: Final = "phase1_selection_candidates.jsonl"
EXCLUDED_FILENAME: Final = "phase1_excluded_candidates.jsonl"
MANIFEST_FILENAME: Final = "phase1_selection_manifest.json"

CHUNK_ID_PATHS: Final = (("chunk_id",), ("metadata", "chunk_id"), ("id",))
TEXT_PATHS: Final = (
    ("text",),
    ("chunk_text",),
    ("content",),
    ("page_content",),
    ("metadata", "text"),
)
SOURCE_ID_PATHS: Final = (
    ("source_id",),
    ("metadata", "source_id"),
    ("source", "source_id"),
    ("source", "id"),
)
DOMAIN_PATHS: Final = (("domain",), ("metadata", "domain"), ("source", "domain"))
SOURCE_TITLE_PATHS: Final = (
    ("source_title",),
    ("metadata", "source_title"),
    ("source", "title"),
)
SECTION_ID_PATHS: Final = (
    ("section_id",),
    ("metadata", "section_id"),
    ("section", "section_id"),
    ("section", "id"),
)
SECTION_TITLE_PATHS: Final = (
    ("section_title",),
    ("heading",),
    ("metadata", "section_title"),
    ("section", "section_title"),
    ("section", "title"),
)
PARENT_SECTION_PATHS: Final = (
    ("parent_section",),
    ("parent_section_title",),
    ("metadata", "parent_section"),
    ("section", "parent_section"),
    ("section", "parent_title"),
)
UNIT_TYPE_PATHS: Final = (
    ("unit_type",),
    ("metadata", "unit_type"),
    ("section", "unit_type"),
)
STRUCTURAL_LOCATOR_PATHS: Final = (
    ("structural_locator",),
    ("locator",),
    ("citation_locator",),
    ("metadata", "structural_locator"),
    ("metadata", "locator"),
)
TOKEN_COUNT_PATHS: Final = (("token_count",), ("metadata", "token_count"))
WARNING_PATHS: Final = (
    ("warnings",),
    ("parser_warnings",),
    ("metadata", "warnings"),
    ("metadata", "parser_warnings"),
)
CHECKSUM_PATHS: Final = (
    ("content_checksum",),
    ("text_checksum",),
    ("checksum",),
    ("metadata", "content_checksum"),
)


class SelectionError(RuntimeError):
    """Raised when deterministic Phase 1 selection cannot complete safely."""


@dataclass(frozen=True)
class SectionInfo:
    source_id: str
    section_id: str
    section_title: str
    parent_section: str
    structural_locator: str
    unit_type: str
    proposed_action: str
    parser_warning_count: int
    report_ocr_noise_score: float | None


@dataclass(frozen=True)
class ConceptRule:
    concept_id: str
    positive_terms: tuple[str, ...]
    contextual_terms: tuple[str, ...]
    exclusion_terms: tuple[str, ...]
    minimum_positive_term_matches: int
    require_substantive_context: bool


@dataclass(frozen=True)
class SourceIncludeRule:
    rule_id: str
    patterns: tuple[str, ...]
    concepts: tuple[str, ...]


@dataclass(frozen=True)
class SourceExcludeRule:
    rule_id: str
    patterns: tuple[str, ...]
    rationale: str
    deferred_concept_family: str | None


@dataclass(frozen=True)
class SourceRule:
    source_id: str
    source_title: str
    domain: str
    scope_status: str
    structural_strategy: str
    structure_assessment: str
    preprocessing_requirements: tuple[str, ...]
    include_rules: tuple[SourceIncludeRule, ...]
    exclude_rules: tuple[SourceExcludeRule, ...]
    hard_negative_targets: tuple[str, ...]
    maximum_pre_review_chunks: int
    maximum_chunks_per_section: int


@dataclass(frozen=True)
class HardNegativeRule:
    category: str
    target_concept: str
    positive_terms: tuple[str, ...]
    exclusion_terms: tuple[str, ...]
    required_domain_contrast: tuple[str, ...]


@dataclass(frozen=True)
class GlobalExclusionRule:
    rule_id: str
    category: str
    match_fields: tuple[str, ...]
    patterns: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class ScopeConfiguration:
    scope_version: str
    corpus_version: str
    status: str
    concepts: tuple[str, ...]
    domains: tuple[str, ...]
    target_minimum: int
    target_maximum: int
    domain_weights: dict[str, float]
    domain_minimums: dict[str, int]
    domain_maximums: dict[str, int]
    hard_negative_minimum: int
    hard_negative_target: int
    hard_negative_maximum: int
    hard_negative_total_cap: int
    hard_negative_category_caps: dict[str, int]
    minimum_alphabetic_ratio: float
    maximum_ocr_noise_score: float
    concept_rules: dict[str, ConceptRule]
    source_rules: dict[str, SourceRule]
    hard_negative_rules: dict[str, HardNegativeRule]
    global_exclusion_rules: tuple[GlobalExclusionRule, ...]
    methodological_constraints: dict[str, object]


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    source_id: str
    domain: str
    source_title: str
    section_id: str
    section_title: str
    parent_section: str
    unit_type: str
    structural_locator: str
    text: str
    token_count: int
    warnings: tuple[str, ...]
    content_checksum: str
    origin_path: str


@dataclass(frozen=True)
class QualityMetrics:
    alphabetic_ratio: float
    digit_ratio: float
    symbol_ratio: float
    single_character_token_ratio: float
    mixed_script_token_ratio: float
    alphanumeric_intrusion_ratio: float
    replacement_character_ratio: float
    ocr_noise_score: float
    severe_ocr_noise: bool


@dataclass
class Evaluation:
    chunk: ChunkRecord
    section_info: SectionInfo | None
    quality: QualityMetrics
    decision: str = "pending"
    selection_class: str = "ineligible"
    active_concepts: tuple[str, ...] = ()
    concept_scores: dict[str, float] = field(default_factory=dict)
    concept_positive_hits: dict[str, tuple[str, ...]] = field(default_factory=dict)
    concept_context_hits: dict[str, tuple[str, ...]] = field(default_factory=dict)
    concept_exclusion_hits: dict[str, tuple[str, ...]] = field(default_factory=dict)
    source_include_rule_ids: tuple[str, ...] = ()
    source_include_pattern_hits: dict[str, tuple[str, ...]] = field(default_factory=dict)
    source_exclude_rule_ids: tuple[str, ...] = ()
    source_exclude_pattern_hits: dict[str, tuple[str, ...]] = field(default_factory=dict)
    hard_negative_categories: tuple[str, ...] = ()
    hard_negative_targets: tuple[str, ...] = ()
    global_exclusion_rule_ids: tuple[str, ...] = ()
    exclusion_reasons: list[str] = field(default_factory=list)
    review_flags: list[str] = field(default_factory=list)
    score: float = 0.0
    selection_rank: int | None = None
    tie_breaker: str = ""


@dataclass
class SelectionState:
    total: int = 0
    by_domain: Counter[str] = field(default_factory=Counter)
    by_source: Counter[str] = field(default_factory=Counter)
    by_section: Counter[tuple[str, str]] = field(default_factory=Counter)
    by_hard_negative_category: Counter[str] = field(default_factory=Counter)
    hard_negative_chunks: int = 0
    selected_ids: set[str] = field(default_factory=set)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a transparent, embedding-independent Phase 1 vertical-slice review corpus."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--chunks-root", type=Path, default=DEFAULT_CHUNKS_ROOT)
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_CANDIDATE_MANIFEST)
    parser.add_argument("--scope-file", type=Path, default=DEFAULT_SCOPE_FILE)
    parser.add_argument("--structure-report", type=Path, default=DEFAULT_STRUCTURE_REPORT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--corpus-version", default=DEFAULT_CORPUS_VERSION)
    parser.add_argument("--expected-chunk-count", type=int, default=DEFAULT_EXPECTED_CHUNK_COUNT)
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help=(
            "Desired pre-review count; defaults to the midpoint of the approved "
            "scope minimum and maximum."
        ),
    )
    parser.add_argument(
        "--allow-below-target",
        action="store_true",
        help="Preserve outputs even when selection is below the approved minimum.",
    )
    parser.add_argument("--replace", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def resolve_from_project(project_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise SelectionError(f"Required file does not exist: {path}")


def require_directory(path: Path) -> None:
    if not path.is_dir():
        raise SelectionError(f"Required directory does not exist: {path}")


def require_mapping(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise SelectionError(f"{description} must be an object")
    result: dict[str, object] = {}
    for raw_key, nested_value in value.items():
        if not isinstance(raw_key, str):
            raise SelectionError(f"{description} contains a non-string key")
        result[raw_key] = nested_value
    return result


def require_list(value: object, description: str) -> list[object]:
    if not isinstance(value, list):
        raise SelectionError(f"{description} must be a list")
    return list(value)


def optional_string(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def require_string(value: object, description: str) -> str:
    normalized = optional_string(value)
    if not normalized:
        raise SelectionError(f"{description} must be a non-empty string")
    return normalized


def optional_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def require_positive_int(value: object, description: str) -> int:
    result = optional_int(value, -1)
    if result <= 0:
        raise SelectionError(f"{description} must be a positive integer")
    return result


def optional_float(value: object, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def require_string_tuple(
    value: object,
    description: str,
    *,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if value is None and allow_empty:
        return ()
    values = require_list(value, description)
    normalized = tuple(require_string(item, f"{description} item") for item in values)
    if not normalized and not allow_empty:
        raise SelectionError(f"{description} must not be empty")
    return normalized


def load_json_value(path: Path) -> object:
    try:
        loaded: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SelectionError(f"Invalid JSON in {path}: {exc}") from exc
    return loaded


def load_yaml_mapping(path: Path) -> dict[str, object]:
    try:
        loaded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SelectionError(f"Invalid YAML in {path}: {exc}") from exc
    return require_mapping(loaded, f"YAML document {path}")


def get_path(mapping: Mapping[str, object], path: Sequence[str]) -> object | None:
    current: object = mapping
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def first_path_value(
    mapping: Mapping[str, object], paths: Sequence[Sequence[str]]
) -> object | None:
    for path in paths:
        value = get_path(mapping, path)
        if value is not None:
            return value
    return None


def find_first_key(value: object, target_key: str) -> object | None:
    if isinstance(value, Mapping):
        mapping = require_mapping(value, "recursive mapping")
        for key, nested_value in mapping.items():
            if key.casefold() == target_key.casefold():
                return nested_value
        for nested_value in mapping.values():
            result = find_first_key(nested_value, target_key)
            if result is not None:
                return result
    if isinstance(value, list | tuple):
        for nested_value in value:
            result = find_first_key(nested_value, target_key)
            if result is not None:
                return result
    return None


def normalize_term(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


class TermMatcher:
    def __init__(self) -> None:
        self._cache: dict[str, re.Pattern[str]] = {}

    def contains(self, normalized_text: str, term: str) -> bool:
        normalized_term = normalize_term(term)
        if not normalized_term:
            return False
        pattern = self._cache.get(normalized_term)
        if pattern is None:
            escaped = re.escape(normalized_term).replace(r"\ ", r"\s+")
            pattern = re.compile(rf"(?<!\w){escaped}(?!\w)", flags=re.UNICODE)
            self._cache[normalized_term] = pattern
        return pattern.search(normalized_text) is not None

    def matched_terms(self, normalized_text: str, terms: Iterable[str]) -> tuple[str, ...]:
        return tuple(term for term in terms if self.contains(normalized_text, term))


def parse_concept_rules(scope: Mapping[str, object]) -> dict[str, ConceptRule]:
    raw_rules = require_mapping(scope.get("concept_scope"), "concept_scope")
    result: dict[str, ConceptRule] = {}
    for concept_id, raw_rule in raw_rules.items():
        rule = require_mapping(raw_rule, f"concept_scope.{concept_id}")
        result[concept_id] = ConceptRule(
            concept_id=concept_id,
            positive_terms=require_string_tuple(
                rule.get("positive_terms"),
                f"{concept_id}.positive_terms",
                allow_empty=False,
            ),
            contextual_terms=require_string_tuple(
                rule.get("contextual_terms"), f"{concept_id}.contextual_terms"
            ),
            exclusion_terms=require_string_tuple(
                rule.get("exclusion_terms"), f"{concept_id}.exclusion_terms"
            ),
            minimum_positive_term_matches=require_positive_int(
                rule.get("minimum_positive_term_matches", 1),
                f"{concept_id}.minimum_positive_term_matches",
            ),
            require_substantive_context=bool(rule.get("require_substantive_context", True)),
        )
    return result


def parse_source_rules(scope: Mapping[str, object]) -> dict[str, SourceRule]:
    raw_source_rules = require_list(scope.get("source_rules"), "source_rules")
    result: dict[str, SourceRule] = {}
    for index, raw_source_rule in enumerate(raw_source_rules, start=1):
        mapping = require_mapping(raw_source_rule, f"source_rules[{index}]")
        source_id = require_string(mapping.get("source_id"), f"source_rules[{index}].source_id")
        if source_id in result:
            raise SelectionError(f"Duplicate source rule: {source_id}")

        include_rules: list[SourceIncludeRule] = []
        raw_includes = require_list(
            mapping.get("include_section_patterns", []),
            f"{source_id}.include_section_patterns",
        )
        for include_index, raw_include in enumerate(raw_includes, start=1):
            include_mapping = require_mapping(
                raw_include, f"{source_id}.include_section_patterns[{include_index}]"
            )
            include_rules.append(
                SourceIncludeRule(
                    rule_id=require_string(
                        include_mapping.get("rule_id"),
                        f"{source_id} include rule {include_index} rule_id",
                    ),
                    patterns=require_string_tuple(
                        include_mapping.get("patterns"),
                        f"{source_id} include rule {include_index} patterns",
                        allow_empty=False,
                    ),
                    concepts=require_string_tuple(
                        include_mapping.get("concepts"),
                        f"{source_id} include rule {include_index} concepts",
                        allow_empty=False,
                    ),
                )
            )

        exclude_rules: list[SourceExcludeRule] = []
        raw_excludes = require_list(
            mapping.get("exclude_section_patterns", []),
            f"{source_id}.exclude_section_patterns",
        )
        for exclude_index, raw_exclude in enumerate(raw_excludes, start=1):
            exclude_mapping = require_mapping(
                raw_exclude, f"{source_id}.exclude_section_patterns[{exclude_index}]"
            )
            deferred = optional_string(exclude_mapping.get("deferred_concept_family"))
            exclude_rules.append(
                SourceExcludeRule(
                    rule_id=require_string(
                        exclude_mapping.get("rule_id"),
                        f"{source_id} exclude rule {exclude_index} rule_id",
                    ),
                    patterns=require_string_tuple(
                        exclude_mapping.get("patterns"),
                        f"{source_id} exclude rule {exclude_index} patterns",
                        allow_empty=False,
                    ),
                    rationale=optional_string(exclude_mapping.get("rationale")),
                    deferred_concept_family=deferred or None,
                )
            )

        source_caps = require_mapping(mapping.get("source_caps"), f"{source_id}.source_caps")
        per_section_value = source_caps.get("maximum_chunks_per_section")
        if per_section_value is None:
            per_section_value = source_caps.get("maximum_chunks_per_karika")

        result[source_id] = SourceRule(
            source_id=source_id,
            source_title=require_string(mapping.get("source_title"), f"{source_id}.source_title"),
            domain=require_string(mapping.get("domain"), f"{source_id}.domain"),
            scope_status=require_string(mapping.get("scope_status"), f"{source_id}.scope_status"),
            structural_strategy=optional_string(mapping.get("structural_strategy")),
            structure_assessment=optional_string(mapping.get("structure_assessment")),
            preprocessing_requirements=require_string_tuple(
                mapping.get("preprocessing_requirements", []),
                f"{source_id}.preprocessing_requirements",
            ),
            include_rules=tuple(include_rules),
            exclude_rules=tuple(exclude_rules),
            hard_negative_targets=require_string_tuple(
                mapping.get("hard_negative_targets", []),
                f"{source_id}.hard_negative_targets",
            ),
            maximum_pre_review_chunks=require_positive_int(
                source_caps.get("maximum_pre_review_chunks"),
                f"{source_id}.maximum_pre_review_chunks",
            ),
            maximum_chunks_per_section=require_positive_int(
                per_section_value, f"{source_id}.maximum_chunks_per_section"
            ),
        )
    return result


def parse_hard_negative_rules(
    scope: Mapping[str, object],
) -> dict[str, HardNegativeRule]:
    raw_rules = require_list(scope.get("hard_negative_categories"), "hard_negative_categories")
    result: dict[str, HardNegativeRule] = {}
    for index, raw_rule in enumerate(raw_rules, start=1):
        mapping = require_mapping(raw_rule, f"hard_negative_categories[{index}]")
        category = require_string(
            mapping.get("category"), f"hard_negative_categories[{index}].category"
        )
        if category in result:
            raise SelectionError(f"Duplicate hard-negative category: {category}")
        result[category] = HardNegativeRule(
            category=category,
            target_concept=require_string(
                mapping.get("target_concept"), f"{category}.target_concept"
            ),
            positive_terms=require_string_tuple(
                mapping.get("positive_terms"),
                f"{category}.positive_terms",
                allow_empty=False,
            ),
            exclusion_terms=require_string_tuple(
                mapping.get("exclusion_terms"), f"{category}.exclusion_terms"
            ),
            required_domain_contrast=require_string_tuple(
                mapping.get("required_domain_contrast", []),
                f"{category}.required_domain_contrast",
            ),
        )
    return result


def parse_global_exclusions(
    scope: Mapping[str, object],
) -> tuple[GlobalExclusionRule, ...]:
    raw_rules = require_list(scope.get("global_exclusion_rules"), "global_exclusion_rules")
    result: list[GlobalExclusionRule] = []
    for index, raw_rule in enumerate(raw_rules, start=1):
        mapping = require_mapping(raw_rule, f"global_exclusion_rules[{index}]")
        patterns = require_string_tuple(
            mapping.get("patterns", []),
            f"global_exclusion_rules[{index}].patterns",
        )
        if not patterns:
            continue
        result.append(
            GlobalExclusionRule(
                rule_id=require_string(
                    mapping.get("rule_id"),
                    f"global_exclusion_rules[{index}].rule_id",
                ),
                category=require_string(
                    mapping.get("category"),
                    f"global_exclusion_rules[{index}].category",
                ),
                match_fields=require_string_tuple(
                    mapping.get("match_fields"),
                    f"global_exclusion_rules[{index}].match_fields",
                    allow_empty=False,
                ),
                patterns=patterns,
                rationale=optional_string(mapping.get("rationale")),
            )
        )
    return tuple(result)


def parse_quality_thresholds(scope: Mapping[str, object]) -> tuple[float, float]:
    minimum_alphabetic_ratio = 0.20
    maximum_ocr_noise_score = 0.20
    raw_rules = require_list(scope.get("global_exclusion_rules"), "global_exclusion_rules")
    for raw_rule in raw_rules:
        mapping = require_mapping(raw_rule, "global exclusion rule")
        if optional_string(mapping.get("rule_id")) != "global_low_text_quality":
            continue
        conditions = require_mapping(
            mapping.get("conditions"), "global_low_text_quality.conditions"
        )
        minimum_alphabetic_ratio = optional_float(
            conditions.get(
                "minimum_alphabetic_ratio",
                conditions.get("maximum_alphabetic_ratio", minimum_alphabetic_ratio),
            ),
            minimum_alphabetic_ratio,
        )
        maximum_ocr_noise_score = optional_float(
            conditions.get("maximum_ocr_noise_score", maximum_ocr_noise_score),
            maximum_ocr_noise_score,
        )
    if not 0.0 <= minimum_alphabetic_ratio <= 1.0:
        raise SelectionError("minimum alphabetic ratio must be between 0 and 1")
    if not 0.0 <= maximum_ocr_noise_score <= 1.0:
        raise SelectionError("maximum OCR noise score must be between 0 and 1")
    return minimum_alphabetic_ratio, maximum_ocr_noise_score


def parse_scope_configuration(scope_path: Path) -> ScopeConfiguration:
    scope = load_yaml_mapping(scope_path)
    status = require_string(scope.get("status"), "scope status")
    if status != "approved":
        raise SelectionError(
            f"The Phase 1 section scope must be approved before selection. Current status: {status}"
        )

    phase = require_mapping(scope.get("phase"), "phase")
    concepts = require_string_tuple(phase.get("concepts"), "phase.concepts", allow_empty=False)
    domains = require_string_tuple(phase.get("domains"), "phase.domains", allow_empty=False)
    selection_targets = require_mapping(scope.get("selection_targets"), "selection_targets")
    target_minimum = require_positive_int(
        selection_targets.get("pre_review_candidate_minimum"),
        "pre_review_candidate_minimum",
    )
    target_maximum = require_positive_int(
        selection_targets.get("pre_review_candidate_maximum"),
        "pre_review_candidate_maximum",
    )
    if target_minimum > target_maximum:
        raise SelectionError("Pre-review minimum exceeds maximum")

    raw_target_by_domain_value = selection_targets.get("pre_review_target_by_domain")
    if raw_target_by_domain_value is None:
        raw_target_by_domain_value = selection_targets.get("target_by_domain")
    raw_target_by_domain = require_mapping(
        raw_target_by_domain_value,
        "selection_targets.pre_review_target_by_domain",
    )
    domain_weights: dict[str, float] = {}
    domain_minimums: dict[str, int] = {}
    domain_maximums: dict[str, int] = {}
    for domain in domains:
        target_mapping = require_mapping(
            raw_target_by_domain.get(domain),
            f"pre_review_target_by_domain.{domain}",
        )
        minimum = require_positive_int(
            target_mapping.get("minimum"),
            f"pre_review_target_by_domain.{domain}.minimum",
        )
        maximum = require_positive_int(
            target_mapping.get("maximum"),
            f"pre_review_target_by_domain.{domain}.maximum",
        )
        if minimum > maximum:
            raise SelectionError(f"Domain minimum exceeds maximum for {domain}")
        domain_minimums[domain] = minimum
        domain_maximums[domain] = maximum
        domain_weights[domain] = float(minimum + maximum) / 2.0

    if sum(domain_minimums.values()) > target_maximum:
        raise SelectionError("The sum of pre-review domain minimums exceeds the global maximum")

    hard_negative_target = require_mapping(
        selection_targets.get("hard_negative_target"),
        "selection_targets.hard_negative_target",
    )
    hard_negative_minimum = require_positive_int(
        hard_negative_target.get("minimum_total"),
        "hard_negative_target.minimum_total",
    )
    hard_negative_target_value = require_positive_int(
        hard_negative_target.get("target_total"),
        "hard_negative_target.target_total",
    )
    hard_negative_maximum = require_positive_int(
        hard_negative_target.get("maximum_total"),
        "hard_negative_target.maximum_total",
    )

    raw_pre_review_hard_negative_caps = selection_targets.get("pre_review_hard_negative_caps")
    hard_negative_total_cap = hard_negative_maximum
    hard_negative_category_caps: dict[str, int] = {}
    if raw_pre_review_hard_negative_caps is not None:
        pre_review_hard_negative_caps = require_mapping(
            raw_pre_review_hard_negative_caps,
            "selection_targets.pre_review_hard_negative_caps",
        )
        hard_negative_total_cap = require_positive_int(
            pre_review_hard_negative_caps.get("total_maximum"),
            "pre_review_hard_negative_caps.total_maximum",
        )
        raw_category_caps = require_mapping(
            pre_review_hard_negative_caps.get("per_category_maximum"),
            "pre_review_hard_negative_caps.per_category_maximum",
        )
        for category, raw_cap in raw_category_caps.items():
            hard_negative_category_caps[category] = require_positive_int(
                raw_cap,
                f"pre_review_hard_negative_caps.per_category_maximum.{category}",
            )

    minimum_alphabetic_ratio, maximum_ocr_noise_score = parse_quality_thresholds(scope)
    source_rules = parse_source_rules(scope)
    for source_rule in source_rules.values():
        if source_rule.scope_status != "approved":
            raise SelectionError(
                "Every source rule must be approved. "
                f"{source_rule.source_id} is {source_rule.scope_status!r}."
            )
        if source_rule.domain not in domains:
            raise SelectionError(
                f"Unknown domain for {source_rule.source_id}: {source_rule.domain}"
            )

    concept_rules = parse_concept_rules(scope)
    missing_concepts = set(concepts) - set(concept_rules)
    if missing_concepts:
        raise SelectionError(
            "Concept rules are missing for: " + ", ".join(sorted(missing_concepts))
        )

    hard_negative_rules = parse_hard_negative_rules(scope)
    for hard_negative_rule in hard_negative_rules.values():
        if hard_negative_rule.target_concept not in concepts:
            raise SelectionError(
                f"Hard-negative category {hard_negative_rule.category} targets "
                f"unknown concept {hard_negative_rule.target_concept}"
            )
    unknown_hard_negative_caps = set(hard_negative_category_caps) - set(hard_negative_rules)
    if unknown_hard_negative_caps:
        raise SelectionError(
            "Hard-negative caps reference unknown categories: "
            + ", ".join(sorted(unknown_hard_negative_caps))
        )
    for category in hard_negative_rules:
        hard_negative_category_caps.setdefault(category, hard_negative_total_cap)

    methodological_constraints = require_mapping(
        scope.get("methodological_constraints"), "methodological_constraints"
    )
    required_true_constraints = (
        "selection_must_be_embedding_independent",
        "prohibit_anchor_similarity_selection",
        "prohibit_chunk_concept_weight_selection",
        "prohibit_llm_labels_as_final_authority",
        "require_human_review_before_activation",
    )
    for constraint in required_true_constraints:
        if methodological_constraints.get(constraint) is not True:
            raise SelectionError(f"Required methodological constraint is not true: {constraint}")

    return ScopeConfiguration(
        scope_version=require_string(scope.get("scope_version"), "scope_version"),
        corpus_version=require_string(scope.get("corpus_version"), "corpus_version"),
        status=status,
        concepts=concepts,
        domains=domains,
        target_minimum=target_minimum,
        target_maximum=target_maximum,
        domain_weights=domain_weights,
        domain_minimums=domain_minimums,
        domain_maximums=domain_maximums,
        hard_negative_minimum=hard_negative_minimum,
        hard_negative_target=hard_negative_target_value,
        hard_negative_maximum=hard_negative_maximum,
        hard_negative_total_cap=hard_negative_total_cap,
        hard_negative_category_caps=hard_negative_category_caps,
        minimum_alphabetic_ratio=minimum_alphabetic_ratio,
        maximum_ocr_noise_score=maximum_ocr_noise_score,
        concept_rules=concept_rules,
        source_rules=source_rules,
        hard_negative_rules=hard_negative_rules,
        global_exclusion_rules=parse_global_exclusions(scope),
        methodological_constraints=methodological_constraints,
    )


def validate_candidate_manifest(
    path: Path,
    *,
    expected_corpus_version: str,
    expected_chunk_count: int,
) -> None:
    manifest = require_mapping(load_json_value(path), "candidate corpus manifest")
    status = optional_string(find_first_key(manifest, "status"))
    if status not in {"candidate_only", "candidate_only_frozen"}:
        raise SelectionError(f"Candidate manifest has unsafe status: {status!r}")
    version = optional_string(find_first_key(manifest, "corpus_version"))
    if not version:
        version = optional_string(find_first_key(manifest, "version"))
    if version != expected_corpus_version:
        raise SelectionError(
            "Candidate corpus version mismatch: "
            f"expected {expected_corpus_version!r}, found {version!r}"
        )
    chunk_count = optional_int(find_first_key(manifest, "chunk_count"), -1)
    if chunk_count != expected_chunk_count:
        raise SelectionError(
            "Candidate manifest chunk count mismatch: "
            f"expected {expected_chunk_count}, found {chunk_count}"
        )


def read_structure_report(
    path: Path,
) -> tuple[dict[tuple[str, str], SectionInfo], dict[str, SectionInfo]]:
    by_source_and_section: dict[tuple[str, str], SectionInfo] = {}
    unique_by_section: dict[str, SectionInfo] = {}
    duplicated_section_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SelectionError(f"Structure report has no header: {path}")
        required = {"source_id", "section_id", "proposed_structure_action"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise SelectionError(
                "Structure report is missing columns: " + ", ".join(sorted(missing))
            )
        for row_number, row in enumerate(reader, start=2):
            source_id = (row.get("source_id") or "").strip()
            section_id = (row.get("section_id") or "").strip()
            if not source_id or not section_id:
                LOGGER.warning("Skipping structure-report row %d with missing IDs", row_number)
                continue
            raw_ocr_score = optional_string(row.get("ocr_noise_score"))
            info = SectionInfo(
                source_id=source_id,
                section_id=section_id,
                section_title=(row.get("section_title") or "").strip(),
                parent_section=(row.get("parent_section") or "").strip(),
                structural_locator=(row.get("structural_locator") or "").strip(),
                unit_type=(row.get("unit_type") or "").strip(),
                proposed_action=(row.get("proposed_structure_action") or "").strip(),
                parser_warning_count=optional_int(row.get("parser_warning_count"), 0),
                report_ocr_noise_score=(
                    optional_float(raw_ocr_score, 0.0) if raw_ocr_score else None
                ),
            )
            by_source_and_section[(source_id, section_id)] = info
            if section_id in unique_by_section:
                duplicated_section_ids.add(section_id)
            else:
                unique_by_section[section_id] = info
    for section_id in duplicated_section_ids:
        unique_by_section.pop(section_id, None)
    return by_source_and_section, unique_by_section


def is_chunk_mapping(mapping: Mapping[str, object]) -> bool:
    chunk_id = first_path_value(mapping, CHUNK_ID_PATHS)
    text = first_path_value(mapping, TEXT_PATHS)
    return bool(optional_string(chunk_id) and optional_string(text))


def iter_chunk_mappings(value: object) -> Iterator[dict[str, object]]:
    if isinstance(value, Mapping):
        mapping = require_mapping(value, "chunk JSON mapping")
        if is_chunk_mapping(mapping):
            yield mapping
            return
        for nested_value in mapping.values():
            yield from iter_chunk_mappings(nested_value)
        return
    if isinstance(value, list | tuple):
        for nested_value in value:
            yield from iter_chunk_mappings(nested_value)


def iter_jsonl_mappings(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                loaded: object = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SelectionError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            yield from iter_chunk_mappings(loaded)


def flatten_warning_values(value: object) -> tuple[str, ...]:
    warnings: list[str] = []

    def visit(nested: object) -> None:
        if nested is None:
            return
        if isinstance(nested, str):
            normalized = nested.strip()
            if normalized:
                warnings.append(normalized)
            return
        if isinstance(nested, Mapping):
            mapping = require_mapping(nested, "warning mapping")
            for key, nested_value in mapping.items():
                if isinstance(nested_value, bool):
                    if nested_value:
                        warnings.append(key)
                else:
                    visit(nested_value)
            return
        if isinstance(nested, list | tuple | set):
            for item in nested:
                visit(item)
            return
        warnings.append(str(nested))

    visit(value)
    return tuple(dict.fromkeys(warnings))


def derive_source_id_from_chunk_id(chunk_id: str, known_source_ids: Iterable[str]) -> str:
    matches = [
        source_id
        for source_id in known_source_ids
        if chunk_id.startswith((f"{source_id}:", f"{source_id}_"))
    ]
    return max(matches, key=len) if matches else ""


def parse_chunk_record(
    mapping: Mapping[str, object],
    *,
    origin_path: Path,
    scope: ScopeConfiguration,
) -> ChunkRecord:
    chunk_id = require_string(
        first_path_value(mapping, CHUNK_ID_PATHS), f"chunk_id in {origin_path}"
    )
    text = require_string(first_path_value(mapping, TEXT_PATHS), f"text for chunk {chunk_id}")
    source_id = optional_string(first_path_value(mapping, SOURCE_ID_PATHS))
    if not source_id:
        source_id = derive_source_id_from_chunk_id(chunk_id, scope.source_rules)
    if source_id not in scope.source_rules:
        raise SelectionError(f"Chunk {chunk_id} refers to unknown source: {source_id!r}")
    source_rule = scope.source_rules[source_id]
    domain = optional_string(first_path_value(mapping, DOMAIN_PATHS))
    if not domain:
        domain = source_rule.domain
    if domain != source_rule.domain:
        raise SelectionError(
            f"Chunk {chunk_id} domain mismatch: chunk={domain!r}, scope={source_rule.domain!r}"
        )
    source_title = optional_string(first_path_value(mapping, SOURCE_TITLE_PATHS))
    if not source_title:
        source_title = source_rule.source_title
    section_id = optional_string(first_path_value(mapping, SECTION_ID_PATHS))
    if not section_id:
        section_id = f"{source_id}:section:unassigned"
    token_count = optional_int(first_path_value(mapping, TOKEN_COUNT_PATHS), 0)
    if token_count <= 0:
        token_count = len(text.split())
    checksum = optional_string(first_path_value(mapping, CHECKSUM_PATHS))
    if not checksum:
        checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return ChunkRecord(
        chunk_id=chunk_id,
        source_id=source_id,
        domain=domain,
        source_title=source_title,
        section_id=section_id,
        section_title=optional_string(first_path_value(mapping, SECTION_TITLE_PATHS)),
        parent_section=optional_string(first_path_value(mapping, PARENT_SECTION_PATHS)),
        unit_type=optional_string(first_path_value(mapping, UNIT_TYPE_PATHS)),
        structural_locator=optional_string(first_path_value(mapping, STRUCTURAL_LOCATOR_PATHS)),
        text=text,
        token_count=token_count,
        warnings=flatten_warning_values(first_path_value(mapping, WARNING_PATHS)),
        content_checksum=checksum,
        origin_path=origin_path.as_posix(),
    )


def load_chunks(chunks_root: Path, *, scope: ScopeConfiguration) -> list[ChunkRecord]:
    paths = sorted(
        (
            path
            for path in chunks_root.rglob("*")
            if path.is_file() and path.suffix.casefold() in {".json", ".jsonl"}
        ),
        key=lambda path: path.as_posix(),
    )
    if not paths:
        raise SelectionError(f"No JSON or JSONL chunks found in {chunks_root}")
    records: list[ChunkRecord] = []
    seen_chunk_ids: set[str] = set()
    for path in paths:
        mappings: Iterator[dict[str, object]]
        if path.suffix.casefold() == ".jsonl":
            mappings = iter_jsonl_mappings(path)
        else:
            mappings = iter_chunk_mappings(load_json_value(path))
        count_before = len(records)
        for mapping in mappings:
            record = parse_chunk_record(mapping, origin_path=path, scope=scope)
            if record.chunk_id in seen_chunk_ids:
                raise SelectionError(f"Duplicate chunk_id: {record.chunk_id}")
            seen_chunk_ids.add(record.chunk_id)
            records.append(record)
        LOGGER.debug("Loaded %d chunks from %s", len(records) - count_before, path)
    records.sort(key=lambda record: record.chunk_id)
    return records


def merge_section_metadata(chunk: ChunkRecord, section_info: SectionInfo | None) -> ChunkRecord:
    if section_info is None:
        return chunk
    return ChunkRecord(
        chunk_id=chunk.chunk_id,
        source_id=chunk.source_id,
        domain=chunk.domain,
        source_title=chunk.source_title,
        section_id=chunk.section_id,
        section_title=chunk.section_title or section_info.section_title,
        parent_section=chunk.parent_section or section_info.parent_section,
        unit_type=chunk.unit_type or section_info.unit_type,
        structural_locator=(chunk.structural_locator or section_info.structural_locator),
        text=chunk.text,
        token_count=chunk.token_count,
        warnings=chunk.warnings,
        content_checksum=chunk.content_checksum,
        origin_path=chunk.origin_path,
    )


def calculate_quality_metrics(
    text: str,
    *,
    minimum_alphabetic_ratio: float,
    maximum_ocr_noise_score: float,
) -> QualityMetrics:
    visible_characters = [character for character in text if not character.isspace()]
    visible_count = max(1, len(visible_characters))
    alphabetic_count = sum(character.isalpha() for character in visible_characters)
    digit_count = sum(character.isdigit() for character in visible_characters)
    replacement_count = sum(character in {"", "□", "■"} for character in visible_characters)
    symbol_count = sum(not character.isalnum() for character in visible_characters)
    raw_tokens = re.findall(r"\S+", text, flags=re.UNICODE)
    token_count = max(1, len(raw_tokens))
    single_character_tokens = sum(len(normalize_term(token)) == 1 for token in raw_tokens)
    mixed_script_tokens = 0
    alphanumeric_intrusion_tokens = 0
    for token in raw_tokens:
        has_latin = any(
            "LATIN" in unicodedata.name(character, "") for character in token if character.isalpha()
        )
        has_devanagari = any(
            "DEVANAGARI" in unicodedata.name(character, "")
            for character in token
            if character.isalpha()
        )
        if has_latin and has_devanagari:
            mixed_script_tokens += 1
        has_letter = any(character.isalpha() for character in token)
        has_digit = any(character.isdigit() for character in token)
        if has_letter and has_digit:
            alphanumeric_intrusion_tokens += 1

    alphabetic_ratio = float(alphabetic_count) / visible_count
    digit_ratio = float(digit_count) / visible_count
    symbol_ratio = float(symbol_count) / visible_count
    replacement_character_ratio = float(replacement_count) / visible_count
    single_character_token_ratio = float(single_character_tokens) / token_count
    mixed_script_token_ratio = float(mixed_script_tokens) / token_count
    alphanumeric_intrusion_ratio = float(alphanumeric_intrusion_tokens) / token_count
    alphabetic_penalty = max(0.0, minimum_alphabetic_ratio - alphabetic_ratio)
    ocr_noise_score = min(
        1.0,
        symbol_ratio * 0.20
        + digit_ratio * 0.20
        + single_character_token_ratio * 0.15
        + mixed_script_token_ratio * 0.25
        + alphanumeric_intrusion_ratio * 0.25
        + replacement_character_ratio * 0.75
        + alphabetic_penalty * 1.50,
    )
    severe = (
        alphabetic_ratio <= minimum_alphabetic_ratio or ocr_noise_score >= maximum_ocr_noise_score
    )
    return QualityMetrics(
        alphabetic_ratio=alphabetic_ratio,
        digit_ratio=digit_ratio,
        symbol_ratio=symbol_ratio,
        single_character_token_ratio=single_character_token_ratio,
        mixed_script_token_ratio=mixed_script_token_ratio,
        alphanumeric_intrusion_ratio=alphanumeric_intrusion_ratio,
        replacement_character_ratio=replacement_character_ratio,
        ocr_noise_score=ocr_noise_score,
        severe_ocr_noise=severe,
    )


def build_search_fields(chunk: ChunkRecord) -> dict[str, str]:
    return {
        "chunk_text": chunk.text,
        "text": chunk.text,
        "sample_text": chunk.text[:1000],
        "section_title": chunk.section_title,
        "parent_section": chunk.parent_section,
        "structural_locator": chunk.structural_locator,
        "unit_type": chunk.unit_type,
        "source_title": chunk.source_title,
    }


def match_global_exclusions(
    fields: Mapping[str, str], rules: Sequence[GlobalExclusionRule]
) -> tuple[str, ...]:
    matched_rule_ids: list[str] = []
    for rule in rules:
        matched = False
        for field_name in rule.match_fields:
            field_value = fields.get(field_name, "")
            if not field_value:
                continue
            for pattern in rule.patterns:
                try:
                    if re.search(pattern, field_value, flags=re.IGNORECASE | re.UNICODE):
                        matched = True
                        break
                except re.error as exc:
                    raise SelectionError(
                        f"Invalid regex {pattern!r} in {rule.rule_id}: {exc}"
                    ) from exc
            if matched:
                break
        if matched:
            matched_rule_ids.append(rule.rule_id)
    return tuple(matched_rule_ids)


def match_source_include_rules(
    normalized_all_text: str,
    source_rule: SourceRule,
    matcher: TermMatcher,
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]], dict[str, float]]:
    matched_rule_ids: list[str] = []
    pattern_hits: dict[str, tuple[str, ...]] = {}
    concept_boosts: defaultdict[str, float] = defaultdict(float)
    for rule in source_rule.include_rules:
        hits = matcher.matched_terms(normalized_all_text, rule.patterns)
        if not hits:
            continue
        matched_rule_ids.append(rule.rule_id)
        pattern_hits[rule.rule_id] = hits
        boost = 2.0 + min(3.0, float(len(hits)))
        for concept_id in rule.concepts:
            concept_boosts[concept_id] += boost
    return tuple(matched_rule_ids), pattern_hits, dict(concept_boosts)


def match_source_exclude_rules(
    normalized_structure_text: str,
    normalized_chunk_text: str,
    source_rule: SourceRule,
    matcher: TermMatcher,
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]], bool]:
    matched_rule_ids: list[str] = []
    pattern_hits: dict[str, tuple[str, ...]] = {}
    structural_match = False
    for rule in source_rule.exclude_rules:
        structure_hits = matcher.matched_terms(normalized_structure_text, rule.patterns)
        text_hits = matcher.matched_terms(normalized_chunk_text, rule.patterns)
        hits = tuple(dict.fromkeys((*structure_hits, *text_hits)))
        if not hits:
            continue
        matched_rule_ids.append(rule.rule_id)
        pattern_hits[rule.rule_id] = hits
        if structure_hits:
            structural_match = True
    return tuple(matched_rule_ids), pattern_hits, structural_match


def evaluate_concepts(
    normalized_chunk_text: str,
    normalized_all_text: str,
    scope: ScopeConfiguration,
    matcher: TermMatcher,
    source_concept_boosts: Mapping[str, float],
) -> tuple[
    tuple[str, ...],
    dict[str, float],
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
]:
    active_concepts: list[str] = []
    scores: dict[str, float] = {}
    positive_hits_by_concept: dict[str, tuple[str, ...]] = {}
    context_hits_by_concept: dict[str, tuple[str, ...]] = {}
    exclusion_hits_by_concept: dict[str, tuple[str, ...]] = {}
    for concept_id in scope.concepts:
        rule = scope.concept_rules[concept_id]
        positive_hits = matcher.matched_terms(normalized_all_text, rule.positive_terms)
        context_hits = matcher.matched_terms(normalized_chunk_text, rule.contextual_terms)
        exclusion_hits = matcher.matched_terms(normalized_chunk_text, rule.exclusion_terms)
        positive_hits_by_concept[concept_id] = positive_hits
        context_hits_by_concept[concept_id] = context_hits
        exclusion_hits_by_concept[concept_id] = exclusion_hits
        score = (
            len(positive_hits) * 3.0
            + min(len(context_hits), 5) * 0.75
            + source_concept_boosts.get(concept_id, 0.0)
            - len(exclusion_hits) * 2.0
        )
        scores[concept_id] = round(score, 6)
        enough_positive_hits = len(positive_hits) >= rule.minimum_positive_term_matches
        source_rule_support = source_concept_boosts.get(concept_id, 0.0) > 0.0
        has_substantive_context = (
            bool(context_hits)
            or source_rule_support
            or any(" " in normalize_term(term) for term in positive_hits)
        )
        if not (enough_positive_hits or source_rule_support):
            continue
        if rule.require_substantive_context and not has_substantive_context:
            continue
        if score > 0.0:
            active_concepts.append(concept_id)
    return (
        tuple(active_concepts),
        scores,
        positive_hits_by_concept,
        context_hits_by_concept,
        exclusion_hits_by_concept,
    )


def evaluate_hard_negatives(
    normalized_chunk_text: str,
    chunk_domain: str,
    scope: ScopeConfiguration,
    matcher: TermMatcher,
    active_concepts: Sequence[str],
    source_rule: SourceRule,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    categories: list[str] = []
    targets: list[str] = []
    for category, rule in scope.hard_negative_rules.items():
        if source_rule.hard_negative_targets and category not in source_rule.hard_negative_targets:
            continue
        if rule.required_domain_contrast and chunk_domain not in rule.required_domain_contrast:
            continue
        positive_hits = matcher.matched_terms(normalized_chunk_text, rule.positive_terms)
        exclusion_hits = matcher.matched_terms(normalized_chunk_text, rule.exclusion_terms)
        if not positive_hits or exclusion_hits:
            continue
        if rule.target_concept in active_concepts:
            continue
        categories.append(category)
        targets.append(rule.target_concept)
    return tuple(categories), tuple(dict.fromkeys(targets))


def stable_tie_breaker(chunk_id: str) -> str:
    return hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()


def evaluate_chunk(
    chunk: ChunkRecord,
    *,
    section_info: SectionInfo | None,
    scope: ScopeConfiguration,
    matcher: TermMatcher,
) -> Evaluation:
    merged_chunk = merge_section_metadata(chunk, section_info)
    quality = calculate_quality_metrics(
        merged_chunk.text,
        minimum_alphabetic_ratio=scope.minimum_alphabetic_ratio,
        maximum_ocr_noise_score=scope.maximum_ocr_noise_score,
    )
    evaluation = Evaluation(
        chunk=merged_chunk,
        section_info=section_info,
        quality=quality,
        tie_breaker=stable_tie_breaker(merged_chunk.chunk_id),
    )
    source_rule = scope.source_rules[merged_chunk.source_id]
    fields = build_search_fields(merged_chunk)
    normalized_chunk_text = normalize_term(merged_chunk.text)
    normalized_structure_text = normalize_term(
        " ".join(
            (
                merged_chunk.section_title,
                merged_chunk.parent_section,
                merged_chunk.structural_locator,
                merged_chunk.unit_type,
            )
        )
    )
    normalized_all_text = normalize_term(" ".join(fields.values()))

    if section_info is None:
        evaluation.review_flags.append("missing_structure_report_join")
    else:
        if section_info.proposed_action == "exclude_candidate":
            evaluation.exclusion_reasons.append("structure_report_exclude_candidate")
        elif section_info.proposed_action == "source_specific_preprocessing":
            evaluation.review_flags.append("source_specific_preprocessing")
        elif section_info.proposed_action == "manual_section_review":
            evaluation.review_flags.append("manual_section_review")
        if section_info.parser_warning_count > 0:
            evaluation.review_flags.append("parser_warnings_present")

    if quality.severe_ocr_noise:
        evaluation.exclusion_reasons.append("severe_ocr_noise")

    evaluation.global_exclusion_rule_ids = match_global_exclusions(
        fields, scope.global_exclusion_rules
    )
    evaluation.exclusion_reasons.extend(
        f"global_exclusion:{rule_id}" for rule_id in evaluation.global_exclusion_rule_ids
    )

    (
        evaluation.source_include_rule_ids,
        evaluation.source_include_pattern_hits,
        source_concept_boosts,
    ) = match_source_include_rules(normalized_all_text, source_rule, matcher)
    (
        evaluation.source_exclude_rule_ids,
        evaluation.source_exclude_pattern_hits,
        source_exclusion_is_structural,
    ) = match_source_exclude_rules(
        normalized_structure_text,
        normalized_chunk_text,
        source_rule,
        matcher,
    )
    (
        evaluation.active_concepts,
        evaluation.concept_scores,
        evaluation.concept_positive_hits,
        evaluation.concept_context_hits,
        evaluation.concept_exclusion_hits,
    ) = evaluate_concepts(
        normalized_chunk_text,
        normalized_all_text,
        scope,
        matcher,
        source_concept_boosts,
    )

    if evaluation.source_exclude_rule_ids:
        if source_exclusion_is_structural:
            evaluation.exclusion_reasons.extend(
                f"source_structural_exclusion:{rule_id}"
                for rule_id in evaluation.source_exclude_rule_ids
            )
        elif not evaluation.active_concepts:
            evaluation.exclusion_reasons.extend(
                f"source_textual_exclusion:{rule_id}"
                for rule_id in evaluation.source_exclude_rule_ids
            )
        else:
            evaluation.review_flags.extend(
                f"source_exclusion_overlap:{rule_id}"
                for rule_id in evaluation.source_exclude_rule_ids
            )

    (
        evaluation.hard_negative_categories,
        evaluation.hard_negative_targets,
    ) = evaluate_hard_negatives(
        normalized_chunk_text,
        merged_chunk.domain,
        scope,
        matcher,
        evaluation.active_concepts,
        source_rule,
    )
    contextual_hit_count = sum(len(hits) for hits in evaluation.concept_context_hits.values())
    positive_hit_count = sum(len(hits) for hits in evaluation.concept_positive_hits.values())
    source_rule_hit_count = sum(
        len(hits) for hits in evaluation.source_include_pattern_hits.values()
    )

    if evaluation.active_concepts and evaluation.hard_negative_categories:
        evaluation.selection_class = "mixed_positive_hard_negative"
    elif evaluation.active_concepts:
        evaluation.selection_class = "positive"
    elif evaluation.hard_negative_categories:
        evaluation.selection_class = "hard_negative"
    elif contextual_hit_count >= 2 or source_rule_hit_count > 0:
        evaluation.selection_class = "ambiguous_review"
        evaluation.review_flags.append("context_only_or_weak_match")
    else:
        evaluation.selection_class = "ineligible"
        evaluation.exclusion_reasons.append("no_phase1_lexical_evidence")

    highest_concept_score = max(evaluation.concept_scores.values(), default=0.0)
    class_bonus = {
        "mixed_positive_hard_negative": 15.0,
        "hard_negative": 12.0,
        "positive": 8.0,
        "ambiguous_review": 2.0,
        "ineligible": 0.0,
    }[evaluation.selection_class]
    structure_bonus = 0.0
    if merged_chunk.section_title:
        structure_bonus += 0.5
    if merged_chunk.structural_locator:
        structure_bonus += 0.5
    if section_info is not None:
        structure_bonus += 0.25
    warning_penalty = min(3.0, float(len(merged_chunk.warnings)) * 0.25)
    quality_penalty = quality.ocr_noise_score * 10.0
    evaluation.score = round(
        class_bonus
        + highest_concept_score
        + min(10, positive_hit_count) * 0.5
        + min(8, contextual_hit_count) * 0.25
        + min(6, source_rule_hit_count) * 0.75
        + len(evaluation.hard_negative_categories) * 2.0
        + structure_bonus
        - warning_penalty
        - quality_penalty,
        6,
    )
    evaluation.decision = "excluded" if evaluation.exclusion_reasons else "eligible"
    return evaluation


def evaluation_sort_key(evaluation: Evaluation) -> tuple[int, float, str]:
    class_priority = {
        "mixed_positive_hard_negative": 0,
        "hard_negative": 1,
        "positive": 2,
        "ambiguous_review": 3,
        "ineligible": 4,
    }[evaluation.selection_class]
    return class_priority, -evaluation.score, evaluation.tie_breaker


def allocate_with_capacity(
    total: int,
    *,
    weights: Mapping[str, float],
    capacities: Mapping[str, int],
    order: Sequence[str],
) -> dict[str, int]:
    allocation = dict.fromkeys(order, 0)
    remaining = total
    order_index = {key: index for index, key in enumerate(order)}
    while remaining > 0:
        active = [key for key in order if allocation[key] < capacities.get(key, 0)]
        if not active:
            break
        active_weight = sum(max(0.0, weights.get(key, 0.0)) for key in active)
        progressed = False
        if active_weight <= 0.0:
            for key in active:
                if remaining <= 0:
                    break
                allocation[key] += 1
                remaining -= 1
                progressed = True
            continue
        shares: list[tuple[float, str]] = []
        starting_remaining = remaining
        for key in active:
            raw_share = starting_remaining * max(0.0, weights.get(key, 0.0)) / active_weight
            whole_share = min(capacities[key] - allocation[key], math.floor(raw_share))
            if whole_share > 0:
                allocation[key] += whole_share
                remaining -= whole_share
                progressed = True
            shares.append((raw_share - math.floor(raw_share), key))
        if remaining <= 0:
            break
        for _, key in sorted(shares, key=lambda item: (-item[0], order_index[item[1]])):
            if remaining <= 0:
                break
            if allocation[key] >= capacities[key]:
                continue
            allocation[key] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            break
    return allocation


def can_select(
    evaluation: Evaluation,
    *,
    state: SelectionState,
    scope: ScopeConfiguration,
    domain_caps: Mapping[str, int],
    total_cap: int,
) -> bool:
    chunk = evaluation.chunk
    source_rule = scope.source_rules[chunk.source_id]
    if (
        chunk.chunk_id in state.selected_ids
        or state.total >= total_cap
        or state.by_domain[chunk.domain] >= domain_caps.get(chunk.domain, 0)
        or state.by_source[chunk.source_id] >= source_rule.maximum_pre_review_chunks
        or state.by_section[(chunk.source_id, chunk.section_id)]
        >= source_rule.maximum_chunks_per_section
    ):
        return False

    if evaluation.hard_negative_categories:
        if state.hard_negative_chunks >= scope.hard_negative_total_cap:
            return False
        selectable_category = any(
            state.by_hard_negative_category[category]
            < scope.hard_negative_category_caps.get(
                category,
                scope.hard_negative_total_cap,
            )
            for category in evaluation.hard_negative_categories
        )
        if not selectable_category:
            return False

    return True


def add_selection(evaluation: Evaluation, state: SelectionState) -> None:
    chunk = evaluation.chunk
    state.total += 1
    state.by_domain[chunk.domain] += 1
    state.by_source[chunk.source_id] += 1
    state.by_section[(chunk.source_id, chunk.section_id)] += 1
    if evaluation.hard_negative_categories:
        state.hard_negative_chunks += 1
        for category in evaluation.hard_negative_categories:
            state.by_hard_negative_category[category] += 1
    state.selected_ids.add(chunk.chunk_id)
    evaluation.decision = "selected"
    evaluation.selection_rank = state.total


def _select_first_eligible(
    pool: Sequence[Evaluation],
    *,
    state: SelectionState,
    scope: ScopeConfiguration,
    domain_caps: Mapping[str, int],
    total_cap: int,
    predicate: Callable[[Evaluation], bool],
) -> bool:
    for evaluation in pool:
        if evaluation.chunk.chunk_id in state.selected_ids:
            continue
        if not predicate(evaluation):
            continue
        if can_select(
            evaluation,
            state=state,
            scope=scope,
            domain_caps=domain_caps,
            total_cap=total_cap,
        ):
            add_selection(evaluation, state)
            return True
    return False


def select_candidates(
    evaluations: Sequence[Evaluation],
    *,
    scope: ScopeConfiguration,
    target_count: int,
) -> tuple[list[Evaluation], SelectionState, dict[str, int], dict[str, int]]:
    eligible = [evaluation for evaluation in evaluations if evaluation.decision == "eligible"]
    ordered_eligible = sorted(eligible, key=evaluation_sort_key)
    available_by_domain = Counter(evaluation.chunk.domain for evaluation in eligible)
    source_capacity_by_domain: Counter[str] = Counter()
    for source_rule in scope.source_rules.values():
        source_capacity_by_domain[source_rule.domain] += source_rule.maximum_pre_review_chunks

    effective_capacity = {
        domain: min(
            available_by_domain[domain],
            source_capacity_by_domain[domain],
            scope.domain_maximums[domain],
        )
        for domain in scope.domains
    }
    for domain in scope.domains:
        if effective_capacity[domain] < scope.domain_minimums[domain]:
            LOGGER.warning(
                "Domain %s has effective capacity %d below configured minimum %d",
                domain,
                effective_capacity[domain],
                scope.domain_minimums[domain],
            )

    domain_targets = allocate_with_capacity(
        target_count,
        weights=scope.domain_weights,
        capacities=effective_capacity,
        order=scope.domains,
    )
    for domain in scope.domains:
        domain_targets[domain] = max(
            min(scope.domain_minimums[domain], effective_capacity[domain]),
            min(domain_targets[domain], effective_capacity[domain]),
        )

    overflow = sum(domain_targets.values()) - target_count
    if overflow > 0:
        for domain in reversed(scope.domains):
            reducible = max(
                0,
                domain_targets[domain]
                - min(scope.domain_minimums[domain], effective_capacity[domain]),
            )
            reduction = min(reducible, overflow)
            domain_targets[domain] -= reduction
            overflow -= reduction
            if overflow == 0:
                break

    domain_maximums = dict(effective_capacity)
    state = SelectionState()

    positive_pool = [evaluation for evaluation in ordered_eligible if evaluation.active_concepts]
    non_hard_negative_pool = [
        evaluation for evaluation in ordered_eligible if not evaluation.hard_negative_categories
    ]
    hard_negative_pool = [
        evaluation for evaluation in ordered_eligible if evaluation.hard_negative_categories
    ]

    # Stage 1A: reserve transparent positive coverage for every domain x concept cell.
    progressed = True
    while progressed and state.total < target_count:
        progressed = False
        for domain in scope.domains:
            if state.by_domain[domain] >= min(
                scope.domain_minimums[domain],
                domain_maximums[domain],
            ):
                continue
            for concept in scope.concepts:

                def stage1a_predicate(
                    evaluation: Evaluation, d: str = domain, c: str = concept
                ) -> bool:
                    return (
                        evaluation.chunk.domain == d
                        and c in evaluation.active_concepts
                        and evaluation.selection_class
                        in {"positive", "mixed_positive_hard_negative"}
                    )

                selected_now = _select_first_eligible(
                    positive_pool,
                    state=state,
                    scope=scope,
                    domain_caps=domain_maximums,
                    total_cap=target_count,
                    predicate=stage1a_predicate,
                )
                progressed = progressed or selected_now
                if state.by_domain[domain] >= min(
                    scope.domain_minimums[domain],
                    domain_maximums[domain],
                ):
                    break

    # Stage 1B: fill each domain minimum using non-hard-negative positives first.
    for domain in scope.domains:
        domain_minimum = min(
            scope.domain_minimums[domain],
            domain_maximums[domain],
        )
        while state.by_domain[domain] < domain_minimum and state.total < target_count:

            def stage1b_predicate_primary(evaluation: Evaluation, d: str = domain) -> bool:
                return evaluation.chunk.domain == d and evaluation.selection_class in {
                    "positive",
                    "ambiguous_review",
                }

            selected_now = _select_first_eligible(
                non_hard_negative_pool,
                state=state,
                scope=scope,
                domain_caps=domain_maximums,
                total_cap=target_count,
                predicate=stage1b_predicate_primary,
            )
            if not selected_now:

                def stage1b_predicate_fallback(evaluation: Evaluation, d: str = domain) -> bool:
                    return evaluation.chunk.domain == d

                selected_now = _select_first_eligible(
                    positive_pool,
                    state=state,
                    scope=scope,
                    domain_caps=domain_maximums,
                    total_cap=target_count,
                    predicate=stage1b_predicate_fallback,
                )
            if not selected_now:
                break

    # Stage 2: reserve a bounded, category-balanced hard-negative review pool.
    desired_hard_negative_count = min(
        scope.hard_negative_total_cap,
        max(
            scope.hard_negative_target,
            round(target_count * 0.22),
        ),
    )
    while state.hard_negative_chunks < desired_hard_negative_count and state.total < target_count:
        progressed = False
        for category in scope.hard_negative_rules:
            category_cap = scope.hard_negative_category_caps.get(
                category,
                scope.hard_negative_total_cap,
            )
            if state.by_hard_negative_category[category] >= category_cap:
                continue

            def stage2_predicate(evaluation: Evaluation, c: str = category) -> bool:
                return c in evaluation.hard_negative_categories

            selected_now = _select_first_eligible(
                hard_negative_pool,
                state=state,
                scope=scope,
                domain_caps=domain_maximums,
                total_cap=target_count,
                predicate=stage2_predicate,
            )
            progressed = progressed or selected_now
            if (
                state.hard_negative_chunks >= desired_hard_negative_count
                or state.total >= target_count
            ):
                break
        if not progressed:
            break

    # Stage 3: fill each domain toward its proportional target, prioritizing positives.
    for domain in scope.domains:
        while state.by_domain[domain] < domain_targets[domain] and state.total < target_count:

            def stage3_predicate_primary(evaluation: Evaluation, d: str = domain) -> bool:
                return evaluation.chunk.domain == d

            selected_now = _select_first_eligible(
                non_hard_negative_pool,
                state=state,
                scope=scope,
                domain_caps=domain_maximums,
                total_cap=target_count,
                predicate=stage3_predicate_primary,
            )
            if not selected_now:

                def stage3_predicate_fallback(evaluation: Evaluation, d: str = domain) -> bool:
                    return evaluation.chunk.domain == d

                selected_now = _select_first_eligible(
                    ordered_eligible,
                    state=state,
                    scope=scope,
                    domain_caps=domain_maximums,
                    total_cap=target_count,
                    predicate=stage3_predicate_fallback,
                )
            if not selected_now:
                break

    # Stage 4: fill remaining global capacity without exceeding domain or HN caps.
    while state.total < target_count:

        def stage4_predicate(evaluation: Evaluation) -> bool:
            return True

        selected_now = _select_first_eligible(
            ordered_eligible,
            state=state,
            scope=scope,
            domain_caps=domain_maximums,
            total_cap=target_count,
            predicate=stage4_predicate,
        )
        if not selected_now:
            break

    selected = [evaluation for evaluation in evaluations if evaluation.decision == "selected"]
    selected.sort(
        key=lambda evaluation: (
            evaluation.selection_rank if evaluation.selection_rank is not None else sys.maxsize
        )
    )
    return selected, state, domain_targets, domain_maximums


def finalize_exclusion_reasons(
    evaluations: Sequence[Evaluation],
    *,
    state: SelectionState,
    scope: ScopeConfiguration,
    domain_maximums: Mapping[str, int],
    target_count: int,
) -> None:
    for evaluation in evaluations:
        if evaluation.decision == "selected":
            continue
        if evaluation.exclusion_reasons:
            evaluation.decision = "excluded"
            continue
        chunk = evaluation.chunk
        source_rule = scope.source_rules[chunk.source_id]
        section_key = (chunk.source_id, chunk.section_id)
        if state.by_section[section_key] >= source_rule.maximum_chunks_per_section:
            evaluation.exclusion_reasons.append("selection_cap:section")
        elif state.by_source[chunk.source_id] >= source_rule.maximum_pre_review_chunks:
            evaluation.exclusion_reasons.append("selection_cap:source")
        elif state.by_domain[chunk.domain] >= domain_maximums.get(chunk.domain, 0):
            evaluation.exclusion_reasons.append("selection_cap:domain")
        elif state.total >= target_count:
            evaluation.exclusion_reasons.append("selection_target_reached")
        else:
            evaluation.exclusion_reasons.append("lower_ranked_eligible_candidate")
        evaluation.decision = "excluded"


def rounded(value: float) -> float:
    return round(value, 6)


def evaluation_to_output(evaluation: Evaluation) -> dict[str, object]:
    chunk = evaluation.chunk
    section_action = (
        evaluation.section_info.proposed_action if evaluation.section_info is not None else ""
    )
    report_ocr_noise_score = (
        evaluation.section_info.report_ocr_noise_score
        if evaluation.section_info is not None
        else None
    )
    return {
        "chunk_id": chunk.chunk_id,
        "source_id": chunk.source_id,
        "domain": chunk.domain,
        "source_title": chunk.source_title,
        "section_id": chunk.section_id,
        "section_title": chunk.section_title,
        "parent_section": chunk.parent_section,
        "unit_type": chunk.unit_type,
        "structural_locator": chunk.structural_locator,
        "token_count": chunk.token_count,
        "content_checksum": chunk.content_checksum,
        "origin_path": chunk.origin_path,
        "text": chunk.text,
        "decision": evaluation.decision,
        "selection_rank": evaluation.selection_rank,
        "selection_class": evaluation.selection_class,
        "selection_score": evaluation.score,
        "active_concepts": list(evaluation.active_concepts),
        "concept_scores": evaluation.concept_scores,
        "concept_positive_hits": {
            concept_id: list(hits) for concept_id, hits in evaluation.concept_positive_hits.items()
        },
        "concept_context_hits": {
            concept_id: list(hits) for concept_id, hits in evaluation.concept_context_hits.items()
        },
        "concept_exclusion_hits": {
            concept_id: list(hits) for concept_id, hits in evaluation.concept_exclusion_hits.items()
        },
        "source_include_rule_ids": list(evaluation.source_include_rule_ids),
        "source_include_pattern_hits": {
            rule_id: list(hits) for rule_id, hits in evaluation.source_include_pattern_hits.items()
        },
        "source_exclude_rule_ids": list(evaluation.source_exclude_rule_ids),
        "source_exclude_pattern_hits": {
            rule_id: list(hits) for rule_id, hits in evaluation.source_exclude_pattern_hits.items()
        },
        "hard_negative_categories": list(evaluation.hard_negative_categories),
        "hard_negative_targets": list(evaluation.hard_negative_targets),
        "global_exclusion_rule_ids": list(evaluation.global_exclusion_rule_ids),
        "exclusion_reasons": list(dict.fromkeys(evaluation.exclusion_reasons)),
        "review_flags": list(dict.fromkeys(evaluation.review_flags)),
        "structure_report_action": section_action,
        "quality": {
            "alphabetic_ratio": rounded(evaluation.quality.alphabetic_ratio),
            "digit_ratio": rounded(evaluation.quality.digit_ratio),
            "symbol_ratio": rounded(evaluation.quality.symbol_ratio),
            "single_character_token_ratio": rounded(
                evaluation.quality.single_character_token_ratio
            ),
            "mixed_script_token_ratio": rounded(evaluation.quality.mixed_script_token_ratio),
            "alphanumeric_intrusion_ratio": rounded(
                evaluation.quality.alphanumeric_intrusion_ratio
            ),
            "replacement_character_ratio": rounded(evaluation.quality.replacement_character_ratio),
            "ocr_noise_score": rounded(evaluation.quality.ocr_noise_score),
            "severe_ocr_noise": evaluation.quality.severe_ocr_noise,
            "report_ocr_noise_score": report_ocr_noise_score,
        },
        "parser_warnings": list(chunk.warnings),
        "selector_version": SELECTOR_VERSION,
    }


def atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    temporary_path.replace(path)


def atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def count_selected_concepts(selected: Sequence[Evaluation]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for evaluation in selected:
        for concept_id in evaluation.active_concepts:
            counts[concept_id] += 1
    return counts


def count_hard_negative_categories(
    selected: Sequence[Evaluation],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for evaluation in selected:
        for category in evaluation.hard_negative_categories:
            counts[category] += 1
    return counts


def count_exclusion_reasons(
    evaluations: Sequence[Evaluation],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for evaluation in evaluations:
        if evaluation.decision == "selected":
            continue
        for reason in dict.fromkeys(evaluation.exclusion_reasons):
            counts[reason] += 1
    return counts


def eligible_before_caps(evaluation: Evaluation) -> bool:
    if evaluation.decision == "selected":
        return True
    allowed_reasons = {
        "selection_cap:section",
        "selection_cap:source",
        "selection_cap:domain",
        "selection_target_reached",
        "lower_ranked_eligible_candidate",
    }
    return bool(evaluation.exclusion_reasons) and set(evaluation.exclusion_reasons).issubset(
        allowed_reasons
    )


def build_manifest(
    *,
    generated_at: str,
    project_root: Path,
    chunks_root: Path,
    candidate_manifest_path: Path,
    scope_path: Path,
    structure_report_path: Path,
    candidates_path: Path,
    excluded_path: Path,
    scope: ScopeConfiguration,
    expected_chunk_count: int,
    target_count: int,
    selected: Sequence[Evaluation],
    evaluations: Sequence[Evaluation],
    state: SelectionState,
    domain_targets: Mapping[str, int],
    domain_maximums: Mapping[str, int],
    allow_below_target: bool,
) -> dict[str, object]:
    selected_by_class = Counter(evaluation.selection_class for evaluation in selected)
    evaluated_by_class = Counter(evaluation.selection_class for evaluation in evaluations)
    selected_by_domain = Counter(evaluation.chunk.domain for evaluation in selected)
    selected_by_source = Counter(evaluation.chunk.source_id for evaluation in selected)
    selected_hard_negative_chunks = sum(
        bool(evaluation.hard_negative_categories) for evaluation in selected
    )
    warnings: list[str] = []
    if len(selected) < scope.target_minimum:
        warnings.append("Selected candidate count is below the approved minimum.")
    if selected_hard_negative_chunks < scope.hard_negative_minimum:
        warnings.append("Selected hard-negative chunk count is below the approved minimum.")

    return {
        "selector_version": SELECTOR_VERSION,
        "generated_at": generated_at,
        "status": "pre_review_candidates",
        "method": "deterministic_structural_lexical_selection",
        "embedding_independent": True,
        "scope": {
            "scope_version": scope.scope_version,
            "scope_status": scope.status,
            "corpus_version": scope.corpus_version,
            "concepts": list(scope.concepts),
            "domains": list(scope.domains),
            "methodological_constraints": scope.methodological_constraints,
        },
        "inputs": {
            "project_root": project_root.as_posix(),
            "chunks_root": chunks_root.as_posix(),
            "candidate_manifest": {
                "path": candidate_manifest_path.as_posix(),
                "sha256": sha256_file(candidate_manifest_path),
            },
            "scope_file": {
                "path": scope_path.as_posix(),
                "sha256": sha256_file(scope_path),
            },
            "structure_report": {
                "path": structure_report_path.as_posix(),
                "sha256": sha256_file(structure_report_path),
            },
        },
        "outputs": {
            "candidates": {
                "path": candidates_path.as_posix(),
                "sha256": sha256_file(candidates_path),
            },
            "excluded": {
                "path": excluded_path.as_posix(),
                "sha256": sha256_file(excluded_path),
            },
        },
        "selection_parameters": {
            "expected_chunk_count": expected_chunk_count,
            "approved_target_minimum": scope.target_minimum,
            "approved_target_maximum": scope.target_maximum,
            "target_count": target_count,
            "allow_below_target": allow_below_target,
            "minimum_alphabetic_ratio": scope.minimum_alphabetic_ratio,
            "maximum_ocr_noise_score": scope.maximum_ocr_noise_score,
            "hard_negative_minimum": scope.hard_negative_minimum,
            "hard_negative_target": scope.hard_negative_target,
            "hard_negative_maximum": scope.hard_negative_maximum,
            "domain_minimums": dict(scope.domain_minimums),
            "domain_targets": dict(domain_targets),
            "domain_maximums": dict(domain_maximums),
            "hard_negative_total_cap": scope.hard_negative_total_cap,
            "hard_negative_category_caps": dict(sorted(scope.hard_negative_category_caps.items())),
            "source_caps": {
                source_id: {
                    "maximum_pre_review_chunks": (source_rule.maximum_pre_review_chunks),
                    "maximum_chunks_per_section": (source_rule.maximum_chunks_per_section),
                }
                for source_id, source_rule in scope.source_rules.items()
            },
        },
        "counts": {
            "input_chunks": len(evaluations),
            "eligible_before_caps": sum(
                eligible_before_caps(evaluation) for evaluation in evaluations
            ),
            "selected_candidates": len(selected),
            "excluded_candidates": len(evaluations) - len(selected),
            "selected_by_domain": dict(sorted(selected_by_domain.items())),
            "selected_by_source": dict(sorted(selected_by_source.items())),
            "selected_by_class": dict(sorted(selected_by_class.items())),
            "evaluated_by_class": dict(sorted(evaluated_by_class.items())),
            "selected_by_concept": dict(sorted(count_selected_concepts(selected).items())),
            "selected_hard_negative_chunks": selected_hard_negative_chunks,
            "selected_hard_negative_categories": dict(
                sorted(count_hard_negative_categories(selected).items())
            ),
            "exclusion_reasons": dict(sorted(count_exclusion_reasons(evaluations).items())),
            "actual_domain_counts": dict(sorted(state.by_domain.items())),
            "actual_source_counts": dict(sorted(state.by_source.items())),
            "actual_hard_negative_category_counts": dict(
                sorted(state.by_hard_negative_category.items())
            ),
        },
        "warnings": warnings,
        "next_step": (
            "Generate and human-review the Phase 1 selection review packet. "
            "These records are not approved for embedding, activation, "
            "retrieval, or evaluation use."
        ),
    }


def prepare_output_paths(output_directory: Path, *, replace: bool) -> tuple[Path, Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    candidates_path = output_directory / CANDIDATES_FILENAME
    excluded_path = output_directory / EXCLUDED_FILENAME
    manifest_path = output_directory / MANIFEST_FILENAME
    existing = [path for path in (candidates_path, excluded_path, manifest_path) if path.exists()]
    if existing and not replace:
        raise SelectionError(
            "Selection outputs already exist. Use --replace: "
            + ", ".join(path.as_posix() for path in existing)
        )
    return candidates_path, excluded_path, manifest_path


def select_phase1_vertical_slice(
    *,
    project_root: Path,
    chunks_root: Path,
    candidate_manifest_path: Path,
    scope_path: Path,
    structure_report_path: Path,
    output_directory: Path,
    corpus_version: str,
    expected_chunk_count: int,
    requested_target_count: int | None,
    allow_below_target: bool,
    replace: bool,
) -> dict[str, object]:
    project_root = project_root.resolve()
    chunks_root = resolve_from_project(project_root, chunks_root)
    candidate_manifest_path = resolve_from_project(project_root, candidate_manifest_path)
    scope_path = resolve_from_project(project_root, scope_path)
    structure_report_path = resolve_from_project(project_root, structure_report_path)
    output_directory = resolve_from_project(project_root, output_directory)

    require_directory(chunks_root)
    require_file(candidate_manifest_path)
    require_file(scope_path)
    require_file(structure_report_path)

    scope = parse_scope_configuration(scope_path)
    if scope.corpus_version != corpus_version:
        raise SelectionError(
            "Scope corpus version mismatch: "
            f"expected {corpus_version!r}, found {scope.corpus_version!r}"
        )
    validate_candidate_manifest(
        candidate_manifest_path,
        expected_corpus_version=corpus_version,
        expected_chunk_count=expected_chunk_count,
    )
    structure_by_source_and_section, structure_by_unique_section = read_structure_report(
        structure_report_path
    )
    chunks = load_chunks(chunks_root, scope=scope)
    if len(chunks) != expected_chunk_count:
        raise SelectionError(
            f"Loaded chunk count mismatch: expected {expected_chunk_count}, loaded {len(chunks)}"
        )

    target_count = requested_target_count
    if target_count is None:
        target_count = (scope.target_minimum + scope.target_maximum) // 2
    if not scope.target_minimum <= target_count <= scope.target_maximum:
        raise SelectionError(
            f"target_count must be between {scope.target_minimum} and {scope.target_maximum}"
        )

    matcher = TermMatcher()
    evaluations: list[Evaluation] = []
    for chunk in chunks:
        section_info = structure_by_source_and_section.get((chunk.source_id, chunk.section_id))
        if section_info is None:
            section_info = structure_by_unique_section.get(chunk.section_id)
        evaluations.append(
            evaluate_chunk(
                chunk,
                section_info=section_info,
                scope=scope,
                matcher=matcher,
            )
        )

    selected, state, domain_targets, domain_maximums = select_candidates(
        evaluations, scope=scope, target_count=target_count
    )
    finalize_exclusion_reasons(
        evaluations,
        state=state,
        scope=scope,
        domain_maximums=domain_maximums,
        target_count=target_count,
    )

    if len(selected) < scope.target_minimum and not allow_below_target:
        raise SelectionError(
            f"Only {len(selected)} candidates could be selected; the approved "
            f"minimum is {scope.target_minimum}. Rerun with "
            "--allow-below-target to preserve diagnostic outputs."
        )

    candidates_path, excluded_path, manifest_path = prepare_output_paths(
        output_directory, replace=replace
    )
    excluded = [evaluation for evaluation in evaluations if evaluation.decision != "selected"]
    excluded.sort(
        key=lambda evaluation: (
            evaluation.chunk.source_id,
            evaluation.chunk.section_id,
            evaluation.chunk.chunk_id,
        )
    )
    atomic_write_jsonl(
        candidates_path,
        (evaluation_to_output(evaluation) for evaluation in selected),
    )
    atomic_write_jsonl(
        excluded_path,
        (evaluation_to_output(evaluation) for evaluation in excluded),
    )
    generated_at = datetime.now(UTC).isoformat()
    manifest = build_manifest(
        generated_at=generated_at,
        project_root=project_root,
        chunks_root=chunks_root,
        candidate_manifest_path=candidate_manifest_path,
        scope_path=scope_path,
        structure_report_path=structure_report_path,
        candidates_path=candidates_path,
        excluded_path=excluded_path,
        scope=scope,
        expected_chunk_count=expected_chunk_count,
        target_count=target_count,
        selected=selected,
        evaluations=evaluations,
        state=state,
        domain_targets=domain_targets,
        domain_maximums=domain_maximums,
        allow_below_target=allow_below_target,
    )
    atomic_write_json(manifest_path, manifest)

    LOGGER.info("Phase 1 vertical-slice selection completed")
    LOGGER.info("Selected candidates: %d", len(selected))
    LOGGER.info("Excluded candidates: %d", len(excluded))
    LOGGER.info("Selected by domain: %s", dict(sorted(state.by_domain.items())))
    LOGGER.info(
        "Outputs: %s, %s, %s",
        candidates_path,
        excluded_path,
        manifest_path,
    )
    return manifest


def main() -> int:
    arguments = parse_arguments()
    configure_logging(arguments.log_level)
    try:
        select_phase1_vertical_slice(
            project_root=arguments.project_root,
            chunks_root=arguments.chunks_root,
            candidate_manifest_path=arguments.candidate_manifest,
            scope_path=arguments.scope_file,
            structure_report_path=arguments.structure_report,
            output_directory=arguments.output_directory,
            corpus_version=arguments.corpus_version,
            expected_chunk_count=arguments.expected_chunk_count,
            requested_target_count=arguments.target_count,
            allow_below_target=arguments.allow_below_target,
            replace=arguments.replace,
        )
    except SelectionError:
        LOGGER.exception("Phase 1 vertical slice selection failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
