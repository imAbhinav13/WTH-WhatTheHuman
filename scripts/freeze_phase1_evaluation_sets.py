from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

LOGGER = logging.getLogger("wth.phase1.freeze_phase1_evaluation_sets")

SPLITTER_VERSION: Final = "1.1.0"
SPLIT_VERSION: Final = "phase1-evaluation-splits-v1"
CHECKSUM_ALGORITHM: Final = "sha256-canonical-jsonl-v1"

DEFAULT_GOLD_CORPUS: Final = Path("artifacts/phase1/reviewed/phase1_reviewed_gold_corpus.jsonl")
DEFAULT_REVIEW_MANIFEST: Final = Path("artifacts/phase1/reviewed/phase1_human_review_manifest.json")
DEFAULT_OUTPUT_DIRECTORY: Final = Path("data/evaluation")
DEFAULT_EXPECTED_COUNT: Final = 318

BUILD_FILENAME: Final = "phase1_build.jsonl"
DEVELOPMENT_FILENAME: Final = "phase1_development.jsonl"
HELDOUT_FILENAME: Final = "phase1_heldout.jsonl"
MANIFEST_FILENAME: Final = "phase1_split_manifest.json"

SPLITS: Final = ("build", "development", "heldout")
SPLIT_RATIOS: Final = {
    "build": 0.50,
    "development": 0.25,
    "heldout": 0.25,
}

HARD_NEGATIVE_SPLIT_RATIOS: Final = {
    "build": 0.50,
    "development": 0.25,
    "heldout": 0.25,
}

CONCEPTS: Final = (
    "consciousness",
    "self_identity",
    "reality_appearance",
)

VALID_GOLD_STATUSES: Final = {
    "phase1_reviewed_gold_candidate",
    "phase1_reviewed_gold",
}

OVERLAP_JACCARD_THRESHOLD: Final = 0.72
CROSS_SOURCE_DUPLICATE_THRESHOLD: Final = 0.94
MIN_OVERLAP_TOKEN_COUNT: Final = 12

EXPLICIT_FAMILY_FIELDS: Final = (
    "passage_family_id",
    "overlap_group_id",
    "verse_sequence_id",
    "paragraph_group_id",
    "duplicate_group_id",
    "translation_group_id",
)

SEQUENCE_FIELDS: Final = (
    "verse_number",
    "verse_range",
    "sutra_number",
    "sutra_range",
    "karika_number",
    "karika_range",
    "paragraph_number",
    "paragraph_range",
)


class SplitError(RuntimeError):
    """Raised when Phase 1 evaluation sets cannot be frozen safely."""


@dataclass(frozen=True)
class GoldRecord:
    index: int
    raw: dict[str, object]
    chunk_id: str
    source_id: str
    domain: str
    reviewed_text: str
    section_title: str
    structural_locator: str
    labels: dict[str, str]
    primary_concept: str
    hard_negative_category: str
    normalized_text: str
    token_set: frozenset[str]


@dataclass
class UnionFind:
    parent: list[int]
    rank: list[int]

    @classmethod
    def create(cls, size: int) -> UnionFind:
        return cls(parent=list(range(size)), rank=[0] * size)

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            self.parent[left_root] = right_root
            return
        if self.rank[left_root] > self.rank[right_root]:
            self.parent[right_root] = left_root
            return
        self.parent[right_root] = left_root
        self.rank[left_root] += 1


@dataclass(frozen=True)
class PassageFamily:
    family_id: str
    member_indices: tuple[int, ...]
    feature_counts: Counter[str]

    @property
    def size(self) -> int:
        return len(self.member_indices)


@dataclass
class SplitState:
    target_counts: dict[str, int]
    assigned_families: dict[str, list[PassageFamily]] = field(
        default_factory=lambda: {split: [] for split in SPLITS}
    )
    record_counts: Counter[str] = field(default_factory=Counter)
    feature_counts: dict[str, Counter[str]] = field(
        default_factory=lambda: {split: Counter() for split in SPLITS}
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze Phase 1 reviewed gold chunks into deterministic build, "
            "development, and held-out evaluation sets with leakage prevention."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--gold-corpus", type=Path, default=DEFAULT_GOLD_CORPUS)
    parser.add_argument("--review-manifest", type=Path, default=DEFAULT_REVIEW_MANIFEST)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--expected-count", type=int, default=DEFAULT_EXPECTED_COUNT)
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
        raise SplitError(f"Required file does not exist: {path}")


def require_mapping(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise SplitError(f"{description} must be an object.")
    result: dict[str, object] = {}
    for raw_key, nested_value in value.items():
        if not isinstance(raw_key, str):
            raise SplitError(f"{description} contains a non-string key.")
        result[raw_key] = nested_value
    return result


def require_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SplitError(f"{description} must be a non-empty string.")
    return value.strip()


def optional_string(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def text_tokens(value: str) -> frozenset[str]:
    return frozenset(token for token in normalize_text(value).split() if len(token) >= 2)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash exact file bytes for non-JSONL artifacts."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    """Return deterministic JSON bytes for semantic checksums."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_jsonl_content(path: Path) -> str:
    """Hash JSONL semantically and independently of LF/CRLF formatting.

    Object-key order and insignificant JSON whitespace are ignored.
    Record order remains significant.
    """

    digest = hashlib.sha256()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue

            try:
                value: object = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SplitError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc

            digest.update(canonical_json_bytes(value))
            digest.update(b"\n")

    return digest.hexdigest()


def load_json_mapping(path: Path) -> dict[str, object]:
    try:
        loaded: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SplitError(f"Invalid JSON in {path}: {exc}") from exc
    return require_mapping(loaded, f"JSON document {path}")


def iter_jsonl_mappings(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                loaded: object = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SplitError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            yield require_mapping(
                loaded,
                f"gold corpus record at {path}:{line_number}",
            )


def nested_mapping(mapping: Mapping[str, object], key: str) -> dict[str, object]:
    value = mapping.get(key)
    if value is None:
        return {}
    return require_mapping(value, key)


def find_first_string(value: object, keys: Sequence[str]) -> str:
    if isinstance(value, Mapping):
        mapping = require_mapping(value, "recursive mapping")
        for key in keys:
            candidate = optional_string(mapping.get(key))
            if candidate:
                return candidate
        for nested_value in mapping.values():
            result = find_first_string(nested_value, keys)
            if result:
                return result
    elif isinstance(value, list | tuple):
        for nested_value in value:
            result = find_first_string(nested_value, keys)
            if result:
                return result
    return ""


def parse_gold_record(raw: dict[str, object], *, index: int) -> GoldRecord:
    chunk_id = require_string(raw.get("chunk_id"), f"record {index} chunk_id")
    source_id = require_string(raw.get("source_id"), f"{chunk_id} source_id")
    domain = require_string(raw.get("domain"), f"{chunk_id} domain").casefold()
    reviewed_text = require_string(raw.get("reviewed_text"), f"{chunk_id} reviewed_text")

    status = optional_string(raw.get("corpus_status"))
    if status and status not in VALID_GOLD_STATUSES:
        raise SplitError(f"{chunk_id} has unexpected corpus_status {status!r}.")

    review = nested_mapping(raw, "review")
    labels_raw = nested_mapping(review, "labels")
    labels: dict[str, str] = {}
    for concept in CONCEPTS:
        labels[concept] = require_string(
            labels_raw.get(concept), f"{chunk_id} review.labels.{concept}"
        ).casefold()

    primary_concept = optional_string(review.get("primary_concept")).casefold()
    hard_negative_category = optional_string(review.get("hard_negative_category")).casefold()

    return GoldRecord(
        index=index,
        raw=raw,
        chunk_id=chunk_id,
        source_id=source_id,
        domain=domain,
        reviewed_text=reviewed_text,
        section_title=optional_string(raw.get("section_title")),
        structural_locator=optional_string(raw.get("structural_locator")),
        labels=labels,
        primary_concept=primary_concept,
        hard_negative_category=hard_negative_category,
        normalized_text=normalize_text(reviewed_text),
        token_set=text_tokens(reviewed_text),
    )


def load_gold_records(path: Path, *, expected_count: int) -> list[GoldRecord]:
    records = [
        parse_gold_record(raw, index=index) for index, raw in enumerate(iter_jsonl_mappings(path))
    ]
    if len(records) != expected_count:
        raise SplitError(f"Expected {expected_count} reviewed gold chunks, found {len(records)}.")
    chunk_ids = [record.chunk_id for record in records]
    if len(chunk_ids) != len(set(chunk_ids)):
        duplicates = [chunk_id for chunk_id, count in Counter(chunk_ids).items() if count > 1]
        raise SplitError("Duplicate chunk IDs in gold corpus: " + ", ".join(sorted(duplicates)))
    return records


def validate_review_manifest(path: Path, *, expected_count: int) -> dict[str, object]:
    manifest = load_json_mapping(path)
    if manifest.get("strict_gate_passed") is not True:
        raise SplitError("Phase 5 human-review manifest has not passed its strict gate.")
    status = optional_string(manifest.get("status"))
    if status != "phase1_human_review_complete":
        raise SplitError(f"Unexpected human-review manifest status: {status!r}")
    summary = require_mapping(manifest.get("summary"), "human-review manifest summary")
    approved = summary.get("approved_rows")
    if not isinstance(approved, int) or approved != expected_count:
        raise SplitError(
            "Human-review manifest approved count does not match expected count: "
            f"{approved!r} != {expected_count}."
        )
    return manifest


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    union = len(left | right)
    if union == 0:
        return 0.0
    return len(left & right) / union


def explicit_family_key(record: GoldRecord) -> str:
    for field_name in EXPLICIT_FAMILY_FIELDS:
        value = find_first_string(record.raw, (field_name,))
        if value:
            return f"explicit:{field_name}:{normalize_text(value)}"

    sequence_parts: list[str] = []
    for field_name in SEQUENCE_FIELDS:
        value = find_first_string(record.raw, (field_name,))
        if value:
            sequence_parts.append(f"{field_name}={normalize_text(value)}")
    if sequence_parts:
        return f"sequence:{record.source_id}:" + "|".join(sorted(sequence_parts))

    locator = normalize_text(record.structural_locator)
    if locator:
        return f"locator:{record.source_id}:{locator}"
    return ""


def record_features(record: GoldRecord) -> tuple[str, ...]:
    features: list[str] = [
        f"domain={record.domain}",
        f"source={record.source_id}",
    ]
    for concept in CONCEPTS:
        features.append(f"label:{concept}={record.labels[concept]}")
    if record.primary_concept:
        features.append(f"primary_concept={record.primary_concept}")
    if record.hard_negative_category:
        features.append(f"hard_negative={record.hard_negative_category}")
    return tuple(features)


def build_passage_families(
    records: Sequence[GoldRecord],
) -> tuple[list[PassageFamily], dict[str, object]]:
    union_find = UnionFind.create(len(records))
    reasons: Counter[str] = Counter()

    explicit_groups: defaultdict[str, list[int]] = defaultdict(list)
    exact_text_groups: defaultdict[str, list[int]] = defaultdict(list)

    for record in records:
        family_key = explicit_family_key(record)
        if family_key:
            explicit_groups[family_key].append(record.index)
        exact_text_groups[sha256_text(record.normalized_text)].append(record.index)

    for members in explicit_groups.values():
        if len(members) < 2:
            continue
        first = members[0]
        for member in members[1:]:
            union_find.union(first, member)
            reasons["explicit_or_locator_family"] += 1

    for members in exact_text_groups.values():
        if len(members) < 2:
            continue
        first = members[0]
        for member in members[1:]:
            union_find.union(first, member)
            reasons["exact_text_duplicate"] += 1

    by_source_section: defaultdict[tuple[str, str], list[GoldRecord]] = defaultdict(list)
    for record in records:
        section_key = normalize_text(record.section_title)
        if section_key:
            by_source_section[(record.source_id, section_key)].append(record)

    for section_records in by_source_section.values():
        for left_offset, left in enumerate(section_records):
            if len(left.token_set) < MIN_OVERLAP_TOKEN_COUNT:
                continue
            for right in section_records[left_offset + 1 :]:
                if len(right.token_set) < MIN_OVERLAP_TOKEN_COUNT:
                    continue
                if jaccard(left.token_set, right.token_set) >= OVERLAP_JACCARD_THRESHOLD:
                    union_find.union(left.index, right.index)
                    reasons["same_section_text_overlap"] += 1

    for left_offset, left in enumerate(records):
        if len(left.token_set) < MIN_OVERLAP_TOKEN_COUNT:
            continue
        for right in records[left_offset + 1 :]:
            if left.source_id == right.source_id:
                continue
            if len(right.token_set) < MIN_OVERLAP_TOKEN_COUNT:
                continue
            if jaccard(left.token_set, right.token_set) >= CROSS_SOURCE_DUPLICATE_THRESHOLD:
                union_find.union(left.index, right.index)
                reasons["cross_source_near_duplicate"] += 1

    grouped_indices: defaultdict[int, list[int]] = defaultdict(list)
    for record in records:
        grouped_indices[union_find.find(record.index)].append(record.index)

    families: list[PassageFamily] = []
    for members in grouped_indices.values():
        sorted_members = tuple(sorted(members))
        family_hash = hashlib.sha256(
            "|".join(records[index].chunk_id for index in sorted_members).encode("utf-8")
        ).hexdigest()[:20]
        feature_counts: Counter[str] = Counter()
        for index in sorted_members:
            for feature in record_features(records[index]):
                feature_counts[feature] += 1
        families.append(
            PassageFamily(
                family_id=f"pf_{family_hash}",
                member_indices=sorted_members,
                feature_counts=feature_counts,
            )
        )

    families.sort(key=lambda family: (-family.size, family.family_id))
    diagnostics: dict[str, object] = {
        "family_count": len(families),
        "multi_record_family_count": sum(family.size > 1 for family in families),
        "largest_family_size": max((family.size for family in families), default=0),
        "link_reasons": dict(sorted(reasons.items())),
    }
    return families, diagnostics


def calculate_target_counts(total: int) -> dict[str, int]:
    raw = {split: total * SPLIT_RATIOS[split] for split in SPLITS}
    target = {split: math.floor(raw[split]) for split in SPLITS}
    remaining = total - sum(target.values())
    remainder_order = sorted(
        SPLITS,
        key=lambda split: (-(raw[split] - target[split]), SPLITS.index(split)),
    )
    for split in remainder_order[:remaining]:
        target[split] += 1
    return target


def global_feature_counts(records: Sequence[GoldRecord]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        for feature in record_features(record):
            counts[feature] += 1
    return counts


def feature_weight(
    feature: str,
) -> float:
    if feature.startswith("domain="):
        return 12.0

    if feature.startswith("label:"):
        return 8.0

    if feature.startswith("hard_negative="):
        return 10.0

    if feature.startswith("primary_concept="):
        return 6.0

    if feature.startswith("source="):
        return 4.0

    return 1.0


def validate_hard_negative_distribution(
    split_records: Mapping[
        str,
        Sequence[GoldRecord],
    ],
) -> dict[str, object]:
    """Validate hard-negative availability across evaluation splits."""

    counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}

    totals: Counter[str] = Counter()

    for split in SPLITS:
        for record in split_records[split]:
            category = record.hard_negative_category

            if not category:
                continue

            counts[split][category] += 1
            totals[split] += 1

    if totals["build"] < 20:
        raise SplitError(
            f"Build set has too few hard negatives for prototype construction: {totals['build']}"
        )

    if totals["development"] < 10:
        raise SplitError(f"Development set has too few hard negatives: {totals['development']}")

    if totals["heldout"] < 10:
        raise SplitError(f"Held-out set has too few hard negatives: {totals['heldout']}")

    all_categories = {category for split_counts in counts.values() for category in split_counts}

    missing_build_categories = sorted(
        category for category in all_categories if counts["build"][category] == 0
    )

    if missing_build_categories:
        raise SplitError(
            "Build set lacks hard-negative categories: " + ", ".join(missing_build_categories)
        )

    return {
        "totals": {split: totals[split] for split in SPLITS},
        "categories": {split: dict(sorted(counts[split].items())) for split in SPLITS},
    }


def assignment_score(
    family: PassageFamily,
    *,
    split: str,
    state: SplitState,
    global_features: Counter[str],
    total_records: int,
) -> float:
    target_size = state.target_counts[split]
    projected_size = state.record_counts[split] + family.size
    overflow = max(0, projected_size - target_size)
    size_difference = abs(projected_size - target_size)
    score = overflow * 75.0 + (size_difference / max(1, target_size)) * 8.0
    split_ratio = target_size / total_records

    for feature, family_count in family.feature_counts.items():
        target_feature = global_features[feature] * split_ratio
        projected_feature = state.feature_counts[split][feature] + family_count
        difference = abs(projected_feature - target_feature)
        normalized_difference = difference / max(1.0, target_feature)
        score += normalized_difference * feature_weight(feature)
    return score


def assign_families(
    families: Sequence[PassageFamily],
    *,
    records: Sequence[GoldRecord],
) -> SplitState:
    """Assign passage families using constraint-first stratification."""

    target_counts = calculate_target_counts(len(records))

    state = SplitState(target_counts=target_counts)

    # Stage 1:
    # Reserve hard negatives before generic allocation.
    reserved_family_ids = reserve_hard_negative_families(
        families=families,
        state=state,
    )

    remaining_families = [
        family for family in families if family.family_id not in reserved_family_ids
    ]

    global_features = global_feature_counts(records)

    feature_rarity = {feature: 1.0 / max(1, count) for feature, count in global_features.items()}

    ordered_families = sorted(
        remaining_families,
        key=lambda family: (
            -family.size,
            -sum(
                feature_rarity[feature] * count for feature, count in family.feature_counts.items()
            ),
            family.family_id,
        ),
    )

    # Stage 2:
    # Allocate remaining families while balancing
    # domain, concept labels and source.
    for family in ordered_families:
        available_splits = [
            split
            for split in SPLITS
            if (state.record_counts[split] + family.size <= state.target_counts[split])
        ]

        if not available_splits:
            available_splits = list(SPLITS)

        scored_splits = [
            (
                assignment_score(
                    family,
                    split=split,
                    state=state,
                    global_features=global_features,
                    total_records=len(records),
                ),
                SPLITS.index(split),
                split,
            )
            for split in available_splits
        ]

        _, _, chosen_split = min(scored_splits)

        state.assigned_families[chosen_split].append(family)

        state.record_counts[chosen_split] += family.size

        state.feature_counts[chosen_split].update(family.feature_counts)

    return state


def build_record_assignments(state: SplitState) -> dict[int, tuple[str, str]]:
    assignments: dict[int, tuple[str, str]] = {}
    for split in SPLITS:
        for family in state.assigned_families[split]:
            for index in family.member_indices:
                if index in assignments:
                    raise SplitError(f"Record index {index} assigned twice.")
                assignments[index] = (split, family.family_id)
    return assignments


def validate_leakage(
    records: Sequence[GoldRecord],
    assignments: Mapping[int, tuple[str, str]],
) -> dict[str, object]:
    if len(assignments) != len(records):
        raise SplitError("Not all gold records received a split assignment.")

    family_to_split: dict[str, str] = {}
    normalized_text_to_split: dict[str, str] = {}

    for record in records:
        split, family_id = assignments[record.index]
        existing_family_split = family_to_split.get(family_id)
        if existing_family_split is not None and existing_family_split != split:
            raise SplitError(f"Passage family {family_id} crosses splits.")
        family_to_split[family_id] = split

        text_hash = sha256_text(record.normalized_text)
        existing_text_split = normalized_text_to_split.get(text_hash)
        if existing_text_split is not None and existing_text_split != split:
            raise SplitError("Exact duplicate reviewed text crosses splits.")
        normalized_text_to_split[text_hash] = split

    cross_split_overlap_pairs = 0
    for left_offset, left in enumerate(records):
        left_split, _ = assignments[left.index]
        if len(left.token_set) < MIN_OVERLAP_TOKEN_COUNT:
            continue
        for right in records[left_offset + 1 :]:
            right_split, _ = assignments[right.index]
            if left_split == right_split:
                continue
            if len(right.token_set) < MIN_OVERLAP_TOKEN_COUNT:
                continue
            threshold = (
                OVERLAP_JACCARD_THRESHOLD
                if left.source_id == right.source_id
                else CROSS_SOURCE_DUPLICATE_THRESHOLD
            )
            if jaccard(left.token_set, right.token_set) >= threshold:
                cross_split_overlap_pairs += 1

    if cross_split_overlap_pairs:
        raise SplitError(
            f"Detected {cross_split_overlap_pairs} high-overlap passage pair(s) crossing splits."
        )

    return {
        "family_leakage_count": 0,
        "exact_text_leakage_count": 0,
        "high_overlap_cross_split_pairs": 0,
    }


def enriched_record(record: GoldRecord, *, split: str, family_id: str) -> dict[str, object]:
    result = dict(record.raw)
    result["evaluation_split"] = split
    result["evaluation_split_version"] = SPLIT_VERSION
    result["passage_family_id"] = family_id
    result["frozen"] = True

    if split == "build":
        result["dataset_role"] = "prototype_creation_prompt_development_debugging"
        result["read_only"] = False
    elif split == "development":
        result["dataset_role"] = "threshold_calibration_weighting_topk_reranking"
        result["read_only"] = False
    else:
        result["dataset_role"] = "final_evaluation_only"
        result["read_only"] = True
        result["heldout_policy"] = (
            "Must not influence prototype construction, anchor wording, thresholds, "
            "feature selection, or post-lock model choice."
        )
    return result


def atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    temporary.replace(path)


def atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def distribution_for_records(records: Sequence[GoldRecord]) -> dict[str, object]:
    domain = Counter(record.domain for record in records)
    source = Counter(record.source_id for record in records)
    primary = Counter(record.primary_concept for record in records if record.primary_concept)
    hard_negative = Counter(
        record.hard_negative_category for record in records if record.hard_negative_category
    )
    concept_labels: dict[str, dict[str, int]] = {}
    for concept in CONCEPTS:
        concept_labels[concept] = dict(
            sorted(Counter(record.labels[concept] for record in records).items())
        )
    return {
        "count": len(records),
        "by_domain": dict(sorted(domain.items())),
        "by_source": dict(sorted(source.items())),
        "by_primary_concept": dict(sorted(primary.items())),
        "concept_labels": concept_labels,
        "hard_negative_categories": dict(sorted(hard_negative.items())),
    }


def prepare_output_paths(output_directory: Path, *, replace: bool) -> dict[str, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "build": output_directory / BUILD_FILENAME,
        "development": output_directory / DEVELOPMENT_FILENAME,
        "heldout": output_directory / HELDOUT_FILENAME,
        "manifest": output_directory / MANIFEST_FILENAME,
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not replace:
        raise SplitError(
            "Evaluation split outputs already exist. Use --replace: "
            + ", ".join(path.as_posix() for path in existing)
        )
    return paths


def freeze_phase1_evaluation_sets(
    *,
    project_root: Path,
    gold_corpus_path: Path,
    review_manifest_path: Path,
    output_directory: Path,
    expected_count: int,
    replace: bool,
) -> dict[str, object]:
    project_root = project_root.resolve()
    gold_corpus_path = resolve_from_project(project_root, gold_corpus_path)
    review_manifest_path = resolve_from_project(project_root, review_manifest_path)
    output_directory = resolve_from_project(project_root, output_directory)

    require_file(gold_corpus_path)
    require_file(review_manifest_path)

    review_manifest = validate_review_manifest(review_manifest_path, expected_count=expected_count)
    records = load_gold_records(gold_corpus_path, expected_count=expected_count)

    families, family_diagnostics = build_passage_families(records)
    state = assign_families(families, records=records)
    assignments = build_record_assignments(state)
    leakage = validate_leakage(records, assignments)

    paths = prepare_output_paths(output_directory, replace=replace)
    split_records: dict[str, list[GoldRecord]] = {split: [] for split in SPLITS}
    for record in records:
        split, _ = assignments[record.index]
        split_records[split].append(record)

    hard_negative_distribution = validate_hard_negative_distribution(split_records)

    for split in SPLITS:
        split_records[split].sort(
            key=lambda record: (record.domain, record.source_id, record.chunk_id)
        )
        atomic_write_jsonl(
            paths[split],
            (
                enriched_record(
                    record,
                    split=split,
                    family_id=assignments[record.index][1],
                )
                for record in split_records[split]
            ),
        )

    target_counts = calculate_target_counts(len(records))
    actual_counts = {split: len(split_records[split]) for split in SPLITS}
    manifest: dict[str, object] = {
        "splitter_version": SPLITTER_VERSION,
        "split_version": SPLIT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "frozen",
        "frozen": True,
        "checksum_policy": {
            "jsonl_algorithm": CHECKSUM_ALGORITHM,
            "jsonl_semantics": (
                "Canonical JSON per non-empty record; object keys sorted; "
                "insignificant JSON whitespace ignored; LF/CRLF differences "
                "ignored; record order preserved."
            ),
            "non_jsonl_algorithm": "sha256-file-bytes",
        },
        "input": {
            "gold_corpus": {
                "path": gold_corpus_path.as_posix(),
                "sha256": sha256_jsonl_content(gold_corpus_path),
                "sha256_algorithm": CHECKSUM_ALGORITHM,
                "record_count": len(records),
            },
            "human_review_manifest": {
                "path": review_manifest_path.as_posix(),
                "sha256": sha256_file(review_manifest_path),
                "sha256_algorithm": "sha256-file-bytes",
                "status": optional_string(review_manifest.get("status")),
                "strict_gate_passed": review_manifest.get("strict_gate_passed"),
            },
        },
        "split_policy": {
            "ratios": SPLIT_RATIOS,
            "hard_negative_ratios": HARD_NEGATIVE_SPLIT_RATIOS,
            "target_counts": target_counts,
            "actual_counts": actual_counts,
            "stratification_features": [
                "domain",
                "source",
                "concept label",
                "primary concept",
                "hard-negative category",
            ],
            "deterministic": True,
            "random_sampling": False,
            "passage_family_policy": {
                "explicit_family_fields": list(EXPLICIT_FAMILY_FIELDS),
                "sequence_fields": list(SEQUENCE_FIELDS),
                "same_source_section_jaccard_threshold": OVERLAP_JACCARD_THRESHOLD,
                "cross_source_duplicate_jaccard_threshold": CROSS_SOURCE_DUPLICATE_THRESHOLD,
            },
            "heldout_policy": {
                "read_only": True,
                "allowed_use": "final evaluation only",
                "prohibited_influence": [
                    "anchor wording",
                    "prototype construction",
                    "threshold tuning",
                    "feature selection",
                    "ambiguity-margin tuning",
                    "top-k tuning",
                    "reranking tuning",
                    "post-lock model choice",
                ],
            },
        },
        "passage_families": family_diagnostics,
        "leakage_validation": leakage,
        "hard_negative_distribution": hard_negative_distribution,
        "distribution_report": {
            split: distribution_for_records(split_records[split]) for split in SPLITS
        },
        "outputs": {
            split: {
                "path": paths[split].as_posix(),
                "sha256": sha256_jsonl_content(paths[split]),
                "sha256_algorithm": CHECKSUM_ALGORITHM,
                "record_count": len(split_records[split]),
                "read_only": split == "heldout",
            }
            for split in SPLITS
        },
        "exit_gate": {
            "splits_checksummed": True,
            "canonical_jsonl_checksums": True,
            "heldout_marked_read_only": True,
            "passage_family_leakage": False,
            "exact_duplicate_leakage": False,
            "high_overlap_leakage": False,
            "distribution_report_generated": True,
            "hard_negative_distribution_validated": True,
        },
        "next_step": (
            "Construct Phase 1 concept prototypes from the frozen Build set only. "
            "Development remains reserved for calibration and Held-out remains "
            "untouched until final evaluation."
        ),
    }
    atomic_write_json(paths["manifest"], manifest)

    LOGGER.info("Phase 1 evaluation sets frozen successfully")
    LOGGER.info("Target counts: %s", target_counts)
    LOGGER.info("Actual counts: %s", actual_counts)
    LOGGER.info(
        "Passage families: %d total, %d multi-record",
        family_diagnostics["family_count"],
        family_diagnostics["multi_record_family_count"],
    )
    LOGGER.info("Leakage validation: PASS")
    LOGGER.info("Checksum algorithm: %s", CHECKSUM_ALGORITHM)
    LOGGER.info("Build checksum: %s", sha256_jsonl_content(paths["build"]))
    LOGGER.info(
        "Development checksum: %s",
        sha256_jsonl_content(paths["development"]),
    )
    LOGGER.info("Held-out checksum: %s", sha256_jsonl_content(paths["heldout"]))
    LOGGER.info("Build: %s", paths["build"])
    LOGGER.info("Development: %s", paths["development"])
    LOGGER.info("Held-out: %s", paths["heldout"])
    LOGGER.info("Manifest: %s", paths["manifest"])
    return manifest


def calculate_feature_targets(
    total: int,
    *,
    ratios: Mapping[str, float],
) -> dict[str, int]:
    """Calculate deterministic integer targets across splits."""

    raw_targets = {split: total * ratios[split] for split in SPLITS}

    targets = {split: math.floor(raw_targets[split]) for split in SPLITS}

    remainder = total - sum(targets.values())

    ordered = sorted(
        SPLITS,
        key=lambda split: (
            -(raw_targets[split] - targets[split]),
            SPLITS.index(split),
        ),
    )

    for split in ordered[:remainder]:
        targets[split] += 1

    return targets


def family_hard_negative_categories(
    family: PassageFamily,
) -> tuple[str, ...]:
    """Return hard-negative categories represented by a family."""

    categories = {
        feature.split("=", maxsplit=1)[1]
        for feature in family.feature_counts
        if feature.startswith("hard_negative=")
    }

    return tuple(sorted(categories))


def reserve_hard_negative_families(
    *,
    families: Sequence[PassageFamily],
    state: SplitState,
) -> set[str]:
    """Reserve hard-negative families across all splits before general allocation."""

    hard_negative_families = [
        family for family in families if family_hard_negative_categories(family)
    ]

    if not hard_negative_families:
        return set()

    total_hard_negative_records = sum(family.size for family in hard_negative_families)

    total_targets = calculate_feature_targets(
        total_hard_negative_records,
        ratios=HARD_NEGATIVE_SPLIT_RATIOS,
    )

    categories = sorted(
        {
            category
            for family in hard_negative_families
            for category in family_hard_negative_categories(family)
        }
    )

    category_totals: Counter[str] = Counter()

    for family in hard_negative_families:
        for category in family_hard_negative_categories(family):
            category_totals[category] += family.size

    category_targets = {
        category: calculate_feature_targets(
            category_totals[category],
            ratios=HARD_NEGATIVE_SPLIT_RATIOS,
        )
        for category in categories
    }

    assigned_ids: set[str] = set()
    hard_negative_counts: Counter[str] = Counter()

    category_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}

    ordered = sorted(
        hard_negative_families,
        key=lambda family: (
            -family.size,
            len(family_hard_negative_categories(family)),
            family.family_id,
        ),
    )

    for family in ordered:
        family_categories = family_hard_negative_categories(family)

        eligible_splits = [
            split
            for split in SPLITS
            if (state.record_counts[split] + family.size <= state.target_counts[split])
        ]

        if not eligible_splits:
            eligible_splits = list(SPLITS)

        def split_score(
            split: str,
            *,
            current_family: PassageFamily = family,
            current_categories: tuple[str, ...] = family_categories,
        ) -> tuple[float, float, float, int]:
            """Prefer splits with the largest unmet hard-negative targets."""

            hard_negative_target = total_targets[split]
            hard_negative_current = hard_negative_counts[split]

            if hard_negative_target > 0:
                hard_negative_fill_ratio = hard_negative_current / hard_negative_target
            else:
                hard_negative_fill_ratio = 1.0

            category_fill_ratios: list[float] = []

            for category in current_categories:
                category_target = category_targets[category][split]

                category_current = category_counts[split][category]

                if category_target > 0:
                    category_fill_ratios.append(category_current / category_target)
                else:
                    category_fill_ratios.append(1.0)

            average_category_fill = (
                sum(category_fill_ratios) / len(category_fill_ratios)
                if category_fill_ratios
                else 1.0
            )

            projected_total = state.record_counts[split] + current_family.size

            split_capacity = state.target_counts[split]

            capacity_fill_ratio = projected_total / split_capacity if split_capacity > 0 else 1.0

            return (
                hard_negative_fill_ratio,
                average_category_fill,
                capacity_fill_ratio,
                SPLITS.index(split),
            )

        chosen_split = min(
            eligible_splits,
            key=split_score,
        )

        state.assigned_families[chosen_split].append(family)

        state.record_counts[chosen_split] += family.size

        state.feature_counts[chosen_split].update(family.feature_counts)

        hard_negative_counts[chosen_split] += family.size

        for category in family_categories:
            category_counts[chosen_split][category] += family.size

        assigned_ids.add(family.family_id)

    return assigned_ids


def main() -> int:
    arguments = parse_arguments()
    configure_logging(arguments.log_level)
    try:
        freeze_phase1_evaluation_sets(
            project_root=arguments.project_root,
            gold_corpus_path=arguments.gold_corpus,
            review_manifest_path=arguments.review_manifest,
            output_directory=arguments.output_directory,
            expected_count=arguments.expected_count,
            replace=arguments.replace,
        )
    except SplitError:
        LOGGER.exception("Phase 1 evaluation-set freezing failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
