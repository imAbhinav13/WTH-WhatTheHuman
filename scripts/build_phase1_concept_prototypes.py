from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import yaml

LOGGER = logging.getLogger("wth.phase1.build_phase1_concept_prototypes")

PROTOTYPE_VERSION: Final = "phase1-prototype-v2"

DEFAULT_BUILD_PATH: Final = Path("data/evaluation/phase1_build.jsonl")
DEFAULT_SPLIT_MANIFEST: Final = Path("data/evaluation/phase1_split_manifest.json")
DEFAULT_OUTPUT_PATH: Final = Path("data/concepts/phase1_concept_prototypes.yaml")

CONCEPTS: Final = (
    "consciousness",
    "self_identity",
    "reality_appearance",
)

DOMAINS: Final = (
    "science",
    "advaita",
    "samkhya",
)

VALID_LABELS: Final = {
    "positive",
    "partial",
    "negative",
    "uncertain",
}

POSITIVE_PROTOTYPES_PER_DOMAIN: Final = 3
POSITIVE_PROTOTYPES_PER_CONCEPT: Final = POSITIVE_PROTOTYPES_PER_DOMAIN * len(DOMAINS)

# Conservative lexical guards used only to prevent a passage that is explicitly
# explaining another framework from becoming a canonical domain prototype.
# These rules do not relabel the corpus and do not affect retrieval/evaluation.
DOMAIN_FOREIGN_MARKERS: Final[dict[str, tuple[str, ...]]] = {
    "science": (),
    "advaita": (
        "samkhya",
        "sankhya",
    ),
    "samkhya": (
        "vedanta",
        "vedantin",
        "vedantins",
        "vedantist",
        "vedantists",
        "vedantic",
        "sankara",
        "shankara",
    ),
}

DOMAIN_FOREIGN_SUPPORT_MARKERS: Final[dict[str, tuple[str, ...]]] = {
    "science": (),
    "advaita": (
        "purusha",
        "prakriti",
        "ahamkara",
        "buddhi",
    ),
    "samkhya": (
        "brahman",
        "maya",
        "avidya",
        "isvara",
        "jiva",
        "atman",
    ),
}

MIN_HARD_NEGATIVES_PER_CONCEPT: Final = 3
MAX_HARD_NEGATIVES_PER_CONCEPT: Final = 6


QUESTION_PROTOTYPES: Final[dict[str, tuple[str, ...]]] = {
    "consciousness": (
        "What makes an experience conscious?",
        "Is awareness distinct from thought or cognition?",
        "What is the difference between consciousness and the mental processes that appear within it?",
        "How is witnessing awareness related to perception and experience?",
        "Can cognition occur without conscious awareness?",
        "What distinguishes a conscious subject from an information-processing system?",
    ),
    "self_identity": (
        "What is the self?",
        "What makes a person or subject the same self across experience?",
        "Is the ego the same as the true self?",
        "How is the self related to body, mind, and first-person perspective?",
        "How does Atman differ from ego or personality?",
        "How does Purusha differ from ahamkara and the individual personality?",
    ),
    "reality_appearance": (
        "What is the difference between reality and appearance?",
        "When does perception present an appearance rather than reality itself?",
        "How can something appear real while being illusory or misperceived?",
        "How does superimposition affect what is taken to be real?",
        "How is the experienced world related to representation or perceptual construction?",
        "Does a cosmological description by itself explain the distinction between reality and appearance?",
    ),
}


ADJACENT_CONCEPTS: Final[dict[str, tuple[str, ...]]] = {
    "consciousness": (
        "cognition_without_experience",
        "attention_without_awareness",
        "intellect_or_buddhi_as_if_conscious",
        "ego_or_self_reference_without_witnessing_awareness",
    ),
    "self_identity": (
        "ego_or_ahamkara",
        "personality_or_autobiographical_identity",
        "body_representation_without_claim_about_selfhood",
        "advaita_atman_treated_as_identical_to_samkhya_purusha",
    ),
    "reality_appearance": (
        "cosmology_without_reality_appearance_claim",
        "perceptual_description_without_ontological_contrast",
        "representation_without_illusion_or_reality_contrast",
        "manifestation_sequence_without_appearance_analysis",
    ),
}


class PrototypeBuildError(RuntimeError):
    """Raised when Phase 1 prototypes cannot be constructed safely."""


@dataclass(frozen=True)
class BuildRecord:
    chunk_id: str
    source_id: str
    domain: str
    reviewed_text: str
    labels: dict[str, str]
    primary_concept: str
    secondary_concepts: tuple[str, ...]
    hard_negative_for: tuple[str, ...]
    hard_negative_category: str
    citation_verified: str
    text_quality_status: str
    ocr_review_status: str


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct Phase 1 concept prototypes strictly from the frozen "
            "Build set. Development and Held-out data are never read."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--build",
        type=Path,
        default=DEFAULT_BUILD_PATH,
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace an existing prototype artifact.",
    )
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


def resolve_from_project(
    project_root: Path,
    path: Path,
) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise PrototypeBuildError(f"Required file does not exist: {path}")


def require_mapping(
    value: object,
    description: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise PrototypeBuildError(f"{description} must be an object.")

    result: dict[str, object] = {}
    for raw_key, nested_value in value.items():
        if not isinstance(raw_key, str):
            raise PrototypeBuildError(f"{description} contains a non-string key.")
        result[raw_key] = nested_value
    return result


def require_string(
    value: object,
    description: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PrototypeBuildError(f"{description} must be a non-empty string.")
    return value.strip()


def optional_string(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def parse_string_list(value: object) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(
            item.strip().casefold() for item in value if isinstance(item, str) and item.strip()
        )

    if isinstance(value, str) and value.strip():
        normalized = value.replace(";", "|").replace(",", "|")
        return tuple(part.strip().casefold() for part in normalized.split("|") if part.strip())

    return ()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha256_jsonl_content(path: Path) -> str:
    """Hash JSONL semantically, independent of LF/CRLF formatting."""

    digest = hashlib.sha256()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue

            try:
                value: object = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise PrototypeBuildError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc

            canonical = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            digest.update(canonical.encode("utf-8"))
            digest.update(b"\n")

    return digest.hexdigest()


def load_json_mapping(path: Path) -> dict[str, object]:
    try:
        loaded: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PrototypeBuildError(f"Invalid JSON in {path}: {exc}") from exc

    return require_mapping(
        loaded,
        f"JSON document {path}",
    )


def iter_jsonl_mappings(
    path: Path,
) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(
            handle,
            start=1,
        ):
            stripped = raw_line.strip()
            if not stripped:
                continue

            try:
                loaded: object = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise PrototypeBuildError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc

            yield require_mapping(
                loaded,
                f"record at {path}:{line_number}",
            )


def validate_split_manifest(
    path: Path,
    *,
    build_path: Path,
) -> dict[str, object]:
    manifest = load_json_mapping(path)

    if manifest.get("status") != "frozen":
        raise PrototypeBuildError("Phase 6 split manifest is not frozen.")

    if manifest.get("frozen") is not True:
        raise PrototypeBuildError("Phase 6 split manifest does not have frozen=true.")

    outputs = require_mapping(
        manifest.get("outputs"),
        "split manifest outputs",
    )
    build_output = require_mapping(
        outputs.get("build"),
        "split manifest build output",
    )

    expected_sha = require_string(
        build_output.get("sha256"),
        "build output sha256",
    )
    actual_sha = sha256_jsonl_content(build_path)

    if actual_sha != expected_sha:
        raise PrototypeBuildError(
            "Build-set checksum does not match the frozen Phase 6 "
            "manifest. Do not construct prototypes from a modified split."
        )

    development_output = require_mapping(
        outputs.get("development"),
        "split manifest development output",
    )
    heldout_output = require_mapping(
        outputs.get("heldout"),
        "split manifest heldout output",
    )

    if heldout_output.get("read_only") is not True:
        raise PrototypeBuildError("Held-out set is not marked read-only.")

    return {
        "split_version": optional_string(manifest.get("split_version")),
        "build_sha256": expected_sha,
        "development_sha256": require_string(
            development_output.get("sha256"),
            "development output sha256",
        ),
        "heldout_sha256": require_string(
            heldout_output.get("sha256"),
            "heldout output sha256",
        ),
    }


def parse_build_record(
    raw: Mapping[str, object],
) -> BuildRecord:
    split = require_string(
        raw.get("evaluation_split"),
        "evaluation_split",
    )
    if split != "build":
        raise PrototypeBuildError(f"Prototype input contains non-Build record: {split!r}")

    review = require_mapping(
        raw.get("review"),
        "review",
    )
    labels_raw = require_mapping(
        review.get("labels"),
        "review.labels",
    )

    labels: dict[str, str] = {}
    for concept in CONCEPTS:
        label = require_string(
            labels_raw.get(concept),
            f"review.labels.{concept}",
        ).casefold()

        if label not in VALID_LABELS:
            raise PrototypeBuildError(f"Invalid label {label!r} for {concept}.")

        labels[concept] = label

    domain = require_string(
        raw.get("domain"),
        "domain",
    ).casefold()
    if domain not in DOMAINS:
        raise PrototypeBuildError(f"Unknown domain in Build set: {domain!r}")

    return BuildRecord(
        chunk_id=require_string(
            raw.get("chunk_id"),
            "chunk_id",
        ),
        source_id=require_string(
            raw.get("source_id"),
            "source_id",
        ),
        domain=domain,
        reviewed_text=require_string(
            raw.get("reviewed_text"),
            "reviewed_text",
        ),
        labels=labels,
        primary_concept=optional_string(review.get("primary_concept")).casefold(),
        secondary_concepts=parse_string_list(review.get("secondary_concepts")),
        hard_negative_for=parse_string_list(review.get("hard_negative_for")),
        hard_negative_category=optional_string(review.get("hard_negative_category")).casefold(),
        citation_verified=optional_string(review.get("citation_verified")).casefold(),
        text_quality_status=optional_string(review.get("text_quality_status")).casefold(),
        ocr_review_status=optional_string(review.get("ocr_review_status")).casefold(),
    )


def load_build_records(
    path: Path,
) -> list[BuildRecord]:
    records = [parse_build_record(raw) for raw in iter_jsonl_mappings(path)]

    if not records:
        raise PrototypeBuildError("Build set is empty.")

    chunk_ids = [record.chunk_id for record in records]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise PrototypeBuildError("Build set contains duplicate chunk IDs.")

    return records


def quality_eligible(record: BuildRecord) -> bool:
    return (
        record.citation_verified == "yes"
        and record.text_quality_status in {"acceptable", "needs_edit"}
        and record.ocr_review_status in {"clean", "acceptable", "corrected"}
    )


def normalized_search_text(value: str) -> str:
    """Normalize text for conservative domain-fidelity checks.

    This removes Sanskrit/Indic transliteration diacritics and joins
    apostrophe-separated transliterations so forms such as:

    Vedântists -> vedantists
    mâyâ       -> maya
    jîva       -> jiva
    Îs'vara    -> isvara
    S'ankara   -> sankara

    can be matched deterministically.
    """

    normalized = unicodedata.normalize(
        "NFKD",
        value,
    )

    without_diacritics = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )

    folded = without_diacritics.casefold()

    # Sanskrit transliterations in older texts frequently contain
    # apostrophes inside names, for example S'ankara and Is'vara.
    folded = re.sub(
        r"[''`']",
        "",
        folded,
    )

    folded = re.sub(
        r"[^\w]+",
        " ",
        folded,
        flags=re.UNICODE,
    )

    return " ".join(folded.split())


def domain_fidelity_issue(
    record: BuildRecord,
) -> str:
    """Reject clearly cross-domain passages as canonical prototypes.

    This does not relabel or remove a Build record.

    It only prevents a passage that is substantially explaining another
    framework from becoming a canonical positive prototype for the
    record's nominal domain.
    """

    foreign_markers = DOMAIN_FOREIGN_MARKERS[record.domain]
    support_markers = DOMAIN_FOREIGN_SUPPORT_MARKERS[record.domain]

    if not foreign_markers:
        return ""

    text = normalized_search_text(record.reviewed_text)

    foreign_hits = sorted({marker for marker in foreign_markers if marker in text})

    if not foreign_hits:
        return ""

    support_hits = sorted({marker for marker in support_markers if marker in text})

    # Strongest evidence:
    # the passage explicitly invokes another framework and also
    # uses terminology strongly associated with that framework.
    if support_hits:
        return "cross_domain_discussion:" + ",".join(foreign_hits + support_hits)

    explicit_framework_markers = {
        "vedantin",
        "vedantins",
        "vedantist",
        "vedantists",
        "vedantic",
        "sankara",
        "shankara",
        "samkhya",
        "sankhya",
    }

    explicit_hits = sorted(set(foreign_hits) & explicit_framework_markers)

    # Even without support terminology, explicit discussion of
    # another school/author makes the passage undesirable as a
    # canonical domain prototype.
    if explicit_hits:
        return "explicit_foreign_framework:" + ",".join(explicit_hits)

    return ""


def positive_selection_tier(
    record: BuildRecord,
    *,
    concept: str,
) -> int:
    """Rank positive prototype candidates from strongest to fallback.

    0 = clean positive + requested concept is primary
    1 = clean positive + requested concept is non-primary
    2 = positive + other hard-negative role + requested concept primary
    3 = positive + other hard-negative role + requested concept non-primary
    4 = clean partial + requested concept primary
    5 = clean partial + requested concept non-primary
    6 = partial + other hard-negative role + requested concept primary
    7 = partial + other hard-negative role + requested concept non-primary
    9 = ineligible
    """

    if not quality_eligible(record):
        return 9

    if domain_fidelity_issue(record):
        return 9

    label = record.labels[concept]

    if label not in {
        "positive",
        "partial",
    }:
        return 9

    # Never allow something explicitly reviewed as a hard negative
    # for this concept to become a positive prototype for it.
    if concept in record.hard_negative_for:
        return 9

    is_primary = record.primary_concept == concept

    has_other_hard_negative_role = bool(record.hard_negative_category or record.hard_negative_for)

    if label == "positive":
        if is_primary and not has_other_hard_negative_role:
            return 0

        if not is_primary and not has_other_hard_negative_role:
            return 1

        if is_primary and has_other_hard_negative_role:
            return 2

        return 3

    if label == "partial":
        if is_primary and not has_other_hard_negative_role:
            return 4

        if not is_primary and not has_other_hard_negative_role:
            return 5

        if is_primary and has_other_hard_negative_role:
            return 6

        return 7

    return 9


def positive_rank(
    record: BuildRecord,
    concept: str,
) -> tuple[int, int, str]:
    text_length_penalty = abs(len(record.reviewed_text) - 1200)

    return (
        positive_selection_tier(
            record,
            concept=concept,
        ),
        text_length_penalty,
        record.chunk_id,
    )


def select_domain_positive_prototypes(
    records: Sequence[BuildRecord],
    *,
    concept: str,
    domain: str,
) -> list[BuildRecord]:
    candidates = [
        record
        for record in records
        if record.domain == domain and positive_selection_tier(record, concept=concept) < 9
    ]

    candidates.sort(key=lambda record: positive_rank(record, concept))

    selected: list[BuildRecord] = []
    selected_ids: set[str] = set()
    selected_sources: Counter[str] = Counter()

    # First pass: source diversity where possible.
    for record in candidates:
        if len(selected) >= POSITIVE_PROTOTYPES_PER_DOMAIN:
            break

        if record.chunk_id in selected_ids:
            continue

        unseen_source_exists = any(
            candidate.chunk_id not in selected_ids and selected_sources[candidate.source_id] == 0
            for candidate in candidates
        )

        if selected_sources[record.source_id] > 0 and unseen_source_exists:
            continue

        selected.append(record)
        selected_ids.add(record.chunk_id)
        selected_sources[record.source_id] += 1

    # Second pass: fill the domain quota deterministically if source diversity
    # alone could not produce three records.
    if len(selected) < POSITIVE_PROTOTYPES_PER_DOMAIN:
        for record in candidates:
            if len(selected) >= POSITIVE_PROTOTYPES_PER_DOMAIN:
                break
            if record.chunk_id in selected_ids:
                continue

            selected.append(record)
            selected_ids.add(record.chunk_id)
            selected_sources[record.source_id] += 1

    if len(selected) < POSITIVE_PROTOTYPES_PER_DOMAIN:
        tier_counts = Counter(
            positive_selection_tier(record, concept=concept)
            for record in records
            if record.domain == domain
        )
        fidelity_rejections = sum(
            1 for record in records if record.domain == domain and domain_fidelity_issue(record)
        )
        raise PrototypeBuildError(
            f"{concept}/{domain} has only {len(selected)} eligible positive "
            f"Build prototypes; {POSITIVE_PROTOTYPES_PER_DOMAIN} are required. "
            f"candidate_tiers={dict(sorted(tier_counts.items()))}, "
            f"domain_fidelity_rejections={fidelity_rejections}. "
            "Review the Build annotations rather than weakening the prototype gate."
        )

    return selected


def select_positive_prototypes(
    records: Sequence[BuildRecord],
    *,
    concept: str,
) -> list[BuildRecord]:
    selected: list[BuildRecord] = []

    for domain in DOMAINS:
        selected.extend(
            select_domain_positive_prototypes(
                records,
                concept=concept,
                domain=domain,
            )
        )

    if len(selected) != POSITIVE_PROTOTYPES_PER_CONCEPT:
        raise PrototypeBuildError(
            f"{concept} produced {len(selected)} positive prototypes; "
            f"expected exactly {POSITIVE_PROTOTYPES_PER_CONCEPT}."
        )

    domain_counts = Counter(record.domain for record in selected)
    invalid_domains = {
        domain: domain_counts[domain]
        for domain in DOMAINS
        if domain_counts[domain] != POSITIVE_PROTOTYPES_PER_DOMAIN
    }
    if invalid_domains:
        raise PrototypeBuildError(
            f"{concept} positive prototypes are not balanced 3/3/3: {invalid_domains}."
        )

    return selected


def hard_negative_rank(
    record: BuildRecord,
) -> tuple[int, int, str]:
    text_length_penalty = abs(len(record.reviewed_text) - 1200)
    return (
        text_length_penalty,
        len(record.reviewed_text),
        record.chunk_id,
    )


def select_hard_negative_prototypes(
    records: Sequence[BuildRecord],
    *,
    concept: str,
) -> list[BuildRecord]:
    candidates = [
        record
        for record in records
        if quality_eligible(record)
        and concept in record.hard_negative_for
        and record.hard_negative_category
    ]

    if len(candidates) < MIN_HARD_NEGATIVES_PER_CONCEPT:
        raise PrototypeBuildError(
            f"{concept} has only {len(candidates)} hard-negative "
            "Build candidates; at least "
            f"{MIN_HARD_NEGATIVES_PER_CONCEPT} are required."
        )

    by_category: defaultdict[str, list[BuildRecord]] = defaultdict(list)
    for record in candidates:
        by_category[record.hard_negative_category].append(record)

    selected: list[BuildRecord] = []
    selected_ids: set[str] = set()
    selected_domains: Counter[str] = Counter()
    selected_sources: Counter[str] = Counter()

    # First pass: one representative per hard-negative category.
    for category in sorted(by_category):
        category_candidates = sorted(
            by_category[category],
            key=lambda record: (
                selected_domains[record.domain],
                selected_sources[record.source_id],
                hard_negative_rank(record),
            ),
        )
        if not category_candidates:
            continue

        record = category_candidates[0]
        selected.append(record)
        selected_ids.add(record.chunk_id)
        selected_domains[record.domain] += 1
        selected_sources[record.source_id] += 1

        if len(selected) >= MAX_HARD_NEGATIVES_PER_CONCEPT:
            break

    # Second pass: fill while preferring domain/source diversity.
    remaining = sorted(
        (record for record in candidates if record.chunk_id not in selected_ids),
        key=lambda record: (
            selected_domains[record.domain],
            selected_sources[record.source_id],
            hard_negative_rank(record),
        ),
    )

    for record in remaining:
        if len(selected) >= MAX_HARD_NEGATIVES_PER_CONCEPT:
            break

        selected.append(record)
        selected_ids.add(record.chunk_id)
        selected_domains[record.domain] += 1
        selected_sources[record.source_id] += 1

    if len(selected) < MIN_HARD_NEGATIVES_PER_CONCEPT:
        raise PrototypeBuildError(
            f"{concept} could not satisfy the minimum hard-negative prototype count."
        )

    return selected


def record_reference(
    record: BuildRecord,
    *,
    concept: str,
) -> dict[str, object]:
    selection_tier = positive_selection_tier(
        record,
        concept=concept,
    )

    return {
        "chunk_id": record.chunk_id,
        "domain": record.domain,
        "source_id": record.source_id,
        "concept_label": record.labels[concept],
        "primary_concept": record.primary_concept,
        "selection_tier": selection_tier,
        "selection_tier_description": {
            0: "clean_positive_primary",
            1: "clean_positive_non_primary",
            2: "positive_primary_with_other_hard_negative_role",
            3: "positive_non_primary_with_other_hard_negative_role",
            4: "clean_partial_primary_fallback",
            5: "clean_partial_non_primary_fallback",
            6: "partial_primary_with_other_hard_negative_role_fallback",
            7: "partial_non_primary_with_other_hard_negative_role_fallback",
        }[selection_tier],
        "domain_fidelity_checked": True,
        "domain_fidelity_issue": (domain_fidelity_issue(record)),
    }


def hard_negative_reference(
    record: BuildRecord,
) -> dict[str, object]:
    return {
        "chunk_id": record.chunk_id,
        "domain": record.domain,
        "source_id": record.source_id,
        "hard_negative_category": (record.hard_negative_category),
        "hard_negative_for": list(record.hard_negative_for),
    }


def build_concept_entry(
    records: Sequence[BuildRecord],
    *,
    concept: str,
) -> dict[str, object]:
    positive_records = select_positive_prototypes(
        records,
        concept=concept,
    )
    hard_negative_records = select_hard_negative_prototypes(
        records,
        concept=concept,
    )

    positive_domains = Counter(record.domain for record in positive_records)
    hard_negative_domains = Counter(record.domain for record in hard_negative_records)

    return {
        "concept_slug": concept,
        "prototype_version": PROTOTYPE_VERSION,
        "review_status": "needs_human_review",
        "construction_policy": {
            "source_split": "build_only",
            "uses_development": False,
            "uses_heldout": False,
            "uses_embeddings_for_selection": False,
            "question_and_passage_prototypes_separate": True,
            "domain_diversity_required": True,
            "positive_domain_quota": {
                "science": POSITIVE_PROTOTYPES_PER_DOMAIN,
                "advaita": POSITIVE_PROTOTYPES_PER_DOMAIN,
                "samkhya": POSITIVE_PROTOTYPES_PER_DOMAIN,
            },
            "positive_label_preferred": "positive",
            "partial_label_policy": "fallback_only",
            "same_concept_hard_negative_as_positive": False,
            "domain_fidelity_filter_enabled": True,
        },
        "question_examples": list(QUESTION_PROTOTYPES[concept]),
        "positive_passage_ids": [record.chunk_id for record in positive_records],
        "positive_passages": [
            record_reference(
                record,
                concept=concept,
            )
            for record in positive_records
        ],
        "positive_domain_distribution": {domain: positive_domains[domain] for domain in DOMAINS},
        "hard_negative_passage_ids": [record.chunk_id for record in hard_negative_records],
        "hard_negative_passages": [
            hard_negative_reference(record) for record in hard_negative_records
        ],
        "hard_negative_domain_distribution": {
            domain: hard_negative_domains[domain] for domain in DOMAINS
        },
        "excluded_adjacent_concepts": list(ADJACENT_CONCEPTS[concept]),
        "human_review": {
            "reviewer": "",
            "reviewed_at": "",
            "question_prototypes_reviewed": False,
            "positive_passages_reviewed": False,
            "hard_negatives_reviewed": False,
            "notes": "",
        },
    }


def build_artifact(
    records: Sequence[BuildRecord],
    *,
    build_path: Path,
    split_metadata: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "prototype_version": PROTOTYPE_VERSION,
        "status": "review_required",
        "phase": "phase_8_concept_prototypes",
        "concepts": [
            build_concept_entry(
                records,
                concept=concept,
            )
            for concept in CONCEPTS
        ],
        "provenance": {
            "source_split": "build",
            "build_path": build_path.as_posix(),
            "build_sha256": split_metadata["build_sha256"],
            "split_version": split_metadata["split_version"],
            "development_sha256_recorded_only": (split_metadata["development_sha256"]),
            "heldout_sha256_recorded_only": (split_metadata["heldout_sha256"]),
            "development_content_read": False,
            "heldout_content_read": False,
        },
        "exit_gate": {
            "all_concepts_have_question_prototypes": True,
            "all_concepts_have_positive_passage_prototypes": True,
            "all_concepts_have_hard_negative_prototypes": True,
            "all_passage_prototypes_from_build": True,
            "domain_diversity_checked": True,
            "positive_domain_balance_3_3_3": True,
            "domain_fidelity_checked": True,
            "human_review_complete": False,
        },
    }


def write_yaml(
    path: Path,
    artifact: Mapping[str, object],
    *,
    replace: bool,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists() and not replace:
        raise PrototypeBuildError(f"Output already exists: {path}. Use --replace to overwrite it.")

    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        yaml.safe_dump(
            dict(artifact),
            sort_keys=False,
            allow_unicode=True,
            width=100,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def summarize_artifact(
    artifact: Mapping[str, object],
) -> None:
    concepts_value = artifact.get("concepts")
    if not isinstance(concepts_value, list):
        return

    LOGGER.info("Phase 1 concept prototype artifact created")
    for concept_value in concepts_value:
        if not isinstance(concept_value, Mapping):
            continue

        concept = optional_string(concept_value.get("concept_slug"))
        positives = concept_value.get("positive_passage_ids")
        hard_negatives = concept_value.get("hard_negative_passage_ids")
        questions = concept_value.get("question_examples")

        LOGGER.info(
            "%s: %d questions, %d positives, %d hard negatives",
            concept,
            (len(questions) if isinstance(questions, list) else 0),
            (len(positives) if isinstance(positives, list) else 0),
            (len(hard_negatives) if isinstance(hard_negatives, list) else 0),
        )


def build_phase1_concept_prototypes(
    *,
    project_root: Path,
    build_path: Path,
    split_manifest_path: Path,
    output_path: Path,
    replace: bool,
) -> dict[str, object]:
    project_root = project_root.resolve()
    build_path = resolve_from_project(
        project_root,
        build_path,
    )
    split_manifest_path = resolve_from_project(
        project_root,
        split_manifest_path,
    )
    output_path = resolve_from_project(
        project_root,
        output_path,
    )

    require_file(build_path)
    require_file(split_manifest_path)

    split_metadata = validate_split_manifest(
        split_manifest_path,
        build_path=build_path,
    )
    records = load_build_records(build_path)

    artifact = build_artifact(
        records,
        build_path=build_path,
        split_metadata=split_metadata,
    )

    write_yaml(
        output_path,
        artifact,
        replace=replace,
    )
    summarize_artifact(artifact)

    LOGGER.info(
        "Output: %s",
        output_path,
    )
    LOGGER.info("Development content read: NO")
    LOGGER.info("Held-out content read: NO")
    LOGGER.info("Human review still required before Phase 8 exit gate.")

    return artifact


def main() -> int:
    arguments = parse_arguments()
    configure_logging(arguments.log_level)

    try:
        build_phase1_concept_prototypes(
            project_root=arguments.project_root,
            build_path=arguments.build,
            split_manifest_path=(arguments.split_manifest),
            output_path=arguments.output,
            replace=arguments.replace,
        )
    except PrototypeBuildError:
        LOGGER.exception("Phase 1 concept prototype construction failed")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
