from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import yaml
from scripts import tune_phase1_concept_mapping as phase10

LOGGER = logging.getLogger("wth.phase1.activate_phase1_corpus")

SCRIPT_VERSION: Final = "1.0.0"
ACTIVATION_VERSION: Final = "phase1-active-corpus-activation-v1"
DEFAULT_CORPUS_VERSION: Final = "phase1_active_corpus_v1"

CONCEPTS: Final = (
    "consciousness",
    "self_identity",
    "reality_appearance",
)
VALID_HUMAN_LABELS: Final = {"positive", "partial", "negative"}
VALID_REVIEW_DECISIONS: Final = {
    "include",
    "include_with_edits",
    "approved",
    "accept",
    "accepted",
}
BLOCKED_RIGHTS_VALUES: Final = {
    "unknown",
    "unreviewed",
    "pending",
    "restricted",
    "prohibited",
    "not_permitted",
    "not permitted",
    "denied",
    "blocked",
    "missing",
    "none",
    "n/a",
    "na",
}
EXPECTED_APPROVED_CHUNKS: Final = 318
MIN_ACTIVE_CHUNKS: Final = 250
MAX_ACTIVE_CHUNKS: Final = 350

DEFAULT_GOLD_CORPUS: Final = Path("artifacts/phase1/reviewed/phase1_reviewed_gold_corpus.jsonl")
DEFAULT_REVIEW_MANIFEST: Final = Path("artifacts/phase1/reviewed/phase1_human_review_manifest.json")
DEFAULT_SOURCE_CATALOGUE: Final = Path("docs/catalogues/phase1_sources.yaml")
DEFAULT_APPROVED_EMBEDDINGS: Final = Path(
    "artifacts/phase1/embeddings/approved_chunk_embeddings.jsonl"
)
DEFAULT_EMBEDDING_MANIFEST: Final = Path("artifacts/phase1/embeddings/embedding_manifest.json")
DEFAULT_WEIGHTED_TAGS: Final = Path(
    "artifacts/phase1/concepts/phase1_reviewed_weighted_concept_tags.jsonl"
)
DEFAULT_WEIGHTED_TAGS_MANIFEST: Final = Path(
    "artifacts/phase1/concepts/phase1_reviewed_weighted_concept_tags_manifest.json"
)
DEFAULT_OUTPUT_DIRECTORY: Final = Path("artifacts/phase1/active")

ACTIVE_BUNDLES_FILENAME: Final = "active_chunk_bundles.jsonl"
REVIEWED_CONCEPTS_FILENAME: Final = "reviewed_chunk_concepts.jsonl"
ACTIVATION_MANIFEST_FILENAME: Final = "activation_manifest.json"

SOURCE_ID_KEYS: Final = ("source_id", "id")
RIGHTS_KEYS: Final = (
    "rights_status",
    "source_rights_status",
    "license_status",
    "licence_status",
    "rights",
    "license",
    "licence",
)
SOURCE_STATUS_KEYS: Final = (
    "source_status",
    "status",
    "review_status",
    "acquisition_status",
)
SOURCE_TITLE_KEYS: Final = ("source_title", "title")
AUTHOR_KEYS: Final = ("author", "authors", "creator")
TRANSLATOR_KEYS: Final = ("translator", "translators")
PUBLICATION_YEAR_KEYS: Final = ("publication_year", "year")
CITATION_KEYS: Final = ("citation", "source_citation", "citation_text")
SOURCE_CHECKSUM_KEYS: Final = (
    "source_checksum",
    "source_sha256",
    "source_hash",
    "checksum",
    "sha256",
)
TEXT_CHECKSUM_KEYS: Final = (
    "text_checksum",
    "reviewed_text_checksum",
    "content_checksum",
)
REVIEW_DECISION_KEYS: Final = (
    "review_decision",
    "decision",
)
REVIEWER_KEYS: Final = ("reviewer", "reviewed_by")
REVIEWED_AT_KEYS: Final = ("reviewed_at", "review_timestamp")
CITATION_VERIFIED_KEYS: Final = (
    "citation_verified",
    "citation_reviewed",
)
SECTION_TITLE_KEYS: Final = ("section_title", "heading")
LOCATOR_KEYS: Final = (
    "structural_locator",
    "locator",
    "citation_locator",
)


class ActivationError(RuntimeError):
    """Raised when Phase 13 cannot safely activate the reviewed corpus."""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 13: validate and promote only reviewed Phase 1 gold chunks "
            "into database-ready active retrieval-corpus artifacts."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--gold-corpus", type=Path, default=DEFAULT_GOLD_CORPUS)
    parser.add_argument(
        "--review-manifest",
        type=Path,
        default=DEFAULT_REVIEW_MANIFEST,
    )
    parser.add_argument(
        "--source-catalogue",
        type=Path,
        default=DEFAULT_SOURCE_CATALOGUE,
    )
    parser.add_argument(
        "--approved-embeddings",
        type=Path,
        default=DEFAULT_APPROVED_EMBEDDINGS,
    )
    parser.add_argument(
        "--embedding-manifest",
        type=Path,
        default=DEFAULT_EMBEDDING_MANIFEST,
    )
    parser.add_argument(
        "--weighted-tags",
        type=Path,
        default=DEFAULT_WEIGHTED_TAGS,
    )
    parser.add_argument(
        "--weighted-tags-manifest",
        type=Path,
        default=DEFAULT_WEIGHTED_TAGS_MANIFEST,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--corpus-version",
        default=DEFAULT_CORPUS_VERSION,
    )
    parser.add_argument(
        "--expected-chunks",
        type=int,
        default=EXPECTED_APPROVED_CHUNKS,
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing derived Phase 13 activation artifacts.",
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


def resolve(project_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise ActivationError(f"Required file does not exist: {path}")


def require_mapping(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ActivationError(f"{description} must be an object.")

    result: dict[str, object] = {}
    for key, nested in value.items():
        if not isinstance(key, str):
            raise ActivationError(f"{description} contains a non-string key.")
        result[key] = nested
    return result


def require_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ActivationError(f"{description} must be a non-empty string.")
    return value.strip()


def optional_string(value: object) -> str:
    if isinstance(value, str):
        return value.strip()

    if isinstance(value, int | float):
        return str(value)

    return ""


def require_float(value: object, description: str) -> float:
    if not isinstance(value, int | float):
        raise ActivationError(f"{description} must be numeric.")
    return float(value)


def require_int(value: object, description: str) -> int:
    if not isinstance(value, int):
        raise ActivationError(f"{description} must be an integer.")
    return value


def load_json(path: Path) -> dict[str, object]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ActivationError(f"Invalid JSON in {path}: {exc}") from exc

    return require_mapping(raw, f"JSON document {path}")


def load_yaml(path: Path) -> object:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ActivationError(f"Invalid YAML in {path}: {exc}") from exc


def iter_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                raw: object = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ActivationError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc

            yield require_mapping(
                raw,
                f"JSONL record {path}:{line_number}",
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_json(
    path: Path,
    value: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")

    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
            handle.write("\n")

    temporary.replace(path)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def find_first_string(
    value: object,
    keys: Sequence[str],
) -> str:
    if isinstance(value, Mapping):
        mapping = require_mapping(value, "recursive mapping")

        for key in keys:
            candidate = optional_string(mapping.get(key))
            if candidate:
                return candidate

        for nested in mapping.values():
            result = find_first_string(nested, keys)
            if result:
                return result

    elif isinstance(value, list | tuple):
        for nested in value:
            result = find_first_string(nested, keys)
            if result:
                return result

    return ""


def parse_string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, list | tuple):
        return tuple(
            item.strip().casefold() for item in value if isinstance(item, str) and item.strip()
        )

    if isinstance(value, str) and value.strip():
        normalized = value.replace(",", "|").replace(";", "|")
        return tuple(item.strip().casefold() for item in normalized.split("|") if item.strip())

    return ()


def scalar_or_joined(value: object) -> str:
    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list | tuple):
        return "; ".join(optional_string(item) for item in value if optional_string(item))

    if isinstance(value, int | float):
        return str(value)

    return ""


def find_first_scalar_or_joined(
    value: object,
    keys: Sequence[str],
) -> str:
    if isinstance(value, Mapping):
        mapping = require_mapping(value, "recursive mapping")

        for key in keys:
            candidate = scalar_or_joined(mapping.get(key))
            if candidate:
                return candidate

        for nested in mapping.values():
            result = find_first_scalar_or_joined(nested, keys)
            if result:
                return result

    elif isinstance(value, list | tuple):
        for nested in value:
            result = find_first_scalar_or_joined(nested, keys)
            if result:
                return result

    return ""


def catalogue_entries(value: object) -> list[dict[str, object]]:
    if isinstance(value, list):
        return [require_mapping(item, "source catalogue entry") for item in value]

    if not isinstance(value, Mapping):
        raise ActivationError("Source catalogue root must be a list or object.")

    root = require_mapping(value, "source catalogue")

    sources = root.get("sources")
    if isinstance(sources, list):
        return [require_mapping(item, "source catalogue sources entry") for item in sources]

    if isinstance(sources, Mapping):
        result: list[dict[str, object]] = []
        for source_id, raw_entry in sources.items():
            if not isinstance(source_id, str):
                raise ActivationError("Source catalogue contains a non-string source key.")
            entry = require_mapping(
                raw_entry,
                f"source catalogue entry {source_id}",
            )
            if not find_first_string(entry, SOURCE_ID_KEYS):
                entry = dict(entry)
                entry["source_id"] = source_id
            result.append(entry)
        return result

    # Also support a mapping keyed directly by source_id.
    direct_entries: list[dict[str, object]] = []
    for source_id, raw_entry in root.items():
        if not isinstance(raw_entry, Mapping):
            continue

        entry = require_mapping(
            raw_entry,
            f"source catalogue entry {source_id}",
        )
        if not find_first_string(entry, SOURCE_ID_KEYS):
            entry = dict(entry)
            entry["source_id"] = source_id
        direct_entries.append(entry)

    if direct_entries:
        return direct_entries

    raise ActivationError("Could not locate source entries in source catalogue.")


def load_source_catalogue(
    path: Path,
) -> dict[str, dict[str, object]]:
    entries = catalogue_entries(load_yaml(path))
    result: dict[str, dict[str, object]] = {}

    for entry in entries:
        source_id = find_first_string(entry, SOURCE_ID_KEYS)
        if not source_id:
            raise ActivationError("Source catalogue entry is missing source_id.")

        if source_id in result:
            raise ActivationError(f"Duplicate source_id in source catalogue: {source_id}")

        result[source_id] = entry

    return result


def source_rights_status(
    entry: Mapping[str, object],
    *,
    source_id: str,
) -> str:
    rights = find_first_scalar_or_joined(entry, RIGHTS_KEYS)
    if not rights:
        raise ActivationError(f"{source_id} has no source rights/license status.")

    if rights.casefold() in BLOCKED_RIGHTS_VALUES:
        raise ActivationError(f"{source_id} has non-activatable rights status: {rights!r}.")

    return rights


def validate_review_manifest(
    path: Path,
    *,
    expected_chunks: int,
) -> dict[str, object]:
    manifest = load_json(path)

    if manifest.get("strict_gate_passed") is not True:
        raise ActivationError("Human-review manifest did not pass its strict gate.")

    if optional_string(manifest.get("status")) != "phase1_human_review_complete":
        raise ActivationError("Human-review manifest is not complete.")

    summary = require_mapping(
        manifest.get("summary"),
        "human-review summary",
    )
    approved = require_int(
        summary.get("approved_rows"),
        "human-review approved_rows",
    )

    if approved != expected_chunks:
        raise ActivationError(
            f"Expected {expected_chunks} approved chunks, review manifest contains {approved}."
        )

    return manifest


def validate_phase12_manifest(
    path: Path,
    *,
    tags_path: Path,
    expected_chunks: int,
) -> dict[str, object]:
    manifest = load_json(path)

    if optional_string(manifest.get("status")) != "production_concept_weights_complete":
        raise ActivationError("Phase 12 production concept weights are not complete.")

    gate = require_mapping(
        manifest.get("exit_gate"),
        "Phase 12 exit_gate",
    )

    required_true = (
        "all_approved_chunks_present",
        "all_calibrated_weights_in_0_1",
        "all_chunks_have_valid_phase1_role",
        "all_reviewed_labels_present",
        "exactly_three_concept_rows_per_chunk",
        "full_model_provenance_present",
        "full_prototype_provenance_present",
        "human_labels_authoritative",
    )

    for field_name in required_true:
        if gate.get(field_name) is not True:
            raise ActivationError(f"Phase 12 exit gate failed: {field_name}.")

    counts = require_mapping(
        manifest.get("counts"),
        "Phase 12 counts",
    )
    approved_chunks = require_int(
        counts.get("approved_chunks"),
        "Phase 12 approved_chunks",
    )
    tag_count = require_int(
        counts.get("tag_count"),
        "Phase 12 tag_count",
    )

    if approved_chunks != expected_chunks:
        raise ActivationError("Phase 12 approved chunk count does not match Phase 13.")

    if tag_count != expected_chunks * len(CONCEPTS):
        raise ActivationError("Phase 12 concept-tag count does not equal chunks x concepts.")

    output = require_mapping(
        manifest.get("output"),
        "Phase 12 output",
    )
    expected_sha = require_string(
        output.get("sha256"),
        "Phase 12 output sha256",
    )
    actual_sha = phase10.sha256_jsonl(tags_path)

    if actual_sha != expected_sha:
        raise ActivationError("Phase 12 weighted-tag checksum mismatch.")

    return manifest


def load_gold_rows(
    path: Path,
    *,
    expected_chunks: int,
) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}

    for raw in iter_jsonl(path):
        chunk_id = require_string(raw.get("chunk_id"), "gold chunk_id")

        if chunk_id in rows:
            raise ActivationError(f"Duplicate chunk_id in gold corpus: {chunk_id}")

        reviewed_text = require_string(
            raw.get("reviewed_text"),
            f"{chunk_id} reviewed_text",
        )

        domain = require_string(
            raw.get("domain"),
            f"{chunk_id} domain",
        ).casefold()

        citation = find_first_string(raw, CITATION_KEYS)
        if not citation:
            raise ActivationError(f"{chunk_id} has no citation.")

        source_checksum = find_first_string(
            raw,
            SOURCE_CHECKSUM_KEYS,
        )
        if not source_checksum:
            raise ActivationError(f"{chunk_id} has no source checksum.")

        review_decision = find_first_string(
            raw,
            REVIEW_DECISION_KEYS,
        ).casefold()
        if review_decision not in VALID_REVIEW_DECISIONS:
            raise ActivationError(
                f"{chunk_id} has invalid/missing approved review decision: {review_decision!r}."
            )

        review = require_mapping(
            raw.get("review"),
            f"{chunk_id} review",
        )
        labels_raw = require_mapping(
            review.get("labels"),
            f"{chunk_id} review.labels",
        )

        for concept in CONCEPTS:
            label = require_string(
                labels_raw.get(concept),
                f"{chunk_id} label {concept}",
            ).casefold()
            if label not in VALID_HUMAN_LABELS:
                raise ActivationError(f"{chunk_id} has invalid reviewed label {concept}={label!r}.")

        rows[chunk_id] = raw

        if domain not in {"science", "advaita", "samkhya"}:
            raise ActivationError(f"{chunk_id} has unsupported Phase 1 domain {domain!r}.")

        # This also ensures text is materially present.
        if not reviewed_text.strip():
            raise ActivationError(f"{chunk_id} has blank reviewed text.")

    if len(rows) != expected_chunks:
        raise ActivationError(
            f"Expected {expected_chunks} reviewed gold chunks, found {len(rows)}."
        )

    return rows


def load_weighted_tags(
    path: Path,
    *,
    expected_chunks: int,
) -> dict[str, dict[str, dict[str, object]]]:
    by_chunk: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    row_count = 0

    for raw in iter_jsonl(path):
        chunk_id = require_string(
            raw.get("chunk_id"),
            "weighted tag chunk_id",
        )
        concept_id = require_string(
            raw.get("concept_id"),
            f"{chunk_id} concept_id",
        )

        if concept_id not in CONCEPTS:
            raise ActivationError(f"{chunk_id} has unexpected concept tag {concept_id!r}.")

        if concept_id in by_chunk[chunk_id]:
            raise ActivationError(f"Duplicate weighted tag: {chunk_id}/{concept_id}")

        human_label = require_string(
            raw.get("human_label"),
            f"{chunk_id}/{concept_id} human_label",
        ).casefold()
        if human_label not in VALID_HUMAN_LABELS:
            raise ActivationError(f"{chunk_id}/{concept_id} has invalid human label.")

        weight = require_float(
            raw.get("calibrated_weight"),
            f"{chunk_id}/{concept_id} calibrated_weight",
        )
        if not 0.0 <= weight <= 1.0:
            raise ActivationError(f"{chunk_id}/{concept_id} weight is outside [0,1].")

        production_active = raw.get("production_active")
        if not isinstance(production_active, bool):
            raise ActivationError(f"{chunk_id}/{concept_id} production_active must be boolean.")

        expected_active = human_label in {"positive", "partial"}
        if production_active is not expected_active:
            raise ActivationError(f"{chunk_id}/{concept_id} violates human-authoritative policy.")

        by_chunk[chunk_id][concept_id] = raw
        row_count += 1

    expected_rows = expected_chunks * len(CONCEPTS)
    if row_count != expected_rows:
        raise ActivationError(f"Expected {expected_rows} weighted tag rows, found {row_count}.")

    if len(by_chunk) != expected_chunks:
        raise ActivationError(f"Expected tags for {expected_chunks} chunks, found {len(by_chunk)}.")

    for chunk_id, concepts in by_chunk.items():
        if set(concepts) != set(CONCEPTS):
            raise ActivationError(f"{chunk_id} does not have all three Phase 1 concept rows.")

    return dict(by_chunk)


def embedding_identity_dict(identity: object) -> dict[str, object]:
    as_dict = getattr(identity, "as_dict", None)
    if not callable(as_dict):
        raise ActivationError("Embedding identity object does not expose as_dict().")

    return require_mapping(as_dict(), "embedding identity")


def validate_embedding_provenance(
    *,
    phase12_manifest: Mapping[str, object],
    embedding_identity: Mapping[str, object],
) -> None:
    mapping_provenance = require_mapping(
        phase12_manifest.get("mapping_provenance"),
        "Phase 12 mapping_provenance",
    )
    phase12_identity = require_mapping(
        mapping_provenance.get("embedding_identity"),
        "Phase 12 embedding_identity",
    )

    keys = (
        "provider",
        "model",
        "model_revision",
        "dimensions",
        "normalization",
    )

    for key in keys:
        if phase12_identity.get(key) != embedding_identity.get(key):
            raise ActivationError(
                f"Embedding identity mismatch for {key}: "
                f"{phase12_identity.get(key)!r} != "
                f"{embedding_identity.get(key)!r}."
            )


def source_summary(
    *,
    source_id: str,
    gold_rows: Sequence[Mapping[str, object]],
    catalogue_entry: Mapping[str, object],
) -> dict[str, object]:
    checksums = {find_first_string(row, SOURCE_CHECKSUM_KEYS) for row in gold_rows}
    checksums.discard("")

    if len(checksums) != 1:
        raise ActivationError(
            f"{source_id} does not have one consistent source checksum: {sorted(checksums)}"
        )

    citations = {find_first_string(row, CITATION_KEYS) for row in gold_rows}
    citations.discard("")

    if not citations:
        raise ActivationError(f"{source_id} has no citations in reviewed gold rows.")

    rights = source_rights_status(
        catalogue_entry,
        source_id=source_id,
    )

    first_row = gold_rows[0]

    return {
        "source_id": source_id,
        "approved_for_phase1": True,
        "approval_basis": "human_reviewed_gold_corpus",
        "source_catalogue_status": find_first_string(
            catalogue_entry,
            SOURCE_STATUS_KEYS,
        ),
        "source_rights_status": rights,
        "source_checksum": next(iter(checksums)),
        "source_title": (
            find_first_scalar_or_joined(first_row, SOURCE_TITLE_KEYS)
            or find_first_scalar_or_joined(
                catalogue_entry,
                SOURCE_TITLE_KEYS,
            )
        ),
        "author": (
            find_first_scalar_or_joined(first_row, AUTHOR_KEYS)
            or find_first_scalar_or_joined(
                catalogue_entry,
                AUTHOR_KEYS,
            )
        ),
        "translator": (
            find_first_scalar_or_joined(first_row, TRANSLATOR_KEYS)
            or find_first_scalar_or_joined(
                catalogue_entry,
                TRANSLATOR_KEYS,
            )
        ),
        "publication_year": (
            find_first_scalar_or_joined(
                first_row,
                PUBLICATION_YEAR_KEYS,
            )
            or find_first_scalar_or_joined(
                catalogue_entry,
                PUBLICATION_YEAR_KEYS,
            )
        ),
        "citations_observed": sorted(citations),
        "catalogue_metadata": dict(catalogue_entry),
    }


def concept_labels_from_gold(
    raw: Mapping[str, object],
    *,
    chunk_id: str,
) -> dict[str, str]:
    review = require_mapping(
        raw.get("review"),
        f"{chunk_id} review",
    )
    labels_raw = require_mapping(
        review.get("labels"),
        f"{chunk_id} review.labels",
    )

    return {
        concept: require_string(
            labels_raw.get(concept),
            f"{chunk_id} label {concept}",
        ).casefold()
        for concept in CONCEPTS
    }


def build_artifacts(
    *,
    gold_rows: Mapping[str, Mapping[str, object]],
    weighted_tags: Mapping[
        str,
        Mapping[str, Mapping[str, object]],
    ],
    embeddings: Mapping[str, phase10.FloatArray],
    embedding_identity: Mapping[str, object],
    source_catalogue: Mapping[str, Mapping[str, object]],
    phase12_manifest: Mapping[str, object],
    corpus_version: str,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    grouped_gold: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for raw in gold_rows.values():
        source_id = require_string(
            raw.get("source_id"),
            "gold source_id",
        )
        grouped_gold[source_id].append(raw)

    source_metadata: dict[str, dict[str, object]] = {}
    for source_id, rows in grouped_gold.items():
        catalogue_entry = source_catalogue.get(source_id)
        if catalogue_entry is None:
            raise ActivationError(
                f"Approved source {source_id} is missing from data/catalogues/phase1_sources.yaml."
            )

        source_metadata[source_id] = source_summary(
            source_id=source_id,
            gold_rows=rows,
            catalogue_entry=catalogue_entry,
        )

    mapping_provenance = require_mapping(
        phase12_manifest.get("mapping_provenance"),
        "Phase 12 mapping_provenance",
    )
    model_version = require_string(
        mapping_provenance.get("model_version"),
        "Phase 12 model_version",
    )
    prototype_version = require_string(
        mapping_provenance.get("prototype_version"),
        "Phase 12 prototype_version",
    )
    mapping_method = require_string(
        mapping_provenance.get("mapping_method"),
        "Phase 12 mapping_method",
    )

    active_bundles: list[dict[str, object]] = []
    reviewed_concepts: list[dict[str, object]] = []

    domain_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    production_active_counts: Counter[str] = Counter()
    negative_relation_counts: Counter[str] = Counter()
    hard_negative_chunk_count = 0
    edited_chunk_count = 0

    for chunk_id in sorted(gold_rows):
        raw = gold_rows[chunk_id]
        source_id = require_string(
            raw.get("source_id"),
            f"{chunk_id} source_id",
        )
        domain = require_string(
            raw.get("domain"),
            f"{chunk_id} domain",
        ).casefold()
        reviewed_text = require_string(
            raw.get("reviewed_text"),
            f"{chunk_id} reviewed_text",
        )
        citation = find_first_string(raw, CITATION_KEYS)
        source_checksum = find_first_string(
            raw,
            SOURCE_CHECKSUM_KEYS,
        )
        review_decision = find_first_string(
            raw,
            REVIEW_DECISION_KEYS,
        ).casefold()

        if review_decision == "include_with_edits":
            edited_chunk_count += 1

        review = require_mapping(
            raw.get("review"),
            f"{chunk_id} review",
        )
        labels = concept_labels_from_gold(
            raw,
            chunk_id=chunk_id,
        )
        hard_negative_for = parse_string_tuple(review.get("hard_negative_for"))
        hard_negative_category = optional_string(review.get("hard_negative_category")).casefold()

        if hard_negative_for:
            hard_negative_chunk_count += 1

        if chunk_id not in embeddings:
            raise ActivationError(f"{chunk_id} has no selected approved embedding.")

        vector = embeddings[chunk_id]
        dimensions = require_int(
            embedding_identity.get("dimensions"),
            "embedding dimensions",
        )
        if len(vector) != dimensions:
            raise ActivationError(
                f"{chunk_id} embedding dimension mismatch: {len(vector)} != {dimensions}."
            )

        tag_rows = weighted_tags.get(chunk_id)
        if tag_rows is None:
            raise ActivationError(f"{chunk_id} has no Phase 12 concept tags.")

        concept_payloads: list[dict[str, object]] = []
        valid_roles: list[str] = []

        for concept in CONCEPTS:
            tag = tag_rows[concept]
            human_label = require_string(
                tag.get("human_label"),
                f"{chunk_id}/{concept} human_label",
            ).casefold()

            if human_label != labels[concept]:
                raise ActivationError(
                    f"{chunk_id}/{concept} Phase 12 label differs from gold review."
                )

            production_active = tag.get("production_active")
            if not isinstance(production_active, bool):
                raise ActivationError(f"{chunk_id}/{concept} production_active is invalid.")

            calibrated_weight = require_float(
                tag.get("calibrated_weight"),
                f"{chunk_id}/{concept} calibrated_weight",
            )
            raw_similarity = require_float(
                tag.get("raw_similarity"),
                f"{chunk_id}/{concept} raw_similarity",
            )
            raw_mapping_score = require_float(
                tag.get("raw_mapping_score"),
                f"{chunk_id}/{concept} raw_mapping_score",
            )

            if production_active:
                production_active_counts[concept] += 1
                valid_roles.append(f"evidence:{concept}")
            else:
                negative_relation_counts[concept] += 1

            if concept in hard_negative_for:
                valid_roles.append(f"hard_negative:{concept}")

            # Explicitly type concept_record as dict[str, object]
            concept_record: dict[str, object] = {
                "chunk_id": chunk_id,
                "concept_id": concept,
                "corpus_version": corpus_version,
                "lifecycle_status": "active",
                "human_label": human_label,
                "production_active": production_active,
                "calibrated_weight": calibrated_weight,
                "raw_similarity": raw_similarity,
                "raw_mapping_score": raw_mapping_score,
                "human_override": bool(tag.get("human_override")),
                "automated_active": bool(tag.get("automated_active")),
                "activation_threshold": require_float(
                    tag.get("activation_threshold"),
                    f"{chunk_id}/{concept} activation_threshold",
                ),
                "review_status": require_string(
                    tag.get("review_status"),
                    f"{chunk_id}/{concept} review_status",
                ),
                "mapping_method": require_string(
                    tag.get("mapping_method"),
                    f"{chunk_id}/{concept} mapping_method",
                ),
                "prototype_version": require_string(
                    tag.get("prototype_version"),
                    f"{chunk_id}/{concept} prototype_version",
                ),
                "model_version": require_string(
                    tag.get("model_version"),
                    f"{chunk_id}/{concept} model_version",
                ),
                "hard_negative_for_concept": concept in hard_negative_for,
                "hard_negative_category": hard_negative_category,
            }

            reviewed_concepts.append(concept_record)
            concept_payloads.append(concept_record)

        if not valid_roles:
            raise ActivationError(
                f"{chunk_id} has no valid Phase 1 evidence or hard-negative role."
            )

        text_checksum = find_first_string(
            raw,
            TEXT_CHECKSUM_KEYS,
        )
        if not text_checksum:
            text_checksum = sha256_text(reviewed_text)

        source = source_metadata[source_id]

        # Explicitly type bundle as dict[str, object]
        bundle: dict[str, object] = {
            "chunk_id": chunk_id,
            "corpus_version": corpus_version,
            "lifecycle_status": "active",
            "queryable": True,
            "source": {
                "source_id": source_id,
                "domain": domain,
                "approved_for_phase1": True,
                "approval_basis": source["approval_basis"],
                "source_rights_status": source["source_rights_status"],
                "source_checksum": source_checksum,
                "source_title": source["source_title"],
                "author": source["author"],
                "translator": source["translator"],
                "publication_year": source["publication_year"],
                "source_catalogue_status": source["source_catalogue_status"],
            },
            "citation": {
                "citation": citation,
                "citation_verified": find_first_string(
                    raw,
                    CITATION_VERIFIED_KEYS,
                ),
                "section_title": find_first_string(
                    raw,
                    SECTION_TITLE_KEYS,
                ),
                "structural_locator": find_first_string(
                    raw,
                    LOCATOR_KEYS,
                ),
            },
            "content": {
                "reviewed_text": reviewed_text,
                "text_checksum": text_checksum,
            },
            "embedding": {
                "vector": [float(value) for value in vector],
                "provider": embedding_identity.get("provider"),
                "model": embedding_identity.get("model"),
                "model_revision": embedding_identity.get("model_revision"),
                "model_version": model_version,
                "dimensions": dimensions,
                "normalization": embedding_identity.get("normalization"),
                "input_role": "approved_chunk_document",
                "selected_for_production": True,
            },
            "concept_labels": labels,
            "concepts": concept_payloads,
            "phase1_roles": sorted(set(valid_roles)),
            "review": {
                "review_decision": review_decision,
                "reviewer": find_first_string(
                    raw,
                    REVIEWER_KEYS,
                ),
                "reviewed_at": find_first_string(
                    raw,
                    REVIEWED_AT_KEYS,
                ),
                "primary_concept": optional_string(review.get("primary_concept")).casefold(),
                "secondary_concepts": list(parse_string_tuple(review.get("secondary_concepts"))),
                "hard_negative_for": list(hard_negative_for),
                "hard_negative_category": hard_negative_category,
                "human_authoritative": True,
            },
            "mapping_provenance": {
                "mapping_method": mapping_method,
                "prototype_version": prototype_version,
                "model_version": model_version,
            },
            "database_ready": {
                "logical_entities": [
                    "sources",
                    "chunks",
                    "chunk_embeddings",
                    "chunk_concepts",
                    "review_metadata",
                    "corpus_version",
                ],
                "schema_mapping_required_before_upsert": True,
            },
        }

        if not valid_roles:
            raise ActivationError(
                f"{chunk_id} has no valid Phase 1 evidence or hard-negative role."
            )

        text_checksum = find_first_string(
            raw,
            TEXT_CHECKSUM_KEYS,
        )
        if not text_checksum:
            text_checksum = sha256_text(reviewed_text)

        source = source_metadata[source_id]

        bundle = {
            "chunk_id": chunk_id,
            "corpus_version": corpus_version,
            "lifecycle_status": "active",
            "queryable": True,
            "source": {
                "source_id": source_id,
                "domain": domain,
                "approved_for_phase1": True,
                "approval_basis": source["approval_basis"],
                "source_rights_status": source["source_rights_status"],
                "source_checksum": source_checksum,
                "source_title": source["source_title"],
                "author": source["author"],
                "translator": source["translator"],
                "publication_year": source["publication_year"],
                "source_catalogue_status": source["source_catalogue_status"],
            },
            "citation": {
                "citation": citation,
                "citation_verified": find_first_string(
                    raw,
                    CITATION_VERIFIED_KEYS,
                ),
                "section_title": find_first_string(
                    raw,
                    SECTION_TITLE_KEYS,
                ),
                "structural_locator": find_first_string(
                    raw,
                    LOCATOR_KEYS,
                ),
            },
            "content": {
                "reviewed_text": reviewed_text,
                "text_checksum": text_checksum,
            },
            "embedding": {
                "vector": [float(value) for value in vector],
                "provider": embedding_identity.get("provider"),
                "model": embedding_identity.get("model"),
                "model_revision": embedding_identity.get("model_revision"),
                "model_version": model_version,
                "dimensions": dimensions,
                "normalization": embedding_identity.get("normalization"),
                "input_role": "approved_chunk_document",
                "selected_for_production": True,
            },
            "concept_labels": labels,
            "concepts": concept_payloads,
            "phase1_roles": sorted(set(valid_roles)),
            "review": {
                "review_decision": review_decision,
                "reviewer": find_first_string(
                    raw,
                    REVIEWER_KEYS,
                ),
                "reviewed_at": find_first_string(
                    raw,
                    REVIEWED_AT_KEYS,
                ),
                "primary_concept": optional_string(review.get("primary_concept")).casefold(),
                "secondary_concepts": list(parse_string_tuple(review.get("secondary_concepts"))),
                "hard_negative_for": list(hard_negative_for),
                "hard_negative_category": hard_negative_category,
                "human_authoritative": True,
            },
            "mapping_provenance": {
                "mapping_method": mapping_method,
                "prototype_version": prototype_version,
                "model_version": model_version,
            },
            "database_ready": {
                "logical_entities": [
                    "sources",
                    "chunks",
                    "chunk_embeddings",
                    "chunk_concepts",
                    "review_metadata",
                    "corpus_version",
                ],
                "schema_mapping_required_before_upsert": True,
            },
        }

        active_bundles.append(bundle)
        domain_counts[domain] += 1
        source_counts[source_id] += 1

    active_count = len(active_bundles)
    if not MIN_ACTIVE_CHUNKS <= active_count <= MAX_ACTIVE_CHUNKS:
        raise ActivationError(
            "Active corpus count is outside Phase 1 exit range: "
            f"{active_count} not in [{MIN_ACTIVE_CHUNKS}, {MAX_ACTIVE_CHUNKS}]."
        )

    if len(reviewed_concepts) != active_count * len(CONCEPTS):
        raise ActivationError(
            "Reviewed concept output does not contain exactly three relations per active chunk."
        )

    diagnostics: dict[str, object] = {
        "active_chunk_count": active_count,
        "reviewed_chunk_concept_count": len(reviewed_concepts),
        "approved_source_count": len(source_metadata),
        "by_domain": dict(sorted(domain_counts.items())),
        "by_source": dict(sorted(source_counts.items())),
        "production_active_relations_by_concept": dict(sorted(production_active_counts.items())),
        "negative_relations_by_concept": dict(sorted(negative_relation_counts.items())),
        "hard_negative_chunk_count": hard_negative_chunk_count,
        "include_with_edits_chunk_count": edited_chunk_count,
    }

    return active_bundles, reviewed_concepts, diagnostics


def validate_bundle_contract(
    bundles: Sequence[Mapping[str, object]],
    concepts: Sequence[Mapping[str, object]],
    *,
    expected_chunks: int,
    corpus_version: str,
) -> dict[str, object]:
    if len(bundles) != expected_chunks:
        raise ActivationError(f"Expected {expected_chunks} active bundles, found {len(bundles)}.")

    if len(concepts) != expected_chunks * len(CONCEPTS):
        raise ActivationError("Reviewed concept row count is incomplete.")

    seen_chunks: set[str] = set()

    for bundle in bundles:
        chunk_id = require_string(
            bundle.get("chunk_id"),
            "active bundle chunk_id",
        )
        if chunk_id in seen_chunks:
            raise ActivationError(f"Duplicate active bundle chunk_id: {chunk_id}")
        seen_chunks.add(chunk_id)

        if bundle.get("queryable") is not True:
            raise ActivationError(f"{chunk_id} is not marked queryable.")

        if optional_string(bundle.get("lifecycle_status")) != "active":
            raise ActivationError(f"{chunk_id} is not marked active.")

        if optional_string(bundle.get("corpus_version")) != corpus_version:
            raise ActivationError(f"{chunk_id} corpus_version mismatch.")

        source = require_mapping(
            bundle.get("source"),
            f"{chunk_id} source",
        )
        if source.get("approved_for_phase1") is not True:
            raise ActivationError(f"{chunk_id} source is not approved.")
        require_string(
            source.get("source_rights_status"),
            f"{chunk_id} source_rights_status",
        )
        require_string(
            source.get("source_checksum"),
            f"{chunk_id} source_checksum",
        )

        citation = require_mapping(
            bundle.get("citation"),
            f"{chunk_id} citation",
        )
        require_string(
            citation.get("citation"),
            f"{chunk_id} citation text",
        )

        content = require_mapping(
            bundle.get("content"),
            f"{chunk_id} content",
        )
        require_string(
            content.get("reviewed_text"),
            f"{chunk_id} reviewed_text",
        )

        embedding = require_mapping(
            bundle.get("embedding"),
            f"{chunk_id} embedding",
        )
        vector = embedding.get("vector")
        if not isinstance(vector, list):
            raise ActivationError(f"{chunk_id} embedding vector must be a list.")
        dimensions = require_int(
            embedding.get("dimensions"),
            f"{chunk_id} embedding dimensions",
        )
        if len(vector) != dimensions:
            raise ActivationError(f"{chunk_id} embedding vector length mismatch.")

        labels = require_mapping(
            bundle.get("concept_labels"),
            f"{chunk_id} concept_labels",
        )
        if set(labels) != set(CONCEPTS):
            raise ActivationError(f"{chunk_id} does not have all Phase 1 labels.")

        review = require_mapping(
            bundle.get("review"),
            f"{chunk_id} review",
        )
        decision = require_string(
            review.get("review_decision"),
            f"{chunk_id} review_decision",
        ).casefold()
        if decision not in VALID_REVIEW_DECISIONS:
            raise ActivationError(f"{chunk_id} has invalid review decision.")

        roles = bundle.get("phase1_roles")
        if not isinstance(roles, list) or not roles:
            raise ActivationError(f"{chunk_id} has no valid Phase 1 role.")

    return {
        "approved_source_present_for_every_chunk": True,
        "source_rights_status_present_for_every_chunk": True,
        "source_checksum_present_for_every_chunk": True,
        "citation_present_for_every_chunk": True,
        "reviewed_text_present_for_every_chunk": True,
        "selected_embedding_present_for_every_chunk": True,
        "embedding_metadata_present_for_every_chunk": True,
        "reviewed_concept_labels_present_for_every_chunk": True,
        "calibrated_weights_present_for_every_chunk": True,
        "corpus_version_present_for_every_chunk": True,
        "review_decision_present_for_every_chunk": True,
        "active_chunk_count_within_250_350": True,
        "artifact_queryable_flag_set": True,
        "database_schema_mapping_pending": True,
    }


def prepare_output_paths(
    output_directory: Path,
    *,
    replace: bool,
) -> dict[str, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)

    paths = {
        "active_bundles": output_directory / ACTIVE_BUNDLES_FILENAME,
        "reviewed_concepts": (output_directory / REVIEWED_CONCEPTS_FILENAME),
        "manifest": output_directory / ACTIVATION_MANIFEST_FILENAME,
    }

    existing = [path for path in paths.values() if path.exists()]

    if existing and not replace:
        raise ActivationError(
            "Phase 13 activation artifacts already exist. "
            "Use --replace to regenerate derived files: "
            + ", ".join(path.as_posix() for path in existing)
        )

    return paths


def run_phase13(
    *,
    project_root: Path,
    gold_corpus_path: Path,
    review_manifest_path: Path,
    source_catalogue_path: Path,
    approved_embeddings_path: Path,
    embedding_manifest_path: Path,
    weighted_tags_path: Path,
    weighted_tags_manifest_path: Path,
    output_directory: Path,
    corpus_version: str,
    expected_chunks: int,
    replace: bool,
) -> dict[str, object]:
    project_root = project_root.resolve()

    gold_corpus_path = resolve(project_root, gold_corpus_path)
    review_manifest_path = resolve(
        project_root,
        review_manifest_path,
    )
    source_catalogue_path = resolve(
        project_root,
        source_catalogue_path,
    )
    approved_embeddings_path = resolve(
        project_root,
        approved_embeddings_path,
    )
    embedding_manifest_path = resolve(
        project_root,
        embedding_manifest_path,
    )
    weighted_tags_path = resolve(
        project_root,
        weighted_tags_path,
    )
    weighted_tags_manifest_path = resolve(
        project_root,
        weighted_tags_manifest_path,
    )
    output_directory = resolve(
        project_root,
        output_directory,
    )

    for path in (
        gold_corpus_path,
        review_manifest_path,
        source_catalogue_path,
        approved_embeddings_path,
        embedding_manifest_path,
        weighted_tags_path,
        weighted_tags_manifest_path,
    ):
        require_file(path)

    paths = prepare_output_paths(
        output_directory,
        replace=replace,
    )

    LOGGER.info(
        "Phase 13 starting: corpus_version=%s expected_chunks=%d",
        corpus_version,
        expected_chunks,
    )

    review_manifest = validate_review_manifest(
        review_manifest_path,
        expected_chunks=expected_chunks,
    )
    phase12_manifest = validate_phase12_manifest(
        weighted_tags_manifest_path,
        tags_path=weighted_tags_path,
        expected_chunks=expected_chunks,
    )
    gold_rows = load_gold_rows(
        gold_corpus_path,
        expected_chunks=expected_chunks,
    )
    weighted_tags = load_weighted_tags(
        weighted_tags_path,
        expected_chunks=expected_chunks,
    )
    source_catalogue = load_source_catalogue(source_catalogue_path)

    identity, embedding_manifest = phase10.load_identity(embedding_manifest_path)
    identity_dict = embedding_identity_dict(identity)
    validate_embedding_provenance(
        phase12_manifest=phase12_manifest,
        embedding_identity=identity_dict,
    )

    embeddings = phase10.load_needed_embeddings(
        approved_embeddings_path,
        set(gold_rows),
        identity,
    )

    bundles, reviewed_concepts, diagnostics = build_artifacts(
        gold_rows=gold_rows,
        weighted_tags=weighted_tags,
        embeddings=embeddings,
        embedding_identity=identity_dict,
        source_catalogue=source_catalogue,
        phase12_manifest=phase12_manifest,
        corpus_version=corpus_version,
    )

    exit_gate = validate_bundle_contract(
        bundles,
        reviewed_concepts,
        expected_chunks=expected_chunks,
        corpus_version=corpus_version,
    )

    atomic_jsonl(
        paths["active_bundles"],
        bundles,
    )
    atomic_jsonl(
        paths["reviewed_concepts"],
        reviewed_concepts,
    )

    manifest: dict[str, object] = {
        "script_version": SCRIPT_VERSION,
        "activation_version": ACTIVATION_VERSION,
        "generated_at": utc_now(),
        "phase": "phase_13_activate_approved_phase1_corpus",
        "status": "activation_artifacts_complete",
        "corpus_version": corpus_version,
        "lifecycle_status": "active",
        "activation_policy": {
            "only_human_reviewed_approved_chunks_promoted": True,
            "human_labels_authoritative": True,
            "all_three_concept_relations_preserved": True,
            "production_active_relations_drive_concept_eligibility": True,
            "calibrated_weights_drive_ranking_not_inclusion": True,
            "hard_negative_roles_preserved": True,
            "candidate_corpus_not_activated": True,
        },
        "inputs": {
            "gold_corpus": {
                "path": gold_corpus_path.as_posix(),
                "sha256": phase10.sha256_jsonl(gold_corpus_path),
                "record_count": len(gold_rows),
            },
            "review_manifest": {
                "path": review_manifest_path.as_posix(),
                "sha256_file_bytes": sha256_file(review_manifest_path),
                "status": optional_string(review_manifest.get("status")),
            },
            "source_catalogue": {
                "path": source_catalogue_path.as_posix(),
                "sha256_file_bytes": sha256_file(source_catalogue_path),
                "source_count": len(source_catalogue),
            },
            "approved_embeddings": {
                "path": approved_embeddings_path.as_posix(),
                "sha256": phase10.sha256_jsonl(approved_embeddings_path),
            },
            "embedding_manifest": {
                "path": embedding_manifest_path.as_posix(),
                "sha256_file_bytes": sha256_file(embedding_manifest_path),
                "status": optional_string(embedding_manifest.get("status")),
            },
            "weighted_tags": {
                "path": weighted_tags_path.as_posix(),
                "sha256": phase10.sha256_jsonl(weighted_tags_path),
            },
            "weighted_tags_manifest": {
                "path": weighted_tags_manifest_path.as_posix(),
                "sha256_file_bytes": sha256_file(weighted_tags_manifest_path),
                "status": optional_string(phase12_manifest.get("status")),
            },
        },
        "embedding_identity": identity_dict,
        "counts": diagnostics,
        "outputs": {
            "active_chunk_bundles": {
                "path": paths["active_bundles"].as_posix(),
                "sha256": phase10.sha256_jsonl(paths["active_bundles"]),
                "record_count": len(bundles),
            },
            "reviewed_chunk_concepts": {
                "path": paths["reviewed_concepts"].as_posix(),
                "sha256": phase10.sha256_jsonl(paths["reviewed_concepts"]),
                "record_count": len(reviewed_concepts),
            },
        },
        "database_step": {
            "status": "pending_schema_mapped_upsert",
            "logical_upsert_order": [
                "corpus_version",
                "sources",
                "chunks",
                "chunk_embeddings",
                "chunk_concepts",
                "review_metadata",
            ],
            "idempotency": (
                "Upsert by stable source_id/chunk_id/concept_id plus "
                "corpus_version; never create duplicate active relations."
            ),
            "required_post_upsert_validation": [
                "active chunk count equals activation artifact count",
                "every active chunk has one selected embedding",
                "every active chunk has three reviewed concept rows",
                "retrieval queries filter lifecycle_status=active",
                "concept retrieval filters production_active=true",
                "corpus_version resolves on every retrieved chunk",
            ],
            "reason_pending": (
                "The actual Supabase migration/schema was not available in "
                "the accessible project files, so Phase 13 does not guess "
                "physical table or column names."
            ),
        },
        "exit_gate": exit_gate,
        "next_step": (
            "Complete the Phase 13 database upsert against the actual "
            "Supabase migration/schema, then verify the same active count "
            "is queryable. After that, begin Phase 14 concept-and-domain "
            "retrieval."
        ),
    }

    atomic_json(paths["manifest"], manifest)

    LOGGER.info("Phase 13 activation artifacts complete")
    LOGGER.info(
        "Active chunks: %d",
        require_int(
            diagnostics.get("active_chunk_count"),
            "active_chunk_count",
        ),
    )
    LOGGER.info(
        "Reviewed concept rows: %d",
        require_int(
            diagnostics.get("reviewed_chunk_concept_count"),
            "reviewed_chunk_concept_count",
        ),
    )
    LOGGER.info(
        "Approved sources: %d",
        require_int(
            diagnostics.get("approved_source_count"),
            "approved_source_count",
        ),
    )
    LOGGER.info("Corpus version: %s", corpus_version)
    LOGGER.info("Active bundles: %s", paths["active_bundles"])
    LOGGER.info(
        "Reviewed concepts: %s",
        paths["reviewed_concepts"],
    )
    LOGGER.info("Manifest: %s", paths["manifest"])
    LOGGER.info("Database upsert: PENDING actual schema mapping")

    return manifest


def main() -> int:
    arguments = parse_arguments()
    configure_logging(arguments.log_level)

    try:
        run_phase13(
            project_root=arguments.project_root,
            gold_corpus_path=arguments.gold_corpus,
            review_manifest_path=arguments.review_manifest,
            source_catalogue_path=arguments.source_catalogue,
            approved_embeddings_path=arguments.approved_embeddings,
            embedding_manifest_path=arguments.embedding_manifest,
            weighted_tags_path=arguments.weighted_tags,
            weighted_tags_manifest_path=arguments.weighted_tags_manifest,
            output_directory=arguments.output_directory,
            corpus_version=arguments.corpus_version,
            expected_chunks=arguments.expected_chunks,
            replace=arguments.replace,
        )
    except ActivationError:
        LOGGER.exception("Phase 13 activation failed")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
