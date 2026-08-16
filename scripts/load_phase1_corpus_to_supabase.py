from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, cast

import yaml
from supabase import create_client

LOGGER = logging.getLogger("wth.production.load_phase1_corpus")

SCRIPT_VERSION: Final = "stage1-corpus-migration-v1.2"
STAGE: Final = "stage_1_corpus_migration"
STATUS_COMPLETE: Final = "corpus_migration_complete"

EXPECTED_CORPUS_VERSION: Final = "phase1_active_corpus_v1"
EXPECTED_ACTIVE_CHUNKS: Final = 318
EXPECTED_CONCEPT_RELATIONS: Final = 954
EXPECTED_ACTIVE_SOURCES: Final = 10
EXPECTED_EMBEDDING_DIMENSION: Final = 768
EXPECTED_PHASE1_CONCEPTS: Final = (
    "consciousness",
    "self_identity",
    "reality_appearance",
)
EXPECTED_DOMAINS: Final = ("science", "advaita", "samkhya")

PHASE1_CONCEPT_UUIDS: Final = {
    "self_identity": "10000000-0000-4000-8000-000000000001",
    "consciousness": "10000000-0000-4000-8000-000000000002",
    "reality_appearance": "10000000-0000-4000-8000-000000000003",
}

DEFAULT_PHASE20_MANIFEST: Final = Path(
    "artifacts/phase1/completion/phase1_completion_manifest.json"
)
DEFAULT_ACTIVATION_MANIFEST: Final = Path("artifacts/phase1/active/activation_manifest.json")
DEFAULT_GOLD_CORPUS: Final = Path("artifacts/phase1/reviewed/phase1_reviewed_gold_corpus.jsonl")
DEFAULT_ACTIVE_BUNDLES: Final = Path("artifacts/phase1/active/active_chunk_bundles.jsonl")
DEFAULT_CONCEPT_RELATIONS: Final = Path("artifacts/phase1/active/reviewed_chunk_concepts.jsonl")
DEFAULT_EMBEDDINGS: Final = Path("artifacts/phase1/embeddings/approved_chunk_embeddings.jsonl")
DEFAULT_SOURCE_CATALOGUE: Final = Path("docs/catalogues/phase1_sources.yaml")
DEFAULT_OUTPUT: Final = Path("artifacts/production/corpus_migration_manifest.json")

SOURCE_BATCH_SIZE: Final = 25
CHUNK_BATCH_SIZE: Final = 25
RELATION_BATCH_SIZE: Final = 100

SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")

SCHEMA_REQUIRED_COLUMNS: Final = {
    "sources": (
        "id",
        "corpus_version_id",
        "title",
        "source_type",
        "license_attribution",
        "license_verified",
        "rights_status",
        "source_checksum",
    ),
    "chunks": (
        "id",
        "source_id",
        "domain",
        "citation",
        "full_text",
        "claim_type",
        "review_status",
        "embedding_model",
        "embedding_dimension",
        "embedding",
        "content_hash",
        "review_decision",
        "embedding_provider",
        "embedding_model_revision",
        "embedding_normalization",
        "embedding_task_type",
        "embedding_checksum",
    ),
    "chunk_concepts": (
        "chunk_id",
        "concept_id",
        "weight",
        "raw_similarity",
        "raw_mapping_score",
        "human_label",
        "human_override",
        "production_active",
        "phase1_role",
        "review_status",
        "mapping_method",
        "prototype_version",
        "model_version",
        "updated_at",
    ),
}

CLAIM_TYPE_FALLBACK: Final = {
    "science": "empirical",
    "advaita": "metaphysical",
    "samkhya": "metaphysical",
}

RECONCILE_FLOAT_ABS_TOL: Final = 1e-10
RECONCILE_VECTOR_ABS_TOL: Final = 1e-7


class MigrationError(RuntimeError):
    """Raised when the frozen corpus cannot be migrated safely."""


@dataclass(frozen=True)
class LocalCorpus:
    phase20_manifest: dict[str, Any]
    activation_manifest: dict[str, Any]
    gold_by_id: dict[str, dict[str, Any]]
    bundle_by_id: dict[str, dict[str, Any]]
    embedding_by_id: dict[str, dict[str, Any]]
    source_catalogue_by_id: dict[str, dict[str, Any]]
    concept_relations: list[dict[str, Any]]
    active_source_ids: frozenset[str]


@dataclass(frozen=True)
class PreparedPayload:
    sources: list[dict[str, Any]]
    chunks: list[dict[str, Any]]
    neighbor_updates: list[tuple[str, str | None, str | None]]
    relations: list[dict[str, Any]]
    claim_type_fallback_count: int
    source_rights_mapping: dict[str, str]


@dataclass(frozen=True)
class Reconciliation:
    counts: dict[str, int]
    checks: dict[str, bool]
    semantic_fingerprint_sha256: str
    domain_counts: dict[str, int]
    sample_review: list[dict[str, Any]]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 1: idempotently migrate the frozen WTH Phase 1 corpus "
            "from reviewed local artifacts into live Supabase."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--phase20-manifest",
        type=Path,
        default=DEFAULT_PHASE20_MANIFEST,
    )
    parser.add_argument(
        "--activation-manifest",
        type=Path,
        default=DEFAULT_ACTIVATION_MANIFEST,
    )
    parser.add_argument("--gold-corpus", type=Path, default=DEFAULT_GOLD_CORPUS)
    parser.add_argument(
        "--active-bundles",
        type=Path,
        default=DEFAULT_ACTIVE_BUNDLES,
    )
    parser.add_argument(
        "--concept-relations",
        type=Path,
        default=DEFAULT_CONCEPT_RELATIONS,
    )
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument(
        "--source-catalogue",
        type=Path,
        default=DEFAULT_SOURCE_CATALOGUE,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate frozen artifacts, live schema, and concept seeds only.",
    )
    parser.add_argument(
        "--verify-rerun",
        action="store_true",
        help=("Run the same upserts a second time and prove counts/fingerprint do not change."),
    )
    parser.add_argument(
        "--replace-manifest",
        action="store_true",
        help="Replace an existing corpus migration manifest.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def resolve(project_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise MigrationError(f"Required file is missing: {path}")
    return path


def require_mapping(value: object, description: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MigrationError(f"{description} must be an object.")
    return {str(key): nested for key, nested in value.items()}


def require_sequence(value: object, description: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise MigrationError(f"{description} must be a list.")
    return list(value)


def require_string(value: object, description: str) -> str:
    if not isinstance(value, str):
        raise MigrationError(f"{description} must be a string.")
    stripped = value.strip()
    if not stripped:
        raise MigrationError(f"{description} must be non-empty.")
    return stripped


def optional_string(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value).strip() or None


def require_bool(value: object, description: str) -> bool:
    if not isinstance(value, bool):
        raise MigrationError(f"{description} must be a boolean.")
    return value


def require_int(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MigrationError(f"{description} must be an integer.")
    return value


def require_float(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MigrationError(f"{description} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise MigrationError(f"{description} must be finite.")
    return result


def require_sha256(value: object, description: str) -> str:
    result = require_string(value, description).lower()
    if not SHA256_RE.fullmatch(result):
        raise MigrationError(f"{description} must be a lowercase SHA-256 hex digest.")
    return result


def load_json(path: Path) -> dict[str, Any]:
    require_file(path)
    value: object = json.loads(path.read_text(encoding="utf-8"))
    return require_mapping(value, f"JSON file {path}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    require_file(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value: object = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MigrationError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            rows.append(require_mapping(value, f"{path}:{line_number}"))
    return rows


def load_yaml(path: Path) -> object:
    require_file(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    require_file(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(payload)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            handle.write(text)
        Path(temp_name).replace(path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def deep_values_for_keys(
    value: object,
    keys: set[str],
) -> list[object]:
    found: list[object] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            if key_text in keys:
                found.append(nested)
            found.extend(deep_values_for_keys(nested, keys))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            found.extend(deep_values_for_keys(nested, keys))
    return found


def first_nonblank_deep_string(
    value: object,
    keys: Sequence[str],
) -> str | None:
    candidates = deep_values_for_keys(value, set(keys))
    for candidate in candidates:
        text = optional_string(candidate)
        if text is not None:
            return text
    return None


def first_deep_bool(
    value: object,
    keys: Sequence[str],
) -> bool | None:
    candidates = deep_values_for_keys(value, set(keys))
    for candidate in candidates:
        if isinstance(candidate, bool):
            return candidate
    return None


def textify_lossless(value: object, description: str) -> str:
    if isinstance(value, str):
        result = value.strip()
        if result:
            return result
    if value is None:
        raise MigrationError(f"{description} is missing.")
    if isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes)):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    result = str(value).strip()
    if not result:
        raise MigrationError(f"{description} is blank.")
    return result


def row_id(record: Mapping[str, Any], description: str) -> str:
    for key in ("chunk_id", "id"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise MigrationError(f"{description} is missing chunk_id/id.")


def index_unique_rows(
    records: Iterable[dict[str, Any]],
    *,
    description: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        identifier = row_id(record, description)
        if identifier in result:
            raise MigrationError(f"Duplicate {description} ID: {identifier}")
        result[identifier] = record
    return result


def find_artifact_hash(
    phase20: Mapping[str, Any],
    artifact_name: str,
) -> str | None:
    baseline = phase20.get("baseline")
    if not isinstance(baseline, Mapping):
        return None
    hashes = baseline.get("artifact_hashes")
    if not isinstance(hashes, Mapping):
        return None
    entry = hashes.get(artifact_name)
    if not isinstance(entry, Mapping):
        return None
    value = entry.get("sha256")
    if isinstance(value, str) and SHA256_RE.fullmatch(value.lower()):
        return value.lower()
    return None


def validate_phase20_manifest(
    phase20: Mapping[str, Any],
) -> str:
    if phase20.get("phase") != "phase_20_phase1_completion_and_freeze":
        raise MigrationError("Unexpected Phase 20 manifest phase.")
    if phase20.get("status") != "phase1_frozen_complete":
        raise MigrationError("Phase 20 manifest is not frozen complete.")
    freeze_fingerprint = require_sha256(
        phase20.get("freeze_fingerprint_sha256"),
        "Phase 20 freeze_fingerprint_sha256",
    )

    gate = require_mapping(phase20.get("exit_gate"), "Phase 20 exit_gate")
    if gate.get("passed") is not True:
        raise MigrationError("Phase 20 exit gate did not pass.")

    baseline = require_mapping(phase20.get("baseline"), "Phase 20 baseline")
    if baseline.get("corpus_version") != EXPECTED_CORPUS_VERSION:
        raise MigrationError("Phase 20 corpus version changed.")
    if baseline.get("active_chunk_count") != EXPECTED_ACTIVE_CHUNKS:
        raise MigrationError("Phase 20 active chunk count changed.")
    if baseline.get("reviewed_concept_relation_count") != EXPECTED_CONCEPT_RELATIONS:
        raise MigrationError("Phase 20 concept relation count changed.")

    return freeze_fingerprint


def validate_activation_manifest(
    activation: Mapping[str, Any],
) -> None:
    if activation.get("phase") != "phase_13_activate_approved_phase1_corpus":
        raise MigrationError("Unexpected Phase 13 activation manifest.")
    if activation.get("status") != "activation_artifacts_complete":
        raise MigrationError("Phase 13 activation artifacts are not complete.")
    if activation.get("lifecycle_status") != "active":
        raise MigrationError("Phase 13 lifecycle status is not active.")
    if activation.get("corpus_version") != EXPECTED_CORPUS_VERSION:
        raise MigrationError("Phase 13 corpus version changed.")

    counts = require_mapping(activation.get("counts"), "activation counts")
    if counts.get("active_chunk_count") != EXPECTED_ACTIVE_CHUNKS:
        raise MigrationError("Activation chunk count changed.")
    if counts.get("reviewed_chunk_concept_count") != EXPECTED_CONCEPT_RELATIONS:
        raise MigrationError("Activation concept relation count changed.")
    if counts.get("approved_source_count") != EXPECTED_ACTIVE_SOURCES:
        raise MigrationError("Activation approved source count changed.")


def validate_frozen_artifact_hashes(
    *,
    phase20: Mapping[str, Any],
    activation: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> dict[str, str]:
    aliases = {
        "gold_corpus": "gold_corpus",
        "active_chunk_bundles": "active_chunk_bundles",
        "reviewed_chunk_concepts": "reviewed_chunk_concepts",
        "approved_chunk_embeddings": "approved_chunk_embeddings",
    }

    observed: dict[str, str] = {}
    for logical_name, path in paths.items():
        digest = sha256_file(path)
        observed[logical_name] = digest
        phase20_key = aliases.get(logical_name)
        if phase20_key is None:
            continue
        expected = find_artifact_hash(phase20, phase20_key)
        if expected is not None and digest != expected:
            raise MigrationError(
                f"Frozen artifact hash mismatch for {logical_name}: "
                f"expected {expected}, observed {digest}"
            )

    inputs = activation.get("inputs")
    if isinstance(inputs, Mapping):
        source_entry = inputs.get("source_catalogue")
        if isinstance(source_entry, Mapping):
            expected_source_hash = source_entry.get("sha256_file_bytes")
            if isinstance(expected_source_hash, str):
                observed_source_hash = observed["source_catalogue"]
                if observed_source_hash != expected_source_hash:
                    raise MigrationError(
                        "Source catalogue hash does not match the activation manifest."
                    )

    return observed


def source_catalogue_records(value: object) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        for key in ("sources", "source_catalogue", "items"):
            nested = value.get(key)
            if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
                return [require_mapping(item, "source catalogue item") for item in nested]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [require_mapping(item, "source catalogue item") for item in value]
    raise MigrationError("Could not locate the source list in phase1_sources.yaml.")


def catalogue_source_id(record: Mapping[str, Any]) -> str:
    return require_string(record.get("source_id"), "catalogue source_id")


def load_local_corpus(
    *,
    phase20_path: Path,
    activation_path: Path,
    gold_path: Path,
    bundles_path: Path,
    concept_relations_path: Path,
    embeddings_path: Path,
    source_catalogue_path: Path,
) -> tuple[LocalCorpus, dict[str, str]]:
    phase20 = load_json(phase20_path)
    activation = load_json(activation_path)
    validate_phase20_manifest(phase20)
    validate_activation_manifest(activation)

    frozen_hashes = validate_frozen_artifact_hashes(
        phase20=phase20,
        activation=activation,
        paths={
            "gold_corpus": gold_path,
            "active_chunk_bundles": bundles_path,
            "reviewed_chunk_concepts": concept_relations_path,
            "approved_chunk_embeddings": embeddings_path,
            "source_catalogue": source_catalogue_path,
        },
    )

    gold_rows = load_jsonl(gold_path)
    bundle_rows = load_jsonl(bundles_path)
    relation_rows = load_jsonl(concept_relations_path)
    embedding_rows = load_jsonl(embeddings_path)
    catalogue_rows = source_catalogue_records(load_yaml(source_catalogue_path))

    if len(gold_rows) != EXPECTED_ACTIVE_CHUNKS:
        raise MigrationError(
            f"Expected {EXPECTED_ACTIVE_CHUNKS} gold rows; found {len(gold_rows)}."
        )
    if len(bundle_rows) != EXPECTED_ACTIVE_CHUNKS:
        raise MigrationError(
            f"Expected {EXPECTED_ACTIVE_CHUNKS} active bundles; found {len(bundle_rows)}."
        )
    if len(embedding_rows) != EXPECTED_ACTIVE_CHUNKS:
        raise MigrationError(
            f"Expected {EXPECTED_ACTIVE_CHUNKS} embeddings; found {len(embedding_rows)}."
        )
    if len(relation_rows) != EXPECTED_CONCEPT_RELATIONS:
        raise MigrationError(
            f"Expected {EXPECTED_CONCEPT_RELATIONS} concept relations; found {len(relation_rows)}."
        )

    gold_by_id = index_unique_rows(gold_rows, description="gold chunk")
    bundle_by_id = index_unique_rows(bundle_rows, description="active bundle")
    embedding_by_id = index_unique_rows(
        embedding_rows,
        description="embedding",
    )

    gold_ids = set(gold_by_id)
    if set(bundle_by_id) != gold_ids:
        raise MigrationError("Active bundle chunk IDs do not exactly match the frozen gold corpus.")
    if set(embedding_by_id) != gold_ids:
        raise MigrationError(
            "Approved embedding chunk IDs do not exactly match the frozen gold corpus."
        )

    relation_ids: Counter[str] = Counter()
    relation_concepts: defaultdict[str, set[str]] = defaultdict(set)
    for relation in relation_rows:
        identifier = require_string(
            relation.get("chunk_id"),
            "concept relation chunk_id",
        )
        concept = require_string(
            relation.get("concept_id"),
            f"{identifier} concept_id",
        )
        if identifier not in gold_ids:
            raise MigrationError(f"Concept relation references non-gold chunk: {identifier}")
        if concept not in EXPECTED_PHASE1_CONCEPTS:
            raise MigrationError(f"Unexpected Phase 1 concept {concept!r} for {identifier}.")
        relation_ids[identifier] += 1
        relation_concepts[identifier].add(concept)

    for identifier in sorted(gold_ids):
        if relation_ids[identifier] != 3:
            raise MigrationError(f"{identifier} must have exactly three concept relations.")
        if relation_concepts[identifier] != set(EXPECTED_PHASE1_CONCEPTS):
            raise MigrationError(f"{identifier} concept relation set is incomplete.")

    source_catalogue_by_id: dict[str, dict[str, Any]] = {}
    for record in catalogue_rows:
        identifier = catalogue_source_id(record)
        if identifier in source_catalogue_by_id:
            raise MigrationError(f"Duplicate catalogue source ID: {identifier}")
        source_catalogue_by_id[identifier] = record

    active_source_ids = frozenset(
        require_string(
            record.get("source_id"),
            f"{identifier} source_id",
        )
        for identifier, record in gold_by_id.items()
    )
    if len(active_source_ids) != EXPECTED_ACTIVE_SOURCES:
        raise MigrationError(
            f"Expected {EXPECTED_ACTIVE_SOURCES} active source IDs; found {len(active_source_ids)}."
        )

    missing_catalogue_sources = sorted(active_source_ids - set(source_catalogue_by_id))
    if missing_catalogue_sources:
        raise MigrationError(
            "Active sources missing from source catalogue: " + ", ".join(missing_catalogue_sources)
        )

    return (
        LocalCorpus(
            phase20_manifest=phase20,
            activation_manifest=activation,
            gold_by_id=gold_by_id,
            bundle_by_id=bundle_by_id,
            embedding_by_id=embedding_by_id,
            source_catalogue_by_id=source_catalogue_by_id,
            concept_relations=relation_rows,
            active_source_ids=active_source_ids,
        ),
        frozen_hashes,
    )


def read_environment_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        value = raw_value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def build_supabase_client(project_root: Path) -> Any:
    read_environment_file(project_root / ".env")
    read_environment_file(project_root / ".env.local")

    url = os.getenv("SUPABASE_URL", "").strip()
    key = (
        os.getenv("SUPABASE_SECRET_KEY", "").strip()
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    )

    if not url:
        raise MigrationError("SUPABASE_URL is not configured in the environment/.env.")
    if not key:
        raise MigrationError("SUPABASE_SECRET_KEY or SUPABASE_SERVICE_ROLE_KEY is not configured.")

    try:
        return create_client(url, key)
    except Exception as exc:
        raise MigrationError(f"Could not create Supabase client: {exc}") from exc


def response_rows(response: object, description: str) -> list[dict[str, Any]]:
    data = getattr(response, "data", None)
    if data is None:
        raise MigrationError(f"{description} returned no data payload.")
    if isinstance(data, Mapping):
        return [require_mapping(data, description)]
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
        raise MigrationError(f"{description} returned an unexpected payload.")
    return [require_mapping(item, description) for item in data]


def execute_select(
    client: Any,
    table: str,
    columns: str,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    try:
        query = client.table(table).select(columns)
        if limit is not None:
            query = query.limit(limit)
        response = query.execute()
    except Exception as exc:
        raise MigrationError(f"Supabase select failed for {table}: {exc}") from exc
    return response_rows(response, f"select {table}")


def validate_live_schema(client: Any) -> None:
    for table, columns in SCHEMA_REQUIRED_COLUMNS.items():
        try:
            execute_select(
                client,
                table,
                ",".join(columns),
                limit=1,
            )
        except MigrationError as exc:
            raise MigrationError(
                "Stage 0 hardening migration is not fully applied. "
                f"Schema contract check failed for {table}: {exc}"
            ) from exc

    corpus_rows = execute_select(
        client,
        "corpus_versions",
        "id,version,description,is_active,created_at",
        limit=2,
    )
    if not corpus_rows:
        raise MigrationError("corpus_versions table has no rows.")

    LOGGER.info("Live Stage 0 schema contract: PASS")


def validate_required_concepts(
    client: Any,
) -> dict[str, str]:
    rows = execute_select(
        client,
        "concepts",
        "id,slug,display_name,is_active",
    )
    by_slug = {require_string(row.get("slug"), "concept slug"): row for row in rows}

    resolved: dict[str, str] = {}
    for slug in EXPECTED_PHASE1_CONCEPTS:
        row = by_slug.get(slug)
        if row is None:
            raise MigrationError(f"Required canonical concept is missing from Supabase: {slug}")
        identifier = require_string(row.get("id"), f"{slug} concept id")
        expected_uuid = PHASE1_CONCEPT_UUIDS[slug]
        if identifier != expected_uuid:
            raise MigrationError(
                f"Canonical concept UUID changed for {slug}: "
                f"expected {expected_uuid}, observed {identifier}"
            )
        if row.get("is_active") is not True:
            raise MigrationError(f"Canonical concept is not active: {slug}")
        resolved[slug] = identifier

    LOGGER.info("Required Phase 1 concepts resolved: %d/3", len(resolved))
    return resolved


def source_rights_status(
    source_id: str,
    catalogue: Mapping[str, Any],
    bundles: Iterable[Mapping[str, Any]],
) -> tuple[str, str]:
    relevant_bundles = [
        bundle
        for bundle in bundles
        if first_nonblank_deep_string(
            bundle,
            ("source_id",),
        )
        == source_id
    ]

    for bundle in relevant_bundles:
        value = first_nonblank_deep_string(
            bundle,
            (
                "source_rights_status",
                "rights_status",
            ),
        )
        if value is not None:
            return value, "active_bundle"

    raise MigrationError(
        f"Could not resolve frozen source_rights_status for active source "
        f"{source_id} from the Phase 13 active bundle. Do not substitute the "
        "older source-catalogue acquisition status."
    )


def source_license_attribution(catalogue: Mapping[str, Any]) -> str:
    for key in (
        "license_attribution",
        "license_name",
        "rights_statement",
    ):
        value = optional_string(catalogue.get(key))
        if value is not None:
            return value
    raise MigrationError(f"Source {catalogue.get('source_id')} has no license attribution text.")


def source_checksum(
    source_id: str,
    catalogue: Mapping[str, Any],
    gold_rows: Iterable[Mapping[str, Any]],
) -> str:
    catalogue_checksum = require_sha256(
        catalogue.get("checksum"),
        f"{source_id} catalogue checksum",
    )
    observed = {
        require_sha256(
            row.get("source_checksum"),
            f"{source_id} gold source_checksum",
        )
        for row in gold_rows
        if row.get("source_id") == source_id
    }
    if observed != {catalogue_checksum}:
        raise MigrationError(
            f"Source checksum mismatch for {source_id}: "
            f"catalogue={catalogue_checksum}, gold={sorted(observed)}"
        )
    return catalogue_checksum


def parse_embedding_vector(record: Mapping[str, Any], chunk_id: str) -> list[float]:
    raw = record.get("embedding")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MigrationError(f"{chunk_id} embedding string is not valid JSON.") from exc

    values = require_sequence(raw, f"{chunk_id} embedding")
    vector = [
        require_float(value, f"{chunk_id} embedding[{index}]") for index, value in enumerate(values)
    ]
    if len(vector) != EXPECTED_EMBEDDING_DIMENSION:
        raise MigrationError(
            f"{chunk_id} embedding has {len(vector)} dimensions; "
            f"expected {EXPECTED_EMBEDDING_DIMENSION}."
        )
    return vector


def embedding_field(
    record: Mapping[str, Any],
    key: str,
    aliases: Sequence[str],
    *,
    chunk_id: str,
) -> object:
    if key in record and record[key] is not None:
        return record[key]
    for alias in aliases:
        if alias in record and record[alias] is not None:
            return record[alias]
    nested = first_nonblank_deep_string(record, (key, *aliases))
    if nested is not None:
        return nested
    raise MigrationError(f"{chunk_id} embedding record is missing {key}.")


def embedding_identity(
    record: Mapping[str, Any],
    chunk_id: str,
) -> dict[str, Any]:
    provider = require_string(
        embedding_field(
            record,
            "provider",
            ("embedding_provider",),
            chunk_id=chunk_id,
        ),
        f"{chunk_id} provider",
    )
    model = require_string(
        embedding_field(
            record,
            "model",
            ("embedding_model",),
            chunk_id=chunk_id,
        ),
        f"{chunk_id} model",
    )
    revision = require_string(
        embedding_field(
            record,
            "model_revision",
            ("embedding_model_revision", "revision"),
            chunk_id=chunk_id,
        ),
        f"{chunk_id} model_revision",
    )
    dimensions = require_int(
        embedding_field(
            record,
            "dimensions",
            ("embedding_dimension", "dimension"),
            chunk_id=chunk_id,
        ),
        f"{chunk_id} dimensions",
    )
    normalization = require_string(
        embedding_field(
            record,
            "normalization",
            ("embedding_normalization",),
            chunk_id=chunk_id,
        ),
        f"{chunk_id} normalization",
    )
    task_type = require_string(
        embedding_field(
            record,
            "task_type",
            ("embedding_task_type",),
            chunk_id=chunk_id,
        ),
        f"{chunk_id} task_type",
    )
    text_checksum = require_sha256(
        embedding_field(
            record,
            "text_checksum",
            ("content_hash",),
            chunk_id=chunk_id,
        ),
        f"{chunk_id} embedding text_checksum",
    )
    embedding_checksum = require_sha256(
        embedding_field(
            record,
            "embedding_checksum",
            (),
            chunk_id=chunk_id,
        ),
        f"{chunk_id} embedding checksum",
    )

    if provider != "Google Gemini API":
        raise MigrationError(f"{chunk_id} provider drifted: {provider}")
    if model != "gemini-embedding-2":
        raise MigrationError(f"{chunk_id} embedding model drifted: {model}")
    if revision != "2":
        raise MigrationError(f"{chunk_id} model revision drifted: {revision}")
    if dimensions != EXPECTED_EMBEDDING_DIMENSION:
        raise MigrationError(f"{chunk_id} dimensions drifted: {dimensions}")
    if normalization != "provider_auto_l2":
        raise MigrationError(f"{chunk_id} normalization drifted: {normalization}")

    created_at = first_nonblank_deep_string(
        record,
        ("created_at", "embedding_created_at"),
    )

    return {
        "provider": provider,
        "model": model,
        "model_revision": revision,
        "dimensions": dimensions,
        "normalization": normalization,
        "task_type": task_type,
        "text_checksum": text_checksum,
        "embedding_checksum": embedding_checksum,
        "created_at": created_at,
    }


def candidate_reviewed_texts(record: Mapping[str, Any]) -> list[str]:
    candidates: list[str] = []
    for key in (
        "edited_text",
        "reviewed_text",
        "final_text",
        "chunk_text",
        "full_text",
        "text",
        "content",
    ):
        for value in deep_values_for_keys(record, {key}):
            text = optional_string(value)
            if text is not None and text not in candidates:
                candidates.append(text)
    return candidates


def select_reviewed_text(
    *,
    chunk_id: str,
    gold: Mapping[str, Any],
    bundle: Mapping[str, Any],
    expected_text_checksum: str,
) -> str:
    candidates = candidate_reviewed_texts(gold)
    for candidate in candidate_reviewed_texts(bundle):
        if candidate not in candidates:
            candidates.append(candidate)

    matches = [
        candidate for candidate in candidates if sha256_text(candidate) == expected_text_checksum
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        if all(candidate == matches[0] for candidate in matches):
            return matches[0]
        raise MigrationError(
            f"{chunk_id} has multiple distinct reviewed texts matching the "
            "frozen embedding checksum."
        )

    raise MigrationError(
        f"{chunk_id} reviewed text does not match the frozen embedding "
        f"text_checksum {expected_text_checksum}."
    )


def resolve_review_metadata(
    chunk_id: str,
    gold: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> dict[str, str | None]:
    review_decision = first_nonblank_deep_string(
        gold,
        ("review_decision", "decision"),
    ) or first_nonblank_deep_string(
        bundle,
        ("review_decision", "decision"),
    )
    if review_decision is None:
        raise MigrationError(
            f"{chunk_id} is missing frozen review_decision in both the gold "
            "record and active bundle. Stage 1 must not invent review "
            "provenance."
        )

    if review_decision not in {"include", "include_with_edits"}:
        raise MigrationError(f"{chunk_id} has non-active review decision {review_decision!r}.")

    reviewer = first_nonblank_deep_string(
        gold,
        ("reviewer", "reviewed_by"),
    ) or first_nonblank_deep_string(
        bundle,
        ("reviewer", "reviewed_by"),
    )
    reviewed_at = first_nonblank_deep_string(
        gold,
        ("reviewed_at",),
    ) or first_nonblank_deep_string(
        bundle,
        ("reviewed_at",),
    )
    review_notes = first_nonblank_deep_string(
        gold,
        ("review_notes",),
    ) or first_nonblank_deep_string(
        bundle,
        ("review_notes",),
    )

    return {
        "review_decision": review_decision,
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
        "review_notes": review_notes,
    }


def resolve_claim_type(
    *,
    chunk_id: str,
    domain: str,
    gold: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> tuple[str, bool]:
    value = first_nonblank_deep_string(
        gold,
        ("claim_type",),
    ) or first_nonblank_deep_string(
        bundle,
        ("claim_type",),
    )
    if value is not None:
        if value not in {"empirical", "metaphysical", "normative"}:
            raise MigrationError(f"{chunk_id} has invalid claim_type {value!r}.")
        return value, False

    fallback = CLAIM_TYPE_FALLBACK.get(domain)
    if fallback is None:
        raise MigrationError(f"{chunk_id} cannot resolve claim_type for domain {domain!r}.")
    return fallback, True


def resolve_neighbor(
    record: Mapping[str, Any],
    bundle: Mapping[str, Any],
    key: str,
) -> str | None:
    value = first_nonblank_deep_string(record, (key,))
    if value is not None:
        return value
    return first_nonblank_deep_string(bundle, (key,))


def prepare_sources(
    local: LocalCorpus,
    *,
    corpus_version_id: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    gold_rows = list(local.gold_by_id.values())
    active_bundles = list(local.bundle_by_id.values())

    prepared: list[dict[str, Any]] = []
    rights_mapping: dict[str, str] = {}

    for source_id in sorted(local.active_source_ids):
        catalogue = local.source_catalogue_by_id[source_id]
        checksum = source_checksum(source_id, catalogue, gold_rows)
        rights_status, rights_origin = source_rights_status(
            source_id,
            catalogue,
            active_bundles,
        )
        rights_mapping[source_id] = rights_origin

        source_type = require_string(
            catalogue.get("source_type"),
            f"{source_id} source_type",
        )
        if source_type not in {"paper", "primary_text", "commentary"}:
            raise MigrationError(
                f"{source_id} source_type {source_type!r} is not allowed by the live schema."
            )

        license_verified = first_deep_bool(
            catalogue,
            ("license_verified",),
        )
        if license_verified is None:
            # The catalogue's explicit inclusion/right statuses are preserved
            # separately. Do not infer legal verification from an approval-like
            # string.
            license_verified = False

        prepared.append(
            {
                "id": source_id,
                "corpus_version_id": corpus_version_id,
                "title": require_string(
                    catalogue.get("title"),
                    f"{source_id} title",
                ),
                "author": optional_string(catalogue.get("author")),
                "translator": optional_string(catalogue.get("translator")),
                "editor": optional_string(catalogue.get("editor")),
                "edition": optional_string(catalogue.get("edition")),
                "publication_year": catalogue.get("publication_year"),
                "source_type": source_type,
                "source_url": optional_string(catalogue.get("canonical_url"))
                or optional_string(catalogue.get("source_url")),
                "download_url": optional_string(catalogue.get("download_url")),
                "license_name": optional_string(catalogue.get("license_name")),
                "license_url": optional_string(catalogue.get("license_url")),
                "license_attribution": source_license_attribution(catalogue),
                "license_verified": license_verified,
                "rights_status": rights_status,
                "rights_statement": optional_string(catalogue.get("rights_statement")),
                "rights_jurisdiction": optional_string(catalogue.get("rights_jurisdiction")),
                "accessed_at": optional_string(catalogue.get("accessed_at")),
                "source_checksum": checksum,
            }
        )

    if len(prepared) != EXPECTED_ACTIVE_SOURCES:
        raise MigrationError("Prepared source count changed unexpectedly.")
    return prepared, rights_mapping


def prepare_chunks(
    local: LocalCorpus,
) -> tuple[
    list[dict[str, Any]],
    list[tuple[str, str | None, str | None]],
    int,
]:
    prepared: list[dict[str, Any]] = []
    neighbors: list[tuple[str, str | None, str | None]] = []
    fallback_count = 0

    for chunk_id in sorted(local.gold_by_id):
        gold = local.gold_by_id[chunk_id]
        bundle = local.bundle_by_id[chunk_id]
        embedding = local.embedding_by_id[chunk_id]

        source_id = require_string(
            gold.get("source_id"),
            f"{chunk_id} source_id",
        )
        if source_id not in local.active_source_ids:
            raise MigrationError(f"{chunk_id} references non-active source {source_id}.")

        domain = require_string(gold.get("domain"), f"{chunk_id} domain")
        if domain not in EXPECTED_DOMAINS:
            raise MigrationError(f"{chunk_id} has unexpected domain {domain!r}.")

        citation = require_string(
            gold.get("citation"),
            f"{chunk_id} citation",
        )

        identity = embedding_identity(embedding, chunk_id)
        vector = parse_embedding_vector(embedding, chunk_id)
        reviewed_text = select_reviewed_text(
            chunk_id=chunk_id,
            gold=gold,
            bundle=bundle,
            expected_text_checksum=cast(str, identity["text_checksum"]),
        )
        review = resolve_review_metadata(chunk_id, gold, bundle)
        claim_type, used_fallback = resolve_claim_type(
            chunk_id=chunk_id,
            domain=domain,
            gold=gold,
            bundle=bundle,
        )
        fallback_count += int(used_fallback)

        neighbor_prev = resolve_neighbor(
            gold,
            bundle,
            "neighbor_prev_id",
        )
        neighbor_next = resolve_neighbor(
            gold,
            bundle,
            "neighbor_next_id",
        )
        neighbors.append((chunk_id, neighbor_prev, neighbor_next))

        prepared.append(
            {
                "id": chunk_id,
                "source_id": source_id,
                "domain": domain,
                "citation": citation,
                "full_text": reviewed_text,
                "claim_type": claim_type,
                # Self-referential FKs are attached only after every chunk
                # exists, so batching cannot fail on a future neighbor.
                "neighbor_prev_id": None,
                "neighbor_next_id": None,
                "review_status": "active",
                "embedding_model": identity["model"],
                "embedding_dimension": identity["dimensions"],
                "embedding": vector,
                "content_hash": identity["text_checksum"],
                "review_decision": review["review_decision"],
                "reviewer": review["reviewer"],
                "reviewed_at": review["reviewed_at"],
                "review_notes": review["review_notes"],
                "embedding_provider": identity["provider"],
                "embedding_model_revision": identity["model_revision"],
                "embedding_normalization": identity["normalization"],
                "embedding_task_type": identity["task_type"],
                "embedding_checksum": identity["embedding_checksum"],
                "embedding_created_at": identity["created_at"],
            }
        )

    if len(prepared) != EXPECTED_ACTIVE_CHUNKS:
        raise MigrationError("Prepared chunk count changed unexpectedly.")

    chunk_ids = {row["id"] for row in prepared}
    for chunk_id, previous, following in neighbors:
        for role, candidate in (
            ("neighbor_prev_id", previous),
            ("neighbor_next_id", following),
        ):
            if candidate is not None and candidate not in chunk_ids:
                raise MigrationError(f"{chunk_id} {role} references non-active chunk {candidate}.")

    return prepared, neighbors, fallback_count


def relation_value(
    relation: Mapping[str, Any],
    field: str,
    aliases: Sequence[str] = (),
) -> object:
    if field in relation and relation[field] is not None:
        return relation[field]
    for alias in aliases:
        if alias in relation and relation[alias] is not None:
            return relation[alias]
    raise MigrationError(
        f"Concept relation {relation.get('chunk_id')} / "
        f"{relation.get('concept_id')} is missing {field}."
    )


def prepare_relations(
    local: LocalCorpus,
    concept_uuid_by_slug: Mapping[str, str],
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []

    for relation in sorted(
        local.concept_relations,
        key=lambda row: (
            require_string(row.get("chunk_id"), "relation chunk_id"),
            require_string(row.get("concept_id"), "relation concept_id"),
        ),
    ):
        chunk_id = require_string(
            relation.get("chunk_id"),
            "relation chunk_id",
        )
        concept_slug = require_string(
            relation.get("concept_id"),
            f"{chunk_id} concept_id",
        )
        concept_uuid = concept_uuid_by_slug.get(concept_slug)
        if concept_uuid is None:
            raise MigrationError(f"Cannot resolve concept UUID for {concept_slug}.")

        weight = require_float(
            relation_value(
                relation,
                "calibrated_weight",
                ("weight",),
            ),
            f"{chunk_id}/{concept_slug} calibrated_weight",
        )
        if not 0.0 <= weight <= 1.0:
            raise MigrationError(f"{chunk_id}/{concept_slug} calibrated_weight outside [0,1].")

        human_label = require_string(
            relation_value(relation, "human_label"),
            f"{chunk_id}/{concept_slug} human_label",
        )
        if human_label not in {"positive", "partial", "negative"}:
            raise MigrationError(f"{chunk_id}/{concept_slug} invalid human_label {human_label!r}.")

        prepared.append(
            {
                "chunk_id": chunk_id,
                "concept_id": concept_uuid,
                "weight": weight,
                "raw_similarity": require_float(
                    relation_value(relation, "raw_similarity"),
                    f"{chunk_id}/{concept_slug} raw_similarity",
                ),
                "raw_mapping_score": require_float(
                    relation_value(relation, "raw_mapping_score"),
                    f"{chunk_id}/{concept_slug} raw_mapping_score",
                ),
                "human_label": human_label,
                "human_override": require_bool(
                    relation_value(relation, "human_override"),
                    f"{chunk_id}/{concept_slug} human_override",
                ),
                "production_active": require_bool(
                    relation_value(relation, "production_active"),
                    f"{chunk_id}/{concept_slug} production_active",
                ),
                "phase1_role": (
                    textify_lossless(
                        relation.get("phase1_role"),
                        f"{chunk_id}/{concept_slug} phase1_role",
                    )
                    if relation.get("phase1_role") is not None
                    else None
                ),
                "review_status": textify_lossless(
                    relation_value(relation, "review_status"),
                    f"{chunk_id}/{concept_slug} review_status",
                ),
                "mapping_method": require_string(
                    relation_value(relation, "mapping_method"),
                    f"{chunk_id}/{concept_slug} mapping_method",
                ),
                "prototype_version": require_string(
                    relation_value(relation, "prototype_version"),
                    f"{chunk_id}/{concept_slug} prototype_version",
                ),
                "model_version": require_string(
                    relation_value(relation, "model_version"),
                    f"{chunk_id}/{concept_slug} model_version",
                ),
            }
        )

    if len(prepared) != EXPECTED_CONCEPT_RELATIONS:
        raise MigrationError("Prepared relation count changed unexpectedly.")

    pair_keys = {(row["chunk_id"], row["concept_id"]) for row in prepared}
    if len(pair_keys) != EXPECTED_CONCEPT_RELATIONS:
        raise MigrationError("Prepared concept relations contain duplicates.")

    return prepared


def batched(
    rows: Sequence[dict[str, Any]],
    size: int,
) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield list(rows[start : start + size])


def upsert_batches(
    client: Any,
    *,
    table: str,
    rows: Sequence[dict[str, Any]],
    on_conflict: str,
    batch_size: int,
) -> None:
    for batch_number, batch in enumerate(
        batched(rows, batch_size),
        start=1,
    ):
        try:
            (
                client.table(table)
                .upsert(
                    batch,
                    on_conflict=on_conflict,
                    returning="minimal",
                )
                .execute()
            )
        except Exception as exc:
            raise MigrationError(
                f"Supabase upsert failed for {table}, batch {batch_number}: {exc}"
            ) from exc

    LOGGER.info("Upserted %s rows=%d", table, len(rows))


def update_neighbors(
    client: Any,
    updates: Sequence[tuple[str, str | None, str | None]],
) -> None:
    nonempty = [
        (chunk_id, previous, following)
        for chunk_id, previous, following in updates
        if previous is not None or following is not None
    ]
    for chunk_id, previous, following in nonempty:
        try:
            (
                client.table("chunks")
                .update(
                    {
                        "neighbor_prev_id": previous,
                        "neighbor_next_id": following,
                    }
                )
                .eq("id", chunk_id)
                .execute()
            )
        except Exception as exc:
            raise MigrationError(f"Could not attach neighbors for {chunk_id}: {exc}") from exc
    if nonempty:
        LOGGER.info("Attached chunk neighbor links rows=%d", len(nonempty))


def ensure_corpus_version(
    client: Any,
) -> tuple[str, list[dict[str, Any]]]:
    rows = execute_select(
        client,
        "corpus_versions",
        "id,version,description,is_active,created_at",
    )
    previous_active = [
        row
        for row in rows
        if row.get("is_active") is True and row.get("version") != EXPECTED_CORPUS_VERSION
    ]

    matching = [row for row in rows if row.get("version") == EXPECTED_CORPUS_VERSION]
    if len(matching) > 1:
        raise MigrationError(f"Duplicate corpus_versions row for {EXPECTED_CORPUS_VERSION}.")

    if matching:
        identifier = require_string(
            matching[0].get("id"),
            "Phase 1 corpus version id",
        )
        return identifier, previous_active

    try:
        response = (
            client.table("corpus_versions")
            .upsert(
                {
                    "version": EXPECTED_CORPUS_VERSION,
                    "description": (
                        "Frozen WTH Phase 1 active corpus: 318 reviewed chunks, "
                        "954 reviewed concept relations."
                    ),
                    "is_active": False,
                },
                on_conflict="version",
            )
            .select("id,version,is_active")
            .execute()
        )
    except Exception as exc:
        raise MigrationError(f"Could not create Phase 1 corpus version: {exc}") from exc

    inserted = response_rows(response, "corpus_versions upsert")
    if len(inserted) != 1:
        raise MigrationError("Corpus version upsert did not return exactly one row.")
    identifier = require_string(
        inserted[0].get("id"),
        "Phase 1 corpus version id",
    )
    return identifier, previous_active


def activate_corpus_version(
    client: Any,
    *,
    corpus_version_id: str,
    previous_active: Sequence[Mapping[str, Any]],
) -> None:
    current = execute_select(
        client,
        "corpus_versions",
        "id,version,is_active",
    )
    phase1_rows = [row for row in current if row.get("id") == corpus_version_id]
    if len(phase1_rows) != 1:
        raise MigrationError("Phase 1 corpus version disappeared.")
    if phase1_rows[0].get("is_active") is True:
        return

    previous_ids = [
        require_string(row.get("id"), "previous active corpus id")
        for row in current
        if row.get("is_active") is True and row.get("id") != corpus_version_id
    ]

    try:
        for identifier in previous_ids:
            (
                client.table("corpus_versions")
                .update({"is_active": False})
                .eq("id", identifier)
                .execute()
            )
        (
            client.table("corpus_versions")
            .update({"is_active": True})
            .eq("id", corpus_version_id)
            .execute()
        )
    except Exception as exc:
        # Best-effort restoration if the active-version switch fails between
        # the two HTTP operations.
        for row in previous_active:
            restore_id = optional_string(row.get("id"))
            if not restore_id:
                continue
            try:
                (
                    client.table("corpus_versions")
                    .update({"is_active": True})
                    .eq("id", restore_id)
                    .execute()
                )
            except Exception:
                LOGGER.exception("Best-effort previous corpus activation restore failed.")
        raise MigrationError(f"Could not activate {EXPECTED_CORPUS_VERSION}: {exc}") from exc

    active_rows = [
        row
        for row in execute_select(
            client,
            "corpus_versions",
            "id,version,is_active",
        )
        if row.get("is_active") is True
    ]
    if len(active_rows) != 1:
        raise MigrationError("Exactly one corpus version must be active after migration.")
    if active_rows[0].get("id") != corpus_version_id:
        raise MigrationError("The active corpus version is not Phase 1 after migration.")


def parse_db_vector(value: object, chunk_id: str) -> list[float]:
    if isinstance(value, str):
        raw = value.strip()
        try:
            parsed: object = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MigrationError(f"Database vector is not parseable for {chunk_id}.") from exc
        value = parsed
    values = require_sequence(value, f"{chunk_id} database embedding")
    return [
        require_float(item, f"{chunk_id} database embedding[{index}]")
        for index, item in enumerate(values)
    ]


def assert_float_equal(
    expected: float,
    observed: object,
    *,
    description: str,
    abs_tol: float = RECONCILE_FLOAT_ABS_TOL,
) -> None:
    actual = require_float(observed, description)
    if not math.isclose(
        expected,
        actual,
        rel_tol=0.0,
        abs_tol=abs_tol,
    ):
        raise MigrationError(f"{description} mismatch: expected {expected!r}, observed {actual!r}")


def semantic_source_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "id",
            "corpus_version_id",
            "title",
            "author",
            "translator",
            "editor",
            "edition",
            "publication_year",
            "source_type",
            "source_url",
            "download_url",
            "license_name",
            "license_url",
            "license_attribution",
            "license_verified",
            "rights_status",
            "rights_statement",
            "rights_jurisdiction",
            "accessed_at",
            "source_checksum",
        )
    }


def semantic_chunk_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "id",
            "source_id",
            "domain",
            "citation",
            "full_text",
            "claim_type",
            "neighbor_prev_id",
            "neighbor_next_id",
            "review_status",
            "embedding_model",
            "embedding_dimension",
            "content_hash",
            "review_decision",
            "reviewer",
            "reviewed_at",
            "review_notes",
            "embedding_provider",
            "embedding_model_revision",
            "embedding_normalization",
            "embedding_task_type",
            "embedding_checksum",
            "embedding_created_at",
        )
    }


def semantic_relation_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "chunk_id",
            "concept_id",
            "weight",
            "raw_similarity",
            "raw_mapping_score",
            "human_label",
            "human_override",
            "production_active",
            "phase1_role",
            "review_status",
            "mapping_method",
            "prototype_version",
            "model_version",
        )
    }


def reconcile(
    client: Any,
    *,
    corpus_version_id: str,
    prepared: PreparedPayload,
    concept_uuid_by_slug: Mapping[str, str],
) -> Reconciliation:
    source_rows = execute_select(
        client,
        "sources",
        (
            "id,corpus_version_id,title,author,translator,editor,edition,"
            "publication_year,source_type,source_url,download_url,license_name,"
            "license_url,license_attribution,license_verified,rights_status,"
            "rights_statement,rights_jurisdiction,accessed_at,source_checksum"
        ),
    )
    source_rows = [row for row in source_rows if row.get("corpus_version_id") == corpus_version_id]

    chunk_rows = execute_select(
        client,
        "chunks",
        (
            "id,source_id,domain,citation,full_text,claim_type,neighbor_prev_id,"
            "neighbor_next_id,review_status,embedding_model,embedding_dimension,"
            "embedding,content_hash,review_decision,reviewer,reviewed_at,"
            "review_notes,embedding_provider,embedding_model_revision,"
            "embedding_normalization,embedding_task_type,embedding_checksum,"
            "embedding_created_at"
        ),
    )
    expected_chunk_ids = {str(row["id"]) for row in prepared.chunks}
    chunk_rows = [row for row in chunk_rows if row.get("id") in expected_chunk_ids]

    relation_rows = execute_select(
        client,
        "chunk_concepts",
        (
            "chunk_id,concept_id,weight,raw_similarity,raw_mapping_score,"
            "human_label,human_override,production_active,phase1_role,"
            "review_status,mapping_method,prototype_version,model_version"
        ),
    )
    relation_rows = [row for row in relation_rows if row.get("chunk_id") in expected_chunk_ids]

    expected_sources = {str(row["id"]): row for row in prepared.sources}
    observed_sources = {
        require_string(row.get("id"), "database source id"): row for row in source_rows
    }
    if set(observed_sources) != set(expected_sources):
        raise MigrationError(
            "Database source ID set does not exactly match the frozen active source set."
        )

    expected_chunks = {str(row["id"]): row for row in prepared.chunks}
    observed_chunks = {
        require_string(row.get("id"), "database chunk id"): row for row in chunk_rows
    }
    if set(observed_chunks) != set(expected_chunks):
        raise MigrationError(
            "Database chunk ID set does not exactly match the frozen active chunk set."
        )

    expected_relations = {
        (str(row["chunk_id"]), str(row["concept_id"])): row for row in prepared.relations
    }
    observed_relations = {
        (
            require_string(row.get("chunk_id"), "database relation chunk_id"),
            require_string(
                row.get("concept_id"),
                "database relation concept_id",
            ),
        ): row
        for row in relation_rows
    }
    if set(observed_relations) != set(expected_relations):
        raise MigrationError(
            "Database concept-relation key set does not exactly match the "
            "frozen 954-row relation set."
        )

    for source_id, expected in expected_sources.items():
        observed = observed_sources[source_id]
        for key, expected_value in semantic_source_projection(expected).items():
            if observed.get(key) != expected_value:
                raise MigrationError(
                    f"Source reconciliation mismatch {source_id}.{key}: "
                    f"expected {expected_value!r}, observed {observed.get(key)!r}"
                )

    neighbor_by_id = {
        chunk_id: (previous, following)
        for chunk_id, previous, following in prepared.neighbor_updates
    }

    for chunk_id, expected in expected_chunks.items():
        observed = observed_chunks[chunk_id]
        expected_projection = semantic_chunk_projection(expected)
        expected_projection["neighbor_prev_id"] = neighbor_by_id[chunk_id][0]
        expected_projection["neighbor_next_id"] = neighbor_by_id[chunk_id][1]

        for key, expected_value in expected_projection.items():
            observed_value = observed.get(key)
            if key in {"reviewed_at", "embedding_created_at"}:
                # Supabase/Postgres may normalize equivalent ISO timestamps.
                if expected_value is None and observed_value is None:
                    continue
                if str(expected_value) != str(observed_value):
                    LOGGER.debug(
                        "Timestamp representation differs for %s.%s: %r vs %r",
                        chunk_id,
                        key,
                        expected_value,
                        observed_value,
                    )
                continue
            if observed_value != expected_value:
                raise MigrationError(
                    f"Chunk reconciliation mismatch {chunk_id}.{key}: "
                    f"expected {expected_value!r}, observed {observed_value!r}"
                )

        expected_vector = cast(list[float], expected["embedding"])
        observed_vector = parse_db_vector(observed.get("embedding"), chunk_id)
        if len(observed_vector) != EXPECTED_EMBEDDING_DIMENSION:
            raise MigrationError(f"Database embedding dimension mismatch for {chunk_id}.")
        for index, (expected_value, observed_value) in enumerate(
            zip(expected_vector, observed_vector, strict=True)
        ):
            if not math.isclose(
                expected_value,
                observed_value,
                rel_tol=0.0,
                abs_tol=RECONCILE_VECTOR_ABS_TOL,
            ):
                raise MigrationError(
                    f"Embedding mismatch {chunk_id}[{index}]: "
                    f"expected {expected_value}, observed {observed_value}"
                )

    for relation_key, expected in expected_relations.items():
        observed = observed_relations[relation_key]
        for key, expected_value in semantic_relation_projection(expected).items():
            observed_value = observed.get(key)
            if key in {"weight", "raw_similarity", "raw_mapping_score"}:
                assert_float_equal(
                    float(expected_value),
                    observed_value,
                    description=f"{relation_key}.{key}",
                )
                continue
            if observed_value != expected_value:
                raise MigrationError(
                    f"Relation reconciliation mismatch {relation_key}.{key}: "
                    f"expected {expected_value!r}, observed {observed_value!r}"
                )

    source_ids = set(observed_sources)
    orphan_chunk_source_count = sum(
        1 for row in observed_chunks.values() if row.get("source_id") not in source_ids
    )
    orphan_relation_count = sum(
        1 for row in observed_relations.values() if row.get("chunk_id") not in expected_chunk_ids
    )
    missing_embedding_count = sum(
        1 for row in observed_chunks.values() if row.get("embedding") is None
    )
    missing_citation_count = sum(
        1 for row in observed_chunks.values() if not optional_string(row.get("citation"))
    )

    source_content_pairs = [
        (row.get("source_id"), row.get("content_hash")) for row in observed_chunks.values()
    ]
    duplicate_logical_chunk_count = len(source_content_pairs) - len(set(source_content_pairs))

    domain_counts = Counter(
        require_string(row.get("domain"), "database chunk domain")
        for row in observed_chunks.values()
    )

    semantic_payload = {
        "corpus_version": EXPECTED_CORPUS_VERSION,
        "sources": [
            semantic_source_projection(observed_sources[key]) for key in sorted(observed_sources)
        ],
        "chunks": [
            semantic_chunk_projection(observed_chunks[key]) for key in sorted(observed_chunks)
        ],
        "relations": [
            semantic_relation_projection(observed_relations[key])
            for key in sorted(observed_relations)
        ],
    }
    semantic_fingerprint = canonical_json_sha256(semantic_payload)

    samples: list[dict[str, Any]] = []
    rng = random.Random(20260812)
    concept_slug_by_uuid = {uuid: slug for slug, uuid in concept_uuid_by_slug.items()}

    for domain in EXPECTED_DOMAINS:
        domain_ids = sorted(
            chunk_id for chunk_id, row in observed_chunks.items() if row.get("domain") == domain
        )
        if not domain_ids:
            raise MigrationError(f"No database chunks found for domain {domain}.")
        chosen = rng.choice(domain_ids)
        chunk = observed_chunks[chosen]
        chunk_relation_rows = [
            row for (chunk_id, _), row in observed_relations.items() if chunk_id == chosen
        ]
        samples.append(
            {
                "domain": domain,
                "chunk_id": chosen,
                "source_id": chunk.get("source_id"),
                "citation": chunk.get("citation"),
                "text_checksum": chunk.get("content_hash"),
                "embedding_checksum": chunk.get("embedding_checksum"),
                "embedding_dimension": len(parse_db_vector(chunk.get("embedding"), chosen)),
                "concepts": sorted(
                    (
                        {
                            "concept": concept_slug_by_uuid[
                                require_string(
                                    row.get("concept_id"),
                                    "sample concept id",
                                )
                            ],
                            "human_label": row.get("human_label"),
                            "calibrated_weight": row.get("weight"),
                            "production_active": row.get("production_active"),
                        }
                        for row in chunk_relation_rows
                    ),
                    key=lambda item: str(item["concept"]),
                ),
            }
        )

    counts = {
        "active_sources": len(observed_sources),
        "active_chunks": len(observed_chunks),
        "chunk_concepts": len(observed_relations),
        "required_phase1_concepts": len(concept_uuid_by_slug),
        "orphan_chunk_sources": orphan_chunk_source_count,
        "orphan_chunk_concepts": orphan_relation_count,
        "chunks_without_embeddings": missing_embedding_count,
        "chunks_without_citations": missing_citation_count,
        "duplicate_logical_chunks": duplicate_logical_chunk_count,
    }

    checks = {
        "318_active_chunks_loaded": len(observed_chunks) == EXPECTED_ACTIVE_CHUNKS,
        "954_chunk_concept_rows_loaded": len(observed_relations) == EXPECTED_CONCEPT_RELATIONS,
        "3_required_concepts_resolved": len(concept_uuid_by_slug) == 3,
        "all_source_references_resolve": orphan_chunk_source_count == 0,
        "all_embeddings_present": missing_embedding_count == 0,
        "all_embeddings_dimension_768": all(
            len(parse_db_vector(row.get("embedding"), chunk_id)) == EXPECTED_EMBEDDING_DIMENSION
            for chunk_id, row in observed_chunks.items()
        ),
        "all_citations_present": missing_citation_count == 0,
        "all_chunk_ids_match_frozen_baseline": set(observed_chunks) == set(expected_chunks),
        "all_concept_weights_match_frozen_baseline": True,
        "no_orphan_records": (orphan_chunk_source_count == 0 and orphan_relation_count == 0),
        "no_duplicate_logical_records": duplicate_logical_chunk_count == 0,
    }

    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise MigrationError("Post-load reconciliation failed: " + ", ".join(failed))

    return Reconciliation(
        counts=counts,
        checks=checks,
        semantic_fingerprint_sha256=semantic_fingerprint,
        domain_counts=dict(sorted(domain_counts.items())),
        sample_review=samples,
    )


def prepare_payload(
    local: LocalCorpus,
    *,
    corpus_version_id: str,
    concept_uuid_by_slug: Mapping[str, str],
) -> PreparedPayload:
    sources, source_rights_mapping = prepare_sources(
        local,
        corpus_version_id=corpus_version_id,
    )
    chunks, neighbors, fallback_count = prepare_chunks(local)
    relations = prepare_relations(local, concept_uuid_by_slug)

    return PreparedPayload(
        sources=sources,
        chunks=chunks,
        neighbor_updates=neighbors,
        relations=relations,
        claim_type_fallback_count=fallback_count,
        source_rights_mapping=source_rights_mapping,
    )


def perform_upserts(
    client: Any,
    prepared: PreparedPayload,
) -> None:
    upsert_batches(
        client,
        table="sources",
        rows=prepared.sources,
        on_conflict="id",
        batch_size=SOURCE_BATCH_SIZE,
    )
    upsert_batches(
        client,
        table="chunks",
        rows=prepared.chunks,
        on_conflict="id",
        batch_size=CHUNK_BATCH_SIZE,
    )
    update_neighbors(client, prepared.neighbor_updates)
    upsert_batches(
        client,
        table="chunk_concepts",
        rows=prepared.relations,
        on_conflict="chunk_id,concept_id",
        batch_size=RELATION_BATCH_SIZE,
    )


def output_manifest(
    *,
    freeze_fingerprint: str,
    frozen_hashes: Mapping[str, str],
    corpus_version_id: str,
    concept_uuid_by_slug: Mapping[str, str],
    prepared: PreparedPayload,
    reconciliation: Reconciliation,
    rerun_reconciliation: Reconciliation | None,
) -> dict[str, Any]:
    rerun_verified = rerun_reconciliation is not None
    if rerun_reconciliation is not None:
        if (
            rerun_reconciliation.semantic_fingerprint_sha256
            != reconciliation.semantic_fingerprint_sha256
        ):
            raise MigrationError(
                "Second-run semantic fingerprint differs from first-run fingerprint."
            )
        if rerun_reconciliation.counts != reconciliation.counts:
            raise MigrationError("Second-run database counts differ from first-run counts.")

    exit_gate = {
        **reconciliation.checks,
        "second_loader_run_produces_no_duplication": rerun_verified,
        "supabase_semantically_matches_frozen_phase1": True,
    }
    passed = all(exit_gate.values())

    return {
        "stage": STAGE,
        "status": STATUS_COMPLETE if passed else "corpus_migration_incomplete",
        "script_version": SCRIPT_VERSION,
        "generated_at": utc_now(),
        "frozen_phase1": {
            "freeze_fingerprint_sha256": freeze_fingerprint,
            "corpus_version": EXPECTED_CORPUS_VERSION,
            "active_chunks": EXPECTED_ACTIVE_CHUNKS,
            "active_sources": EXPECTED_ACTIVE_SOURCES,
            "concept_relations": EXPECTED_CONCEPT_RELATIONS,
            "embedding_dimension": EXPECTED_EMBEDDING_DIMENSION,
            "artifact_sha256": dict(sorted(frozen_hashes.items())),
        },
        "database": {
            "corpus_version_id": corpus_version_id,
            "required_phase1_concepts": dict(sorted(concept_uuid_by_slug.items())),
            "counts": reconciliation.counts,
            "domain_counts": reconciliation.domain_counts,
            "semantic_fingerprint_sha256": (reconciliation.semantic_fingerprint_sha256),
        },
        "migration_policy": {
            "idempotent_upserts": True,
            "source_conflict_key": "id",
            "chunk_conflict_key": "id",
            "concept_relation_conflict_key": "chunk_id,concept_id",
            "stable_chunk_ids_preserved": True,
            "human_labels_authoritative": True,
            "chunk_concepts_weight_semantics": "calibrated_weight",
            "concept_eligibility_field": "production_active",
            "corpus_version_resolution": (
                "chunks.source_id -> sources.corpus_version_id -> corpus_versions.id"
            ),
            "embeddings_regenerated": False,
            "semantic_pipeline_changed": False,
            "claim_type_fallback_policy": CLAIM_TYPE_FALLBACK,
            "claim_type_fallback_count": prepared.claim_type_fallback_count,
            "source_rights_status_mapping_origin": (prepared.source_rights_mapping),
        },
        "technical_review_samples": reconciliation.sample_review,
        "rerun_verification": {
            "requested": rerun_verified,
            "passed": rerun_verified,
            "first_semantic_fingerprint_sha256": (reconciliation.semantic_fingerprint_sha256),
            "second_semantic_fingerprint_sha256": (
                rerun_reconciliation.semantic_fingerprint_sha256
                if rerun_reconciliation is not None
                else None
            ),
            "counts_unchanged": (
                rerun_reconciliation is not None
                and rerun_reconciliation.counts == reconciliation.counts
            ),
        },
        "exit_gate": {
            "passed": passed,
            **exit_gate,
        },
        "exit_condition": (
            "The Supabase representation is semantically equivalent to the "
            "frozen Phase 1 artifact representation."
        ),
        "next_stage": (
            "Stage 2: database connectivity API and GET /api/chunk/{id}." if passed else None
        ),
    }


def run_stage1(args: argparse.Namespace) -> int:
    project_root = args.project_root.resolve()
    output_path = resolve(project_root, args.output)

    if output_path.exists() and not args.replace_manifest and not args.dry_run:
        raise MigrationError(
            f"Migration manifest already exists: {output_path}. "
            "Use --replace-manifest after intentionally rerunning validation."
        )

    LOGGER.info("Stage 1 starting: %s", SCRIPT_VERSION)
    LOGGER.info(
        "Target frozen corpus: %s chunks=%d relations=%d",
        EXPECTED_CORPUS_VERSION,
        EXPECTED_ACTIVE_CHUNKS,
        EXPECTED_CONCEPT_RELATIONS,
    )

    phase20_path = resolve(project_root, args.phase20_manifest)
    activation_path = resolve(project_root, args.activation_manifest)
    gold_path = resolve(project_root, args.gold_corpus)
    bundles_path = resolve(project_root, args.active_bundles)
    relation_path = resolve(project_root, args.concept_relations)
    embeddings_path = resolve(project_root, args.embeddings)
    source_catalogue_path = resolve(project_root, args.source_catalogue)

    local, frozen_hashes = load_local_corpus(
        phase20_path=phase20_path,
        activation_path=activation_path,
        gold_path=gold_path,
        bundles_path=bundles_path,
        concept_relations_path=relation_path,
        embeddings_path=embeddings_path,
        source_catalogue_path=source_catalogue_path,
    )
    freeze_fingerprint = require_sha256(
        local.phase20_manifest.get("freeze_fingerprint_sha256"),
        "Phase 20 freeze_fingerprint_sha256",
    )
    LOGGER.info(
        "Frozen local corpus validated: chunks=%d sources=%d relations=%d",
        len(local.gold_by_id),
        len(local.active_source_ids),
        len(local.concept_relations),
    )

    client = build_supabase_client(project_root)
    validate_live_schema(client)
    concept_uuid_by_slug = validate_required_concepts(client)

    if args.dry_run:
        # Preparation without writes catches shape/checksum/provenance problems,
        # while a placeholder UUID lets the source rows be assembled.
        prepared = prepare_payload(
            local,
            corpus_version_id="00000000-0000-0000-0000-000000000000",
            concept_uuid_by_slug=concept_uuid_by_slug,
        )
        LOGGER.info(
            "DRY RUN PASS: prepared sources=%d chunks=%d relations=%d",
            len(prepared.sources),
            len(prepared.chunks),
            len(prepared.relations),
        )
        LOGGER.info("No Supabase corpus rows were written. Run without --dry-run to migrate.")
        return 0

    corpus_version_id, previous_active = ensure_corpus_version(client)
    prepared = prepare_payload(
        local,
        corpus_version_id=corpus_version_id,
        concept_uuid_by_slug=concept_uuid_by_slug,
    )

    perform_upserts(client, prepared)

    # Reconcile before activation so a partially bad upload cannot become the
    # production corpus version.
    first = reconcile(
        client,
        corpus_version_id=corpus_version_id,
        prepared=prepared,
        concept_uuid_by_slug=concept_uuid_by_slug,
    )
    LOGGER.info(
        "Initial reconciliation PASS: fingerprint=%s",
        first.semantic_fingerprint_sha256,
    )

    activate_corpus_version(
        client,
        corpus_version_id=corpus_version_id,
        previous_active=previous_active,
    )

    # Validate again after activation.
    first = reconcile(
        client,
        corpus_version_id=corpus_version_id,
        prepared=prepared,
        concept_uuid_by_slug=concept_uuid_by_slug,
    )

    second: Reconciliation | None = None
    if args.verify_rerun:
        LOGGER.info("Starting second identical upsert pass for idempotency proof.")
        perform_upserts(client, prepared)
        second = reconcile(
            client,
            corpus_version_id=corpus_version_id,
            prepared=prepared,
            concept_uuid_by_slug=concept_uuid_by_slug,
        )
        if first.semantic_fingerprint_sha256 != second.semantic_fingerprint_sha256:
            raise MigrationError("Idempotency failure: semantic fingerprint changed on rerun.")
        if first.counts != second.counts:
            raise MigrationError("Idempotency failure: database counts changed on rerun.")
        LOGGER.info("Second-run idempotency proof: PASS")

    manifest = output_manifest(
        freeze_fingerprint=freeze_fingerprint,
        frozen_hashes=frozen_hashes,
        corpus_version_id=corpus_version_id,
        concept_uuid_by_slug=concept_uuid_by_slug,
        prepared=prepared,
        reconciliation=first,
        rerun_reconciliation=second,
    )
    atomic_write_json(output_path, manifest)

    gate = require_mapping(manifest.get("exit_gate"), "manifest exit_gate")
    if gate.get("passed") is not True:
        raise MigrationError("Stage 1 exit gate did not pass.")

    LOGGER.info(
        "Stage 1 COMPLETE: active_chunks=%d chunk_concepts=%d active_sources=%d",
        first.counts["active_chunks"],
        first.counts["chunk_concepts"],
        first.counts["active_sources"],
    )
    LOGGER.info(
        "DB semantic fingerprint=%s",
        first.semantic_fingerprint_sha256,
    )
    LOGGER.info("Exit gate passed: True")
    LOGGER.info("Migration manifest: %s", output_path)
    return 0


def main() -> int:
    args = parse_arguments()
    configure_logging(args.log_level)
    try:
        return run_stage1(args)
    except MigrationError:
        LOGGER.exception("Stage 1 FAILED")
        return 1
    except KeyboardInterrupt:
        LOGGER.exception("Stage 1 interrupted.")
        return 130
    except Exception:
        LOGGER.exception("Stage 1 failed with an unexpected error.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
