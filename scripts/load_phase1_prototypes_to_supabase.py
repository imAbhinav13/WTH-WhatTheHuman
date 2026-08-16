from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from supabase import create_client

LOGGER = logging.getLogger("wth.production.load_phase1_prototypes")

SCRIPT_VERSION: Final = "stage3.5d-prototype-migration-v1.0"
STAGE: Final = "stage_3_5d_concept_prototype_migration"
STATUS_COMPLETE: Final = "concept_prototype_migration_complete"

TABLE_NAME: Final = "concept_prototype_embeddings"

PROTOTYPE_VERSION: Final = "phase1-prototype-v2"

EXPECTED_CONCEPTS: Final = (
    "consciousness",
    "self_identity",
    "reality_appearance",
)

PHASE1_CONCEPT_UUIDS: Final = {
    "self_identity": ("10000000-0000-4000-8000-000000000001"),
    "consciousness": ("10000000-0000-4000-8000-000000000002"),
    "reality_appearance": ("10000000-0000-4000-8000-000000000003"),
}

EXPECTED_PROVIDER: Final = "Google Gemini API"
EXPECTED_MODEL: Final = "gemini-embedding-2"
EXPECTED_MODEL_REVISION: Final = "2"
EXPECTED_DIMENSIONS: Final = 768
EXPECTED_NORMALIZATION: Final = "provider_auto_l2"
EXPECTED_EMBEDDING_ORIGIN: Final = "provider"

EXPECTED_QUERY_COUNT: Final = 18
EXPECTED_PASSAGE_COUNT: Final = 45
EXPECTED_TOTAL_COUNT: Final = 63

EXPECTED_ROLE_COUNTS: Final = {
    "question": 18,
    "positive": 27,
    "hard_negative": 18,
}

EXPECTED_PER_CONCEPT_ROLE_COUNTS: Final = {
    "question": 6,
    "positive": 9,
    "hard_negative": 6,
}

# These are semantic JSONL hashes: each parsed row is canonicalized with
# sorted keys and compact JSON, followed by "\n", then SHA-256 hashed.
EXPECTED_QUERY_JSONL_SHA256: Final = (
    "d4dd9c89525332bfbc492d5464e13b83432fe14096fb15d22e4247fb72ef05b4"
)
EXPECTED_PASSAGE_JSONL_SHA256: Final = (
    "5081c2771b5a7f173763194fad904e660b46ef337d460805ff7ae066fe6b6d6b"
)

EXPECTED_EMBEDDING_MANIFEST_FILE_SHA256: Final = (
    "efbf7641d74360ed70f3b6ebf4fa894b0252f0107a8ca11cb5cefea659c85e27"
)
EXPECTED_PROTOTYPE_YAML_FILE_SHA256: Final = (
    "397d1052601e418a83d891ee9285ff4ec9f4766e26903f216d4d4e6f52006104"
)

DEFAULT_QUERY_PROTOTYPES: Final = Path(
    "artifacts/phase1/embeddings/query_prototype_embeddings.jsonl"
)
DEFAULT_PASSAGE_PROTOTYPES: Final = Path(
    "artifacts/phase1/embeddings/passage_prototype_embeddings.jsonl"
)
DEFAULT_EMBEDDING_MANIFEST: Final = Path("artifacts/phase1/embeddings/embedding_manifest.json")
DEFAULT_PROTOTYPE_YAML: Final = Path("data/concepts/phase1_concept_prototypes.yaml")
DEFAULT_OUTPUT: Final = Path("artifacts/production/concept_prototype_migration_manifest.json")

UPSERT_BATCH_SIZE: Final = 25
VECTOR_ABS_TOL: Final = 1e-7

SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")


class PrototypeMigrationError(RuntimeError):
    """Raised when frozen prototype migration is unsafe."""


@dataclass(frozen=True)
class FrozenPrototypeRow:
    """Validated local prototype row."""

    record_id: str
    concept_slug: str
    prototype_role: str
    record_type: str

    embedding: tuple[float, ...]

    provider: str
    model: str
    model_revision: str
    dimensions: int
    normalization: str
    task_type: str
    task_instruction: str

    text_checksum: str
    embedding_input_checksum: str
    embedding_checksum: str
    embedding_origin: str

    source_artifact_sha256: str

    chunk_id: str | None
    source_id: str | None
    domain: str | None
    evaluation_split: str | None
    citation: str | None
    title: str | None
    reviewed_labels: dict[str, str] | None

    estimated_tokens: int | None
    actual_tokens: int | None
    created_at: str


@dataclass(frozen=True)
class LocalPrototypeSet:
    """Complete validated frozen prototype set."""

    rows: tuple[FrozenPrototypeRow, ...]
    query_semantic_sha256: str
    passage_semantic_sha256: str
    embedding_manifest_file_sha256: str
    prototype_yaml_file_sha256: str


@dataclass(frozen=True)
class Reconciliation:
    """Post-upsert database reconciliation."""

    row_count: int
    role_counts: dict[str, int]
    concept_role_counts: dict[str, dict[str, int]]
    semantic_fingerprint_sha256: str
    checks: dict[str, bool]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_iso_datetime(
    value: object,
    description: str,
) -> datetime:
    text = require_string(
        value,
        description,
    )

    try:
        parsed = datetime.fromisoformat(
            text.replace(
                "Z",
                "+00:00",
            )
        )
    except ValueError as exc:
        raise PrototypeMigrationError(f"{description} is not a valid ISO-8601 timestamp.") from exc

    if parsed.tzinfo is None:
        raise PrototypeMigrationError(f"{description} must include a timezone.")

    return parsed.astimezone(UTC)


def configure_logging(
    level: str,
) -> None:
    logging.basicConfig(
        level=getattr(
            logging,
            level.upper(),
            logging.INFO,
        ),
        format=("%(asctime)s %(levelname)s %(name)s %(message)s"),
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 3.5D: idempotently load the frozen "
            "Phase 10 concept-prototype embeddings into "
            "Supabase for runtime concept activation."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--query-prototypes",
        type=Path,
        default=DEFAULT_QUERY_PROTOTYPES,
    )
    parser.add_argument(
        "--passage-prototypes",
        type=Path,
        default=DEFAULT_PASSAGE_PROTOTYPES,
    )
    parser.add_argument(
        "--embedding-manifest",
        type=Path,
        default=DEFAULT_EMBEDDING_MANIFEST,
    )
    parser.add_argument(
        "--prototype-yaml",
        type=Path,
        default=DEFAULT_PROTOTYPE_YAML,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate frozen local artifacts plus the "
            "live Supabase schema/concept seeds without "
            "writing prototype rows."
        ),
    )
    parser.add_argument(
        "--verify-rerun",
        action="store_true",
        help=("Repeat the same upsert and prove the DB fingerprint/counts remain unchanged."),
    )
    parser.add_argument(
        "--replace-manifest",
        action="store_true",
        help=("Replace an existing Stage 3.5D migration manifest."),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
    )

    return parser.parse_args()


def resolve(
    project_root: Path,
    path: Path,
) -> Path:
    if path.is_absolute():
        return path.resolve()

    return (project_root / path).resolve()


def require_file(
    path: Path,
) -> Path:
    if not path.is_file():
        raise PrototypeMigrationError(f"Required file is missing: {path}")

    return path


def require_mapping(
    value: object,
    description: str,
) -> dict[str, Any]:
    if not isinstance(
        value,
        Mapping,
    ):
        raise PrototypeMigrationError(f"{description} must be an object.")

    return {str(key): nested for key, nested in value.items()}


def require_sequence(
    value: object,
    description: str,
) -> list[Any]:
    if not isinstance(
        value,
        Sequence,
    ) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        raise PrototypeMigrationError(f"{description} must be a list.")

    return list(value)


def require_string(
    value: object,
    description: str,
) -> str:
    if not isinstance(
        value,
        str,
    ):
        raise PrototypeMigrationError(f"{description} must be a string.")

    result = value.strip()

    if not result:
        raise PrototypeMigrationError(f"{description} must be non-empty.")

    return result


def optional_string(
    value: object,
) -> str | None:
    if value is None:
        return None

    if not isinstance(
        value,
        str,
    ):
        raise PrototypeMigrationError("Optional string field has a non-string value.")

    result = value.strip()

    return result or None


def require_int(
    value: object,
    description: str,
) -> int:
    if isinstance(value, bool) or not isinstance(
        value,
        int,
    ):
        raise PrototypeMigrationError(f"{description} must be an integer.")

    return value


def optional_int(
    value: object,
    description: str,
) -> int | None:
    if value is None:
        return None

    return require_int(
        value,
        description,
    )


def require_float(
    value: object,
    description: str,
) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise PrototypeMigrationError(f"{description} must be numeric.")

    result = float(value)

    if not math.isfinite(result):
        raise PrototypeMigrationError(f"{description} must be finite.")

    return result


def require_sha256(
    value: object,
    description: str,
) -> str:
    result = require_string(
        value,
        description,
    ).lower()

    if not SHA256_RE.fullmatch(result):
        raise PrototypeMigrationError(f"{description} must be a lowercase SHA-256 digest.")

    return result


def load_json(
    path: Path,
) -> dict[str, Any]:
    require_file(path)

    try:
        raw: object = json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )
    except json.JSONDecodeError as exc:
        raise PrototypeMigrationError(f"Invalid JSON in {path}: {exc}") from exc

    return require_mapping(
        raw,
        f"JSON file {path}",
    )


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    require_file(path)

    rows: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, raw_line in enumerate(
            handle,
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                continue

            try:
                raw: object = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PrototypeMigrationError(
                    f"Invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc

            rows.append(
                require_mapping(
                    raw,
                    f"{path}:{line_number}",
                )
            )

    return rows


def sha256_file(
    path: Path,
) -> str:
    require_file(path)

    digest = hashlib.sha256()

    with path.open(
        "rb",
    ) as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def canonical_json_bytes(
    value: object,
) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_json_value(
    value: object,
) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_jsonl_content(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, raw_line in enumerate(
            handle,
            start=1,
        ):
            stripped = raw_line.strip()

            if not stripped:
                continue

            try:
                value: object = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise PrototypeMigrationError(
                    f"Invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc

            digest.update(canonical_json_bytes(value))
            digest.update(b"\n")

    return digest.hexdigest()


def embedding_checksum(
    vector: Sequence[float],
) -> str:
    """Exact Phase 9 embedding-checksum algorithm."""

    return sha256_json_value([float(value) for value in vector])


def atomic_write_json(
    path: Path,
    payload: Mapping[str, object],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def read_environment_file(
    path: Path,
) -> None:
    if not path.is_file():
        return

    for raw_line in path.read_text(
        encoding="utf-8",
    ).splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, raw_value = line.split(
            "=",
            1,
        )

        key = key.strip()
        value = raw_value.strip()

        if not key or key in os.environ:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        os.environ[key] = value


def build_supabase_client(
    project_root: Path,
) -> Any:
    read_environment_file(project_root / ".env")
    read_environment_file(project_root / ".env.local")
    read_environment_file(project_root / "apps" / "api" / ".env")
    read_environment_file(project_root / "apps" / "api" / ".env.local")

    url = os.getenv(
        "SUPABASE_URL",
        "",
    ).strip()

    key = (
        os.getenv(
            "SUPABASE_SECRET_KEY",
            "",
        ).strip()
        or os.getenv(
            "SUPABASE_SERVICE_ROLE_KEY",
            "",
        ).strip()
    )

    if not url:
        raise PrototypeMigrationError("SUPABASE_URL is not configured.")

    if not key:
        raise PrototypeMigrationError(
            "SUPABASE_SECRET_KEY or SUPABASE_SERVICE_ROLE_KEY is not configured."
        )

    try:
        return create_client(
            url,
            key,
        )
    except Exception as exc:
        raise PrototypeMigrationError("Could not create the backend Supabase client.") from exc


def response_rows(
    response: object,
    description: str,
) -> list[dict[str, Any]]:
    data = getattr(
        response,
        "data",
        None,
    )

    if data is None:
        raise PrototypeMigrationError(f"{description} returned no data payload.")

    if isinstance(
        data,
        Mapping,
    ):
        return [
            require_mapping(
                data,
                description,
            )
        ]

    if not isinstance(
        data,
        Sequence,
    ) or isinstance(
        data,
        (str, bytes, bytearray),
    ):
        raise PrototypeMigrationError(f"{description} returned an unexpected payload.")

    return [
        require_mapping(
            item,
            description,
        )
        for item in data
    ]


def validate_embedding_manifest(
    manifest: Mapping[str, Any],
) -> None:
    if manifest.get("status") != "complete":
        raise PrototypeMigrationError("Embedding manifest status is not complete.")

    identity = require_mapping(
        manifest.get("embedding_identity"),
        "embedding manifest identity",
    )

    expected_identity = {
        "provider": EXPECTED_PROVIDER,
        "model": EXPECTED_MODEL,
        "model_revision": (EXPECTED_MODEL_REVISION),
        "dimensions": (EXPECTED_DIMENSIONS),
        "normalization": (EXPECTED_NORMALIZATION),
        "query_task_type": ("search_query"),
        "document_task_type": ("search_document"),
    }

    for key, expected in expected_identity.items():
        if identity.get(key) != expected:
            raise PrototypeMigrationError(
                "Embedding identity changed: "
                f"{key} expected {expected!r}, "
                f"observed {identity.get(key)!r}."
            )

    inputs = require_mapping(
        manifest.get("inputs"),
        "embedding manifest inputs",
    )

    prototypes = require_mapping(
        inputs.get("concept_prototypes"),
        "embedding manifest concept_prototypes",
    )

    if prototypes.get("prototype_version") != PROTOTYPE_VERSION:
        raise PrototypeMigrationError("Frozen prototype version changed.")

    if prototypes.get("sha256_file_bytes") != EXPECTED_PROTOTYPE_YAML_FILE_SHA256:
        raise PrototypeMigrationError("Embedding manifest prototype YAML hash changed.")

    outputs = require_mapping(
        manifest.get("outputs"),
        "embedding manifest outputs",
    )

    query_output = require_mapping(
        outputs.get("query_prototype_embeddings"),
        "query prototype output",
    )

    passage_output = require_mapping(
        outputs.get("passage_prototype_embeddings"),
        "passage prototype output",
    )

    if (
        query_output.get("record_count") != EXPECTED_QUERY_COUNT
        or query_output.get("sha256") != EXPECTED_QUERY_JSONL_SHA256
    ):
        raise PrototypeMigrationError("Frozen query-prototype output contract changed.")

    if (
        passage_output.get("record_count") != EXPECTED_PASSAGE_COUNT
        or passage_output.get("sha256") != EXPECTED_PASSAGE_JSONL_SHA256
    ):
        raise PrototypeMigrationError("Frozen passage-prototype output contract changed.")

    gate = require_mapping(
        manifest.get("exit_gate"),
        "embedding manifest exit_gate",
    )

    required_true = (
        "all_prototype_vectors_same_frozen_configuration",
        "no_mock_vectors",
        "single_embedding_identity",
    )

    for key in required_true:
        if gate.get(key) is not True:
            raise PrototypeMigrationError(f"Embedding manifest exit gate failed: {key}.")


def parse_vector(
    row: Mapping[str, Any],
    *,
    record_id: str,
) -> tuple[float, ...]:
    raw = row.get("embedding")

    if isinstance(
        raw,
        str,
    ):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PrototypeMigrationError(f"{record_id} embedding string is invalid JSON.") from exc

    values = require_sequence(
        raw,
        f"{record_id} embedding",
    )

    if len(values) != EXPECTED_DIMENSIONS:
        raise PrototypeMigrationError(
            f"{record_id} has {len(values)} dimensions; expected {EXPECTED_DIMENSIONS}."
        )

    vector = tuple(
        require_float(
            value,
            f"{record_id} embedding[{index}]",
        )
        for index, value in enumerate(values)
    )

    expected_checksum = require_sha256(
        row.get("embedding_checksum"),
        f"{record_id} embedding_checksum",
    )

    observed_checksum = embedding_checksum(vector)

    if observed_checksum != expected_checksum:
        raise PrototypeMigrationError(f"{record_id} vector checksum mismatch.")

    return vector


def parse_reviewed_labels(
    value: object,
    *,
    record_id: str,
) -> dict[str, str] | None:
    if value is None:
        return None

    mapping = require_mapping(
        value,
        f"{record_id} reviewed_labels",
    )

    result: dict[
        str,
        str,
    ] = {}

    for concept in EXPECTED_CONCEPTS:
        label = require_string(
            mapping.get(concept),
            (f"{record_id} reviewed_labels.{concept}"),
        )

        if label not in {
            "positive",
            "partial",
            "negative",
        }:
            raise PrototypeMigrationError(
                f"{record_id} has invalid reviewed label {concept}={label!r}."
            )

        result[concept] = label

    if set(mapping) != set(EXPECTED_CONCEPTS):
        raise PrototypeMigrationError(
            f"{record_id} reviewed_labels does not contain exactly the three Phase 1 concepts."
        )

    return result


def parse_local_row(
    raw: Mapping[str, Any],
    *,
    source_artifact_sha256: str,
    expected_source: str,
) -> FrozenPrototypeRow:
    record_id = require_string(
        raw.get("record_id"),
        "prototype record_id",
    )

    concept_slug = require_string(
        raw.get("concept_slug"),
        f"{record_id} concept_slug",
    )

    if concept_slug not in EXPECTED_CONCEPTS:
        raise PrototypeMigrationError(f"{record_id} has unexpected concept {concept_slug!r}.")

    prototype_version = require_string(
        raw.get("prototype_version"),
        f"{record_id} prototype_version",
    )

    if prototype_version != PROTOTYPE_VERSION:
        raise PrototypeMigrationError(f"{record_id} prototype version changed.")

    prototype_role = require_string(
        raw.get("prototype_role"),
        f"{record_id} prototype_role",
    )

    record_type = require_string(
        raw.get("record_type"),
        f"{record_id} record_type",
    )

    if expected_source == "query":
        if prototype_role != "question" or record_type != "query_prototype":
            raise PrototypeMigrationError(
                f"{record_id} violates the frozen query-prototype role/type contract."
            )

        expected_task_type = "search_query"

    elif expected_source == "passage":
        if (
            prototype_role
            not in {
                "positive",
                "hard_negative",
            }
            or record_type != "passage_prototype"
        ):
            raise PrototypeMigrationError(
                f"{record_id} violates the frozen passage-prototype role/type contract."
            )

        expected_task_type = "search_document"

    else:
        raise PrototypeMigrationError(f"Unknown local prototype source {expected_source!r}.")

    provider = require_string(
        raw.get("provider"),
        f"{record_id} provider",
    )
    model = require_string(
        raw.get("model"),
        f"{record_id} model",
    )
    model_revision = require_string(
        raw.get("model_revision"),
        f"{record_id} model_revision",
    )
    dimensions = require_int(
        raw.get("dimensions"),
        f"{record_id} dimensions",
    )
    normalization = require_string(
        raw.get("normalization"),
        f"{record_id} normalization",
    )
    task_type = require_string(
        raw.get("task_type"),
        f"{record_id} task_type",
    )
    embedding_origin = require_string(
        raw.get("embedding_origin"),
        f"{record_id} embedding_origin",
    )

    identity = (
        provider,
        model,
        model_revision,
        dimensions,
        normalization,
        task_type,
        embedding_origin,
    )

    expected_identity = (
        EXPECTED_PROVIDER,
        EXPECTED_MODEL,
        EXPECTED_MODEL_REVISION,
        EXPECTED_DIMENSIONS,
        EXPECTED_NORMALIZATION,
        expected_task_type,
        EXPECTED_EMBEDDING_ORIGIN,
    )

    if identity != expected_identity:
        raise PrototypeMigrationError(f"{record_id} embedding identity changed.")

    chunk_id = optional_string(raw.get("chunk_id"))
    source_id = optional_string(raw.get("source_id"))
    domain = optional_string(raw.get("domain"))
    evaluation_split = optional_string(raw.get("evaluation_split"))

    if expected_source == "query":
        if any(
            value is not None
            for value in (
                chunk_id,
                source_id,
                domain,
                evaluation_split,
            )
        ):
            raise PrototypeMigrationError(
                f"{record_id} query prototype unexpectedly contains passage provenance."
            )

    else:
        if not all(
            value is not None
            for value in (
                chunk_id,
                source_id,
                domain,
                evaluation_split,
            )
        ):
            raise PrototypeMigrationError(
                f"{record_id} passage prototype is missing build provenance."
            )

        if domain not in {
            "science",
            "advaita",
            "samkhya",
        }:
            raise PrototypeMigrationError(f"{record_id} has invalid domain {domain!r}.")

        if evaluation_split != "build":
            raise PrototypeMigrationError(f"{record_id} is not from the frozen build split.")

    return FrozenPrototypeRow(
        record_id=record_id,
        concept_slug=concept_slug,
        prototype_role=prototype_role,
        record_type=record_type,
        embedding=parse_vector(
            raw,
            record_id=record_id,
        ),
        provider=provider,
        model=model,
        model_revision=model_revision,
        dimensions=dimensions,
        normalization=normalization,
        task_type=task_type,
        task_instruction=require_string(
            raw.get("task_instruction"),
            f"{record_id} task_instruction",
        ),
        text_checksum=require_sha256(
            raw.get("text_checksum"),
            f"{record_id} text_checksum",
        ),
        embedding_input_checksum=(
            require_sha256(
                raw.get("embedding_input_checksum"),
                (f"{record_id} embedding_input_checksum"),
            )
        ),
        embedding_checksum=(
            require_sha256(
                raw.get("embedding_checksum"),
                f"{record_id} embedding_checksum",
            )
        ),
        embedding_origin=embedding_origin,
        source_artifact_sha256=(source_artifact_sha256),
        chunk_id=chunk_id,
        source_id=source_id,
        domain=domain,
        evaluation_split=(evaluation_split),
        citation=optional_string(raw.get("citation")),
        title=optional_string(raw.get("title")),
        reviewed_labels=(
            parse_reviewed_labels(
                raw.get("reviewed_labels"),
                record_id=record_id,
            )
            if expected_source == "passage"
            else None
        ),
        estimated_tokens=(
            optional_int(
                raw.get("estimated_tokens"),
                f"{record_id} estimated_tokens",
            )
        ),
        actual_tokens=(
            optional_int(
                raw.get("actual_tokens"),
                f"{record_id} actual_tokens",
            )
        ),
        created_at=require_string(
            raw.get("created_at"),
            f"{record_id} created_at",
        ),
    )


def validate_local_prototypes(
    *,
    query_path: Path,
    passage_path: Path,
    embedding_manifest_path: Path,
    prototype_yaml_path: Path,
) -> LocalPrototypeSet:
    query_hash = sha256_jsonl_content(query_path)
    passage_hash = sha256_jsonl_content(passage_path)
    manifest_hash = sha256_file(embedding_manifest_path)
    yaml_hash = sha256_file(prototype_yaml_path)

    if query_hash != EXPECTED_QUERY_JSONL_SHA256:
        raise PrototypeMigrationError("Frozen query prototype JSONL semantic SHA-256 changed.")

    if passage_hash != EXPECTED_PASSAGE_JSONL_SHA256:
        raise PrototypeMigrationError("Frozen passage prototype JSONL semantic SHA-256 changed.")

    if manifest_hash != EXPECTED_EMBEDDING_MANIFEST_FILE_SHA256:
        raise PrototypeMigrationError("Frozen embedding manifest file SHA-256 changed.")

    if yaml_hash != EXPECTED_PROTOTYPE_YAML_FILE_SHA256:
        raise PrototypeMigrationError("Frozen concept prototype YAML SHA-256 changed.")

    validate_embedding_manifest(load_json(embedding_manifest_path))

    query_rows = load_jsonl(query_path)
    passage_rows = load_jsonl(passage_path)

    if len(query_rows) != EXPECTED_QUERY_COUNT:
        raise PrototypeMigrationError(
            f"Expected {EXPECTED_QUERY_COUNT} query prototypes; found {len(query_rows)}."
        )

    if len(passage_rows) != EXPECTED_PASSAGE_COUNT:
        raise PrototypeMigrationError(
            f"Expected {EXPECTED_PASSAGE_COUNT} passage prototypes; found {len(passage_rows)}."
        )

    parsed = [
        *(
            parse_local_row(
                row,
                source_artifact_sha256=(query_hash),
                expected_source="query",
            )
            for row in query_rows
        ),
        *(
            parse_local_row(
                row,
                source_artifact_sha256=(passage_hash),
                expected_source="passage",
            )
            for row in passage_rows
        ),
    ]

    if len(parsed) != EXPECTED_TOTAL_COUNT:
        raise PrototypeMigrationError("Frozen prototype total changed.")

    record_ids = [row.record_id for row in parsed]

    if len(set(record_ids)) != len(record_ids):
        raise PrototypeMigrationError("Frozen prototype record IDs are not unique.")

    role_counts = Counter(row.prototype_role for row in parsed)

    if dict(role_counts) != EXPECTED_ROLE_COUNTS:
        raise PrototypeMigrationError(
            f"Frozen prototype role distribution changed: {dict(role_counts)}."
        )

    concept_role_counts = Counter(
        (
            row.concept_slug,
            row.prototype_role,
        )
        for row in parsed
    )

    for concept in EXPECTED_CONCEPTS:
        for role, expected in EXPECTED_PER_CONCEPT_ROLE_COUNTS.items():
            observed = concept_role_counts[
                (
                    concept,
                    role,
                )
            ]

            if observed != expected:
                raise PrototypeMigrationError(
                    "Frozen prototype concept/role "
                    "distribution changed: "
                    f"{concept}/{role} "
                    f"expected {expected}, "
                    f"observed {observed}."
                )

    LOGGER.info(
        "Frozen local prototype validation: PASS (%d rows)",
        len(parsed),
    )

    return LocalPrototypeSet(
        rows=tuple(
            sorted(
                parsed,
                key=lambda row: row.record_id,
            )
        ),
        query_semantic_sha256=(query_hash),
        passage_semantic_sha256=(passage_hash),
        embedding_manifest_file_sha256=(manifest_hash),
        prototype_yaml_file_sha256=(yaml_hash),
    )


def execute_select(
    client: Any,
    table: str,
    columns: str,
) -> list[dict[str, Any]]:
    try:
        response = client.table(table).select(columns).execute()
    except Exception as exc:
        raise PrototypeMigrationError(f"Supabase select failed for {table}.") from exc

    return response_rows(
        response,
        f"select {table}",
    )


def validate_live_schema(
    client: Any,
) -> None:
    columns = (
        "record_id,concept_id,prototype_version,"
        "prototype_role,record_type,embedding,"
        "provider,model,model_revision,dimensions,"
        "normalization,task_type,task_instruction,"
        "text_checksum,embedding_input_checksum,"
        "embedding_checksum,embedding_origin,"
        "source_artifact_sha256,chunk_id,source_id,"
        "domain,evaluation_split,citation,title,"
        "reviewed_labels,estimated_tokens,"
        "actual_tokens,embedding_created_at,"
        "loaded_at,updated_at"
    )

    try:
        (client.table(TABLE_NAME).select(columns).limit(1).execute())
    except Exception as exc:
        raise PrototypeMigrationError(
            "Stage 3.5D migration 0006 is not fully applied; prototype table schema check failed."
        ) from exc

    LOGGER.info("Live Stage 3.5D schema contract: PASS")


def resolve_required_concepts(
    client: Any,
) -> dict[str, str]:
    rows = execute_select(
        client,
        "concepts",
        "id,slug,is_active",
    )

    by_slug = {
        require_string(
            row.get("slug"),
            "concept slug",
        ): row
        for row in rows
    }

    result: dict[
        str,
        str,
    ] = {}

    for slug in EXPECTED_CONCEPTS:
        row = by_slug.get(slug)

        if row is None:
            raise PrototypeMigrationError(f"Canonical concept is missing: {slug}.")

        concept_id = require_string(
            row.get("id"),
            f"{slug} concept id",
        )

        expected_id = PHASE1_CONCEPT_UUIDS[slug]

        if concept_id != expected_id:
            raise PrototypeMigrationError(
                f"Canonical concept UUID changed for {slug}: "
                f"expected {expected_id}, "
                f"observed {concept_id}."
            )

        if row.get("is_active") is not True:
            raise PrototypeMigrationError(f"Canonical concept is inactive: {slug}.")

        result[slug] = concept_id

    LOGGER.info("Required Phase 1 concepts resolved: 3/3")

    return result


def payload_for_row(
    row: FrozenPrototypeRow,
    *,
    concept_ids: Mapping[str, str],
) -> dict[str, object]:
    return {
        "record_id": row.record_id,
        "concept_id": (concept_ids[row.concept_slug]),
        "prototype_version": (PROTOTYPE_VERSION),
        "prototype_role": (row.prototype_role),
        "record_type": (row.record_type),
        "embedding": list(row.embedding),
        "provider": row.provider,
        "model": row.model,
        "model_revision": (row.model_revision),
        "dimensions": (row.dimensions),
        "normalization": (row.normalization),
        "task_type": (row.task_type),
        "task_instruction": (row.task_instruction),
        "text_checksum": (row.text_checksum),
        "embedding_input_checksum": (row.embedding_input_checksum),
        "embedding_checksum": (row.embedding_checksum),
        "embedding_origin": (row.embedding_origin),
        "source_artifact_sha256": (row.source_artifact_sha256),
        "chunk_id": row.chunk_id,
        "source_id": row.source_id,
        "domain": row.domain,
        "evaluation_split": (row.evaluation_split),
        "citation": row.citation,
        "title": row.title,
        "reviewed_labels": (row.reviewed_labels),
        "estimated_tokens": (row.estimated_tokens),
        "actual_tokens": (row.actual_tokens),
        "embedding_created_at": (row.created_at),
    }


def batched(
    values: Sequence[dict[str, object]],
    size: int,
) -> Sequence[Sequence[dict[str, object]]]:
    return [
        values[index : index + size]
        for index in range(
            0,
            len(values),
            size,
        )
    ]


def upsert_rows(
    client: Any,
    payloads: Sequence[dict[str, object]],
) -> None:
    for batch in batched(
        list(payloads),
        UPSERT_BATCH_SIZE,
    ):
        try:
            (
                client.table(TABLE_NAME)
                .upsert(
                    list(batch),
                    on_conflict="record_id",
                )
                .execute()
            )
        except Exception as exc:
            raise PrototypeMigrationError("Supabase prototype upsert failed.") from exc


def parse_db_vector(
    value: object,
    *,
    record_id: str,
) -> tuple[float, ...]:
    raw = value

    if isinstance(
        raw,
        str,
    ):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PrototypeMigrationError(f"DB vector for {record_id} is invalid JSON.") from exc

    values = require_sequence(
        raw,
        f"DB vector for {record_id}",
    )

    if len(values) != EXPECTED_DIMENSIONS:
        raise PrototypeMigrationError(f"DB vector for {record_id} has {len(values)} dimensions.")

    return tuple(
        require_float(
            value,
            f"DB vector {record_id}[{index}]",
        )
        for index, value in enumerate(values)
    )


def fetch_prototype_rows(
    client: Any,
) -> list[dict[str, Any]]:
    columns = (
        "record_id,concept_id,prototype_version,"
        "prototype_role,record_type,embedding,"
        "provider,model,model_revision,dimensions,"
        "normalization,task_type,task_instruction,"
        "text_checksum,embedding_input_checksum,"
        "embedding_checksum,embedding_origin,"
        "source_artifact_sha256,chunk_id,source_id,"
        "domain,evaluation_split,citation,title,"
        "reviewed_labels,estimated_tokens,"
        "actual_tokens,embedding_created_at"
    )

    try:
        response = (
            client.table(TABLE_NAME)
            .select(columns)
            .eq(
                "prototype_version",
                PROTOTYPE_VERSION,
            )
            .execute()
        )
    except Exception as exc:
        raise PrototypeMigrationError("Could not read back frozen prototype rows.") from exc

    return response_rows(
        response,
        "prototype reconciliation select",
    )


def database_fingerprint(
    *,
    rows: Sequence[Mapping[str, Any]],
    concept_slug_by_id: Mapping[str, str],
) -> str:
    normalized: list[dict[str, object]] = []

    for row in rows:
        concept_id = require_string(
            row.get("concept_id"),
            "DB concept_id",
        )

        concept_slug = concept_slug_by_id.get(concept_id)

        if concept_slug is None:
            raise PrototypeMigrationError(f"Unknown prototype concept UUID {concept_id}.")

        normalized.append(
            {
                "record_id": (
                    require_string(
                        row.get("record_id"),
                        "DB record_id",
                    )
                ),
                "concept_slug": (concept_slug),
                "prototype_version": (
                    require_string(
                        row.get("prototype_version"),
                        "DB prototype_version",
                    )
                ),
                "prototype_role": (
                    require_string(
                        row.get("prototype_role"),
                        "DB prototype_role",
                    )
                ),
                "embedding_checksum": (
                    require_sha256(
                        row.get("embedding_checksum"),
                        "DB embedding_checksum",
                    )
                ),
                "text_checksum": (
                    require_sha256(
                        row.get("text_checksum"),
                        "DB text_checksum",
                    )
                ),
                "source_artifact_sha256": (
                    require_sha256(
                        row.get("source_artifact_sha256"),
                        "DB source_artifact_sha256",
                    )
                ),
            }
        )

    normalized.sort(key=lambda item: str(item["record_id"]))

    return sha256_json_value(normalized)


def reconcile(
    *,
    client: Any,
    local: LocalPrototypeSet,
    concept_ids: Mapping[str, str],
) -> Reconciliation:
    db_rows = fetch_prototype_rows(client)

    if len(db_rows) != EXPECTED_TOTAL_COUNT:
        raise PrototypeMigrationError(
            "Supabase prototype count mismatch: "
            f"expected {EXPECTED_TOTAL_COUNT}, "
            f"observed {len(db_rows)}."
        )

    local_by_id = {row.record_id: row for row in local.rows}

    db_by_id: dict[
        str,
        dict[str, Any],
    ] = {}

    role_counts: Counter[str] = Counter()

    concept_role_counts: dict[
        str,
        Counter[str],
    ] = {concept: Counter() for concept in EXPECTED_CONCEPTS}

    slug_by_id = {concept_id: slug for slug, concept_id in concept_ids.items()}

    for db_row in db_rows:
        record_id = require_string(
            db_row.get("record_id"),
            "DB record_id",
        )

        if record_id in db_by_id:
            raise PrototypeMigrationError(f"Duplicate DB prototype record {record_id}.")

        db_by_id[record_id] = db_row

    if set(db_by_id) != set(local_by_id):
        missing = sorted(set(local_by_id) - set(db_by_id))
        extra = sorted(set(db_by_id) - set(local_by_id))

        raise PrototypeMigrationError(
            "Supabase prototype IDs differ from frozen "
            f"artifacts; missing={missing}, extra={extra}."
        )

    for record_id, local_row in local_by_id.items():
        db_row = db_by_id[record_id]

        concept_id = require_string(
            db_row.get("concept_id"),
            f"{record_id} DB concept_id",
        )

        expected_concept_id = concept_ids[local_row.concept_slug]

        if concept_id != expected_concept_id:
            raise PrototypeMigrationError(f"{record_id} concept UUID changed.")

        expected_scalars: dict[
            str,
            object,
        ] = {
            "prototype_version": (PROTOTYPE_VERSION),
            "prototype_role": (local_row.prototype_role),
            "record_type": (local_row.record_type),
            "provider": (local_row.provider),
            "model": local_row.model,
            "model_revision": (local_row.model_revision),
            "dimensions": (local_row.dimensions),
            "normalization": (local_row.normalization),
            "task_type": (local_row.task_type),
            "task_instruction": (local_row.task_instruction),
            "text_checksum": (local_row.text_checksum),
            "embedding_input_checksum": (local_row.embedding_input_checksum),
            "embedding_checksum": (local_row.embedding_checksum),
            "embedding_origin": (local_row.embedding_origin),
            "source_artifact_sha256": (local_row.source_artifact_sha256),
            "chunk_id": (local_row.chunk_id),
            "source_id": (local_row.source_id),
            "domain": (local_row.domain),
            "evaluation_split": (local_row.evaluation_split),
            "citation": (local_row.citation),
            "title": (local_row.title),
            "estimated_tokens": (local_row.estimated_tokens),
            "actual_tokens": (local_row.actual_tokens),
        }

        for key, expected in expected_scalars.items():
            observed = db_row.get(key)

            if observed != expected:
                raise PrototypeMigrationError(
                    f"{record_id} DB field {key!r} "
                    f"changed: expected {expected!r}, "
                    f"observed {observed!r}."
                )

        expected_embedding_created_at = parse_iso_datetime(
            local_row.created_at,
            (f"{record_id} local embedding_created_at"),
        )

        observed_embedding_created_at = parse_iso_datetime(
            db_row.get("embedding_created_at"),
            (f"{record_id} DB embedding_created_at"),
        )

        if observed_embedding_created_at != expected_embedding_created_at:
            raise PrototypeMigrationError(
                f"{record_id} embedding_created_at changed: "
                f"expected "
                f"{expected_embedding_created_at.isoformat()!r}, "
                f"observed "
                f"{observed_embedding_created_at.isoformat()!r}."
            )

        observed_labels = db_row.get("reviewed_labels")

        if observed_labels != local_row.reviewed_labels:
            raise PrototypeMigrationError(f"{record_id} reviewed_labels changed.")

        db_vector = parse_db_vector(
            db_row.get("embedding"),
            record_id=record_id,
        )

        if len(db_vector) != len(local_row.embedding):
            raise PrototypeMigrationError(f"{record_id} DB vector length changed.")

        for index, (
            observed,
            expected,
        ) in enumerate(
            zip(
                db_vector,
                local_row.embedding,
                strict=True,
            )
        ):
            if abs(observed - expected) > VECTOR_ABS_TOL:
                raise PrototypeMigrationError(
                    f"{record_id} DB vector differs at dimension {index}."
                )

        if embedding_checksum(db_vector) != local_row.embedding_checksum:
            # pgvector textual round-trip may have tiny numeric representation
            # differences; exact source checksum is already persisted. The
            # tolerance comparison above is authoritative for DB vector parity.
            # This condition is therefore informative rather than fatal.
            LOGGER.debug(
                "%s DB textual vector checksum differs "
                "after pgvector round-trip but remains "
                "within %.1e tolerance.",
                record_id,
                VECTOR_ABS_TOL,
            )

        role_counts[local_row.prototype_role] += 1

        concept_role_counts[local_row.concept_slug][local_row.prototype_role] += 1

    checks = {
        "63_rows_loaded": (len(db_rows) == EXPECTED_TOTAL_COUNT),
        "all_record_ids_match_frozen_artifacts": (set(db_by_id) == set(local_by_id)),
        "all_concept_ids_match_canonical_seeds": True,
        "all_embedding_checksums_preserved": True,
        "all_vectors_match_frozen_values_within_tolerance": True,
        "all_embedding_identity_fields_preserved": True,
        "all_source_artifact_hashes_preserved": True,
        "prototype_role_distribution_preserved": (dict(role_counts) == EXPECTED_ROLE_COUNTS),
    }

    if not all(checks.values()):
        raise PrototypeMigrationError("Prototype reconciliation did not pass.")

    normalized_concept_counts = {
        concept: {
            role: (concept_role_counts[concept][role])
            for role in (
                "question",
                "positive",
                "hard_negative",
            )
        }
        for concept in EXPECTED_CONCEPTS
    }

    return Reconciliation(
        row_count=len(db_rows),
        role_counts=dict(sorted(role_counts.items())),
        concept_role_counts=(normalized_concept_counts),
        semantic_fingerprint_sha256=(
            database_fingerprint(
                rows=db_rows,
                concept_slug_by_id=(slug_by_id),
            )
        ),
        checks=checks,
    )


def load_phase1_prototypes(
    *,
    project_root: Path,
    query_prototypes_path: Path,
    passage_prototypes_path: Path,
    embedding_manifest_path: Path,
    prototype_yaml_path: Path,
    output_path: Path,
    dry_run: bool,
    verify_rerun: bool,
    replace_manifest: bool,
) -> dict[str, object]:
    project_root = project_root.resolve()

    query_prototypes_path = resolve(
        project_root,
        query_prototypes_path,
    )
    passage_prototypes_path = resolve(
        project_root,
        passage_prototypes_path,
    )
    embedding_manifest_path = resolve(
        project_root,
        embedding_manifest_path,
    )
    prototype_yaml_path = resolve(
        project_root,
        prototype_yaml_path,
    )
    output_path = resolve(
        project_root,
        output_path,
    )

    for path in (
        query_prototypes_path,
        passage_prototypes_path,
        embedding_manifest_path,
        prototype_yaml_path,
    ):
        require_file(path)

    if output_path.exists() and not replace_manifest and not dry_run:
        raise PrototypeMigrationError(
            "Prototype migration manifest already exists. Use --replace-manifest to replace it."
        )

    local = validate_local_prototypes(
        query_path=query_prototypes_path,
        passage_path=passage_prototypes_path,
        embedding_manifest_path=(embedding_manifest_path),
        prototype_yaml_path=(prototype_yaml_path),
    )

    client = build_supabase_client(project_root)

    validate_live_schema(client)

    concept_ids = resolve_required_concepts(client)

    if dry_run:
        manifest: dict[
            str,
            object,
        ] = {
            "stage": STAGE,
            "status": ("dry_run_complete"),
            "script_version": (SCRIPT_VERSION),
            "generated_at": (utc_now()),
            "prototype_version": (PROTOTYPE_VERSION),
            "dry_run": True,
            "local_validation": {
                "row_count": (len(local.rows)),
                "query_semantic_sha256": (local.query_semantic_sha256),
                "passage_semantic_sha256": (local.passage_semantic_sha256),
                "embedding_manifest_file_sha256": (local.embedding_manifest_file_sha256),
                "prototype_yaml_file_sha256": (local.prototype_yaml_file_sha256),
            },
            "live_schema_validated": True,
            "canonical_concepts_resolved": (len(concept_ids)),
            "writes_performed": False,
        }

        LOGGER.info("Stage 3.5D dry run: PASS")

        return manifest

    payloads = [
        payload_for_row(
            row,
            concept_ids=concept_ids,
        )
        for row in local.rows
    ]

    upsert_rows(
        client,
        payloads,
    )

    first = reconcile(
        client=client,
        local=local,
        concept_ids=concept_ids,
    )

    rerun_verified = False

    if verify_rerun:
        upsert_rows(
            client,
            payloads,
        )

        second = reconcile(
            client=client,
            local=local,
            concept_ids=concept_ids,
        )

        if (
            second.row_count != first.row_count
            or second.role_counts != first.role_counts
            or second.concept_role_counts != first.concept_role_counts
            or second.semantic_fingerprint_sha256 != first.semantic_fingerprint_sha256
        ):
            raise PrototypeMigrationError(
                "Second prototype loader run changed counts or semantic fingerprint."
            )

        rerun_verified = True

        LOGGER.info("Second-run prototype idempotency proof: PASS")

    manifest = {
        "stage": STAGE,
        "status": STATUS_COMPLETE,
        "script_version": (SCRIPT_VERSION),
        "generated_at": utc_now(),
        "prototype_version": (PROTOTYPE_VERSION),
        "embedding_identity": {
            "provider": (EXPECTED_PROVIDER),
            "model": EXPECTED_MODEL,
            "model_revision": (EXPECTED_MODEL_REVISION),
            "dimensions": (EXPECTED_DIMENSIONS),
            "normalization": (EXPECTED_NORMALIZATION),
        },
        "frozen_inputs": {
            "query_prototype_embeddings": {
                "path": (query_prototypes_path.as_posix()),
                "record_count": (EXPECTED_QUERY_COUNT),
                "semantic_sha256": (local.query_semantic_sha256),
            },
            "passage_prototype_embeddings": {
                "path": (passage_prototypes_path.as_posix()),
                "record_count": (EXPECTED_PASSAGE_COUNT),
                "semantic_sha256": (local.passage_semantic_sha256),
            },
            "embedding_manifest": {
                "path": (embedding_manifest_path.as_posix()),
                "sha256_file_bytes": (local.embedding_manifest_file_sha256),
            },
            "concept_prototypes": {
                "path": (prototype_yaml_path.as_posix()),
                "sha256_file_bytes": (local.prototype_yaml_file_sha256),
            },
        },
        "database": {
            "table": TABLE_NAME,
            "row_count": (first.row_count),
            "role_counts": (first.role_counts),
            "concept_role_counts": (first.concept_role_counts),
            "semantic_fingerprint_sha256": (first.semantic_fingerprint_sha256),
        },
        "idempotency": {
            "verify_rerun_requested": (verify_rerun),
            "second_run_passed": (rerun_verified),
        },
        "exit_gate": {
            **first.checks,
            "exact_query_jsonl_hash_validated": True,
            "exact_passage_jsonl_hash_validated": True,
            "embedding_manifest_hash_validated": True,
            "prototype_yaml_hash_validated": True,
            "no_embedding_provider_calls": True,
            "no_vectors_regenerated": True,
            "runtime_local_jsonl_dependency_removed": True,
            "passed": True,
        },
        "next_step": (
            "Use apps/api/repositories/concept_repository.py "
            "to load and cache the frozen PrototypeBank from "
            "Supabase, then assemble the Phase 14 runtime "
            "retrieval service."
        ),
    }

    atomic_write_json(
        output_path,
        manifest,
    )

    LOGGER.info("Stage 3.5D prototype migration: COMPLETE")
    LOGGER.info(
        "Prototype rows: %d",
        first.row_count,
    )
    LOGGER.info(
        "Database semantic fingerprint: %s",
        first.semantic_fingerprint_sha256,
    )
    LOGGER.info(
        "Migration manifest: %s",
        output_path,
    )

    return manifest


def main() -> int:
    arguments = parse_arguments()

    configure_logging(arguments.log_level)

    try:
        load_phase1_prototypes(
            project_root=(arguments.project_root),
            query_prototypes_path=(arguments.query_prototypes),
            passage_prototypes_path=(arguments.passage_prototypes),
            embedding_manifest_path=(arguments.embedding_manifest),
            prototype_yaml_path=(arguments.prototype_yaml),
            output_path=(arguments.output),
            dry_run=arguments.dry_run,
            verify_rerun=(arguments.verify_rerun),
            replace_manifest=(arguments.replace_manifest),
        )
    except PrototypeMigrationError:
        LOGGER.exception("Stage 3.5D prototype migration failed")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
