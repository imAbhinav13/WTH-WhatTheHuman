from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types

LOGGER = logging.getLogger("wth.phase1.generate_phase1_embeddings")

SCRIPT_VERSION: Final = "1.0.0"
EMBEDDING_RUN_VERSION: Final = "phase1-embeddings-v1"

PROVIDER: Final = "Google Gemini API"
MODEL: Final = "gemini-embedding-2"
DIMENSIONS: Final = 768
NORMALIZATION: Final = "provider_auto_l2"
QUERY_TASK_TYPE: Final = "search_query"
DOCUMENT_TASK_TYPE: Final = "search_document"
QUERY_TEMPLATE: Final = "task: search result | query: {content}"
DOCUMENT_TEMPLATE: Final = "title: {title} | text: {content}"

DEFAULT_GOLD_CORPUS: Final = Path("artifacts/phase1/reviewed/phase1_reviewed_gold_corpus.jsonl")
DEFAULT_PROTOTYPES: Final = Path("data/concepts/phase1_concept_prototypes.yaml")
DEFAULT_SPLIT_MANIFEST: Final = Path("data/evaluation/phase1_split_manifest.json")
DEFAULT_BUILD: Final = Path("data/evaluation/phase1_build.jsonl")
DEFAULT_DEVELOPMENT: Final = Path("data/evaluation/phase1_development.jsonl")
DEFAULT_HELDOUT: Final = Path("data/evaluation/phase1_heldout.jsonl")
DEFAULT_OUTPUT_DIRECTORY: Final = Path("artifacts/phase1/embeddings")

APPROVED_OUTPUT_FILENAME: Final = "approved_chunk_embeddings.jsonl"
QUERY_PROTOTYPE_OUTPUT_FILENAME: Final = "query_prototype_embeddings.jsonl"
PASSAGE_PROTOTYPE_OUTPUT_FILENAME: Final = "passage_prototype_embeddings.jsonl"
MANIFEST_FILENAME: Final = "embedding_manifest.json"

CACHE_FILENAME: Final = "embedding_cache.jsonl"
CHECKPOINT_FILENAME: Final = "embedding_checkpoint.json"

EXPECTED_APPROVED_COUNT: Final = 318
EXPECTED_SPLIT_COUNTS: Final = {
    "build": 159,
    "development": 80,
    "heldout": 79,
}

MAX_BATCH_ITEMS: Final = 12
MAX_BATCH_ESTIMATED_TOKENS: Final = 18_000
TOKEN_ESTIMATE_CHARS_PER_TOKEN: Final = 3.0
MIN_REQUEST_INTERVAL_SECONDS: Final = 0.75
MAX_RETRIES: Final = 6
BACKOFF_BASE_SECONDS: Final = 2.0
BACKOFF_MAX_SECONDS: Final = 60.0
NORMALIZATION_MIN_NORM: Final = 0.95
NORMALIZATION_MAX_NORM: Final = 1.05
LARGE_INPUT_EXACT_COUNT_THRESHOLD: Final = 6_000


class EmbeddingRunError(RuntimeError):
    """Raised when Phase 9 cannot proceed safely."""


class RetryClass(StrEnum):
    RETRYABLE_RATE_LIMIT = "retryable_rate_limit"
    RETRYABLE_SERVER = "retryable_server"
    RETRYABLE_NETWORK = "retryable_network"
    FATAL_AUTH = "fatal_auth"
    FATAL_REQUEST = "fatal_request"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ModelIdentity:
    provider: str
    model: str
    model_revision: str
    dimensions: int
    normalization: str
    model_resource_name: str
    base_model_id: str
    input_token_limit: int

    def common_payload(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "model_revision": self.model_revision,
            "dimensions": self.dimensions,
            "normalization": self.normalization,
        }

    def fingerprint(self) -> str:
        return sha256_json_value(self.common_payload())


@dataclass(frozen=True)
class WorkItem:
    record_id: str
    output_kind: str
    raw_text: str
    title: str
    embedding_input: str
    task_type: str
    task_instruction: str
    text_checksum: str
    embedding_input_checksum: str
    estimated_tokens: int
    chunk_id: str = ""
    source_id: str = ""
    domain: str = ""
    evaluation_split: str = ""
    concept_slug: str = ""
    prototype_role: str = ""
    prototype_version: str = ""
    citation: str = ""
    labels: dict[str, str] = field(default_factory=dict)

    def cache_key(
        self,
        identity: ModelIdentity,
    ) -> str:
        return sha256_json_value(
            {
                **identity.common_payload(),
                "task_type": self.task_type,
                "embedding_input_checksum": (self.embedding_input_checksum),
            }
        )


@dataclass(frozen=True)
class CacheEntry:
    cache_key: str
    provider: str
    model: str
    model_revision: str
    dimensions: int
    task_type: str
    normalization: str
    text_checksum: str
    embedding_input_checksum: str
    embedding_checksum: str
    created_at: str
    embedding: tuple[float, ...]
    estimated_tokens: int
    actual_tokens: int | None
    source: str

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, object],
    ) -> CacheEntry:
        embedding_raw = raw.get("embedding")
        if not isinstance(
            embedding_raw,
            list,
        ):
            raise EmbeddingRunError("Cache entry embedding must be a list.")

        embedding: list[float] = []
        for value in embedding_raw:
            if not isinstance(
                value,
                int | float,
            ):
                raise EmbeddingRunError("Cache embedding contains a non-numeric value.")
            embedding.append(float(value))

        actual_tokens_raw = raw.get("actual_tokens")
        actual_tokens: int | None
        if actual_tokens_raw is None:
            actual_tokens = None
        elif isinstance(
            actual_tokens_raw,
            int,
        ):
            actual_tokens = actual_tokens_raw
        else:
            raise EmbeddingRunError("Cache actual_tokens must be an integer or null.")

        return cls(
            cache_key=require_string(
                raw.get("cache_key"),
                "cache cache_key",
            ),
            provider=require_string(
                raw.get("provider"),
                "cache provider",
            ),
            model=require_string(
                raw.get("model"),
                "cache model",
            ),
            model_revision=require_string(
                raw.get("model_revision"),
                "cache model_revision",
            ),
            dimensions=require_int(
                raw.get("dimensions"),
                "cache dimensions",
            ),
            task_type=require_string(
                raw.get("task_type"),
                "cache task_type",
            ),
            normalization=require_string(
                raw.get("normalization"),
                "cache normalization",
            ),
            text_checksum=require_string(
                raw.get("text_checksum"),
                "cache text_checksum",
            ),
            embedding_input_checksum=(
                require_string(
                    raw.get("embedding_input_checksum"),
                    "cache embedding_input_checksum",
                )
            ),
            embedding_checksum=(
                require_string(
                    raw.get("embedding_checksum"),
                    "cache embedding_checksum",
                )
            ),
            created_at=require_string(
                raw.get("created_at"),
                "cache created_at",
            ),
            embedding=tuple(embedding),
            estimated_tokens=require_int(
                raw.get("estimated_tokens"),
                "cache estimated_tokens",
            ),
            actual_tokens=actual_tokens,
            source=require_string(
                raw.get("source"),
                "cache source",
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "cache_key": self.cache_key,
            "provider": self.provider,
            "model": self.model,
            "model_revision": (self.model_revision),
            "dimensions": self.dimensions,
            "task_type": self.task_type,
            "normalization": (self.normalization),
            "text_checksum": (self.text_checksum),
            "embedding_input_checksum": (self.embedding_input_checksum),
            "embedding_checksum": (self.embedding_checksum),
            "created_at": self.created_at,
            "embedding": list(self.embedding),
            "estimated_tokens": (self.estimated_tokens),
            "actual_tokens": (self.actual_tokens),
            "source": self.source,
        }


@dataclass
class RunStats:
    work_items_total: int = 0
    unique_embedding_inputs: int = 0
    cache_hits: int = 0
    provider_vectors_generated: int = 0
    provider_requests: int = 0
    retry_count: int = 0
    rate_limit_retries: int = 0
    server_retries: int = 0
    network_retries: int = 0
    token_count_requests: int = 0
    exact_token_counts: int = 0
    estimated_token_counts: int = 0
    batches_completed: int = 0
    started_monotonic: float = field(default_factory=time.monotonic)


@dataclass
class RequestPacer:
    minimum_interval_seconds: float
    last_request_started: float | None = None

    def wait(self) -> None:
        if self.last_request_started is None:
            self.last_request_started = time.monotonic()
            return

        elapsed = time.monotonic() - self.last_request_started
        remaining = self.minimum_interval_seconds - elapsed

        if remaining > 0:
            time.sleep(remaining)

        self.last_request_started = time.monotonic()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the real, resumable Phase 1 Gemini embedding "
            "artifacts for approved chunks and reviewed prototypes."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--gold-corpus",
        type=Path,
        default=DEFAULT_GOLD_CORPUS,
    )
    parser.add_argument(
        "--prototypes",
        type=Path,
        default=DEFAULT_PROTOTYPES,
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST,
    )
    parser.add_argument(
        "--build",
        type=Path,
        default=DEFAULT_BUILD,
    )
    parser.add_argument(
        "--development",
        type=Path,
        default=DEFAULT_DEVELOPMENT,
    )
    parser.add_argument(
        "--heldout",
        type=Path,
        default=DEFAULT_HELDOUT,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--replace-final-outputs",
        action="store_true",
        help=("Replace final JSONL/manifest outputs. The checksum cache is retained and reused."),
    )
    parser.add_argument(
        "--reset-cache",
        action="store_true",
        help=(
            "Delete the Phase 9 embedding cache/checkpoint before running. "
            "Use only when intentionally starting a new embedding identity."
        ),
    )
    parser.add_argument(
        "--max-batch-items",
        type=int,
        default=MAX_BATCH_ITEMS,
    )
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=MAX_BATCH_ESTIMATED_TOKENS,
    )
    parser.add_argument(
        "--request-interval-seconds",
        type=float,
        default=MIN_REQUEST_INTERVAL_SECONDS,
    )
    parser.add_argument(
        "--log-level",
        choices=(
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
        ),
        default="INFO",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format=("%(asctime)s %(levelname)s %(name)s %(message)s"),
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
        raise EmbeddingRunError(f"Required file does not exist: {path}")


def require_mapping(
    value: object,
    description: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise EmbeddingRunError(f"{description} must be an object.")

    result: dict[str, object] = {}
    for raw_key, nested_value in value.items():
        if not isinstance(
            raw_key,
            str,
        ):
            raise EmbeddingRunError(f"{description} contains a non-string key.")
        result[raw_key] = nested_value

    return result


def require_list(
    value: object,
    description: str,
) -> list[object]:
    if not isinstance(value, list):
        raise EmbeddingRunError(f"{description} must be a list.")
    return value


def require_string(
    value: object,
    description: str,
) -> str:
    if (
        not isinstance(
            value,
            str,
        )
        or not value.strip()
    ):
        raise EmbeddingRunError(f"{description} must be a non-empty string.")
    return value.strip()


def optional_string(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def require_int(
    value: object,
    description: str,
) -> int:
    if not isinstance(value, int):
        raise EmbeddingRunError(f"{description} must be an integer.")
    return value


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


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
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)

    return digest.hexdigest()


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
                raise EmbeddingRunError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc

            digest.update(canonical_json_bytes(value))
            digest.update(b"\n")

    return digest.hexdigest()


def embedding_checksum(
    values: Sequence[float],
) -> str:
    return sha256_json_value([float(value) for value in values])


def load_json_mapping(
    path: Path,
) -> dict[str, object]:
    try:
        loaded: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EmbeddingRunError(f"Invalid JSON in {path}: {exc}") from exc

    return require_mapping(
        loaded,
        f"JSON document {path}",
    )


def load_yaml_mapping(
    path: Path,
) -> dict[str, object]:
    try:
        loaded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise EmbeddingRunError(f"Invalid YAML in {path}: {exc}") from exc

    return require_mapping(
        loaded,
        f"YAML document {path}",
    )


def iter_jsonl_mappings(
    path: Path,
) -> Iterable[dict[str, object]]:
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
                loaded: object = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise EmbeddingRunError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc

            yield require_mapping(
                loaded,
                f"record at {path}:{line_number}",
            )


def atomic_write_json(
    path: Path,
    value: Mapping[str, object],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
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


def atomic_write_jsonl(
    path: Path,
    records: Iterable[Mapping[str, object]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = path.with_suffix(f"{path.suffix}.tmp")

    with temporary.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            handle.write("\n")

    temporary.replace(path)


def append_cache_entries(
    path: Path,
    entries: Sequence[CacheEntry],
) -> None:
    if not entries:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        for entry in entries:
            handle.write(
                json.dumps(
                    entry.to_mapping(),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            handle.write("\n")

        handle.flush()
        os.fsync(handle.fileno())


def estimate_tokens(text: str) -> int:
    return max(
        1,
        math.ceil(len(text) / TOKEN_ESTIMATE_CHARS_PER_TOKEN),
    )


def format_document(
    *,
    title: str,
    text: str,
) -> str:
    resolved_title = title.strip() if title.strip() else "none"
    return DOCUMENT_TEMPLATE.format(
        title=resolved_title,
        content=text.strip(),
    )


def format_query(text: str) -> str:
    return QUERY_TEMPLATE.format(content=text.strip())


def extract_labels(
    raw: Mapping[str, object],
) -> dict[str, str]:
    review_raw = raw.get("review")
    if not isinstance(
        review_raw,
        Mapping,
    ):
        return {}

    review = require_mapping(
        review_raw,
        "review",
    )
    labels_raw = review.get("labels")
    if not isinstance(
        labels_raw,
        Mapping,
    ):
        return {}

    labels_mapping = require_mapping(
        labels_raw,
        "review.labels",
    )

    labels: dict[str, str] = {}
    for key, value in labels_mapping.items():
        if isinstance(value, str):
            labels[key] = value.strip()

    return labels


def record_title(
    raw: Mapping[str, object],
) -> str:
    for key in (
        "source_title",
        "section_title",
        "title",
    ):
        candidate = optional_string(raw.get(key))
        if candidate:
            return candidate

    return optional_string(raw.get("source_id"))


def load_approved_corpus(
    path: Path,
) -> tuple[
    list[dict[str, object]],
    dict[str, dict[str, object]],
]:
    records = list(iter_jsonl_mappings(path))

    if len(records) != EXPECTED_APPROVED_COUNT:
        raise EmbeddingRunError(
            "Approved corpus count mismatch: "
            f"expected {EXPECTED_APPROVED_COUNT}, "
            f"found {len(records)}."
        )

    by_id: dict[
        str,
        dict[str, object],
    ] = {}

    for raw in records:
        chunk_id = require_string(
            raw.get("chunk_id"),
            "approved chunk_id",
        )
        if chunk_id in by_id:
            raise EmbeddingRunError(f"Duplicate approved chunk_id: {chunk_id}")
        by_id[chunk_id] = raw

    return records, by_id


def validate_split_manifest_and_load_map(
    *,
    manifest_path: Path,
    split_paths: Mapping[str, Path],
    approved_ids: set[str],
) -> tuple[
    dict[str, str],
    dict[str, object],
]:
    manifest = load_json_mapping(manifest_path)

    if manifest.get("frozen") is not True:
        raise EmbeddingRunError("Phase 6 split manifest is not frozen.")

    if optional_string(manifest.get("status")) != "frozen":
        raise EmbeddingRunError("Phase 6 split manifest status is not 'frozen'.")

    outputs = require_mapping(
        manifest.get("outputs"),
        "split manifest outputs",
    )

    chunk_to_split: dict[str, str] = {}
    split_diagnostics: dict[
        str,
        object,
    ] = {}

    for split in (
        "build",
        "development",
        "heldout",
    ):
        path = split_paths[split]
        require_file(path)

        output_meta = require_mapping(
            outputs.get(split),
            f"split manifest outputs.{split}",
        )

        expected_count = EXPECTED_SPLIT_COUNTS[split]
        manifest_count = require_int(
            output_meta.get("record_count"),
            f"{split} record_count",
        )

        if manifest_count != expected_count:
            raise EmbeddingRunError(
                f"{split} manifest count mismatch: {manifest_count} != {expected_count}."
            )

        expected_sha = require_string(
            output_meta.get("sha256"),
            f"{split} sha256",
        )
        algorithm = optional_string(output_meta.get("sha256_algorithm"))

        if algorithm and (algorithm != "sha256-canonical-jsonl-v1"):
            raise EmbeddingRunError(f"Unsupported {split} checksum algorithm: {algorithm!r}.")

        actual_sha = sha256_jsonl_content(path)
        if actual_sha != expected_sha:
            raise EmbeddingRunError(f"{split} checksum does not match the frozen Phase 6 manifest.")

        split_records = list(iter_jsonl_mappings(path))
        if len(split_records) != expected_count:
            raise EmbeddingRunError(f"{split} actual count mismatch.")

        for raw in split_records:
            chunk_id = require_string(
                raw.get("chunk_id"),
                f"{split} chunk_id",
            )

            if optional_string(raw.get("evaluation_split")) != split:
                raise EmbeddingRunError(f"{chunk_id} has incorrect evaluation_split.")

            if chunk_id in (chunk_to_split):
                raise EmbeddingRunError(f"{chunk_id} occurs in multiple evaluation splits.")

            chunk_to_split[chunk_id] = split

        split_diagnostics[split] = {
            "path": path.as_posix(),
            "record_count": (len(split_records)),
            "sha256": actual_sha,
            "read_only": (output_meta.get("read_only")),
        }

    if set(chunk_to_split) != approved_ids:
        missing = sorted(approved_ids - set(chunk_to_split))
        unexpected = sorted(set(chunk_to_split) - approved_ids)
        raise EmbeddingRunError(
            "Evaluation splits do not exactly partition the approved corpus. "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    heldout_meta = require_mapping(
        outputs.get("heldout"),
        "heldout output metadata",
    )
    if heldout_meta.get("read_only") is not True:
        raise EmbeddingRunError("Held-out split is not marked read-only.")

    return (
        chunk_to_split,
        split_diagnostics,
    )


def validate_frozen_prototypes(
    path: Path,
) -> dict[str, object]:
    prototypes = load_yaml_mapping(path)

    if optional_string(prototypes.get("status")) != "frozen":
        raise EmbeddingRunError(
            "Phase 8 concept prototypes are not frozen. "
            "Complete human review and set top-level status='frozen' "
            "before Phase 9."
        )

    exit_gate = require_mapping(
        prototypes.get("exit_gate"),
        "prototype exit_gate",
    )

    required_true = (
        "all_concepts_have_question_prototypes",
        "all_concepts_have_positive_passage_prototypes",
        "all_concepts_have_hard_negative_prototypes",
        "all_passage_prototypes_from_build",
        "domain_diversity_checked",
        "human_review_complete",
    )

    for field_name in required_true:
        if exit_gate.get(field_name) is not True:
            raise EmbeddingRunError(f"Phase 8 exit gate is incomplete: {field_name}=false.")

    concepts = require_list(
        prototypes.get("concepts"),
        "prototype concepts",
    )

    if len(concepts) != 3:
        raise EmbeddingRunError(f"Expected 3 Phase 1 concepts, found {len(concepts)}.")

    for concept_raw in concepts:
        concept = require_mapping(
            concept_raw,
            "prototype concept",
        )

        if optional_string(concept.get("review_status")) != "reviewed":
            raise EmbeddingRunError("Every concept prototype must have review_status='reviewed'.")

        human_review = require_mapping(
            concept.get("human_review"),
            "prototype human_review",
        )

        for review_field in (
            "question_prototypes_reviewed",
            "positive_passages_reviewed",
            "hard_negatives_reviewed",
        ):
            if human_review.get(review_field) is not True:
                raise EmbeddingRunError(f"Prototype human review incomplete: {review_field}=false.")

    return prototypes


def make_approved_items(
    records: Sequence[Mapping[str, object]],
    *,
    chunk_to_split: Mapping[
        str,
        str,
    ],
) -> list[WorkItem]:
    items: list[WorkItem] = []

    for raw in records:
        chunk_id = require_string(
            raw.get("chunk_id"),
            "approved chunk_id",
        )
        reviewed_text = require_string(
            raw.get("reviewed_text"),
            f"{chunk_id} reviewed_text",
        )
        title = record_title(raw)
        embedding_input = format_document(
            title=title,
            text=reviewed_text,
        )

        items.append(
            WorkItem(
                record_id=(f"approved:{chunk_id}"),
                output_kind=("approved_chunk"),
                raw_text=reviewed_text,
                title=title,
                embedding_input=(embedding_input),
                task_type=(DOCUMENT_TASK_TYPE),
                task_instruction=(DOCUMENT_TEMPLATE),
                text_checksum=(sha256_text(reviewed_text)),
                embedding_input_checksum=(sha256_text(embedding_input)),
                estimated_tokens=(estimate_tokens(embedding_input)),
                chunk_id=chunk_id,
                source_id=optional_string(raw.get("source_id")),
                domain=optional_string(raw.get("domain")),
                evaluation_split=(chunk_to_split[chunk_id]),
                citation=optional_string(raw.get("citation")),
                labels=extract_labels(raw),
            )
        )

    return items


def make_query_prototype_items(
    prototypes: Mapping[str, object],
) -> list[WorkItem]:
    prototype_version = require_string(
        prototypes.get("prototype_version"),
        "prototype_version",
    )
    concepts = require_list(
        prototypes.get("concepts"),
        "prototype concepts",
    )

    items: list[WorkItem] = []

    for concept_raw in concepts:
        concept = require_mapping(
            concept_raw,
            "prototype concept",
        )
        concept_slug = require_string(
            concept.get("concept_slug"),
            "concept_slug",
        )
        questions = require_list(
            concept.get("question_examples"),
            f"{concept_slug} question_examples",
        )

        for index, question_raw in enumerate(
            questions,
            start=1,
        ):
            question = require_string(
                question_raw,
                f"{concept_slug} question {index}",
            )
            embedding_input = format_query(question)
            question_sha = sha256_text(question)

            items.append(
                WorkItem(
                    record_id=(f"query:{concept_slug}:{index:02d}:{question_sha[:12]}"),
                    output_kind=("query_prototype"),
                    raw_text=question,
                    title="",
                    embedding_input=(embedding_input),
                    task_type=(QUERY_TASK_TYPE),
                    task_instruction=(QUERY_TEMPLATE),
                    text_checksum=(question_sha),
                    embedding_input_checksum=(sha256_text(embedding_input)),
                    estimated_tokens=(estimate_tokens(embedding_input)),
                    concept_slug=(concept_slug),
                    prototype_role=("question"),
                    prototype_version=(prototype_version),
                )
            )

    return items


def make_passage_prototype_items(
    prototypes: Mapping[str, object],
    *,
    approved_by_id: Mapping[
        str,
        Mapping[str, object],
    ],
    chunk_to_split: Mapping[
        str,
        str,
    ],
) -> list[WorkItem]:
    prototype_version = require_string(
        prototypes.get("prototype_version"),
        "prototype_version",
    )
    concepts = require_list(
        prototypes.get("concepts"),
        "prototype concepts",
    )

    items: list[WorkItem] = []
    seen_role_ids: set[str] = set()

    role_fields = (
        (
            "positive",
            "positive_passage_ids",
        ),
        (
            "hard_negative",
            "hard_negative_passage_ids",
        ),
    )

    for concept_raw in concepts:
        concept = require_mapping(
            concept_raw,
            "prototype concept",
        )
        concept_slug = require_string(
            concept.get("concept_slug"),
            "concept_slug",
        )

        for (
            prototype_role,
            field_name,
        ) in role_fields:
            passage_ids = require_list(
                concept.get(field_name),
                f"{concept_slug} {field_name}",
            )

            for passage_id_raw in passage_ids:
                chunk_id = require_string(
                    passage_id_raw,
                    (f"{concept_slug} {field_name} chunk_id"),
                )

                if chunk_id not in approved_by_id:
                    raise EmbeddingRunError(
                        f"Prototype passage is not in approved corpus: {chunk_id}"
                    )

                role_id = f"passage:{concept_slug}:{prototype_role}:{chunk_id}"

                if role_id in seen_role_ids:
                    raise EmbeddingRunError(f"Duplicate prototype role: {role_id}")
                seen_role_ids.add(role_id)

                raw = approved_by_id[chunk_id]
                reviewed_text = require_string(
                    raw.get("reviewed_text"),
                    f"{chunk_id} reviewed_text",
                )
                title = record_title(raw)
                embedding_input = format_document(
                    title=title,
                    text=reviewed_text,
                )

                items.append(
                    WorkItem(
                        record_id=role_id,
                        output_kind=("passage_prototype"),
                        raw_text=(reviewed_text),
                        title=title,
                        embedding_input=(embedding_input),
                        task_type=(DOCUMENT_TASK_TYPE),
                        task_instruction=(DOCUMENT_TEMPLATE),
                        text_checksum=(sha256_text(reviewed_text)),
                        embedding_input_checksum=(sha256_text(embedding_input)),
                        estimated_tokens=(estimate_tokens(embedding_input)),
                        chunk_id=chunk_id,
                        source_id=optional_string(raw.get("source_id")),
                        domain=optional_string(raw.get("domain")),
                        evaluation_split=(chunk_to_split[chunk_id]),
                        concept_slug=(concept_slug),
                        prototype_role=(prototype_role),
                        prototype_version=(prototype_version),
                        citation=optional_string(raw.get("citation")),
                        labels=extract_labels(raw),
                    )
                )

    return items


def get_api_key() -> str:
    key = (
        os.environ.get(
            "GOOGLE_API_KEY",
            "",
        ).strip()
        or os.environ.get(
            "GEMINI_API_KEY",
            "",
        ).strip()
    )

    if not key:
        raise EmbeddingRunError("Missing GOOGLE_API_KEY or GEMINI_API_KEY.")

    return key


def model_supported_actions(
    model_info: Any,
) -> tuple[str, ...]:
    raw = getattr(
        model_info,
        "supported_actions",
        None,
    )

    if raw is None:
        raw = getattr(
            model_info,
            "supported_generation_methods",
            None,
        )

    if not isinstance(
        raw,
        list | tuple,
    ):
        return ()

    return tuple(str(action) for action in raw)


def validate_model_metadata(
    client: Any,
) -> ModelIdentity:
    LOGGER.info(
        "Validating Gemini model metadata for %s",
        MODEL,
    )

    model_info = client.models.get(model=MODEL)

    model_resource_name = optional_string(
        getattr(
            model_info,
            "name",
            "",
        )
    )
    base_model_id = optional_string(
        getattr(
            model_info,
            "base_model_id",
            "",
        )
    )
    model_revision = optional_string(
        getattr(
            model_info,
            "version",
            "",
        )
    )
    input_token_limit_raw = getattr(
        model_info,
        "input_token_limit",
        None,
    )

    if not model_revision:
        raise EmbeddingRunError(
            "Gemini models.get() did not return a model revision. "
            "Phase 9 requires exact model_revision provenance."
        )

    if not isinstance(
        input_token_limit_raw,
        int,
    ):
        raise EmbeddingRunError("Gemini models.get() did not return input_token_limit.")

    actions = model_supported_actions(model_info)
    if actions and not any(action.casefold() == "embedcontent" for action in actions):
        raise EmbeddingRunError(f"{MODEL} does not advertise embedContent support.")

    if model_resource_name and MODEL not in model_resource_name and base_model_id != MODEL:
        raise EmbeddingRunError(
            "Model metadata identity mismatch: "
            f"name={model_resource_name!r}, "
            f"base_model_id={base_model_id!r}."
        )

    if input_token_limit_raw < 8_192:
        raise EmbeddingRunError(
            f"Unexpected Gemini Embedding 2 token limit: {input_token_limit_raw}."
        )

    identity = ModelIdentity(
        provider=PROVIDER,
        model=MODEL,
        model_revision=(model_revision),
        dimensions=DIMENSIONS,
        normalization=(NORMALIZATION),
        model_resource_name=(model_resource_name),
        base_model_id=(base_model_id),
        input_token_limit=(input_token_limit_raw),
    )

    LOGGER.info(
        "Model metadata validated: revision=%s input_token_limit=%d",
        identity.model_revision,
        identity.input_token_limit,
    )

    return identity


def vector_norm(
    values: Sequence[float],
) -> float:
    return math.sqrt(sum(value * value for value in values))


def validate_embedding_vector(
    values: Sequence[float],
) -> tuple[float, ...]:
    if len(values) != DIMENSIONS:
        raise EmbeddingRunError(f"Embedding dimension mismatch: {len(values)} != {DIMENSIONS}.")

    normalized_values: list[float] = []

    for value in values:
        number = float(value)
        if not math.isfinite(number):
            raise EmbeddingRunError("Embedding contains NaN or infinity.")
        normalized_values.append(number)

    if not any(value != 0.0 for value in normalized_values):
        raise EmbeddingRunError("Embedding is an all-zero vector.")

    norm = vector_norm(normalized_values)
    if not (NORMALIZATION_MIN_NORM <= norm <= NORMALIZATION_MAX_NORM):
        raise EmbeddingRunError(
            f"Gemini 768-dimensional embedding is not approximately L2-normalized: norm={norm:.6f}."
        )

    return tuple(normalized_values)


def validate_cache_entry(
    entry: CacheEntry,
) -> None:
    values = validate_embedding_vector(entry.embedding)

    if embedding_checksum(values) != entry.embedding_checksum:
        raise EmbeddingRunError(f"Cache embedding checksum mismatch for {entry.cache_key}.")

    if entry.source != "provider":
        raise EmbeddingRunError(
            "Cache contains a non-provider embedding source. "
            "Mock or synthetic vectors are not allowed."
        )


def load_cache(
    path: Path,
) -> dict[str, CacheEntry]:
    if not path.exists():
        return {}

    cache: dict[
        str,
        CacheEntry,
    ] = {}

    for raw in iter_jsonl_mappings(path):
        entry = CacheEntry.from_mapping(raw)
        validate_cache_entry(entry)

        existing = cache.get(entry.cache_key)
        if existing is not None and existing.embedding_checksum != entry.embedding_checksum:
            raise EmbeddingRunError(f"Cache contains conflicting embeddings for {entry.cache_key}.")

        cache[entry.cache_key] = entry

    return cache


def validate_cached_identity(
    entry: CacheEntry,
    *,
    item: WorkItem,
    identity: ModelIdentity,
) -> bool:
    return (
        entry.provider == identity.provider
        and entry.model == identity.model
        and entry.model_revision == identity.model_revision
        and entry.dimensions == identity.dimensions
        and entry.normalization == identity.normalization
        and entry.task_type == item.task_type
        and entry.embedding_input_checksum == item.embedding_input_checksum
    )


def extract_status_code(
    exc: BaseException,
) -> int | None:
    for attribute in (
        "status_code",
        "code",
    ):
        candidate = getattr(
            exc,
            attribute,
            None,
        )
        if isinstance(candidate, int):
            return candidate

    response = getattr(
        exc,
        "response",
        None,
    )
    if response is not None:
        candidate = getattr(
            response,
            "status_code",
            None,
        )
        if isinstance(candidate, int):
            return candidate

    return None


def classify_exception(
    exc: BaseException,
) -> RetryClass:
    """Classify provider/SDK failures for retry handling."""

    # Local SDK/configuration errors are deterministic.
    # Retrying them will never help.
    if isinstance(exc, ValueError):
        return RetryClass.FATAL_REQUEST

    status_code = extract_status_code(exc)
    message = str(exc).casefold()

    if status_code in {
        401,
        403,
    }:
        return RetryClass.FATAL_AUTH

    if status_code in {
        400,
        404,
        409,
        422,
    }:
        return RetryClass.FATAL_REQUEST

    if status_code == 429:
        return RetryClass.RETRYABLE_RATE_LIMIT

    if status_code in {
        408,
        425,
        500,
        502,
        503,
        504,
    }:
        return RetryClass.RETRYABLE_SERVER

    rate_markers = (
        "resource_exhausted",
        "rate limit",
        "rate-limit",
        "quota exceeded",
        "quota",
        "too many requests",
    )

    if any(marker in message for marker in rate_markers):
        return RetryClass.RETRYABLE_RATE_LIMIT

    network_markers = (
        "timeout",
        "timed out",
        "connection",
        "connecterror",
        "network",
        "temporarily unavailable",
        "reset by peer",
        "connection reset",
        "connection refused",
    )

    if any(marker in message for marker in network_markers):
        return RetryClass.RETRYABLE_NETWORK

    server_markers = (
        "internal server error",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
    )

    if any(marker in message for marker in server_markers):
        return RetryClass.RETRYABLE_SERVER

    return RetryClass.UNKNOWN


def retry_delay_seconds(
    *,
    attempt: int,
    retry_class: RetryClass,
) -> float:
    base = min(
        BACKOFF_MAX_SECONDS,
        BACKOFF_BASE_SECONDS * (2 ** max(0, attempt - 1)),
    )

    if retry_class == RetryClass.RETRYABLE_RATE_LIMIT:
        base = max(
            base,
            5.0,
        )

    jitter = random.uniform(
        0.0,
        min(
            2.0,
            base * 0.25,
        ),
    )
    return float(
        min(
            BACKOFF_MAX_SECONDS,
            base + jitter,
        )
    )


def exact_token_count_if_needed(
    *,
    client: Any,
    item: WorkItem,
    identity: ModelIdentity,
    pacer: RequestPacer,
    stats: RunStats,
) -> int | None:
    if item.estimated_tokens < LARGE_INPUT_EXACT_COUNT_THRESHOLD:
        stats.estimated_token_counts += 1
        return None

    try:
        pacer.wait()
        stats.provider_requests += 1
        stats.token_count_requests += 1

        response = client.models.count_tokens(
            model=identity.model,
            contents=item.embedding_input,
        )
    except Exception as exc:
        LOGGER.warning(
            "Exact token count unavailable for %s; using conservative estimate. Error: %s",
            item.record_id,
            exc,
        )
        stats.estimated_token_counts += 1
        return None
    else:
        total_tokens = getattr(
            response,
            "total_tokens",
            None,
        )
        if not isinstance(
            total_tokens,
            int,
        ):
            raise EmbeddingRunError("count_tokens did not return total_tokens.")

        stats.exact_token_counts += 1
        return total_tokens


def enforce_item_token_limit(
    *,
    item: WorkItem,
    actual_tokens: int | None,
    identity: ModelIdentity,
) -> None:
    safe_limit = min(
        7_800,
        math.floor(identity.input_token_limit * 0.95),
    )

    effective_tokens = actual_tokens if actual_tokens is not None else item.estimated_tokens

    if effective_tokens > safe_limit:
        raise EmbeddingRunError(
            "Embedding input is too large and auto-truncation is disabled: "
            f"{item.record_id} tokens={effective_tokens} "
            f"safe_limit={safe_limit}."
        )


def make_batches(
    items: Sequence[WorkItem],
    *,
    max_items: int,
    max_tokens: int,
) -> list[list[WorkItem]]:
    if max_items <= 0:
        raise EmbeddingRunError("--max-batch-items must be > 0.")
    if max_tokens <= 0:
        raise EmbeddingRunError("--max-batch-tokens must be > 0.")

    batches: list[list[WorkItem]] = []
    current: list[WorkItem] = []
    current_tokens = 0

    for item in items:
        would_exceed_items = len(current) >= max_items
        would_exceed_tokens = current and (current_tokens + item.estimated_tokens > max_tokens)

        if would_exceed_items or would_exceed_tokens:
            batches.append(current)
            current = []
            current_tokens = 0

        current.append(item)
        current_tokens += item.estimated_tokens

    if current:
        batches.append(current)

    return batches


def call_embedding_batch_once(
    *,
    client: Any,
    batch: Sequence[WorkItem],
    identity: ModelIdentity,
) -> list[tuple[float, ...]]:
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=item.embedding_input)],
        )
        for item in batch
    ]

    response = client.models.embed_content(
        model=identity.model,
        contents=contents,
        config=types.EmbedContentConfig(
            output_dimensionality=(identity.dimensions),
        ),
    )

    embeddings_raw = getattr(
        response,
        "embeddings",
        None,
    )
    if not isinstance(
        embeddings_raw,
        list,
    ):
        raise EmbeddingRunError("Gemini embed_content returned no embeddings list.")

    if len(embeddings_raw) != len(batch):
        raise EmbeddingRunError(
            "Gemini returned a different number of embeddings "
            f"than requested: {len(embeddings_raw)} != {len(batch)}. "
            "Inputs were explicitly wrapped as separate Content objects."
        )

    vectors: list[tuple[float, ...]] = []

    for embedding_raw in embeddings_raw:
        values_raw = getattr(
            embedding_raw,
            "values",
            None,
        )
        if not isinstance(
            values_raw,
            list,
        ):
            raise EmbeddingRunError("Gemini embedding object has no values list.")

        vectors.append(validate_embedding_vector(values_raw))

    return vectors


def call_embedding_batch_with_retry(
    *,
    client: Any,
    batch: Sequence[WorkItem],
    identity: ModelIdentity,
    pacer: RequestPacer,
    stats: RunStats,
) -> list[tuple[float, ...]]:
    for attempt in range(
        1,
        MAX_RETRIES + 2,
    ):
        try:
            pacer.wait()
            stats.provider_requests += 1

            return call_embedding_batch_once(
                client=client,
                batch=batch,
                identity=identity,
            )

        except EmbeddingRunError:
            raise

        except Exception as exc:
            retry_class = classify_exception(exc)

            if retry_class == RetryClass.FATAL_AUTH:
                raise EmbeddingRunError("Gemini authentication/authorization failed.") from exc

            if retry_class == RetryClass.FATAL_REQUEST:
                raise EmbeddingRunError(
                    f"Gemini rejected the embedding request as invalid: {exc}"
                ) from exc

            if attempt > MAX_RETRIES:
                raise EmbeddingRunError(
                    f"Gemini embedding batch exhausted retries. class={retry_class}, error={exc}"
                ) from exc

            stats.retry_count += 1

            if retry_class == RetryClass.RETRYABLE_RATE_LIMIT:
                stats.rate_limit_retries += 1
            elif retry_class == RetryClass.RETRYABLE_SERVER:
                stats.server_retries += 1
            elif retry_class == RetryClass.RETRYABLE_NETWORK:
                stats.network_retries += 1

            delay = retry_delay_seconds(
                attempt=attempt,
                retry_class=retry_class,
            )

            LOGGER.warning(
                "Embedding batch retry %d/%d class=%s delay=%.2fs error=%s",
                attempt,
                MAX_RETRIES,
                retry_class,
                delay,
                exc,
            )
            time.sleep(delay)

    raise AssertionError("unreachable")


def build_checkpoint_payload(
    *,
    status: str,
    identity: ModelIdentity,
    stats: RunStats,
    cache: Mapping[str, CacheEntry],
    last_batch_number: int,
) -> dict[str, object]:
    return {
        "checkpoint_version": "1.0",
        "embedding_run_version": (EMBEDDING_RUN_VERSION),
        "status": status,
        "updated_at": utc_now(),
        "identity": {
            **identity.common_payload(),
            "identity_fingerprint": (identity.fingerprint()),
        },
        "cache_entries": len(cache),
        "last_batch_number": (last_batch_number),
        "statistics": {
            "work_items_total": (stats.work_items_total),
            "unique_embedding_inputs": (stats.unique_embedding_inputs),
            "cache_hits": (stats.cache_hits),
            "provider_vectors_generated": (stats.provider_vectors_generated),
            "provider_requests": (stats.provider_requests),
            "retry_count": (stats.retry_count),
            "rate_limit_retries": (stats.rate_limit_retries),
            "server_retries": (stats.server_retries),
            "network_retries": (stats.network_retries),
            "batches_completed": (stats.batches_completed),
        },
    }


def prepare_unique_missing_items(
    *,
    work_items: Sequence[WorkItem],
    identity: ModelIdentity,
    cache: Mapping[str, CacheEntry],
    stats: RunStats,
) -> list[WorkItem]:
    unique_by_cache_key: dict[
        str,
        WorkItem,
    ] = {}

    for item in work_items:
        key = item.cache_key(identity)
        cached = cache.get(key)

        if cached is not None:
            if not validate_cached_identity(
                cached,
                item=item,
                identity=identity,
            ):
                raise EmbeddingRunError(
                    f"Cache key collision or identity mismatch for {item.record_id}."
                )
            stats.cache_hits += 1
            continue

        unique_by_cache_key.setdefault(
            key,
            item,
        )

    stats.unique_embedding_inputs = len({item.cache_key(identity) for item in work_items})

    return list(unique_by_cache_key.values())


def create_cache_entries_for_batch(
    *,
    batch: Sequence[WorkItem],
    vectors: Sequence[tuple[float, ...]],
    identity: ModelIdentity,
    actual_token_counts: Mapping[
        str,
        int | None,
    ],
) -> list[CacheEntry]:
    if len(batch) != len(vectors):
        raise EmbeddingRunError("Internal batch/vector count mismatch.")

    created_at = utc_now()
    entries: list[CacheEntry] = []

    for item, vector in zip(
        batch,
        vectors,
        strict=True,
    ):
        entries.append(
            CacheEntry(
                cache_key=item.cache_key(identity),
                provider=(identity.provider),
                model=identity.model,
                model_revision=(identity.model_revision),
                dimensions=(identity.dimensions),
                task_type=(item.task_type),
                normalization=(identity.normalization),
                text_checksum=(item.text_checksum),
                embedding_input_checksum=(item.embedding_input_checksum),
                embedding_checksum=(embedding_checksum(vector)),
                created_at=created_at,
                embedding=vector,
                estimated_tokens=(item.estimated_tokens),
                actual_tokens=(actual_token_counts.get(item.cache_key(identity))),
                source="provider",
            )
        )

    return entries


def embedding_output_record(
    *,
    item: WorkItem,
    entry: CacheEntry,
    identity: ModelIdentity,
) -> dict[str, object]:
    if not validate_cached_identity(
        entry,
        item=item,
        identity=identity,
    ):
        raise EmbeddingRunError(f"Output identity mismatch for {item.record_id}.")

    validate_cache_entry(entry)

    result: dict[str, object] = {
        "record_id": item.record_id,
        "record_type": item.output_kind,
        "provider": entry.provider,
        "model": entry.model,
        "model_revision": (entry.model_revision),
        "dimensions": entry.dimensions,
        "task_type": entry.task_type,
        "task_instruction": (item.task_instruction),
        "normalization": (entry.normalization),
        "text_checksum": (entry.text_checksum),
        "embedding_input_checksum": (entry.embedding_input_checksum),
        "embedding_checksum": (entry.embedding_checksum),
        "created_at": entry.created_at,
        "embedding_origin": (entry.source),
        "embedding": list(entry.embedding),
        "estimated_tokens": (entry.estimated_tokens),
        "actual_tokens": (entry.actual_tokens),
    }

    optional_fields: dict[
        str,
        object,
    ] = {
        "chunk_id": item.chunk_id,
        "source_id": item.source_id,
        "domain": item.domain,
        "evaluation_split": (item.evaluation_split),
        "concept_slug": (item.concept_slug),
        "prototype_role": (item.prototype_role),
        "prototype_version": (item.prototype_version),
        "citation": item.citation,
        "title": item.title,
    }

    for key, value in optional_fields.items():
        if value != "":
            result[key] = value

    if item.labels:
        result["reviewed_labels"] = item.labels

    return result


def records_for_items(
    *,
    items: Sequence[WorkItem],
    cache: Mapping[str, CacheEntry],
    identity: ModelIdentity,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    for item in items:
        key = item.cache_key(identity)
        entry = cache.get(key)

        if entry is None:
            raise EmbeddingRunError(f"Missing final embedding for {item.record_id}.")

        records.append(
            embedding_output_record(
                item=item,
                entry=entry,
                identity=identity,
            )
        )

    return records


def validate_final_records(
    *,
    approved_records: Sequence[Mapping[str, object]],
    query_records: Sequence[Mapping[str, object]],
    passage_records: Sequence[Mapping[str, object]],
    identity: ModelIdentity,
) -> dict[str, object]:
    if len(approved_records) != EXPECTED_APPROVED_COUNT:
        raise EmbeddingRunError("Approved embedding count does not equal 318.")

    approved_chunk_ids = {
        require_string(
            record.get("chunk_id"),
            "approved output chunk_id",
        )
        for record in approved_records
    }
    if len(approved_chunk_ids) != EXPECTED_APPROVED_COUNT:
        raise EmbeddingRunError("Approved embedding output contains duplicate chunk IDs.")

    all_records = [
        *approved_records,
        *query_records,
        *passage_records,
    ]

    identities: set[
        tuple[
            str,
            str,
            str,
            int,
            str,
        ]
    ] = set()

    for record in all_records:
        vector_raw = record.get("embedding")
        if not isinstance(
            vector_raw,
            list,
        ):
            raise EmbeddingRunError("Final embedding record has no vector.")

        vector = validate_embedding_vector([float(value) for value in vector_raw])

        expected_checksum = require_string(
            record.get("embedding_checksum"),
            "final embedding_checksum",
        )
        if embedding_checksum(vector) != expected_checksum:
            raise EmbeddingRunError("Final embedding checksum validation failed.")

        if optional_string(record.get("embedding_origin")) != "provider":
            raise EmbeddingRunError("Final output contains a non-provider vector.")

        identities.add(
            (
                require_string(
                    record.get("provider"),
                    "provider",
                ),
                require_string(
                    record.get("model"),
                    "model",
                ),
                require_string(
                    record.get("model_revision"),
                    "model_revision",
                ),
                require_int(
                    record.get("dimensions"),
                    "dimensions",
                ),
                require_string(
                    record.get("normalization"),
                    "normalization",
                ),
            )
        )

    expected_identity = (
        identity.provider,
        identity.model,
        identity.model_revision,
        identity.dimensions,
        identity.normalization,
    )

    if identities != {expected_identity}:
        raise EmbeddingRunError("Final outputs contain mixed embedding identities.")

    query_task_types = {
        require_string(
            record.get("task_type"),
            "query task_type",
        )
        for record in query_records
    }
    document_task_types = {
        require_string(
            record.get("task_type"),
            "document task_type",
        )
        for record in [
            *approved_records,
            *passage_records,
        ]
    }

    if query_task_types != {QUERY_TASK_TYPE}:
        raise EmbeddingRunError("Query prototype task types are inconsistent.")

    if document_task_types != {DOCUMENT_TASK_TYPE}:
        raise EmbeddingRunError("Document/passage prototype task types are inconsistent.")

    split_counts: dict[str, int] = {
        "build": 0,
        "development": 0,
        "heldout": 0,
    }
    for record in approved_records:
        split = require_string(
            record.get("evaluation_split"),
            "evaluation_split",
        )
        if split not in split_counts:
            raise EmbeddingRunError(f"Unexpected evaluation split {split!r}.")
        split_counts[split] += 1

    if split_counts != EXPECTED_SPLIT_COUNTS:
        raise EmbeddingRunError(
            f"Approved embeddings do not preserve the frozen 159/80/79 split: {split_counts}."
        )

    return {
        "approved_chunks": (len(approved_records)),
        "query_prototypes": (len(query_records)),
        "passage_prototype_roles": (len(passage_records)),
        "evaluation_split_counts": (split_counts),
        "embedding_identity_count": (len(identities)),
        "mock_vector_count": 0,
    }


def prepare_output_paths(
    output_directory: Path,
) -> dict[str, Path]:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return {
        "approved": (output_directory / APPROVED_OUTPUT_FILENAME),
        "query": (output_directory / QUERY_PROTOTYPE_OUTPUT_FILENAME),
        "passage": (output_directory / PASSAGE_PROTOTYPE_OUTPUT_FILENAME),
        "manifest": (output_directory / MANIFEST_FILENAME),
        "cache": (output_directory / CACHE_FILENAME),
        "checkpoint": (output_directory / CHECKPOINT_FILENAME),
    }


def reset_cache_if_requested(
    paths: Mapping[str, Path],
    *,
    reset_cache: bool,
) -> None:
    if not reset_cache:
        return

    for key in (
        "cache",
        "checkpoint",
    ):
        path = paths[key]
        if path.exists():
            path.unlink()

    LOGGER.warning("Phase 9 cache/checkpoint reset by explicit request.")


def ensure_final_output_policy(
    paths: Mapping[str, Path],
    *,
    replace: bool,
) -> None:
    final_keys = (
        "approved",
        "query",
        "passage",
        "manifest",
    )

    existing = [paths[key] for key in final_keys if paths[key].exists()]

    if existing and not replace:
        raise EmbeddingRunError(
            "Final Phase 9 outputs already exist. "
            "Use --replace-final-outputs to regenerate them "
            "from the existing checksum cache: " + ", ".join(path.as_posix() for path in existing)
        )


def phase9_generate_embeddings(
    *,
    project_root: Path,
    gold_corpus_path: Path,
    prototypes_path: Path,
    split_manifest_path: Path,
    build_path: Path,
    development_path: Path,
    heldout_path: Path,
    output_directory: Path,
    replace_final_outputs: bool,
    reset_cache: bool,
    max_batch_items: int,
    max_batch_tokens: int,
    request_interval_seconds: float,
) -> dict[str, object]:
    project_root = project_root.resolve()

    gold_corpus_path = resolve_from_project(
        project_root,
        gold_corpus_path,
    )
    prototypes_path = resolve_from_project(
        project_root,
        prototypes_path,
    )
    split_manifest_path = resolve_from_project(
        project_root,
        split_manifest_path,
    )
    build_path = resolve_from_project(
        project_root,
        build_path,
    )
    development_path = resolve_from_project(
        project_root,
        development_path,
    )
    heldout_path = resolve_from_project(
        project_root,
        heldout_path,
    )
    output_directory = resolve_from_project(
        project_root,
        output_directory,
    )

    for path in (
        gold_corpus_path,
        prototypes_path,
        split_manifest_path,
        build_path,
        development_path,
        heldout_path,
    ):
        require_file(path)

    paths = prepare_output_paths(output_directory)
    reset_cache_if_requested(
        paths,
        reset_cache=reset_cache,
    )
    ensure_final_output_policy(
        paths,
        replace=(replace_final_outputs),
    )

    prototypes = validate_frozen_prototypes(prototypes_path)

    approved_records_raw, approved_by_id = load_approved_corpus(gold_corpus_path)

    chunk_to_split, split_diagnostics = validate_split_manifest_and_load_map(
        manifest_path=(split_manifest_path),
        split_paths={
            "build": build_path,
            "development": (development_path),
            "heldout": heldout_path,
        },
        approved_ids=set(approved_by_id),
    )

    approved_items = make_approved_items(
        approved_records_raw,
        chunk_to_split=(chunk_to_split),
    )
    query_items = make_query_prototype_items(prototypes)
    passage_items = make_passage_prototype_items(
        prototypes,
        approved_by_id=(approved_by_id),
        chunk_to_split=(chunk_to_split),
    )

    all_items = [
        *approved_items,
        *query_items,
        *passage_items,
    ]

    record_ids = [item.record_id for item in all_items]
    if len(record_ids) != len(set(record_ids)):
        raise EmbeddingRunError("Phase 9 work contains duplicate record IDs.")

    load_dotenv()
    api_key = get_api_key()
    client = genai.Client(api_key=api_key)

    identity = validate_model_metadata(client)

    stats = RunStats(work_items_total=len(all_items))

    cache = load_cache(paths["cache"])

    missing_items = prepare_unique_missing_items(
        work_items=all_items,
        identity=identity,
        cache=cache,
        stats=stats,
    )

    LOGGER.info(
        "Phase 9 work items=%d unique_inputs=%d cache_hits=%d missing_unique_inputs=%d",
        len(all_items),
        stats.unique_embedding_inputs,
        stats.cache_hits,
        len(missing_items),
    )

    pacer = RequestPacer(minimum_interval_seconds=(request_interval_seconds))

    actual_token_counts: dict[
        str,
        int | None,
    ] = {}

    for item in missing_items:
        key = item.cache_key(identity)
        actual_tokens = exact_token_count_if_needed(
            client=client,
            item=item,
            identity=identity,
            pacer=pacer,
            stats=stats,
        )
        enforce_item_token_limit(
            item=item,
            actual_tokens=(actual_tokens),
            identity=identity,
        )
        actual_token_counts[key] = actual_tokens

    batches = make_batches(
        missing_items,
        max_items=max_batch_items,
        max_tokens=max_batch_tokens,
    )

    atomic_write_json(
        paths["checkpoint"],
        build_checkpoint_payload(
            status="running",
            identity=identity,
            stats=stats,
            cache=cache,
            last_batch_number=0,
        ),
    )

    for batch_number, batch in enumerate(
        batches,
        start=1,
    ):
        LOGGER.info(
            "Embedding batch %d/%d items=%d estimated_tokens=%d",
            batch_number,
            len(batches),
            len(batch),
            sum(item.estimated_tokens for item in batch),
        )

        vectors = call_embedding_batch_with_retry(
            client=client,
            batch=batch,
            identity=identity,
            pacer=pacer,
            stats=stats,
        )

        new_entries = create_cache_entries_for_batch(
            batch=batch,
            vectors=vectors,
            identity=identity,
            actual_token_counts=(actual_token_counts),
        )

        append_cache_entries(
            paths["cache"],
            new_entries,
        )

        for entry in new_entries:
            cache[entry.cache_key] = entry

        stats.provider_vectors_generated += len(new_entries)
        stats.batches_completed += 1

        atomic_write_json(
            paths["checkpoint"],
            build_checkpoint_payload(
                status="running",
                identity=identity,
                stats=stats,
                cache=cache,
                last_batch_number=(batch_number),
            ),
        )

    approved_output_records = records_for_items(
        items=approved_items,
        cache=cache,
        identity=identity,
    )
    query_output_records = records_for_items(
        items=query_items,
        cache=cache,
        identity=identity,
    )
    passage_output_records = records_for_items(
        items=passage_items,
        cache=cache,
        identity=identity,
    )

    final_validation = validate_final_records(
        approved_records=(approved_output_records),
        query_records=(query_output_records),
        passage_records=(passage_output_records),
        identity=identity,
    )

    atomic_write_jsonl(
        paths["approved"],
        approved_output_records,
    )
    atomic_write_jsonl(
        paths["query"],
        query_output_records,
    )
    atomic_write_jsonl(
        paths["passage"],
        passage_output_records,
    )

    elapsed_seconds = time.monotonic() - stats.started_monotonic

    manifest: dict[str, object] = {
        "script_version": (SCRIPT_VERSION),
        "embedding_run_version": (EMBEDDING_RUN_VERSION),
        "generated_at": utc_now(),
        "status": "complete",
        "embedding_identity": {
            **identity.common_payload(),
            "model_resource_name": (identity.model_resource_name),
            "base_model_id": (identity.base_model_id),
            "input_token_limit": (identity.input_token_limit),
            "identity_fingerprint": (identity.fingerprint()),
            "query_task_type": (QUERY_TASK_TYPE),
            "query_instruction": (QUERY_TEMPLATE),
            "document_task_type": (DOCUMENT_TASK_TYPE),
            "document_instruction": (DOCUMENT_TEMPLATE),
        },
        "inputs": {
            "approved_corpus": {
                "path": (gold_corpus_path.as_posix()),
                "sha256": (sha256_jsonl_content(gold_corpus_path)),
                "record_count": (len(approved_records_raw)),
            },
            "concept_prototypes": {
                "path": (prototypes_path.as_posix()),
                "sha256_file_bytes": (sha256_file(prototypes_path)),
                "prototype_version": (
                    require_string(
                        prototypes.get("prototype_version"),
                        "prototype_version",
                    )
                ),
                "status": (optional_string(prototypes.get("status"))),
            },
            "split_manifest": {
                "path": (split_manifest_path.as_posix()),
                "sha256_file_bytes": (sha256_file(split_manifest_path)),
            },
            "evaluation_splits": (split_diagnostics),
        },
        "architecture": {
            "resumable": True,
            "content_checksum_cache": True,
            "cache_key_includes_model_identity": True,
            "token_aware_batching": True,
            "token_strategy": {
                "default": (
                    "conservative character estimate "
                    f"at {TOKEN_ESTIMATE_CHARS_PER_TOKEN} chars/token"
                ),
                "exact_count_threshold_estimated_tokens": (LARGE_INPUT_EXACT_COUNT_THRESHOLD),
                "auto_truncate": False,
            },
            "per_batch_checkpointing": True,
            "retry_classification": [retry_class.value for retry_class in RetryClass],
            "exponential_backoff_with_jitter": True,
            "quota_aware_pacing_seconds": (request_interval_seconds),
            "model_metadata_validated": True,
            "max_batch_items": (max_batch_items),
            "max_batch_estimated_tokens": (max_batch_tokens),
        },
        "statistics": {
            "work_items_total": (stats.work_items_total),
            "unique_embedding_inputs": (stats.unique_embedding_inputs),
            "cache_hits": (stats.cache_hits),
            "provider_vectors_generated": (stats.provider_vectors_generated),
            "provider_requests": (stats.provider_requests),
            "retry_count": (stats.retry_count),
            "rate_limit_retries": (stats.rate_limit_retries),
            "server_retries": (stats.server_retries),
            "network_retries": (stats.network_retries),
            "token_count_requests": (stats.token_count_requests),
            "exact_token_counts": (stats.exact_token_counts),
            "estimated_token_counts": (stats.estimated_token_counts),
            "batches_completed": (stats.batches_completed),
            "elapsed_seconds": (elapsed_seconds),
        },
        "validation": (final_validation),
        "outputs": {
            "approved_chunk_embeddings": {
                "path": (paths["approved"].as_posix()),
                "sha256": (sha256_jsonl_content(paths["approved"])),
                "record_count": (len(approved_output_records)),
            },
            "query_prototype_embeddings": {
                "path": (paths["query"].as_posix()),
                "sha256": (sha256_jsonl_content(paths["query"])),
                "record_count": (len(query_output_records)),
            },
            "passage_prototype_embeddings": {
                "path": (paths["passage"].as_posix()),
                "sha256": (sha256_jsonl_content(paths["passage"])),
                "record_count": (len(passage_output_records)),
            },
            "cache": {
                "path": (paths["cache"].as_posix()),
                "entry_count": (len(cache)),
                "purpose": ("resumable content-checksum embedding cache"),
            },
            "checkpoint": {
                "path": (paths["checkpoint"].as_posix()),
                "purpose": ("per-batch resumability checkpoint"),
            },
        },
        "exit_gate": {
            "every_approved_chunk_has_one_valid_embedding": True,
            "approved_embedding_count": (len(approved_output_records)),
            "no_mock_vectors": True,
            "single_embedding_identity": True,
            "all_prototype_vectors_same_frozen_configuration": True,
            "evaluation_splits_embedded_in_same_space": True,
            "resume_without_repeated_provider_calls": True,
        },
        "next_step": (
            "Phase 10: tune concept mapping on Development data only. "
            "Held-out embeddings may exist but must remain untouched for tuning."
        ),
    }

    atomic_write_json(
        paths["manifest"],
        manifest,
    )

    atomic_write_json(
        paths["checkpoint"],
        build_checkpoint_payload(
            status="complete",
            identity=identity,
            stats=stats,
            cache=cache,
            last_batch_number=(stats.batches_completed),
        ),
    )

    LOGGER.info("Phase 9 embedding generation complete")
    LOGGER.info(
        "Approved chunks: %d",
        len(approved_output_records),
    )
    LOGGER.info(
        "Query prototypes: %d",
        len(query_output_records),
    )
    LOGGER.info(
        "Passage prototype roles: %d",
        len(passage_output_records),
    )
    LOGGER.info(
        "Provider vectors generated this run: %d",
        stats.provider_vectors_generated,
    )
    LOGGER.info(
        "Cache hits this run: %d",
        stats.cache_hits,
    )
    LOGGER.info(
        "Provider requests: %d retries: %d",
        stats.provider_requests,
        stats.retry_count,
    )
    LOGGER.info(
        "Model revision: %s",
        identity.model_revision,
    )
    LOGGER.info(
        "Manifest: %s",
        paths["manifest"],
    )

    return manifest


def main() -> int:
    arguments = parse_arguments()
    configure_logging(arguments.log_level)

    try:
        phase9_generate_embeddings(
            project_root=(arguments.project_root),
            gold_corpus_path=(arguments.gold_corpus),
            prototypes_path=(arguments.prototypes),
            split_manifest_path=(arguments.split_manifest),
            build_path=(arguments.build),
            development_path=(arguments.development),
            heldout_path=(arguments.heldout),
            output_directory=(arguments.output_directory),
            replace_final_outputs=(arguments.replace_final_outputs),
            reset_cache=(arguments.reset_cache),
            max_batch_items=(arguments.max_batch_items),
            max_batch_tokens=(arguments.max_batch_tokens),
            request_interval_seconds=(arguments.request_interval_seconds),
        )
    except EmbeddingRunError:
        LOGGER.exception("Phase 9 embedding generation failed")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
