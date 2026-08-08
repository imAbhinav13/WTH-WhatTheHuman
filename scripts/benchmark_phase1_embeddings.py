from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import logging
import math
import os
import statistics
import time
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Protocol, cast

LOGGER = logging.getLogger("wth.phase1.benchmark_phase1_embeddings")
BENCHMARK_VERSION: Final = "1.0.0"

DEFAULT_BUILD: Final = Path("data/evaluation/phase1_build.jsonl")
DEFAULT_DEVELOPMENT: Final = Path("data/evaluation/phase1_development.jsonl")
DEFAULT_HELDOUT: Final = Path("data/evaluation/phase1_heldout.jsonl")
DEFAULT_SPLIT_MANIFEST: Final = Path("data/evaluation/phase1_split_manifest.json")
DEFAULT_OUTPUT_DIR: Final = Path("artifacts/phase1/evaluation")
DEFAULT_CACHE_DIR: Final = Path("artifacts/phase1/embedding_benchmark/cache")
RESULTS_FILENAME: Final = "embedding_benchmark_results.json"
QUERY_RESULTS_FILENAME: Final = "embedding_benchmark_query_results.csv"

GEMINI_MODEL: Final = "gemini-embedding-2"
GEMINI_DIMENSIONS: Final = 768
GEMINI_ENDPOINT: Final = (
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent"
)
LOCAL_MODEL: Final = "intfloat/multilingual-e5-base"
LOCAL_DIMENSIONS: Final = 768
COHERE_MODEL: Final = "embed-v4.0"
COHERE_DIMENSIONS: Final = 1024
COHERE_ENDPOINT: Final = "https://api.cohere.com/v2/embed"

CONCEPTS: Final = ("consciousness", "self_identity", "reality_appearance")
VALID_LABELS: Final = {"positive", "partial", "negative", "uncertain"}


class BenchmarkError(RuntimeError):
    """Raised when the Phase 7 benchmark cannot complete safely."""


@dataclass(frozen=True)
class BenchmarkQuery:
    query_id: str
    text: str
    target_concept: str
    target_domain: str | None
    focus: str
    kind: str


QUERY_SUITE: Final[tuple[BenchmarkQuery, ...]] = (
    BenchmarkQuery(
        "consciousness_general_01",
        "What makes an experience conscious rather than merely processed?",
        "consciousness",
        None,
        "consciousness_vs_cognition",
        "general",
    ),
    BenchmarkQuery(
        "consciousness_general_02",
        "Is awareness distinct from thought, intellect, or cognition?",
        "consciousness",
        None,
        "consciousness_vs_cognition",
        "general",
    ),
    BenchmarkQuery(
        "consciousness_science",
        "How does science explain conscious experience and bodily self-consciousness?",
        "consciousness",
        "science",
        "awareness_vs_processing",
        "domain",
    ),
    BenchmarkQuery(
        "consciousness_advaita",
        "How does Advaita describe witnessing awareness or consciousness?",
        "consciousness",
        "advaita",
        "witness_vs_mind",
        "domain",
    ),
    BenchmarkQuery(
        "consciousness_samkhya",
        "How does Samkhya distinguish conscious Purusha from unconscious Prakriti or buddhi?",
        "consciousness",
        "samkhya",
        "purusha_vs_prakriti",
        "domain",
    ),
    BenchmarkQuery(
        "self_general_01",
        "What is the self, and what makes subjective identity persist?",
        "self_identity",
        None,
        "self_vs_identity",
        "general",
    ),
    BenchmarkQuery(
        "self_general_02",
        "Is the ego the same thing as the true self or conscious subject?",
        "self_identity",
        None,
        "self_vs_ego",
        "general",
    ),
    BenchmarkQuery(
        "self_science",
        "What do bodily ownership and first-person perspective reveal about the scientific self-model?",
        "self_identity",
        "science",
        "self_model_vs_personality",
        "domain",
    ),
    BenchmarkQuery(
        "self_advaita",
        "How does Advaita distinguish Atman from ego, body, and mind?",
        "self_identity",
        "advaita",
        "atman_vs_ego",
        "domain",
    ),
    BenchmarkQuery(
        "self_samkhya",
        "How does Samkhya distinguish Purusha from ahamkara, buddhi, and individuality?",
        "self_identity",
        "samkhya",
        "purusha_vs_ego",
        "domain",
    ),
    BenchmarkQuery(
        "reality_general_01",
        "What is the relation between reality, appearance, and perception?",
        "reality_appearance",
        None,
        "reality_vs_description",
        "general",
    ),
    BenchmarkQuery(
        "reality_general_02",
        "When is an experienced world an appearance, illusion, or representation rather than reality itself?",
        "reality_appearance",
        None,
        "appearance_vs_reality",
        "general",
    ),
    BenchmarkQuery(
        "reality_science",
        "How can perceptual construction or illusion affect what a person experiences as real?",
        "reality_appearance",
        "science",
        "perception_vs_reality",
        "domain",
    ),
    BenchmarkQuery(
        "reality_advaita",
        "How does Advaita explain Maya, superimposition, and appearance versus ultimate reality?",
        "reality_appearance",
        "advaita",
        "maya_vs_reality",
        "domain",
    ),
    BenchmarkQuery(
        "reality_samkhya",
        "How does Samkhya understand the manifest world and the distinction between seer and seen?",
        "reality_appearance",
        "samkhya",
        "seer_vs_seen",
        "domain",
    ),
    BenchmarkQuery(
        "hardneg_cognition",
        "If the intellect reasons and remembers, does that alone make it conscious?",
        "consciousness",
        None,
        "consciousness_vs_cognition",
        "hard_negative",
    ),
    BenchmarkQuery(
        "hardneg_ego",
        "Does having an ego or individuality establish the existence of the true Self?",
        "self_identity",
        None,
        "self_vs_ego",
        "hard_negative",
    ),
    BenchmarkQuery(
        "hardneg_cosmology",
        "Does listing cosmic elements or stages of creation by itself explain reality versus appearance?",
        "reality_appearance",
        None,
        "reality_appearance_vs_cosmology",
        "hard_negative",
    ),
    BenchmarkQuery(
        "hardneg_atman_purusha",
        "Are Advaita Atman and Samkhya Purusha equivalent concepts of self?",
        "self_identity",
        None,
        "advaita_atman_vs_samkhya_purusha",
        "hard_negative",
    ),
    BenchmarkQuery(
        "adjacent_consciousness_self",
        "How is the conscious subject related to the sense of self without treating consciousness and identity as identical?",
        "consciousness",
        None,
        "consciousness_vs_self",
        "adjacent",
    ),
    BenchmarkQuery(
        "adjacent_self_reality",
        "How can misidentification of the self shape what appears to be real?",
        "self_identity",
        None,
        "self_vs_reality_appearance",
        "adjacent",
    ),
    BenchmarkQuery(
        "adjacent_reality_consciousness",
        "Does the fact that something appears in consciousness determine whether it is ultimately real?",
        "reality_appearance",
        None,
        "reality_appearance_vs_consciousness",
        "adjacent",
    ),
)


@dataclass(frozen=True)
class GoldDocument:
    chunk_id: str
    source_id: str
    domain: str
    source_title: str
    reviewed_text: str
    labels: dict[str, str]
    hard_negative_for: tuple[str, ...]
    evaluation_split: str


@dataclass
class ProviderStats:
    provider: str
    model: str
    dimensions: int
    document_seconds: float = 0.0
    query_seconds: float = 0.0
    query_latencies: list[float] = field(default_factory=list)
    input_tokens: int = 0
    api_requests: int = 0
    retries: int = 0
    rate_limit_retries: int = 0
    cache_hits: int = 0
    generated_vectors: int = 0
    memory_mb_before: float | None = None
    memory_mb_after: float | None = None


@dataclass(frozen=True)
class QueryMetrics:
    provider_key: str
    query_id: str
    target_concept: str
    target_domain: str
    focus: str
    kind: str
    relevant_count: int
    recall_at_10: float
    precision_at_5: float
    mrr: float
    ndcg_at_5: float
    hard_negative_fp_rate_at_10: float
    hard_negative_hits_at_10: int
    score_margin: float
    top_chunk_ids: tuple[str, ...]


class EmbeddingProvider(Protocol):
    key: str
    provider_name: str
    model_id: str
    dimensions: int
    stats: ProviderStats

    def embed_documents(self, documents: Sequence[GoldDocument]) -> list[list[float]]: ...

    def embed_queries(self, queries: Sequence[BenchmarkQuery]) -> list[list[float]]: ...


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Phase 1 embedding candidates using frozen Build and "
            "Development only; Held-out content is never parsed."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--heldout", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-directory", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--models",
        default="gemini,local,cohere",
        help="Comma-separated subset of gemini,local,cohere.",
    )
    parser.add_argument("--allow-unavailable", action="store_true")
    parser.add_argument("--gemini-min-interval-seconds", type=float, default=0.75)
    parser.add_argument("--cohere-batch-size", type=int, default=48)
    parser.add_argument("--gemini-price-per-million-input-tokens", type=float)
    parser.add_argument("--cohere-price-per-million-input-tokens", type=float)
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


def resolve(project_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise BenchmarkError(f"Required file does not exist: {path}")


def require_mapping(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise BenchmarkError(f"{description} must be an object")
    result: dict[str, object] = {}
    for key, nested in value.items():
        if not isinstance(key, str):
            raise BenchmarkError(f"{description} contains a non-string key")
        result[key] = nested
    return result


def require_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkError(f"{description} must be a non-empty string")
    return value.strip()


def optional_string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def load_json(path: Path) -> dict[str, object]:
    try:
        loaded: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"Invalid JSON in {path}: {exc}") from exc
    return require_mapping(loaded, f"JSON document {path}")


def iter_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                loaded: object = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise BenchmarkError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            yield require_mapping(loaded, f"record at {path}:{line_number}")


def parse_string_list(value: object) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(
            item.strip().casefold() for item in value if isinstance(item, str) and item.strip()
        )
    if isinstance(value, str) and value.strip():
        return tuple(
            part.strip().casefold()
            for part in value.replace(";", "|").replace(",", "|").split("|")
            if part.strip()
        )
    return ()


def parse_document(raw: Mapping[str, object]) -> GoldDocument:
    review = require_mapping(raw.get("review"), "review")
    labels_raw = require_mapping(review.get("labels"), "review.labels")
    labels = {
        concept: require_string(labels_raw.get(concept), f"review.labels.{concept}").casefold()
        for concept in CONCEPTS
    }
    if any(label not in VALID_LABELS for label in labels.values()):
        raise BenchmarkError(f"Invalid labels in {raw.get('chunk_id')!r}: {labels}")

    split = require_string(raw.get("evaluation_split"), "evaluation_split")
    if split not in {"build", "development"}:
        raise BenchmarkError(
            f"Benchmark input contains forbidden split {split!r}; Held-out must not be benchmark input"
        )

    return GoldDocument(
        chunk_id=require_string(raw.get("chunk_id"), "chunk_id"),
        source_id=require_string(raw.get("source_id"), "source_id"),
        domain=require_string(raw.get("domain"), "domain").casefold(),
        source_title=optional_string(raw.get("source_title")),
        reviewed_text=require_string(raw.get("reviewed_text"), "reviewed_text"),
        labels=labels,
        hard_negative_for=parse_string_list(review.get("hard_negative_for")),
        evaluation_split=split,
    )


def load_documents(build: Path, development: Path) -> list[GoldDocument]:
    documents = [parse_document(raw) for raw in iter_jsonl(build)]
    documents.extend(parse_document(raw) for raw in iter_jsonl(development))
    ids = [document.chunk_id for document in documents]
    if len(ids) != len(set(ids)):
        raise BenchmarkError("Duplicate chunk IDs across Build and Development")
    return documents


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def validate_split_manifest(manifest_path: Path, heldout_path: Path) -> dict[str, object]:
    manifest = load_json(manifest_path)
    if manifest.get("status") != "frozen" or manifest.get("frozen") is not True:
        raise BenchmarkError("Phase 6 split manifest is not frozen")
    outputs = require_mapping(manifest.get("outputs"), "split outputs")
    heldout = require_mapping(outputs.get("heldout"), "heldout output")
    if heldout.get("read_only") is not True:
        raise BenchmarkError("Held-out is not marked read-only")
    expected = require_string(heldout.get("sha256"), "heldout sha256")
    if expected != sha256_file(heldout_path):
        raise BenchmarkError("Held-out checksum changed after freezing")
    return manifest


def query_suite_sha256() -> str:
    payload = [query.__dict__ for query in QUERY_SUITE]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def canonical_document_text(document: GoldDocument) -> str:
    title = document.source_title or "none"
    return f"title: {title} | text: {document.reviewed_text}"


def l2(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0:
        raise BenchmarkError("Zero-norm embedding")
    return [float(value / norm) for value in vector]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise BenchmarkError(f"Dimension mismatch {len(left)} != {len(right)}")
    return float(sum(a * b for a, b in zip(left, right, strict=True)))


class VectorCache:
    def __init__(self, directory: Path, namespace: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in namespace
        )
        self.path = directory / f"{safe_name}.json"
        self.vectors: dict[str, list[float]] = {}
        if self.path.is_file():
            raw = load_json(self.path)
            for key, value in raw.items():
                if isinstance(value, list) and all(isinstance(item, int | float) for item in value):
                    self.vectors[key] = [float(item) for item in value]

    def get(self, key: str) -> list[float] | None:
        vector = self.vectors.get(key)
        return list(vector) if vector is not None else None

    def put(self, key: str, vector: Sequence[float]) -> None:
        self.vectors[key] = [float(value) for value in vector]

    def save(self) -> None:
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.vectors, sort_keys=True) + "\n", encoding="utf-8")
        temp.replace(self.path)


def vector_cache_key(
    provider: str,
    model: str,
    dimensions: int,
    role: str,
    text: str,
) -> str:
    payload = "\x1f".join((provider, model, str(dimensions), role, text))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def process_memory_mb() -> float | None:
    try:
        module = importlib.import_module("psutil")
    except ModuleNotFoundError:
        return None
    process_class = getattr(module, "Process", None)
    if process_class is None:
        return None
    process = process_class(os.getpid())
    info = process.memory_info()
    rss = getattr(info, "rss", None)
    if not isinstance(rss, int | float):
        return None
    return float(rss) / (1024.0 * 1024.0)


class JsonHttpClient:
    def __init__(self, stats: ProviderStats) -> None:
        self.stats = stats

    def post(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        body: Mapping[str, object],
        attempts: int = 6,
    ) -> tuple[dict[str, object], float]:
        encoded = json.dumps(body).encode("utf-8")
        for attempt in range(attempts):
            request = urllib.request.Request(
                url,
                data=encoded,
                headers=dict(headers),
                method="POST",
            )
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    response_bytes = response.read()
                elapsed = time.perf_counter() - started
                loaded: object = json.loads(response_bytes.decode("utf-8"))
                self.stats.api_requests += 1
                return require_mapping(loaded, "HTTP response"), elapsed
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    self.stats.rate_limit_retries += 1
                if exc.code not in {429, 500, 502, 503, 504} or attempt + 1 >= attempts:
                    detail = exc.read().decode("utf-8", errors="replace")
                    raise BenchmarkError(f"HTTP {exc.code}: {detail[:1000]}") from exc
                self.stats.retries += 1
                time.sleep(min(30.0, 1.5 * (2**attempt)))
            except urllib.error.URLError as exc:
                if attempt + 1 >= attempts:
                    raise BenchmarkError(f"Network error: {exc}") from exc
                self.stats.retries += 1
                time.sleep(min(30.0, 1.5 * (2**attempt)))
        raise BenchmarkError(f"Failed to call {url}")


class GeminiProvider:
    key = "gemini"

    def __init__(self, api_key: str, cache_dir: Path, min_interval: float) -> None:
        self.provider_name = "Google Gemini API"
        self.model_id = GEMINI_MODEL
        self.dimensions = GEMINI_DIMENSIONS
        self.stats = ProviderStats(self.provider_name, self.model_id, self.dimensions)
        self.api_key = api_key
        self.cache = VectorCache(cache_dir, f"gemini_{self.model_id}_{self.dimensions}")
        self.http = JsonHttpClient(self.stats)
        self.min_interval = max(0.0, min_interval)
        self.last_request_at = 0.0

    def embed_one(self, role: str, text: str) -> tuple[list[float], float, bool]:
        cache_key = vector_cache_key(self.key, self.model_id, self.dimensions, role, text)
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.stats.cache_hits += 1
            return cached, 0.0, True

        elapsed_since = time.perf_counter() - self.last_request_at
        if elapsed_since < self.min_interval:
            time.sleep(self.min_interval - elapsed_since)

        payload, latency = self.http.post(
            url=GEMINI_ENDPOINT,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            body={
                "content": {"parts": [{"text": text}]},
                "output_dimensionality": self.dimensions,
            },
        )
        self.last_request_at = time.perf_counter()
        embedding_value = payload.get("embedding")
        if embedding_value is None:
            embeddings = payload.get("embeddings")
            if isinstance(embeddings, list) and embeddings:
                embedding_value = embeddings[0]
        embedding = require_mapping(embedding_value, "Gemini embedding")
        values = embedding.get("values")
        if not isinstance(values, list) or not all(
            isinstance(item, int | float) for item in values
        ):
            raise BenchmarkError("Gemini response missing numeric values")
        vector = l2([float(item) for item in values])
        if len(vector) != self.dimensions:
            raise BenchmarkError(f"Gemini dimensions {len(vector)} != {self.dimensions}")

        usage = payload.get("usageMetadata")
        if isinstance(usage, Mapping):
            for field_name in ("promptTokenCount", "inputTokenCount", "totalTokenCount"):
                token_count = usage.get(field_name)
                if isinstance(token_count, int):
                    self.stats.input_tokens += token_count
                    break

        self.cache.put(cache_key, vector)
        self.cache.save()
        self.stats.generated_vectors += 1
        return vector, latency, False

    def embed_documents(self, documents: Sequence[GoldDocument]) -> list[list[float]]:
        started = time.perf_counter()
        vectors: list[list[float]] = []
        for index, document in enumerate(documents, start=1):
            vector, _, _ = self.embed_one("document", canonical_document_text(document))
            vectors.append(vector)
            if index % 25 == 0:
                LOGGER.info("Gemini documents %d/%d", index, len(documents))
        self.stats.document_seconds = time.perf_counter() - started
        return vectors

    def embed_queries(self, queries: Sequence[BenchmarkQuery]) -> list[list[float]]:
        started = time.perf_counter()
        vectors: list[list[float]] = []
        for query in queries:
            prompt = f"task: search result | query: {query.text}"
            vector, latency, cached = self.embed_one("query", prompt)
            vectors.append(vector)
            if not cached:
                self.stats.query_latencies.append(latency)
        self.stats.query_seconds = time.perf_counter() - started
        return vectors


class CohereProvider:
    key = "cohere"

    def __init__(self, api_key: str, cache_dir: Path, batch_size: int) -> None:
        self.provider_name = "Cohere"
        self.model_id = COHERE_MODEL
        self.dimensions = COHERE_DIMENSIONS
        self.stats = ProviderStats(self.provider_name, self.model_id, self.dimensions)
        self.api_key = api_key
        self.cache = VectorCache(cache_dir, f"cohere_{self.model_id}_{self.dimensions}")
        self.http = JsonHttpClient(self.stats)
        self.batch_size = max(1, min(96, batch_size))

    def request(self, texts: Sequence[str], input_type: str) -> tuple[list[list[float]], float]:
        payload, latency = self.http.post(
            url=COHERE_ENDPOINT,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            body={
                "model": self.model_id,
                "texts": list(texts),
                "input_type": input_type,
                "embedding_types": ["float"],
                "output_dimension": self.dimensions,
            },
        )
        embeddings = payload.get("embeddings")
        vectors_value: object = embeddings
        if isinstance(embeddings, Mapping):
            vectors_value = embeddings.get("float")
        if not isinstance(vectors_value, list):
            raise BenchmarkError("Cohere response missing embeddings")
        vectors: list[list[float]] = []
        for item in vectors_value:
            if not isinstance(item, list) or not all(
                isinstance(value, int | float) for value in item
            ):
                raise BenchmarkError("Cohere embedding is not numeric")
            vector = l2([float(value) for value in item])
            if len(vector) != self.dimensions:
                raise BenchmarkError(f"Cohere dimensions {len(vector)} != {self.dimensions}")
            vectors.append(vector)
        if len(vectors) != len(texts):
            raise BenchmarkError("Cohere vector count mismatch")

        meta = payload.get("meta")
        if isinstance(meta, Mapping):
            billed = meta.get("billed_units")
            if isinstance(billed, Mapping):
                input_tokens = billed.get("input_tokens")
                if isinstance(input_tokens, int):
                    self.stats.input_tokens += input_tokens
        return vectors, latency

    def embed_documents(self, documents: Sequence[GoldDocument]) -> list[list[float]]:
        started = time.perf_counter()
        output: list[list[float] | None] = [None] * len(documents)
        pending: list[tuple[int, str, str]] = []
        for index, document in enumerate(documents):
            text = canonical_document_text(document)
            key = vector_cache_key(self.key, self.model_id, self.dimensions, "document", text)
            cached = self.cache.get(key)
            if cached is not None:
                output[index] = cached
                self.stats.cache_hits += 1
            else:
                pending.append((index, key, text))

        for offset in range(0, len(pending), self.batch_size):
            batch = pending[offset : offset + self.batch_size]
            vectors, _ = self.request([item[2] for item in batch], "search_document")
            for (index, key, _), vector in zip(batch, vectors, strict=True):
                output[index] = vector
                self.cache.put(key, vector)
                self.stats.generated_vectors += 1
            self.cache.save()
        self.stats.document_seconds = time.perf_counter() - started
        if any(vector is None for vector in output):
            raise BenchmarkError("Cohere left missing document vectors")
        return [cast(list[float], vector) for vector in output]

    def embed_queries(self, queries: Sequence[BenchmarkQuery]) -> list[list[float]]:
        started = time.perf_counter()
        output: list[list[float]] = []
        for query in queries:
            key = vector_cache_key(self.key, self.model_id, self.dimensions, "query", query.text)
            cached = self.cache.get(key)
            if cached is not None:
                output.append(cached)
                self.stats.cache_hits += 1
                continue
            vectors, latency = self.request([query.text], "search_query")
            vector = vectors[0]
            self.cache.put(key, vector)
            self.cache.save()
            self.stats.generated_vectors += 1
            self.stats.query_latencies.append(latency)
            output.append(vector)
        self.stats.query_seconds = time.perf_counter() - started
        return output


class LocalE5Provider:
    key = "local"

    def __init__(self, cache_dir: Path) -> None:
        self.provider_name = "Local SentenceTransformers"
        self.model_id = LOCAL_MODEL
        self.dimensions = LOCAL_DIMENSIONS
        self.stats = ProviderStats(self.provider_name, self.model_id, self.dimensions)
        self.cache = VectorCache(cache_dir, f"local_{self.model_id}_{self.dimensions}")
        self.model: Any = None

    def load_model(self) -> Any:
        if self.model is not None:
            return self.model
        self.stats.memory_mb_before = process_memory_mb()
        try:
            module = importlib.import_module("sentence_transformers")
        except ModuleNotFoundError as exc:
            raise BenchmarkError(
                "Local benchmark requires sentence-transformers; install it before final " \
                "Phase 7 run"
            ) from exc
        model_class = getattr(module, "SentenceTransformer", None)
        if model_class is None:
            raise BenchmarkError("SentenceTransformer class unavailable")
        self.model = model_class(self.model_id)
        self.stats.memory_mb_after = process_memory_mb()
        return self.model

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        model = self.load_model()
        encoded = model.encode(
            list(texts),
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        output: list[list[float]] = []
        for row in encoded:
            row_list = cast(Sequence[float], row.tolist())
            vector = l2(row_list)
            if len(vector) != self.dimensions:
                raise BenchmarkError(f"Local E5 dimensions {len(vector)} != {self.dimensions}")
            output.append(vector)
        return output

    def embed_role(self, role: str, texts: Sequence[str]) -> list[list[float]]:
        output: list[list[float] | None] = [None] * len(texts)
        pending_indices: list[int] = []
        pending_keys: list[str] = []
        pending_texts: list[str] = []
        for index, text in enumerate(texts):
            key = vector_cache_key(self.key, self.model_id, self.dimensions, role, text)
            cached = self.cache.get(key)
            if cached is not None:
                output[index] = cached
                self.stats.cache_hits += 1
            else:
                pending_indices.append(index)
                pending_keys.append(key)
                pending_texts.append(text)
        if pending_texts:
            vectors = self.encode(pending_texts)
            for index, key, vector in zip(pending_indices, pending_keys, vectors, strict=True):
                output[index] = vector
                self.cache.put(key, vector)
                self.stats.generated_vectors += 1
            self.cache.save()
        if any(vector is None for vector in output):
            raise BenchmarkError("Local E5 left missing vectors")
        return [cast(list[float], vector) for vector in output]

    def embed_documents(self, documents: Sequence[GoldDocument]) -> list[list[float]]:
        started = time.perf_counter()
        vectors = self.embed_role(
            "document",
            [f"passage: {canonical_document_text(document)}" for document in documents],
        )
        self.stats.document_seconds = time.perf_counter() - started
        return vectors

    def embed_queries(self, queries: Sequence[BenchmarkQuery]) -> list[list[float]]:
        started = time.perf_counter()
        output: list[list[float]] = []
        for query in queries:
            per_query_start = time.perf_counter()
            vector = self.embed_role("query", [f"query: {query.text}"])[0]
            self.stats.query_latencies.append(time.perf_counter() - per_query_start)
            output.append(vector)
        self.stats.query_seconds = time.perf_counter() - started
        return output


def relevance_grade(document: GoldDocument, query: BenchmarkQuery) -> int:
    if query.target_domain is not None and document.domain != query.target_domain:
        return 0
    label = document.labels[query.target_concept]
    if label == "positive":
        return 2
    if label == "partial":
        return 1
    return 0


def is_hard_negative(document: GoldDocument, query: BenchmarkQuery) -> bool:
    return query.target_concept in document.hard_negative_for


def dcg(grades: Sequence[int]) -> float:
    return float(sum((2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(grades)))


def evaluate_query(
    provider_key: str,
    query: BenchmarkQuery,
    query_vector: Sequence[float],
    documents: Sequence[GoldDocument],
    document_vectors: Sequence[Sequence[float]],
) -> QueryMetrics:
    ranked = sorted(
        (
            (cosine(query_vector, document_vector), index)
            for index, document_vector in enumerate(document_vectors)
        ),
        key=lambda item: (-item[0], documents[item[1]].chunk_id),
    )
    grades = [relevance_grade(document, query) for document in documents]
    relevant = {index for index, grade in enumerate(grades) if grade > 0}
    relevant_count = len(relevant)
    top10 = ranked[:10]
    top5 = ranked[:5]
    recall = (
        sum(index in relevant for _, index in top10) / relevant_count if relevant_count else 0.0
    )
    precision = sum(index in relevant for _, index in top5) / 5.0

    mrr = 0.0
    for rank, (_, index) in enumerate(ranked, start=1):
        if index in relevant:
            mrr = 1.0 / rank
            break

    actual_grades = [relevance_grade(documents[index], query) for _, index in top5]
    ideal = sorted(grades, reverse=True)[:5]
    ideal_dcg = dcg(ideal)
    ndcg = dcg(actual_grades) / ideal_dcg if ideal_dcg else 0.0

    hard_negative_hits = sum(is_hard_negative(documents[index], query) for _, index in top10)
    hard_negative_fp_rate = hard_negative_hits / 10.0
    relevant_scores = [score for score, index in ranked if index in relevant]
    irrelevant_scores = [score for score, index in ranked if index not in relevant]
    score_margin = (
        max(relevant_scores) - max(irrelevant_scores)
        if relevant_scores and irrelevant_scores
        else 0.0
    )

    return QueryMetrics(
        provider_key=provider_key,
        query_id=query.query_id,
        target_concept=query.target_concept,
        target_domain=query.target_domain or "all",
        focus=query.focus,
        kind=query.kind,
        relevant_count=relevant_count,
        recall_at_10=recall,
        precision_at_5=precision,
        mrr=mrr,
        ndcg_at_5=ndcg,
        hard_negative_fp_rate_at_10=hard_negative_fp_rate,
        hard_negative_hits_at_10=hard_negative_hits,
        score_margin=score_margin,
        top_chunk_ids=tuple(documents[index].chunk_id for _, index in top10),
    )


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else 0.0


def p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return ordered[index]


def aggregate(metrics: Sequence[QueryMetrics]) -> dict[str, float]:
    return {
        "recall_at_10": mean(metric.recall_at_10 for metric in metrics),
        "precision_at_5": mean(metric.precision_at_5 for metric in metrics),
        "mrr": mean(metric.mrr for metric in metrics),
        "ndcg_at_5": mean(metric.ndcg_at_5 for metric in metrics),
        "hard_negative_false_positive_rate_at_10": mean(
            metric.hard_negative_fp_rate_at_10 for metric in metrics
        ),
        "score_margin": mean(metric.score_margin for metric in metrics),
    }


def aggregate_by(metrics: Sequence[QueryMetrics], attribute: str) -> dict[str, dict[str, float]]:
    grouped: defaultdict[str, list[QueryMetrics]] = defaultdict(list)
    for metric in metrics:
        value = getattr(metric, attribute)
        if isinstance(value, str):
            grouped[value].append(metric)
    return {key: aggregate(group) for key, group in sorted(grouped.items())}


def estimate_cost(tokens: int, price: float | None) -> float | None:
    return None if price is None else tokens / 1_000_000.0 * price


def provider_summary(
    provider: EmbeddingProvider,
    metrics: Sequence[QueryMetrics],
    price_per_million: float | None,
) -> dict[str, object]:
    stats = provider.stats
    memory_delta = None
    if stats.memory_mb_before is not None and stats.memory_mb_after is not None:
        memory_delta = stats.memory_mb_after - stats.memory_mb_before
    return {
        "provider_key": provider.key,
        "provider": provider.provider_name,
        "model": provider.model_id,
        "dimensions": provider.dimensions,
        "normalization": "L2 before cosine evaluation",
        "metrics": aggregate(metrics),
        "metrics_by_domain": aggregate_by(metrics, "target_domain"),
        "metrics_by_concept": aggregate_by(metrics, "target_concept"),
        "metrics_by_query_kind": aggregate_by(metrics, "kind"),
        "operational": {
            "document_embedding_seconds": stats.document_seconds,
            "query_embedding_seconds": stats.query_seconds,
            "average_query_latency_seconds": mean(stats.query_latencies),
            "p95_query_latency_seconds": p95(stats.query_latencies),
            "input_tokens_reported": stats.input_tokens,
            "api_requests": stats.api_requests,
            "retries": stats.retries,
            "rate_limit_retries": stats.rate_limit_retries,
            "cache_hits": stats.cache_hits,
            "generated_vectors": stats.generated_vectors,
            "local_memory_mb_before": stats.memory_mb_before,
            "local_memory_mb_after": stats.memory_mb_after,
            "local_memory_delta_mb": memory_delta,
            "price_per_million_input_tokens": price_per_million,
            "estimated_input_cost": estimate_cost(stats.input_tokens, price_per_million),
        },
    }


def create_provider(
    key: str,
    cache_dir: Path,
    gemini_interval: float,
    cohere_batch_size: int,
) -> EmbeddingProvider:
    if key == "gemini":
        api_key = (
            os.environ.get("GOOGLE_API_KEY", "").strip()
            or os.environ.get("GEMINI_API_KEY", "").strip()
            )
        if not api_key:
                raise BenchmarkError(
                    "Neither GOOGLE_API_KEY nor GEMINI_API_KEY is set."
                )
        if not api_key:
            raise BenchmarkError("GEMINI_API_KEY is not set")
        return GeminiProvider(api_key, cache_dir, gemini_interval)
    if key == "local":
        return LocalE5Provider(cache_dir)
    if key == "cohere":
        api_key = os.environ.get("COHERE_API_KEY", "").strip()
        if not api_key:
            raise BenchmarkError("COHERE_API_KEY is not set")
        return CohereProvider(api_key, cache_dir, cohere_batch_size)
    raise BenchmarkError(f"Unknown provider key {key!r}")


def write_query_csv(path: Path, metrics: Sequence[QueryMetrics]) -> None:
    fieldnames = [
        "provider_key",
        "query_id",
        "target_concept",
        "target_domain",
        "focus",
        "kind",
        "relevant_count",
        "recall_at_10",
        "precision_at_5",
        "mrr",
        "ndcg_at_5",
        "hard_negative_fp_rate_at_10",
        "hard_negative_hits_at_10",
        "score_margin",
        "top_chunk_ids",
    ]
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric in metrics:
            writer.writerow(
                {
                    "provider_key": metric.provider_key,
                    "query_id": metric.query_id,
                    "target_concept": metric.target_concept,
                    "target_domain": metric.target_domain,
                    "focus": metric.focus,
                    "kind": metric.kind,
                    "relevant_count": metric.relevant_count,
                    "recall_at_10": f"{metric.recall_at_10:.8f}",
                    "precision_at_5": f"{metric.precision_at_5:.8f}",
                    "mrr": f"{metric.mrr:.8f}",
                    "ndcg_at_5": f"{metric.ndcg_at_5:.8f}",
                    "hard_negative_fp_rate_at_10": f"{metric.hard_negative_fp_rate_at_10:.8f}",
                    "hard_negative_hits_at_10": metric.hard_negative_hits_at_10,
                    "score_margin": f"{metric.score_margin:.8f}",
                    "top_chunk_ids": "|".join(metric.top_chunk_ids),
                }
            )
    temp.replace(path)


def atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def benchmark(
    *,
    project_root: Path,
    build: Path,
    development: Path,
    heldout: Path,
    split_manifest: Path,
    output_dir: Path,
    cache_dir: Path,
    model_keys: Sequence[str],
    allow_unavailable: bool,
    gemini_interval: float,
    cohere_batch_size: int,
    gemini_price: float | None,
    cohere_price: float | None,
    replace: bool,
) -> dict[str, object]:
    project_root = project_root.resolve()
    build = resolve(project_root, build)
    development = resolve(project_root, development)
    heldout = resolve(project_root, heldout)
    split_manifest = resolve(project_root, split_manifest)
    output_dir = resolve(project_root, output_dir)
    cache_dir = resolve(project_root, cache_dir)

    for path in (build, development, heldout, split_manifest):
        require_file(path)

    manifest = validate_split_manifest(split_manifest, heldout)
    documents = load_documents(build, development)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / RESULTS_FILENAME
    query_csv_path = output_dir / QUERY_RESULTS_FILENAME
    existing = [path for path in (results_path, query_csv_path) if path.exists()]
    if existing and not replace:
        raise BenchmarkError(
            "Benchmark outputs already exist; use --replace: "
            + ", ".join(path.as_posix() for path in existing)
        )

    all_query_metrics: list[QueryMetrics] = []
    model_results: list[dict[str, object]] = []
    successful: list[str] = []
    prices = {"gemini": gemini_price, "local": None, "cohere": cohere_price}

    for key in model_keys:
        LOGGER.info("Benchmarking %s", key)
        try:
            provider = create_provider(key, cache_dir, gemini_interval, cohere_batch_size)
            doc_vectors = provider.embed_documents(documents)
            query_vectors = provider.embed_queries(QUERY_SUITE)
            metrics = [
                evaluate_query(provider.key, query, query_vector, documents, doc_vectors)
                for query, query_vector in zip(QUERY_SUITE, query_vectors, strict=True)
            ]
            all_query_metrics.extend(metrics)
            model_results.append(provider_summary(provider, metrics, prices[key]))
            successful.append(key)
        except Exception as exc:
            LOGGER.exception("Candidate %s failed", key)
            if not allow_unavailable:
                raise
            model_results.append(
                {
                    "provider_key": key,
                    "availability": False,
                    "error": str(exc),
                }
            )

    if not allow_unavailable and not {"gemini", "local"}.issubset(successful):
        raise BenchmarkError("Final Phase 7 run requires successful Gemini and local benchmarks")

    write_query_csv(query_csv_path, all_query_metrics)
    outputs = require_mapping(manifest.get("outputs"), "split outputs")
    heldout_meta = require_mapping(outputs.get("heldout"), "heldout output")

    result: dict[str, object] = {
        "benchmark_version": BENCHMARK_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "benchmark_complete",
        "decision_status": "pending_human_review",
        "evaluation_policy": {
            "uses_build": True,
            "uses_development": True,
            "parses_heldout_content": False,
            "heldout_checksum_verified": True,
            "heldout_read_only": heldout_meta.get("read_only") is True,
            "similarity": "cosine on L2-normalized vectors",
            "relevance_grades": {"positive": 2, "partial": 1, "negative": 0, "uncertain": 0},
        },
        "inputs": {
            "build": {"path": build.as_posix(), "sha256": sha256_file(build)},
            "development": {
                "path": development.as_posix(),
                "sha256": sha256_file(development),
            },
            "heldout": {
                "path": heldout.as_posix(),
                "sha256": sha256_file(heldout),
                "content_parsed": False,
            },
            "split_manifest": {
                "path": split_manifest.as_posix(),
                "sha256": sha256_file(split_manifest),
            },
            "query_suite_sha256": query_suite_sha256(),
            "query_count": len(QUERY_SUITE),
            "document_count": len(documents),
        },
        "candidate_configurations": {
            "gemini": {
                "provider": "Google Gemini API",
                "model": GEMINI_MODEL,
                "dimensions": GEMINI_DIMENSIONS,
                "query_instruction": "task: search result | query: {content}",
                "document_instruction": "title: {title} | text: {content}",
                "quota_strategy": f">= {gemini_interval:.2f}s/request + cache + 429/5xx backoff",
            },
            "local": {
                "provider": "Local SentenceTransformers",
                "model": LOCAL_MODEL,
                "dimensions": LOCAL_DIMENSIONS,
                "query_instruction": "query: {content}",
                "document_instruction": "passage: title: {title} | text: {content}",
                "batch_size": 32,
            },
            "cohere": {
                "provider": "Cohere",
                "model": COHERE_MODEL,
                "dimensions": COHERE_DIMENSIONS,
                "query_instruction": "input_type=search_query",
                "document_instruction": "input_type=search_document",
                "batch_size": cohere_batch_size,
            },
        },
        "models": model_results,
        "outputs": {"query_results_csv": query_csv_path.as_posix()},
        "next_step": (
            "Review metrics and operational trade-offs, then create "
            "docs/architecture/phase1_embedding_decision.md and freeze one configuration."
        ),
    }
    atomic_json(results_path, result)
    LOGGER.info("Phase 7 benchmark complete")
    LOGGER.info("Documents: %d; queries: %d", len(documents), len(QUERY_SUITE))
    LOGGER.info("Successful candidates: %s", successful)
    LOGGER.info("Results: %s", results_path)
    LOGGER.info("Query CSV: %s", query_csv_path)
    LOGGER.info("Held-out content parsed: NO")
    return result


def main() -> int:
    args = parse_arguments()
    configure_logging(args.log_level)
    model_keys = tuple(part.strip().casefold() for part in args.models.split(",") if part.strip())
    unknown = set(model_keys) - {"gemini", "local", "cohere"}
    if unknown:
        LOGGER.error("Unknown model key(s): %s", sorted(unknown))
        return 1
    try:
        benchmark(
            project_root=args.project_root,
            build=args.build,
            development=args.development,
            heldout=args.heldout,
            split_manifest=args.split_manifest,
            output_dir=args.output_directory,
            cache_dir=args.cache_directory,
            model_keys=model_keys,
            allow_unavailable=args.allow_unavailable,
            gemini_interval=args.gemini_min_interval_seconds,
            cohere_batch_size=args.cohere_batch_size,
            gemini_price=args.gemini_price_per_million_input_tokens,
            cohere_price=args.cohere_price_per_million_input_tokens,
            replace=args.replace,
        )
    except BenchmarkError:
        LOGGER.exception("Phase 7 embedding benchmark failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
