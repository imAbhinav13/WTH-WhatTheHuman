from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import httpx

LOGGER = logging.getLogger("wth.phase1.build_domain_generation")

SCRIPT_VERSION: Final = "1.0.0"
GENERATION_VERSION: Final = "phase1-domain-generation-v1"
PROMPT_VERSION: Final = "phase1-domain-grounding-prompt-v1"

DOMAINS: Final = ("science", "advaita", "samkhya")
CONCEPTS: Final = (
    "consciousness",
    "self_identity",
    "reality_appearance",
)

DEFAULT_EVIDENCE_PACKAGE: Final = Path("artifacts/phase1/retrieval/evidence_package.json")
DEFAULT_RETRIEVAL_MANIFEST: Final = Path("artifacts/phase1/retrieval/retrieval_manifest.json")
DEFAULT_OUTPUT_DIRECTORY: Final = Path("artifacts/phase1/generation")

GROQ_API_URL: Final = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_REASONING_EFFORT: Final = "low"
DEFAULT_TEMPERATURE: Final = 0.2
DEFAULT_MAX_COMPLETION_TOKENS: Final = 1800
DEFAULT_TIMEOUT_SECONDS: Final = 60.0
DEFAULT_MAX_PROVIDER_ATTEMPTS: Final = 3

MAX_CLAIMS_PER_DOMAIN: Final = 6
MAX_SUMMARY_CHARS: Final = 2200
MAX_CLAIM_CHARS: Final = 1400
MAX_LIMITATION_CHARS: Final = 1200
MAX_UNSUPPORTED_CHARS: Final = 1200

REASONING_EFFORT_MODELS: Final = {
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
}

JSON_OBJECT_MODELS: Final = {
    "llama-3.3-70b-versatile",
}

DOMAIN_DISPLAY_NAMES: Final = {
    "science": "Science",
    "advaita": "Advaita Vedanta",
    "samkhya": "Samkhya",
}

DOMAIN_SYSTEM_RULES: Final = {
    "science": (
        "You are the Science domain generator. Use ONLY the supplied Science "
        "evidence. Treat it as empirical/scientific literature, not as "
        "metaphysical proof. Do not introduce Advaita Vedanta or Samkhya "
        "doctrines, and do not claim that perceptual or self-model research "
        "proves Maya, Atman, Purusha, or any ultimate metaphysical conclusion."
    ),
    "advaita": (
        "You are the Advaita Vedanta domain generator. Use ONLY the supplied "
        "Advaita evidence. Describe Advaita on its own terms. Do not import "
        "Samkhya ontology such as Purusha/Prakriti into Advaita claims. "
        "Do not present Advaita claims as scientific findings."
    ),
    "samkhya": (
        "You are the Samkhya domain generator. Use ONLY the supplied Samkhya "
        "evidence. Describe Samkhya on its own terms. Do not import Advaita "
        "ontology such as Atman/Brahman/Maya into Samkhya claims. Never merge "
        "Purusha with Atman."
    ),
}

SCIENCE_FORBIDDEN_TERMS: Final = (
    "atman",
    "brahman",
    "maya",
    "purusha",
    "prakriti",
    "advaita",
    "samkhya",
)

ADVAITA_FORBIDDEN_TERMS: Final = (
    "purusha",
    "prakriti",
    "samkhya",
)

SAMKHYA_FORBIDDEN_TERMS: Final = (
    "atman",
    "brahman",
    "maya",
    "advaita",
)

METAPHYSICAL_PROOF_PATTERNS: Final = (
    re.compile(
        r"\b(science|scientific|experiment(?:s|al)?)\b.{0,80}"
        r"\b(proves?|demonstrates?|establishes?)\b.{0,80}"
        r"\b(illusion|maya|atman|purusha|ultimate reality|metaphysical)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(proves?|demonstrates?|establishes?)\b.{0,80}"
        r"\b(reality is (?:an )?illusion|maya is|atman is|purusha is)\b",
        re.IGNORECASE,
    ),
)

ATMAN_PURUSHA_MERGE_PATTERNS: Final = (
    re.compile(
        r"\b(atman|ātman)\b.{0,50}\b"
        r"(same as|identical to|equivalent to|is|equals?)\b.{0,30}"
        r"\bpurusha\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bpurusha\b.{0,50}\b"
        r"(same as|identical to|equivalent to|is|equals?)\b.{0,30}"
        r"\b(atman|ātman)\b",
        re.IGNORECASE,
    ),
)


class GenerationError(RuntimeError):
    """Raised when Phase 15 cannot safely produce grounded domain output."""


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    source_id: str
    domain: str
    citation: str
    reviewed_text: str
    corpus_version: str
    concepts: dict[str, bool]


@dataclass(frozen=True)
class DomainEvidence:
    domain: str
    status: str
    chunks: tuple[EvidenceChunk, ...]


@dataclass(frozen=True)
class ClaimDraft:
    claim_id: str
    text: str
    supporting_chunk_ids: tuple[str, ...]
    concepts_covered: tuple[str, ...]


@dataclass(frozen=True)
class DomainDraft:
    domain: str
    summary: str
    claims: tuple[ClaimDraft, ...]
    concepts_covered: tuple[str, ...]
    limitations: tuple[str, ...]
    unsupported_aspects: tuple[str, ...]


@dataclass(frozen=True)
class ProviderConfig:
    api_key: str
    model: str
    reasoning_effort: str
    temperature: float
    max_completion_tokens: int
    timeout_seconds: float
    max_attempts: int


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 15: generate three independently grounded domain responses "
            "from a Phase 14 evidence package using parallel Groq calls."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--evidence-package",
        type=Path,
        default=DEFAULT_EVIDENCE_PACKAGE,
    )
    parser.add_argument(
        "--retrieval-manifest",
        type=Path,
        default=DEFAULT_RETRIEVAL_MANIFEST,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(f"Groq model ID. Defaults to GROQ_MODEL or {DEFAULT_GROQ_MODEL}."),
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default=DEFAULT_REASONING_EFFORT,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=DEFAULT_MAX_COMPLETION_TOKENS,
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--max-provider-attempts",
        type=int,
        default=DEFAULT_MAX_PROVIDER_ATTEMPTS,
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing Phase 15 derived outputs.",
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
        raise GenerationError(f"Required file does not exist: {path}")


def require_mapping(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise GenerationError(f"{description} must be an object.")

    result: dict[str, object] = {}
    for key, nested in value.items():
        if not isinstance(key, str):
            raise GenerationError(f"{description} contains a non-string key.")
        result[key] = nested
    return result


def require_list(value: object, description: str) -> list[object]:
    if not isinstance(value, list):
        raise GenerationError(f"{description} must be a list.")
    return value


def require_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GenerationError(f"{description} must be a non-empty string.")
    return value.strip()


def optional_string(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def require_bool(value: object, description: str) -> bool:
    if not isinstance(value, bool):
        raise GenerationError(f"{description} must be boolean.")
    return value


def load_json(path: Path) -> dict[str, object]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GenerationError(f"Invalid JSON in {path}: {exc}") from exc
    return require_mapping(raw, f"JSON document {path}")


def atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f"{path.suffix}.tmp")
    temp.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def api_key_from_env(project_root: Path) -> str:
    value = os.getenv("GROQ_API_KEY")
    if value and value.strip():
        return value.strip()

    candidates = (
        project_root / ".env",
        project_root / ".env.local",
        project_root / "apps" / "api" / ".env",
        project_root / "apps" / "api" / ".env.local",
    )

    for path in candidates:
        if not path.is_file():
            continue
        for line in path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, raw_value = stripped.split("=", 1)
            if key.strip() != "GROQ_API_KEY":
                continue
            parsed = raw_value.strip().strip('"').strip("'")
            if parsed:
                return parsed

    raise GenerationError("GROQ_API_KEY not found in environment or project .env files.")


def model_from_configuration(
    project_root: Path,
    explicit_model: str | None,
) -> str:
    if explicit_model and explicit_model.strip():
        return explicit_model.strip()

    env_value = os.getenv("GROQ_MODEL")
    if env_value and env_value.strip():
        return env_value.strip()

    candidates = (
        project_root / ".env",
        project_root / ".env.local",
        project_root / "apps" / "api" / ".env",
        project_root / "apps" / "api" / ".env.local",
    )
    for path in candidates:
        if not path.is_file():
            continue
        for line in path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, raw_value = stripped.split("=", 1)
            if key.strip() != "GROQ_MODEL":
                continue
            parsed = raw_value.strip().strip('"').strip("'")
            if parsed:
                return parsed

    return DEFAULT_GROQ_MODEL


def validate_provider_config(config: ProviderConfig) -> None:
    if config.model != DEFAULT_GROQ_MODEL:
        LOGGER.warning(
            "Phase 15 was designed and schema-validated for %s; configured model is %s.",
            DEFAULT_GROQ_MODEL,
            config.model,
        )
    if not 0.0 <= config.temperature <= 2.0:
        raise GenerationError("temperature must be between 0 and 2.")
    if config.max_completion_tokens <= 0:
        raise GenerationError("max_completion_tokens must be positive.")
    if config.timeout_seconds <= 0:
        raise GenerationError("timeout_seconds must be positive.")
    if config.max_attempts <= 0:
        raise GenerationError("max_attempts must be positive.")


def validate_retrieval_manifest(path: Path) -> dict[str, object]:
    manifest = load_json(path)

    phase = optional_string(manifest.get("phase"))
    if phase != "phase_14_build_retrieval_by_concept_and_domain":
        raise GenerationError("Retrieval manifest is not a Phase 14 manifest.")

    status = optional_string(manifest.get("status"))
    if status != "evaluation_complete":
        raise GenerationError("Phase 14 retrieval evaluation is not complete.")

    gate = require_mapping(
        manifest.get("exit_gate"),
        "Phase 14 exit_gate",
    )
    required_true = (
        "question_embedding_uses_frozen_model",
        "weighted_concept_activation_uses_frozen_phase10",
        "only_active_chunks_retrieved",
        "domain_separation_enforced",
        "source_diversity_enforced",
        "deduplication_enforced",
        "per_domain_context_budgets_enforced",
        "retrieval_evaluation_complete",
    )
    for field_name in required_true:
        if gate.get(field_name) is not True:
            raise GenerationError(f"Phase 14 exit gate failed: {field_name}.")

    if gate.get("concept_aware_retained") is not True:
        raise GenerationError("Concept-aware retrieval was not retained in Phase 14.")

    return manifest


def parse_concepts(
    raw: Mapping[str, object],
    *,
    description: str,
) -> dict[str, bool]:
    concepts: dict[str, bool] = {}
    for concept in CONCEPTS:
        relation_raw = raw.get(concept)
        if relation_raw is None:
            concepts[concept] = False
            continue
        relation = require_mapping(
            relation_raw,
            f"{description}/{concept}",
        )
        concepts[concept] = require_bool(
            relation.get("production_active"),
            f"{description}/{concept}/production_active",
        )
    return concepts


def parse_domain_evidence(
    *,
    domain: str,
    raw: Mapping[str, object],
    corpus_version: str,
) -> DomainEvidence:
    status = require_string(
        raw.get("status"),
        f"{domain} evidence status",
    )
    evidence_raw = require_list(
        raw.get("evidence"),
        f"{domain} evidence",
    )

    chunks: list[EvidenceChunk] = []
    seen: set[str] = set()

    for index, item_raw in enumerate(evidence_raw, start=1):
        item = require_mapping(
            item_raw,
            f"{domain} evidence item {index}",
        )
        item_domain = require_string(
            item.get("domain"),
            f"{domain} evidence item domain",
        ).casefold()
        if item_domain != domain:
            raise GenerationError(
                f"Domain leakage in retrieval package: expected {domain}, found {item_domain}."
            )

        item_corpus_version = require_string(
            item.get("corpus_version"),
            f"{domain} evidence corpus_version",
        )
        if item_corpus_version != corpus_version:
            raise GenerationError(f"{domain} evidence corpus version mismatch.")

        chunk_id = require_string(
            item.get("chunk_id"),
            f"{domain} evidence chunk_id",
        )
        if chunk_id in seen:
            raise GenerationError(f"Duplicate {domain} evidence chunk: {chunk_id}")
        seen.add(chunk_id)

        concepts_raw = require_mapping(
            item.get("concepts"),
            f"{domain}/{chunk_id} concepts",
        )

        chunks.append(
            EvidenceChunk(
                chunk_id=chunk_id,
                source_id=require_string(
                    item.get("source_id"),
                    f"{domain}/{chunk_id} source_id",
                ),
                domain=domain,
                citation=require_string(
                    item.get("citation"),
                    f"{domain}/{chunk_id} citation",
                ),
                reviewed_text=require_string(
                    item.get("reviewed_text"),
                    f"{domain}/{chunk_id} reviewed_text",
                ),
                corpus_version=item_corpus_version,
                concepts=parse_concepts(
                    concepts_raw,
                    description=f"{domain}/{chunk_id} concepts",
                ),
            )
        )

    if status == "evidence_found" and not chunks:
        raise GenerationError(f"{domain} says evidence_found but has no evidence.")

    return DomainEvidence(
        domain=domain,
        status=status,
        chunks=tuple(chunks),
    )


def parse_evidence_package(
    path: Path,
) -> tuple[str, str, dict[str, object], dict[str, DomainEvidence]]:
    package = load_json(path)

    if optional_string(package.get("retrieval_mode")) != "concept_aware":
        raise GenerationError("Phase 15 requires a concept-aware Phase 14 evidence package.")

    question = require_string(
        package.get("question"),
        "evidence package question",
    )
    corpus_version = require_string(
        package.get("corpus_version"),
        "evidence package corpus_version",
    )
    query_activation = require_mapping(
        package.get("query_activation"),
        "evidence package query_activation",
    )
    domains_raw = require_mapping(
        package.get("domains"),
        "evidence package domains",
    )

    domains: dict[str, DomainEvidence] = {}
    for domain in DOMAINS:
        raw = require_mapping(
            domains_raw.get(domain),
            f"{domain} domain package",
        )
        domains[domain] = parse_domain_evidence(
            domain=domain,
            raw=raw,
            corpus_version=corpus_version,
        )

    return question, corpus_version, query_activation, domains


def generation_schema(domain: str) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "domain",
            "summary",
            "claims",
            "concepts_covered",
            "limitations",
            "unsupported_aspects",
        ],
        "properties": {
            "domain": {
                "type": "string",
                "enum": [domain],
            },
            "summary": {
                "type": "string",
            },
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "claim_id",
                        "text",
                        "supporting_chunk_ids",
                        "concepts_covered",
                    ],
                    "properties": {
                        "claim_id": {
                            "type": "string",
                        },
                        "text": {
                            "type": "string",
                        },
                        "supporting_chunk_ids": {
                            "type": "array",
                            "items": {
                                "type": "string",
                            },
                        },
                        "concepts_covered": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": list(CONCEPTS),
                            },
                        },
                    },
                },
            },
            "concepts_covered": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": list(CONCEPTS),
                },
            },
            "limitations": {
                "type": "array",
                "items": {
                    "type": "string",
                },
            },
            "unsupported_aspects": {
                "type": "array",
                "items": {
                    "type": "string",
                },
            },
        },
    }


def active_query_concepts(
    query_activation: Mapping[str, object],
) -> tuple[str, ...]:
    raw = require_list(
        query_activation.get("active_concepts"),
        "query active_concepts",
    )
    result: list[str] = []
    for value in raw:
        concept = require_string(
            value,
            "query active concept",
        )
        if concept not in CONCEPTS:
            raise GenerationError(f"Unexpected active concept: {concept!r}")
        if concept not in result:
            result.append(concept)
    return tuple(result)


def prompt_payload(
    *,
    question: str,
    domain_evidence: DomainEvidence,
    query_activation: Mapping[str, object],
    corpus_version: str,
) -> str:
    active_concepts = active_query_concepts(query_activation)

    evidence_blocks: list[dict[str, object]] = []
    for chunk in domain_evidence.chunks:
        evidence_blocks.append(
            {
                "chunk_id": chunk.chunk_id,
                "source_id": chunk.source_id,
                "citation": chunk.citation,
                "corpus_version": chunk.corpus_version,
                "reviewed_text": chunk.reviewed_text,
                "production_active_concepts": [
                    concept for concept in CONCEPTS if chunk.concepts[concept]
                ],
            }
        )

    instructions = {
        "task": (
            "Answer the user question from this domain only. "
            "The supplied evidence is the only authority."
        ),
        "hard_rules": [
            "Do not use outside knowledge or model memory as factual support.",
            "Do not cite or mention a chunk_id that is not supplied.",
            "Every substantive claim must cite at least one supplied chunk_id.",
            (
                "A claim's concepts_covered may include a concept only when "
                "at least one of its supporting chunks has that concept marked "
                "production_active."
            ),
            "If evidence does not support an aspect, put it in unsupported_aspects.",
            "State limitations instead of filling evidence gaps.",
            "Do not compare this domain with other domains.",
            "Do not editorialize about which worldview is correct.",
            (
                "Keep the summary concise and make it a synthesis of the "
                "claim-cited evidence, not a new unsupported claim."
            ),
            (
                "Your reply MUST be a single raw JSON object that validates "
                "against response_schema below: every property in "
                "response_schema.required must be present, no properties "
                "outside response_schema.properties are allowed, and every "
                "enum must be respected."
            ),
            (
                "The top-level 'domain' field of your reply must be exactly "
                f"{domain_evidence.domain!r} (see response_schema.properties.domain.enum)."
            ),
            (
                "response_schema is a JSON Schema describing the shape your "
                "answer must satisfy. Do not return the schema itself, do "
                "not wrap your answer in another object or key, and do not "
                "add markdown fences or commentary outside the JSON object."
            ),
        ],
        "domain_rule": DOMAIN_SYSTEM_RULES[domain_evidence.domain],
        "domain": domain_evidence.domain,
        "domain_display_name": DOMAIN_DISPLAY_NAMES[domain_evidence.domain],
        "question": question,
        "active_query_concepts": list(active_concepts),
        "corpus_version": corpus_version,
        "evidence": evidence_blocks,
        "response_schema": generation_schema(domain_evidence.domain),
    }

    return json.dumps(
        instructions,
        ensure_ascii=False,
        indent=2,
    )


def system_prompt(domain: str) -> str:
    return (
        "You are a strict evidence-grounded RAG domain generator for WTH. "
        "The user message contains a JSON evidence packet, including a "
        "'response_schema' field. "
        "Follow the packet's hard rules. "
        "Do not use hidden knowledge to add factual content. "
        "Return only a single raw JSON object conforming exactly to "
        "response_schema: no missing required properties, no extra "
        "properties, no wrapping object, no markdown fences, no prose "
        "outside the JSON. " + DOMAIN_SYSTEM_RULES[domain]
    )


def groq_request_payload(
    *,
    domain: str,
    question: str,
    domain_evidence: DomainEvidence,
    query_activation: Mapping[str, object],
    corpus_version: str,
    config: ProviderConfig,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": config.model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt(domain),
            },
            {
                "role": "user",
                "content": prompt_payload(
                    question=question,
                    domain_evidence=domain_evidence,
                    query_activation=query_activation,
                    corpus_version=corpus_version,
                ),
            },
        ],
        "temperature": config.temperature,
        "max_completion_tokens": config.max_completion_tokens,
    }

    # reasoning_effort is supported by GPT-OSS,
    # but NOT by llama-3.3-70b-versatile.
    if config.model in REASONING_EFFORT_MODELS:
        payload["reasoning_effort"] = config.reasoning_effort

    # Llama 3.3 supports JSON Object Mode.
    # Exact schema/grounding validation remains application-side.
    if config.model in JSON_OBJECT_MODELS:
        payload["response_format"] = {
            "type": "json_object",
        }
    else:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": f"phase1_{domain}_domain_response",
                "strict": True,
                "schema": generation_schema(domain),
            },
        }

    return payload


def extract_completion_content(raw: object) -> str:
    root = require_mapping(raw, "Groq response")
    choices = require_list(root.get("choices"), "Groq choices")
    if not choices:
        raise GenerationError("Groq response has no choices.")
    first = require_mapping(choices[0], "Groq first choice")
    message = require_mapping(
        first.get("message"),
        "Groq first choice message",
    )
    return require_string(
        message.get("content"),
        "Groq message content",
    )


def retry_wait_seconds(
    response: httpx.Response,
    attempt: int,
) -> float:
    retry_after = response.headers.get("retry-after")

    if retry_after:
        try:
            return float(retry_after) + 0.5
        except ValueError:
            pass

    match = re.search(
        r"try again in ([\d.]+)s",
        response.text,
        re.IGNORECASE,
    )
    if match:
        return float(match.group(1)) + 0.5

    return float(attempt * 3)


def provider_call(
    *,
    domain: str,
    question: str,
    domain_evidence: DomainEvidence,
    query_activation: Mapping[str, object],
    corpus_version: str,
    config: ProviderConfig,
) -> tuple[dict[str, object], dict[str, object]]:
    payload = groq_request_payload(
        domain=domain,
        question=question,
        domain_evidence=domain_evidence,
        query_activation=query_activation,
        corpus_version=corpus_version,
        config=config,
    )

    last_error = ""

    for attempt in range(1, config.max_attempts + 1):
        started = time.perf_counter()
        try:
            with httpx.Client(
                timeout=config.timeout_seconds,
            ) as client:
                response = client.post(
                    GROQ_API_URL,
                    headers={
                        "Authorization": f"Bearer {config.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
        except httpx.HTTPError as exc:
            last_error = str(exc)
            if attempt == config.max_attempts:
                raise GenerationError(f"{domain} Groq request failed: {exc}") from exc
            LOGGER.warning(
                "%s provider retry %d/%d after HTTP error: %s",
                domain,
                attempt,
                config.max_attempts,
                exc,
            )
            time.sleep(float(attempt * 2))
            continue

        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)

        if response.status_code == 200:
            try:
                raw_response: object = response.json()
            except json.JSONDecodeError as exc:
                raise GenerationError(f"{domain} Groq returned invalid response JSON.") from exc

            content = extract_completion_content(raw_response)
            try:
                parsed: object = json.loads(content)
            except json.JSONDecodeError as exc:
                raise GenerationError(
                    f"{domain} strict structured output was not valid JSON."
                ) from exc

            parsed_mapping = require_mapping(
                parsed,
                f"{domain} structured response",
            )
            root = require_mapping(
                raw_response,
                f"{domain} Groq response",
            )

            provider_metadata: dict[str, object] = {
                "model_requested": config.model,
                "model_returned": optional_string(root.get("model")),
                "provider": "Groq",
                "attempt": attempt,
                "latency_ms": elapsed_ms,
                "usage": (
                    require_mapping(
                        root.get("usage"),
                        f"{domain} Groq usage",
                    )
                    if isinstance(root.get("usage"), Mapping)
                    else {}
                ),
                "system_fingerprint": optional_string(root.get("system_fingerprint")),
                "reasoning_effort": (
                    config.reasoning_effort if config.model in REASONING_EFFORT_MODELS else None
                ),
                "structured_output_strict": (config.model not in JSON_OBJECT_MODELS),
                "json_object_mode": (config.model in JSON_OBJECT_MODELS),
                "temperature": config.temperature,
                "max_completion_tokens": (config.max_completion_tokens),
            }
            return parsed_mapping, provider_metadata

        body = response.text[:1000]
        last_error = f"status={response.status_code} body={body}"

        retryable = response.status_code in {
            408,
            409,
            429,
            500,
            502,
            503,
            504,
        }
        if retryable and attempt < config.max_attempts:
            if response.status_code == 429:
                wait_seconds = retry_wait_seconds(
                    response,
                    attempt,
                )
            else:
                wait_seconds = float(attempt * 3)

            LOGGER.warning(
                "%s provider retry %d/%d status=%d wait_seconds=%.2f",
                domain,
                attempt,
                config.max_attempts,
                response.status_code,
                wait_seconds,
            )

            time.sleep(wait_seconds)
            continue

        raise GenerationError(f"{domain} Groq request failed {last_error}")

    raise GenerationError(f"{domain} Groq request exhausted retries: {last_error}")


def parse_string_list(
    value: object,
    *,
    description: str,
    allowed: set[str] | None = None,
) -> tuple[str, ...]:
    raw = require_list(value, description)
    result: list[str] = []

    for index, item in enumerate(raw, start=1):
        text = require_string(
            item,
            f"{description}[{index}]",
        )
        if allowed is not None and text not in allowed:
            raise GenerationError(f"{description} contains invalid value {text!r}.")
        if text not in result:
            result.append(text)

    return tuple(result)


def unwrap_single_key_domain_payload(
    domain: str,
    raw: Mapping[str, object],
) -> Mapping[str, object]:
    """Defensively unwrap a single top-level wrapper key.

    Models running in unconstrained JSON-object mode (no API-level schema
    enforcement) occasionally nest the real payload under one extra key
    (e.g. {"response": {...}} or {domain: {...}}) even when the exact
    schema is spelled out in the prompt. If `raw` is already well-formed
    (has a non-empty string 'domain'), it is returned unchanged. Otherwise,
    if `raw` has exactly one key whose value is itself a mapping with a
    'domain' key, that inner mapping is used instead.
    """
    existing_domain = raw.get("domain")
    if isinstance(existing_domain, str) and existing_domain.strip():
        return raw

    if len(raw) == 1:
        (only_value,) = raw.values()
        if isinstance(only_value, Mapping) and isinstance(only_value.get("domain"), str):
            LOGGER.warning(
                "%s response was nested under a wrapper key; unwrapping.",
                domain,
            )
            return only_value

    return raw


def parse_domain_draft(
    domain: str,
    raw: Mapping[str, object],
) -> DomainDraft:
    returned_domain = require_string(
        raw.get("domain"),
        f"{domain} response domain",
    )
    if returned_domain != domain:
        raise GenerationError(f"{domain} response returned domain {returned_domain!r}.")

    claims_raw = require_list(
        raw.get("claims"),
        f"{domain} claims",
    )
    if len(claims_raw) > MAX_CLAIMS_PER_DOMAIN:
        raise GenerationError(f"{domain} returned too many claims.")

    claims: list[ClaimDraft] = []
    claim_ids: set[str] = set()

    for index, item_raw in enumerate(claims_raw, start=1):
        item = require_mapping(
            item_raw,
            f"{domain} claim {index}",
        )
        claim_id = require_string(
            item.get("claim_id"),
            f"{domain} claim_id",
        )
        if claim_id in claim_ids:
            raise GenerationError(f"{domain} duplicate claim_id {claim_id!r}.")
        claim_ids.add(claim_id)

        claims.append(
            ClaimDraft(
                claim_id=claim_id,
                text=require_string(
                    item.get("text"),
                    f"{domain}/{claim_id} text",
                ),
                supporting_chunk_ids=parse_string_list(
                    item.get("supporting_chunk_ids"),
                    description=(f"{domain}/{claim_id}/supporting_chunk_ids"),
                ),
                concepts_covered=parse_string_list(
                    item.get("concepts_covered"),
                    description=(f"{domain}/{claim_id}/concepts_covered"),
                    allowed=set(CONCEPTS),
                ),
            )
        )

    summary = require_string(
        raw.get("summary"),
        f"{domain} summary",
    )
    if len(summary) > MAX_SUMMARY_CHARS:
        raise GenerationError(f"{domain} summary exceeds {MAX_SUMMARY_CHARS} characters.")

    for claim in claims:
        if len(claim.text) > MAX_CLAIM_CHARS:
            raise GenerationError(
                f"{domain}/{claim.claim_id} claim exceeds {MAX_CLAIM_CHARS} characters."
            )

    limitations = parse_string_list(
        raw.get("limitations"),
        description=f"{domain} limitations",
    )
    unsupported_aspects = parse_string_list(
        raw.get("unsupported_aspects"),
        description=f"{domain} unsupported_aspects",
    )

    if len(limitations) > 6:
        raise GenerationError(f"{domain} returned too many limitations.")
    if len(unsupported_aspects) > 6:
        raise GenerationError(f"{domain} returned too many unsupported_aspects.")

    if any(len(item) > MAX_LIMITATION_CHARS for item in limitations):
        raise GenerationError(f"{domain} limitation exceeds {MAX_LIMITATION_CHARS} characters.")
    if any(len(item) > MAX_UNSUPPORTED_CHARS for item in unsupported_aspects):
        raise GenerationError(
            f"{domain} unsupported aspect exceeds {MAX_UNSUPPORTED_CHARS} characters."
        )

    return DomainDraft(
        domain=domain,
        summary=summary,
        claims=tuple(claims),
        concepts_covered=parse_string_list(
            raw.get("concepts_covered"),
            description=f"{domain} concepts_covered",
            allowed=set(CONCEPTS),
        ),
        limitations=limitations,
        unsupported_aspects=unsupported_aspects,
    )


def leakage_issues(
    *,
    domain: str,
    draft: DomainDraft,
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []

    texts_to_check = [
        ("summary", draft.summary),
        *[(claim.claim_id, claim.text) for claim in draft.claims],
    ]

    forbidden_terms: tuple[str, ...]
    if domain == "science":
        forbidden_terms = SCIENCE_FORBIDDEN_TERMS
    elif domain == "advaita":
        forbidden_terms = ADVAITA_FORBIDDEN_TERMS
    else:
        forbidden_terms = SAMKHYA_FORBIDDEN_TERMS

    for claim_id, text in texts_to_check:
        normalized = text.casefold()

        for term in forbidden_terms:
            if re.search(
                rf"\b{re.escape(term)}\b",
                normalized,
                re.IGNORECASE,
            ):
                issues.append(
                    {
                        "severity": "error",
                        "code": "cross_domain_ontology",
                        "claim_id": claim_id,
                        "message": (
                            f"{domain} claim contains forbidden cross-domain term {term!r}."
                        ),
                    }
                )

        for pattern in ATMAN_PURUSHA_MERGE_PATTERNS:
            if pattern.search(text):
                issues.append(
                    {
                        "severity": "error",
                        "code": "atman_purusha_merge",
                        "claim_id": claim_id,
                        "message": ("Claim merges or equates Atman and Purusha."),
                    }
                )

        if domain == "science":
            for pattern in METAPHYSICAL_PROOF_PATTERNS:
                if pattern.search(text):
                    issues.append(
                        {
                            "severity": "error",
                            "code": "science_to_metaphysical_proof",
                            "claim_id": claim_id,
                            "message": (
                                "Science claim converts empirical/model "
                                "evidence into metaphysical proof."
                            ),
                        }
                    )

    return issues


def grounded_response(
    *,
    domain_evidence: DomainEvidence,
    draft: DomainDraft,
    corpus_version: str,
    provider_metadata: Mapping[str, object],
) -> dict[str, object]:
    evidence_by_id = {chunk.chunk_id: chunk for chunk in domain_evidence.chunks}

    validation_issues: list[dict[str, str]] = []
    canonical_citations: dict[str, dict[str, str]] = {}
    claim_payloads: list[dict[str, object]] = []

    if domain_evidence.status != "evidence_found" and draft.claims:
        validation_issues.append(
            {
                "severity": "error",
                "code": "claims_without_evidence",
                "claim_id": "",
                "message": (
                    "Model returned substantive claims when retrieval reported no evidence."
                ),
            }
        )

    for claim in draft.claims:
        support_payloads: list[dict[str, str]] = []

        if not claim.supporting_chunk_ids:
            validation_issues.append(
                {
                    "severity": "error",
                    "code": "claim_without_support",
                    "claim_id": claim.claim_id,
                    "message": "Claim has no supporting chunk IDs.",
                }
            )

        for chunk_id in claim.supporting_chunk_ids:
            chunk = evidence_by_id.get(chunk_id)
            if chunk is None:
                validation_issues.append(
                    {
                        "severity": "error",
                        "code": "unknown_supporting_chunk",
                        "claim_id": claim.claim_id,
                        "message": (
                            f"Claim cites chunk {chunk_id!r}, which was "
                            "not retrieved for this domain."
                        ),
                    }
                )
                continue

            if chunk.domain != draft.domain:
                validation_issues.append(
                    {
                        "severity": "error",
                        "code": "supporting_chunk_domain_mismatch",
                        "claim_id": claim.claim_id,
                        "message": (
                            f"Supporting chunk {chunk_id!r} belongs to "
                            f"{chunk.domain}, not {draft.domain}."
                        ),
                    }
                )
                continue

            if chunk.corpus_version != corpus_version:
                validation_issues.append(
                    {
                        "severity": "error",
                        "code": "corpus_version_mismatch",
                        "claim_id": claim.claim_id,
                        "message": (f"Supporting chunk {chunk_id!r} has wrong corpus version."),
                    }
                )
                continue

            support = {
                "chunk_id": chunk.chunk_id,
                "source_id": chunk.source_id,
                "citation": chunk.citation,
                "corpus_version": chunk.corpus_version,
            }
            support_payloads.append(support)
            canonical_citations[chunk.chunk_id] = support

        for concept in claim.concepts_covered:
            concept_supported = any(
                evidence_by_id[chunk_id].concepts[concept]
                for chunk_id in claim.supporting_chunk_ids
                if chunk_id in evidence_by_id
            )
            if not concept_supported:
                validation_issues.append(
                    {
                        "severity": "error",
                        "code": "unsupported_claim_concept",
                        "claim_id": claim.claim_id,
                        "message": (
                            f"Claim labels concept {concept!r}, but none "
                            "of its cited chunks has production_active=true "
                            "for that concept."
                        ),
                    }
                )

        claim_payloads.append(
            {
                "claim_id": claim.claim_id,
                "text": claim.text,
                "supporting_chunk_ids": list(claim.supporting_chunk_ids),
                "citations": support_payloads,
                "concepts_covered": list(claim.concepts_covered),
            }
        )

    leakage = leakage_issues(
        domain=draft.domain,
        draft=draft,
    )
    validation_issues.extend(leakage)

    union_claim_concepts = {concept for claim in draft.claims for concept in claim.concepts_covered}
    invalid_summary_concepts = set(draft.concepts_covered) - union_claim_concepts
    for concept in sorted(invalid_summary_concepts):
        validation_issues.append(
            {
                "severity": "error",
                "code": "domain_concept_without_claim_support",
                "claim_id": "",
                "message": (f"Domain response lists concept {concept!r} but no claim covers it."),
            }
        )

    validation_passed = not any(issue.get("severity") == "error" for issue in validation_issues)

    return {
        "generation_version": GENERATION_VERSION,
        "prompt_version": PROMPT_VERSION,
        "domain": draft.domain,
        "domain_display_name": DOMAIN_DISPLAY_NAMES[draft.domain],
        "summary": draft.summary,
        "claims": claim_payloads,
        "citations": [canonical_citations[chunk_id] for chunk_id in sorted(canonical_citations)],
        "concepts_covered": list(draft.concepts_covered),
        "limitations": list(draft.limitations),
        "unsupported_aspects": list(draft.unsupported_aspects),
        "corpus_version": corpus_version,
        "grounding": {
            "evidence_chunk_count": len(domain_evidence.chunks),
            "retrieval_status": domain_evidence.status,
            "all_claims_use_retrieved_chunks": not any(
                issue.get("code")
                in {
                    "claim_without_support",
                    "unknown_supporting_chunk",
                    "supporting_chunk_domain_mismatch",
                }
                for issue in validation_issues
            ),
            "all_citations_canonicalized_from_evidence": True,
            "all_claims_preserve_corpus_version": not any(
                issue.get("code") == "corpus_version_mismatch" for issue in validation_issues
            ),
        },
        "domain_leakage": {
            "passed": not leakage,
            "issues": leakage,
        },
        "validation": {
            "passed": validation_passed,
            "issue_count": len(validation_issues),
            "issues": validation_issues,
        },
        "provider": dict(provider_metadata),
    }


def no_evidence_response(
    *,
    domain_evidence: DomainEvidence,
    corpus_version: str,
    provider_config: ProviderConfig,
) -> dict[str, object]:
    return {
        "generation_version": GENERATION_VERSION,
        "prompt_version": PROMPT_VERSION,
        "domain": domain_evidence.domain,
        "domain_display_name": DOMAIN_DISPLAY_NAMES[domain_evidence.domain],
        "summary": ("No sufficiently strong active-corpus evidence was retrieved for this domain."),
        "claims": [],
        "citations": [],
        "concepts_covered": [],
        "limitations": [
            "No domain-specific generation was attempted because retrieval "
            "did not return a strong evidence package."
        ],
        "unsupported_aspects": [
            "The question cannot be grounded in this domain from the current Phase 1 active corpus."
        ],
        "corpus_version": corpus_version,
        "grounding": {
            "evidence_chunk_count": 0,
            "retrieval_status": domain_evidence.status,
            "all_claims_use_retrieved_chunks": True,
            "all_citations_canonicalized_from_evidence": True,
            "all_claims_preserve_corpus_version": True,
        },
        "domain_leakage": {
            "passed": True,
            "issues": [],
        },
        "validation": {
            "passed": True,
            "issue_count": 0,
            "issues": [],
        },
        "provider": {
            "provider": "Groq",
            "model_requested": provider_config.model,
            "call_skipped": True,
            "reason": "no_strong_domain_evidence",
        },
    }


def generate_one_domain(
    *,
    domain: str,
    question: str,
    domain_evidence: DomainEvidence,
    query_activation: Mapping[str, object],
    corpus_version: str,
    provider_config: ProviderConfig,
) -> dict[str, object]:
    if domain_evidence.status != "evidence_found" or not domain_evidence.chunks:
        LOGGER.info(
            "%s generation skipped: no strong evidence",
            domain,
        )
        return no_evidence_response(
            domain_evidence=domain_evidence,
            corpus_version=corpus_version,
            provider_config=provider_config,
        )

    LOGGER.info(
        "%s generation starting with %d evidence chunks",
        domain,
        len(domain_evidence.chunks),
    )
    raw, provider_metadata = provider_call(
        domain=domain,
        question=question,
        domain_evidence=domain_evidence,
        query_activation=query_activation,
        corpus_version=corpus_version,
        config=provider_config,
    )
    raw = require_mapping(
        unwrap_single_key_domain_payload(domain, raw), f"{domain} unwrapped payload"
    )
    draft = parse_domain_draft(domain, raw)
    response = grounded_response(
        domain_evidence=domain_evidence,
        draft=draft,
        corpus_version=corpus_version,
        provider_metadata=provider_metadata,
    )

    validation = require_mapping(
        response.get("validation"),
        f"{domain} validation",
    )
    LOGGER.info(
        "%s generation complete claims=%d validation_passed=%s",
        domain,
        len(draft.claims),
        validation.get("passed"),
    )
    return response


def run_parallel_generation(
    *,
    question: str,
    domains: Mapping[str, DomainEvidence],
    query_activation: Mapping[str, object],
    corpus_version: str,
    provider_config: ProviderConfig,
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}

    with ThreadPoolExecutor(
        max_workers=len(DOMAINS),
        thread_name_prefix="phase15-domain",
    ) as executor:
        future_to_domain: dict[
            Future[dict[str, object]],
            str,
        ] = {}

        for index, domain in enumerate(DOMAINS):
            future = executor.submit(
                generate_one_domain,
                domain=domain,
                question=question,
                domain_evidence=domains[domain],
                query_activation=query_activation,
                corpus_version=corpus_version,
                provider_config=provider_config,
            )
            future_to_domain[future] = domain

            if index < len(DOMAINS) - 1:
                time.sleep(0.3)

        for future in as_completed(future_to_domain):
            domain = future_to_domain[future]
            try:
                results[domain] = future.result()
            except Exception as exc:
                raise GenerationError(f"{domain} generation failed.") from exc

    missing = set(DOMAINS) - set(results)
    if missing:
        raise GenerationError(f"Missing domain generation results: {sorted(missing)}")

    return results


def output_paths(
    output_directory: Path,
) -> dict[str, Path]:
    return {
        "science": output_directory / "science_response.json",
        "advaita": output_directory / "advaita_response.json",
        "samkhya": output_directory / "samkhya_response.json",
        "combined": output_directory / "domain_responses.json",
        "manifest": output_directory / "generation_manifest.json",
    }


def ensure_replace_policy(
    *,
    paths: Mapping[str, Path],
    replace: bool,
) -> None:
    if replace:
        return
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise GenerationError(
            "Phase 15 outputs already exist. Use --replace: "
            + ", ".join(path.as_posix() for path in existing)
        )


def run_phase15(
    *,
    project_root: Path,
    evidence_package_path: Path,
    retrieval_manifest_path: Path,
    output_directory: Path,
    provider_config: ProviderConfig,
    replace: bool,
) -> dict[str, object]:
    validate_provider_config(provider_config)

    project_root = project_root.resolve()
    evidence_package_path = resolve(
        project_root,
        evidence_package_path,
    )
    retrieval_manifest_path = resolve(
        project_root,
        retrieval_manifest_path,
    )
    output_directory = resolve(
        project_root,
        output_directory,
    )

    require_file(evidence_package_path)
    require_file(retrieval_manifest_path)

    LOGGER.info(
        "Phase 15 starting: generation_version=%s model=%s",
        GENERATION_VERSION,
        provider_config.model,
    )

    retrieval_manifest = validate_retrieval_manifest(retrieval_manifest_path)
    question, corpus_version, query_activation, domains = parse_evidence_package(
        evidence_package_path
    )

    manifest_corpus = require_string(
        retrieval_manifest.get("corpus_version"),
        "retrieval manifest corpus_version",
    )
    if manifest_corpus != corpus_version:
        raise GenerationError("Retrieval manifest and evidence package corpus versions differ.")

    paths = output_paths(output_directory)
    ensure_replace_policy(paths=paths, replace=replace)
    output_directory.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    responses = run_parallel_generation(
        question=question,
        domains=domains,
        query_activation=query_activation,
        corpus_version=corpus_version,
        provider_config=provider_config,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)

    domain_validation: dict[str, bool] = {}
    total_claims = 0
    total_citations = 0

    for domain in DOMAINS:
        response = responses[domain]
        validation = require_mapping(
            response.get("validation"),
            f"{domain} validation",
        )
        passed = validation.get("passed") is True
        domain_validation[domain] = passed

        claims = require_list(
            response.get("claims"),
            f"{domain} claims",
        )
        citations = require_list(
            response.get("citations"),
            f"{domain} citations",
        )
        total_claims += len(claims)
        total_citations += len(citations)

        atomic_json(paths[domain], response)

    combined: dict[str, object] = {
        "generation_version": GENERATION_VERSION,
        "prompt_version": PROMPT_VERSION,
        "generated_at": utc_now(),
        "question": question,
        "query_activation": query_activation,
        "corpus_version": corpus_version,
        "domains": {domain: responses[domain] for domain in DOMAINS},
    }
    atomic_json(paths["combined"], combined)

    all_domains_passed = all(domain_validation.values())
    all_claims_grounded = all(
        require_mapping(
            responses[domain].get("grounding"),
            f"{domain} grounding",
        ).get("all_claims_use_retrieved_chunks")
        is True
        for domain in DOMAINS
    )
    all_citations_canonical = all(
        require_mapping(
            responses[domain].get("grounding"),
            f"{domain} grounding",
        ).get("all_citations_canonicalized_from_evidence")
        is True
        for domain in DOMAINS
    )
    no_domain_leakage = all(
        require_mapping(
            responses[domain].get("domain_leakage"),
            f"{domain} domain_leakage",
        ).get("passed")
        is True
        for domain in DOMAINS
    )

    exit_gate_passed = (
        all_domains_passed and all_claims_grounded and all_citations_canonical and no_domain_leakage
    )

    manifest: dict[str, object] = {
        "phase": "phase_15_build_domain_specific_generation",
        "status": (
            "domain_generation_complete"
            if exit_gate_passed
            else "domain_generation_validation_failed"
        ),
        "script_version": SCRIPT_VERSION,
        "generation_version": GENERATION_VERSION,
        "prompt_version": PROMPT_VERSION,
        "generated_at": utc_now(),
        "question": question,
        "corpus_version": corpus_version,
        "provider": {
            "provider": "Groq",
            "model": provider_config.model,
            "reasoning_effort": (
                provider_config.reasoning_effort
                if provider_config.model in REASONING_EFFORT_MODELS
                else None
            ),
            "temperature": provider_config.temperature,
            "structured_output_strict": (provider_config.model not in JSON_OBJECT_MODELS),
            "json_object_mode": (provider_config.model in JSON_OBJECT_MODELS),
            "parallel_domain_calls": True,
            "maximum_domain_calls": 3,
        },
        "counts": {
            "domain_count": len(DOMAINS),
            "total_claim_count": total_claims,
            "canonical_citation_count": total_citations,
        },
        "timing": {
            "parallel_generation_elapsed_ms": elapsed_ms,
        },
        "domain_validation": domain_validation,
        "outputs": {key: path.as_posix() for key, path in paths.items() if key != "manifest"},
        "exit_gate": {
            "science_uses_only_science_evidence": (domain_validation["science"]),
            "advaita_uses_only_advaita_evidence": (domain_validation["advaita"]),
            "samkhya_uses_only_samkhya_evidence": (domain_validation["samkhya"]),
            "every_substantive_claim_maps_to_retrieved_chunks": (all_claims_grounded),
            "citations_resolve_from_retrieved_evidence": (all_citations_canonical),
            "active_corpus_version_preserved": all(
                require_mapping(
                    responses[domain].get("grounding"),
                    f"{domain} grounding",
                ).get("all_claims_preserve_corpus_version")
                is True
                for domain in DOMAINS
            ),
            "domain_leakage_validation_passed": (no_domain_leakage),
            "atman_purusha_merge_rejected_or_flagged": True,
            "science_metaphysical_proof_rejected_or_flagged": True,
            "independently_grounded_and_claim_cited": (exit_gate_passed),
        },
        "next_step": (
            "If the exit gate passes, freeze Phase 15 prompt/generation "
            "version and begin Phase 16 synthesis and tension detection. "
            "Phase 16 should consume these structured domain claims, "
            "citations, and limitations rather than resending the raw corpus."
        ),
    }

    atomic_json(paths["manifest"], manifest)

    LOGGER.info("Phase 15 domain generation complete")
    LOGGER.info(
        "Parallel generation elapsed: %.2f ms",
        elapsed_ms,
    )
    for domain in DOMAINS:
        response = responses[domain]
        claims = require_list(
            response.get("claims"),
            f"{domain} claims",
        )
        validation = require_mapping(
            response.get("validation"),
            f"{domain} validation",
        )
        LOGGER.info(
            "%s claims=%d validation_passed=%s",
            domain,
            len(claims),
            validation.get("passed"),
        )
    LOGGER.info(
        "Exit gate passed: %s",
        exit_gate_passed,
    )
    LOGGER.info(
        "Generation manifest: %s",
        paths["manifest"],
    )

    if not exit_gate_passed:
        raise GenerationError(
            "Phase 15 generated outputs but failed grounding/domain-leakage "
            "validation. Inspect generation_manifest.json and per-domain "
            "validation issues."
        )

    return manifest


def main() -> int:
    arguments = parse_arguments()
    configure_logging(arguments.log_level)

    project_root = arguments.project_root.resolve()
    provider_config = ProviderConfig(
        api_key=api_key_from_env(project_root),
        model=model_from_configuration(
            project_root,
            arguments.model,
        ),
        reasoning_effort=arguments.reasoning_effort,
        temperature=arguments.temperature,
        max_completion_tokens=arguments.max_completion_tokens,
        timeout_seconds=arguments.timeout_seconds,
        max_attempts=arguments.max_provider_attempts,
    )

    try:
        run_phase15(
            project_root=project_root,
            evidence_package_path=arguments.evidence_package,
            retrieval_manifest_path=arguments.retrieval_manifest,
            output_directory=arguments.output_directory,
            provider_config=provider_config,
            replace=arguments.replace,
        )
    except GenerationError:
        LOGGER.exception("Phase 15 generation failed")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
