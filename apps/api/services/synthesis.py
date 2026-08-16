from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, Protocol

import httpx

from apps.api.models.runtime_contracts import (
    DomainResponses,
    FrozenRuntimeContract,
    GenerationManifest,
    SynthesisManifest,
    SynthesisResult,
)

LOGGER = logging.getLogger("wth.api.synthesis")


LOGGER = logging.getLogger("wth.phase1.build_synthesis")

SCRIPT_VERSION: Final = "1.1.0"
SYNTHESIS_VERSION: Final = "phase1-cross-domain-synthesis-v4"
PROMPT_VERSION: Final = "phase1-synthesis-tension-prompt-v5"

DOMAINS: Final = ("science", "advaita", "samkhya")
DOMAIN_PAIRS: Final = (
    ("science", "advaita"),
    ("science", "samkhya"),
    ("advaita", "samkhya"),
)
CONCEPTS: Final = (
    "consciousness",
    "self_identity",
    "reality_appearance",
)

CATEGORIES: Final = (
    "surface_similarity",
    "functional_analogy",
    "substantive_agreement",
    "partial_overlap",
    "direct_tension",
    "non_equivalence",
    "insufficient_corpus_coverage",
)

CATEGORY_CODES: Final = {
    "ss": "surface_similarity",
    "fa": "functional_analogy",
    "sa": "substantive_agreement",
    "po": "partial_overlap",
    "dt": "direct_tension",
    "ne": "non_equivalence",
    "ic": "insufficient_corpus_coverage",
}
CATEGORY_CODE_BY_NAME: Final = {value: key for key, value in CATEGORY_CODES.items()}

GROQ_API_URL: Final = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_SYNTHESIS_MODEL: Final = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE: Final = 0.1
DEFAULT_MAX_COMPLETION_TOKENS: Final = 1150
DEFAULT_TIMEOUT_SECONDS: Final = 30.0
DEFAULT_MAX_PROVIDER_ATTEMPTS: Final = 2

MAX_COMPARISONS: Final = 12
MAX_SYNTHESIS_SUMMARY_CHARS: Final = 2200
MAX_EXPLANATION_CHARS: Final = 650
MAX_NON_CONCLUSION_CHARS: Final = 1400

DOMAIN_DISPLAY_NAMES: Final = {
    "science": "Science",
    "advaita": "Advaita Vedanta",
    "samkhya": "Samkhya",
}

CATEGORY_DEFINITIONS: Final = {
    "surface_similarity": "similar wording or description only",
    "functional_analogy": "similar role, different ontology or method",
    "substantive_agreement": "materially compatible claims at same level",
    "partial_overlap": "meaningful overlap plus important differences",
    "direct_tension": "materially incompatible commitments",
    "non_equivalence": "comparable but not the same concept or doctrine",
    "insufficient_corpus_coverage": "evidence too weak for reliable comparison",
}

SCIENCE_PROCESS_TERMS: Final = (
    "neural",
    "brain",
    "cognitive",
    "cognition",
    "process",
    "self-model",
    "self model",
    "constructed",
    "construction",
    "perceptual",
    "perception",
)

ADVAITA_SELF_TERMS: Final = (
    "atman",
    "ātman",
    "brahman",
    "nondual",
    "non-dual",
    "unchanging",
    "permanent self",
    "witness",
    "irreducible",
)

ADVAITA_APPEARANCE_TERMS: Final = (
    "maya",
    "māyā",
    "appearance",
    "dependent appearance",
    "dependent reality",
)

SAMKHYA_PURUSHA_TERMS: Final = (
    "purusha",
    "puruṣa",
    "purushas",
    "puruṣas",
    "plural purusha",
)

SAMKHYA_PRAKRITI_TERMS: Final = (
    "prakriti",
    "prakṛti",
    "real prakriti",
    "real prakṛti",
    "real principle",
)

ATMAN_PURUSHA_EQUIVALENCE_PATTERNS: Final = (
    re.compile(
        r"\b(atman|ātman)\b.{0,55}\b"
        r"(same as|identical to|equivalent to|equals?|is)\b.{0,35}"
        r"\b(purusha|puruṣa)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(purusha|puruṣa)\b.{0,55}\b"
        r"(same as|identical to|equivalent to|equals?|is)\b.{0,35}"
        r"\b(atman|ātman)\b",
        re.IGNORECASE,
    ),
)

ADVAITA_SAMKHYA_SHARED_NONDUAL_PATTERN: Final = re.compile(
    r"\b("
    r"advaita(?: vedanta)?\s+and\s+samkhya"
    r"|samkhya\s+and\s+advaita(?: vedanta)?"
    r"|both"
    r"|the two domains"
    r"|these two"
    r")\b"
    r".{0,140}\b(nondual|non-dual)\b",
    re.IGNORECASE,
)

SCIENCE_METAPHYSICS_PATTERNS: Final = (
    re.compile(
        r"\b(science|scientific|neural|cognitive|perceptual)\b.{0,100}"
        r"\b(proves?|establishes?|demonstrates?)\b.{0,100}"
        r"\b(maya|māyā|atman|ātman|brahman|purusha|puruṣa|"
        r"ultimate reality|metaphysical)\b",
        re.IGNORECASE,
    ),
)

ENTAILMENT_SENSITIVE_TERMS: Final = (
    "fundamental",
    "ultimate reality",
    "permanent",
    "unchanging",
    "unchangeable",
    "irreducible",
    "eternal",
    "nondual",
    "non-dual",
    "dualistic",
    "essence",
    "constant",
    "illusion",
    "maya",
    "māyā",
    "atman",
    "ātman",
    "brahman",
    "purusha",
    "puruṣa",
    "prakriti",
    "prakṛti",
    "self-model",
    "self model",
    "constructed",
    "construction",
    "neural",
    "cognitive",
    "perceptual",
)

COLLECTIVE_STRONG_TERM_PATTERNS: Final = (
    re.compile(
        r"\b(both|the two domains|these two)\b.{0,80}\b"
        r"(fundamental|permanent|unchanging|unchangeable|irreducible|"
        r"ultimate reality|eternal|nondual|non-dual|constant|essence)\b",
        re.IGNORECASE,
    ),
)

CONCEPT_HINT_TERMS: Final = {
    "consciousness": (
        "consciousness",
        "conscious",
        "awareness",
        "experience",
        "experiencing",
        "subjectivity",
    ),
    "self_identity": (
        "self",
        "identity",
        "ego",
        "subject",
        "body ownership",
        "bodily",
    ),
    "reality_appearance": (
        "reality",
        "appearance",
        "perception",
        "perceptual",
        "world",
        "maya",
        "māyā",
        "prakriti",
        "prakṛti",
    ),
}

OVERGENERAL_AGREEMENT_PATTERNS: Final = (
    re.compile(
        r"\b(all three|science,? advaita(?: vedanta)?,? and samkhya)\b"
        r".{0,60}\b(agree|say the same|are equivalent|are identical)\b",
        re.IGNORECASE,
    ),
)


class SynthesisError(RuntimeError):
    """Raised when Phase 16 cannot safely synthesize Phase 15 outputs."""


@dataclass(frozen=True, slots=True)
class Citation:
    chunk_id: str
    source_id: str
    citation: str
    corpus_version: str


@dataclass(frozen=True, slots=True)
class DomainClaim:
    domain: str
    claim_id: str
    text: str
    concepts: tuple[str, ...]
    citations: tuple[Citation, ...]


@dataclass(frozen=True, slots=True)
class DomainInput:
    domain: str
    claims: tuple[DomainClaim, ...]
    limitations: tuple[str, ...]
    unsupported_aspects: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SynthesisProviderConfig:
    api_key: str
    model: str
    temperature: float
    max_completion_tokens: int
    timeout_seconds: float
    max_attempts: int


@dataclass(frozen=True, slots=True)
class ComparisonSlot:
    slot: int
    comparison_id: str
    concept: str
    left_domain: str
    right_domain: str
    left_claim_refs: tuple[str, ...]
    right_claim_refs: tuple[str, ...]
    required_unsupported_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ComparisonDraft:
    comparison_id: str
    category: str
    domains: tuple[str, ...]
    claim_refs: tuple[str, ...]
    limitation_refs: tuple[str, ...]
    concepts: tuple[str, ...]
    explanation: str


@dataclass(frozen=True, slots=True)
class SynthesisDraft:
    synthesis_summary: str
    comparisons: tuple[ComparisonDraft, ...]
    non_conclusion: str


class SynthesisProviderRunner(Protocol):
    """Synchronous provider boundary used by the validated Phase 16 logic."""

    def __call__(
        self,
        *,
        packet: Mapping[str, object],
        slots: tuple[ComparisonSlot, ...],
        config: SynthesisProviderConfig,
    ) -> tuple[dict[str, object], dict[str, object]]: ...


@dataclass(frozen=True, slots=True)
class SynthesisServiceResult:
    """Runtime Phase 16 outputs."""

    synthesis: SynthesisResult
    manifest: SynthesisManifest


def require_mapping(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise SynthesisError(f"{description} must be an object.")

    result: dict[str, object] = {}
    for key, nested in value.items():
        if not isinstance(key, str):
            raise SynthesisError(f"{description} contains a non-string key.")
        result[key] = nested
    return result


def require_list(value: object, description: str) -> list[object]:
    if not isinstance(value, list):
        raise SynthesisError(f"{description} must be a list.")
    return value


def require_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SynthesisError(f"{description} must be a non-empty string.")
    return value.strip()


def optional_string(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def validate_provider_config(config: SynthesisProviderConfig) -> None:
    if not 0.0 <= config.temperature <= 2.0:
        raise SynthesisError("temperature must be between 0 and 2.")
    if config.max_completion_tokens <= 0:
        raise SynthesisError("max_completion_tokens must be positive.")
    if config.timeout_seconds <= 0:
        raise SynthesisError("timeout_seconds must be positive.")
    if config.max_attempts <= 0:
        raise SynthesisError("max_attempts must be positive.")


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
            raise SynthesisError(f"{description} contains invalid value {text!r}.")
        if text not in result:
            result.append(text)

    return tuple(result)


def active_query_concepts(
    query_activation: Mapping[str, object],
) -> tuple[str, ...]:
    raw = require_list(
        query_activation.get("active_concepts"),
        "query active_concepts",
    )
    result: list[str] = []
    for item in raw:
        concept = require_string(
            item,
            "query active concept",
        )
        if concept not in CONCEPTS:
            raise SynthesisError(f"Unexpected active concept {concept!r}.")
        if concept not in result:
            result.append(concept)
    return tuple(result)


def infer_limitation_concepts(text: str) -> tuple[str, ...]:
    normalized = text.casefold()
    inferred: list[str] = []
    for concept in CONCEPTS:
        if any(term.casefold() in normalized for term in CONCEPT_HINT_TERMS[concept]):
            inferred.append(concept)
    return tuple(inferred)


def pairwise_comparison_id(
    concept: str,
    left_domain: str,
    right_domain: str,
) -> str:
    return f"{concept}__{left_domain}__{right_domain}"


def parse_citation(
    raw: Mapping[str, object],
    *,
    corpus_version: str,
    description: str,
) -> Citation:
    citation_corpus_version = require_string(
        raw.get("corpus_version"),
        f"{description}/corpus_version",
    )
    if citation_corpus_version != corpus_version:
        raise SynthesisError(f"{description} corpus version mismatch.")

    return Citation(
        chunk_id=require_string(
            raw.get("chunk_id"),
            f"{description}/chunk_id",
        ),
        source_id=require_string(
            raw.get("source_id"),
            f"{description}/source_id",
        ),
        citation=require_string(
            raw.get("citation"),
            f"{description}/citation",
        ),
        corpus_version=citation_corpus_version,
    )


def parse_domain_input(
    *,
    domain: str,
    raw: Mapping[str, object],
    corpus_version: str,
) -> DomainInput:
    returned_domain = require_string(
        raw.get("domain"),
        f"{domain} response domain",
    )
    if returned_domain != domain:
        raise SynthesisError(f"{domain} response returned domain {returned_domain!r}.")

    validation = require_mapping(
        raw.get("validation"),
        f"{domain} validation",
    )
    if validation.get("passed") is not True:
        raise SynthesisError(f"{domain} Phase 15 response did not pass validation.")

    grounding = require_mapping(
        raw.get("grounding"),
        f"{domain} grounding",
    )
    for field_name in (
        "all_claims_use_retrieved_chunks",
        "all_citations_canonicalized_from_evidence",
        "all_claims_preserve_corpus_version",
    ):
        if grounding.get(field_name) is not True:
            raise SynthesisError(f"{domain} grounding failed: {field_name}.")

    leakage = require_mapping(
        raw.get("domain_leakage"),
        f"{domain} domain_leakage",
    )
    if leakage.get("passed") is not True:
        raise SynthesisError(f"{domain} Phase 15 response failed domain leakage validation.")

    response_corpus_version = require_string(
        raw.get("corpus_version"),
        f"{domain} corpus_version",
    )
    if response_corpus_version != corpus_version:
        raise SynthesisError(f"{domain} corpus version differs from Phase 15 manifest.")

    claims_raw = require_list(
        raw.get("claims"),
        f"{domain} claims",
    )
    claims: list[DomainClaim] = []
    seen_claim_ids: set[str] = set()

    for index, claim_raw in enumerate(claims_raw, start=1):
        claim = require_mapping(
            claim_raw,
            f"{domain} claim {index}",
        )
        claim_id = require_string(
            claim.get("claim_id"),
            f"{domain} claim_id",
        )
        if claim_id in seen_claim_ids:
            raise SynthesisError(f"{domain} duplicate claim_id {claim_id!r}.")
        seen_claim_ids.add(claim_id)

        citations_raw = require_list(
            claim.get("citations"),
            f"{domain}/{claim_id}/citations",
        )
        if not citations_raw:
            raise SynthesisError(f"{domain}/{claim_id} has no citations.")

        citations = tuple(
            parse_citation(
                require_mapping(
                    citation_raw,
                    f"{domain}/{claim_id} citation {citation_index}",
                ),
                corpus_version=corpus_version,
                description=(f"{domain}/{claim_id}/citation/{citation_index}"),
            )
            for citation_index, citation_raw in enumerate(
                citations_raw,
                start=1,
            )
        )

        claims.append(
            DomainClaim(
                domain=domain,
                claim_id=claim_id,
                text=require_string(
                    claim.get("text"),
                    f"{domain}/{claim_id}/text",
                ),
                concepts=parse_string_list(
                    claim.get("concepts_covered"),
                    description=(f"{domain}/{claim_id}/concepts_covered"),
                    allowed=set(CONCEPTS),
                ),
                citations=citations,
            )
        )

    return DomainInput(
        domain=domain,
        claims=tuple(claims),
        limitations=parse_string_list(
            raw.get("limitations"),
            description=f"{domain} limitations",
        ),
        unsupported_aspects=parse_string_list(
            raw.get("unsupported_aspects"),
            description=f"{domain} unsupported_aspects",
        ),
    )


def claim_reference(domain: str, claim_id: str) -> str:
    return f"{domain}:{claim_id}"


def limitation_reference(domain: str, kind: str, index: int) -> str:
    return f"{domain}:{kind}{index}"


def build_comparison_slots(
    *,
    active_concepts: tuple[str, ...],
    domains: Mapping[str, DomainInput],
) -> tuple[ComparisonSlot, ...]:
    claims_by_domain_concept: dict[
        tuple[str, str],
        list[str],
    ] = {}
    for domain in DOMAINS:
        for claim in domains[domain].claims:
            ref = claim_reference(
                domain,
                claim.claim_id,
            )
            for concept in claim.concepts:
                claims_by_domain_concept.setdefault(
                    (domain, concept),
                    [],
                ).append(ref)

    unsupported_by_domain_concept: dict[
        tuple[str, str],
        list[str],
    ] = {}
    for domain in DOMAINS:
        for index, unsupported in enumerate(
            domains[domain].unsupported_aspects,
            start=1,
        ):
            ref = limitation_reference(
                domain,
                "U",
                index,
            )
            for concept in infer_limitation_concepts(unsupported):
                unsupported_by_domain_concept.setdefault(
                    (domain, concept),
                    [],
                ).append(ref)

    slots: list[ComparisonSlot] = []
    slot_number = 1

    for concept in active_concepts:
        for left_domain, right_domain in DOMAIN_PAIRS:
            slots.append(
                ComparisonSlot(
                    slot=slot_number,
                    comparison_id=pairwise_comparison_id(
                        concept,
                        left_domain,
                        right_domain,
                    ),
                    concept=concept,
                    left_domain=left_domain,
                    right_domain=right_domain,
                    left_claim_refs=tuple(
                        claims_by_domain_concept.get(
                            (left_domain, concept),
                            [],
                        )
                    ),
                    right_claim_refs=tuple(
                        claims_by_domain_concept.get(
                            (right_domain, concept),
                            [],
                        )
                    ),
                    required_unsupported_refs=tuple(
                        dict.fromkeys(
                            [
                                *unsupported_by_domain_concept.get(
                                    (left_domain, concept),
                                    [],
                                ),
                                *unsupported_by_domain_concept.get(
                                    (right_domain, concept),
                                    [],
                                ),
                            ]
                        )
                    ),
                )
            )
            slot_number += 1

    return tuple(slots)


def synthesis_input_packet(
    *,
    question: str,
    corpus_version: str,
    domains: Mapping[str, DomainInput],
    slots: tuple[ComparisonSlot, ...],
) -> dict[str, object]:
    claim_texts: dict[str, str] = {}
    limitation_texts: dict[str, str] = {}

    for domain in DOMAINS:
        domain_input = domains[domain]

        for claim in domain_input.claims:
            claim_texts[
                claim_reference(
                    domain,
                    claim.claim_id,
                )
            ] = claim.text

        for index, limitation in enumerate(
            domain_input.limitations,
            start=1,
        ):
            limitation_texts[
                limitation_reference(
                    domain,
                    "L",
                    index,
                )
            ] = limitation

        for index, unsupported in enumerate(
            domain_input.unsupported_aspects,
            start=1,
        ):
            limitation_texts[
                limitation_reference(
                    domain,
                    "U",
                    index,
                )
            ] = unsupported

    slot_payload: dict[str, object] = {}
    for slot in slots:
        slot_payload[str(slot.slot)] = {
            "x": slot.concept,
            "l": slot.left_domain,
            "r": slot.right_domain,
            "lc": list(slot.left_claim_refs),
            "rc": list(slot.right_claim_refs),
            "u": list(slot.required_unsupported_refs),
            "ic_allowed": bool(slot.required_unsupported_refs),
        }

    return {
        "q": question,
        "v": corpus_version,
        "claims": claim_texts,
        "limits": limitation_texts,
        "slots": slot_payload,
    }


def response_shape_description(
    slots: tuple[ComparisonSlot, ...],
) -> dict[str, object]:
    return {
        "slots": [
            {
                "i": slot.slot,
                "c": (
                    "ss|fa|sa|po|dt|ne|ic"
                    if slot.required_unsupported_refs
                    else "ss|fa|sa|po|dt|ne"
                ),
                "e": "short grounded comparison sentence",
            }
            for slot in slots
        ]
    }

def synthesis_system_prompt() -> str:
    return (
        "Classify pre-built cross-domain comparison slots from validated "
        "Phase 15 claims only. Preserve domain differences. Use no outside "
        "knowledge. Return one JSON object only."
    )


def synthesis_user_prompt(
    *,
    packet: Mapping[str, object],
    slots: tuple[ComparisonSlot, ...],
) -> str:
    instructions = {
        "codes": CATEGORY_CODES,
        "rules": [
            f"Return exactly {len(slots)} slots.",
            "Return every supplied slot id ('i') exactly once.",
            "For c use exactly one supplied category code.",
            "Compare only the claims supplied for that slot.",
            "Use ic (insufficient_corpus_coverage) only when the supplied slot has ic_allowed=true. When ic_allowed=false, choosing ic is invalid.",
            "The slot field 'u' is Python-owned. Never invent, infer, add, or return limitation references that were not supplied in 'u'.",
            "When ic_allowed=true and c='ic', base the comparison only on the supplied limitation references in 'u'; do not invent a new limitation.",
            "Do not equate Atman/Brahman with Purusha.",
            "Do not present Advaita concepts such as Atman, Brahman, Maya, or non-duality as equivalent to Samkhya concepts such as Purusha, Prakriti, or their dualist ontology.",
            "Never describe non-duality as shared by Advaita and Samkhya; non-duality may be attributed to Advaita only when supported by the supplied claims.",
            "Shared terms such as eternal, constant, unchanging, essence, self, consciousness, witness, or awareness do not by themselves justify substantive_agreement between Advaita and Samkhya. Prefer partial_overlap or functional_analogy unless the supplied claims support compatible meaning at the same conceptual and ontological level.",
            "Do not convert scientific claims, empirical findings, cognitive models, neural observations, perceptual findings, or experimental results into metaphysical proof.",
            "Do not claim that Science proves or disproves Atman, Brahman, Maya, Purusha, Prakriti, non-duality, or any ultimate metaphysical reality unless such a claim is explicitly present in the supplied scientific claim text.",
            "A similarity in function, vocabulary, phenomenology, or explanatory role does not by itself establish ontological identity or substantive agreement.",
            "Use functional_analogy when two claims perform a comparable explanatory or functional role but differ in ontology, mechanism, epistemic status, or level of explanation.",
            "Use partial_overlap when the supplied claims share a limited feature or implication but also contain meaningful differences.",
            "Use substantive_agreement only when the supplied claims support compatible meaning at the same relevant level of explanation and there is no material ontological, epistemic, or semantic conflict.",
            "Use substantive_disagreement only when the supplied claims support an actual incompatible position; absence of evidence, different terminology, or different explanatory levels alone are not substantive disagreement.",
            "Do not infer agreement or disagreement from domain labels alone. Classify only from the supplied claims and supplied limitations.",
            "Do not strengthen, universalize, generalize, or add certainty to the supplied claims.",
            "Do not introduce facts, doctrines, interpretations, mechanisms, definitions, citations, claim ids, or limitation ids that are not present in the supplied slot.",
            "The explanation e must accurately reflect category c and must not state a stronger relationship than c permits.",
            "For functional_analogy, e must explicitly preserve the relevant difference rather than implying identity.",
            "For partial_overlap, e must state both the shared aspect and the important difference when they are supported by the supplied claims.",
            "For substantive_disagreement, e must identify the incompatible positions supported by the supplied claims.",
            "For insufficient_corpus_coverage, e must explain that the supplied evidence is insufficient for that comparison and must not fabricate a substantive relationship.",
            "e must be one short grounded sentence.",
            "Return JSON only.",
        ],        
        "shape": response_shape_description(slots),
        "input": packet,
    }
    return json.dumps(
        instructions,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def groq_payload(
    *,
    packet: Mapping[str, object],
    slots: tuple[ComparisonSlot, ...],
    config: SynthesisProviderConfig,
) -> dict[str, object]:
    return {
        "model": config.model,
        "messages": [
            {
                "role": "system",
                "content": synthesis_system_prompt(),
            },
            {
                "role": "user",
                "content": synthesis_user_prompt(packet=packet, slots=slots),
            },
        ],
        "temperature": config.temperature,
        "max_completion_tokens": config.max_completion_tokens,
        "response_format": {
            "type": "json_object",
        },
    }


def extract_completion_content(raw: object) -> str:
    root = require_mapping(raw, "Groq response")
    choices = require_list(root.get("choices"), "Groq choices")
    if not choices:
        raise SynthesisError("Groq response has no choices.")

    first = require_mapping(
        choices[0],
        "Groq first choice",
    )
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

    return float(attempt * 2)


def provider_call(
    *,
    packet: Mapping[str, object],
    slots: tuple[ComparisonSlot, ...],
    config: SynthesisProviderConfig,
) -> tuple[dict[str, object], dict[str, object]]:
    payload = groq_payload(
        packet=packet,
        slots=slots,
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
                raise SynthesisError(f"Groq synthesis request failed: {exc}") from exc
            wait_seconds = float(attempt * 2)
            LOGGER.warning(
                "Synthesis provider retry %d/%d after HTTP error wait_seconds=%.2f error=%s",
                attempt,
                config.max_attempts,
                wait_seconds,
                exc,
            )
            time.sleep(wait_seconds)
            continue

        elapsed_ms = round(
            (time.perf_counter() - started) * 1000.0,
            2,
        )

        if response.status_code == 200:
            try:
                raw_response: object = response.json()
            except json.JSONDecodeError as exc:
                raise SynthesisError("Groq returned invalid response JSON.") from exc

            content = extract_completion_content(raw_response)
            try:
                parsed: object = json.loads(content)
            except json.JSONDecodeError as exc:
                raise SynthesisError("Synthesis JSON output was not valid JSON.") from exc

            parsed_mapping = require_mapping(
                parsed,
                "synthesis structured response",
            )
            root = require_mapping(
                raw_response,
                "Groq synthesis response",
            )

            choices = require_list(
                root.get("choices"),
                "Groq synthesis choices",
            )
            first_choice = require_mapping(
                choices[0],
                "Groq synthesis first choice",
            )
            finish_reason = optional_string(first_choice.get("finish_reason"))
            if finish_reason and finish_reason != "stop":
                raise SynthesisError(
                    "Groq synthesis completion did not finish normally: "
                    f"finish_reason={finish_reason!r}."
                )

            provider_metadata: dict[str, object] = {
                "provider": "Groq",
                "model_requested": config.model,
                "model_returned": optional_string(root.get("model")),
                "attempt": attempt,
                "latency_ms": elapsed_ms,
                "temperature": config.temperature,
                "max_completion_tokens": config.max_completion_tokens,
                "json_object_mode": True,
                "structured_output_strict": False,
                "finish_reason": finish_reason,
                "slot_count": len(slots),
                "usage": (
                    require_mapping(
                        root.get("usage"),
                        "Groq synthesis usage",
                    )
                    if isinstance(root.get("usage"), Mapping)
                    else {}
                ),
                "system_fingerprint": optional_string(root.get("system_fingerprint")),
            }
            return parsed_mapping, provider_metadata

        body = response.text[:1200]
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
            wait_seconds = (
                retry_wait_seconds(response, attempt)
                if response.status_code == 429
                else float(attempt * 2)
            )
            LOGGER.warning(
                "Synthesis provider retry %d/%d status=%d wait_seconds=%.2f",
                attempt,
                config.max_attempts,
                response.status_code,
                wait_seconds,
            )
            time.sleep(wait_seconds)
            continue

        raise SynthesisError(f"Groq synthesis request failed {last_error}")

    raise SynthesisError(f"Groq synthesis request exhausted retries: {last_error}")


def normalize_synthesis_payload(
    raw: Mapping[str, object],
) -> dict[str, object]:
    candidate = dict(raw)

    if "slots" in candidate or "results" in candidate:
        return candidate

    for wrapper_key in ("synthesis", "response", "result"):
        wrapped = candidate.get(wrapper_key)

        if isinstance(wrapped, Mapping):
            normalized = require_mapping(
                wrapped,
                f"synthesis wrapper {wrapper_key}",
            )

            if "slots" in normalized or "results" in normalized:
                LOGGER.warning(
                    "Synthesis response was wrapped under %r; unwrapping.",
                    wrapper_key,
                )
                return normalized

    return candidate


def normalize_singleton_string(
    value: object,
    *,
    description: str,
) -> str:
    if isinstance(value, str):
        return require_string(value, description)
    if isinstance(value, list) and len(value) == 1:
        return require_string(value[0], description)
    raise SynthesisError(f"{description} must be a string or one-item list.")


def parse_slot_result(
    *,
    slot: ComparisonSlot,
    raw: Mapping[str, object],
) -> ComparisonDraft:
    category_code = normalize_singleton_string(
        raw.get("c"),
        description=f"slot {slot.slot}/category_code",
    ).casefold()
    category = CATEGORY_CODES.get(category_code)
    if category is None:
        raise SynthesisError(f"slot {slot.slot} has invalid category code {category_code!r}.")

    if (category== "insufficient_corpus_coverage" and not slot.required_unsupported_refs):
        raise SynthesisError(
            "Synthesis provider returned insufficient_corpus_coverage for slot "
            f"{slot.slot!r}, but Python supplied no "
            "valid limitation references."
        )

    explanation = require_string(
        raw.get("e"),
        f"slot {slot.slot}/explanation",
    )
    if len(explanation) > MAX_EXPLANATION_CHARS:
        raise SynthesisError(
            f"slot {slot.slot} explanation exceeds {MAX_EXPLANATION_CHARS} characters."
        )

    if category == "insufficient_corpus_coverage":
        claim_refs: tuple[str, ...] = ()
    else:
        claim_refs = tuple(
            dict.fromkeys(
                [
                    *slot.left_claim_refs,
                    *slot.right_claim_refs,
                ]
            )
        )

    limitation_refs = slot.required_unsupported_refs

    return ComparisonDraft(
        comparison_id=slot.comparison_id,
        category=category,
        domains=(
            slot.left_domain,
            slot.right_domain,
        ),
        claim_refs=claim_refs,
        limitation_refs=limitation_refs,
        concepts=(slot.concept,),
        explanation=explanation,
    )


def parse_synthesis_draft(
    raw: Mapping[str, object],
    *,
    slots: tuple[ComparisonSlot, ...],
) -> SynthesisDraft:
    normalized = normalize_synthesis_payload(raw)

    raw_slots = normalized.get("slots")
    slot_by_id = {slot.slot: slot for slot in slots}

    if isinstance(raw_slots, list):
        if len(raw_slots) != len(slots):
            raise SynthesisError(
                "Synthesis returned incomplete slot matrix: "
                f"expected={len(slots)} actual={len(raw_slots)}."
            )

        comparisons: list[ComparisonDraft] = []

        for index, raw_item in enumerate(raw_slots, start=1):
            item = require_mapping(raw_item, f"slot {index} item")
            slot_id_val = item.get("i")

            if isinstance(slot_id_val, int) and slot_id_val in slot_by_id:
                target_slot = slot_by_id[slot_id_val]
            elif (
                isinstance(slot_id_val, str)
                and slot_id_val.isdigit()
                and int(slot_id_val) in slot_by_id
            ):
                target_slot = slot_by_id[int(slot_id_val)]
            else:
                target_slot = slots[index - 1]

            comparisons.append(
                parse_slot_result(
                    slot=target_slot,
                    raw=item,
                )
            )

        slot_order = {slot.comparison_id: slot.slot for slot in slots}
        comparisons.sort(key=lambda c: slot_order.get(c.comparison_id, 0))

        return SynthesisDraft(
            synthesis_summary="",
            comparisons=tuple(comparisons),
            non_conclusion="",
        )

    raw_results = normalized.get("results")

    if isinstance(raw_results, Mapping):
        results = require_mapping(
            raw_results,
            "results",
        )

        expected_keys = {str(slot.slot) for slot in slots}
        returned_keys = set(results)

        if returned_keys != expected_keys:
            missing = sorted(expected_keys - returned_keys)
            unexpected = sorted(returned_keys - expected_keys)

            raise SynthesisError(
                "Synthesis returned incomplete result matrix: "
                f"missing={missing} unexpected={unexpected}."
            )

        comparisons = []

        for slot in slots:
            raw_result = require_mapping(
                results[str(slot.slot)],
                f"slot {slot.slot} result",
            )

            comparisons.append(
                parse_slot_result(
                    slot=slot,
                    raw=raw_result,
                )
            )

        return SynthesisDraft(
            synthesis_summary="",
            comparisons=tuple(comparisons),
            non_conclusion="",
        )

    raise SynthesisError(
        "Synthesis response must contain either 'slots' as a list or 'results' as an object."
    )


def build_reference_maps(
    domains: Mapping[str, DomainInput],
) -> tuple[
    dict[str, DomainClaim],
    dict[str, tuple[str, str]],
]:
    claims: dict[str, DomainClaim] = {}
    limitations: dict[str, tuple[str, str]] = {}

    for domain in DOMAINS:
        domain_input = domains[domain]

        for claim in domain_input.claims:
            ref = claim_reference(
                domain,
                claim.claim_id,
            )
            if ref in claims:
                raise SynthesisError(f"Duplicate global claim reference {ref!r}.")
            claims[ref] = claim

        for index, limitation in enumerate(
            domain_input.limitations,
            start=1,
        ):
            ref = limitation_reference(
                domain,
                "L",
                index,
            )
            limitations[ref] = (domain, limitation)

        for index, unsupported in enumerate(
            domain_input.unsupported_aspects,
            start=1,
        ):
            ref = limitation_reference(
                domain,
                "U",
                index,
            )
            limitations[ref] = (domain, unsupported)

    return claims, limitations


def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    normalized = text.casefold()
    return any(term.casefold() in normalized for term in terms)


def high_risk_false_equivalence(
    *,
    category: str,
    claim_objects: tuple[DomainClaim, ...],
) -> str | None:
    if category not in {
        "surface_similarity",
        "substantive_agreement",
        "partial_overlap",
    }:
        return None

    by_domain: dict[str, list[DomainClaim]] = {domain: [] for domain in DOMAINS}
    for claim in claim_objects:
        by_domain[claim.domain].append(claim)

    science_text = " ".join(claim.text for claim in by_domain["science"])
    advaita_text = " ".join(claim.text for claim in by_domain["advaita"])
    samkhya_text = " ".join(claim.text for claim in by_domain["samkhya"])

    if (
        category == "substantive_agreement"
        and contains_any(science_text, SCIENCE_PROCESS_TERMS)
        and contains_any(advaita_text, ADVAITA_SELF_TERMS)
    ):
        return (
            "Potential false equivalence: scientific process/self-model language "
            "was classified as substantive agreement with Advaita permanent/"
            "irreducible Self language."
        )

    if (
        category == "substantive_agreement"
        and contains_any(advaita_text, ADVAITA_SELF_TERMS)
        and contains_any(samkhya_text, SAMKHYA_PURUSHA_TERMS)
    ):
        return (
            "Potential false equivalence: Advaita Atman/Brahman language was "
            "classified as substantive agreement with Samkhya Purusha language."
        )

    if (
        category == "substantive_agreement"
        and contains_any(advaita_text, ADVAITA_APPEARANCE_TERMS)
        and contains_any(samkhya_text, SAMKHYA_PRAKRITI_TERMS)
    ):
        return (
            "Potential false equivalence: Advaita dependent appearance language "
            "was classified as substantive agreement with Samkhya real Prakriti."
        )

    if (
        category == "substantive_agreement"
        and contains_any(science_text, SCIENCE_PROCESS_TERMS)
        and contains_any(advaita_text, ADVAITA_APPEARANCE_TERMS)
    ):
        return (
            "Potential false equivalence: scientific perceptual construction was "
            "classified as substantive agreement with metaphysical Maya/appearance."
        )

    return None


def expected_pairwise_targets(
    *,
    active_concepts: tuple[str, ...],
) -> dict[str, tuple[str, str, str]]:
    targets: dict[str, tuple[str, str, str]] = {}
    for concept in active_concepts:
        for left_domain, right_domain in DOMAIN_PAIRS:
            comparison_id = pairwise_comparison_id(
                concept,
                left_domain,
                right_domain,
            )
            targets[comparison_id] = (
                concept,
                left_domain,
                right_domain,
            )
    return targets


def explanation_entailment_issues(
    *,
    comparison: ComparisonDraft,
    claim_objects: tuple[DomainClaim, ...],
) -> list[str]:
    issues: list[str] = []
    explanation = comparison.explanation.casefold()
    combined_claim_text = " ".join(claim.text for claim in claim_objects).casefold()

    for term in ENTAILMENT_SENSITIVE_TERMS:
        normalized_term = term.casefold()
        if normalized_term in explanation and normalized_term not in combined_claim_text:
            issues.append(
                "Explanation introduces entailment-sensitive term "
                f"{term!r} that is absent from the cited claim text."
            )

    if any(pattern.search(comparison.explanation) for pattern in COLLECTIVE_STRONG_TERM_PATTERNS):
        by_domain = {
            domain: " ".join(
                claim.text for claim in claim_objects if claim.domain == domain
            ).casefold()
            for domain in comparison.domains
        }
        strong_terms = (
            "fundamental",
            "permanent",
            "unchanging",
            "unchangeable",
            "irreducible",
            "ultimate reality",
            "eternal",
            "nondual",
            "non-dual",
            "constant",
            "essence",
        )
        for term in strong_terms:
            if term in explanation and not all(
                term in by_domain[domain] for domain in comparison.domains
            ):
                issues.append(
                    "Collective explanation attributes strong term "
                    f"{term!r} to both domains without support in each "
                    "domain's cited claims."
                )

    return issues


def global_synthesis_issues(
    draft: SynthesisDraft,
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []

    text = "\n".join(comparison.explanation for comparison in draft.comparisons)

    for pattern in ATMAN_PURUSHA_EQUIVALENCE_PATTERNS:
        if pattern.search(text):
            issues.append(
                {
                    "severity": "error",
                    "code": "atman_purusha_false_equivalence",
                    "message": "Synthesis equates Atman and Purusha.",
                }
            )

    for pattern in SCIENCE_METAPHYSICS_PATTERNS:
        if pattern.search(text):
            issues.append(
                {
                    "severity": "error",
                    "code": "science_to_metaphysical_proof",
                    "message": ("Synthesis converts scientific evidence into metaphysical proof."),
                }
            )

    for comparison in draft.comparisons:
        if set(comparison.domains) != {
            "advaita",
            "samkhya",
        }:
            continue

        if comparison.category not in {
            "surface_similarity",
            "functional_analogy",
            "substantive_agreement",
            "partial_overlap",
        }:
            continue

        if ADVAITA_SAMKHYA_SHARED_NONDUAL_PATTERN.search(comparison.explanation):
            issues.append(
                {
                    "severity": "error",
                    "code": "advaita_samkhya_shared_nonduality",
                    "comparison_id": comparison.comparison_id,
                    "message": (
                        "Synthesis incorrectly attributes "
                        "non-duality as a shared Advaita/Samkhya "
                        "property."
                    ),
                }
            )

    return issues


def deterministic_summary_and_non_conclusion(
    comparison_payloads: Sequence[Mapping[str, object]],
) -> tuple[str, str]:
    category_counts = dict.fromkeys(CATEGORIES, 0)
    for payload in comparison_payloads:
        category = payload.get("category")
        if isinstance(category, str) and category in category_counts:
            category_counts[category] += 1

    represented = [
        (
            category.replace("_", " "),
            category_counts[category],
        )
        for category in CATEGORIES
        if category_counts[category] > 0
    ]

    if represented:
        relation_text = ", ".join(f"{count} {label}" for label, count in represented)
        summary = (
            "Validated pairwise synthesis identified "
            f"{relation_text}. Relationships vary by active concept and "
            "domain pair and are not collapsed into one shared view."
        )
    else:
        summary = (
            "No validated cross-domain comparison relation was established "
            "from the current Phase 15 claims."
        )

    non_conclusion = (
        "These comparisons do not establish that Science, Advaita Vedanta, "
        "or Samkhya proves, reduces to, or is identical with another framework."
    )

    return summary, non_conclusion


def deterministic_three_way_overview(
    comparison_payloads: Sequence[Mapping[str, object]],
    *,
    active_concepts: tuple[str, ...],
) -> dict[str, object]:
    """Generates a structured three-way cross-domain overview payload."""
    tensions = [
        p.get("comparison_id") for p in comparison_payloads if p.get("category") == "direct_tension"
    ]
    non_eqs = [
        p.get("comparison_id")
        for p in comparison_payloads
        if p.get("category") == "non_equivalence"
    ]

    return {
        "active_concepts": list(active_concepts),
        "total_pairwise_comparisons": len(comparison_payloads),
        "direct_tension_count": len(tensions),
        "non_equivalence_count": len(non_eqs),
        "direct_tension_comparison_ids": tensions,
        "non_equivalence_comparison_ids": non_eqs,
        "methodological_note": (
            "Three-way comparison is constructed from pairwise analysis "
            "grounded strictly in domain-specific evidence claims."
        ),
    }


def canonicalize_synthesis(
    *,
    draft: SynthesisDraft,
    domains: Mapping[str, DomainInput],
    active_concepts: tuple[str, ...],
    corpus_version: str,
    provider_metadata: Mapping[str, object],
) -> dict[str, object]:
    claim_map, limitation_map = build_reference_maps(domains)
    expected_targets = expected_pairwise_targets(
        active_concepts=active_concepts,
    )

    validation_issues: list[dict[str, str]] = []
    comparison_payloads: list[dict[str, object]] = []
    seen_comparison_ids: set[str] = set()

    for comparison in draft.comparisons:
        if comparison.comparison_id in seen_comparison_ids:
            validation_issues.append(
                {
                    "severity": "warning",
                    "code": "duplicate_comparison_id",
                    "comparison_id": comparison.comparison_id,
                    "message": "Duplicate comparison_id.",
                }
            )
        seen_comparison_ids.add(comparison.comparison_id)

        expected_target = expected_targets.get(comparison.comparison_id)
        if expected_target is None:
            validation_issues.append(
                {
                    "severity": "warning",
                    "code": "unexpected_pairwise_comparison",
                    "comparison_id": comparison.comparison_id,
                    "message": (
                        "Comparison is not one of the required active-concept domain-pair targets."
                    ),
                }
            )
        else:
            expected_concept, left_domain, right_domain = expected_target
            if tuple(comparison.domains) != (left_domain, right_domain):
                validation_issues.append(
                    {
                        "severity": "error",
                        "code": "pairwise_domain_mismatch",
                        "comparison_id": comparison.comparison_id,
                        "message": (
                            "Comparison domains do not match the required ordered domain pair."
                        ),
                    }
                )
            if comparison.concepts != (expected_concept,):
                validation_issues.append(
                    {
                        "severity": "error",
                        "code": "pairwise_concept_mismatch",
                        "comparison_id": comparison.comparison_id,
                        "message": (
                            "Comparison concept does not match the required pairwise target."
                        ),
                    }
                )

        valid_claims: list[DomainClaim] = []
        valid_claim_refs: list[str] = []
        valid_limitation_refs: list[str] = []

        for ref in comparison.claim_refs:
            claim = claim_map.get(ref)
            if claim is not None:
                valid_claims.append(claim)
                valid_claim_refs.append(ref)

        for ref in comparison.limitation_refs:
            limitation = limitation_map.get(ref)
            if limitation is not None:
                valid_limitation_refs.append(ref)

        if comparison.category == "insufficient_corpus_coverage":
            if not valid_limitation_refs:
                validation_issues.append(
                    {
                        "severity": "error",
                        "code": "coverage_without_limitation",
                        "comparison_id": comparison.comparison_id,
                        "message": (
                            "insufficient_corpus_coverage requires at least "
                            "one valid limitation_ref."
                        ),
                    }
                )
        else:
            if len(valid_claims) < 2:
                validation_issues.append(
                    {
                        "severity": "error",
                        "code": "comparison_without_two_claims",
                        "comparison_id": comparison.comparison_id,
                        "message": ("Grounded comparison requires at least two valid claims."),
                    }
                )

        false_equivalence = high_risk_false_equivalence(
            category=comparison.category,
            claim_objects=tuple(valid_claims),
        )
        if false_equivalence is not None:
            validation_issues.append(
                {
                    "severity": "error",
                    "code": "high_risk_false_equivalence",
                    "comparison_id": comparison.comparison_id,
                    "message": false_equivalence,
                }
            )

        for entailment_issue in explanation_entailment_issues(
            comparison=comparison,
            claim_objects=tuple(valid_claims),
        ):
            validation_issues.append(
                {
                    "severity": "warning",
                    "code": "explanation_not_entailed_by_claims",
                    "comparison_id": comparison.comparison_id,
                    "message": entailment_issue,
                }
            )

        citation_map: dict[str, dict[str, str]] = {}
        for claim in valid_claims:
            for citation in claim.citations:
                citation_map[citation.chunk_id] = {
                    "chunk_id": citation.chunk_id,
                    "source_id": citation.source_id,
                    "citation": citation.citation,
                    "corpus_version": citation.corpus_version,
                }

        limitation_payloads = [
            {
                "limitation_ref": ref,
                "domain": limitation_map[ref][0],
                "text": limitation_map[ref][1],
            }
            for ref in valid_limitation_refs
            if ref in limitation_map
        ]

        payload_item: dict[str, object] = {
            "comparison_id": comparison.comparison_id,
            "category": comparison.category,
            "domains": list(comparison.domains),
            "claim_refs": valid_claim_refs,
            "limitation_refs": valid_limitation_refs,
            "concepts_covered": list(comparison.concepts),
            "explanation": comparison.explanation,
            "citations": [citation_map[chunk_id] for chunk_id in sorted(citation_map)],
            "limitations": limitation_payloads,
        }
        comparison_payloads.append(payload_item)

    missing_targets = set(expected_targets) - seen_comparison_ids
    for comparison_id in sorted(missing_targets):
        validation_issues.append(
            {
                "severity": "error",
                "code": "missing_required_pairwise_comparison",
                "comparison_id": comparison_id,
                "message": ("Required active-concept domain-pair comparison was not returned."),
            }
        )

    validation_issues.extend(global_synthesis_issues(draft))

    (
        deterministic_summary,
        deterministic_non_conclusion,
    ) = deterministic_summary_and_non_conclusion(comparison_payloads)

    validation_passed = not any(issue.get("severity") == "error" for issue in validation_issues)

    key_tensions = [item for item in comparison_payloads if item["category"] == "direct_tension"]
    non_equivalences = [
        item for item in comparison_payloads if item["category"] == "non_equivalence"
    ]
    insufficient_coverage = [
        item for item in comparison_payloads if item["category"] == "insufficient_corpus_coverage"
    ]

    all_input_limitations: list[dict[str, str]] = []
    for ref in sorted(limitation_map):
        domain, text = limitation_map[ref]
        all_input_limitations.append(
            {
                "limitation_ref": ref,
                "domain": domain,
                "text": text,
            }
        )

    three_way_overview = deterministic_three_way_overview(
        comparison_payloads,
        active_concepts=active_concepts,
    )

    return {
        "synthesis_version": SYNTHESIS_VERSION,
        "prompt_version": PROMPT_VERSION,
        "corpus_version": corpus_version,
        "synthesis_summary": deterministic_summary,
        "three_way_overview": three_way_overview,
        "comparisons": comparison_payloads,
        "pairwise_comparisons": comparison_payloads,
        "key_tensions": key_tensions,
        "non_equivalences": non_equivalences,
        "insufficient_corpus_coverage": insufficient_coverage,
        "domain_limitations": all_input_limitations,
        "non_conclusion": deterministic_non_conclusion,
        "validation": {
            "passed": validation_passed,
            "issue_count": len(validation_issues),
            "issues": validation_issues,
        },
        "provider": dict(provider_metadata),
    }


def validate_phase15_manifest_document(
    manifest: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    """Validate an in-memory Phase 15 manifest using the frozen Phase 16 rules."""

    document = require_mapping(
        manifest,
        "Phase 15 manifest",
    )

    if optional_string(document.get("phase")) != "phase_15_build_domain_specific_generation":
        raise SynthesisError("Generation manifest is not a Phase 15 manifest.")

    if optional_string(document.get("status")) != "domain_generation_complete":
        raise SynthesisError("Phase 15 generation is not complete.")

    exit_gate = require_mapping(
        document.get("exit_gate"),
        "Phase 15 exit_gate",
    )

    required_true = (
        "science_uses_only_science_evidence",
        "advaita_uses_only_advaita_evidence",
        "samkhya_uses_only_samkhya_evidence",
        "every_substantive_claim_maps_to_retrieved_chunks",
        "citations_resolve_from_retrieved_evidence",
        "active_corpus_version_preserved",
        "domain_leakage_validation_passed",
        "independently_grounded_and_claim_cited",
    )

    for field_name in required_true:
        if exit_gate.get(field_name) is not True:
            raise SynthesisError(f"Phase 15 exit gate failed: {field_name}.")

    corpus_version = require_string(
        document.get("corpus_version"),
        "Phase 15 corpus_version",
    )

    return document, corpus_version


def parse_domain_responses_document(
    document: Mapping[str, object],
    *,
    expected_corpus_version: str,
) -> tuple[
    str,
    dict[str, object],
    dict[str, DomainInput],
]:
    """Parse in-memory Phase 15 domain responses with frozen Phase 16 rules."""

    payload = require_mapping(
        document,
        "Phase 15 domain responses",
    )

    corpus_version = require_string(
        payload.get("corpus_version"),
        "domain responses corpus_version",
    )

    if corpus_version != expected_corpus_version:
        raise SynthesisError("Phase 15 manifest and domain_responses corpus versions differ.")

    question = require_string(
        payload.get("question"),
        "domain responses question",
    )

    query_activation = require_mapping(
        payload.get("query_activation"),
        "domain responses query_activation",
    )

    domains_raw = require_mapping(
        payload.get("domains"),
        "domain responses domains",
    )

    domains: dict[str, DomainInput] = {}

    for domain in DOMAINS:
        raw = require_mapping(
            domains_raw.get(domain),
            f"{domain} domain response",
        )
        domains[domain] = parse_domain_input(
            domain=domain,
            raw=raw,
            corpus_version=corpus_version,
        )

    return question, query_activation, domains


def _dump_contract(
    contract: FrozenRuntimeContract,
) -> dict[str, object]:
    raw = contract.model_dump(
        mode="python",
        by_alias=True,
    )

    result: dict[str, object] = {}

    for key, value in raw.items():
        result[key] = value

    return result


class SynthesisService:
    """Execute validated Phase 16 from runtime objects only."""

    async def synthesize(
        self,
        *,
        domain_responses: DomainResponses,
        generation_manifest: GenerationManifest,
        provider_config: SynthesisProviderConfig,
        provider_runner: SynthesisProviderRunner | None = None,
        generated_at: str | None = None,
        generation_elapsed_ms: float | None = None,
        synthesis_output_path: str = ("artifacts/phase1/synthesis/synthesis.json"),
        raise_on_validation_failure: bool = True,
    ) -> SynthesisServiceResult:
        """Generate and validate Phase 16 without artifact file I/O."""

        validate_provider_config(provider_config)

        domain_responses_document = _dump_contract(domain_responses)
        generation_manifest_document = _dump_contract(generation_manifest)

        (
            phase15_manifest,
            corpus_version,
        ) = validate_phase15_manifest_document(generation_manifest_document)

        (
            question,
            query_activation,
            domains,
        ) = parse_domain_responses_document(
            domain_responses_document,
            expected_corpus_version=corpus_version,
        )

        manifest_question = require_string(
            phase15_manifest.get("question"),
            "Phase 15 manifest question",
        )

        if manifest_question != question:
            raise SynthesisError("Phase 15 manifest and domain_responses questions differ.")

        active_concepts = active_query_concepts(query_activation)

        slots = build_comparison_slots(
            active_concepts=active_concepts,
            domains=domains,
        )

        LOGGER.info(
            "Phase 16 runtime slot matrix: active_concepts=%d slots=%d",
            len(active_concepts),
            len(slots),
        )

        packet = synthesis_input_packet(
            question=question,
            corpus_version=corpus_version,
            domains=domains,
            slots=slots,
        )

        # This packet contains Phase 15 structured claims, citations represented
        # only through local claim references, limitations, unsupported-aspect
        # references, and Python-owned slots. Raw Phase 14 chunks/corpus are not
        # accepted by this service and therefore cannot enter the provider call.
        runner = provider_runner or provider_call

        started = time.perf_counter()

        raw, provider_metadata = await asyncio.to_thread(
            runner,
            packet=packet,
            slots=slots,
            config=provider_config,
        )

        elapsed_ms = (
            generation_elapsed_ms
            if generation_elapsed_ms is not None
            else round(
                (time.perf_counter() - started) * 1000.0,
                2,
            )
        )

        try:
            draft = parse_synthesis_draft(
                raw,
                slots=slots,
            )
        except SynthesisError:
            LOGGER.exception(
                "Raw synthesis response: %s",
                json.dumps(
                    raw,
                    ensure_ascii=False,
                    sort_keys=True,
                )[:8000],
            )
            raise

        synthesis_payload = canonicalize_synthesis(
            draft=draft,
            domains=domains,
            active_concepts=active_concepts,
            corpus_version=corpus_version,
            provider_metadata=provider_metadata,
        )

        validation = require_mapping(
            synthesis_payload.get("validation"),
            "synthesis validation",
        )

        exit_gate_passed = validation.get("passed") is True

        full_synthesis_payload: dict[str, object] = {
            "question": question,
            "query_activation": query_activation,
            **synthesis_payload,
        }

        synthesis_model = SynthesisResult.model_validate(full_synthesis_payload)

        comparisons = require_list(
            synthesis_payload.get("comparisons"),
            "synthesis comparisons",
        )
        tensions = require_list(
            synthesis_payload.get("key_tensions"),
            "synthesis key_tensions",
        )
        non_equivalences = require_list(
            synthesis_payload.get("non_equivalences"),
            "synthesis non_equivalences",
        )
        insufficient = require_list(
            synthesis_payload.get("insufficient_corpus_coverage"),
            "synthesis insufficient_corpus_coverage",
        )

        all_comparisons_canonical = all(
            isinstance(item, Mapping) and isinstance(item.get("citations"), list)
            for item in comparisons
        )

        timestamp = generated_at or utc_now()

        manifest_payload: dict[str, object] = {
            "phase": ("phase_16_synthesis_and_tension_detection"),
            "status": ("synthesis_complete" if exit_gate_passed else "synthesis_validation_failed"),
            "script_version": SCRIPT_VERSION,
            "synthesis_version": SYNTHESIS_VERSION,
            "prompt_version": PROMPT_VERSION,
            "generated_at": timestamp,
            "question": question,
            "corpus_version": corpus_version,
            "provider": {
                "provider": "Groq",
                "model": provider_config.model,
                "temperature": provider_config.temperature,
                "json_object_mode": True,
                "structured_output_strict": False,
                "maximum_api_calls": 1,
            },
            "input_policy": {
                "phase15_structured_claims_used": True,
                "phase15_citations_used": True,
                "phase15_limitations_used": True,
                "raw_retrieval_chunks_sent": False,
                "raw_corpus_sent": False,
                "outside_knowledge_allowed": False,
                "pairwise_comparison_required": True,
                "comparison_slots_generated_locally": True,
                "claim_refs_attached_locally": True,
                "limitation_refs_attached_locally": True,
                "claim_texts_deduplicated_in_prompt": True,
                "one_user_question_per_synthesis_request": True,
                "relevant_unsupported_aspects_propagated": True,
            },
            "counts": {
                "active_concept_count": len(active_concepts),
                "required_slot_count": len(slots),
                "comparison_count": len(comparisons),
                "direct_tension_count": len(tensions),
                "non_equivalence_count": len(non_equivalences),
                "insufficient_coverage_count": len(insufficient),
            },
            "timing": {
                "synthesis_generation_elapsed_ms": elapsed_ms,
            },
            "outputs": {
                "synthesis": synthesis_output_path,
            },
            "exit_gate": {
                "phase15_grounded_input_only": True,
                "raw_corpus_not_resent": True,
                "domain_differences_preserved": exit_gate_passed,
                ("unsupported_comparisons_identified_or_left_unasserted"): exit_gate_passed,
                "all_comparison_references_validated": exit_gate_passed,
                ("comparison_citations_canonicalized_from_phase15"): all_comparisons_canonical,
                "known_tension_guardrails_enforced": True,
                ("pairwise_active_concept_matrix_complete"): (
                    exit_gate_passed and len(comparisons) == len(slots)
                ),
                "comparison_identity_owned_by_python": True,
                "limitation_identity_owned_by_python": True,
                ("relevant_unsupported_aspects_propagated"): exit_gate_passed,
                ("synthesis_explanations_entailment_guarded"): exit_gate_passed,
                ("three_way_overview_derived_from_pairwise_relations"): True,
                ("summary_and_non_conclusion_derived_deterministically"): True,
                "atman_purusha_false_equivalence_rejected": True,
                "science_metaphysical_proof_rejected": True,
                (
                    "synthesis_preserves_domain_differences_and_identifies_unsupported_comparisons"
                ): exit_gate_passed,
            },
            "next_step": (
                "If the exit gate passes, freeze Phase 16 synthesis/prompt "
                "versions and begin Phase 17 coverage classification using "
                "activated concept weights, retrieved evidence counts, "
                "domain coverage, citation quality, unsupported "
                "subquestions, and retrieval confidence."
            ),
        }

        manifest_model = SynthesisManifest.model_validate(manifest_payload)

        if raise_on_validation_failure and not exit_gate_passed:
                raw_issues = validation.get("issues")
                issues = raw_issues if isinstance(raw_issues, list) else []

                error_issues = [
                    issue for issue in issues
                    if isinstance(issue, Mapping) and issue.get("severity") == "error"
                ]

                LOGGER.error(
                    "Phase 16 synthesis comparison validation failed: %s",
                    json.dumps(error_issues, ensure_ascii=False, sort_keys=True),
                )

                issue_summary_parts = []
                for issue in error_issues:
                    parts = [
                        str(issue[key]) for key in ("code", "comparison_id", "message")
                        if isinstance(issue.get(key), str)
                    ]
                    if parts:
                        issue_summary_parts.append(" | ".join(parts))

                issue_summary = (
                    "; ".join(issue_summary_parts)
                    if issue_summary_parts
                    else "validation failed with no error details"
                )

                raise SynthesisError(f"Phase 16 generated synthesis but failed comparison validation: {issue_summary}")

        return SynthesisServiceResult(
            synthesis=synthesis_model,
            manifest=manifest_model,
        )


__all__ = [
    "CATEGORIES",
    "PROMPT_VERSION",
    "SYNTHESIS_VERSION",
    "ComparisonSlot",
    "SynthesisError",
    "SynthesisProviderConfig",
    "SynthesisProviderRunner",
    "SynthesisService",
    "SynthesisServiceResult",
    "build_comparison_slots",
    "canonicalize_synthesis",
    "parse_synthesis_draft",
    "provider_call",
    "synthesis_input_packet",
]
