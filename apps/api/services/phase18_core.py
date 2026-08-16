from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Final

ASSEMBLY_VERSION: Final = "phase1-final-response-assembly-v1"

DOMAINS: Final = ("science", "advaita", "samkhya")
DOMAIN_DISPLAY_NAMES: Final = {
    "science": "Science",
    "advaita": "Advaita Vedanta",
    "samkhya": "Samkhya",
}
CONCEPT_DISPLAY_NAMES: Final = {
    "consciousness": "Consciousness",
    "self_identity": "Self / identity",
    "reality_appearance": "Reality / appearance",
}

ALLOWED_COVERAGE_STATUSES: Final = {
    "Supported",
    "Partially Supported",
    "Out of Corpus",
}
ALLOWED_CONCEPT_STATUSES: Final = {
    "Supported",
    "Partially Supported",
    "Unsupported",
}
ALLOWED_SYNTHESIS_CATEGORIES: Final = {
    "surface_similarity",
    "functional_analogy",
    "substantive_agreement",
    "partial_overlap",
    "direct_tension",
    "non_equivalence",
    "insufficient_corpus_coverage",
}

ATMAN_TERMS: Final = ("atman", "ātman")
PURUSHA_TERMS: Final = ("purusha", "puruṣa")
EQUIVALENCE_TERMS: Final = (
    "same as",
    "identical",
    "equivalent",
    "the same concept",
    "the same entity",
)


class AssemblyError(RuntimeError):
    """Raised when Phase 18 cannot safely assemble the final response."""


def require_mapping(
    value: object,
    description: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise AssemblyError(f"{description} must be an object.")

    result: dict[str, object] = {}
    for key, nested in value.items():
        if not isinstance(key, str):
            raise AssemblyError(f"{description} contains a non-string key.")
        result[key] = nested
    return result


def require_list(
    value: object,
    description: str,
) -> list[object]:
    if not isinstance(value, list):
        raise AssemblyError(f"{description} must be a list.")
    return value


def require_string(
    value: object,
    description: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AssemblyError(f"{description} must be a non-empty string.")
    return value.strip()


def optional_string(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def parse_string_list(
    value: object,
    *,
    description: str,
) -> tuple[str, ...]:
    raw = require_list(
        value,
        description,
    )
    result: list[str] = []

    for index, item in enumerate(
        raw,
        start=1,
    ):
        text = require_string(
            item,
            f"{description}[{index}]",
        )
        if text not in result:
            result.append(text)

    return tuple(result)


def same_question(
    *,
    expected: str,
    actual: str,
    description: str,
) -> None:
    if normalized_text(expected) != normalized_text(actual):
        raise AssemblyError(f"{description} question differs from the Phase 14 question.")


def same_corpus_version(
    *,
    expected: str,
    actual: str,
    description: str,
) -> None:
    if expected != actual:
        raise AssemblyError(f"{description} corpus version {actual!r} differs from {expected!r}.")


def parse_query_activation(
    evidence_package: Mapping[str, object],
) -> tuple[
    dict[str, object],
    tuple[str, ...],
    dict[str, float],
]:
    activation = require_mapping(
        evidence_package.get("query_activation"),
        "evidence package query_activation",
    )

    active_raw = require_list(
        activation.get("active_concepts"),
        "query active_concepts",
    )
    active: list[str] = []
    for item in active_raw:
        concept = require_string(
            item,
            "active concept",
        )
        if concept not in CONCEPT_DISPLAY_NAMES:
            raise AssemblyError(f"Unknown active concept {concept!r}.")
        if concept not in active:
            active.append(concept)

    weights_raw = activation.get("calibrated_weights")
    weights: dict[str, float] = {}
    if isinstance(weights_raw, Mapping):
        for concept in active:
            value = optional_float(weights_raw.get(concept))
            if value is None:
                weights[concept] = 0.0
            else:
                weights[concept] = max(
                    0.0,
                    min(1.0, value),
                )
    else:
        weights = dict.fromkeys(active, 0.0)

    return (
        activation,
        tuple(active),
        weights,
    )


def citation_key(
    *,
    chunk_id: str,
    source_id: str,
) -> tuple[str, str]:
    return (chunk_id, source_id)


def build_evidence_index(
    evidence_package: Mapping[str, object],
    *,
    corpus_version: str,
) -> tuple[
    dict[tuple[str, str], dict[str, str]],
    dict[str, str],
]:
    if (
        require_string(
            evidence_package.get("retrieval_mode"),
            "evidence retrieval_mode",
        )
        != "concept_aware"
    ):
        raise AssemblyError("Phase 18 requires the concept-aware Phase 14 evidence package.")

    domains_raw = require_mapping(
        evidence_package.get("domains"),
        "evidence package domains",
    )

    evidence_index: dict[
        tuple[str, str],
        dict[str, str],
    ] = {}
    chunk_domain: dict[str, str] = {}

    for domain in DOMAINS:
        domain_raw = require_mapping(
            domains_raw.get(domain),
            f"{domain} evidence package",
        )
        evidence_raw = require_list(
            domain_raw.get("evidence"),
            f"{domain} evidence",
        )

        for index, item_raw in enumerate(
            evidence_raw,
            start=1,
        ):
            item = require_mapping(
                item_raw,
                f"{domain} evidence {index}",
            )

            item_domain = require_string(
                item.get("domain"),
                f"{domain} evidence domain",
            ).casefold()
            if item_domain != domain:
                raise AssemblyError(
                    f"Phase 14 domain leakage: expected={domain} actual={item_domain}."
                )

            chunk_id = require_string(
                item.get("chunk_id"),
                f"{domain} evidence chunk_id",
            )
            source_id = require_string(
                item.get("source_id"),
                f"{domain} evidence source_id",
            )
            citation = require_string(
                item.get("citation"),
                f"{domain} evidence citation",
            )
            item_corpus_version = require_string(
                item.get("corpus_version"),
                f"{domain} evidence corpus_version",
            )

            same_corpus_version(
                expected=corpus_version,
                actual=item_corpus_version,
                description=(f"{domain}/{chunk_id}"),
            )

            key = citation_key(
                chunk_id=chunk_id,
                source_id=source_id,
            )
            canonical = {
                "chunk_id": chunk_id,
                "source_id": source_id,
                "citation": citation,
                "corpus_version": (corpus_version),
                "domain": domain,
            }

            existing = evidence_index.get(key)
            if existing is not None and existing != canonical:
                raise AssemblyError(f"Conflicting canonical citation for chunk {chunk_id!r}.")

            evidence_index[key] = canonical

            existing_domain = chunk_domain.get(chunk_id)
            if existing_domain is not None and existing_domain != domain:
                raise AssemblyError(f"Chunk {chunk_id!r} appears in multiple domains.")
            chunk_domain[chunk_id] = domain

    return evidence_index, chunk_domain


def parse_claim_citations(
    *,
    claim: Mapping[str, object],
    domain: str,
    claim_id: str,
    corpus_version: str,
    evidence_index: Mapping[
        tuple[str, str],
        Mapping[str, str],
    ],
) -> list[dict[str, str]]:
    citations_raw = require_list(
        claim.get("citations"),
        f"{domain}/{claim_id}/citations",
    )
    if not citations_raw:
        raise AssemblyError(f"{domain}/{claim_id} has no citations.")

    result: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for index, citation_raw in enumerate(
        citations_raw,
        start=1,
    ):
        citation = require_mapping(
            citation_raw,
            (f"{domain}/{claim_id}/citation/{index}"),
        )
        chunk_id = require_string(
            citation.get("chunk_id"),
            "claim citation chunk_id",
        )
        source_id = require_string(
            citation.get("source_id"),
            "claim citation source_id",
        )
        cited_corpus_version = require_string(
            citation.get("corpus_version"),
            "claim citation corpus_version",
        )

        same_corpus_version(
            expected=corpus_version,
            actual=cited_corpus_version,
            description=(f"{domain}/{claim_id}/citation"),
        )

        key = citation_key(
            chunk_id=chunk_id,
            source_id=source_id,
        )
        canonical = evidence_index.get(key)
        if canonical is None:
            raise AssemblyError(
                f"{domain}/{claim_id} citation "
                f"{chunk_id!r} does not resolve to "
                "the Phase 14 active retrieval evidence."
            )

        if canonical["domain"] != domain:
            raise AssemblyError(
                f"{domain}/{claim_id} citation {chunk_id!r} belongs to {canonical['domain']!r}."
            )

        supplied_citation = optional_string(citation.get("citation"))
        if supplied_citation and supplied_citation != canonical["citation"]:
            raise AssemblyError(
                f"{domain}/{claim_id} citation text does not match the canonical Phase 14 citation."
            )

        if key not in seen:
            seen.add(key)
            result.append(dict(canonical))

    return result


def validate_domain_response(
    *,
    domain: str,
    raw: Mapping[str, object],
    corpus_version: str,
    evidence_index: Mapping[
        tuple[str, str],
        Mapping[str, str],
    ],
) -> dict[str, object]:
    response_domain = require_string(
        raw.get("domain"),
        f"{domain} response domain",
    ).casefold()
    if response_domain != domain:
        raise AssemblyError(f"{domain} response declares domain {response_domain!r}.")

    response_corpus_version = require_string(
        raw.get("corpus_version"),
        f"{domain} corpus_version",
    )
    same_corpus_version(
        expected=corpus_version,
        actual=response_corpus_version,
        description=f"Phase 15 {domain}",
    )

    validation = require_mapping(
        raw.get("validation"),
        f"{domain} validation",
    )
    if validation.get("passed") is not True:
        raise AssemblyError(f"Phase 15 {domain} validation did not pass.")

    leakage = require_mapping(
        raw.get("domain_leakage"),
        f"{domain} domain_leakage",
    )
    if leakage.get("passed") is not True:
        raise AssemblyError(f"Phase 15 {domain} failed domain-leakage validation.")

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
            raise AssemblyError(f"Phase 15 {domain} grounding failed: {field_name}.")

    summary = require_string(
        raw.get("summary"),
        f"{domain} summary",
    )
    claims_raw = require_list(
        raw.get("claims"),
        f"{domain} claims",
    )

    claims: list[dict[str, object]] = []
    seen_claim_ids: set[str] = set()

    for index, claim_raw in enumerate(
        claims_raw,
        start=1,
    ):
        claim = require_mapping(
            claim_raw,
            f"{domain} claim {index}",
        )
        claim_id = require_string(
            claim.get("claim_id"),
            f"{domain} claim_id",
        )
        if claim_id in seen_claim_ids:
            raise AssemblyError(f"{domain} duplicate claim_id {claim_id!r}.")
        seen_claim_ids.add(claim_id)

        text = require_string(
            claim.get("text"),
            f"{domain}/{claim_id}/text",
        )
        concepts = parse_string_list(
            claim.get("concepts_covered"),
            description=(f"{domain}/{claim_id}/concepts_covered"),
        )
        for concept in concepts:
            if concept not in CONCEPT_DISPLAY_NAMES:
                raise AssemblyError(f"{domain}/{claim_id} has unknown concept {concept!r}.")

        citations = parse_claim_citations(
            claim=claim,
            domain=domain,
            claim_id=claim_id,
            corpus_version=corpus_version,
            evidence_index=evidence_index,
        )

        claims.append(
            {
                "claim_id": claim_id,
                "claim_ref": (f"{domain}:{claim_id}"),
                "text": text,
                "concepts_covered": list(concepts),
                "citations": citations,
            }
        )

    limitations_raw = raw.get("limitations")
    limitations: list[str] = []
    if isinstance(limitations_raw, list):
        limitations = [
            item.strip() for item in limitations_raw if isinstance(item, str) and item.strip()
        ]

    unsupported_raw = raw.get("unsupported_aspects")
    unsupported_aspects: list[str] = []
    if isinstance(unsupported_raw, list):
        unsupported_aspects = [
            item.strip() for item in unsupported_raw if isinstance(item, str) and item.strip()
        ]

    return {
        "domain": domain,
        "display_name": (DOMAIN_DISPLAY_NAMES[domain]),
        "summary": summary,
        "claims": claims,
        "limitations": limitations,
        "unsupported_aspects": (unsupported_aspects),
    }


def parse_domain_responses(
    document: Mapping[str, object],
    *,
    question: str,
    corpus_version: str,
    evidence_index: Mapping[
        tuple[str, str],
        Mapping[str, str],
    ],
) -> tuple[
    dict[str, object],
    dict[str, dict[str, object]],
]:
    same_question(
        expected=question,
        actual=require_string(
            document.get("question"),
            "Phase 15 question",
        ),
        description="Phase 15",
    )
    same_corpus_version(
        expected=corpus_version,
        actual=require_string(
            document.get("corpus_version"),
            "Phase 15 corpus_version",
        ),
        description="Phase 15",
    )

    query_activation = require_mapping(
        document.get("query_activation"),
        "Phase 15 query_activation",
    )
    domains_raw = require_mapping(
        document.get("domains"),
        "Phase 15 domains",
    )

    domains: dict[
        str,
        dict[str, object],
    ] = {}
    for domain in DOMAINS:
        domains[domain] = validate_domain_response(
            domain=domain,
            raw=require_mapping(
                domains_raw.get(domain),
                f"{domain} response",
            ),
            corpus_version=corpus_version,
            evidence_index=evidence_index,
        )

    return query_activation, domains


def build_claim_index(
    domains: Mapping[
        str,
        Mapping[str, object],
    ],
) -> dict[str, dict[str, object]]:
    result: dict[
        str,
        dict[str, object],
    ] = {}

    for domain in DOMAINS:
        claims_raw = require_list(
            domains[domain].get("claims"),
            f"{domain} canonical claims",
        )
        for claim_raw in claims_raw:
            claim = require_mapping(
                claim_raw,
                f"{domain} canonical claim",
            )
            ref = require_string(
                claim.get("claim_ref"),
                "claim_ref",
            )
            if ref in result:
                raise AssemblyError(f"Duplicate claim reference {ref!r}.")
            result[ref] = claim

    return result


def validate_synthesis_citation(
    citation_raw: object,
    *,
    corpus_version: str,
    evidence_index: Mapping[
        tuple[str, str],
        Mapping[str, str],
    ],
) -> dict[str, str]:
    citation = require_mapping(
        citation_raw,
        "synthesis citation",
    )
    chunk_id = require_string(
        citation.get("chunk_id"),
        "synthesis citation chunk_id",
    )
    source_id = require_string(
        citation.get("source_id"),
        "synthesis citation source_id",
    )
    cited_corpus_version = require_string(
        citation.get("corpus_version"),
        "synthesis citation corpus_version",
    )
    same_corpus_version(
        expected=corpus_version,
        actual=cited_corpus_version,
        description="synthesis citation",
    )

    key = citation_key(
        chunk_id=chunk_id,
        source_id=source_id,
    )
    canonical = evidence_index.get(key)
    if canonical is None:
        raise AssemblyError(
            f"Synthesis citation {chunk_id!r} "
            "does not resolve to Phase 14 "
            "active retrieval evidence."
        )

    supplied_text = optional_string(citation.get("citation"))
    if supplied_text and supplied_text != canonical["citation"]:
        raise AssemblyError("Synthesis citation text does not match Phase 14 canonical citation.")

    return dict(canonical)


def contains_atman_purusha_false_equivalence(
    *,
    category: str,
    explanation: str,
) -> bool:
    if category not in {
        "substantive_agreement",
        "surface_similarity",
    }:
        return False

    normalized = explanation.casefold()
    has_atman = any(term in normalized for term in ATMAN_TERMS)
    has_purusha = any(term in normalized for term in PURUSHA_TERMS)
    has_equivalence = any(term in normalized for term in EQUIVALENCE_TERMS)

    return has_atman and has_purusha and has_equivalence


def parse_synthesis_comparison(
    *,
    raw: object,
    index: int,
    corpus_version: str,
    claim_index: Mapping[
        str,
        Mapping[str, object],
    ],
    evidence_index: Mapping[
        tuple[str, str],
        Mapping[str, str],
    ],
) -> dict[str, object]:
    comparison = require_mapping(
        raw,
        f"synthesis comparison {index}",
    )
    comparison_id = require_string(
        comparison.get("comparison_id"),
        f"comparison {index} id",
    )
    category = require_string(
        comparison.get("category"),
        f"{comparison_id} category",
    )
    if category not in ALLOWED_SYNTHESIS_CATEGORIES:
        raise AssemblyError(f"{comparison_id} has invalid category {category!r}.")

    explanation = require_string(
        comparison.get("explanation"),
        f"{comparison_id} explanation",
    )

    domains = parse_string_list(
        comparison.get("domains"),
        description=(f"{comparison_id} domains"),
    )
    for domain in domains:
        if domain not in DOMAINS:
            raise AssemblyError(f"{comparison_id} has unknown domain {domain!r}.")

    concepts = parse_string_list(
        comparison.get("concepts_covered"),
        description=(f"{comparison_id} concepts"),
    )
    for concept in concepts:
        if concept not in CONCEPT_DISPLAY_NAMES:
            raise AssemblyError(f"{comparison_id} has unknown concept {concept!r}.")

    claim_refs = parse_string_list(
        comparison.get("claim_refs"),
        description=(f"{comparison_id} claim_refs"),
    )
    for claim_ref in claim_refs:
        claim = claim_index.get(claim_ref)
        if claim is None:
            raise AssemblyError(f"{comparison_id} references unknown claim {claim_ref!r}.")

        claim_domain = claim_ref.split(
            ":",
            1,
        )[0]
        if claim_domain not in domains:
            raise AssemblyError(
                f"{comparison_id} claim {claim_ref!r} is outside the comparison domains."
            )

    citations_raw = require_list(
        comparison.get("citations"),
        f"{comparison_id} citations",
    )
    if category != "insufficient_corpus_coverage" and not citations_raw:
        raise AssemblyError(f"{comparison_id} has no citations.")

    citations: list[dict[str, str]] = []
    seen_citations: set[tuple[str, str]] = set()
    for citation_raw in citations_raw:
        canonical = validate_synthesis_citation(
            citation_raw,
            corpus_version=corpus_version,
            evidence_index=evidence_index,
        )
        key = citation_key(
            chunk_id=canonical["chunk_id"],
            source_id=canonical["source_id"],
        )
        if key in seen_citations:
            continue
        seen_citations.add(key)
        citations.append(canonical)

    citation_domains = {item["domain"] for item in citations}
    if not citation_domains.issubset(set(domains)):
        raise AssemblyError(
            f"{comparison_id} contains citation from a domain outside the comparison."
        )

    if contains_atman_purusha_false_equivalence(
        category=category,
        explanation=explanation,
    ):
        raise AssemblyError(f"{comparison_id} risks an unsupported Atman/Purusha equivalence.")

    limitations_raw = comparison.get("limitations")
    limitations: list[dict[str, str]] = []
    if isinstance(limitations_raw, list):
        for limitation_raw in limitations_raw:
            if not isinstance(
                limitation_raw,
                Mapping,
            ):
                continue
            domain = optional_string(limitation_raw.get("domain"))
            text = optional_string(limitation_raw.get("text"))
            ref = optional_string(limitation_raw.get("limitation_ref"))
            if domain in DOMAINS and text:
                limitations.append(
                    {
                        "domain": domain,
                        "limitation_ref": ref,
                        "text": text,
                    }
                )

    return {
        "comparison_id": comparison_id,
        "category": category,
        "domains": list(domains),
        "concepts_covered": list(concepts),
        "claim_refs": list(claim_refs),
        "explanation": explanation,
        "citations": citations,
        "limitations": limitations,
    }


def parse_synthesis(
    document: Mapping[str, object],
    *,
    question: str,
    corpus_version: str,
    claim_index: Mapping[
        str,
        Mapping[str, object],
    ],
    evidence_index: Mapping[
        tuple[str, str],
        Mapping[str, str],
    ],
) -> dict[str, object]:
    same_question(
        expected=question,
        actual=require_string(
            document.get("question"),
            "Phase 16 question",
        ),
        description="Phase 16",
    )
    same_corpus_version(
        expected=corpus_version,
        actual=require_string(
            document.get("corpus_version"),
            "Phase 16 corpus_version",
        ),
        description="Phase 16",
    )

    validation = require_mapping(
        document.get("validation"),
        "Phase 16 validation",
    )
    if validation.get("passed") is not True:
        raise AssemblyError("Phase 16 synthesis validation did not pass.")

    synthesis_version = require_string(
        document.get("synthesis_version"),
        "Phase 16 synthesis_version",
    )
    prompt_version = require_string(
        document.get("prompt_version"),
        "Phase 16 prompt_version",
    )
    summary = require_string(
        document.get("synthesis_summary"),
        "Phase 16 synthesis_summary",
    )
    non_conclusion = require_string(
        document.get("non_conclusion"),
        "Phase 16 non_conclusion",
    )

    comparisons_raw = document.get("pairwise_comparisons")
    if not isinstance(
        comparisons_raw,
        list,
    ):
        comparisons_raw = require_list(
            document.get("comparisons"),
            "Phase 16 comparisons",
        )

    comparisons = [
        parse_synthesis_comparison(
            raw=raw,
            index=index,
            corpus_version=corpus_version,
            claim_index=claim_index,
            evidence_index=evidence_index,
        )
        for index, raw in enumerate(
            comparisons_raw,
            start=1,
        )
    ]

    key_tensions = [
        comparison for comparison in comparisons if comparison["category"] == "direct_tension"
    ]
    non_equivalences = [
        comparison for comparison in comparisons if comparison["category"] == "non_equivalence"
    ]
    insufficient = [
        comparison
        for comparison in comparisons
        if comparison["category"] == "insufficient_corpus_coverage"
    ]

    overview_raw = document.get("three_way_overview")
    overview: object
    overview = overview_raw if isinstance(overview_raw, (str, list, dict)) else summary

    return {
        "synthesis_version": (synthesis_version),
        "prompt_version": prompt_version,
        "summary": summary,
        "three_way_overview": overview,
        "comparisons": comparisons,
        "key_tensions": key_tensions,
        "non_equivalences": (non_equivalences),
        "insufficient_corpus_coverage": (insufficient),
        "non_conclusion": non_conclusion,
    }


def parse_coverage(
    document: Mapping[str, object],
    *,
    question: str,
    corpus_version: str,
    active_concepts: tuple[str, ...],
) -> dict[str, object]:
    same_question(
        expected=question,
        actual=require_string(
            document.get("question"),
            "Phase 17 question",
        ),
        description="Phase 17",
    )
    same_corpus_version(
        expected=corpus_version,
        actual=require_string(
            document.get("corpus_version"),
            "Phase 17 corpus_version",
        ),
        description="Phase 17",
    )

    exit_gate = require_mapping(
        document.get("exit_gate"),
        "Phase 17 exit_gate",
    )
    if exit_gate.get("passed") is not True:
        raise AssemblyError("Phase 17 exit gate did not pass.")

    status = require_string(
        document.get("coverage_status"),
        "coverage_status",
    )
    if status not in ALLOWED_COVERAGE_STATUSES:
        raise AssemblyError(f"Invalid coverage status {status!r}.")

    score = optional_float(document.get("coverage_score"))
    if score is None:
        raise AssemblyError("Phase 17 coverage_score is missing.")
    if not 0.0 <= score <= 100.0:
        raise AssemblyError("coverage_score must be 0-100.")

    reason = require_string(
        document.get("coverage_reason"),
        "coverage_reason",
    )

    concept_coverage_raw = require_list(
        document.get("concept_coverage"),
        "concept_coverage",
    )
    concept_coverage: list[dict[str, object]] = []
    seen_concepts: set[str] = set()

    for index, item_raw in enumerate(
        concept_coverage_raw,
        start=1,
    ):
        item = require_mapping(
            item_raw,
            f"concept coverage {index}",
        )
        concept = require_string(
            item.get("concept"),
            "coverage concept",
        )
        if concept not in active_concepts:
            raise AssemblyError(f"Coverage contains inactive concept {concept!r}.")
        if concept in seen_concepts:
            raise AssemblyError(f"Duplicate concept coverage {concept!r}.")
        seen_concepts.add(concept)

        concept_status = require_string(
            item.get("status"),
            f"{concept} status",
        )
        if concept_status not in ALLOWED_CONCEPT_STATUSES:
            raise AssemblyError(f"{concept} has invalid status {concept_status!r}.")

        concept_score = optional_float(item.get("coverage_score"))
        if concept_score is None:
            raise AssemblyError(f"{concept} coverage_score missing.")

        concept_coverage.append(dict(item))

    if set(seen_concepts) != set(active_concepts):
        raise AssemblyError("Phase 17 concept coverage does not match active concepts.")

    supported = tuple(
        require_string(
            item,
            "supported concept",
        )
        for item in require_list(
            document.get("supported_concepts"),
            "supported_concepts",
        )
    )
    partial_raw = document.get("partially_supported_concepts")
    partial: tuple[str, ...] = ()
    if isinstance(partial_raw, list):
        partial = tuple(
            require_string(
                item,
                "partially supported concept",
            )
            for item in partial_raw
        )
    unsupported = tuple(
        require_string(
            item,
            "unsupported concept",
        )
        for item in require_list(
            document.get("unsupported_concepts"),
            "unsupported_concepts",
        )
    )

    derived_supported = {
        require_string(
            item.get("concept"),
            "concept",
        )
        for item in concept_coverage
        if item.get("status") == "Supported"
    }
    derived_partial = {
        require_string(
            item.get("concept"),
            "concept",
        )
        for item in concept_coverage
        if item.get("status") == "Partially Supported"
    }
    derived_unsupported = {
        require_string(
            item.get("concept"),
            "concept",
        )
        for item in concept_coverage
        if item.get("status") == "Unsupported"
    }

    if set(supported) != derived_supported:
        raise AssemblyError("Phase 17 supported_concepts does not match concept statuses.")
    if set(partial) != derived_partial:
        raise AssemblyError(
            "Phase 17 partially_supported_concepts does not match concept statuses."
        )
    if set(unsupported) != derived_unsupported:
        raise AssemblyError("Phase 17 unsupported_concepts does not match concept statuses.")

    policy = require_mapping(
        document.get("response_policy"),
        "Phase 17 response_policy",
    )

    if status == "Supported":
        if derived_partial or derived_unsupported:
            raise AssemblyError(
                "Coverage status says Supported but not all concepts are Supported."
            )
        if policy.get("corpus_answer_allowed") is not True:
            raise AssemblyError("Supported coverage must allow a corpus answer.")

    elif status == "Partially Supported":
        if not (derived_partial or derived_unsupported):
            raise AssemblyError("Partially Supported status has no partial/unsupported concept.")
        if policy.get("corpus_answer_allowed") is not True:
            raise AssemblyError(
                "Partially Supported coverage must allow supported corpus components."
            )

    else:
        if policy.get("corpus_answer_allowed") is not False:
            raise AssemblyError("Out of Corpus must forbid a corpus-grounded answer.")

    if policy.get("general_knowledge_must_be_labeled") is not True:
        raise AssemblyError("Phase 17 must require labeling of general-knowledge fallback.")
    if policy.get("general_knowledge_must_not_use_corpus_citations") is not True:
        raise AssemblyError("Phase 17 must forbid WTH citations for general-knowledge fallback.")

    covered_domains = parse_string_list(
        document.get("covered_domains"),
        description="covered_domains",
    )
    missing_domains = parse_string_list(
        document.get("missing_domains"),
        description="missing_domains",
    )

    hard_overrides_raw = document.get("hard_overrides", [])
    hard_overrides: list[object] = (
        list(hard_overrides_raw) if isinstance(hard_overrides_raw, list) else []
    )

    return {
        "coverage_version": require_string(
            document.get("coverage_version"),
            "coverage_version",
        ),
        "coverage_status": status,
        "coverage_score": round(score, 2),
        "coverage_reason": reason,
        "concept_coverage": (concept_coverage),
        "supported_concepts": list(supported),
        "partially_supported_concepts": (list(partial)),
        "unsupported_concepts": list(unsupported),
        "covered_domains": list(covered_domains),
        "missing_domains": list(missing_domains),
        "response_policy": policy,
        "hard_overrides": hard_overrides,
    }


def interpretation_text(
    *,
    question: str,
    active_concepts: tuple[str, ...],
) -> str:
    if not active_concepts:
        return "The question does not map cleanly to the current Phase 1 concept set."

    concept_names = [CONCEPT_DISPLAY_NAMES[concept].casefold() for concept in active_concepts]
    if len(concept_names) == 1:
        concept_phrase = concept_names[0]
    elif len(concept_names) == 2:
        concept_phrase = f"{concept_names[0]} and {concept_names[1]}"
    else:
        concept_phrase = ", ".join(concept_names[:-1]) + f", and {concept_names[-1]}"

    return (
        "I interpret the question as asking how "
        f"{concept_phrase} relate, and how that "
        "relationship is described differently "
        "by Science, Advaita Vedanta, and Samkhya."
    )


def citation_registry(
    *,
    domains: Mapping[
        str,
        Mapping[str, object],
    ],
    synthesis: Mapping[str, object],
) -> tuple[
    dict[tuple[str, str], str],
    list[dict[str, str]],
]:
    canonical_by_key: dict[
        tuple[str, str],
        dict[str, str],
    ] = {}

    for domain in DOMAINS:
        claims = require_list(
            domains[domain].get("claims"),
            f"{domain} claims",
        )
        for claim_raw in claims:
            claim = require_mapping(
                claim_raw,
                f"{domain} claim",
            )
            citations = require_list(
                claim.get("citations"),
                "claim citations",
            )
            for citation_raw in citations:
                citation = require_mapping(
                    citation_raw,
                    "canonical claim citation",
                )
                key = citation_key(
                    chunk_id=require_string(
                        citation.get("chunk_id"),
                        "citation chunk_id",
                    ),
                    source_id=require_string(
                        citation.get("source_id"),
                        "citation source_id",
                    ),
                )
                canonical_by_key[key] = {
                    key_name: require_string(
                        citation.get(key_name),
                        f"citation {key_name}",
                    )
                    for key_name in (
                        "chunk_id",
                        "source_id",
                        "citation",
                        "corpus_version",
                        "domain",
                    )
                }

    comparisons = require_list(
        synthesis.get("comparisons"),
        "synthesis comparisons",
    )
    for comparison_raw in comparisons:
        comparison = require_mapping(
            comparison_raw,
            "synthesis comparison",
        )
        citations = require_list(
            comparison.get("citations"),
            "synthesis citations",
        )
        for citation_raw in citations:
            citation = require_mapping(
                citation_raw,
                "canonical synthesis citation",
            )
            key = citation_key(
                chunk_id=require_string(
                    citation.get("chunk_id"),
                    "citation chunk_id",
                ),
                source_id=require_string(
                    citation.get("source_id"),
                    "citation source_id",
                ),
            )
            canonical_by_key[key] = {
                key_name: require_string(
                    citation.get(key_name),
                    f"citation {key_name}",
                )
                for key_name in (
                    "chunk_id",
                    "source_id",
                    "citation",
                    "corpus_version",
                    "domain",
                )
            }

    ordered = sorted(
        canonical_by_key.values(),
        key=lambda item: (
            item["domain"],
            item["source_id"],
            item["chunk_id"],
        ),
    )

    key_to_ref: dict[
        tuple[str, str],
        str,
    ] = {}
    registry: list[dict[str, str]] = []

    for index, citation_entry in enumerate(
        ordered,
        start=1,
    ):
        ref = f"C{index}"
        key = citation_key(
            chunk_id=citation_entry["chunk_id"],
            source_id=citation_entry["source_id"],
        )
        key_to_ref[key] = ref
        registry.append(
            {
                "citation_ref": ref,
                **citation_entry,
            }
        )

    return key_to_ref, registry


def attach_citation_refs(
    *,
    domains: dict[
        str,
        dict[str, object],
    ],
    synthesis: dict[str, object],
    key_to_ref: Mapping[
        tuple[str, str],
        str,
    ],
) -> None:
    for domain in DOMAINS:
        claims = require_list(
            domains[domain].get("claims"),
            f"{domain} claims",
        )
        for claim_raw in claims:
            if not isinstance(claim_raw, dict):
                raise AssemblyError(f"{domain} canonical claim must be mutable.")
            claim = claim_raw
            refs: list[str] = []
            citations = require_list(
                claim.get("citations"),
                "claim citations",
            )
            for citation_raw in citations:
                citation = require_mapping(
                    citation_raw,
                    "claim citation",
                )
                key = citation_key(
                    chunk_id=require_string(
                        citation.get("chunk_id"),
                        "citation chunk_id",
                    ),
                    source_id=require_string(
                        citation.get("source_id"),
                        "citation source_id",
                    ),
                )
                ref = key_to_ref.get(key)
                if ref is None:
                    raise AssemblyError("Internal citation registry is incomplete.")
                refs.append(ref)
            claim["citation_refs"] = refs

    comparisons = require_list(
        synthesis.get("comparisons"),
        "synthesis comparisons",
    )
    for comparison_raw in comparisons:
        if not isinstance(comparison_raw, dict):
            raise AssemblyError("Canonical synthesis comparison must be mutable.")
        comparison = comparison_raw
        comparison_refs: list[str] = []
        citations = require_list(
            comparison.get("citations"),
            "comparison citations",
        )
        for citation_raw in citations:
            citation = require_mapping(
                citation_raw,
                "comparison citation",
            )
            key = citation_key(
                chunk_id=require_string(
                    citation.get("chunk_id"),
                    "citation chunk_id",
                ),
                source_id=require_string(
                    citation.get("source_id"),
                    "citation source_id",
                ),
            )
            ref = key_to_ref.get(key)
            if ref is None:
                raise AssemblyError("Internal synthesis citation registry is incomplete.")
            comparison_refs.append(ref)
        comparison["citation_refs"] = comparison_refs


def format_overview(
    value: object,
) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        return " ".join(parts)
    if isinstance(value, Mapping):
        mapping_parts: list[str] = []
        for nested in value.values():
            if isinstance(nested, str) and nested.strip():
                mapping_parts.append(nested.strip())
        return " ".join(mapping_parts)
    return ""


def final_sections(
    *,
    question: str,
    active_concepts: tuple[str, ...],
    weights: Mapping[str, float],
    domains: Mapping[
        str,
        Mapping[str, object],
    ],
    synthesis: Mapping[str, object],
    coverage: Mapping[str, object],
) -> dict[str, object]:
    concept_status_map: dict[
        str,
        dict[str, object],
    ] = {}
    concept_coverage = require_list(
        coverage.get("concept_coverage"),
        "concept_coverage",
    )
    for item_raw in concept_coverage:
        item = require_mapping(
            item_raw,
            "concept coverage item",
        )
        concept = require_string(
            item.get("concept"),
            "concept",
        )
        concept_status_map[concept] = item

    activated: list[dict[str, object]] = []
    for concept in active_concepts:
        item = concept_status_map[concept]
        activated.append(
            {
                "concept": concept,
                "display_name": (CONCEPT_DISPLAY_NAMES[concept]),
                "activation_weight": round(
                    weights.get(
                        concept,
                        0.0,
                    ),
                    6,
                ),
                "coverage_status": (
                    require_string(
                        item.get("status"),
                        f"{concept} status",
                    )
                ),
                "coverage_score": (optional_float(item.get("coverage_score"))),
            }
        )

    policy = require_mapping(
        coverage.get("response_policy"),
        "response_policy",
    )
    corpus_answer_allowed = policy.get("corpus_answer_allowed") is True

    domain_sections: dict[
        str,
        object,
    ] = {}
    if corpus_answer_allowed:
        domain_sections = {domain: dict(domains[domain]) for domain in DOMAINS}

    comparisons = require_list(
        synthesis.get("comparisons"),
        "synthesis comparisons",
    )
    tensions = [
        dict(
            require_mapping(
                item,
                "key tension",
            )
        )
        for item in comparisons
        if (isinstance(item, Mapping) and item.get("category") == "direct_tension")
    ]
    non_equivalences = [
        dict(
            require_mapping(
                item,
                "non-equivalence",
            )
        )
        for item in comparisons
        if (isinstance(item, Mapping) and item.get("category") == "non_equivalence")
    ]

    synthesis_section: dict[
        str,
        object,
    ] = {
        "summary": (
            require_string(
                synthesis.get("summary"),
                "synthesis summary",
            )
            if corpus_answer_allowed
            else ""
        ),
        "three_way_overview": (
            format_overview(synthesis.get("three_way_overview")) if corpus_answer_allowed else ""
        ),
        "comparisons": (list(comparisons) if corpus_answer_allowed else []),
        "non_conclusion": (
            require_string(
                synthesis.get("non_conclusion"),
                "synthesis non_conclusion",
            )
            if corpus_answer_allowed
            else ""
        ),
    }

    fallback_allowed = policy.get("general_knowledge_fallback_allowed") is True

    if fallback_allowed:
        fallback_instruction = (
            "If the application chooses to provide a "
            "general-knowledge answer, generate it in a "
            "separate clearly labeled section. Do not "
            "present it as reviewed-corpus support and "
            "do not attach WTH corpus citations."
        )
    else:
        fallback_instruction = (
            "General-knowledge fallback is not allowed "
            "for this response."
        )

    fallback: dict[str, object] = {
        "allowed": fallback_allowed,
        "generated_in_phase18": False,
        "instruction": fallback_instruction,
        "must_be_clearly_labeled": True,
        "may_use_wth_corpus_citations": False,
    }

    return {
        "interpretation": interpretation_text(
            question=question,
            active_concepts=active_concepts,
        ),
        "activated_concepts": activated,
        "domain_perspectives": (domain_sections),
        "comparative_synthesis": (synthesis_section),
        "key_tensions": (tensions if corpus_answer_allowed else []),
        "non_equivalences": (non_equivalences if corpus_answer_allowed else []),
        "coverage": {
            key: coverage[key]
            for key in (
                "coverage_status",
                "coverage_score",
                "coverage_reason",
                "supported_concepts",
                "partially_supported_concepts",
                "unsupported_concepts",
                "covered_domains",
                "missing_domains",
                "hard_overrides",
            )
            if key in coverage
        },
        "general_knowledge_fallback": (fallback),
    }


def markdown_response(
    *,
    sections: Mapping[str, object],
    citation_registry_rows: list[dict[str, str]],
) -> str:
    lines: list[str] = []

    lines.append("## Interpretation of the question")
    lines.append("")
    lines.append(
        require_string(
            sections.get("interpretation"),
            "interpretation",
        )
    )
    lines.append("")

    lines.append("## Activated concepts")
    lines.append("")
    activated = require_list(
        sections.get("activated_concepts"),
        "activated_concepts",
    )
    if not activated:
        lines.append("- No Phase 1 concept was activated.")
    else:
        for item_raw in activated:
            item = require_mapping(
                item_raw,
                "activated concept",
            )
            display_name = require_string(
                item.get("display_name"),
                "concept display name",
            )
            weight = optional_float(item.get("activation_weight"))
            status = require_string(
                item.get("coverage_status"),
                "concept coverage status",
            )
            concept_score = optional_float(item.get("coverage_score"))
            weight_text = f"{weight:.3f}" if weight is not None else "n/a"
            score_text = f"{concept_score:.1f}/100" if concept_score is not None else "n/a"
            lines.append(
                f"- **{display_name}** — activation {weight_text}; coverage {status} ({score_text})"
            )
    lines.append("")

    domain_sections = require_mapping(
        sections.get("domain_perspectives"),
        "domain_perspectives",
    )
    for domain in DOMAINS:
        lines.append(f"## {DOMAIN_DISPLAY_NAMES[domain]} perspective")
        lines.append("")
        raw = domain_sections.get(domain)
        if not isinstance(raw, Mapping):
            lines.append(
                "The reviewed corpus is not being used "
                "to provide a domain answer for this "
                "question."
            )
            lines.append("")
            continue

        perspective = require_mapping(
            raw,
            f"{domain} perspective",
        )
        lines.append(
            require_string(
                perspective.get("summary"),
                f"{domain} summary",
            )
        )
        lines.append("")

        claims = require_list(
            perspective.get("claims"),
            f"{domain} claims",
        )
        for claim_raw in claims:
            claim = require_mapping(
                claim_raw,
                f"{domain} claim",
            )
            text = require_string(
                claim.get("text"),
                "claim text",
            )
            refs = require_list(
                claim.get("citation_refs"),
                "claim citation_refs",
            )
            ref_text = " ".join(f"[{require_string(ref, 'citation ref')}]" for ref in refs)
            lines.append(f"- {text} {ref_text}".rstrip())

        limitations = perspective.get("limitations")
        if (
            isinstance(
                limitations,
                list,
            )
            and limitations
        ):
            lines.append("")
            lines.append("**Limitations:**")
            for item in limitations:
                if isinstance(item, str) and item.strip():
                    lines.append(f"- {item.strip()}")
        lines.append("")

    lines.append("## Comparative synthesis")
    lines.append("")
    synthesis = require_mapping(
        sections.get("comparative_synthesis"),
        "comparative_synthesis",
    )
    synthesis_summary = optional_string(synthesis.get("summary"))
    if synthesis_summary:
        lines.append(synthesis_summary)
        lines.append("")

    overview = optional_string(synthesis.get("three_way_overview"))
    if overview and overview != synthesis_summary:
        lines.append(overview)
        lines.append("")

    comparisons = synthesis.get("comparisons")
    if isinstance(comparisons, list):
        for item_raw in comparisons:
            if not isinstance(
                item_raw,
                Mapping,
            ):
                continue
            item = require_mapping(
                item_raw,
                "comparison",
            )
            explanation = optional_string(item.get("explanation"))
            category = optional_string(item.get("category"))
            comparison_refs_raw = item.get("citation_refs")
            ref_text = ""
            if isinstance(comparison_refs_raw, list):
                ref_text = " ".join(
                    f"[{ref}]" for ref in comparison_refs_raw if isinstance(ref, str) and ref
                )
            if explanation:
                category_label = (
                    category.replace(
                        "_",
                        " ",
                    ).title()
                    if category
                    else "Comparison"
                )
                lines.append(f"- **{category_label}:** {explanation} {ref_text}".rstrip())
    lines.append("")

    lines.append("## Key tensions and non-equivalences")
    lines.append("")
    tensions = require_list(
        sections.get("key_tensions"),
        "key_tensions",
    )
    non_equivalences = require_list(
        sections.get("non_equivalences"),
        "non_equivalences",
    )

    if not tensions and not non_equivalences:
        lines.append(
            "No separately validated direct tension "
            "or non-equivalence is asserted beyond "
            "the comparative synthesis above."
        )
    else:
        for label, items in (
            ("Tension", tensions),
            (
                "Non-equivalence",
                non_equivalences,
            ),
        ):
            for item_raw in items:
                item = require_mapping(
                    item_raw,
                    label,
                )
                explanation = require_string(
                    item.get("explanation"),
                    f"{label} explanation",
                )
                refs = require_list(
                    item.get(
                        "citation_refs",
                        [],
                    ),
                    f"{label} citation_refs",
                )
                ref_text = " ".join(f"[{require_string(ref, 'citation ref')}]" for ref in refs)
                lines.append(f"- **{label}:** {explanation} {ref_text}".rstrip())
    lines.append("")

    lines.append("## Coverage classification")
    lines.append("")
    coverage = require_mapping(
        sections.get("coverage"),
        "coverage",
    )
    coverage_status = require_string(
        coverage.get("coverage_status"),
        "coverage status",
    )
    coverage_score = optional_float(coverage.get("coverage_score"))
    coverage_reason = require_string(
        coverage.get("coverage_reason"),
        "coverage reason",
    )
    score_text = f" — {coverage_score:.1f}/100" if coverage_score is not None else ""
    lines.append(f"**{coverage_status}{score_text}**")
    lines.append("")
    lines.append(coverage_reason)
    lines.append("")

    lines.append("## Claim-level citations")
    lines.append("")
    if not citation_registry_rows:
        lines.append("No WTH corpus citations are attached to this response.")
    else:
        for citation in citation_registry_rows:
            lines.append(
                f"- [{citation['citation_ref']}] {citation['citation']} — `{citation['chunk_id']}`"
            )
    lines.append("")

    return "\n".join(lines)


