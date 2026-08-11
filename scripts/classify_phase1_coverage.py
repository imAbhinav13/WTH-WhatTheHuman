from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

LOGGER = logging.getLogger("wth.phase1.classify_coverage")

SCRIPT_VERSION: Final = "1.1.0"
COVERAGE_VERSION: Final = "phase1-coverage-classification-v2"

DOMAINS: Final = ("science", "advaita", "samkhya")
CONCEPTS: Final = ("consciousness", "self_identity", "reality_appearance")
CoverageStatus = Literal["Supported", "Partially Supported", "Out of Corpus"]

DEFAULT_EVIDENCE_PACKAGE: Final = Path("artifacts/phase1/retrieval/evidence_package.json")
DEFAULT_RETRIEVAL_MANIFEST: Final = Path("artifacts/phase1/retrieval/retrieval_manifest.json")
DEFAULT_DOMAIN_RESPONSES: Final = Path("artifacts/phase1/generation/domain_responses.json")
DEFAULT_GENERATION_MANIFEST: Final = Path("artifacts/phase1/generation/generation_manifest.json")
DEFAULT_SYNTHESIS: Final = Path("artifacts/phase1/synthesis/synthesis.json")
DEFAULT_SYNTHESIS_MANIFEST: Final = Path("artifacts/phase1/synthesis/synthesis_manifest.json")
DEFAULT_OUTPUT_DIRECTORY: Final = Path("artifacts/phase1/coverage")

SUPPORTED_SCORE_THRESHOLD: Final = 70.0
PARTIAL_SCORE_THRESHOLD: Final = 40.0

SCORE_WEIGHT_ACTIVATION: Final = 25.0
SCORE_WEIGHT_EVIDENCE: Final = 20.0
SCORE_WEIGHT_DOMAIN_COVERAGE: Final = 20.0
SCORE_WEIGHT_CITATION_QUALITY: Final = 15.0
SCORE_WEIGHT_RETRIEVAL_CONFIDENCE: Final = 10.0
SCORE_WEIGHT_UNSUPPORTED_COMPONENT: Final = 10.0

TARGET_EVIDENCE_PER_CONCEPT: Final = 9
EXPLICIT_UNSUPPORTED_PENALTY: Final = 6.0
INSUFFICIENT_COMPARISON_PENALTY: Final = 4.0

ConceptStatus = Literal[
    "Supported",
    "Partially Supported",
    "Unsupported",
]

CONCEPT_HINT_TERMS: Final = {
    "consciousness": ("consciousness", "conscious", "awareness", "experience", "subjectivity"),
    "self_identity": ("self", "identity", "ego", "subject", "body ownership", "bodily"),
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

RETRIEVAL_SCORE_FIELDS: Final = (
    "final_score",
    "retrieval_score",
    "score",
    "rerank_score",
    "vector_similarity",
    "similarity",
)

OUT_OF_CORPUS_INTRO: Final = (
    "The reviewed WTH corpus does not currently contain enough evidence "
    "to answer this question reliably."
)
PARTIAL_SUPPORT_INTRO: Final = (
    "The reviewed WTH corpus supports part of this question, but some "
    "components are not sufficiently covered."
)


class CoverageError(RuntimeError):
    """Raised when Phase 17 cannot safely classify corpus coverage."""


@dataclass(frozen=True)
class EvidenceItem:
    chunk_id: str
    source_id: str
    domain: str
    citation: str
    corpus_version: str
    active_concepts: tuple[str, ...]
    retrieval_score: float | None


@dataclass(frozen=True)
class DomainEvidence:
    domain: str
    status: str
    evidence: tuple[EvidenceItem, ...]


@dataclass(frozen=True)
class DomainGeneration:
    domain: str
    claim_count: int
    citation_count: int
    concepts_covered: tuple[str, ...]
    unsupported_aspects: tuple[str, ...]
    validation_passed: bool


@dataclass(frozen=True)
class ConceptCoverage:
    concept: str
    activation_weight: float
    evidence_count: int
    covered_domains: tuple[str, ...]
    claim_domains: tuple[str, ...]
    citation_quality: float
    retrieval_confidence: float
    retrieval_confidence_source: str
    explicit_unsupported: bool
    insufficient_comparison: bool
    coverage_score: float
    score_components: dict[str, float]
    hard_overrides: tuple[str, ...]
    status: ConceptStatus
    reasons: tuple[str, ...]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 17: classify reviewed Phase 1 corpus coverage deterministically."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--evidence-package", type=Path, default=DEFAULT_EVIDENCE_PACKAGE)
    parser.add_argument("--retrieval-manifest", type=Path, default=DEFAULT_RETRIEVAL_MANIFEST)
    parser.add_argument("--domain-responses", type=Path, default=DEFAULT_DOMAIN_RESPONSES)
    parser.add_argument("--generation-manifest", type=Path, default=DEFAULT_GENERATION_MANIFEST)
    parser.add_argument("--synthesis", type=Path, default=DEFAULT_SYNTHESIS)
    parser.add_argument("--synthesis-manifest", type=Path, default=DEFAULT_SYNTHESIS_MANIFEST)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
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
        raise CoverageError(f"Required file does not exist: {path}")


def require_mapping(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise CoverageError(f"{description} must be an object.")
    result: dict[str, object] = {}
    for key, nested in value.items():
        if not isinstance(key, str):
            raise CoverageError(f"{description} contains a non-string key.")
        result[key] = nested
    return result


def require_list(value: object, description: str) -> list[object]:
    if not isinstance(value, list):
        raise CoverageError(f"{description} must be a list.")
    return value


def require_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CoverageError(f"{description} must be a non-empty string.")
    return value.strip()


def optional_string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def load_json(path: Path) -> dict[str, object]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CoverageError(f"Invalid JSON in {path}: {exc}") from exc
    return require_mapping(raw, f"JSON document {path}")


def atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f"{path.suffix}.tmp")
    temp.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_string_list(value: object, description: str) -> tuple[str, ...]:
    raw = require_list(value, description)
    result: list[str] = []
    for index, item in enumerate(raw, start=1):
        text = require_string(item, f"{description}[{index}]")
        if text not in result:
            result.append(text)
    return tuple(result)


def active_query_concepts(query_activation: Mapping[str, object]) -> tuple[str, ...]:
    raw = query_activation.get("active_concepts")
    if not isinstance(raw, list):
        return ()
    result: list[str] = []
    for item in raw:
        if isinstance(item, str) and item in CONCEPTS and item not in result:
            result.append(item)
    return tuple(result)


def calibrated_weights(query_activation: Mapping[str, object]) -> dict[str, float]:
    raw = query_activation.get("calibrated_weights")
    if not isinstance(raw, Mapping):
        return dict.fromkeys(CONCEPTS, 0.0)
    result: dict[str, float] = {}
    for concept in CONCEPTS:
        value = optional_float(raw.get(concept))
        result[concept] = max(0.0, min(1.0, value)) if value is not None else 0.0
    return result


def infer_concepts_from_text(text: str) -> tuple[str, ...]:
    normalized = text.casefold()
    return tuple(
        concept
        for concept in CONCEPTS
        if any(term.casefold() in normalized for term in CONCEPT_HINT_TERMS[concept])
    )


def extract_retrieval_score(raw: Mapping[str, object]) -> float | None:
    for field in RETRIEVAL_SCORE_FIELDS:
        score = optional_float(raw.get(field))
        if score is not None:
            return max(0.0, min(1.0, score))
    scoring = raw.get("scoring")
    if isinstance(scoring, Mapping):
        for field in RETRIEVAL_SCORE_FIELDS:
            score = optional_float(scoring.get(field))
            if score is not None:
                return max(0.0, min(1.0, score))
    return None


def parse_active_concepts_from_evidence(raw: Mapping[str, object]) -> tuple[str, ...]:
    concepts_raw = raw.get("concepts")
    if not isinstance(concepts_raw, Mapping):
        return ()
    result: list[str] = []
    for concept in CONCEPTS:
        relation = concepts_raw.get(concept)
        if (
            isinstance(relation, Mapping) and relation.get("production_active") is True
        ) or relation is True:
            result.append(concept)
    return tuple(result)


def parse_evidence_package(
    path: Path,
) -> tuple[str, str, dict[str, object], dict[str, DomainEvidence]]:
    package = load_json(path)
    if optional_string(package.get("retrieval_mode")) != "concept_aware":
        raise CoverageError("Phase 17 requires a concept-aware Phase 14 evidence package.")

    question = require_string(package.get("question"), "evidence package question")
    corpus_version = require_string(
        package.get("corpus_version"), "evidence package corpus_version"
    )
    query_activation = require_mapping(
        package.get("query_activation"), "evidence package query_activation"
    )
    domains_raw = require_mapping(package.get("domains"), "evidence package domains")

    domains: dict[str, DomainEvidence] = {}
    for domain in DOMAINS:
        domain_raw = require_mapping(domains_raw.get(domain), f"{domain} evidence package")
        status = optional_string(domain_raw.get("status"))
        evidence_raw = require_list(domain_raw.get("evidence"), f"{domain} evidence")
        evidence: list[EvidenceItem] = []
        seen: set[str] = set()
        for index, item_raw in enumerate(evidence_raw, start=1):
            item = require_mapping(item_raw, f"{domain} evidence item {index}")
            item_domain = require_string(item.get("domain"), f"{domain} evidence domain").casefold()
            if item_domain != domain:
                raise CoverageError(
                    f"Phase 14 domain leakage: expected={domain} actual={item_domain}."
                )
            item_corpus = require_string(
                item.get("corpus_version"), f"{domain} evidence corpus_version"
            )
            if item_corpus != corpus_version:
                raise CoverageError(f"{domain} evidence corpus version mismatch.")
            chunk_id = require_string(item.get("chunk_id"), f"{domain} evidence chunk_id")
            if chunk_id in seen:
                raise CoverageError(f"Duplicate evidence chunk {chunk_id!r}.")
            seen.add(chunk_id)
            evidence.append(
                EvidenceItem(
                    chunk_id=chunk_id,
                    source_id=require_string(
                        item.get("source_id"), f"{domain}/{chunk_id}/source_id"
                    ),
                    domain=domain,
                    citation=require_string(item.get("citation"), f"{domain}/{chunk_id}/citation"),
                    corpus_version=item_corpus,
                    active_concepts=parse_active_concepts_from_evidence(item),
                    retrieval_score=extract_retrieval_score(item),
                )
            )
        domains[domain] = DomainEvidence(domain=domain, status=status, evidence=tuple(evidence))

    return question, corpus_version, query_activation, domains


def parse_domain_generations(
    path: Path,
    *,
    expected_question: str,
    expected_corpus_version: str,
) -> dict[str, DomainGeneration]:
    document = load_json(path)
    if require_string(document.get("question"), "domain responses question") != expected_question:
        raise CoverageError("Phase 14 and Phase 15 questions differ.")
    if (
        require_string(document.get("corpus_version"), "domain responses corpus_version")
        != expected_corpus_version
    ):
        raise CoverageError("Phase 14 and Phase 15 corpus versions differ.")

    domains_raw = require_mapping(document.get("domains"), "domain responses domains")
    result: dict[str, DomainGeneration] = {}
    for domain in DOMAINS:
        raw = require_mapping(domains_raw.get(domain), f"{domain} domain response")
        validation_raw = raw.get("validation")
        validation_passed = (
            isinstance(validation_raw, Mapping) and validation_raw.get("passed") is True
        )
        claims_raw = require_list(raw.get("claims"), f"{domain} claims")
        concepts: list[str] = []
        citation_keys: set[tuple[str, str]] = set()
        for index, claim_raw in enumerate(claims_raw, start=1):
            claim = require_mapping(claim_raw, f"{domain} claim {index}")
            claim_concepts = parse_string_list(
                claim.get("concepts_covered"), f"{domain} claim {index} concepts"
            )
            for concept in claim_concepts:
                if concept in CONCEPTS and concept not in concepts:
                    concepts.append(concept)
            citations_raw = claim.get("citations")
            if isinstance(citations_raw, list):
                for citation_raw in citations_raw:
                    if isinstance(citation_raw, Mapping):
                        chunk_id = optional_string(citation_raw.get("chunk_id"))
                        source_id = optional_string(citation_raw.get("source_id"))
                        if chunk_id and source_id:
                            citation_keys.add((chunk_id, source_id))

        top_level_citations = raw.get("citations")
        if isinstance(top_level_citations, list):
            for citation_raw in top_level_citations:
                if isinstance(citation_raw, Mapping):
                    chunk_id = optional_string(citation_raw.get("chunk_id"))
                    source_id = optional_string(citation_raw.get("source_id"))
                    if chunk_id and source_id:
                        citation_keys.add((chunk_id, source_id))

        unsupported_raw = raw.get("unsupported_aspects")
        unsupported = (
            tuple(
                item.strip() for item in unsupported_raw if isinstance(item, str) and item.strip()
            )
            if isinstance(unsupported_raw, list)
            else ()
        )
        result[domain] = DomainGeneration(
            domain=domain,
            claim_count=len(claims_raw),
            citation_count=len(citation_keys),
            concepts_covered=tuple(concepts),
            unsupported_aspects=unsupported,
            validation_passed=validation_passed,
        )
    return result


def validate_manifest(
    path: Path,
    *,
    expected_phase: str,
    allowed_statuses: tuple[str, ...],
    description: str,
) -> dict[str, object]:
    manifest = load_json(path)
    phase = optional_string(manifest.get("phase"))
    status = optional_string(manifest.get("status"))
    if phase != expected_phase:
        raise CoverageError(f"{description} has unexpected phase {phase!r}.")
    if status not in allowed_statuses:
        raise CoverageError(f"{description} status {status!r} is not complete.")
    return manifest


def validate_upstream_manifests(
    *,
    retrieval_manifest_path: Path,
    generation_manifest_path: Path,
    synthesis_manifest_path: Path,
    corpus_version: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    retrieval = validate_manifest(
        retrieval_manifest_path,
        expected_phase="phase_14_build_retrieval_by_concept_and_domain",
        allowed_statuses=("evaluation_complete",),
        description="Phase 14 manifest",
    )
    generation = validate_manifest(
        generation_manifest_path,
        expected_phase="phase_15_build_domain_specific_generation",
        allowed_statuses=("domain_generation_complete",),
        description="Phase 15 manifest",
    )
    synthesis = validate_manifest(
        synthesis_manifest_path,
        expected_phase="phase_16_synthesis_and_tension_detection",
        allowed_statuses=("synthesis_complete",),
        description="Phase 16 manifest",
    )
    for name, manifest in (
        ("Phase 14", retrieval),
        ("Phase 15", generation),
        ("Phase 16", synthesis),
    ):
        if (
            require_string(manifest.get("corpus_version"), f"{name} corpus_version")
            != corpus_version
        ):
            raise CoverageError(f"{name} corpus version does not match evidence package.")
    return retrieval, generation, synthesis


def parse_synthesis(
    path: Path,
    *,
    expected_question: str,
    expected_corpus_version: str,
) -> tuple[dict[str, object], tuple[str, ...]]:
    synthesis = load_json(path)
    if require_string(synthesis.get("question"), "synthesis question") != expected_question:
        raise CoverageError("Phase 16 question differs from Phase 14/15.")
    if (
        require_string(synthesis.get("corpus_version"), "synthesis corpus_version")
        != expected_corpus_version
    ):
        raise CoverageError("Phase 16 corpus version differs from Phase 14/15.")
    validation = require_mapping(synthesis.get("validation"), "Phase 16 synthesis validation")
    if validation.get("passed") is not True:
        raise CoverageError("Phase 16 synthesis validation did not pass.")

    insufficient_raw = synthesis.get("insufficient_corpus_coverage")
    concepts: list[str] = []
    if isinstance(insufficient_raw, list):
        for item_raw in insufficient_raw:
            if not isinstance(item_raw, Mapping):
                continue
            raw_concepts = item_raw.get("concepts_covered")
            if not isinstance(raw_concepts, list):
                continue
            for concept in raw_concepts:
                if isinstance(concept, str) and concept in CONCEPTS and concept not in concepts:
                    concepts.append(concept)
    return synthesis, tuple(concepts)


def evidence_counts_for_concept(
    *, concept: str, evidence_domains: Mapping[str, DomainEvidence]
) -> tuple[int, tuple[str, ...], tuple[float, ...], int]:
    count = 0
    covered_domains: list[str] = []
    scores: list[float] = []
    citation_complete = 0
    for domain in DOMAINS:
        domain_count = 0
        for item in evidence_domains[domain].evidence:
            if concept not in item.active_concepts:
                continue
            count += 1
            domain_count += 1
            if item.chunk_id and item.source_id and item.citation and item.corpus_version:
                citation_complete += 1
            if item.retrieval_score is not None:
                scores.append(item.retrieval_score)
        if domain_count > 0:
            covered_domains.append(domain)
    return count, tuple(covered_domains), tuple(scores), citation_complete


def claim_domains_for_concept(
    *, concept: str, generations: Mapping[str, DomainGeneration]
) -> tuple[str, ...]:
    return tuple(
        domain
        for domain in DOMAINS
        if generations[domain].validation_passed and concept in generations[domain].concepts_covered
    )


def unsupported_for_concept(*, concept: str, generations: Mapping[str, DomainGeneration]) -> bool:
    return any(
        concept in infer_concepts_from_text(unsupported)
        for domain in DOMAINS
        for unsupported in generations[domain].unsupported_aspects
    )


def retrieval_confidence(
    *,
    activation_weight: float,
    covered_domain_count: int,
    evidence_count: int,
    explicit_scores: tuple[float, ...],
) -> tuple[float, str]:
    if explicit_scores:
        return round(sum(explicit_scores) / len(explicit_scores), 6), "phase14_score_mean"

    domain_factor = covered_domain_count / len(DOMAINS)
    evidence_factor = min(1.0, evidence_count / len(DOMAINS))
    proxy = 0.50 * activation_weight + 0.30 * domain_factor + 0.20 * evidence_factor
    return round(max(0.0, min(1.0, proxy)), 6), "derived_proxy"


def coverage_score_components(
    *,
    activation_weight: float,
    evidence_count: int,
    covered_domain_count: int,
    citation_quality: float,
    retrieval_confidence_value: float,
    explicit_unsupported: bool,
    insufficient_comparison: bool,
) -> dict[str, float]:
    activation_component = max(0.0, min(1.0, activation_weight)) * SCORE_WEIGHT_ACTIVATION
    evidence_component = (
        min(1.0, evidence_count / TARGET_EVIDENCE_PER_CONCEPT) * SCORE_WEIGHT_EVIDENCE
    )
    domain_component = covered_domain_count / len(DOMAINS) * SCORE_WEIGHT_DOMAIN_COVERAGE
    citation_component = max(0.0, min(1.0, citation_quality)) * SCORE_WEIGHT_CITATION_QUALITY
    confidence_component = (
        max(0.0, min(1.0, retrieval_confidence_value)) * SCORE_WEIGHT_RETRIEVAL_CONFIDENCE
    )

    unsupported_component = SCORE_WEIGHT_UNSUPPORTED_COMPONENT
    if explicit_unsupported:
        unsupported_component -= EXPLICIT_UNSUPPORTED_PENALTY
    if insufficient_comparison:
        unsupported_component -= INSUFFICIENT_COMPARISON_PENALTY
    unsupported_component = max(0.0, unsupported_component)

    return {
        "activated_concept_weight": round(activation_component, 4),
        "retrieved_evidence": round(evidence_component, 4),
        "domain_coverage": round(domain_component, 4),
        "citation_quality": round(citation_component, 4),
        "retrieval_confidence": round(confidence_component, 4),
        "unsupported_subquestion_component": round(unsupported_component, 4),
    }


def classify_concept(
    *,
    concept: str,
    activation_weight: float,
    evidence_domains: Mapping[str, DomainEvidence],
    generations: Mapping[str, DomainGeneration],
    insufficient_concepts: tuple[str, ...],
) -> ConceptCoverage:
    (
        evidence_count,
        covered_domains,
        scores,
        citation_complete,
    ) = evidence_counts_for_concept(
        concept=concept,
        evidence_domains=evidence_domains,
    )
    claim_domains = claim_domains_for_concept(concept=concept, generations=generations)
    citation_quality = citation_complete / evidence_count if evidence_count else 0.0
    confidence, confidence_source = retrieval_confidence(
        activation_weight=activation_weight,
        covered_domain_count=len(covered_domains),
        evidence_count=evidence_count,
        explicit_scores=scores,
    )
    explicit_unsupported = unsupported_for_concept(concept=concept, generations=generations)
    insufficient_comparison = concept in insufficient_concepts

    components = coverage_score_components(
        activation_weight=activation_weight,
        evidence_count=evidence_count,
        covered_domain_count=len(covered_domains),
        citation_quality=citation_quality,
        retrieval_confidence_value=confidence,
        explicit_unsupported=explicit_unsupported,
        insufficient_comparison=insufficient_comparison,
    )
    raw_score = round(sum(components.values()), 2)

    reasons: list[str] = []
    hard_overrides: list[str] = []

    if evidence_count == 0:
        reasons.append("no reviewed retrieval evidence")
        hard_overrides.append("no_reviewed_evidence")

    missing_evidence = [domain for domain in DOMAINS if domain not in covered_domains]
    if missing_evidence:
        reasons.append("missing reviewed evidence in " + ", ".join(missing_evidence))

    missing_claims = [domain for domain in DOMAINS if domain not in claim_domains]
    if missing_claims:
        reasons.append("no grounded Phase 15 claim in " + ", ".join(missing_claims))

    if citation_quality < 1.0:
        reasons.append("citation provenance is incomplete")

    if explicit_unsupported:
        reasons.append("Phase 15 identified a narrower unsupported aspect for this concept")

    if insufficient_comparison:
        reasons.append("Phase 16 identified insufficient comparison coverage")
        hard_overrides.append("insufficient_comparison_caps_at_partial")

    if not claim_domains:
        hard_overrides.append("no_grounded_phase15_claims")

    if len(covered_domains) <= 1:
        hard_overrides.append("single_or_zero_domain_coverage")

    if raw_score >= SUPPORTED_SCORE_THRESHOLD:
        status: ConceptStatus = "Supported"
    elif raw_score >= PARTIAL_SCORE_THRESHOLD:
        status = "Partially Supported"
    else:
        status = "Unsupported"

    if "no_reviewed_evidence" in hard_overrides or "no_grounded_phase15_claims" in hard_overrides:
        status = "Unsupported"
    elif "single_or_zero_domain_coverage" in hard_overrides and status == "Supported":
        status = "Partially Supported"

    if insufficient_comparison and status == "Supported":
        status = "Partially Supported"

    return ConceptCoverage(
        concept=concept,
        activation_weight=activation_weight,
        evidence_count=evidence_count,
        covered_domains=covered_domains,
        claim_domains=claim_domains,
        citation_quality=round(citation_quality, 6),
        retrieval_confidence=confidence,
        retrieval_confidence_source=confidence_source,
        explicit_unsupported=explicit_unsupported,
        insufficient_comparison=insufficient_comparison,
        coverage_score=raw_score,
        score_components=components,
        hard_overrides=tuple(hard_overrides),
        status=status,
        reasons=tuple(reasons),
    )


def weighted_overall_score(
    concept_results: tuple[ConceptCoverage, ...],
) -> float:
    if not concept_results:
        return 0.0

    weight_total = sum(max(result.activation_weight, 0.01) for result in concept_results)
    weighted = sum(
        result.coverage_score * max(result.activation_weight, 0.01) for result in concept_results
    )
    return round(weighted / weight_total, 2)


def classify_overall(
    *,
    query_unsupported: bool,
    active_concepts: tuple[str, ...],
    concept_results: tuple[ConceptCoverage, ...],
    covered_domains: tuple[str, ...],
) -> tuple[
    CoverageStatus,
    float,
    tuple[str, ...],
    str,
]:
    hard_overrides: list[str] = []

    if query_unsupported or not active_concepts:
        return (
            "Out of Corpus",
            0.0,
            ("query_outside_phase1_concepts",),
            (
                "The question is not sufficiently covered by the three "
                "Phase 1 concepts in the reviewed WTH corpus."
            ),
        )

    if not concept_results:
        return (
            "Out of Corpus",
            0.0,
            ("no_concept_coverage",),
            "No Phase 1 concept coverage could be established for the question.",
        )

    total_evidence = sum(result.evidence_count for result in concept_results)
    if total_evidence == 0 or not covered_domains:
        return (
            "Out of Corpus",
            0.0,
            ("no_reviewed_evidence",),
            (
                "The question activates Phase 1 concepts, but no reviewed "
                "retrieval evidence is available to support an answer."
            ),
        )

    score = weighted_overall_score(concept_results)
    statuses = {result.status for result in concept_results}

    if len(covered_domains) <= 1:
        hard_overrides.append("single_or_zero_domain_coverage")

    if score >= SUPPORTED_SCORE_THRESHOLD:
        status: CoverageStatus = "Supported"
    elif score >= PARTIAL_SCORE_THRESHOLD:
        status = "Partially Supported"
    else:
        status = "Out of Corpus"

    if status == "Supported" and statuses != {"Supported"}:
        status = "Partially Supported"
        hard_overrides.append("one_or_more_active_concepts_not_fully_supported")

    if status == "Supported" and len(covered_domains) <= 1:
        status = "Partially Supported"

    if statuses == {"Unsupported"}:
        status = "Out of Corpus"
        hard_overrides.append("all_active_concepts_unsupported")

    supported_names = [result.concept for result in concept_results if result.status == "Supported"]
    partial_names = [
        result.concept for result in concept_results if result.status == "Partially Supported"
    ]
    unsupported_names = [
        result.concept for result in concept_results if result.status == "Unsupported"
    ]

    if status == "Supported":
        reason = (
            f"Coverage score {score:.1f}/100. The reviewed corpus directly "
            "supports all major active concepts across the required domains "
            "with grounded claims and citation provenance."
        )
    elif status == "Partially Supported":
        details: list[str] = []
        if supported_names:
            details.append("supported: " + ", ".join(supported_names))
        if partial_names:
            details.append("partial: " + ", ".join(partial_names))
        if unsupported_names:
            details.append("unsupported: " + ", ".join(unsupported_names))
        reason = (
            f"Coverage score {score:.1f}/100. The reviewed corpus supports "
            "useful parts of the question but one or more components require "
            "qualification"
        )
        if details:
            reason += " (" + "; ".join(details) + ")"
        reason += "."
    else:
        reason = (
            f"Coverage score {score:.1f}/100. The reviewed corpus does not "
            "contain enough grounded evidence to answer the major components "
            "of this question reliably."
        )

    return status, score, tuple(hard_overrides), reason


def response_policy(status: CoverageStatus) -> dict[str, object]:
    if status == "Supported":
        return {
            "corpus_answer_allowed": True,
            "corpus_answer_scope": "supported_evidence_only",
            "general_knowledge_fallback_allowed": False,
            "general_knowledge_must_be_labeled": True,
            "general_knowledge_must_not_use_corpus_citations": True,
            "suggested_disclosure": "",
        }
    if status == "Partially Supported":
        return {
            "corpus_answer_allowed": True,
            "corpus_answer_scope": "supported_components_only",
            "general_knowledge_fallback_allowed": True,
            "general_knowledge_must_be_labeled": True,
            "general_knowledge_must_not_use_corpus_citations": True,
            "suggested_disclosure": PARTIAL_SUPPORT_INTRO,
        }
    return {
        "corpus_answer_allowed": False,
        "corpus_answer_scope": "none",
        "general_knowledge_fallback_allowed": True,
        "general_knowledge_must_be_labeled": True,
        "general_knowledge_must_not_use_corpus_citations": True,
        "suggested_disclosure": OUT_OF_CORPUS_INTRO,
        "fallback_structure": [
            "state reviewed-corpus limitation",
            "briefly interpret the user's question",
            "label the next section as general knowledge",
            "answer from general knowledge without WTH corpus citations",
        ],
    }


def concept_payload(result: ConceptCoverage) -> dict[str, object]:
    return {
        "concept": result.concept,
        "status": result.status,
        "coverage_score": result.coverage_score,
        "activation_weight": result.activation_weight,
        "evidence_count": result.evidence_count,
        "covered_domains": list(result.covered_domains),
        "claim_domains": list(result.claim_domains),
        "citation_quality": result.citation_quality,
        "retrieval_confidence": result.retrieval_confidence,
        "retrieval_confidence_source": result.retrieval_confidence_source,
        "explicit_unsupported": result.explicit_unsupported,
        "insufficient_comparison": result.insufficient_comparison,
        "score_components": dict(result.score_components),
        "hard_overrides": list(result.hard_overrides),
        "reasons": list(result.reasons),
    }


def output_paths(output_directory: Path) -> dict[str, Path]:
    return {
        "coverage": output_directory / "coverage.json",
        "manifest": output_directory / "coverage_manifest.json",
    }


def ensure_replace_policy(*, paths: Mapping[str, Path], replace: bool) -> None:
    if replace:
        return
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise CoverageError(
            "Phase 17 outputs already exist. Use --replace: "
            + ", ".join(path.as_posix() for path in existing)
        )


def run_phase17(
    *,
    project_root: Path,
    evidence_package_path: Path,
    retrieval_manifest_path: Path,
    domain_responses_path: Path,
    generation_manifest_path: Path,
    synthesis_path: Path,
    synthesis_manifest_path: Path,
    output_directory: Path,
    replace: bool,
) -> dict[str, object]:
    project_root = project_root.resolve()
    evidence_package_path = resolve(project_root, evidence_package_path)
    retrieval_manifest_path = resolve(project_root, retrieval_manifest_path)
    domain_responses_path = resolve(project_root, domain_responses_path)
    generation_manifest_path = resolve(project_root, generation_manifest_path)
    synthesis_path = resolve(project_root, synthesis_path)
    synthesis_manifest_path = resolve(project_root, synthesis_manifest_path)
    output_directory = resolve(project_root, output_directory)

    for path in (
        evidence_package_path,
        retrieval_manifest_path,
        domain_responses_path,
        generation_manifest_path,
        synthesis_path,
        synthesis_manifest_path,
    ):
        require_file(path)

    LOGGER.info("Phase 17 starting: coverage_version=%s", COVERAGE_VERSION)

    question, corpus_version, query_activation, evidence_domains = parse_evidence_package(
        evidence_package_path
    )
    retrieval_manifest, generation_manifest, synthesis_manifest = validate_upstream_manifests(
        retrieval_manifest_path=retrieval_manifest_path,
        generation_manifest_path=generation_manifest_path,
        synthesis_manifest_path=synthesis_manifest_path,
        corpus_version=corpus_version,
    )
    generations = parse_domain_generations(
        domain_responses_path,
        expected_question=question,
        expected_corpus_version=corpus_version,
    )
    synthesis, insufficient_concepts = parse_synthesis(
        synthesis_path,
        expected_question=question,
        expected_corpus_version=corpus_version,
    )

    active_concepts = active_query_concepts(query_activation)
    weights = calibrated_weights(query_activation)
    query_unsupported = query_activation.get("unsupported") is True

    concept_results = tuple(
        classify_concept(
            concept=concept,
            activation_weight=weights.get(concept, 0.0),
            evidence_domains=evidence_domains,
            generations=generations,
            insufficient_concepts=insufficient_concepts,
        )
        for concept in active_concepts
    )

    covered_domains = tuple(domain for domain in DOMAINS if evidence_domains[domain].evidence)
    missing_domains = tuple(domain for domain in DOMAINS if domain not in covered_domains)
    (
        status,
        coverage_score,
        overall_hard_overrides,
        reason,
    ) = classify_overall(
        query_unsupported=query_unsupported,
        active_concepts=active_concepts,
        concept_results=concept_results,
        covered_domains=covered_domains,
    )

    supported_concepts = tuple(
        result.concept for result in concept_results if result.status == "Supported"
    )
    partially_supported_concepts = tuple(
        result.concept for result in concept_results if result.status == "Partially Supported"
    )
    unsupported_concepts = tuple(
        result.concept for result in concept_results if result.status == "Unsupported"
    )
    policy = response_policy(status)

    total_evidence = sum(len(evidence_domains[domain].evidence) for domain in DOMAINS)
    total_claims = sum(generations[domain].claim_count for domain in DOMAINS)
    total_citations = sum(generations[domain].citation_count for domain in DOMAINS)

    coverage: dict[str, object] = {
        "coverage_version": COVERAGE_VERSION,
        "generated_at": utc_now(),
        "question": question,
        "corpus_version": corpus_version,
        "coverage_status": status,
        "coverage_score": coverage_score,
        "coverage_reason": reason,
        "supported_concepts": list(supported_concepts),
        "partially_supported_concepts": list(partially_supported_concepts),
        "unsupported_concepts": list(unsupported_concepts),
        "hard_overrides": list(overall_hard_overrides),
        "covered_domains": list(covered_domains),
        "missing_domains": list(missing_domains),
        "signals": {
            "query_unsupported": query_unsupported,
            "query_ambiguous": query_activation.get("ambiguous") is True,
            "active_concepts": list(active_concepts),
            "activated_concept_weights": {concept: weights[concept] for concept in active_concepts},
            "retrieved_evidence_count": total_evidence,
            "grounded_claim_count": total_claims,
            "claim_citation_count": total_citations,
            "insufficient_comparison_concepts": list(insufficient_concepts),
            "coverage_thresholds": {
                "supported_min": SUPPORTED_SCORE_THRESHOLD,
                "partial_min": PARTIAL_SCORE_THRESHOLD,
            },
            "score_weights": {
                "activated_concept_weight": SCORE_WEIGHT_ACTIVATION,
                "retrieved_evidence": SCORE_WEIGHT_EVIDENCE,
                "domain_coverage": SCORE_WEIGHT_DOMAIN_COVERAGE,
                "citation_quality": SCORE_WEIGHT_CITATION_QUALITY,
                "retrieval_confidence": SCORE_WEIGHT_RETRIEVAL_CONFIDENCE,
                "unsupported_subquestion_component": SCORE_WEIGHT_UNSUPPORTED_COMPONENT,
            },
        },
        "concept_coverage": [concept_payload(result) for result in concept_results],
        "response_policy": policy,
        "boundary": {
            "reviewed_corpus_and_general_knowledge_are_separate": True,
            "corpus_claims_require_reviewed_evidence": True,
            "general_knowledge_may_not_be_presented_as_corpus_supported": True,
            "general_knowledge_may_not_reuse_wth_corpus_citations": True,
        },
        "upstream_versions": {
            "retrieval_manifest_status": optional_string(retrieval_manifest.get("status")),
            "generation_version": optional_string(generation_manifest.get("generation_version")),
            "synthesis_version": optional_string(synthesis.get("synthesis_version")),
            "synthesis_manifest_status": optional_string(synthesis_manifest.get("status")),
        },
    }

    exit_gate_passed = (
        status in {"Supported", "Partially Supported", "Out of Corpus"}
        and bool(reason)
        and (status != "Out of Corpus" or policy["corpus_answer_allowed"] is False)
        and policy["general_knowledge_must_be_labeled"] is True
        and policy["general_knowledge_must_not_use_corpus_citations"] is True
    )
    coverage["exit_gate"] = {
        "passed": exit_gate_passed,
        "no_corpus_fabrication_when_out_of_corpus": (
            status != "Out of Corpus" or policy["corpus_answer_allowed"] is False
        ),
        "partial_answers_limited_to_supported_components": (
            status != "Partially Supported"
            or policy["corpus_answer_scope"] == "supported_components_only"
        ),
        "general_knowledge_boundary_explicit": True,
        "corpus_citations_forbidden_for_general_fallback": True,
    }

    paths = output_paths(output_directory)
    ensure_replace_policy(paths=paths, replace=replace)
    output_directory.mkdir(parents=True, exist_ok=True)
    atomic_json(paths["coverage"], coverage)

    manifest: dict[str, object] = {
        "phase": "phase_17_coverage_classification",
        "status": (
            "coverage_classification_complete"
            if exit_gate_passed
            else "coverage_classification_failed"
        ),
        "script_version": SCRIPT_VERSION,
        "coverage_version": COVERAGE_VERSION,
        "generated_at": utc_now(),
        "question": question,
        "corpus_version": corpus_version,
        "classification": {
            "coverage_status": status,
            "coverage_score": coverage_score,
            "supported_concept_count": len(supported_concepts),
            "partially_supported_concept_count": len(partially_supported_concepts),
            "unsupported_concept_count": len(unsupported_concepts),
            "covered_domain_count": len(covered_domains),
            "missing_domain_count": len(missing_domains),
        },
        "calculation_policy": {
            "deterministic": True,
            "llm_calls": 0,
            "activated_concept_weights_used": True,
            "retrieved_evidence_count_used": True,
            "domain_coverage_used": True,
            "citation_quality_used": True,
            "unsupported_subquestions_used": True,
            "retrieval_confidence_used": True,
            "score_then_hard_override_policy": True,
            "supported_score_threshold": SUPPORTED_SCORE_THRESHOLD,
            "partial_score_threshold": PARTIAL_SCORE_THRESHOLD,
            "strict_supported_requires_all_active_concepts_supported": True,
            "single_domain_cannot_be_fully_supported": True,
        },
        "knowledge_boundary": {
            "reviewed_corpus_support_separate_from_general_knowledge": True,
            "out_of_corpus_corpus_answer_forbidden": True,
            "general_fallback_allowed_when_labeled": True,
            "general_fallback_cannot_use_wth_corpus_citations": True,
        },
        "outputs": {"coverage": paths["coverage"].as_posix()},
        "exit_gate": coverage["exit_gate"],
        "next_step": (
            "If the exit gate passes, freeze Phase 17 v2 score/override policy and begin Phase 18 final response assembly. "
            "Phase 18 must obey response_policy and keep any general-knowledge fallback clearly separated."
        ),
    }
    atomic_json(paths["manifest"], manifest)

    LOGGER.info("Phase 17 coverage classification complete")
    LOGGER.info(
        "coverage_status=%s coverage_score=%.2f supported_concepts=%d "
        "partially_supported_concepts=%d unsupported_concepts=%d "
        "covered_domains=%d missing_domains=%d",
        status,
        coverage_score,
        len(supported_concepts),
        len(partially_supported_concepts),
        len(unsupported_concepts),
        len(covered_domains),
        len(missing_domains),
    )
    LOGGER.info("Exit gate passed: %s", exit_gate_passed)
    LOGGER.info("Coverage output: %s", paths["coverage"])
    LOGGER.info("Coverage manifest: %s", paths["manifest"])

    if not exit_gate_passed:
        raise CoverageError("Phase 17 failed its anti-fabrication exit gate.")
    return manifest


def main() -> int:
    arguments = parse_arguments()
    configure_logging(arguments.log_level)
    try:
        run_phase17(
            project_root=arguments.project_root,
            evidence_package_path=arguments.evidence_package,
            retrieval_manifest_path=arguments.retrieval_manifest,
            domain_responses_path=arguments.domain_responses,
            generation_manifest_path=arguments.generation_manifest,
            synthesis_path=arguments.synthesis,
            synthesis_manifest_path=arguments.synthesis_manifest,
            output_directory=arguments.output_directory,
            replace=arguments.replace,
        )
    except CoverageError:
        LOGGER.exception("Phase 17 coverage classification failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
