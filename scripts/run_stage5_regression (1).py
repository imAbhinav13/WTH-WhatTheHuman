from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, Mapping

import httpx


API_PATH: Final = "/api/query"
EXPECTED_CORPUS_VERSION: Final = "phase1_active_corpus_v1"
VALID_COVERAGE_STATUSES: Final = {
    "Supported",
    "Partially Supported",
    "Out of Corpus",
}
EXPECTED_DOMAINS: Final = {
    "science",
    "advaita",
    "samkhya",
}

ATMAN_PURUSHA_EQUIVALENCE_RE: Final = re.compile(
    r"\b(?:atman|ātman)\b.{0,120}\b"
    r"(?:same|identical|equivalent|same concept|same entity)\b"
    r".{0,120}\b(?:purusha|puruṣa)\b"
    r"|"
    r"\b(?:purusha|puruṣa)\b.{0,120}\b"
    r"(?:same|identical|equivalent|same concept|same entity)\b"
    r".{0,120}\b(?:atman|ātman)\b",
    re.IGNORECASE | re.DOTALL,
)

COGNITION_CONSCIOUSNESS_EQUIVALENCE_RE: Final = re.compile(
    r"\bcognition\b.{0,120}\b"
    r"(?:same|identical|equivalent)\b.{0,120}\bconsciousness\b"
    r"|"
    r"\bconsciousness\b.{0,120}\b"
    r"(?:same|identical|equivalent)\b.{0,120}\bcognition\b",
    re.IGNORECASE | re.DOTALL,
)

SCIENCE_MAYA_PROOF_RE: Final = re.compile(
    r"\b(?:science|scientific|neuroscience|empirical|perceptual)\b"
    r".{0,200}\b(?:prove|proves|proven|establishes|confirms)\b"
    r".{0,160}\b(?:maya|māyā)\b",
    re.IGNORECASE | re.DOTALL,
)

SCIENCE_METAPHYSICS_PROOF_RE: Final = re.compile(
    r"\b(?:science|scientific|empirical|neuroscience)\b"
    r".{0,200}\b(?:prove|proves|proven|disprove|disproves|disproven)\b"
    r".{0,180}\b(?:atman|ātman|purusha|puruṣa|brahman|maya|māyā)\b",
    re.IGNORECASE | re.DOTALL,
)

ADVAITA_SAMKHYA_COLLAPSE_RE: Final = re.compile(
    r"\b(?:advaita|non[- ]?dual(?:ity)?)\b.{0,180}\b"
    r"(?:same|identical|equivalent)\b.{0,180}\b"
    r"(?:samkhya|sāṃkhya|dual(?:ism|ist)?)\b"
    r"|"
    r"\b(?:samkhya|sāṃkhya|dual(?:ism|ist)?)\b.{0,180}\b"
    r"(?:same|identical|equivalent)\b.{0,180}\b"
    r"(?:advaita|non[- ]?dual(?:ity)?)\b",
    re.IGNORECASE | re.DOTALL,
)


class RegressionStatus(StrEnum):
    PASS = "PASS"
    TRUE_REGRESSION = "TRUE_REGRESSION"
    PROVIDER_CAPACITY_BLOCKED = "PROVIDER_CAPACITY_BLOCKED"
    ENVIRONMENT_FAILURE = "ENVIRONMENT_FAILURE"


@dataclass(frozen=True, slots=True)
class RegressionCase:
    key: str
    category: str
    question: str
    expected_http_statuses: tuple[int, ...] = (200,)
    expected_error_code: str | None = None
    expected_coverage: tuple[str, ...] = (
        "Supported",
        "Partially Supported",
        "Out of Corpus",
    )
    required_active_concepts: tuple[str, ...] = ()
    minimum_comparisons: int = 0
    hard_negative_guard: str | None = None


@dataclass(frozen=True, slots=True)
class RegressionResult:
    case: str
    category: str
    status: RegressionStatus
    http_status: int | None
    request_id: str | None
    coverage_status: str | None
    validation_passed: bool | None
    citation_count: int | None
    comparison_count: int | None
    error_code: str | None
    error_phase: str | None
    retry_after_seconds: float | None
    elapsed_ms: float
    detail: str


CASES: Final[dict[str, RegressionCase]] = {
    "consciousness": RegressionCase(
        key="consciousness",
        category="normal",
        question=(
            "How do Science, Advaita Vedanta, and Samkhya "
            "understand consciousness?"
        ),
        expected_coverage=("Supported", "Partially Supported"),
        required_active_concepts=("consciousness",),
    ),
    "self": RegressionCase(
        key="self",
        category="normal",
        question=(
            "How is the self understood in Science, Advaita Vedanta, "
            "and Samkhya?"
        ),
        expected_coverage=("Supported", "Partially Supported"),
        required_active_concepts=("self_identity",),
    ),
    "reality": RegressionCase(
        key="reality",
        category="normal",
        question=(
            "How do the three perspectives distinguish experienced "
            "appearance from reality?"
        ),
        expected_coverage=("Supported", "Partially Supported"),
        required_active_concepts=("reality_appearance",),
    ),
    "multi": RegressionCase(
        key="multi",
        category="multi-concept",
        question=(
            "How is consciousness related to the self and "
            "experienced reality?"
        ),
        expected_coverage=("Supported", "Partially Supported"),
        required_active_concepts=(
            "consciousness",
            "self_identity",
            "reality_appearance",
        ),
        minimum_comparisons=9,
    ),
    "ambiguous": RegressionCase(
        key="ambiguous",
        category="ambiguous",
        question="What is the observer?",
    ),
    "atman_purusha": RegressionCase(
        key="atman_purusha",
        category="hard-negative",
        question="Are Atman and Purusha the same concept?",
        expected_coverage=("Supported", "Partially Supported"),
        required_active_concepts=("self_identity",),
        hard_negative_guard="atman_purusha",
    ),
    "cognition": RegressionCase(
        key="cognition",
        category="hard-negative",
        question="Is cognition the same thing as consciousness?",
        hard_negative_guard="cognition",
    ),
    "maya_science": RegressionCase(
        key="maya_science",
        category="hard-negative",
        question=(
            "Does a perceptual illusion prove that Advaita's Maya "
            "is scientifically true?"
        ),
        expected_coverage=("Partially Supported", "Out of Corpus"),
        hard_negative_guard="maya_science",
    ),
    "out_of_corpus": RegressionCase(
        key="out_of_corpus",
        category="out-of-corpus",
        question="Does quantum entanglement prove Advaita Vedanta?",
        expected_coverage=("Out of Corpus",),
        hard_negative_guard="out_of_corpus",
    ),
    "too_short": RegressionCase(
        key="too_short",
        category="request-validation",
        question="x",
        expected_http_statuses=(422,),
        expected_error_code="invalid_request",
    ),
}


class ContractFailure(AssertionError):
    """Raised when a response violates frozen Stage 4/Phase 1 behavior."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage 5.2 core/safety regression cases through "
            "POST /api/query."
        )
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--case",
        choices=tuple(CASES),
        default="multi",
    )
    parser.add_argument(
        "--all",
        action="store_true",
    )
    parser.add_argument(
        "--category",
        choices=tuple(
            sorted({case.category for case in CASES.values()})
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=135.0,
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--report-json",
        type=Path,
    )
    return parser.parse_args()


def _mapping(
    value: object,
    description: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ContractFailure(
            f"{description} must be a JSON object."
        )
    return {str(key): nested for key, nested in value.items()}


def _list(
    value: object,
    description: str,
) -> list[object]:
    if not isinstance(value, list):
        raise ContractFailure(
            f"{description} must be a JSON array."
        )
    return value


def _error_fields(
    document: Mapping[str, object],
) -> tuple[str | None, str | None, float | None]:
    error_raw = document.get("error")
    if not isinstance(error_raw, Mapping):
        return None, None, None

    code_raw = error_raw.get("code")
    phase_raw = error_raw.get("phase")
    retry_raw = error_raw.get("retry_after_seconds")

    code = code_raw if isinstance(code_raw, str) else None
    phase = phase_raw if isinstance(phase_raw, str) else None

    retry_after: float | None = None
    if (
        isinstance(retry_raw, (int, float))
        and not isinstance(retry_raw, bool)
    ):
        retry_after = float(retry_raw)

    return code, phase, retry_after


def _request_id(
    response: httpx.Response,
    document: Mapping[str, object],
) -> str | None:
    header_value = response.headers.get("X-Request-ID")
    body_value = document.get("request_id")

    if isinstance(header_value, str) and header_value:
        return header_value
    if isinstance(body_value, str) and body_value:
        return body_value
    return None


def _generated_sections_text(
    document: Mapping[str, object],
) -> str:
    return json.dumps(
        document.get("sections", {}),
        ensure_ascii=False,
        sort_keys=True,
    )


def _active_concept_ids(
    sections: Mapping[str, object],
) -> set[str]:
    active_raw = _list(
        sections.get("activated_concepts"),
        "sections.activated_concepts",
    )

    active: set[str] = set()
    for raw in active_raw:
        concept = _mapping(
            raw,
            "activated concept",
        )
        concept_id = concept.get("concept")
        if not isinstance(concept_id, str) or not concept_id:
            raise ContractFailure(
                "Activated concept is missing concept identifier."
            )
        active.add(concept_id)

    return active


def _validate_citation_contract(
    document: Mapping[str, object],
    sections: Mapping[str, object],
) -> tuple[int, int]:
    citations_raw = _list(
        document.get("claim_level_citations"),
        "FinalResponse.claim_level_citations",
    )
    citation_registry: dict[str, dict[str, object]] = {}

    for raw_citation in citations_raw:
        citation = _mapping(
            raw_citation,
            "claim-level citation",
        )
        ref = citation.get("citation_ref")
        if not isinstance(ref, str) or not ref:
            raise ContractFailure(
                "claim_level_citations contains an invalid citation_ref."
            )
        if ref in citation_registry:
            raise ContractFailure(
                f"Duplicate citation_ref: {ref}."
            )

        if citation.get("corpus_version") != EXPECTED_CORPUS_VERSION:
            raise ContractFailure(
                f"Citation {ref} has wrong corpus_version."
            )

        chunk_id = citation.get("chunk_id")
        source_id = citation.get("source_id")
        domain = citation.get("domain")

        if not isinstance(chunk_id, str) or not chunk_id:
            raise ContractFailure(
                f"Citation {ref} is missing chunk_id."
            )
        if not isinstance(source_id, str) or not source_id:
            raise ContractFailure(
                f"Citation {ref} is missing source_id."
            )
        if domain not in EXPECTED_DOMAINS:
            raise ContractFailure(
                f"Citation {ref} has invalid domain {domain!r}."
            )

        citation_registry[ref] = citation

    perspectives = _mapping(
        sections.get("domain_perspectives"),
        "sections.domain_perspectives",
    )
    if set(perspectives) != EXPECTED_DOMAINS:
        raise ContractFailure(
            "domain_perspectives must contain exactly "
            "science, advaita, and samkhya."
        )

    referenced_refs: set[str] = set()

    for domain, raw_perspective in perspectives.items():
        perspective = _mapping(
            raw_perspective,
            f"{domain} perspective",
        )
        if perspective.get("domain") != domain:
            raise ContractFailure(
                f"{domain} perspective changed domain identity."
            )

        claims = _list(
            perspective.get("claims"),
            f"{domain}.claims",
        )

        for raw_claim in claims:
            claim = _mapping(
                raw_claim,
                f"{domain} claim",
            )

            claim_refs = _list(
                claim.get("citation_refs"),
                f"{domain} claim citation_refs",
            )
            inline_citations = _list(
                claim.get("citations"),
                f"{domain} claim citations",
            )

            normalized_refs: list[str] = []

            for raw_ref in claim_refs:
                if not isinstance(raw_ref, str) or not raw_ref:
                    raise ContractFailure(
                        "Claim citation_refs must contain non-empty strings."
                    )
                normalized_refs.append(raw_ref)
                referenced_refs.add(raw_ref)

            if len(normalized_refs) != len(inline_citations):
                raise ContractFailure(
                    f"{domain} claim citation_refs count does not match "
                    "inline citation count."
                )

            # Frozen Phase 18 contract:
            # - citation_ref lives in top-level claim_level_citations
            # - FinalClaim.citation_refs carries the response-scoped refs
            # - inline FinalClaim.citations are DomainCanonicalCitation and
            #   intentionally do NOT contain citation_ref.
            for raw_ref, raw_citation in zip(
                normalized_refs,
                inline_citations,
                strict=True,
            ):
                citation = _mapping(
                    raw_citation,
                    f"{domain} claim citation",
                )
                inline_domain = citation.get("domain")

                if inline_domain != domain:
                    raise ContractFailure(
                        f"{domain} claim contains cross-domain citation."
                    )

                registry_entry = citation_registry.get(raw_ref)
                if registry_entry is None:
                    raise ContractFailure(
                        f"Unresolved citation_ref: {raw_ref}."
                    )

                for field in (
                    "chunk_id",
                    "source_id",
                    "citation",
                    "corpus_version",
                    "domain",
                ):
                    if citation.get(field) != registry_entry.get(field):
                        raise ContractFailure(
                            f"{domain} inline citation does not match "
                            f"registry entry {raw_ref} for field {field}."
                        )

    comparative = _mapping(
        sections.get("comparative_synthesis"),
        "sections.comparative_synthesis",
    )
    comparisons = _list(
        comparative.get("comparisons"),
        "comparative_synthesis.comparisons",
    )

    for raw_comparison in comparisons:
        comparison = _mapping(
            raw_comparison,
            "comparison",
        )
        for raw_ref in _list(
            comparison.get("citation_refs"),
            "comparison citation_refs",
        ):
            if not isinstance(raw_ref, str):
                raise ContractFailure(
                    "Comparison citation_refs must contain strings."
                )
            referenced_refs.add(raw_ref)

    unresolved = sorted(
        referenced_refs - set(citation_registry)
    )
    if unresolved:
        raise ContractFailure(
            "Unresolved citation_refs: " + ", ".join(unresolved)
        )

    return len(citations_raw), len(comparisons)


def _validate_hard_negative(
    *,
    case: RegressionCase,
    document: Mapping[str, object],
) -> None:
    guard = case.hard_negative_guard
    if guard is None:
        return

    sections = _mapping(
        document.get("sections"),
        "FinalResponse.sections",
    )
    text = _generated_sections_text(document)

    if (
        guard == "atman_purusha"
        and ATMAN_PURUSHA_EQUIVALENCE_RE.search(text)
    ):
        raise ContractFailure(
            "Atman/Purusha false equivalence appeared in generated sections."
        )

    if (
        guard == "cognition"
        and COGNITION_CONSCIOUSNESS_EQUIVALENCE_RE.search(text)
    ):
        raise ContractFailure(
            "Cognition/consciousness false equivalence appeared."
        )

    if (
        guard == "maya_science"
        and SCIENCE_MAYA_PROOF_RE.search(text)
    ):
        raise ContractFailure(
            "Scientific/perceptual evidence was presented as proof of Maya."
        )

    if SCIENCE_METAPHYSICS_PROOF_RE.search(text):
        raise ContractFailure(
            "Science was presented as proving/disproving metaphysical entities."
        )

    if ADVAITA_SAMKHYA_COLLAPSE_RE.search(text):
        raise ContractFailure(
            "Advaita and Samkhya were collapsed into an equivalence."
        )

    if guard == "out_of_corpus":
        coverage = _mapping(
            sections.get("coverage"),
            "sections.coverage",
        )
        if coverage.get("coverage_status") != "Out of Corpus":
            raise ContractFailure(
                "Out-of-Corpus case did not classify Out of Corpus."
            )

        fallback = _mapping(
            sections.get("general_knowledge_fallback"),
            "sections.general_knowledge_fallback",
        )

        if fallback.get("generated_in_phase18") is not False:
            raise ContractFailure(
                "Phase 18 generated general-knowledge fallback."
            )
        if fallback.get("may_use_wth_corpus_citations") is not False:
            raise ContractFailure(
                "Out-of-Corpus fallback may use WTH corpus citations."
            )


def _validate_final_response(
    *,
    case: RegressionCase,
    document: Mapping[str, object],
) -> tuple[str, bool, int, int]:
    question = document.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ContractFailure(
            "FinalResponse.question is missing or empty."
        )

    if question != case.question:
        raise ContractFailure(
            "FinalResponse.question changed from the submitted request."
        )

    if document.get("corpus_version") != EXPECTED_CORPUS_VERSION:
        raise ContractFailure(
            "FinalResponse.corpus_version does not match the frozen corpus."
        )

    validation = _mapping(
        document.get("validation"),
        "FinalResponse.validation",
    )
    if validation.get("passed") is not True:
        raise ContractFailure(
            "FinalResponse.validation.passed is not true."
        )

    sections = _mapping(
        document.get("sections"),
        "FinalResponse.sections",
    )
    coverage = _mapping(
        sections.get("coverage"),
        "FinalResponse.sections.coverage",
    )

    coverage_status = coverage.get("coverage_status")
    if (
        not isinstance(coverage_status, str)
        or coverage_status not in VALID_COVERAGE_STATUSES
    ):
        raise ContractFailure(
            f"Invalid coverage_status: {coverage_status!r}."
        )

    if coverage_status not in case.expected_coverage:
        raise ContractFailure(
            f"Coverage {coverage_status!r} is outside expected "
            f"{case.expected_coverage!r}."
        )

    active = _active_concept_ids(sections)
    missing = set(case.required_active_concepts) - active
    if missing:
        raise ContractFailure(
            "Missing required activated concepts: "
            + ", ".join(sorted(missing))
        )

    citation_count, comparison_count = _validate_citation_contract(
        document,
        sections,
    )

    if comparison_count < case.minimum_comparisons:
        raise ContractFailure(
            f"Expected at least {case.minimum_comparisons} comparisons; "
            f"received {comparison_count}."
        )

    _validate_hard_negative(
        case=case,
        document=document,
    )

    return (
        coverage_status,
        True,
        citation_count,
        comparison_count,
    )


def _validate_expected_error(
    *,
    case: RegressionCase,
    response: httpx.Response,
    document: Mapping[str, object],
) -> None:
    if response.status_code not in case.expected_http_statuses:
        raise ContractFailure(
            f"Expected HTTP {case.expected_http_statuses}; "
            f"received {response.status_code}."
        )

    if case.expected_error_code is None:
        return

    code, _, _ = _error_fields(document)
    if code != case.expected_error_code:
        raise ContractFailure(
            f"Expected error code {case.expected_error_code!r}; "
            f"received {code!r}."
        )


def classify_response(
    *,
    case: RegressionCase,
    response: httpx.Response,
    elapsed_ms: float,
) -> RegressionResult:
    try:
        raw_document: object = response.json()
    except ValueError:
        return RegressionResult(
            case=case.key,
            category=case.category,
            status=RegressionStatus.ENVIRONMENT_FAILURE,
            http_status=response.status_code,
            request_id=response.headers.get("X-Request-ID"),
            coverage_status=None,
            validation_passed=None,
            citation_count=None,
            comparison_count=None,
            error_code=None,
            error_phase=None,
            retry_after_seconds=None,
            elapsed_ms=elapsed_ms,
            detail="Response body was not valid JSON.",
        )

    try:
        document = _mapping(raw_document, "HTTP response")
    except ContractFailure as exc:
        return RegressionResult(
            case=case.key,
            category=case.category,
            status=RegressionStatus.TRUE_REGRESSION,
            http_status=response.status_code,
            request_id=response.headers.get("X-Request-ID"),
            coverage_status=None,
            validation_passed=None,
            citation_count=None,
            comparison_count=None,
            error_code=None,
            error_phase=None,
            retry_after_seconds=None,
            elapsed_ms=elapsed_ms,
            detail=str(exc),
        )

    request_id = _request_id(response, document)
    error_code, error_phase, retry_after = _error_fields(document)

    if case.expected_http_statuses != (200,):
        try:
            _validate_expected_error(
                case=case,
                response=response,
                document=document,
            )
        except ContractFailure as exc:
            return RegressionResult(
                case=case.key,
                category=case.category,
                status=RegressionStatus.TRUE_REGRESSION,
                http_status=response.status_code,
                request_id=request_id,
                coverage_status=None,
                validation_passed=None,
                citation_count=None,
                comparison_count=None,
                error_code=error_code,
                error_phase=error_phase,
                retry_after_seconds=retry_after,
                elapsed_ms=elapsed_ms,
                detail=str(exc),
            )

        return RegressionResult(
            case=case.key,
            category=case.category,
            status=RegressionStatus.PASS,
            http_status=response.status_code,
            request_id=request_id,
            coverage_status=None,
            validation_passed=None,
            citation_count=None,
            comparison_count=None,
            error_code=error_code,
            error_phase=error_phase,
            retry_after_seconds=retry_after,
            elapsed_ms=elapsed_ms,
            detail="Expected API error contract matched.",
        )

    if (
        response.status_code == 429
        and error_code == "provider_rate_limited"
    ):
        return RegressionResult(
            case=case.key,
            category=case.category,
            status=RegressionStatus.PROVIDER_CAPACITY_BLOCKED,
            http_status=429,
            request_id=request_id,
            coverage_status=None,
            validation_passed=None,
            citation_count=None,
            comparison_count=None,
            error_code=error_code,
            error_phase=error_phase,
            retry_after_seconds=retry_after,
            elapsed_ms=elapsed_ms,
            detail=(
                "Provider capacity prevented regression evaluation; "
                "not classified as an application regression."
            ),
        )

    if response.status_code in {502, 503, 504}:
        return RegressionResult(
            case=case.key,
            category=case.category,
            status=RegressionStatus.ENVIRONMENT_FAILURE,
            http_status=response.status_code,
            request_id=request_id,
            coverage_status=None,
            validation_passed=None,
            citation_count=None,
            comparison_count=None,
            error_code=error_code,
            error_phase=error_phase,
            retry_after_seconds=retry_after,
            elapsed_ms=elapsed_ms,
            detail=(
                "External dependency/provider/runtime condition "
                "prevented regression evaluation."
            ),
        )

    if response.status_code != 200:
        return RegressionResult(
            case=case.key,
            category=case.category,
            status=RegressionStatus.TRUE_REGRESSION,
            http_status=response.status_code,
            request_id=request_id,
            coverage_status=None,
            validation_passed=None,
            citation_count=None,
            comparison_count=None,
            error_code=error_code,
            error_phase=error_phase,
            retry_after_seconds=retry_after,
            elapsed_ms=elapsed_ms,
            detail=(
                f"Unexpected HTTP {response.status_code} for a "
                "normal regression case."
            ),
        )

    try:
        (
            coverage_status,
            validation_passed,
            citation_count,
            comparison_count,
        ) = _validate_final_response(
            case=case,
            document=document,
        )
    except ContractFailure as exc:
        return RegressionResult(
            case=case.key,
            category=case.category,
            status=RegressionStatus.TRUE_REGRESSION,
            http_status=200,
            request_id=request_id,
            coverage_status=None,
            validation_passed=False,
            citation_count=None,
            comparison_count=None,
            error_code=None,
            error_phase=None,
            retry_after_seconds=None,
            elapsed_ms=elapsed_ms,
            detail=str(exc),
        )

    return RegressionResult(
        case=case.key,
        category=case.category,
        status=RegressionStatus.PASS,
        http_status=200,
        request_id=request_id,
        coverage_status=coverage_status,
        validation_passed=validation_passed,
        citation_count=citation_count,
        comparison_count=comparison_count,
        error_code=None,
        error_phase=None,
        retry_after_seconds=None,
        elapsed_ms=elapsed_ms,
        detail=(
            "Core HTTP, coverage, concept, citation, domain, and safety "
            "regression checks passed."
        ),
    )


def run_case(
    *,
    client: httpx.Client,
    case: RegressionCase,
) -> RegressionResult:
    started = time.perf_counter()

    try:
        response = client.post(
            API_PATH,
            json={"question": case.question},
        )
    except httpx.HTTPError as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return RegressionResult(
            case=case.key,
            category=case.category,
            status=RegressionStatus.ENVIRONMENT_FAILURE,
            http_status=None,
            request_id=None,
            coverage_status=None,
            validation_passed=None,
            citation_count=None,
            comparison_count=None,
            error_code=None,
            error_phase=None,
            retry_after_seconds=None,
            elapsed_ms=elapsed_ms,
            detail=f"HTTP transport failure: {exc}",
        )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return classify_response(
        case=case,
        response=response,
        elapsed_ms=elapsed_ms,
    )


def selected_cases(args: argparse.Namespace) -> list[RegressionCase]:
    if args.all:
        return list(CASES.values())

    if args.category:
        return [
            case
            for case in CASES.values()
            if case.category == args.category
        ]

    return [CASES[args.case]]


def print_result(result: RegressionResult) -> None:
    print(
        f"[{result.case}] {result.status.value} "
        f"http={result.http_status} "
        f"coverage={result.coverage_status} "
        f"phase={result.error_phase} "
        f"elapsed_ms={result.elapsed_ms:.0f}"
    )

    if result.error_code:
        print(
            f"  error_code={result.error_code} "
            f"retry_after={result.retry_after_seconds}"
        )

    print(f"  {result.detail}")


def write_report(
    *,
    path: Path,
    base_url: str,
    results: list[RegressionResult],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    counts = {
        status.value: sum(
            result.status is status
            for result in results
        )
        for status in RegressionStatus
    }

    payload = {
        "stage": "5.2",
        "api": f"{base_url.rstrip('/')}{API_PATH}",
        "case_count": len(results),
        "counts": counts,
        "results": [
            {
                **asdict(result),
                "status": result.status.value,
            }
            for result in results
        ],
    }

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def exit_code(results: list[RegressionResult]) -> int:
    return (
        1
        if any(
            result.status is RegressionStatus.TRUE_REGRESSION
            for result in results
        )
        else 0
    )


def main() -> int:
    args = parse_args()
    cases = selected_cases(args)
    base_url = args.base_url.rstrip("/")

    print(
        f"Stage 5.2 HTTP core/safety regression: {base_url}{API_PATH}"
    )
    print(
        "Classification: PASS | TRUE_REGRESSION | "
        "PROVIDER_CAPACITY_BLOCKED | ENVIRONMENT_FAILURE"
    )

    results: list[RegressionResult] = []

    with httpx.Client(
        base_url=base_url,
        timeout=args.timeout_seconds,
    ) as client:
        for index, case in enumerate(cases):
            print(
                f"\nRunning {case.key}: {case.question}"
            )
            result = run_case(
                client=client,
                case=case,
            )
            results.append(result)
            print_result(result)

            if (
                args.cooldown_seconds > 0
                and index < len(cases) - 1
            ):
                time.sleep(args.cooldown_seconds)

    print("\nSummary")
    for status in RegressionStatus:
        count = sum(
            result.status is status
            for result in results
        )
        print(f"  {status.value}: {count}")

    if args.report_json:
        write_report(
            path=args.report_json,
            base_url=base_url,
            results=results,
        )
        print(
            f"\nReport: {args.report_json}"
        )

    return exit_code(results)


if __name__ == "__main__":
    raise SystemExit(main())
