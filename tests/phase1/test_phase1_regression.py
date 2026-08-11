from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pytest
import yaml

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
REGRESSION_FILE: Final = PROJECT_ROOT / "tests" / "phase1" / "phase1_regression_questions.yaml"
SNAPSHOT_ROOT: Final = PROJECT_ROOT / "tests" / "phase1" / "fixtures" / "regression"

EXPECTED_SCHEMA_VERSION: Final = "phase1-regression-questions-v1"
EXPECTED_CASE_COUNT: Final = 16

ALLOWED_CATEGORIES: Final = {
    "clear_consciousness",
    "clear_self",
    "clear_reality_appearance",
    "multi_concept",
    "ambiguous",
    "hard_negative",
    "out_of_corpus",
}
ALLOWED_COVERAGE_STATUSES: Final = {
    "Supported",
    "Partially Supported",
    "Out of Corpus",
}
ALLOWED_CONCEPTS: Final = {
    "consciousness",
    "self_identity",
    "reality_appearance",
}
ALLOWED_DOMAINS: Final = {
    "science",
    "advaita",
    "samkhya",
}

ATMAN_PURUSHA_EQUIVALENCE_RE: Final = re.compile(
    r"\b(?:atman|ātman)\b.{0,100}\b"
    r"(?:same|identical|equivalent)\b.{0,100}\b"
    r"(?:purusha|puruṣa)\b"
    r"|"
    r"\b(?:purusha|puruṣa)\b.{0,100}\b"
    r"(?:same|identical|equivalent)\b.{0,100}\b"
    r"(?:atman|ātman)\b",
    re.IGNORECASE | re.DOTALL,
)
COGNITION_CONSCIOUSNESS_EQUIVALENCE_RE: Final = re.compile(
    r"\bcognition\b.{0,80}\b"
    r"(?:same as|identical to|equivalent to|is consciousness)\b"
    r"|"
    r"\bconsciousness\b.{0,80}\b"
    r"(?:same as|identical to|equivalent to|is cognition)\b",
    re.IGNORECASE | re.DOTALL,
)
PERCEPTUAL_MAYA_PROOF_RE: Final = re.compile(
    r"\b(?:perceptual|perception|illusion)\b.{0,140}\b"
    r"(?:prove|proves|proven|demonstrates|establishes)\b.{0,100}\b"
    r"(?:maya|māyā)\b",
    re.IGNORECASE | re.DOTALL,
)
SCIENCE_METAPHYSICS_PROOF_RE: Final = re.compile(
    r"\b(?:science|scientific|neuroscience|neurological|empirical)\b"
    r".{0,160}\b"
    r"(?:prove|proves|proven|establishes|confirms)\b"
    r".{0,120}\b"
    r"(?:atman|ātman|brahman|purusha|puruṣa|maya|māyā|nondual|non-dual)\b",
    re.IGNORECASE | re.DOTALL,
)
SHARED_NONDUALITY_RE: Final = re.compile(
    r"\b(?:advaita(?: vedanta)?\s+and\s+samkhya"
    r"|samkhya\s+and\s+advaita(?: vedanta)?"
    r"|both)\b"
    r".{0,160}\b(?:nondual|non-dual|nonduality|non-duality)\b",
    re.IGNORECASE | re.DOTALL,
)

ALLOWED_EXPECTATION_KEYS: Final = {
    "required_active_concepts",
    "allowed_additional_active_concepts",
    "at_least_one_active_concept_from",
    "allowed_coverage_statuses",
    "expected_coverage_score_range",
    "required_domains",
    "minimum_claim_count",
    "minimum_comparison_count",
    "must_have_corpus_citations",
    "must_have_corpus_citations_if_corpus_answered",
    "must_include_comparative_synthesis",
    "must_preserve_domain_separation",
    "must_preserve_domain_separation_if_answered",
    "must_not_assert_atman_purusha_equivalence",
    "must_not_assert_shared_nonduality_for_advaita_and_samkhya",
    "must_not_assert_science_proves_metaphysics",
    "must_not_assert_cognition_equals_consciousness",
    "must_not_assert_perceptual_construction_proves_maya",
    "must_include_difference_or_non_equivalence",
    "must_not_fabricate_corpus_support",
    "may_be_marked_ambiguous",
    "maximum_coverage_status_if_no_direct_reviewed_evidence",
    "general_knowledge_fallback_may_be_allowed",
    "general_knowledge_must_be_separately_labeled",
    "general_knowledge_must_not_use_wth_corpus_citations",
    "corpus_answer_must_be_forbidden",
}


def require_mapping(
    value: object,
    description: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{description} must be an object")

    result: dict[str, object] = {}
    for key, nested in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{description} contains a non-string key")
        result[key] = nested
    return result


def require_list(
    value: object,
    description: str,
) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{description} must be a list")
    return value


def require_string(
    value: object,
    description: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AssertionError(f"{description} must be a non-empty string")
    return value.strip()


def optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def string_set(
    value: object,
    description: str,
) -> set[str]:
    return {require_string(item, description) for item in require_list(value, description)}


def load_yaml_mapping(path: Path) -> dict[str, object]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return require_mapping(raw, f"YAML document {path}")


def load_json_mapping(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return require_mapping(raw, f"JSON document {path}")


REGRESSION_DOCUMENT: Final = load_yaml_mapping(REGRESSION_FILE)
REGRESSION_CASES: Final = tuple(
    require_mapping(item, "regression case")
    for item in require_list(
        REGRESSION_DOCUMENT.get("cases"),
        "regression cases",
    )
)


def case_id(case: Mapping[str, object]) -> str:
    return require_string(
        case.get("id"),
        "case id",
    )


def case_question(
    case: Mapping[str, object],
) -> str:
    return require_string(
        case.get("question"),
        f"{case_id(case)} question",
    )


def case_expectations(
    case: Mapping[str, object],
) -> dict[str, object]:
    return require_mapping(
        case.get("expected"),
        f"{case_id(case)} expected",
    )


def snapshot_path(
    case: Mapping[str, object],
) -> Path:
    return SNAPSHOT_ROOT / case_id(case) / "final_response.json"


def active_concepts_from_response(
    response: Mapping[str, object],
) -> set[str]:
    sections = require_mapping(
        response.get("sections"),
        "final response sections",
    )
    activated = require_list(
        sections.get("activated_concepts"),
        "activated concepts",
    )

    result: set[str] = set()
    for item_raw in activated:
        item = require_mapping(
            item_raw,
            "activated concept",
        )
        concept = require_string(
            item.get("concept"),
            "activated concept id",
        )
        result.add(concept)

    return result


def coverage_from_response(
    response: Mapping[str, object],
) -> dict[str, object]:
    sections = require_mapping(
        response.get("sections"),
        "final response sections",
    )
    return require_mapping(
        sections.get("coverage"),
        "coverage section",
    )


def domain_perspectives_from_response(
    response: Mapping[str, object],
) -> dict[str, object]:
    sections = require_mapping(
        response.get("sections"),
        "final response sections",
    )
    return require_mapping(
        sections.get("domain_perspectives"),
        "domain perspectives",
    )


def synthesis_from_response(
    response: Mapping[str, object],
) -> dict[str, object]:
    sections = require_mapping(
        response.get("sections"),
        "final response sections",
    )
    return require_mapping(
        sections.get("comparative_synthesis"),
        "comparative synthesis",
    )


def final_response_text(
    response: Mapping[str, object],
) -> str:
    sections = require_mapping(
        response.get("sections"),
        "final response sections",
    )
    # Exclude claim-level citation bibliography metadata from semantic
    # assertions. We care about generated/assembled response semantics here.
    return json.dumps(
        sections,
        ensure_ascii=False,
        sort_keys=True,
    )


def corpus_citations(
    response: Mapping[str, object],
) -> list[object]:
    return require_list(
        response.get("claim_level_citations"),
        "claim_level_citations",
    )


def count_claims(
    response: Mapping[str, object],
) -> int:
    domains = domain_perspectives_from_response(response)
    total = 0

    for domain in ALLOWED_DOMAINS:
        raw = domains.get(domain)
        if not isinstance(raw, Mapping):
            continue
        perspective = require_mapping(
            raw,
            f"{domain} perspective",
        )
        claims = perspective.get("claims")
        if isinstance(claims, list):
            total += len(claims)

    return total


def comparisons_from_response(
    response: Mapping[str, object],
) -> list[object]:
    synthesis = synthesis_from_response(response)
    raw = synthesis.get("comparisons")
    if not isinstance(raw, list):
        return []
    return raw


def citation_refs_from_response(
    response: Mapping[str, object],
) -> set[str]:
    refs: set[str] = set()
    for citation_raw in corpus_citations(response):
        citation = require_mapping(
            citation_raw,
            "claim-level citation",
        )
        refs.add(
            require_string(
                citation.get("citation_ref"),
                "citation_ref",
            )
        )
    return refs


def assert_claim_citation_refs_resolve(
    response: Mapping[str, object],
) -> None:
    valid_refs = citation_refs_from_response(response)
    domains = domain_perspectives_from_response(response)

    for domain, raw in domains.items():
        if not isinstance(raw, Mapping):
            continue
        perspective = require_mapping(
            raw,
            f"{domain} perspective",
        )
        claims = perspective.get("claims")
        if not isinstance(claims, list):
            continue

        for claim_raw in claims:
            claim = require_mapping(
                claim_raw,
                f"{domain} claim",
            )
            refs_raw = require_list(
                claim.get("citation_refs"),
                f"{domain} claim citation_refs",
            )
            assert refs_raw, f"{domain} claim has no citation refs"
            for ref_raw in refs_raw:
                ref = require_string(
                    ref_raw,
                    "claim citation ref",
                )
                assert ref in valid_refs, (
                    f"Claim citation {ref!r} does not resolve to the final citation registry"
                )


def assert_domain_separation(
    response: Mapping[str, object],
) -> None:
    domains = domain_perspectives_from_response(response)
    citations_by_ref: dict[str, str] = {}

    for citation_raw in corpus_citations(response):
        citation = require_mapping(
            citation_raw,
            "claim-level citation",
        )
        ref = require_string(
            citation.get("citation_ref"),
            "citation_ref",
        )
        domain = require_string(
            citation.get("domain"),
            "citation domain",
        )
        citations_by_ref[ref] = domain

    for domain, raw in domains.items():
        if domain not in ALLOWED_DOMAINS:
            raise AssertionError(f"Unknown domain perspective {domain!r}")
        if not isinstance(raw, Mapping):
            continue
        perspective = require_mapping(
            raw,
            f"{domain} perspective",
        )
        claims = perspective.get("claims")
        if not isinstance(claims, list):
            continue

        for claim_raw in claims:
            claim = require_mapping(
                claim_raw,
                f"{domain} claim",
            )
            refs_raw = require_list(
                claim.get("citation_refs"),
                f"{domain} claim citation refs",
            )
            for ref_raw in refs_raw:
                ref = require_string(
                    ref_raw,
                    "citation ref",
                )
                assert citations_by_ref[ref] == domain, (
                    f"{domain} claim uses citation {ref!r} from {citations_by_ref[ref]!r}"
                )


def assert_no_fabricated_corpus_support(
    response: Mapping[str, object],
) -> None:
    coverage = coverage_from_response(response)
    status = require_string(
        coverage.get("coverage_status"),
        "coverage status",
    )
    if status != "Out of Corpus":
        return

    domains = domain_perspectives_from_response(response)
    assert not domains, "Out-of-Corpus response contains corpus domain perspectives"
    assert not corpus_citations(response), "Out-of-Corpus response contains WTH corpus citations"
    assert not comparisons_from_response(response), (
        "Out-of-Corpus response contains corpus comparative synthesis"
    )


def validate_regression_result(
    case: Mapping[str, object],
    response: Mapping[str, object],
) -> None:
    """Validate one frozen regression case against a Phase 18 response."""

    identifier = case_id(case)
    expected = case_expectations(case)

    assert require_string(
        response.get("question"),
        "final response question",
    ) == case_question(case), f"{identifier}: response question mismatch"

    validation = require_mapping(
        response.get("validation"),
        "final response validation",
    )
    assert validation.get("passed") is True, f"{identifier}: Phase 18 validation did not pass"

    active = active_concepts_from_response(response)

    required_raw = expected.get("required_active_concepts")
    if isinstance(required_raw, list):
        required = string_set(
            required_raw,
            "required active concepts",
        )
        assert required.issubset(active), (
            f"{identifier}: missing required concepts {sorted(required - active)}"
        )

        allowed_extra_raw = expected.get(
            "allowed_additional_active_concepts",
            [],
        )
        allowed_extra = string_set(
            allowed_extra_raw,
            "allowed additional concepts",
        )
        unexpected = active - required - allowed_extra
        assert not unexpected, f"{identifier}: unexpected active concepts {sorted(unexpected)}"

    at_least_one_raw = expected.get("at_least_one_active_concept_from")
    if isinstance(at_least_one_raw, list):
        candidates = string_set(
            at_least_one_raw,
            "at least one active concept",
        )
        assert active.intersection(candidates), (
            f"{identifier}: none of the expected ambiguous concepts activated"
        )

    coverage = coverage_from_response(response)
    status = require_string(
        coverage.get("coverage_status"),
        "coverage status",
    )
    allowed_statuses = string_set(
        expected.get("allowed_coverage_statuses"),
        "allowed coverage statuses",
    )
    assert status in allowed_statuses, (
        f"{identifier}: coverage {status!r} not in {sorted(allowed_statuses)}"
    )

    score_range_raw = expected.get("expected_coverage_score_range")
    if isinstance(score_range_raw, Mapping):
        score_range = require_mapping(
            score_range_raw,
            "coverage score range",
        )
        score = optional_float(coverage.get("coverage_score"))
        assert score is not None, f"{identifier}: coverage score missing"
        minimum = optional_float(score_range.get("minimum"))
        maximum = optional_float(score_range.get("maximum"))
        if minimum is not None:
            assert score >= minimum, f"{identifier}: coverage score {score} < {minimum}"
        if maximum is not None:
            assert score <= maximum, f"{identifier}: coverage score {score} > {maximum}"

    required_domains_raw = expected.get("required_domains")
    if isinstance(required_domains_raw, list):
        required_domains = string_set(
            required_domains_raw,
            "required domains",
        )
        domains = set(domain_perspectives_from_response(response))
        assert required_domains.issubset(domains), (
            f"{identifier}: missing required domains {sorted(required_domains - domains)}"
        )

    minimum_claim_count = expected.get("minimum_claim_count")
    if isinstance(minimum_claim_count, int):
        assert count_claims(response) >= minimum_claim_count

    minimum_comparison_count = expected.get("minimum_comparison_count")
    if isinstance(
        minimum_comparison_count,
        int,
    ):
        assert len(comparisons_from_response(response)) >= minimum_comparison_count

    if expected.get("must_have_corpus_citations") is True:
        assert corpus_citations(response), f"{identifier}: corpus citations missing"

    if (
        expected.get("must_have_corpus_citations_if_corpus_answered") is True
        and status != "Out of Corpus"
    ):
        assert corpus_citations(response)

    if expected.get("must_include_comparative_synthesis") is True:
        assert comparisons_from_response(response), f"{identifier}: comparative synthesis missing"

    if expected.get("must_preserve_domain_separation") is True or (
        expected.get("must_preserve_domain_separation_if_answered") is True
        and status != "Out of Corpus"
    ):
        assert_domain_separation(response)

    assert_claim_citation_refs_resolve(response)

    text = final_response_text(response)

    if expected.get("must_not_assert_atman_purusha_equivalence") is True:
        assert not ATMAN_PURUSHA_EQUIVALENCE_RE.search(text), (
            f"{identifier}: Atman/Purusha equivalence appears in final response"
        )

    if expected.get("must_not_assert_cognition_equals_consciousness") is True:
        assert not (COGNITION_CONSCIOUSNESS_EQUIVALENCE_RE.search(text)), (
            f"{identifier}: cognition/consciousness equivalence appears in final response"
        )

    if expected.get("must_not_assert_perceptual_construction_proves_maya") is True:
        assert not PERCEPTUAL_MAYA_PROOF_RE.search(text), (
            f"{identifier}: perceptual evidence is presented as proof of Maya"
        )

    if expected.get("must_not_assert_science_proves_metaphysics") is True:
        assert not SCIENCE_METAPHYSICS_PROOF_RE.search(text), (
            f"{identifier}: scientific evidence is presented as metaphysical proof"
        )

    if expected.get("must_not_assert_shared_nonduality_for_advaita_and_samkhya") is True:
        assert not SHARED_NONDUALITY_RE.search(text), (
            f"{identifier}: non-duality is presented as shared by Advaita and Samkhya"
        )

    if expected.get("must_include_difference_or_non_equivalence") is True:
        synthesis = synthesis_from_response(response)
        non_equivalences = synthesis.get("comparisons")
        has_non_equivalence = False
        if isinstance(non_equivalences, list):
            for item_raw in non_equivalences:
                if not isinstance(
                    item_raw,
                    Mapping,
                ):
                    continue
                item = require_mapping(
                    item_raw,
                    "comparison",
                )
                category = item.get("category")
                explanation = item.get("explanation")
                if category in {
                    "non_equivalence",
                    "direct_tension",
                    "partial_overlap",
                }:
                    has_non_equivalence = True
                    break
                if isinstance(explanation, str) and re.search(
                    r"\b(?:differ|different|distinct|not the same)\b",
                    explanation,
                    re.IGNORECASE,
                ):
                    has_non_equivalence = True
                    break

        assert has_non_equivalence, f"{identifier}: expected an explicit difference/non-equivalence"

    if expected.get("must_not_fabricate_corpus_support") is True:
        assert_no_fabricated_corpus_support(response)

    if expected.get("corpus_answer_must_be_forbidden") is True:
        assert status == "Out of Corpus"
        assert_no_fabricated_corpus_support(response)

    fallback = require_mapping(
        require_mapping(
            response.get("sections"),
            "sections",
        ).get("general_knowledge_fallback"),
        "general knowledge fallback",
    )

    if expected.get("general_knowledge_fallback_may_be_allowed") is True:
        # "May be allowed" means either state is accepted,
        # but Phase 18 itself must never generate it.
        assert fallback.get("generated_in_phase18") is False

    if expected.get("general_knowledge_must_be_separately_labeled") is True:
        assert fallback.get("must_be_clearly_labeled") is True

    if expected.get("general_knowledge_must_not_use_wth_corpus_citations") is True:
        assert fallback.get("may_use_wth_corpus_citations") is False


# ---------------------------------------------------------------------------
# Regression-definition tests
# ---------------------------------------------------------------------------


def test_regression_suite_metadata_is_frozen_candidate() -> None:
    assert REGRESSION_DOCUMENT.get("schema_version") == EXPECTED_SCHEMA_VERSION
    assert REGRESSION_DOCUMENT.get("phase") == 19
    assert REGRESSION_DOCUMENT.get("status") == "frozen_candidate"


def test_regression_suite_has_exactly_16_cases() -> None:
    assert len(REGRESSION_CASES) == EXPECTED_CASE_COUNT

    summary = require_mapping(
        REGRESSION_DOCUMENT.get("summary"),
        "summary",
    )
    assert summary.get("total_cases") == EXPECTED_CASE_COUNT


def test_regression_case_ids_and_questions_are_unique() -> None:
    ids = [case_id(case) for case in REGRESSION_CASES]
    questions = [case_question(case) for case in REGRESSION_CASES]

    assert len(ids) == len(set(ids))
    assert len(questions) == len(set(questions))


@pytest.mark.parametrize(
    "case",
    REGRESSION_CASES,
    ids=[case_id(case) for case in REGRESSION_CASES],
)
def test_regression_case_definition_is_valid(
    case: Mapping[str, object],
) -> None:
    identifier = case_id(case)
    category = require_string(
        case.get("category"),
        f"{identifier} category",
    )
    assert category in ALLOWED_CATEGORIES

    question = case_question(case)
    assert question.endswith("?")

    expected = case_expectations(case)
    unexpected_keys = set(expected) - ALLOWED_EXPECTATION_KEYS
    assert not unexpected_keys, f"{identifier}: unknown expectation keys {sorted(unexpected_keys)}"

    statuses = string_set(
        expected.get("allowed_coverage_statuses"),
        f"{identifier} allowed coverage statuses",
    )
    assert statuses
    assert statuses.issubset(ALLOWED_COVERAGE_STATUSES)

    for key in (
        "required_active_concepts",
        "allowed_additional_active_concepts",
        "at_least_one_active_concept_from",
    ):
        raw = expected.get(key)
        if isinstance(raw, list):
            values = string_set(
                raw,
                f"{identifier} {key}",
            )
            assert values.issubset(ALLOWED_CONCEPTS)

    required_domains_raw = expected.get("required_domains")
    if isinstance(required_domains_raw, list):
        domains = string_set(
            required_domains_raw,
            f"{identifier} required domains",
        )
        assert domains.issubset(ALLOWED_DOMAINS)

    score_range_raw = expected.get("expected_coverage_score_range")
    if isinstance(score_range_raw, Mapping):
        score_range = require_mapping(
            score_range_raw,
            f"{identifier} score range",
        )
        minimum = optional_float(score_range.get("minimum"))
        maximum = optional_float(score_range.get("maximum"))
        assert minimum is not None
        assert maximum is not None
        assert 0.0 <= minimum <= maximum <= 100.0


def test_regression_summary_matches_case_categories() -> None:
    actual = Counter(
        require_string(
            case.get("category"),
            "case category",
        )
        for case in REGRESSION_CASES
    )

    summary = require_mapping(
        REGRESSION_DOCUMENT.get("summary"),
        "summary",
    )
    expected_raw = require_mapping(
        summary.get("categories"),
        "summary categories",
    )
    expected = {
        category: int(count) for category, count in expected_raw.items() if isinstance(count, int)
    }

    assert dict(actual) == expected


# ---------------------------------------------------------------------------
# Snapshot/result tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    REGRESSION_CASES,
    ids=[case_id(case) for case in REGRESSION_CASES],
)
def test_regression_snapshot_when_available(
    case: Mapping[str, object],
) -> None:
    """
    Validate a captured Phase 18 result when one exists.

    The normal offline regression suite does not call Gemini or Groq.
    Phase 19 integration/live-E2E tooling will populate:

      tests/phase1/fixtures/regression/<case_id>/final_response.json

    Once a snapshot exists it is validated automatically on every run.
    """

    path = snapshot_path(case)
    if not path.is_file():
        pytest.skip(f"Regression snapshot not captured yet: {path}")

    response = load_json_mapping(path)
    validate_regression_result(
        case,
        response,
    )
