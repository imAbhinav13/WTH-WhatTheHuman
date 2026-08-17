"""Stage 4.4 live HTTP smoke/exit-gate runner for ``POST /api/query``.

Run the FastAPI server separately:

    uv run uvicorn apps.api.main:app --host 127.0.0.1 --port 8000

Then, from another PowerShell window:

    uv run python -m scripts.run_stage44_http_smoke --case multi

The default executes only one real provider-backed request. ``--all`` runs the
full nine-case Stage 4 exit-gate set and intentionally sleeps between cases to
reduce provider TPM collisions.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import httpx


EXPECTED_CORPUS_VERSION: Final = (
    "phase1_active_corpus_v1"
)
EXPECTED_DOMAINS: Final = {
    "science",
    "advaita",
    "samkhya",
}
ALLOWED_COVERAGE: Final = {
    "Supported",
    "Partially Supported",
    "Out of Corpus",
}

ATMAN_PURUSHA_EQUIVALENCE_RE: Final = re.compile(
    r"\b(?:atman|ātman)\b.{0,100}\b"
    r"(?:same|identical|equivalent|same concept|same entity)\b"
    r".{0,100}\b(?:purusha|puruṣa)\b"
    r"|"
    r"\b(?:purusha|puruṣa)\b.{0,100}\b"
    r"(?:same|identical|equivalent|same concept|same entity)\b"
    r".{0,100}\b(?:atman|ātman)\b",
    re.IGNORECASE | re.DOTALL,
)

COGNITION_CONSCIOUSNESS_EQUIVALENCE_RE: Final = re.compile(
    r"\bcognition\b.{0,100}\b"
    r"(?:same|identical|equivalent)\b.{0,100}\bconsciousness\b"
    r"|"
    r"\bconsciousness\b.{0,100}\b"
    r"(?:same|identical|equivalent)\b.{0,100}\bcognition\b",
    re.IGNORECASE | re.DOTALL,
)

SCIENCE_MAYA_PROOF_RE: Final = re.compile(
    r"\b(?:science|scientific|neuroscience|empirical|perceptual)\b"
    r".{0,180}\b(?:prove|proves|proven|establishes|confirms)\b"
    r".{0,140}\b(?:maya|māyā)\b",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class SmokeCase:
    key: str
    question: str
    expected_coverage: frozenset[str]
    required_active_concepts: frozenset[str] = frozenset()
    minimum_comparisons: int = 0
    guard: str | None = None


CASES: Final[dict[str, SmokeCase]] = {
    "consciousness": SmokeCase(
        key="consciousness",
        question=(
            "How do Science, Advaita Vedanta, and Samkhya "
            "understand consciousness?"
        ),
        expected_coverage=frozenset(
            {"Supported", "Partially Supported"}
        ),
        required_active_concepts=frozenset(
            {"consciousness"}
        ),
    ),
    "self": SmokeCase(
        key="self",
        question=(
            "How is the self understood in Science, "
            "Advaita Vedanta, and Samkhya?"
        ),
        expected_coverage=frozenset(
            {"Supported", "Partially Supported"}
        ),
        required_active_concepts=frozenset(
            {"self_identity"}
        ),
    ),
    "reality": SmokeCase(
        key="reality",
        question=(
            "How do the three perspectives distinguish "
            "experienced appearance from reality?"
        ),
        expected_coverage=frozenset(
            {"Supported", "Partially Supported"}
        ),
        required_active_concepts=frozenset(
            {"reality_appearance"}
        ),
    ),
    "multi": SmokeCase(
        key="multi",
        question=(
            "How is consciousness related to the self "
            "and experienced reality?"
        ),
        expected_coverage=frozenset(
            {"Supported", "Partially Supported"}
        ),
        required_active_concepts=frozenset(
            {
                "consciousness",
                "self_identity",
                "reality_appearance",
            }
        ),
        minimum_comparisons=9,
    ),
    "ambiguous": SmokeCase(
        key="ambiguous",
        question="What is the observer?",
        expected_coverage=frozenset(
            {
                "Supported",
                "Partially Supported",
                "Out of Corpus",
            }
        ),
    ),
    "atman_purusha": SmokeCase(
        key="atman_purusha",
        question=(
            "Are Atman and Purusha the same concept?"
        ),
        expected_coverage=frozenset(
            {"Supported", "Partially Supported"}
        ),
        required_active_concepts=frozenset(
            {"self_identity"}
        ),
        guard="atman_purusha",
    ),
    "cognition": SmokeCase(
        key="cognition",
        question=(
            "Is cognition the same thing as consciousness?"
        ),
        expected_coverage=frozenset(
            {
                "Supported",
                "Partially Supported",
                "Out of Corpus",
            }
        ),
        guard="cognition",
    ),
    "maya_science": SmokeCase(
        key="maya_science",
        question=(
            "Does a perceptual illusion prove that "
            "Advaita's Maya is scientifically true?"
        ),
        expected_coverage=frozenset(
            {"Partially Supported", "Out of Corpus"}
        ),
        guard="maya_science",
    ),
    "out_of_corpus": SmokeCase(
        key="out_of_corpus",
        question=(
            "Does quantum entanglement prove Advaita Vedanta?"
        ),
        expected_coverage=frozenset(
            {"Out of Corpus"}
        ),
        guard="out_of_corpus",
    ),
}


class SmokeFailure(
    AssertionError
):
    """Raised when the Stage 4 HTTP contract/exit gate fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage 4.4 live HTTP checks against POST /api/query."
        )
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--case",
        choices=tuple(
            CASES.keys()
        ),
        default="multi",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Run the full nine-case Stage 4 exit-gate suite."
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
        default=65.0,
        help=(
            "Pause between live cases when --all is used."
        ),
    )
    return parser.parse_args()


def require_mapping(
    value: object,
    description: str,
) -> dict[str, object]:
    if not isinstance(
        value,
        Mapping,
    ):
        raise SmokeFailure(
            f"{description} must be an object."
        )

    return {
        str(key): nested
        for key, nested in value.items()
    }


def require_list(
    value: object,
    description: str,
) -> list[object]:
    if not isinstance(
        value,
        list,
    ):
        raise SmokeFailure(
            f"{description} must be a list."
        )

    return value


def generated_sections_text(
    response: Mapping[str, object],
) -> str:
    return json.dumps(
        response.get(
            "sections",
            {},
        ),
        ensure_ascii=False,
        sort_keys=True,
    )


def validate_citations(
    response: Mapping[str, object],
) -> None:
    registry_raw = require_list(
        response.get(
            "claim_level_citations"
        ),
        "claim_level_citations",
    )

    registry: dict[str, dict[str, object]] = {}

    for raw in registry_raw:
        citation = require_mapping(
            raw,
            "claim-level citation",
        )
        ref = citation.get(
            "citation_ref"
        )

        if (
            not isinstance(ref, str)
            or not ref
        ):
            raise SmokeFailure(
                "Citation registry contains an invalid citation_ref."
            )

        if ref in registry:
            raise SmokeFailure(
                f"Duplicate citation_ref: {ref}"
            )

        registry[ref] = citation

    sections = require_mapping(
        response.get(
            "sections"
        ),
        "sections",
    )
    domain_perspectives = require_mapping(
        sections.get(
            "domain_perspectives"
        ),
        "domain_perspectives",
    )

    if set(
        domain_perspectives
    ) != EXPECTED_DOMAINS:
        raise SmokeFailure(
            "Domain perspectives do not contain exactly "
            "science/advaita/samkhya."
        )

    referenced: set[str] = set()

    for domain, raw_perspective in (
        domain_perspectives.items()
    ):
        perspective = require_mapping(
            raw_perspective,
            f"{domain} perspective",
        )

        if (
            perspective.get("domain")
            != domain
        ):
            raise SmokeFailure(
                f"{domain} perspective changed domain identity."
            )

        claims = require_list(
            perspective.get(
                "claims"
            ),
            f"{domain} claims",
        )

        for raw_claim in claims:
            claim = require_mapping(
                raw_claim,
                f"{domain} claim",
            )

            for ref in require_list(
                claim.get(
                    "citation_refs"
                ),
                f"{domain} claim citation_refs",
            ):
                if not isinstance(
                    ref,
                    str,
                ):
                    raise SmokeFailure(
                        "Claim citation_ref must be a string."
                    )
                referenced.add(
                    ref
                )

            for raw_citation in require_list(
                claim.get(
                    "citations"
                ),
                f"{domain} claim citations",
            ):
                citation = require_mapping(
                    raw_citation,
                    f"{domain} claim citation",
                )
                if (
                    citation.get("domain")
                    != domain
                ):
                    raise SmokeFailure(
                        f"{domain} claim contains a cross-domain citation."
                    )

    comparative = require_mapping(
        sections.get(
            "comparative_synthesis"
        ),
        "comparative_synthesis",
    )

    for raw_comparison in require_list(
        comparative.get(
            "comparisons"
        ),
        "comparative comparisons",
    ):
        comparison = require_mapping(
            raw_comparison,
            "comparison",
        )
        for ref in require_list(
            comparison.get(
                "citation_refs"
            ),
            "comparison citation_refs",
        ):
            if isinstance(
                ref,
                str,
            ):
                referenced.add(
                    ref
                )

    missing = sorted(
        referenced - set(
            registry
        )
    )

    if missing:
        raise SmokeFailure(
            "Citation refs do not resolve: "
            + ", ".join(
                missing
            )
        )


def validate_case(
    case: SmokeCase,
    response: Mapping[str, object],
) -> tuple[str, int]:
    if (
        response.get("question")
        != case.question
    ):
        raise SmokeFailure(
            f"{case.key}: response question changed."
        )

    if (
        response.get("corpus_version")
        != EXPECTED_CORPUS_VERSION
    ):
        raise SmokeFailure(
            f"{case.key}: wrong corpus version."
        )

    validation = require_mapping(
        response.get(
            "validation"
        ),
        "validation",
    )

    if validation.get(
        "passed"
    ) is not True:
        raise SmokeFailure(
            f"{case.key}: FinalResponse validation did not pass."
        )

    sections = require_mapping(
        response.get(
            "sections"
        ),
        "sections",
    )
    coverage = require_mapping(
        sections.get(
            "coverage"
        ),
        "coverage",
    )

    status = coverage.get(
        "coverage_status"
    )

    if not isinstance(
        status,
        str,
    ):
        raise SmokeFailure(
            f"{case.key}: coverage_status missing."
        )

    if status not in ALLOWED_COVERAGE:
        raise SmokeFailure(
            f"{case.key}: invalid coverage status {status!r}."
        )

    if status not in case.expected_coverage:
        raise SmokeFailure(
            f"{case.key}: coverage {status!r} not in "
            f"{sorted(case.expected_coverage)}."
        )

    active_raw = require_list(
        sections.get(
            "activated_concepts"
        ),
        "activated_concepts",
    )
    active = {
        str(
            require_mapping(
                item,
                "activated concept",
            ).get(
                "concept"
            )
        )
        for item in active_raw
    }

    missing_concepts = (
        case.required_active_concepts
        - active
    )

    if missing_concepts:
        raise SmokeFailure(
            f"{case.key}: missing active concepts "
            f"{sorted(missing_concepts)}."
        )

    comparative = require_mapping(
        sections.get(
            "comparative_synthesis"
        ),
        "comparative_synthesis",
    )
    comparisons = require_list(
        comparative.get(
            "comparisons"
        ),
        "comparisons",
    )

    if (
        len(comparisons)
        < case.minimum_comparisons
    ):
        raise SmokeFailure(
            f"{case.key}: expected at least "
            f"{case.minimum_comparisons} comparisons, "
            f"received {len(comparisons)}."
        )

    validate_citations(
        response
    )

    text = generated_sections_text(
        response
    )

    if (
        case.guard == "atman_purusha"
        and ATMAN_PURUSHA_EQUIVALENCE_RE.search(
            text
        )
    ):
        raise SmokeFailure(
            "Atman/Purusha false equivalence appeared in generated sections."
        )

    if (
        case.guard == "cognition"
        and COGNITION_CONSCIOUSNESS_EQUIVALENCE_RE.search(
            text
        )
    ):
        raise SmokeFailure(
            "Cognition/consciousness false equivalence appeared."
        )

    if (
        case.guard == "maya_science"
        and SCIENCE_MAYA_PROOF_RE.search(
            text
        )
    ):
        raise SmokeFailure(
            "Scientific/perceptual evidence was presented as proof of Maya."
        )

    if case.guard == "out_of_corpus":
        if status != "Out of Corpus":
            raise SmokeFailure(
                "Out-of-Corpus case did not classify Out of Corpus."
            )

        fallback = require_mapping(
            sections.get(
                "general_knowledge_fallback"
            ),
            "general_knowledge_fallback",
        )

        if (
            fallback.get(
                "generated_in_phase18"
            )
            is not False
        ):
            raise SmokeFailure(
                "Phase 18 generated general-knowledge fallback."
            )

        if (
            fallback.get(
                "may_use_wth_corpus_citations"
            )
            is not False
        ):
            raise SmokeFailure(
                "Out-of-Corpus fallback may use WTH citations."
            )

    return status, len(
        comparisons
    )


def run_case(
    *,
    client: httpx.Client,
    case: SmokeCase,
) -> None:
    print(
        f"\n[{case.key}] {case.question}"
    )

    response = client.post(
        "/api/query",
        json={
            "question": case.question,
        },
    )

    if response.status_code != 200:
        try:
            body = response.json()
        except ValueError:
            body = response.text

        raise SmokeFailure(
            f"{case.key}: HTTP {response.status_code}: {body}"
        )

    raw: object = response.json()
    document = require_mapping(
        raw,
        "FinalResponse",
    )

    status, comparison_count = (
        validate_case(
            case,
            document,
        )
    )

    request_id = response.headers.get(
        "X-Request-ID",
        "<missing>",
    )

    print(
        "PASS "
        f"coverage={status} "
        f"comparisons={comparison_count} "
        f"request_id={request_id}"
    )


def main() -> int:
    args = parse_args()

    selected = (
        list(
            CASES.values()
        )
        if args.all
        else [
            CASES[
                args.case
            ]
        ]
    )

    base_url = (
        args.base_url.rstrip("/")
    )

    try:
        with httpx.Client(
            base_url=base_url,
            timeout=args.timeout_seconds,
        ) as client:
            for index, case in enumerate(
                selected
            ):
                run_case(
                    client=client,
                    case=case,
                )

                if (
                    args.all
                    and index
                    < len(selected) - 1
                    and args.cooldown_seconds
                    > 0
                ):
                    print(
                        "Cooldown "
                        f"{args.cooldown_seconds:.0f}s "
                        "before next provider-backed case..."
                    )
                    time.sleep(
                        args.cooldown_seconds
                    )

    except (
        httpx.HTTPError,
        SmokeFailure,
    ) as exc:
        print(
            f"\nStage 4.4 HTTP smoke FAILED: {exc}",
            file=sys.stderr,
        )
        return 1

    print(
        "\nStage 4.4 HTTP smoke PASS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
