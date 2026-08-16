from __future__ import annotations

import json
from dataclasses import dataclass

from apps.api.models.runtime_contracts import (
    CoverageManifest,
    CoverageResult,
    DomainResponses,
    EvidencePackage,
    FinalResponse,
    GenerationManifest,
    RetrievalManifest,
    SynthesisManifest,
    SynthesisResult,
)
from apps.api.services.phase18_core import (
    ASSEMBLY_VERSION,
    DOMAINS,
    AssemblyError,
    attach_citation_refs,
    build_claim_index,
    build_evidence_index,
    citation_registry,
    final_sections,
    markdown_response,
    parse_coverage,
    parse_domain_responses,
    parse_query_activation,
    parse_synthesis,
    require_list,
    require_mapping,
    require_string,
    same_corpus_version,
    same_question,
    utc_now,
)


@dataclass(frozen=True, slots=True)
class ResponseAssemblyResult:
    """Deterministic Phase 18 runtime output."""

    response: FinalResponse
    markdown: str


class ResponseAssemblyService:
    """Assemble the validated Phase 18 response from runtime objects only."""

    def assemble(
        self,
        *,
        evidence_package: EvidencePackage,
        retrieval_manifest: RetrievalManifest,
        domain_responses: DomainResponses,
        generation_manifest: GenerationManifest,
        synthesis: SynthesisResult,
        synthesis_manifest: SynthesisManifest,
        coverage: CoverageResult,
        coverage_manifest: CoverageManifest,
        generated_at: str | None = None,
        raise_on_validation_failure: bool = True,
    ) -> ResponseAssemblyResult:
        """Run Phase 18 without local artifact reads or writes.

        ``generated_at`` exists so artifact/service parity tests can inject the
        frozen Phase 18 timestamp. Normal runtime callers should omit it.
        """

        evidence_document = _dump_contract(evidence_package)
        retrieval_manifest_document = _dump_contract(retrieval_manifest)
        domain_responses_document = _dump_contract(domain_responses)
        generation_manifest_document = _dump_contract(generation_manifest)
        synthesis_document = _dump_contract(synthesis)
        synthesis_manifest_document = _dump_contract(synthesis_manifest)
        coverage_document = _dump_contract(coverage)
        coverage_manifest_document = _dump_contract(coverage_manifest)

        question = require_string(
            evidence_document.get("question"),
            "Phase 14 question",
        )
        corpus_version = require_string(
            evidence_document.get("corpus_version"),
            "Phase 14 corpus_version",
        )

        (
            query_activation,
            active_concepts,
            weights,
        ) = parse_query_activation(evidence_document)

        evidence_index, _chunk_domain = build_evidence_index(
            evidence_document,
            corpus_version=corpus_version,
        )

        _validate_manifest_document(
            retrieval_manifest_document,
            expected_phase="phase_14_build_retrieval_by_concept_and_domain",
            allowed_statuses={"evaluation_complete"},
            description="Phase 14 manifest",
        )
        same_corpus_version(
            expected=corpus_version,
            actual=require_string(
                retrieval_manifest_document.get("corpus_version"),
                "Phase 14 manifest corpus_version",
            ),
            description="Phase 14 manifest",
        )

        retrieval_exit_gate = require_mapping(
            retrieval_manifest_document.get("exit_gate"),
            "Phase 14 exit_gate",
        )

        for field_name in (
            "only_active_chunks_retrieved",
            "domain_separation_enforced",
            "concept_aware_retained",
        ):
            if retrieval_exit_gate.get(field_name) is not True:
                raise AssemblyError(
                    f"Phase 14 exit gate failed: {field_name}."
                )

        _validate_manifest_document(
            generation_manifest_document,
            expected_phase="phase_15_build_domain_specific_generation",
            allowed_statuses={"domain_generation_complete"},
            description="Phase 15 manifest",
        )

        same_question(
            expected=question,
            actual=require_string(
                generation_manifest_document.get("question"),
                "Phase 15 manifest question",
            ),
            description="Phase 15 manifest",
        )

        same_corpus_version(
            expected=corpus_version,
            actual=require_string(
                generation_manifest_document.get("corpus_version"),
                "Phase 15 manifest corpus_version",
            ),
            description="Phase 15 manifest",
        )

        if require_string(
            domain_responses_document.get("generation_version"),
            "Phase 15 generation_version",
        ) != require_string(
            generation_manifest_document.get("generation_version"),
            "Phase 15 manifest generation_version",
        ):
            raise AssemblyError(
                "Phase 15 generation versions differ."
            )

        if require_string(
            domain_responses_document.get("prompt_version"),
            "Phase 15 prompt_version",
        ) != require_string(
            generation_manifest_document.get("prompt_version"),
            "Phase 15 manifest prompt_version",
        ):
            raise AssemblyError(
                "Phase 15 prompt versions differ."
            )

        (
            generation_query_activation,
            domains,
        ) = parse_domain_responses(
            domain_responses_document,
            question=question,
            corpus_version=corpus_version,
            evidence_index=evidence_index,
        )

        if json.dumps(
            query_activation,
            sort_keys=True,
            ensure_ascii=False,
        ) != json.dumps(
            generation_query_activation,
            sort_keys=True,
            ensure_ascii=False,
        ):
            raise AssemblyError(
                "Phase 14 and Phase 15 query activation payloads differ."
            )

        claim_index = build_claim_index(domains)

        _validate_manifest_document(
            synthesis_manifest_document,
            expected_phase="phase_16_synthesis_and_tension_detection",
            allowed_statuses={"synthesis_complete"},
            description="Phase 16 manifest",
        )

        same_question(
            expected=question,
            actual=require_string(
                synthesis_manifest_document.get("question"),
                "Phase 16 manifest question",
            ),
            description="Phase 16 manifest",
        )

        same_corpus_version(
            expected=corpus_version,
            actual=require_string(
                synthesis_manifest_document.get("corpus_version"),
                "Phase 16 manifest corpus_version",
            ),
            description="Phase 16 manifest",
        )

        parsed_synthesis = parse_synthesis(
            synthesis_document,
            question=question,
            corpus_version=corpus_version,
            claim_index=claim_index,
            evidence_index=evidence_index,
        )

        synthesis_prompt_version = require_string(
            parsed_synthesis.get("prompt_version"),
            "synthesis prompt_version",
        )

        manifest_synthesis_prompt = require_string(
            synthesis_manifest_document.get("prompt_version"),
            "Phase 16 manifest prompt_version",
        )

        if synthesis_prompt_version != manifest_synthesis_prompt:
            raise AssemblyError(
                "Phase 16 synthesis and manifest prompt versions differ."
            )

        _validate_manifest_document(
            coverage_manifest_document,
            expected_phase="phase_17_coverage_classification",
            allowed_statuses={"coverage_classification_complete"},
            description="Phase 17 manifest",
        )

        same_question(
            expected=question,
            actual=require_string(
                coverage_manifest_document.get("question"),
                "Phase 17 manifest question",
            ),
            description="Phase 17 manifest",
        )

        same_corpus_version(
            expected=corpus_version,
            actual=require_string(
                coverage_manifest_document.get("corpus_version"),
                "Phase 17 manifest corpus_version",
            ),
            description="Phase 17 manifest",
        )

        parsed_coverage = parse_coverage(
            coverage_document,
            question=question,
            corpus_version=corpus_version,
            active_concepts=active_concepts,
        )

        manifest_coverage_version = require_string(
            coverage_manifest_document.get("coverage_version"),
            "Phase 17 manifest coverage_version",
        )

        if (
            parsed_coverage["coverage_version"]
            != manifest_coverage_version
        ):
            raise AssemblyError(
                "Phase 17 coverage and manifest versions differ."
            )

        key_to_ref, citations = citation_registry(
            domains=domains,
            synthesis=parsed_synthesis,
        )

        attach_citation_refs(
            domains=domains,
            synthesis=parsed_synthesis,
            key_to_ref=key_to_ref,
        )

        sections = final_sections(
            question=question,
            active_concepts=active_concepts,
            weights=weights,
            domains=domains,
            synthesis=parsed_synthesis,
            coverage=parsed_coverage,
        )

        generation_prompt_version = require_string(
            generation_manifest_document.get("prompt_version"),
            "Phase 15 prompt_version",
        )

        generation_version = require_string(
            generation_manifest_document.get("generation_version"),
            "Phase 15 generation_version",
        )

        response: dict[str, object] = {
            "assembly_version": ASSEMBLY_VERSION,
            "generated_at": generated_at or utc_now(),
            "question": question,
            "corpus_version": corpus_version,
            "sections": sections,
            "claim_level_citations": citations,
            "versions": {
                "generation_version": generation_version,
                "generation_prompt_version": generation_prompt_version,
                "synthesis_version": require_string(
                    parsed_synthesis.get("synthesis_version"),
                    "synthesis_version",
                ),
                "synthesis_prompt_version": synthesis_prompt_version,
                "coverage_version": require_string(
                    parsed_coverage.get("coverage_version"),
                    "coverage_version",
                ),
                "corpus_version": corpus_version,
            },
            "provider_calls": {
                "phase18_llm_calls": 0,
                "phase18_embedding_calls": 0,
                "phase18_retrieval_calls": 0,
            },
        }

        corpus_answer_allowed = (
            require_mapping(
                parsed_coverage.get("response_policy"),
                "response_policy",
            ).get("corpus_answer_allowed")
            is True
        )

        claim_count = sum(
            len(
                require_list(
                    domains[domain].get("claims"),
                    f"{domain} claims",
                )
            )
            for domain in DOMAINS
        )

        cited_claim_count = sum(
            1
            for domain in DOMAINS
            for claim_raw in require_list(
                domains[domain].get("claims"),
                f"{domain} claims",
            )
            if require_list(
                require_mapping(
                    claim_raw,
                    f"{domain} claim",
                ).get("citation_refs"),
                "claim citation_refs",
            )
        )

        all_claims_cited = (
            claim_count == cited_claim_count
        )

        validation_issues: list[dict[str, str]] = []

        if (
            corpus_answer_allowed
            and not all_claims_cited
        ):
            validation_issues.append(
                {
                    "severity": "error",
                    "code": "uncited_claim",
                    "message": (
                        "At least one corpus claim has no "
                        "claim-level citation."
                    ),
                }
            )

        if (
            not citations
            and corpus_answer_allowed
        ):
            validation_issues.append(
                {
                    "severity": "error",
                    "code": "no_citations",
                    "message": (
                        "Corpus answer is allowed but the "
                        "final citation registry is empty."
                    ),
                }
            )

        final_validation_passed = not any(
            issue["severity"] == "error"
            for issue in validation_issues
        )

        response["validation"] = {
            "passed": final_validation_passed,
            "issue_count": len(validation_issues),
            "issues": validation_issues,
            "checks": {
                "all_phase15_claims_cited": (
                    all_claims_cited
                ),
                (
                    "citations_resolve_to_phase14_"
                    "active_retrieval_evidence"
                ): True,
                "citation_domains_match_claim_domains": True,
                (
                    "phase15_domain_leakage_"
                    "validation_passed"
                ): True,
                (
                    "phase16_synthesis_validation_passed"
                ): True,
                (
                    "unsupported_atman_purusha_"
                    "equivalence_rejected"
                ): True,
                (
                    "coverage_status_consistent_with_"
                    "phase17_concept_statuses"
                ): True,
                "out_of_corpus_blocks_corpus_answer": (
                    parsed_coverage["coverage_status"]
                    != "Out of Corpus"
                    or not corpus_answer_allowed
                ),
                "corpus_and_prompt_versions_recorded": True,
                (
                    "reviewed_corpus_and_"
                    "general_knowledge_separated"
                ): True,
            },
        }

        if (raise_on_validation_failure and not final_validation_passed):
                raise AssemblyError("Phase 18 assembled the response but failed final validation.")

        final_response = FinalResponse.model_validate(response)

        markdown = markdown_response(
            sections=sections,
            citation_registry_rows=citations,
        )

        return ResponseAssemblyResult(
            response=final_response,
            markdown=markdown,
        )


def _validate_manifest_document(
    manifest: dict[str, object],
    *,
    expected_phase: str,
    allowed_statuses: set[str],
    description: str,
) -> None:
    """Validate an already-loaded manifest using Phase 18's frozen rules."""

    phase = require_string(
        manifest.get("phase"),
        f"{description} phase",
    )

    if phase != expected_phase:
        raise AssemblyError(
            f"{description} has unexpected phase {phase!r}."
        )

    status = require_string(
        manifest.get("status"),
        f"{description} status",
    )

    if status not in allowed_statuses:
        raise AssemblyError(
            f"{description} status {status!r} "
            "is not complete."
        )


RuntimeInputContract = (
    EvidencePackage
    | RetrievalManifest
    | DomainResponses
    | GenerationManifest
    | SynthesisResult
    | SynthesisManifest
    | CoverageResult
    | CoverageManifest
)


def _dump_contract(
    contract: RuntimeInputContract,
) -> dict[str, object]:
    """Dump one frozen runtime contract to its artifact-shaped dictionary."""

    raw = contract.model_dump(
        mode="python",
        by_alias=True,
    )

    result: dict[str, object] = {}

    for key, value in raw.items():
        result[key] = value

    return result


__all__ = [
    "ResponseAssemblyResult",
    "ResponseAssemblyService",
]