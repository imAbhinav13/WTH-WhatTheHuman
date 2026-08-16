"""Frozen runtime contracts for the validated WTH Phase 14-18 pipeline.

These models intentionally mirror the existing persisted Phase 14-18 JSON
artifacts. Stage 3.0 is a contract freeze, not a schema redesign. Field names,
container shapes, string timestamps, version fields, validation fields, and
manifest metadata are therefore preserved as-is.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FrozenRuntimeContract(BaseModel):
    """Base class that rejects unreviewed artifact schema drift."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=False,
        validate_assignment=True,
        validate_by_alias=True,
        validate_by_name=True,
        serialize_by_alias=True,
    )


# ---------------------------------------------------------------------------
# Shared Phase 14-18 structures
# ---------------------------------------------------------------------------


class QueryActivationContract(FrozenRuntimeContract):
    raw_scores: dict[str, float]
    calibrated_weights: dict[str, float]
    active_concepts: list[str]
    ambiguous: bool
    unsupported: bool


class CanonicalCitation(FrozenRuntimeContract):
    chunk_id: str
    source_id: str
    citation: str
    corpus_version: str


class DomainCanonicalCitation(CanonicalCitation):
    domain: str


class ValidationSummary(FrozenRuntimeContract):
    passed: bool
    issue_count: int
    issues: list[dict[str, str]]


class DomainLeakageValidation(FrozenRuntimeContract):
    passed: bool
    issues: list[dict[str, str]]


# ---------------------------------------------------------------------------
# Phase 14 - EvidencePackage / RetrievalManifest
# ---------------------------------------------------------------------------


class RetrievalConfigContract(FrozenRuntimeContract):
    candidate_pool_per_domain: int
    max_chunks_per_source: int
    min_vector_similarity: float
    token_budget_per_domain: int
    top_k_per_domain: int


class RetrievalScoringContract(FrozenRuntimeContract):
    citation_quality_weight: float
    concept_alignment_weight: float
    human_relevance_weight: float
    source_repeat_penalty: float
    vector_similarity_weight: float


class EvidenceConceptRelation(FrozenRuntimeContract):
    calibrated_weight: float
    human_label: str
    human_override: bool
    production_active: bool


class EvidenceScores(FrozenRuntimeContract):
    base_score: float
    citation_quality: float
    concept_alignment: float
    diversity_adjusted_score: float
    human_relevance: float
    vector_similarity: float


class RetrievedEvidence(FrozenRuntimeContract):
    chunk_id: str
    citation: str
    concepts: dict[str, EvidenceConceptRelation]
    corpus_version: str
    domain: str
    estimated_tokens: int
    rank: int
    reviewed_text: str
    scores: EvidenceScores
    source_id: str


class DomainEvidencePackage(FrozenRuntimeContract):
    estimated_tokens: int
    evidence: list[RetrievedEvidence]
    evidence_count: int
    status: str
    unique_source_count: int


class EvidencePackage(FrozenRuntimeContract):
    config: RetrievalConfigContract
    corpus_version: str
    domains: dict[str, DomainEvidencePackage]
    generated_at: str
    model_version: str
    prototype_version: str
    query_activation: QueryActivationContract
    question: str
    retrieval_mode: str
    retrieval_version: str
    scoring: RetrievalScoringContract


class RetrievalExitGate(FrozenRuntimeContract):
    concept_aware_retained: bool | None
    deduplication_enforced: bool
    domain_separation_enforced: bool
    only_active_chunks_retrieved: bool
    per_domain_context_budgets_enforced: bool
    question_embedding_uses_frozen_model: bool
    retrieval_evaluation_complete: bool
    source_diversity_enforced: bool
    weighted_concept_activation_uses_frozen_phase10: bool


class RetrievalOutputs(FrozenRuntimeContract):
    evidence_package: str | None
    retrieval_config: str
    retrieval_evaluation_results: str | None
    retrieval_report: str | None


class RetrievalManifest(FrozenRuntimeContract):
    active_chunk_count: int
    corpus_version: str
    exit_gate: RetrievalExitGate
    generated_at: str
    next_step: str
    outputs: RetrievalOutputs
    phase: str
    retrieval_config: RetrievalConfigContract
    retrieval_version: str
    script_version: str
    status: str


# ---------------------------------------------------------------------------
# Phase 15 - DomainResponses / GenerationManifest
# ---------------------------------------------------------------------------


class GeneratedClaim(FrozenRuntimeContract):
    citations: list[CanonicalCitation]
    claim_id: str
    concepts_covered: list[str]
    supporting_chunk_ids: list[str]
    text: str


class DomainGrounding(FrozenRuntimeContract):
    all_citations_canonicalized_from_evidence: bool
    all_claims_preserve_corpus_version: bool
    all_claims_use_retrieved_chunks: bool
    evidence_chunk_count: int
    retrieval_status: str


class ProviderUsage(FrozenRuntimeContract):
    completion_time: float
    completion_tokens: int
    prompt_time: float
    prompt_tokens: int
    queue_time: float
    total_time: float
    total_tokens: int


class CalledDomainGenerationProvider(FrozenRuntimeContract):
    attempt: int
    json_object_mode: bool
    latency_ms: float
    max_completion_tokens: int
    model_requested: str
    model_returned: str
    provider: str
    reasoning_effort: str | None
    structured_output_strict: bool
    system_fingerprint: str | None
    temperature: float
    usage: ProviderUsage


class SkippedDomainGenerationProvider(FrozenRuntimeContract):
    provider: str
    model_requested: str
    call_skipped: bool
    reason: str


class DomainResponse(FrozenRuntimeContract):
    citations: list[CanonicalCitation]
    claims: list[GeneratedClaim]
    concepts_covered: list[str]
    corpus_version: str
    domain: str
    domain_display_name: str
    domain_leakage: DomainLeakageValidation
    generation_version: str
    grounding: DomainGrounding
    limitations: list[str]
    prompt_version: str
    provider: CalledDomainGenerationProvider | SkippedDomainGenerationProvider
    summary: str
    unsupported_aspects: list[str]
    validation: ValidationSummary


class DomainResponses(FrozenRuntimeContract):
    corpus_version: str
    domains: dict[str, DomainResponse]
    generated_at: str
    generation_version: str
    prompt_version: str
    query_activation: QueryActivationContract
    question: str


class GenerationCounts(FrozenRuntimeContract):
    canonical_citation_count: int
    domain_count: int
    total_claim_count: int


class GenerationExitGate(FrozenRuntimeContract):
    active_corpus_version_preserved: bool
    advaita_uses_only_advaita_evidence: bool
    atman_purusha_merge_rejected_or_flagged: bool
    citations_resolve_from_retrieved_evidence: bool
    domain_leakage_validation_passed: bool
    every_substantive_claim_maps_to_retrieved_chunks: bool
    independently_grounded_and_claim_cited: bool
    samkhya_uses_only_samkhya_evidence: bool
    science_metaphysical_proof_rejected_or_flagged: bool
    science_uses_only_science_evidence: bool


class GenerationOutputs(FrozenRuntimeContract):
    advaita: str
    combined: str
    samkhya: str
    science: str


class GenerationManifestProvider(FrozenRuntimeContract):
    json_object_mode: bool
    maximum_domain_calls: int
    model: str
    parallel_domain_calls: bool
    provider: str
    reasoning_effort: str | None
    structured_output_strict: bool
    temperature: float


class GenerationTiming(FrozenRuntimeContract):
    parallel_generation_elapsed_ms: float


class GenerationManifest(FrozenRuntimeContract):
    corpus_version: str
    counts: GenerationCounts
    domain_validation: dict[str, bool]
    exit_gate: GenerationExitGate
    generated_at: str
    generation_version: str
    next_step: str
    outputs: GenerationOutputs
    phase: str
    prompt_version: str
    provider: GenerationManifestProvider
    question: str
    script_version: str
    status: str
    timing: GenerationTiming


# ---------------------------------------------------------------------------
# Phase 16 - SynthesisResult / SynthesisManifest
# ---------------------------------------------------------------------------


class SynthesisLimitation(FrozenRuntimeContract):
    domain: str
    limitation_ref: str
    text: str


class SynthesisComparison(FrozenRuntimeContract):
    category: str
    citations: list[CanonicalCitation]
    claim_refs: list[str]
    comparison_id: str
    concepts_covered: list[str]
    domains: list[str]
    explanation: str
    limitation_refs: list[str]
    limitations: list[SynthesisLimitation]


class SynthesisProviderMetadata(FrozenRuntimeContract):
    attempt: int
    finish_reason: str
    json_object_mode: bool
    latency_ms: float
    max_completion_tokens: int
    model_requested: str
    model_returned: str
    provider: str
    slot_count: int
    structured_output_strict: bool
    system_fingerprint: str | None
    temperature: float
    usage: ProviderUsage


class ThreeWayOverview(FrozenRuntimeContract):
    active_concepts: list[str]
    direct_tension_comparison_ids: list[str]
    direct_tension_count: int
    methodological_note: str
    non_equivalence_comparison_ids: list[str]
    non_equivalence_count: int
    total_pairwise_comparisons: int


class SynthesisResult(FrozenRuntimeContract):
    comparisons: list[SynthesisComparison]
    corpus_version: str
    domain_limitations: list[SynthesisLimitation]
    insufficient_corpus_coverage: list[SynthesisComparison]
    key_tensions: list[SynthesisComparison]
    non_conclusion: str
    non_equivalences: list[SynthesisComparison]
    pairwise_comparisons: list[SynthesisComparison]
    prompt_version: str
    provider: SynthesisProviderMetadata
    query_activation: QueryActivationContract
    question: str
    synthesis_summary: str
    synthesis_version: str
    three_way_overview: ThreeWayOverview
    validation: ValidationSummary


class SynthesisCounts(FrozenRuntimeContract):
    active_concept_count: int
    comparison_count: int
    direct_tension_count: int
    insufficient_coverage_count: int
    non_equivalence_count: int
    required_slot_count: int


class SynthesisExitGate(FrozenRuntimeContract):
    all_comparison_references_validated: bool
    atman_purusha_false_equivalence_rejected: bool
    comparison_citations_canonicalized_from_phase15: bool
    comparison_identity_owned_by_python: bool
    domain_differences_preserved: bool
    known_tension_guardrails_enforced: bool
    limitation_identity_owned_by_python: bool
    pairwise_active_concept_matrix_complete: bool
    phase15_grounded_input_only: bool
    raw_corpus_not_resent: bool
    relevant_unsupported_aspects_propagated: bool
    science_metaphysical_proof_rejected: bool
    summary_and_non_conclusion_derived_deterministically: bool
    synthesis_explanations_entailment_guarded: bool
    synthesis_preserves_domain_differences_and_identifies_unsupported_comparisons: bool
    three_way_overview_derived_from_pairwise_relations: bool
    unsupported_comparisons_identified_or_left_unasserted: bool


class SynthesisInputPolicy(FrozenRuntimeContract):
    claim_refs_attached_locally: bool
    claim_texts_deduplicated_in_prompt: bool
    comparison_slots_generated_locally: bool
    limitation_refs_attached_locally: bool
    one_user_question_per_synthesis_request: bool
    outside_knowledge_allowed: bool
    pairwise_comparison_required: bool
    phase15_citations_used: bool
    phase15_limitations_used: bool
    phase15_structured_claims_used: bool
    raw_corpus_sent: bool
    raw_retrieval_chunks_sent: bool
    relevant_unsupported_aspects_propagated: bool


class SynthesisOutputs(FrozenRuntimeContract):
    synthesis: str


class SynthesisManifestProvider(FrozenRuntimeContract):
    json_object_mode: bool
    maximum_api_calls: int
    model: str
    provider: str
    structured_output_strict: bool
    temperature: float


class SynthesisTiming(FrozenRuntimeContract):
    synthesis_generation_elapsed_ms: float


class SynthesisManifest(FrozenRuntimeContract):
    corpus_version: str
    counts: SynthesisCounts
    exit_gate: SynthesisExitGate
    generated_at: str
    input_policy: SynthesisInputPolicy
    next_step: str
    outputs: SynthesisOutputs
    phase: str
    prompt_version: str
    provider: SynthesisManifestProvider
    question: str
    script_version: str
    status: str
    synthesis_version: str
    timing: SynthesisTiming


# ---------------------------------------------------------------------------
# Phase 17 - CoverageResult / CoverageManifest
# ---------------------------------------------------------------------------


class KnowledgeBoundary(FrozenRuntimeContract):
    corpus_claims_require_reviewed_evidence: bool
    general_knowledge_may_not_be_presented_as_corpus_supported: bool
    general_knowledge_may_not_reuse_wth_corpus_citations: bool
    reviewed_corpus_and_general_knowledge_are_separate: bool


class CoverageScoreComponents(FrozenRuntimeContract):
    activated_concept_weight: float
    citation_quality: float
    domain_coverage: float
    retrieval_confidence: float
    retrieved_evidence: float
    unsupported_subquestion_component: float


class ConceptCoverageResult(FrozenRuntimeContract):
    activation_weight: float
    citation_quality: float
    claim_domains: list[str]
    concept: str
    coverage_score: float
    covered_domains: list[str]
    evidence_count: int
    explicit_unsupported: bool
    hard_overrides: list[str]
    insufficient_comparison: bool
    reasons: list[str]
    retrieval_confidence: float
    retrieval_confidence_source: str
    score_components: CoverageScoreComponents
    status: str


class CoverageExitGate(FrozenRuntimeContract):
    corpus_citations_forbidden_for_general_fallback: bool
    general_knowledge_boundary_explicit: bool
    no_corpus_fabrication_when_out_of_corpus: bool
    partial_answers_limited_to_supported_components: bool
    passed: bool


class CorpusCoverageResponsePolicy(FrozenRuntimeContract):
    corpus_answer_allowed: bool
    corpus_answer_scope: str
    general_knowledge_fallback_allowed: bool
    general_knowledge_must_be_labeled: bool
    general_knowledge_must_not_use_corpus_citations: bool
    suggested_disclosure: str


class OutOfCorpusCoverageResponsePolicy(
    CorpusCoverageResponsePolicy
):
    fallback_structure: list[str]


CoverageResponsePolicy = (
    CorpusCoverageResponsePolicy
    | OutOfCorpusCoverageResponsePolicy
)


class CoverageThresholds(FrozenRuntimeContract):
    partial_min: float
    supported_min: float


class CoverageScoreWeights(FrozenRuntimeContract):
    activated_concept_weight: float
    citation_quality: float
    domain_coverage: float
    retrieval_confidence: float
    retrieved_evidence: float
    unsupported_subquestion_component: float


class CoverageSignals(FrozenRuntimeContract):
    activated_concept_weights: dict[str, float]
    active_concepts: list[str]
    claim_citation_count: int
    coverage_thresholds: CoverageThresholds
    grounded_claim_count: int
    insufficient_comparison_concepts: list[str]
    query_ambiguous: bool
    query_unsupported: bool
    retrieved_evidence_count: int
    score_weights: CoverageScoreWeights


class CoverageUpstreamVersions(FrozenRuntimeContract):
    generation_version: str
    retrieval_manifest_status: str
    synthesis_manifest_status: str
    synthesis_version: str


class CoverageResult(FrozenRuntimeContract):
    boundary: KnowledgeBoundary
    concept_coverage: list[ConceptCoverageResult]
    corpus_version: str
    coverage_reason: str
    coverage_score: float
    coverage_status: str
    coverage_version: str
    covered_domains: list[str]
    exit_gate: CoverageExitGate
    generated_at: str
    hard_overrides: list[str]
    missing_domains: list[str]
    partially_supported_concepts: list[str]
    question: str
    response_policy: CoverageResponsePolicy
    signals: CoverageSignals
    supported_concepts: list[str]
    unsupported_concepts: list[str]
    upstream_versions: CoverageUpstreamVersions


class CoverageCalculationPolicy(FrozenRuntimeContract):
    activated_concept_weights_used: bool
    citation_quality_used: bool
    deterministic: bool
    domain_coverage_used: bool
    llm_calls: int
    partial_score_threshold: float
    retrieval_confidence_used: bool
    retrieved_evidence_count_used: bool
    score_then_hard_override_policy: bool
    single_domain_cannot_be_fully_supported: bool
    strict_supported_requires_all_active_concepts_supported: bool
    supported_score_threshold: float
    unsupported_subquestions_used: bool


class CoverageClassification(FrozenRuntimeContract):
    coverage_score: float
    coverage_status: str
    covered_domain_count: int
    missing_domain_count: int
    partially_supported_concept_count: int
    supported_concept_count: int
    unsupported_concept_count: int


class CoverageKnowledgeBoundary(FrozenRuntimeContract):
    general_fallback_allowed_when_labeled: bool
    general_fallback_cannot_use_wth_corpus_citations: bool
    out_of_corpus_corpus_answer_forbidden: bool
    reviewed_corpus_support_separate_from_general_knowledge: bool


class CoverageOutputs(FrozenRuntimeContract):
    coverage: str


class CoverageManifest(FrozenRuntimeContract):
    calculation_policy: CoverageCalculationPolicy
    classification: CoverageClassification
    corpus_version: str
    coverage_version: str
    exit_gate: CoverageExitGate
    generated_at: str
    knowledge_boundary: CoverageKnowledgeBoundary
    next_step: str
    outputs: CoverageOutputs
    phase: str
    question: str
    script_version: str
    status: str


# ---------------------------------------------------------------------------
# Phase 18 - FinalResponse / FinalResponseManifest
# ---------------------------------------------------------------------------


class FinalCitation(DomainCanonicalCitation):
    citation_ref: str


class FinalActivatedConcept(FrozenRuntimeContract):
    activation_weight: float
    concept: str
    coverage_score: float
    coverage_status: str
    display_name: str


class FinalClaim(FrozenRuntimeContract):
    citation_refs: list[str]
    citations: list[DomainCanonicalCitation]
    claim_id: str
    claim_ref: str
    concepts_covered: list[str]
    text: str


class FinalDomainPerspective(FrozenRuntimeContract):
    claims: list[FinalClaim]
    display_name: str
    domain: str
    limitations: list[str]
    summary: str
    unsupported_aspects: list[str]


class FinalComparison(FrozenRuntimeContract):
    category: str
    citation_refs: list[str]
    citations: list[DomainCanonicalCitation]
    claim_refs: list[str]
    comparison_id: str
    concepts_covered: list[str]
    domains: list[str]
    explanation: str
    limitations: list[SynthesisLimitation]


class FinalComparativeSynthesis(FrozenRuntimeContract):
    comparisons: list[FinalComparison]
    non_conclusion: str
    summary: str
    three_way_overview: str


class FinalCoverageSection(FrozenRuntimeContract):
    coverage_reason: str
    coverage_score: float
    coverage_status: str
    covered_domains: list[str]
    hard_overrides: list[str]
    missing_domains: list[str]
    partially_supported_concepts: list[str]
    supported_concepts: list[str]
    unsupported_concepts: list[str]


class GeneralKnowledgeFallback(FrozenRuntimeContract):
    allowed: bool
    generated_in_phase18: bool
    instruction: str
    may_use_wth_corpus_citations: bool
    must_be_clearly_labeled: bool


class FinalSections(FrozenRuntimeContract):
    activated_concepts: list[FinalActivatedConcept]
    comparative_synthesis: FinalComparativeSynthesis
    coverage: FinalCoverageSection
    domain_perspectives: dict[str, FinalDomainPerspective]
    general_knowledge_fallback: GeneralKnowledgeFallback
    interpretation: str
    key_tensions: list[FinalComparison]
    non_equivalences: list[FinalComparison]


class FinalValidationChecks(FrozenRuntimeContract):
    all_phase15_claims_cited: bool
    citation_domains_match_claim_domains: bool
    citations_resolve_to_phase14_active_retrieval_evidence: bool
    corpus_and_prompt_versions_recorded: bool
    coverage_status_consistent_with_phase17_concept_statuses: bool
    out_of_corpus_blocks_corpus_answer: bool
    phase15_domain_leakage_validation_passed: bool
    phase16_synthesis_validation_passed: bool
    reviewed_corpus_and_general_knowledge_separated: bool
    unsupported_atman_purusha_equivalence_rejected: bool


class FinalValidation(FrozenRuntimeContract):
    checks: FinalValidationChecks
    issue_count: int
    issues: list[dict[str, str]]
    passed: bool


class PipelineVersions(FrozenRuntimeContract):
    corpus_version: str
    coverage_version: str
    generation_prompt_version: str
    generation_version: str
    synthesis_prompt_version: str
    synthesis_version: str


class Phase18ProviderCalls(FrozenRuntimeContract):
    phase18_embedding_calls: int
    phase18_llm_calls: int
    phase18_retrieval_calls: int


class FinalResponse(FrozenRuntimeContract):
    assembly_version: str
    claim_level_citations: list[FinalCitation]
    corpus_version: str
    generated_at: str
    provider_calls: Phase18ProviderCalls
    question: str
    sections: FinalSections
    validation: FinalValidation
    versions: PipelineVersions


class FinalResponseCounts(FrozenRuntimeContract):
    active_concept_count: int
    citation_count: int
    claim_count: int
    comparison_count: int
    domain_count: int


class FinalExecutionPolicy(FrozenRuntimeContract):
    deterministic_assembly: bool
    general_knowledge_not_generated_in_phase18: bool
    phase15_prose_reused: bool
    phase16_synthesis_reused: bool
    phase17_coverage_policy_enforced: bool
    phase18_embedding_calls: int
    phase18_llm_calls: int
    phase18_retrieval_calls: int


class FinalExitGate(FrozenRuntimeContract):
    all_claims_are_cited: bool
    citations_resolve_to_active_chunks: bool
    corpus_and_prompt_versions_recorded: bool
    coverage_status_matches_actual_phase17_evidence_classification: bool
    no_domain_leakage: bool
    no_unsupported_equivalence: bool
    passed: bool


class FinalOutputs(FrozenRuntimeContract):
    # Alias preserves the existing artifact key "json" while avoiding a
    # BaseModel method-name collision warning.
    json_output: str = Field(alias="json")
    markdown: str


class FinalResponseManifest(FrozenRuntimeContract):
    assembly_version: str
    corpus_version: str
    counts: FinalResponseCounts
    coverage_score: float
    coverage_status: str
    execution_policy: FinalExecutionPolicy
    exit_gate: FinalExitGate
    generated_at: str
    next_step: str
    outputs: FinalOutputs
    phase: str
    question: str
    script_version: str
    status: str
    versions: PipelineVersions


RUNTIME_ARTIFACT_MODELS: dict[
    str,
    type[FrozenRuntimeContract],
] = {
    "evidence_package": EvidencePackage,
    "retrieval_manifest": RetrievalManifest,
    "domain_responses": DomainResponses,
    "generation_manifest": GenerationManifest,
    "synthesis": SynthesisResult,
    "synthesis_manifest": SynthesisManifest,
    "coverage": CoverageResult,
    "coverage_manifest": CoverageManifest,
    "final_response": FinalResponse,
    "final_response_manifest": FinalResponseManifest,
}


__all__ = [
    "RUNTIME_ARTIFACT_MODELS",
    "CoverageManifest",
    "CoverageResult",
    "DomainResponses",
    "EvidencePackage",
    "FinalResponse",
    "FinalResponseManifest",
    "FrozenRuntimeContract",
    "GenerationManifest",
    "RetrievalManifest",
    "SynthesisManifest",
    "SynthesisResult",
]
