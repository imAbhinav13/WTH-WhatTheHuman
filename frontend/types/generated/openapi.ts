/* eslint-disable */
/**
 * AUTO-GENERATED from contracts/openapi.json.
 * Do not edit by hand. Run `npm run generate:api` after replacing the contract artifact.
 */

export type ChunkResponse = {
  "chunk_id": string;
  "source_id": string;
  "domain": "science" | "advaita" | "samkhya";
  "text": string;
  "citation": string;
  "corpus_version": string;
};

export type DomainCanonicalCitation = {
  "chunk_id": string;
  "source_id": string;
  "citation": string;
  "corpus_version": string;
  "domain": string;
};

export type FinalActivatedConcept = {
  "activation_weight": number;
  "concept": string;
  "coverage_score": number;
  "coverage_status": string;
  "display_name": string;
};

export type FinalCitation = {
  "chunk_id": string;
  "source_id": string;
  "citation": string;
  "corpus_version": string;
  "domain": string;
  "citation_ref": string;
};

export type FinalClaim = {
  "citation_refs": Array<string>;
  "citations": Array<DomainCanonicalCitation>;
  "claim_id": string;
  "claim_ref": string;
  "concepts_covered": Array<string>;
  "text": string;
};

export type FinalComparativeSynthesis = {
  "comparisons": Array<FinalComparison>;
  "non_conclusion": string;
  "summary": string;
  "three_way_overview": string;
};

export type FinalComparison = {
  "category": string;
  "citation_refs": Array<string>;
  "citations": Array<DomainCanonicalCitation>;
  "claim_refs": Array<string>;
  "comparison_id": string;
  "concepts_covered": Array<string>;
  "domains": Array<string>;
  "explanation": string;
  "limitations": Array<SynthesisLimitation>;
};

export type FinalCoverageSection = {
  "coverage_reason": string;
  "coverage_score": number;
  "coverage_status": string;
  "covered_domains": Array<string>;
  "hard_overrides": Array<string>;
  "missing_domains": Array<string>;
  "partially_supported_concepts": Array<string>;
  "supported_concepts": Array<string>;
  "unsupported_concepts": Array<string>;
};

export type FinalDomainPerspective = {
  "claims": Array<FinalClaim>;
  "display_name": string;
  "domain": string;
  "limitations": Array<string>;
  "summary": string;
  "unsupported_aspects": Array<string>;
};

export type FinalResponse = {
  "assembly_version": string;
  "claim_level_citations": Array<FinalCitation>;
  "corpus_version": string;
  "generated_at": string;
  "provider_calls": Phase18ProviderCalls;
  "question": string;
  "sections": FinalSections;
  "validation": FinalValidation;
  "versions": PipelineVersions;
};

export type FinalSections = {
  "activated_concepts": Array<FinalActivatedConcept>;
  "comparative_synthesis": FinalComparativeSynthesis;
  "coverage": FinalCoverageSection;
  "domain_perspectives": Record<string, FinalDomainPerspective>;
  "general_knowledge_fallback": GeneralKnowledgeFallback;
  "interpretation": string;
  "key_tensions": Array<FinalComparison>;
  "non_equivalences": Array<FinalComparison>;
};

export type FinalValidation = {
  "checks": FinalValidationChecks;
  "issue_count": number;
  "issues": Array<Record<string, string>>;
  "passed": boolean;
};

export type FinalValidationChecks = {
  "all_phase15_claims_cited": boolean;
  "citation_domains_match_claim_domains": boolean;
  "citations_resolve_to_phase14_active_retrieval_evidence": boolean;
  "corpus_and_prompt_versions_recorded": boolean;
  "coverage_status_consistent_with_phase17_concept_statuses": boolean;
  "out_of_corpus_blocks_corpus_answer": boolean;
  "phase15_domain_leakage_validation_passed": boolean;
  "phase16_synthesis_validation_passed": boolean;
  "reviewed_corpus_and_general_knowledge_separated": boolean;
  "unsupported_atman_purusha_equivalence_rejected": boolean;
};

export type GeneralKnowledgeFallback = {
  "allowed": boolean;
  "generated_in_phase18": boolean;
  "instruction": string;
  "may_use_wth_corpus_citations": boolean;
  "must_be_clearly_labeled": boolean;
};

export type HTTPValidationError = {
  "detail"?: Array<ValidationError>;
};

export type Phase18ProviderCalls = {
  "phase18_embedding_calls": number;
  "phase18_llm_calls": number;
  "phase18_retrieval_calls": number;
};

export type PipelineVersions = {
  "corpus_version": string;
  "coverage_version": string;
  "generation_prompt_version": string;
  "generation_version": string;
  "synthesis_prompt_version": string;
  "synthesis_version": string;
};

export type QueryApiError = {
  "code": QueryApiErrorCode;
  "message": string;
  "retryable": boolean;
  "phase"?: QueryApiPhase | null;
  "retry_after_seconds"?: number | null;
};

export type QueryApiErrorCode = "invalid_request" | "request_too_large" | "api_rate_limited" | "provider_rate_limited" | "upstream_provider_error" | "dependency_unavailable" | "query_timeout" | "pipeline_invariant_failed" | "internal_error";

export type QueryApiErrorResponse = {
  "request_id": string;
  "error": QueryApiError;
};

export type QueryApiPhase = "retrieval" | "domain_generation" | "synthesis" | "coverage" | "response_assembly";

export type QueryApiRequest = {
  "question": string;
};

export type SynthesisLimitation = {
  "domain": string;
  "limitation_ref": string;
  "text": string;
};

export type ValidationError = {
  "loc": Array<string | number>;
  "msg": string;
  "type": string;
  "input"?: unknown;
  "ctx"?: {

  };
};
