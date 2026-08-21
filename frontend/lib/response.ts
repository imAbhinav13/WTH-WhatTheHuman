import type {
  ClaimLevelCitation,
  FinalResponse,
  QueryRequest,
} from "@/types/api";

export const QUESTION_MIN_LENGTH = 3;
export const QUESTION_MAX_LENGTH = 1000;

export function createQueryRequest(question: string): QueryRequest {
  const trimmed = question.trim();
  if (trimmed.length < QUESTION_MIN_LENGTH || trimmed.length > QUESTION_MAX_LENGTH) {
    throw new RangeError(
      `Question must be between ${QUESTION_MIN_LENGTH} and ${QUESTION_MAX_LENGTH} characters.`,
    );
  }
  return { question: trimmed };
}

export function resolveCitationRef(
  response: FinalResponse,
  citationRef: string,
): ClaimLevelCitation | undefined {
  return response.claim_level_citations.find(
    (citation) => citation.citation_ref === citationRef,
  );
}

export function isOutOfCorpus(response: FinalResponse): boolean {
  return response.sections.coverage.coverage_status === "Out of Corpus";
}

function isObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function isCitation(value: unknown): boolean {
  if (!isObject(value)) return false;
  return (
    typeof value.citation_ref === "string" &&
    typeof value.chunk_id === "string" &&
    typeof value.source_id === "string" &&
    typeof value.citation === "string" &&
    typeof value.corpus_version === "string" &&
    typeof value.domain === "string"
  );
}

function isDomainCanonicalCitation(value: unknown): boolean {
  if (!isObject(value)) return false;
  return (
    typeof value.source_id === "string" &&
    typeof value.citation === "string" &&
    typeof value.corpus_version === "string" &&
    typeof value.domain === "string"
  );
}

function isClaim(value: unknown): boolean {
  if (!isObject(value)) return false;
  return (
    typeof value.claim_id === "string" &&
    typeof value.claim_ref === "string" &&
    typeof value.text === "string" &&
    isStringArray(value.concepts_covered) &&
    isStringArray(value.citation_refs) &&
    Array.isArray(value.citations) &&
    value.citations.every(isDomainCanonicalCitation)
  );
}

function isDomainPerspective(value: unknown): boolean {
  if (!isObject(value)) return false;
  return (
    typeof value.display_name === "string" &&
    typeof value.domain === "string" &&
    typeof value.summary === "string" &&
    Array.isArray(value.claims) &&
    value.claims.every(isClaim) &&
    isStringArray(value.limitations) &&
    isStringArray(value.unsupported_aspects)
  );
}

function isComparison(value: unknown): boolean {
  if (!isObject(value)) return false;
  return (
    typeof value.category === "string" &&
    typeof value.comparison_id === "string" &&
    typeof value.explanation === "string" &&
    isStringArray(value.citation_refs) &&
    isStringArray(value.claim_refs) &&
    isStringArray(value.concepts_covered) &&
    isStringArray(value.domains) &&
    Array.isArray(value.citations) &&
    Array.isArray(value.limitations) &&
    value.limitations.every(
      (item) =>
        isObject(item) &&
        typeof item.domain === "string" &&
        typeof item.limitation_ref === "string" &&
        typeof item.text === "string",
    )
  );
}

function isCoverage(value: unknown): boolean {
  if (!isObject(value)) return false;
  return (
    typeof value.coverage_reason === "string" &&
    typeof value.coverage_score === "number" &&
    Number.isFinite(value.coverage_score) &&
    typeof value.coverage_status === "string" &&
    isStringArray(value.covered_domains) &&
    isStringArray(value.missing_domains) &&
    isStringArray(value.partially_supported_concepts) &&
    isStringArray(value.supported_concepts) &&
    isStringArray(value.unsupported_concepts) &&
    isStringArray(value.hard_overrides)
  );
}

function isActivatedConcept(value: unknown): boolean {
  if (!isObject(value)) return false;
  return (
    typeof value.activation_weight === "number" &&
    typeof value.concept === "string" &&
    typeof value.coverage_score === "number" &&
    typeof value.coverage_status === "string" &&
    typeof value.display_name === "string"
  );
}

function isComparativeSynthesis(value: unknown): boolean {
  if (!isObject(value)) return false;
  return (
    typeof value.summary === "string" &&
    typeof value.three_way_overview === "string" &&
    typeof value.non_conclusion === "string" &&
    Array.isArray(value.comparisons) &&
    value.comparisons.every(isComparison)
  );
}

function isFallback(value: unknown): boolean {
  if (!isObject(value)) return false;
  return (
    typeof value.allowed === "boolean" &&
    typeof value.generated_in_phase18 === "boolean" &&
    typeof value.instruction === "string" &&
    typeof value.may_use_wth_corpus_citations === "boolean" &&
    typeof value.must_be_clearly_labeled === "boolean"
  );
}

export function isValidFinalResponse(value: unknown): value is FinalResponse {
  if (!isObject(value)) return false;
  if (
    typeof value.assembly_version !== "string" ||
    typeof value.corpus_version !== "string" ||
    typeof value.generated_at !== "string" ||
    typeof value.question !== "string" ||
    !Array.isArray(value.claim_level_citations) ||
    !value.claim_level_citations.every(isCitation) ||
    !isObject(value.provider_calls) ||
    !isObject(value.versions) ||
    !isObject(value.sections) ||
    !isObject(value.validation) ||
    typeof value.validation.passed !== "boolean" ||
    typeof value.validation.issue_count !== "number" ||
    !Array.isArray(value.validation.issues) ||
    !isObject(value.validation.checks)
  ) {
    return false;
  }

  const sections = value.sections;
  if (
    !Array.isArray(sections.activated_concepts) ||
    !sections.activated_concepts.every(isActivatedConcept) ||
    typeof sections.interpretation !== "string" ||
    !isCoverage(sections.coverage) ||
    !isComparativeSynthesis(sections.comparative_synthesis) ||
    !isObject(sections.domain_perspectives) ||
    !Object.values(sections.domain_perspectives).every(isDomainPerspective) ||
    !isFallback(sections.general_knowledge_fallback) ||
    !Array.isArray(sections.key_tensions) ||
    !sections.key_tensions.every(isComparison) ||
    !Array.isArray(sections.non_equivalences) ||
    !sections.non_equivalences.every(isComparison)
  ) {
    return false;
  }

  return true;
}

export function formatCoverageStatus(status: string): string {
  return status;
}

export function humanizeToken(value: string): string {
  return value.replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
}
