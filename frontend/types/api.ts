import type {
  ChunkResponse as GeneratedChunkResponse,
  DomainCanonicalCitation as GeneratedDomainCanonicalCitation,
  FinalActivatedConcept,
  FinalCitation,
  FinalClaim,
  FinalComparativeSynthesis,
  FinalComparison,
  FinalCoverageSection,
  FinalDomainPerspective,
  FinalResponse as GeneratedFinalResponse,
  GeneralKnowledgeFallback as GeneratedGeneralKnowledgeFallback,
  QueryApiError,
  QueryApiErrorResponse,
  QueryApiRequest,
  SynthesisLimitation,
} from "@/types/generated/openapi";

export type Domain = GeneratedChunkResponse["domain"];
export type QueryRequest = QueryApiRequest;
export type FinalResponse = GeneratedFinalResponse;
export type ActivatedConcept = FinalActivatedConcept;
export type DomainPerspective = FinalDomainPerspective;
export type DomainClaim = FinalClaim;
export type DomainCanonicalCitation = GeneratedDomainCanonicalCitation;
export type ClaimLevelCitation = FinalCitation;
export type Comparison = FinalComparison;
export type ComparisonLimitation = SynthesisLimitation;
export type ComparativeSynthesis = FinalComparativeSynthesis;
export type Coverage = FinalCoverageSection;
export type GeneralKnowledgeFallback = GeneratedGeneralKnowledgeFallback;
export type ChunkResponse = GeneratedChunkResponse;
export type ApiError = QueryApiError;
export type ApiErrorResponse = QueryApiErrorResponse;
