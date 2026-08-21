import { describe, expect, it } from "vitest";
import queryFixture from "@/tests/fixtures/query_success.json";
import {
  createQueryRequest,
  isOutOfCorpus,
  isValidFinalResponse,
  resolveCitationRef,
} from "@/lib/response";
import type { FinalResponse } from "@/types/api";

const response = queryFixture as FinalResponse;

describe("response helpers", () => {
  it("creates a trimmed valid QueryRequest", () => {
    expect(createQueryRequest("   What is the self?   ")).toEqual({ question: "What is the self?" });
  });

  it("rejects invalid QueryRequest lengths", () => {
    expect(() => createQueryRequest(" a ")).toThrow(RangeError);
    expect(() => createQueryRequest("x".repeat(1001))).toThrow(RangeError);
  });

  it("recognizes the captured success response", () => {
    expect(isValidFinalResponse(response)).toBe(true);
    expect(response.validation.passed).toBe(true);
  });

  it("resolves citation_ref through the response-scoped registry", () => {
    const citation = resolveCitationRef(response, "C7");
    expect(citation?.chunk_id).toContain("science_ionta_gassert_blanke_2011");
  });

  it("treats Out of Corpus as a valid response state", () => {
    const outOfCorpus = structuredClone(response);
    outOfCorpus.sections.coverage.coverage_status = "Out of Corpus";
    expect(isValidFinalResponse(outOfCorpus)).toBe(true);
    expect(isOutOfCorpus(outOfCorpus)).toBe(true);
  });

  it("keeps citation mappings response-scoped", () => {
    const anotherResponse = structuredClone(response);
    anotherResponse.claim_level_citations = anotherResponse.claim_level_citations.map((citation) =>
      citation.citation_ref === "C1"
        ? { ...citation, chunk_id: "samkhya:test:chunk" }
        : citation,
    );
    expect(resolveCitationRef(response, "C1")?.chunk_id).not.toBe("samkhya:test:chunk");
    expect(resolveCitationRef(anotherResponse, "C1")?.chunk_id).toBe("samkhya:test:chunk");
  });
});
