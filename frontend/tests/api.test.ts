import { afterEach, describe, expect, it, vi } from "vitest";
import queryFixture from "@/tests/fixtures/query_success.json";
import chunkFixture from "@/tests/fixtures/chunk_success.json";
import { getChunk, queryWth } from "@/lib/api";
import { WthApiError } from "@/lib/errors";

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("API client", () => {
  it("parses captured query success", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(queryFixture));
    const result = await queryWth("How is consciousness related to the self and experienced reality?");
    expect(result.question).toBe(queryFixture.question);
    expect(result.sections.domain_perspectives.science.display_name).toBe("Science");
  });

  it("requests a URL-encoded chunk id and parses the captured chunk", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(chunkFixture));
    const id = "source name:chunk:id/with slash";
    const result = await getChunk(id);
    expect(result.chunk_id).toBe(chunkFixture.chunk_id);
    expect(String(fetchSpy.mock.calls[0][0])).toContain(encodeURIComponent(id));
  });

  it("preserves structured error code and retry_after_seconds internally", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(
        {
          request_id: "request-12345678",
          error: {
            code: "provider_rate_limited",
            message: "provider detail not for direct display",
            retryable: true,
            retry_after_seconds: 12,
          },
        },
        429,
      ),
    );

    await expect(queryWth("What is the self?")).rejects.toMatchObject({
      name: "WthApiError",
      status: 429,
      code: "provider_rate_limited",
      retryAfterSeconds: 12,
    } satisfies Partial<WthApiError>);
  });
});
