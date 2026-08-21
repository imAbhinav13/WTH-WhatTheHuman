import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import chunkFixture from "@/tests/fixtures/chunk_success.json";
import { CitationDrawer } from "@/components/CitationDrawer";
import type { ClaimLevelCitation } from "@/types/api";

const citation: ClaimLevelCitation = {
  citation_ref: "C1",
  chunk_id: chunkFixture.chunk_id,
  source_id: chunkFixture.source_id,
  citation: chunkFixture.citation,
  corpus_version: chunkFixture.corpus_version,
  domain: chunkFixture.domain,
};

afterEach(() => vi.restoreAllMocks());

describe("CitationDrawer", () => {
  it("loads and displays the actual chunk passage without requiring metadata", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify(chunkFixture), { status: 200, headers: { "Content-Type": "application/json" } }),
    );
    render(<CitationDrawer selection={{ citationRef: "C1", citation }} onClose={() => undefined} />);
    expect(screen.getByRole("heading", { name: "Evidence C1" })).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText((content: string) => content.includes("That by which a mortal perceives"))).toBeInTheDocument());
    expect(screen.getByText(chunkFixture.corpus_version)).toBeInTheDocument();
  });

  it("supports Escape close", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify(chunkFixture), { status: 200, headers: { "Content-Type": "application/json" } }),
    );
    const onClose = vi.fn();
    render(<CitationDrawer selection={{ citationRef: "C1", citation }} onClose={onClose} />);
    await userEvent.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
