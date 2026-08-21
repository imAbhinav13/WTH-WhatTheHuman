import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { SourcePanel } from "@/components/SourcePanel";
import { GeneralKnowledgeFallbackNotice } from "@/components/GeneralKnowledgeFallbackNotice";

describe("defensive optional rendering", () => {
  it("does not crash when a domain perspective is absent", () => {
    render(<SourcePanel domain="science" perspective={undefined} onCitation={() => undefined} />);
    expect(screen.getByText("No reviewed perspective was returned for this domain.")).toBeInTheDocument();
  });

  it("labels general knowledge as outside the reviewed corpus", () => {
    render(
      <GeneralKnowledgeFallbackNotice
        fallback={{
          allowed: true,
          generated_in_phase18: false,
          instruction: "separate",
          may_use_wth_corpus_citations: false,
          must_be_clearly_labeled: true,
        }}
      />,
    );
    expect(screen.getByText("Not from the reviewed corpus")).toBeInTheDocument();
  });
});
