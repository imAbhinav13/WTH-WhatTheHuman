import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import queryFixture from "@/tests/fixtures/query_success.json";
import type { FinalResponse } from "@/types/api";

const { queryWthMock } = vi.hoisted(() => ({ queryWthMock: vi.fn() }));

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams("q=Is%20the%20self%20stable%3F"),
  usePathname: () => "/",
}));

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return { ...actual, queryWth: queryWthMock };
});

import { AskExperience } from "@/components/AskExperience";

afterEach(() => {
  queryWthMock.mockReset();
  vi.useRealTimers();
});

describe("AskExperience", () => {
  it("prefills q from the URL and does not auto-submit", () => {
    render(<AskExperience />);
    expect(screen.getByLabelText("Ask one question")).toHaveValue("Is the self stable?");
    expect(queryWthMock).not.toHaveBeenCalled();
  });

  it("does not render validation.passed=false as a trusted answer", async () => {
    const invalid = structuredClone(queryFixture as FinalResponse);
    invalid.validation.passed = false;
    queryWthMock.mockResolvedValue(invalid);

    render(<AskExperience />);
    await userEvent.click(screen.getByRole("button", { name: "Ask WTH" }));
    await waitFor(() => {
      expect(screen.getByText(/couldn't validate this answer safely/i)).toBeInTheDocument();
    });
    expect(screen.queryByText("Comparative synthesis")).not.toBeInTheDocument();
  });
});
