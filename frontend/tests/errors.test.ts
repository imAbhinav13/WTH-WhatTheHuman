import { describe, expect, it } from "vitest";
import { mapApiError } from "@/lib/errors";

describe("API error mapping", () => {
  const cases: Array<[number, string | undefined, string]> = [
    [413, "request_too_large", "This question is too large to send. Please shorten it and try again."],
    [422, "invalid_request", "Please enter a valid question."],
    [429, "api_rate_limited", "WTH is temporarily busy. Please try again shortly."],
    [429, "provider_rate_limited", "WTH is temporarily busy. Please try again shortly."],
    [500, "internal_error", "WTH couldn't complete this answer. Please try again."],
    [502, "upstream_provider_error", "WTH couldn't complete this answer. Please try again."],
    [503, "dependency_unavailable", "The WTH knowledge service is temporarily unavailable."],
    [504, "query_timeout", "This question took too long to process. Please try again."],
  ];

  it.each(cases)("maps HTTP %s / %s", (status: number, code: string | undefined, message: string) => {
    expect(mapApiError(status, code)).toBe(message);
  });
});
