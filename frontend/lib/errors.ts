import type { ApiErrorResponse } from "@/types/api";

export type KnownHttpStatus = 413 | 422 | 429 | 500 | 502 | 503 | 504;

const statusMessages: Record<KnownHttpStatus, string> = {
  413: "This question is too large to send. Please shorten it and try again.",
  422: "Please enter a valid question.",
  429: "WTH is temporarily busy. Please try again shortly.",
  500: "WTH couldn't complete this answer. Please try again.",
  502: "WTH couldn't complete this answer. Please try again.",
  503: "The WTH knowledge service is temporarily unavailable.",
  504: "This question took too long to process. Please try again.",
};

export function mapApiError(status: number, _errorCode?: string): string {
  if (status in statusMessages) {
    return statusMessages[status as KnownHttpStatus];
  }
  return "WTH couldn't complete this answer. Please try again.";
}

export class WthApiError extends Error {
  readonly status: number;
  readonly code?: string;
  readonly retryAfterSeconds?: number | null;
  readonly requestId?: string;
  readonly phase?: string | null;

  constructor({
    status,
    payload,
    message,
  }: {
    status: number;
    payload?: Partial<ApiErrorResponse> | null;
    message?: string;
  }) {
    const code = payload?.error?.code;
    super(message ?? mapApiError(status, code));
    this.name = "WthApiError";
    this.status = status;
    this.code = code;
    this.retryAfterSeconds = payload?.error?.retry_after_seconds;
    this.requestId = payload?.request_id;
    this.phase = payload?.error?.phase;
  }
}
