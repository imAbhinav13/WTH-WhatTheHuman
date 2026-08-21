import type { ApiErrorResponse, ChunkResponse, FinalResponse } from "@/types/api";
import { WthApiError } from "@/lib/errors";
import { createQueryRequest, isValidFinalResponse } from "@/lib/response";

function apiBaseUrl(): string {
  const raw = process.env.NEXT_PUBLIC_WTH_API_BASE_URL?.trim();
  if (!raw) {
    return "http://127.0.0.1:8000";
  }
  return raw.replace(/\/$/, "");
}

async function parseJson(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    throw new WthApiError({
      status: response.status || 500,
      message: "WTH returned a response that could not be read. Please try again.",
    });
  }
}

function asErrorPayload(value: unknown): Partial<ApiErrorResponse> | null {
  if (!value || typeof value !== "object") return null;
  return value as Partial<ApiErrorResponse>;
}

export async function queryWth(
  question: string,
  options: { signal?: AbortSignal } = {},
): Promise<FinalResponse> {
  const request = createQueryRequest(question);
  let response: Response;

  try {
    response = await fetch(`${apiBaseUrl()}/api/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
      signal: options.signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new WthApiError({
      status: 0,
      message: "WTH couldn't reach the knowledge service. Please try again.",
    });
  }

  const payload = await parseJson(response);
  if (!response.ok) {
    throw new WthApiError({ status: response.status, payload: asErrorPayload(payload) });
  }
  if (!isValidFinalResponse(payload)) {
    throw new WthApiError({
      status: 500,
      message: "WTH returned an unexpected response. Please try again.",
    });
  }
  return payload;
}

export async function getChunk(
  chunkId: string,
  options: { signal?: AbortSignal } = {},
): Promise<ChunkResponse> {
  let response: Response;
  try {
    response = await fetch(`${apiBaseUrl()}/api/chunk/${encodeURIComponent(chunkId)}`, {
      method: "GET",
      signal: options.signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new Error("Evidence could not be loaded. Please try again.");
  }

  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    throw new Error("Evidence returned a response that could not be read.");
  }
  if (!response.ok) {
    throw new Error("Evidence could not be loaded. Please try again.");
  }
  if (!isChunkResponse(payload)) {
    throw new Error("Evidence returned an unexpected response.");
  }
  return payload;
}

export function isChunkResponse(value: unknown): value is ChunkResponse {
  if (!value || typeof value !== "object") return false;
  const chunk = value as Partial<ChunkResponse>;
  return (
    typeof chunk.chunk_id === "string" &&
    typeof chunk.source_id === "string" &&
    (chunk.domain === "science" || chunk.domain === "advaita" || chunk.domain === "samkhya") &&
    typeof chunk.text === "string" &&
    typeof chunk.citation === "string" &&
    typeof chunk.corpus_version === "string"
  );
}
