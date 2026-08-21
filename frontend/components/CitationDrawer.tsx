"use client";

import { useEffect, useRef, useState } from "react";
import { getChunk } from "@/lib/api";
import type { ClaimLevelCitation, ChunkResponse } from "@/types/api";

export type CitationSelection = {
  citationRef: string;
  citation?: ClaimLevelCitation;
};

const focusableSelector =
  'button:not([disabled]), [href], input:not([disabled]), textarea:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])';

export function CitationDrawer({
  selection,
  onClose,
}: {
  selection: CitationSelection | null;
  onClose: () => void;
}) {
  const panelRef = useRef<HTMLElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);
  const [chunk, setChunk] = useState<ChunkResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [retryKey, setRetryKey] = useState(0);

  useEffect(() => {
    if (!selection) return;
    previousFocusRef.current = document.activeElement as HTMLElement | null;
    const frame = requestAnimationFrame(() => closeButtonRef.current?.focus());
    document.body.style.overflow = "hidden";
    return () => {
      cancelAnimationFrame(frame);
      document.body.style.overflow = "";
      previousFocusRef.current?.focus();
    };
  }, [selection]);

  useEffect(() => {
    if (!selection) return;
    setChunk(null);
    setError(null);
    setLoading(false);
    if (!selection.citation?.chunk_id) {
      setError("This evidence reference could not be resolved in the current response.");
      return;
    }
    const controller = new AbortController();
    setLoading(true);
    getChunk(selection.citation.chunk_id, { signal: controller.signal })
      .then((value) => setChunk(value))
      .catch((reason: unknown) => {
        if (reason instanceof DOMException && reason.name === "AbortError") return;
        setError(reason instanceof Error ? reason.message : "Evidence could not be loaded. Please try again.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [selection, retryKey]);

  useEffect(() => {
    if (!selection) return;
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key !== "Tab" || !panelRef.current) return;
      const focusable = Array.from(panelRef.current.querySelectorAll<HTMLElement>(focusableSelector));
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [selection, onClose]);

  if (!selection) return null;

  return (
    <div className="fixed inset-0 z-50" role="presentation">
      <button
        type="button"
        className="absolute inset-0 cursor-default bg-ink/20"
        aria-label="Close evidence drawer"
        onClick={onClose}
      />
      <aside
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="citation-drawer-title"
        className="absolute right-0 top-0 h-full w-full max-w-xl overflow-y-auto bg-bg-raised px-6 py-6 shadow-[-12px_0_40px_rgba(74,46,30,0.12)] sm:px-9"
      >
        <div className="flex items-start justify-between gap-5 border-b hairline pb-5">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.16em] ink-muted">Reviewed evidence</p>
            <h2 id="citation-drawer-title" className="mt-1 font-serif text-3xl font-semibold">
              Evidence {selection.citationRef}
            </h2>
          </div>
          <button
            ref={closeButtonRef}
            type="button"
            onClick={onClose}
            className="min-h-10 rounded-full border border-ink/20 px-3 text-sm font-semibold hover:bg-bg-base"
          >
            Close
          </button>
        </div>

        {loading ? <p className="mt-8 text-sm ink-muted">Loading reviewed passage…</p> : null}

        {error ? (
          <div className="mt-8 border-l-2 border-danger/45 pl-4">
            <p className="manuscript-text text-lg">{error}</p>
            {selection.citation?.chunk_id ? (
              <button
                type="button"
                onClick={() => setRetryKey((value) => value + 1)}
                className="mt-3 rounded-full border border-ink/25 px-3 py-1.5 text-xs font-semibold text-accent"
              >
                Retry evidence
              </button>
            ) : null}
          </div>
        ) : null}

        {chunk ? (
          <div className="mt-8 space-y-8">
            <section>
              <p className="text-xs font-semibold uppercase tracking-[0.15em] ink-muted">Source</p>
              <p className="manuscript-text mt-2 text-lg italic">{chunk.citation}</p>
            </section>
            <section>
              <p className="text-xs font-semibold uppercase tracking-[0.15em] ink-muted">Full reviewed passage</p>
              <p className="manuscript-text mt-3 whitespace-pre-line text-lg">{chunk.text}</p>
            </section>
            <dl className="grid gap-4 border-t hairline pt-5 text-sm sm:grid-cols-2">
              <div>
                <dt className="text-xs uppercase tracking-[0.12em] ink-muted">Domain</dt>
                <dd className="mt-1 capitalize">{chunk.domain}</dd>
              </div>
              <div>
                <dt className="text-xs uppercase tracking-[0.12em] ink-muted">Corpus version</dt>
                <dd className="mt-1 break-all">{chunk.corpus_version}</dd>
              </div>
            </dl>
          </div>
        ) : null}
      </aside>
    </div>
  );
}
