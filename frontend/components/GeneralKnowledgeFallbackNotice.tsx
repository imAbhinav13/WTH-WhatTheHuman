import type { GeneralKnowledgeFallback } from "@/types/api";

export function GeneralKnowledgeFallbackNotice({ fallback }: { fallback: GeneralKnowledgeFallback }) {
  if (!fallback.allowed) return null;
  return (
    <aside className="border border-accent/25 bg-accent/[0.045] px-5 py-5" aria-label="General knowledge fallback notice">
      <p className="text-xs font-semibold uppercase tracking-[0.16em] text-accent">Not from the reviewed corpus</p>
      <p className="manuscript-text mt-2 text-base">
        General-knowledge fallback is permitted for aspects the reviewed corpus does not cover. No general-knowledge passage is included in this response, and WTH corpus citations must not be attached to fallback material.
      </p>
    </aside>
  );
}
