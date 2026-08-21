"use client";

import type { Comparison } from "@/types/api";
import { CitationChip } from "@/components/CitationChip";
import { humanizeToken } from "@/lib/response";

export function ComparisonCard({
  comparison,
  onCitation,
}: {
  comparison: Comparison;
  onCitation: (citationRef: string) => void;
}) {
  return (
    <article className="border-t hairline py-5 first:border-t-0 first:pt-0">
      <div className="flex flex-wrap items-center gap-2">
        <span className="rounded-full border border-ink/20 px-2.5 py-1 text-xs font-semibold ink-muted">
          {humanizeToken(comparison.category)}
        </span>
        {comparison.domains.length ? (
          <span className="text-xs ink-muted">{comparison.domains.map(humanizeToken).join(" ↔ ")}</span>
        ) : null}
      </div>
      <p className="manuscript-text mt-3 text-[1.06rem]">{comparison.explanation}</p>

      {comparison.limitations.length ? (
        <div className="mt-4 border-l border-ink/20 pl-4">
          <p className="text-xs font-semibold uppercase tracking-[0.14em] ink-muted">Comparison limitations</p>
          <ul className="manuscript-text mt-2 space-y-2 text-base ink-muted">
            {comparison.limitations.map((limitation) => (
              <li key={limitation.limitation_ref}>{limitation.text}</li>
            ))}
          </ul>
        </div>
      ) : null}

      {comparison.citation_refs.length ? (
        <div className="mt-4 flex flex-wrap gap-2" aria-label="Comparison citations">
          {comparison.citation_refs.map((citationRef) => (
            <CitationChip key={citationRef} citationRef={citationRef} onOpen={onCitation} />
          ))}
        </div>
      ) : null}
    </article>
  );
}
