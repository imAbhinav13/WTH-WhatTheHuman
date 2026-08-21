"use client";

import { useState } from "react";
import type { Coverage } from "@/types/api";
import { formatCoverageStatus, humanizeToken } from "@/lib/response";

function statusClass(status: string): string {
  if (status === "Partially Supported") return "text-accent border-accent/35";
  if (status === "Out of Corpus") return "ink-muted border-ink/20";
  return "text-ink border-ink/30";
}

function DetailList({ label, values }: { label: string; values: string[] }) {
  return (
    <div>
      <dt className="text-xs font-semibold uppercase tracking-[0.14em] ink-muted">{label}</dt>
      <dd className="manuscript-text mt-1 text-base">
        {values.length ? values.map(humanizeToken).join(", ") : "None"}
      </dd>
    </div>
  );
}

export function CoverageIndicator({ coverage }: { coverage: Coverage }) {
  const [expanded, setExpanded] = useState(false);
  const status = formatCoverageStatus(coverage.coverage_status);

  return (
    <section className="rounded-sm bg-bg-raised/70 px-4 py-4 sm:px-5" aria-label="Corpus coverage">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <span className={`rounded-full border px-3 py-1 text-xs font-semibold ${statusClass(status)}`}>
            {status}
          </span>
          <span className="text-xs tabular-nums ink-muted">{coverage.coverage_score.toFixed(1)}/100 coverage</span>
        </div>
        <button
          type="button"
          onClick={() => setExpanded((value) => !value)}
          aria-expanded={expanded}
          className="rounded-full px-3 py-1 text-xs font-semibold text-accent hover:bg-bg-base/70"
        >
          {expanded ? "Hide detail" : "Show detail"}
        </button>
      </div>
      <p className="manuscript-text mt-3 text-base">{coverage.coverage_reason}</p>
      {expanded ? (
        <dl className="mt-5 grid gap-4 border-t hairline pt-4 sm:grid-cols-2">
          <DetailList label="Supported concepts" values={coverage.supported_concepts} />
          <DetailList label="Partially supported" values={coverage.partially_supported_concepts} />
          <DetailList label="Unsupported concepts" values={coverage.unsupported_concepts} />
          <DetailList label="Covered domains" values={coverage.covered_domains} />
          <DetailList label="Missing domains" values={coverage.missing_domains} />
          <DetailList label="Coverage constraints" values={coverage.hard_overrides} />
        </dl>
      ) : null}
    </section>
  );
}
