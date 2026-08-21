"use client";

import { motion, useReducedMotion } from "framer-motion";
import type { Domain, DomainPerspective } from "@/types/api";
import { ClaimRow } from "@/components/ClaimRow";

const labels: Record<Domain, string> = {
  science: "Science",
  advaita: "Advaita Vedanta",
  samkhya: "Samkhya",
};

function VisibleList({ title, items }: { title: string; items: string[] }) {
  if (!items.length) return null;
  return (
    <section className="border-t hairline pt-5">
      <h3 className="text-xs font-semibold uppercase tracking-[0.16em] ink-muted">{title}</h3>
      <ul className="manuscript-text mt-3 list-disc space-y-2 pl-5 text-base">
        {items.map((item, index) => (
          <li key={`${item}-${index}`}>{item}</li>
        ))}
      </ul>
    </section>
  );
}

export function SourcePanel({
  domain,
  perspective,
  onCitation,
}: {
  domain: Domain;
  perspective?: DomainPerspective;
  onCitation: (citationRef: string) => void;
}) {
  const reducedMotion = useReducedMotion();
  const title = perspective?.display_name || labels[domain];

  return (
    <motion.section
      aria-labelledby={`${domain}-heading`}
      initial={reducedMotion ? false : { opacity: 0, filter: "blur(8px)", y: 5 }}
      animate={{ opacity: 1, filter: "blur(0px)", y: 0 }}
      transition={{ duration: reducedMotion ? 0 : 1.1, ease: "easeOut" }}
      className="rounded-sm bg-bg-raised px-5 py-6 shadow-[0_1px_0_rgba(74,46,30,0.08)] sm:px-7 sm:py-7"
    >
      <header className="mb-6 border-b hairline pb-5">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] ink-muted">Perspective</p>
        <h2 id={`${domain}-heading`} className="mt-1 font-sans text-xl font-semibold">
          {title}
        </h2>
        {perspective?.summary ? (
          <p className="manuscript-text mt-4 text-[1.08rem]">{perspective.summary}</p>
        ) : (
          <p className="manuscript-text mt-4 text-[1.08rem] ink-muted">
            No reviewed perspective was returned for this domain.
          </p>
        )}
      </header>

      {perspective?.claims?.length ? (
        <div aria-label={`${title} claims`}>
          {perspective.claims.map((claim) => (
            <ClaimRow key={claim.claim_ref} claim={claim} onCitation={onCitation} />
          ))}
        </div>
      ) : null}

      <div className="mt-2 space-y-5">
        <VisibleList title="Limitations" items={perspective?.limitations ?? []} />
        <VisibleList title="Unsupported aspects" items={perspective?.unsupported_aspects ?? []} />
      </div>
    </motion.section>
  );
}
