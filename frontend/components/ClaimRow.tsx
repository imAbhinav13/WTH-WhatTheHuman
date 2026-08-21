"use client";

import { motion, useReducedMotion } from "framer-motion";
import type { DomainClaim } from "@/types/api";
import { CitationChip } from "@/components/CitationChip";

export function ClaimRow({
  claim,
  onCitation,
}: {
  claim: DomainClaim;
  onCitation: (citationRef: string) => void;
}) {
  const reducedMotion = useReducedMotion();

  return (
    <article className="border-t hairline py-5 first:border-t-0 first:pt-0">
      <p className="manuscript-text text-[1.08rem]">{claim.text}</p>
      {claim.citation_refs.length ? (
        <motion.div
          initial={reducedMotion ? false : { opacity: 0, filter: "blur(2px)" }}
          animate={{ opacity: 1, filter: "blur(0px)" }}
          transition={{ duration: reducedMotion ? 0 : 0.35, delay: reducedMotion ? 0 : 0.22 }}
          className="mt-3 flex flex-wrap items-center gap-2"
          aria-label={`Citations for ${claim.claim_ref}`}
        >
          {claim.citation_refs.map((citationRef) => (
            <CitationChip key={citationRef} citationRef={citationRef} onOpen={onCitation} />
          ))}
          <div className="basis-full space-y-1 pt-1">
            {claim.citations.map((citation, index) => (
              <p key={`${citation.source_id}-${index}`} className="manuscript-text text-sm italic ink-muted">
                {citation.citation}
              </p>
            ))}
          </div>
        </motion.div>
      ) : null}
    </article>
  );
}
