"use client";

import { motion, useReducedMotion } from "framer-motion";
import type { ComparativeSynthesis as ComparativeSynthesisType } from "@/types/api";
import { ComparisonCard } from "@/components/ComparisonCard";

export function ComparativeSynthesis({
  synthesis,
  onCitation,
}: {
  synthesis: ComparativeSynthesisType;
  onCitation: (citationRef: string) => void;
}) {
  const reducedMotion = useReducedMotion();
  return (
    <motion.section
      aria-labelledby="comparative-synthesis-heading"
      initial={reducedMotion ? false : { opacity: 0, filter: "blur(8px)", y: 5 }}
      animate={{ opacity: 1, filter: "blur(0px)", y: 0 }}
      transition={{ duration: reducedMotion ? 0 : 1.1, ease: "easeOut" }}
      className="border-t-2 border-ink/25 pt-8"
    >
      <p className="text-xs font-semibold uppercase tracking-[0.18em] ink-muted">Comparison, not a fourth perspective</p>
      <h2 id="comparative-synthesis-heading" className="mt-1 font-serif text-3xl font-semibold">
        Comparative synthesis
      </h2>
      <p className="manuscript-text mt-4 text-lg">{synthesis.summary}</p>
      <p className="manuscript-text mt-3 text-base italic ink-muted">{synthesis.three_way_overview}</p>

      {synthesis.comparisons.length ? (
        <div className="mt-7 rounded-sm bg-bg-raised/55 px-5 py-5 sm:px-7">
          {synthesis.comparisons.map((comparison) => (
            <ComparisonCard key={comparison.comparison_id} comparison={comparison} onCitation={onCitation} />
          ))}
        </div>
      ) : (
        <p className="manuscript-text mt-6 text-base ink-muted">No pairwise comparison entries were returned.</p>
      )}
    </motion.section>
  );
}
