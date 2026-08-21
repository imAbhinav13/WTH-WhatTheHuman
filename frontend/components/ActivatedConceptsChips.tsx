import type { ActivatedConcept } from "@/types/api";

export function ActivatedConceptsChips({ concepts }: { concepts: ActivatedConcept[] }) {
  if (!concepts.length) return null;
  return (
    <section aria-label="Activated concepts" className="flex flex-wrap gap-2">
      {concepts.map((concept) => (
        <span
          key={concept.concept}
          className="rounded-full border border-ink/20 bg-bg-raised/60 px-3 py-1 text-xs ink-muted"
          title={`${concept.coverage_status} · ${concept.coverage_score.toFixed(1)}/100`}
        >
          {concept.display_name}
        </span>
      ))}
    </section>
  );
}
