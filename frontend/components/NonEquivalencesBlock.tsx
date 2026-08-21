import type { Comparison } from "@/types/api";
import { ComparisonCard } from "@/components/ComparisonCard";

export function NonEquivalencesBlock({
  items,
  onCitation,
}: {
  items: Comparison[];
  onCitation: (citationRef: string) => void;
}) {
  if (!items.length) return null;
  return (
    <section aria-labelledby="non-equivalences-heading" className="border-l-2 border-ink/35 bg-bg-raised/45 px-5 py-6 sm:px-7">
      <p className="text-xs font-semibold uppercase tracking-[0.18em] ink-muted">Important distinctions</p>
      <h2 id="non-equivalences-heading" className="mt-1 font-serif text-3xl font-semibold">
        Non-equivalences
      </h2>
      <div className="mt-5">
        {items.map((item) => (
          <ComparisonCard key={item.comparison_id} comparison={item} onCitation={onCitation} />
        ))}
      </div>
    </section>
  );
}
