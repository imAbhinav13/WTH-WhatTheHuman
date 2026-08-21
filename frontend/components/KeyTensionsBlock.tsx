import type { Comparison } from "@/types/api";
import { ComparisonCard } from "@/components/ComparisonCard";

export function KeyTensionsBlock({
  items,
  onCitation,
}: {
  items: Comparison[];
  onCitation: (citationRef: string) => void;
}) {
  if (!items.length) return null;
  return (
    <section aria-labelledby="key-tensions-heading">
      <p className="text-xs font-semibold uppercase tracking-[0.18em] ink-muted">Where positions differ</p>
      <h2 id="key-tensions-heading" className="mt-1 font-serif text-3xl font-semibold">Key tensions</h2>
      <div className="mt-5 rounded-sm bg-bg-raised/45 px-5 py-5 sm:px-7">
        {items.map((item) => (
          <ComparisonCard key={item.comparison_id} comparison={item} onCitation={onCitation} />
        ))}
      </div>
    </section>
  );
}
