export function CitationChip({ citationRef, onOpen }: { citationRef: string; onOpen: (citationRef: string) => void }) {
  return (
    <button
      type="button"
      onClick={() => onOpen(citationRef)}
      className="min-h-8 rounded-full border border-ink/25 bg-bg-raised px-2.5 py-1 font-sans text-xs font-semibold text-accent transition-colors hover:border-accent/50 hover:bg-bg-base"
      aria-label={`Open evidence ${citationRef}`}
    >
      [{citationRef}]
    </button>
  );
}
