export function NonConclusion({ text }: { text: string }) {
  if (!text.trim()) return null;
  return (
    <section className="border-t hairline pt-6" aria-labelledby="non-conclusion-heading">
      <h2 id="non-conclusion-heading" className="text-xs font-semibold uppercase tracking-[0.18em] ink-muted">
        What this comparison does not establish
      </h2>
      <p className="manuscript-text mt-3 text-lg italic">{text}</p>
    </section>
  );
}
