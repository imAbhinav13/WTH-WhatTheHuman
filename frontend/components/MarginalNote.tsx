export function MarginalNote({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <aside
      className="relative my-10 border-l-2 border-ink/35 bg-bg-raised/60 px-5 py-5 sm:px-7"
      aria-label={title}
    >
      <p className="mb-2 text-xs font-semibold uppercase tracking-[0.18em] ink-muted">{title}</p>
      <div className="manuscript-text text-[1.04rem]">{children}</div>
    </aside>
  );
}
