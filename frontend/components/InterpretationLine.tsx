export function InterpretationLine({ text }: { text: string }) {
  if (!text.trim()) return null;
  return (
    <p className="manuscript-text border-l border-ink/20 pl-4 text-[1.02rem] italic ink-muted">
      {text}
    </p>
  );
}
