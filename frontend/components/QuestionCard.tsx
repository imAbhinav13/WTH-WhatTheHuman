import Link from "next/link";

export function QuestionCard({ question, index }: { question: string; index: number }) {
  return (
    <Link
      href={`/?q=${encodeURIComponent(question)}`}
      className="group block border-t hairline py-6 first:border-t-0 focus-visible:rounded-sm"
    >
      <div className="flex gap-4">
        <span className="pt-1 text-xs font-semibold tabular-nums ink-muted">0{index + 1}</span>
        <p className="manuscript-text text-xl leading-relaxed transition-transform group-hover:translate-x-1">
          {question}
        </p>
      </div>
    </Link>
  );
}
