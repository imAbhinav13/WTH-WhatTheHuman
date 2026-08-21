import type { Metadata } from "next";
import { QuestionCard } from "@/components/QuestionCard";
import { TRY_THESE_QUESTIONS } from "@/data/questions";

export const metadata: Metadata = {
  title: "Try These",
  description: "Five example questions to begin exploring WTH.",
};

export default function TryThesePage() {
  return (
    <article>
      <p className="mb-3 text-xs font-semibold uppercase tracking-[0.2em] ink-muted">A few places to begin</p>
      <h1 className="font-serif text-4xl font-semibold tracking-tight sm:text-5xl">Try these</h1>
      <p className="manuscript-text mt-5 max-w-2xl text-lg ink-muted">
        Choose a question to carry it back to the reading page.
      </p>

      <section className="mt-10 rounded-sm bg-bg-raised/50 px-5 py-2 sm:px-7" aria-label="Example questions">
        {TRY_THESE_QUESTIONS.map((question, index) => (
          <QuestionCard key={question} question={question} index={index} />
        ))}
      </section>
    </article>
  );
}
