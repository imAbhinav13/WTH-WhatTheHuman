import type { Metadata } from "next";
import { MarginalNote } from "@/components/MarginalNote";

export const metadata: Metadata = {
  title: "About",
  description: "Why WTH exists, how it works, and where its reviewed corpus stops.",
};

const githubUrl = process.env.NEXT_PUBLIC_WTH_GITHUB_URL?.trim();

export default function AboutPage() {
  return (
    <article className="space-y-12">
      <section>
        <p className="mb-3 text-xs font-semibold uppercase tracking-[0.2em] ink-muted">Origin story</p>
        <h1 className="font-serif text-4xl font-semibold tracking-tight sm:text-5xl">Why this exists</h1>
        <blockquote className="manuscript-text mt-7 border-l border-ink/25 pl-5 text-xl italic leading-relaxed sm:pl-7 sm:text-2xl">
          In a world moving fast enough that most things ask to be believed quickly, I always found myself returning to a slower question - what do we actually mean when we say something is true? Not as a debate to win, but as something worth sitting with from more than one direction at once. WTH grew out of that - a way to hold science, and traditions I wanted to understand more seriously, side by side on the same hard questions, without flattening any of them into the others.
        </blockquote>
      </section>

      <section>
        <h2 className="font-serif text-3xl font-semibold">What this is</h2>
        <p className="manuscript-text mt-4 text-lg">
          WTH is a single-question reading experience. It retrieves evidence from a reviewed corpus, lets Science, Advaita Vedanta, and Samkhya answer independently in their own terms, and then compares those answers without merging them into a synthetic doctrine.
        </p>
      </section>

      <section>
        <h2 className="font-serif text-3xl font-semibold">Why it exists</h2>
        <p className="manuscript-text mt-4 text-lg">
          The project began as a way to learn retrieval-augmented generation rigorously while also exploring questions with personal and cultural weight. The technical goal is traceability: claims should remain inspectable, limitations should remain visible, and a comparison should never become stronger than the evidence that supports it.
        </p>
      </section>

      <section>
        <h2 className="font-serif text-3xl font-semibold">How it works</h2>
        <ol className="manuscript-text mt-5 space-y-3 text-lg">
          <li><strong>1.</strong> WTH interprets the question and activates relevant concepts.</li>
          <li><strong>2.</strong> It retrieves reviewed source passages independently for each domain.</li>
          <li><strong>3.</strong> Each perspective is generated with claim-level citations and explicit limitations.</li>
          <li><strong>4.</strong> Pairwise comparisons identify overlap, analogy, tension, non-equivalence, or insufficient coverage.</li>
          <li><strong>5.</strong> Coverage and validation are returned with the answer so the interface can show what the corpus does—and does not—support.</li>
        </ol>
      </section>

      <MarginalNote title="Scope and limitations">
        <p>
          WTH does not establish metaphysical truth, scientific consensus, or equivalence between traditions. Its claims are bounded by the reviewed corpus available to the backend. “Out of Corpus” is a normal result. When coverage is incomplete, the interface preserves that incompleteness rather than filling it silently.
        </p>
      </MarginalNote>

      <section>
        <h2 className="font-serif text-3xl font-semibold">Tech stack</h2>
        <p className="manuscript-text mt-4 text-lg">
          The reading interface is built with Next.js, TypeScript, Tailwind CSS, and Framer Motion. The backend exposes a small public FastAPI contract for querying and expanding citations; provider and database credentials never enter the browser.
        </p>
        {githubUrl ? (
          <a
            href={githubUrl}
            target="_blank"
            rel="noreferrer"
            className="mt-5 inline-flex rounded-full border border-ink/25 px-4 py-2 text-sm font-semibold text-accent transition-colors hover:bg-bg-raised"
          >
            View source on GitHub
          </a>
        ) : (
          <p className="mt-5 text-sm ink-muted">Repository link can be added with NEXT_PUBLIC_WTH_GITHUB_URL.</p>
        )}
      </section>
    </article>
  );
}
