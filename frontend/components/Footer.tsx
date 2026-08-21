export function Footer() {
  const githubUrl = process.env.NEXT_PUBLIC_WTH_GITHUB_URL?.trim();

  return (
    <footer className="border-t hairline">
      <div className="mx-auto flex w-full max-w-reading flex-col gap-2 px-5 py-8 text-xs ink-muted sm:flex-row sm:items-center sm:justify-between sm:px-8">
        <p>WTH — one question, independently grounded perspectives.</p>
        <div className="flex items-center gap-3">
          <span>Corpus-grounded where cited.</span>
          {githubUrl ? (
            <a href={githubUrl} target="_blank" rel="noreferrer" className="font-semibold text-accent hover:underline">
              GitHub
            </a>
          ) : null}
        </div>
      </div>
    </footer>
  );
}
