export function ErrorState({
  message,
  retryAfterSeconds,
}: {
  message: string;
  retryAfterSeconds?: number | null;
}) {
  return (
    <div className="border-l-2 border-danger/55 bg-bg-raised/55 px-4 py-4" role="status">
      <p className="manuscript-text text-lg">{message}</p>
      {typeof retryAfterSeconds === "number" ? (
        <p className="mt-2 text-xs ink-muted">A retry may work in about {Math.ceil(retryAfterSeconds)} seconds.</p>
      ) : null}
    </div>
  );
}
