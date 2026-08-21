"use client";

import { FormEvent, useId } from "react";
import { QUESTION_MAX_LENGTH, QUESTION_MIN_LENGTH } from "@/lib/response";

export function QuestionInput({
  value,
  onChange,
  onSubmit,
  disabled,
  error,
}: {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  disabled: boolean;
  error?: string | null;
}) {
  const id = useId();
  const errorId = `${id}-error`;

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onSubmit();
  }

  return (
    <form onSubmit={submit} className="mt-8" aria-label="Ask WTH">
      <label htmlFor={id} className="mb-2 block text-sm font-semibold">
        Ask one question
      </label>
      <div className="rounded-sm bg-bg-raised p-2 shadow-[0_1px_0_rgba(74,46,30,0.09)]">
        <textarea
          id={id}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={(event) => {
            if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
              event.preventDefault();
              onSubmit();
            }
          }}
          disabled={disabled}
          rows={4}
          minLength={QUESTION_MIN_LENGTH}
          maxLength={QUESTION_MAX_LENGTH}
          aria-invalid={Boolean(error)}
          aria-describedby={error ? errorId : undefined}
          placeholder="How is consciousness related to the self and experienced reality?"
          className="manuscript-text block w-full resize-y bg-transparent px-3 py-3 text-lg text-ink placeholder:text-ink/40 focus:outline-none disabled:cursor-wait disabled:opacity-70"
        />
        <div className="flex items-center justify-between gap-4 border-t hairline px-3 py-2">
          <span className="text-xs tabular-nums ink-muted">{value.length}/{QUESTION_MAX_LENGTH}</span>
          <button
            type="submit"
            disabled={disabled}
            className="min-h-10 rounded-full bg-accent px-5 py-2 text-sm font-semibold text-bg-raised transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {disabled ? "Considering…" : "Ask WTH"}
          </button>
        </div>
      </div>
      {error ? (
        <p id={errorId} className="mt-2 text-sm text-danger" role="alert">
          {error}
        </p>
      ) : null}
    </form>
  );
}
