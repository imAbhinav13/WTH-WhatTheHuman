"use client";

import { useEffect, useState } from "react";

const SESSION_KEY = "wth-welcome-seen";

export function WelcomeNotice() {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const seen = window.sessionStorage.getItem(SESSION_KEY);

    if (!seen) {
      setOpen(true);
    }
  }, []);

  function handleContinue() {
    window.sessionStorage.setItem(SESSION_KEY, "true");
    setOpen(false);
  }

  if (!open) return null;

  return (
    <div
      className="
        fixed inset-0 z-[9999]
        flex items-center justify-center
        px-5
        bg-black/25
        backdrop-blur-md
      "
      role="dialog"
      aria-modal="true"
      aria-labelledby="welcome-title"
    >
      <div
        className="
          w-full max-w-lg
          bg-white
          text-[#4A2E1E]
          border border-[#4A2E1E]/20
          px-7 py-8
          shadow-2xl
          sm:px-10 sm:py-10
        "
      >
        {/* Small heading */}
        <p className="text-xs font-semibold uppercase tracking-[0.22em] text-[#4A2E1E]/65">
          Before you begin
        </p>

        {/* Main heading */}
        <h2
          id="welcome-title"
          className="
            mt-3
            font-serif
            text-3xl
            font-semibold
            tracking-tight
            text-[#4A2E1E]
          "
        >
          A quick note about WTH
        </h2>

        {/* Message */}
        <div
          className="
            mt-6
            space-y-5
            font-serif
            text-[17px]
            leading-7
            text-[#4A2E1E]
          "
        >
          <p>
            <strong className="font-semibold">
              WTH is a live project running on a lightweight demo setup.
            </strong>{" "}
            Responses may occasionally take a little longer, especially
            on the first request, and temporary LLM-provider rate limits
            can sometimes occur.
          </p>

          <p>
            If something doesn&apos;t work, please wait a moment and try
            again.
          </p>

            <p>
            <strong className="font-semibold">
                Found an issue or have feedback?
            </strong>{" "}
            I&apos;m still learning and improving WTH, and honest feedback
            is always welcome.

            <span className="mt-2 block">
                Reach me at{" "}
                <a
                href="mailto:abhinavnagathan@gmail.com"
                className="
                    font-semibold
                    text-[#A6522F]
                    underline
                    decoration-[#A6522F]/45
                    underline-offset-4
                    transition-opacity
                    hover:opacity-65
                "
                >
                abhinavnagathan@gmail.com
                </a>
                .
            </span>
            </p>
          <p>Thanks for trying WTH !!</p>
        </div>

        {/* Continue */}
        <button
          type="button"
          onClick={handleContinue}
          autoFocus
          className="
            mt-8
            w-full
            bg-[#4A2E1E]
            px-5 py-3.5
            text-sm
            font-semibold
            uppercase
            tracking-[0.16em]
            text-white
            transition-all
            duration-200
            hover:bg-[#A6522F]
            focus-visible:outline-none
            focus-visible:ring-2
            focus-visible:ring-[#A6522F]
            focus-visible:ring-offset-2
          "
        >
          Continue to WTH
        </button>
        {/* Tiny decorative manuscript mark */}
            <div
              className="
                mx-auto mt-6
                flex items-center justify-center
                gap-2
                text-[#4A2E1E]/35
              "
              aria-hidden="true"
            >
              <span className="h-px w-8 bg-current" />
              <span className="font-serif text-sm">◆</span>
              <span className="h-px w-8 bg-current" />
            </div>
      </div>
    </div>
  );
}