"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { useSearchParams } from "next/navigation";
import { useReducedMotion } from "framer-motion";

import { queryWth } from "@/lib/api";
import { WthApiError } from "@/lib/errors";
import {
  QUESTION_MAX_LENGTH,
  QUESTION_MIN_LENGTH,
} from "@/lib/response";

import type { FinalResponse } from "@/types/api";

import { QuestionInput } from "@/components/QuestionInput";
import {
  ShatkonaAnimation,
  type ShatkonaState,
} from "@/components/ShatkonaAnimation";
import { ResponseView } from "@/components/ResponseView";
import { ErrorState } from "@/components/ErrorState";

export function AskExperience() {
  const searchParams = useSearchParams();
  const reducedMotion = useReducedMotion();

  const [question, setQuestion] = useState("");
  const [formError, setFormError] = useState<string | null>(null);

  const [requestError, setRequestError] = useState<{
    message: string;
    retryAfterSeconds?: number | null;
  } | null>(null);

  const [response, setResponse] = useState<FinalResponse | null>(null);
  const [isQuerying, setIsQuerying] = useState(false);

  const [shatkonaState, setShatkonaState] =
    useState<ShatkonaState>("idle");

  const abortRef = useRef<AbortController | null>(null);
  const inFlightRef = useRef(false);
  const settleTimerRef = useRef<number | null>(null);

  /**
   * Marks the beginning of the generated answer.
   *
   * When a valid response arrives, the page scrolls here rather than
   * scrolling all the way to the bottom of the generated content.
   */
  const responseStartRef = useRef<HTMLDivElement | null>(null);

  const prefillQuestion = searchParams.get("q");

  /**
   * /try-these uses ?q= to prefill the question.
   *
   * Important:
   * Prefill only — never auto-submit.
   */
  useEffect(() => {
    if (prefillQuestion !== null) {
      setQuestion(
        prefillQuestion.slice(0, QUESTION_MAX_LENGTH),
      );
    }
  }, [prefillQuestion]);

  /**
   * Automatically move the reader to the start of the answer once
   * a complete, validated response has been mounted.
   *
   * Two requestAnimationFrame calls allow React to commit the response
   * area to the DOM before attempting the scroll.
   */
  useEffect(() => {
    if (!response) return;

    let firstFrame = 0;
    let secondFrame = 0;

    firstFrame = window.requestAnimationFrame(() => {
      secondFrame = window.requestAnimationFrame(() => {
        responseStartRef.current?.scrollIntoView({
          behavior: reducedMotion ? "auto" : "smooth",
          block: "start",
        });
      });
    });

    return () => {
      window.cancelAnimationFrame(firstFrame);

      if (secondFrame) {
        window.cancelAnimationFrame(secondFrame);
      }
    };
  }, [response, reducedMotion]);

  /**
   * Cleanup in-flight requests and animation timers.
   */
  useEffect(() => {
    return () => {
      abortRef.current?.abort();

      if (settleTimerRef.current !== null) {
        window.clearTimeout(settleTimerRef.current);
      }
    };
  }, []);

  /**
   * ResponseView calls this after the intentionally staged
   * client-side reveal has finished.
   *
   * The reveal is visual pacing only.
   * The backend response itself is atomic — there is no streaming.
   */
  const handleRevealComplete = useCallback(() => {
    setShatkonaState("complete");

    if (settleTimerRef.current !== null) {
      window.clearTimeout(settleTimerRef.current);
    }

    settleTimerRef.current = window.setTimeout(() => {
      setShatkonaState("idle");
    }, 800);
  }, []);

  async function submit() {
    /**
     * Prevent duplicate submissions from rapid clicks / Enter presses.
     */
    if (isQuerying || inFlightRef.current) {
      return;
    }

    const trimmed = question.trim();

    /**
     * Frontend soft validation.
     * Backend validation remains authoritative.
     */
    if (trimmed.length < QUESTION_MIN_LENGTH) {
      setFormError("Please enter at least 3 characters.");
      return;
    }

    if (trimmed.length > QUESTION_MAX_LENGTH) {
      setFormError(
        "Please keep the question to 1000 characters or fewer.",
      );
      return;
    }

    inFlightRef.current = true;

    setFormError(null);
    setRequestError(null);

    /**
     * WTH is single-question / single-response.
     *
     * A new query replaces the previous reading rather than
     * creating a conversation thread.
     */
    setResponse(null);

    setIsQuerying(true);
    setShatkonaState("resolving");

    /**
     * Cancel any previous request that might still exist.
     */
    abortRef.current?.abort();

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const result = await queryWth(trimmed, {
        signal: controller.signal,
      });

      /**
       * Defensive trust boundary.
       *
       * Never display a backend response as a normal WTH answer
       * unless Phase 18 validation explicitly passed.
       */
      if (result.validation.passed !== true) {
        setRequestError({
          message:
            "WTH couldn't validate this answer safely, so it has not been shown. Please try again.",
        });

        setShatkonaState("complete");

        if (settleTimerRef.current !== null) {
          window.clearTimeout(settleTimerRef.current);
        }

        settleTimerRef.current = window.setTimeout(() => {
          setShatkonaState("idle");
        }, 800);

        return;
      }

      /**
       * Setting the response mounts the answer area.
       *
       * The response effect above then smoothly scrolls to
       * responseStartRef.
       */
      setResponse(result);
    } catch (error) {
      /**
       * Aborted queries are intentional and should not produce
       * user-facing failures.
       */
      if (
        error instanceof DOMException &&
        error.name === "AbortError"
      ) {
        return;
      }

      if (error instanceof WthApiError) {
        setRequestError({
          message: error.message,
          retryAfterSeconds: error.retryAfterSeconds,
        });
      } else {
        setRequestError({
          message:
            "WTH couldn't complete this answer. Please try again.",
        });
      }

      setShatkonaState("complete");

      if (settleTimerRef.current !== null) {
        window.clearTimeout(settleTimerRef.current);
      }

      settleTimerRef.current = window.setTimeout(() => {
        setShatkonaState("idle");
      }, 800);
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
      }

      inFlightRef.current = false;
      setIsQuerying(false);
    }
  }

  return (
    <div>
      {/* ---------------------------------------------------------
          Intro
      --------------------------------------------------------- */}

      <section className="pt-3 text-center sm:pt-5">
        <p className="text-xs font-semibold uppercase tracking-[0.22em] ink-muted">
          What The Human
        </p>

        <h1 className="mt-3 font-serif text-5xl font-semibold tracking-tight sm:text-6xl">
          One question. Three lenses.
        </h1>

        <p className="manuscript-text mx-auto mt-5 max-w-2xl text-lg ink-muted sm:text-xl">
          Read Science, Advaita Vedanta, and Samkhya independently,
          inspect the evidence behind each claim, then compare where
          their answers overlap, diverge, or do not meaningfully meet.
        </p>
      </section>

      {/* ---------------------------------------------------------
          Question
      --------------------------------------------------------- */}

      <QuestionInput
        value={question}
        onChange={(value) => {
          setQuestion(value);

          if (formError) {
            setFormError(null);
          }
        }}
        onSubmit={submit}
        disabled={isQuerying}
        error={formError}
      />

      {/* ---------------------------------------------------------
          WTH resolving indicator
      --------------------------------------------------------- */}

      <ShatkonaAnimation state={shatkonaState} />

      {/* ---------------------------------------------------------
          Request error
      --------------------------------------------------------- */}

      {requestError ? (
        <ErrorState
          message={requestError.message}
          retryAfterSeconds={
            requestError.retryAfterSeconds
          }
        />
      ) : null}

      {/* ---------------------------------------------------------
          Generated answer

          scroll-mt-32 intentionally leaves visual breathing room
          above the response instead of pinning it directly against
          the top of the viewport.
      --------------------------------------------------------- */}

      {response ? (
        <div
          ref={responseStartRef}
          className="scroll-mt-32"
        >
          <ResponseView
            key={response.generated_at}
            response={response}
            onRevealComplete={handleRevealComplete}
          />
        </div>
      ) : !isQuerying && !requestError ? (
        <section className="mt-3 border-t hairline pt-6 text-center">
          <p className="manuscript-text text-base ink-muted">
            WTH does not hold a conversation thread. Each submission
            replaces the previous reading and its response-scoped
            citations.
          </p>
        </section>
      ) : null}
    </div>
  );
}