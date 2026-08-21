"use client";

import {
  motion,
  useAnimationFrame,
  useMotionValue,
  useReducedMotion,
} from "framer-motion";
import { useRef } from "react";

export type ShatkonaState = "idle" | "resolving" | "complete";

/**
 * Six outer vertices of the Shatkona.
 * These match the two triangle vertices exactly.
 */
const points = [
  [50, 8],  // top
  [88, 28], // upper-right
  [88, 72], // lower-right
  [50, 92], // bottom
  [12, 72], // lower-left
  [12, 28], // upper-left
] as const;

export function ShatkonaAnimation({
  state,
}: {
  state: ShatkonaState;
}) {
  const reducedMotion = useReducedMotion();

  const rotate = useMotionValue(0);

  // Degrees per second
  const speed = useRef(18);

  useAnimationFrame((_time: number, delta: number) => {
    if (reducedMotion) return;

    /**
     * Idle:
     * Slow manuscript-like rotation.
     *
     * Resolving:
     * Intentionally spins very quickly after ASK is clicked.
     * At this speed the rotating Shatkona visually begins to read
     * almost like a circular working indicator.
     *
     * Complete:
     * Smoothly eases back toward idle speed.
     */
    const targetSpeed =
      state === "resolving"
        ? 420
        : state === "complete"
          ? 18
          : 18;

    /**
     * Faster ramp-up while resolving.
     * Slower easing when returning to idle.
     */
    const rampDuration = state === "resolving" ? 220 : 650;

    const blend = Math.min(1, delta / rampDuration);

    speed.current += (targetSpeed - speed.current) * blend;

    rotate.set(
      (rotate.get() + speed.current * (delta / 1000)) % 360,
    );
  });

  const pointVisible = state === "resolving";

  return (
    <div
      className="flex flex-col items-center py-8"
      aria-live="polite"
      aria-label={
        state === "resolving"
          ? "WTH is resolving the question"
          : "WTH ready"
      }
    >
      <motion.div
        style={{
          rotate: reducedMotion ? 0 : rotate,
        }}
        className="h-32 w-32 text-ink sm:h-36 sm:w-36"
      >
        <svg
          viewBox="0 0 100 100"
          className="h-full w-full overflow-visible"
          aria-hidden="true"
        >
          {/* Upward triangle */}
          <polygon
            points="50,8 88,72 12,72"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          />

          {/* Downward triangle */}
          <polygon
            points="50,92 88,28 12,28"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          />

          {/* Six outer points */}
          {points.map(([cx, cy], index) => (
            <motion.circle
              key={`${cx}-${cy}`}
              cx={cx}
              cy={cy}
              r="2.5"
              fill="currentColor"
              initial={false}
              animate={{
                opacity: pointVisible ? 0.9 : 0,
                scale: pointVisible ? 1 : 0.75,
              }}
              transition={
                reducedMotion
                  ? {
                      duration: 0,
                    }
                  : {
                      opacity: {
                        duration: 0.2,
                        delay: pointVisible ? index * 0.07 : 0,
                      },
                      scale: {
                        duration: 0.2,
                        delay: pointVisible ? index * 0.07 : 0,
                      },
                    }
              }
              style={{
                transformOrigin: `${cx}px ${cy}px`,
              }}
            />
          ))}

          {/* Center bindu */}
          <motion.circle
            cx="50"
            cy="50"
            r="3.4"
            fill="currentColor"
            initial={false}
            animate={
              reducedMotion
                ? {
                    opacity: 1,
                    scale: 1,
                  }
                : state === "complete"
                  ? {
                      scale: [1, 1.8, 1],
                      opacity: [1, 0.65, 1],
                    }
                  : state === "resolving"
                    ? {
                        scale: [1, 1.18, 1],
                        opacity: [1, 0.8, 1],
                      }
                    : {
                        scale: 1,
                        opacity: 1,
                      }
            }
            transition={
              state === "resolving"
                ? {
                    duration: 1,
                    repeat: Infinity,
                    ease: "easeInOut",
                  }
                : {
                    duration: 0.6,
                    ease: "easeOut",
                  }
            }
            style={{
              transformOrigin: "50px 50px",
            }}
          />
        </svg>
      </motion.div>

      {state === "resolving" ? (
        <motion.p
          initial={
            reducedMotion
              ? false
              : {
                  opacity: 0,
                  y: 3,
                }
          }
          animate={{
            opacity: 1,
            y: 0,
          }}
          className="mt-4 text-xs ink-muted"
        >
          This spans a few areas — give me a moment.
        </motion.p>
      ) : null}
    </div>
  );
}