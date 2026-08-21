import Link from "next/link";
import type { ComponentType } from "react";
import { FaGithub as FaGithubIcon, FaLinkedinIn as FaLinkedinInIcon } from "react-icons/fa6";

const FaGithub = FaGithubIcon as ComponentType<{ className?: string }>;
const FaLinkedinIn = FaLinkedinInIcon as ComponentType<{ className?: string }>;

const GITHUB_URL = "https://github.com/imAbhinav13/WTH-WhatTheHuman";
const LINKEDIN_URL = "https://www.linkedin.com/in/abhinav-nagathan/";

export function Header() {
  return (
    <header className="border-b hairline">
      <div className="mx-auto flex max-w-5xl items-center px-6 py-5">
        {/* Brand / left side */}
        <Link
          href="/"
          className="font-serif text-lg font-semibold tracking-tight"
        >
          WTH
        </Link>

        {/* Navigation */}
        <nav
          className="ml-auto flex items-center gap-6"
          aria-label="Primary navigation"
        >
          <Link
            href="/"
            className="text-sm ink-muted transition-opacity hover:opacity-60"
          >
            ASK
          </Link>

          <Link
            href="/try-these"
            className="text-sm ink-muted transition-opacity hover:opacity-60"
          >
            TRY THESE
          </Link>

          <Link
            href="/about"
            className="text-sm ink-muted transition-opacity hover:opacity-60"
          >
            ABOUT
          </Link>

          {/* Social links — visually separated from page navigation */}
          <div className="ml-2 flex items-center gap-3 border-l hairline pl-5">
            <a
              href={GITHUB_URL}
              target="_blank"
              rel="noopener noreferrer"
              aria-label="WTH on GitHub"
              title="GitHub"
              className="ink-muted transition-opacity hover:opacity-60"
            >
              <FaGithub className="h-[18px] w-[18px]" />
            </a>

            <a
              href={LINKEDIN_URL}
              target="_blank"
              rel="noopener noreferrer"
              aria-label="Abhinav Nagathan on LinkedIn"
              title="LinkedIn"
              className="ink-muted transition-opacity hover:opacity-60"
            >
              <FaLinkedinIn className="h-[18px] w-[18px]" />
            </a>
          </div>
        </nav>
      </div>
    </header>
  );
}