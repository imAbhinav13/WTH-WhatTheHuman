"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const navItems = [
  { href: "/", label: "Ask" },
  { href: "/try-these", label: "Try These" },
  { href: "/about", label: "About" },
] as const;

export function Header() {
  const pathname = usePathname();

  return (
    <header className="border-b hairline">
      <div className="mx-auto flex w-full max-w-reading items-center justify-between px-5 py-5 sm:px-8">
        <Link href="/" className="font-serif text-xl font-semibold tracking-tight" aria-label="WTH home">
          WTH
        </Link>
        <nav aria-label="Primary navigation">
          <ul className="flex items-center gap-1 sm:gap-2">
            {navItems.map((item) => {
              const active = pathname === item.href;
              return (
                <li key={item.href}>
                  <Link
                    href={item.href}
                    aria-current={active ? "page" : undefined}
                    className={`rounded-full px-3 py-2 text-sm transition-colors ${
                      active
                        ? "bg-bg-raised text-ink"
                        : "ink-muted hover:bg-bg-raised/70 hover:text-ink"
                    }`}
                  >
                    {item.label}
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>
      </div>
    </header>
  );
}
