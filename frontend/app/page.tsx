import { Suspense } from "react";
import { AskExperience } from "@/components/AskExperience";
import { WelcomeNotice } from "@/components/WelcomeNotice";

export default function HomePage() {
  return (
    <Suspense fallback={<div className="min-h-64" aria-hidden="true" />}>
      <WelcomeNotice />
      <AskExperience />
    </Suspense>
  );
}
