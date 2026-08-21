import { Suspense } from "react";
import { AskExperience } from "@/components/AskExperience";

export default function HomePage() {
  return (
    <Suspense fallback={<div className="min-h-64" aria-hidden="true" />}>
      <AskExperience />
    </Suspense>
  );
}
