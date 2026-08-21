"use client";

import { useEffect, useState } from "react";
import { useReducedMotion } from "framer-motion";
import type { Domain, FinalResponse } from "@/types/api";
import { resolveCitationRef } from "@/lib/response";
import { CoverageIndicator } from "@/components/CoverageIndicator";
import { ActivatedConceptsChips } from "@/components/ActivatedConceptsChips";
import { InterpretationLine } from "@/components/InterpretationLine";
import { SourcePanel } from "@/components/SourcePanel";
import { ComparativeSynthesis } from "@/components/ComparativeSynthesis";
import { NonEquivalencesBlock } from "@/components/NonEquivalencesBlock";
import { KeyTensionsBlock } from "@/components/KeyTensionsBlock";
import { NonConclusion } from "@/components/NonConclusion";
import { GeneralKnowledgeFallbackNotice } from "@/components/GeneralKnowledgeFallbackNotice";
import { CitationDrawer, type CitationSelection } from "@/components/CitationDrawer";

const domainOrder: Domain[] = ["science", "advaita", "samkhya"];

export function ResponseView({
  response,
  onRevealComplete,
}: {
  response: FinalResponse;
  onRevealComplete: () => void;
}) {
  const reducedMotion = useReducedMotion();
  const [stage, setStage] = useState(reducedMotion ? 4 : 0);
  const [selection, setSelection] = useState<CitationSelection | null>(null);

  useEffect(() => {
    if (reducedMotion) {
      setStage(4);
      onRevealComplete();
      return;
    }

    // Intentional client-side pacing after one complete atomic /api/query JSON response.
    // This is simulated reveal timing only; it is NOT SSE, streaming, or network-driven arrival.
    setStage(1);
    const advaita = window.setTimeout(() => setStage(2), 500);
    const samkhya = window.setTimeout(() => setStage(3), 1000);
    const synthesis = window.setTimeout(() => setStage(4), 2200);
    const revealComplete = window.setTimeout(() => onRevealComplete(), 3350);

    return () => {
      window.clearTimeout(advaita);
      window.clearTimeout(samkhya);
      window.clearTimeout(synthesis);
      window.clearTimeout(revealComplete);
    };
  }, [onRevealComplete, reducedMotion, response.generated_at]);

  function openCitation(citationRef: string) {
    setSelection({
      citationRef,
      citation: resolveCitationRef(response, citationRef),
    });
  }

  return (
    <section className="mt-7 space-y-8" aria-label="WTH response">
      <CoverageIndicator coverage={response.sections.coverage} />
      <ActivatedConceptsChips concepts={response.sections.activated_concepts} />
      <InterpretationLine text={response.sections.interpretation} />

      <div className="space-y-7">
        {domainOrder.map((domain, index) =>
          stage >= index + 1 ? (
            <SourcePanel
              key={domain}
              domain={domain}
              perspective={response.sections.domain_perspectives[domain]}
              onCitation={openCitation}
            />
          ) : null,
        )}
      </div>

      {stage >= 4 ? (
        <div className="space-y-9">
          <ComparativeSynthesis
            synthesis={response.sections.comparative_synthesis}
            onCitation={openCitation}
          />
          <NonEquivalencesBlock items={response.sections.non_equivalences} onCitation={openCitation} />
          <KeyTensionsBlock items={response.sections.key_tensions} onCitation={openCitation} />
          <NonConclusion text={response.sections.comparative_synthesis.non_conclusion} />
          <GeneralKnowledgeFallbackNotice fallback={response.sections.general_knowledge_fallback} />
        </div>
      ) : null}

      <CitationDrawer selection={selection} onClose={() => setSelection(null)} />
    </section>
  );
}
