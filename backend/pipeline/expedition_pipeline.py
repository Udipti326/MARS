from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, Dict, List
from uuid import uuid4

from backend.agents.fact_check_agent import FactCheckAgent
from backend.agents.fetcher_agent import FetcherAgent
from backend.agents.refinement_agent import RefinementAgent
from backend.agents.summarizer_agent import SummarizerAgent
from backend.pipeline.debate_pipeline import DebatePipeline
from backend.services.expedition_store import ExpeditionStore
from backend.services.scoring_service import ScoringService
from backend.services.source_context_service import SourceContextService


def _normalize_verdict(verdict: Any) -> str:
    v = str(verdict or "").strip().lower()
    if "likely true" in v:
        return "LIKELY_TRUE"
    if v == "true":
        return "TRUE"
    if "false" in v:
        return "FALSE"
    if "uncertain" in v:
        return "UNCERTAIN"
    return "UNCERTAIN"


class ExpeditionPipeline:
    def __init__(self, storage_dir: str = "expeditions"):
        self.fetcher = FetcherAgent()
        self.summarizer = SummarizerAgent()
        self.debate_pipeline = DebatePipeline()
        self.factcheck_agent = FactCheckAgent()
        self.refinement_agent = RefinementAgent()
        self.store = ExpeditionStore(storage_dir)

    def run(self, query: str) -> Dict[str, Any]:
        expedition_id = f"exp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        created_at = datetime.utcnow().isoformat() + "Z"

        sources_data = self.fetcher.run(query)

        if not sources_data:
            expedition = {
                "expedition_id": expedition_id,
                "created_at": created_at,
                "query": query,
                "sources": [],
                "source_bundle": "",
                "summary": {
                    "summary": "No relevant sources were found.",
                    "claims": [],
                },
                "claims": [],
                "refinement": {
                    "max_iterations": 0,
                    "weak_claims": [],
                    "strong_claims": [],
                    "needs_rerun": False,
                    "total_claims": 0,
                    "weak_count": 0,
                    "strong_count": 0,
                    "refinement_plan": [],
                },
                "overall": {
                    "verdict": "UNCERTAIN",
                    "confidence": 0.0,
                    "label": "LOW",
                    "supported_claims": 0,
                    "mixed_claims": 0,
                    "contradicted_claims": 0,
                    "total_claims": 0,
                },
            }
            saved_path = self.store.save(expedition, filename=f"{expedition_id}.json")
            expedition["saved_path"] = str(saved_path)
            return expedition

        compact_sources = SourceContextService.prepare_sources(sources_data, max_sources=12)
        source_bundle = SourceContextService.build_source_bundle(
            compact_sources,
            max_sources=12,
            max_chars_per_source=800,
        )

        summary_data = self.summarizer.run(query, source_bundle)
        if not isinstance(summary_data, dict):
            summary_data = {"summary": str(summary_data), "claims": []}

        claims = summary_data.get("claims", [])
        if not isinstance(claims, list):
            claims = []

        sources_for_scoring = [
            {
                "title": s.get("title", ""),
                "domain": s.get("domain", ""),
                "url": s.get("url", ""),
                "source_type": s.get("source_type", ""),
                "citations": s.get("extra", {}).get("stars", 50),
                "recency": 0.8,
                "rank_score": s.get("rank_score", 0.0),
            }
            for s in compact_sources
        ]

        all_chunks = [{"similarity": 0.7} for _ in compact_sources]
        claim_records: List[Dict[str, Any]] = []

        for idx, claim in enumerate(claims, start=1):
            claim_text = str(claim).strip()
            if not claim_text:
                continue

            debate_result = self.debate_pipeline.run(
                query=query,
                claim=claim_text,
                source_bundle=source_bundle,
                sources=sources_for_scoring,
                all_chunks=all_chunks,
                sources_data=compact_sources,
            )

            support = debate_result["support"]
            skeptic = debate_result["skeptic"]
            judge = debate_result["judge"]
            evidence_pack = debate_result["evidence_pack"]

            factcheck = self.factcheck_agent.run(
                query=query,
                claim=claim_text,
                support_data=support,
                skeptic_data=skeptic,
                sources=sources_for_scoring,
                context=debate_result.get("evidence_context", source_bundle),
            )

            scores = ScoringService.score_claim(
                {
                    "support_chunks": support.get("support_chunks", []),
                    "contradictions": skeptic.get("contradictions", []),
                    "sources": sources_for_scoring,
                    "all_chunks": all_chunks,
                }
            )

            claim_records.append(
                {
                    "claim_index": idx,
                    "claim": claim_text,
                    "support": support,
                    "skeptic": skeptic,
                    "judge": judge,
                    "factcheck": factcheck,
                    "evidence_pack": evidence_pack,
                    "scores": scores,
                }
            )

        refinement = self.refinement_agent.run(claim_records)

        verdicts = [_normalize_verdict(item.get("judge", {}).get("verdict")) for item in claim_records]
        verdict_counts = Counter(verdicts)

        confidence_values = [
            item.get("scores", {}).get("confidence", 0.0)
            for item in claim_records
            if isinstance(item.get("scores", {}), dict)
        ]
        overall_confidence = round(sum(confidence_values) / len(confidence_values), 3) if confidence_values else 0.0

        if verdict_counts:
            overall_verdict = verdict_counts.most_common(1)[0][0]
        else:
            overall_verdict = "UNCERTAIN"

        if overall_confidence >= 0.75:
            overall_label = "HIGH"
        elif overall_confidence >= 0.5:
            overall_label = "MEDIUM"
        else:
            overall_label = "LOW"

        expedition = {
            "expedition_id": expedition_id,
            "created_at": created_at,
            "query": query,
            "sources": compact_sources,
            "source_bundle": source_bundle,
            "summary": summary_data,
            "claims": claim_records,
            "refinement": refinement,
            "overall": {
                "verdict": overall_verdict,
                "confidence": overall_confidence,
                "label": overall_label,
                "supported_claims": verdict_counts.get("TRUE", 0) + verdict_counts.get("LIKELY_TRUE", 0),
                "mixed_claims": verdict_counts.get("UNCERTAIN", 0),
                "contradicted_claims": verdict_counts.get("FALSE", 0),
                "total_claims": len(claim_records),
            },
        }

        saved_path = self.store.save(expedition, filename=f"{expedition_id}.json")
        expedition["saved_path"] = str(saved_path)
        return expedition