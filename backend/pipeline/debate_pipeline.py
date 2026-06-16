from __future__ import annotations

from backend.agents.support_agent import SupportAgent
from backend.agents.skeptic_agent import SkepticAgent
from backend.agents.judge_agent import JudgeAgent
from backend.agents.fact_check_agent import FactCheckAgent
from backend.services.debate_evidence_service import DebateEvidenceService
from backend.utils.parser import safe_json_parse


class DebatePipeline:
    def __init__(self):
        self.support_agent = SupportAgent()
        self.skeptic_agent = SkepticAgent()
        self.judge_agent = JudgeAgent()
        self.factcheck_agent = FactCheckAgent()
        self.evidence_service = DebateEvidenceService()

    def run(self, query, claim, source_bundle, sources, all_chunks, sources_data=None):
        raw_sources = sources_data if sources_data is not None else sources
        raw_sources = raw_sources or []

        evidence_pack = self.evidence_service.build_claim_evidence_pack(
            query=query,
            claim=claim,
            sources=raw_sources,
            source_bundle=source_bundle,
            top_k=8,
        )

        evidence_context = evidence_pack["evidence_context"]

        support_raw = self.support_agent.run(claim, evidence_context, query=query)
        support_data = support_raw if isinstance(support_raw, dict) else safe_json_parse(support_raw)

        skeptic_raw = self.skeptic_agent.run(claim, evidence_context, query=query)
        skeptic_data = skeptic_raw if isinstance(skeptic_raw, dict) else safe_json_parse(skeptic_raw)

        judge_raw = self.judge_agent.run(
            claim=claim,
            support_data=support_data,
            skeptic_data=skeptic_data,
            evidence_context=evidence_context,
            query=query,
        )
        judge_data = judge_raw if isinstance(judge_raw, dict) else safe_json_parse(judge_raw)

        factcheck = self.factcheck_agent.run(
            query=query,
            claim=claim,
            support_data=support_data,
            skeptic_data=skeptic_data,
            sources=raw_sources,
            context=evidence_context,
        )

        return {
            "claim": claim,
            "support": support_data,
            "skeptic": skeptic_data,
            "judge": judge_data,
            "factcheck": factcheck,
            "evidence_pack": evidence_pack,
            "evidence_context": evidence_context,
        }