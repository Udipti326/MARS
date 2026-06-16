# backend/pipeline/research_pipeline.py

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from backend.agents.fetcher_agent import FetcherAgent
from backend.agents.support_agent import SupportAgent
from backend.agents.skeptic_agent import SkepticAgent
from backend.agents.judge_agent import JudgeAgent
from backend.agents.fact_checker_agent import FactCheckerAgent
from backend.agents.summarizer_agent import SummarizerAgent

from backend.services.evidence_retrieval_service import EvidenceRetrievalService
from backend.services.scoring_service import ScoringService
from backend.services.source_context_service import SourceContextService


class ResearchPipeline:
    """
    Full research intelligence pipeline.

    FLOW:
    User Query
        ->
    Retrieval
        ->
    Semantic Chunking
        ->
    Embedding + Similarity + Reranking
        ->
    Top-K Evidence
        ->
    Debate System
            - Support Agent
            - Skeptic Agent
            - Judge Agent
        ->
    Fact Checker
        ->
    Summary + Confidence + Verdicts
        ->
    Save Expedition JSON
    """

    def __init__(self):
        self.fetcher = FetcherAgent()

        self.support_agent = SupportAgent()
        self.skeptic_agent = SkepticAgent()
        self.judge_agent = JudgeAgent()
        self.fact_checker = FactCheckerAgent()
        self.summarizer = SummarizerAgent()

        self.retrieval = EvidenceRetrievalService()

        self.output_dir = Path("expeditions")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # SESSION ID
    # ---------------------------------------------------------

    def _session_id(self, query: str) -> str:
        return f"research::{hashlib.sha1(query.encode('utf-8')).hexdigest()[:16]}"

    # ---------------------------------------------------------
    # SOURCE INGESTION
    # ---------------------------------------------------------

    def _ingest_sources(
        self,
        session_id: str,
        query: str,
        sources: List[Dict[str, Any]],
    ) -> int:

        try:
            count = self.retrieval.ingest_sources(
                session_id=session_id,
                query=query,
                sources=sources,
            )

            print(f"\n[PIPELINE] Stored {count} semantic chunks.\n")

            return count

        except Exception as exc:
            print("\n[PIPELINE] Chunk ingestion failed:")
            print(exc)
            return 0

    # ---------------------------------------------------------
    # CLAIM EXTRACTION
    # ---------------------------------------------------------

    def _extract_claims(
        self,
        query: str,
        sources: List[Dict[str, Any]],
    ) -> List[str]:

        claims = []

        for src in sources[:10]:

            content = str(src.get("content", "")).strip()

            if not content:
                continue

            pieces = content.split(". ")

            for p in pieces:

                p = p.strip()

                if len(p) < 80:
                    continue

                if query.lower() not in p.lower():
                    continue

                claims.append(p)

                if len(claims) >= 5:
                    break

            if len(claims) >= 5:
                break

        # fallback
        if not claims:
            claims = [
                f"{query} is an important concept in modern research."
            ]

        return claims[:5]

    # ---------------------------------------------------------
    # BUILD CLAIM ANALYSIS
    # ---------------------------------------------------------

    def _analyze_claim(
        self,
        session_id: str,
        query: str,
        claim: str,
    ) -> Dict[str, Any]:

        # -------------------------------------------------
        # Retrieve top semantic evidence
        # -------------------------------------------------

        evidence_pack = self.retrieval.build_claim_evidence_pack(
            session_id=session_id,
            query=query,
            claim=claim,
            top_k=8,
        )

        selected_evidence = evidence_pack.get("selected_evidence", [])
        evidence_context = evidence_pack.get("evidence_context", "")

        # -------------------------------------------------
        # Debate Agents
        # -------------------------------------------------

        support_output = self.support_agent.run(
            query=query,
            claim=claim,
            evidence=evidence_context,
        )

        skeptic_output = self.skeptic_agent.run(
            query=query,
            claim=claim,
            evidence=evidence_context,
        )

        judge_output = self.judge_agent.run(
            query=query,
            claim=claim,
            support_argument=support_output,
            skeptic_argument=skeptic_output,
            evidence=evidence_context,
        )

        # -------------------------------------------------
        # Fact Checker
        # -------------------------------------------------

        fact_check = self.fact_checker.run(
            query=query,
            claim=claim,
            evidence=evidence_context,
        )

        # -------------------------------------------------
        # Evidence Split
        # -------------------------------------------------

        support_chunks = []
        contradictions = []

        for ev in selected_evidence:

            score = float(ev.get("retrieval_score", 0.0))

            if score >= 0.60:
                support_chunks.append(ev)

            if any(
                x in ev.get("text", "").lower()
                for x in [
                    "however",
                    "but",
                    "incorrect",
                    "false",
                    "contrary",
                    "limitation",
                ]
            ):
                contradictions.append(ev)

        # -------------------------------------------------
        # Source list
        # -------------------------------------------------

        sources = []

        for ev in selected_evidence:

            sources.append(
                {
                    "domain": ev.get("domain", ""),
                    "citations": 50,
                    "recency": 0.8,
                }
            )

        # -------------------------------------------------
        # Confidence Scores
        # -------------------------------------------------

        scoring_input = {
            "support_chunks": support_chunks,
            "all_chunks": selected_evidence,
            "contradictions": contradictions,
            "sources": sources,
        }

        scores = ScoringService.score_claim(scoring_input)

        # -------------------------------------------------
        # Verdict
        # -------------------------------------------------

        verdict = "UNCERTAIN"

        if scores["confidence"] >= 0.75:
            verdict = "LIKELY_TRUE"

        elif scores["confidence"] >= 0.45:
            verdict = "PARTIALLY_SUPPORTED"

        else:
            verdict = "WEAK_SUPPORT"

        return {
            "claim": claim,
            "verdict": verdict,
            "scores": scores,
            "support_agent": support_output,
            "skeptic_agent": skeptic_output,
            "judge_agent": judge_output,
            "fact_check": fact_check,
            "evidence": selected_evidence,
            "evidence_context": evidence_context,
        }

    # ---------------------------------------------------------
    # OVERALL VERDICT
    # ---------------------------------------------------------

    def _overall_result(
        self,
        claim_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        if not claim_results:
            return {
                "verdict": "UNCERTAIN",
                "confidence": 0.0,
                "label": "LOW",
                "supported_claims": 0,
                "mixed_claims": 0,
                "contradicted_claims": 0,
                "total_claims": 0,
            }

        confidences = [
            float(c["scores"]["confidence"])
            for c in claim_results
        ]

        avg_conf = sum(confidences) / len(confidences)

        supported = 0
        mixed = 0
        contradicted = 0

        for c in claim_results:

            verdict = c.get("verdict", "")

            if verdict == "LIKELY_TRUE":
                supported += 1

            elif verdict == "PARTIALLY_SUPPORTED":
                mixed += 1

            else:
                contradicted += 1

        if avg_conf >= 0.75:
            overall = "LIKELY_TRUE"

        elif avg_conf >= 0.45:
            overall = "PARTIALLY_SUPPORTED"

        else:
            overall = "UNCERTAIN"

        return {
            "verdict": overall,
            "confidence": round(avg_conf, 3),
            "label": ScoringService.get_label(avg_conf),
            "supported_claims": supported,
            "mixed_claims": mixed,
            "contradicted_claims": contradicted,
            "total_claims": len(claim_results),
        }

    # ---------------------------------------------------------
    # SAVE JSON
    # ---------------------------------------------------------

    def _save_expedition(
        self,
        data: Dict[str, Any],
    ) -> str:

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        fname = self.output_dir / f"exp_{timestamp}.json"

        with open(fname, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return str(fname)

    # ---------------------------------------------------------
    # MAIN RUN
    # ---------------------------------------------------------

    def run(
        self,
        query: str,
    ) -> Dict[str, Any]:

        print("\n=== RESEARCH INTELLIGENCE SYSTEM ===\n")

        session_id = self._session_id(query)

        # -------------------------------------------------
        # FETCH SOURCES
        # -------------------------------------------------

        print("[1] Fetching sources...\n")

        sources = self.fetcher.run(query)

        print(f"[PIPELINE] Retrieved {len(sources)} sources.\n")

        # -------------------------------------------------
        # HANDLE NO SOURCES
        # -------------------------------------------------

        if not sources:

            empty_result = {
                "query": query,
                "summary": "No relevant sources were found.",
                "claims": [],
                "overall": {
                    "verdict": "UNCERTAIN",
                    "confidence": 0.0,
                    "label": "LOW",
                },
            }

            path = self._save_expedition(empty_result)

            print("\n=== SUMMARY ===\n")
            print(empty_result["summary"])

            print("\n=== SAVED EXPEDITION FILE ===\n")
            print(path)

            return empty_result

        # -------------------------------------------------
        # INGEST INTO SEMANTIC RETRIEVAL SYSTEM
        # -------------------------------------------------

        print("[2] Semantic chunking + embeddings...\n")

        chunk_count = self._ingest_sources(
            session_id=session_id,
            query=query,
            sources=sources,
        )

        # -------------------------------------------------
        # BUILD SOURCE CONTEXT
        # -------------------------------------------------

        source_bundle = SourceContextService.build_source_bundle(
            sources=sources[:10],
            max_chars_per_source=1200,
        )

        # -------------------------------------------------
        # SUMMARY
        # -------------------------------------------------

        print("[3] Generating summary...\n")

        summary = self.summarizer.run(
            query=query,
            source_context=source_bundle,
        )

        # -------------------------------------------------
        # CLAIM EXTRACTION
        # -------------------------------------------------

        print("[4] Extracting claims...\n")

        claims = self._extract_claims(
            query=query,
            sources=sources,
        )

        print(f"[PIPELINE] Extracted {len(claims)} claims.\n")

        # -------------------------------------------------
        # CLAIM ANALYSIS
        # -------------------------------------------------

        print("[5] Debate + evidence analysis...\n")

        claim_results = []

        for idx, claim in enumerate(claims, start=1):

            print(f"[CLAIM {idx}] {claim[:100]}...\n")

            try:

                result = self._analyze_claim(
                    session_id=session_id,
                    query=query,
                    claim=claim,
                )

                claim_results.append(result)

            except Exception as exc:

                print("\n[PIPELINE] Claim analysis failed:")
                print(exc)

        # -------------------------------------------------
        # OVERALL RESULT
        # -------------------------------------------------

        overall = self._overall_result(claim_results)

        # -------------------------------------------------
        # FINAL DATA
        # -------------------------------------------------

        expedition_data = {
            "query": query,
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "summary": summary,
            "chunk_count": chunk_count,
            "sources": sources,
            "claims": claim_results,
            "overall": overall,
        }

        # -------------------------------------------------
        # SAVE
        # -------------------------------------------------

        save_path = self._save_expedition(expedition_data)

        # -------------------------------------------------
        # PRINT OUTPUT
        # -------------------------------------------------

        print("\n=== SUMMARY ===\n")
        print(summary)

        print("\n=== CLAIMS / VERDICTS / CONFIDENCE ===\n")

        for idx, item in enumerate(claim_results, start=1):

            print(f"\nClaim {idx}: {item['claim']}\n")

            print("Verdict:", item["verdict"])

            print(
                "Confidence:",
                item["scores"]["confidence"],
            )

            print(
                "Label:",
                item["scores"]["label"],
            )

            evidence = item.get("evidence", [])

            print(f"Evidence Count: {len(evidence)}")

            for ev in evidence[:3]:

                print(
                    f" - [{ev.get('retrieval_score', 0):.3f}] "
                    f"{ev.get('source_title', 'Unknown Source')}"
                )

        print("\n=== SAVED EXPEDITION FILE ===\n")
        print(save_path)

        print("\n=== OVERALL ===\n")
        print(json.dumps(overall, indent=2))

        print("\n=== DONE ===\n")

        return expedition_data