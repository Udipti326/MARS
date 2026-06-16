from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.prompts import PromptTemplate

from backend.core.llm import get_llm
from backend.services.evidence_retrieval_service import EvidenceRetrievalService
from backend.services.expedition_repository import ExpeditionRepository
from backend.utils.parser import safe_json_parse


class ExpeditionChatService:
    def __init__(self, repo: ExpeditionRepository | None = None):
        self.repo = repo or ExpeditionRepository()
        self.retrieval = EvidenceRetrievalService()
        self.llm = get_llm()

    def _history_text(self, messages: List[Dict[str, Any]], limit: int = 8) -> str:
        lines = []
        for msg in messages[-limit:]:
            role = str(msg.get("role", "") or "").strip().lower()
            content = str(msg.get("content", "") or "").strip()
            if content:
                lines.append(f"{role.upper()}: {content}")
        return "\n".join(lines)

    def answer_question(self, expedition_id: str, question: str) -> Dict[str, Any]:
        detail = self.repo.get_expedition_detail(expedition_id)
        sources = detail.get("sources", [])
        messages = detail.get("messages", [])
        root_query = detail.get("root_query", "")
        summary = detail.get("summary", {})
        overall = detail.get("overall", {})
        claims = detail.get("claims", [])

        if sources:
            self.retrieval.ingest_sources(session_id=expedition_id, query=root_query, sources=sources)

        evidence_pack = self.retrieval.build_claim_evidence_pack(
            session_id=expedition_id,
            query=root_query or question,
            claim=question,
            top_k=8,
        )

        evidence_context = evidence_pack.get("evidence_context", "")
        history_text = self._history_text(messages, limit=8)

        prompt = PromptTemplate.from_template(
            """
You are the expedition chat assistant.

Use the expedition context, evidence, prior chat, and claims to answer the user's question.
Return ONLY valid JSON.

Root Query:
{root_query}

Summary:
{summary}

Overall:
{overall}

Claims:
{claims}

Recent Chat:
{history}

Evidence:
{evidence_context}

Question:
{question}

Return this JSON schema:
{{
  "answer": "...",
  "citations": [
    {{
      "evidence_id": "E1",
      "source_title": "...",
      "source_url": "...",
      "reason": "..."
    }}
  ],
  "follow_up_questions": ["...", "..."],
  "confidence": 0.0
}}
"""
        )

        response = (prompt | self.llm).invoke(
            {
                "root_query": root_query,
                "summary": summary,
                "overall": overall,
                "claims": claims,
                "history": history_text or "none",
                "evidence_context": evidence_context,
                "question": question,
            }
        )

        parsed = safe_json_parse(response.content if hasattr(response, "content") else str(response))
        if not isinstance(parsed, dict):
            parsed = {
                "answer": str(response.content if hasattr(response, "content") else response),
                "citations": [],
                "follow_up_questions": [],
                "confidence": 0.0,
            }

        answer = str(parsed.get("answer", "") or "").strip()
        citations = parsed.get("citations", [])
        follow_up_questions = parsed.get("follow_up_questions", [])
        confidence = parsed.get("confidence", 0.0)

        if not isinstance(citations, list):
            citations = []
        if not isinstance(follow_up_questions, list):
            follow_up_questions = []

        self.repo.append_message(expedition_id, "user", question, metadata={"kind": "follow_up_question"})
        self.repo.append_message(
            expedition_id,
            "assistant",
            answer,
            metadata={"kind": "follow_up_answer", "citations": citations, "confidence": confidence},
        )

        return {
            "answer": answer,
            "citations": citations,
            "follow_up_questions": [str(x).strip() for x in follow_up_questions if str(x).strip()],
            "confidence": confidence,
            "evidence_pack": evidence_pack,
        }