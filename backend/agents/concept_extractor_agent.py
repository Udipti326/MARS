from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.prompts import PromptTemplate

from backend.core.llm import get_llm
from backend.utils.parser import safe_json_parse


def _dedupe_concepts(concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []

    for concept in concepts:
        if not isinstance(concept, dict):
            continue
        name = str(concept.get("normalized_name") or concept.get("name") or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)

        out.append(
            {
                "name": str(concept.get("name") or name).strip(),
                "normalized_name": key,
                "category": str(concept.get("category") or "general").strip(),
                "importance": str(concept.get("importance") or "medium").strip(),
                "evidence_claims": concept.get("evidence_claims", []),
                "notes": concept.get("notes", ""),
            }
        )

    return out


class ConceptExtractorAgent:
    """
    Extracts key concepts from a summary + claim set and returns a normalized concept list.
    """

    def __init__(self, temperature: float = 0.0):
        self.llm = get_llm(temperature=temperature)

    def run(
        self,
        summary: str,
        claims: List[Any],
        topic: str = "",
    ) -> Dict[str, Any]:
        claim_texts: List[str] = []
        for claim in claims or []:
            if isinstance(claim, str):
                claim_texts.append(claim.strip())
            elif isinstance(claim, dict):
                claim_texts.append(str(claim.get("claim") or claim.get("text") or "").strip())

        prompt = PromptTemplate.from_template(
            """
You are a concept extraction agent for a personalized knowledge system.

Return ONLY valid JSON.
No markdown, no explanation, no code fences.

Topic:
{topic}

Summary:
{summary}

Claims:
{claims}

Return this JSON schema:
{{
  "topic": "{topic}",
  "concepts": [
    {{
      "name": "...",
      "normalized_name": "...",
      "category": "core|supporting|prerequisite|tool|method|dataset|other",
      "importance": "high|medium|low",
      "evidence_claims": ["..."],
      "notes": "..."
    }}
  ],
  "topic_candidates": ["..."],
  "summary_signal": "short string describing the main conceptual signal"
}}
"""
        )

        response = (prompt | self.llm).invoke(
            {
                "topic": topic.strip() or "unspecified",
                "summary": summary.strip(),
                "claims": "\n".join(f"- {x}" for x in claim_texts) or "- none",
            }
        )

        data = safe_json_parse(response.content if hasattr(response, "content") else str(response))
        if not isinstance(data, dict):
            raise ValueError("ConceptExtractorAgent output must be a JSON object")

        concepts = data.get("concepts", [])
        if not isinstance(concepts, list):
            concepts = []

        data["topic"] = topic.strip() or data.get("topic", "unspecified")
        data["concepts"] = _dedupe_concepts(concepts)
        data["claim_count"] = len(claim_texts)
        data["summary"] = summary
        return data