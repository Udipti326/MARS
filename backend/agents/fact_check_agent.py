from __future__ import annotations

from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from langchain_core.prompts import PromptTemplate

from backend.core.llm import get_llm
from backend.utils.parser import safe_json_parse


def _clamp(value: Any, low: float = 0.0, high: float = 1.0, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, value))


def _to_text_list(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    out: List[str] = []
    for item in items:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = str(item.get("text") or item.get("content") or item.get("title") or "").strip()
        else:
            text = str(item).strip()
        if text:
            out.append(text)
    return out


def _normalize_source(source: Dict[str, Any]) -> Dict[str, Any]:
    url = str(source.get("url", "")).strip()
    domain = str(source.get("domain", "")).strip()

    if not domain and url:
        try:
            domain = urlparse(url).netloc
        except Exception:
            domain = ""

    return {
        "title": source.get("title", ""),
        "url": url,
        "domain": domain,
        "source_type": source.get("source_type", source.get("type", "")),
        "credibility_hint": source.get("credibility_score", None),
    }


class FactCheckAgent:
    def __init__(self, temperature: float = 0.0):
        self.llm = get_llm(temperature=temperature)

    def run(
        self,
        query: str,
        claim: str,
        support_data: Optional[Dict[str, Any]] = None,
        skeptic_data: Optional[Dict[str, Any]] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
        context: str = "",
    ) -> Dict[str, Any]:
        support_data = support_data or {}
        skeptic_data = skeptic_data or {}
        sources = sources or []

        support_points = _to_text_list(support_data.get("support_points", []))
        support_chunks = support_data.get("support_chunks", []) if isinstance(support_data.get("support_chunks", []), list) else []
        contradictions = _to_text_list(skeptic_data.get("contradictions", []))
        assumptions = _to_text_list(skeptic_data.get("assumptions", []))
        normalized_sources = [_normalize_source(s) for s in sources if isinstance(s, dict)]

        prompt = PromptTemplate.from_template(
            """
You are the fact-checking agent in a multi-agent research system.

Return ONLY valid JSON.
Do not add markdown, explanations, or code fences.

Query:
{query}

Claim:
{claim}

Support points:
{support_points}

Contradictions:
{contradictions}

Assumptions:
{assumptions}

Sources:
{sources}

Context:
{context}

Return this JSON schema:
{{
  "verdict_hint": "supported | mixed | uncertain | contradicted",
  "support_alignment": 0.0,
  "contradiction_alignment": 0.0,
  "evidence_gaps": ["..."],
  "safety_warnings": ["..."],
  "source_quality_notes": ["..."],
  "supporting_evidence": [
    {{
      "text": "...",
      "source_hint": "...",
      "chunk_hint": "..."
    }}
  ],
  "conflicting_evidence": [
    {{
      "text": "...",
      "source_hint": "...",
      "chunk_hint": "..."
    }}
  ]
}}
"""
        )

        response = (prompt | self.llm).invoke(
            {
                "query": query or "unspecified",
                "claim": claim,
                "support_points": "\n".join(f"- {x}" for x in support_points) or "- none",
                "contradictions": "\n".join(f"- {x}" for x in contradictions) or "- none",
                "assumptions": "\n".join(f"- {x}" for x in assumptions) or "- none",
                "sources": normalized_sources,
                "context": context.strip() or "none",
            }
        )

        data = safe_json_parse(response.content if hasattr(response, "content") else str(response))
        if not isinstance(data, dict):
            raise ValueError("FactCheckAgent output must be a JSON object")

        supporting_evidence = data.get("supporting_evidence", [])
        conflicting_evidence = data.get("conflicting_evidence", [])

        if not isinstance(supporting_evidence, list):
            supporting_evidence = []
        if not isinstance(conflicting_evidence, list):
            conflicting_evidence = []

        if not data.get("safety_warnings") and any(k in claim.lower() for k in ["medical", "legal", "diagnosis", "treatment", "law", "medicine"]):
            data["safety_warnings"] = ["Sensitive domain detected; verify with authoritative sources."]

        data["query"] = query
        data["claim"] = claim
        data["support_alignment"] = _clamp(data.get("support_alignment", 0.0))
        data["contradiction_alignment"] = _clamp(data.get("contradiction_alignment", 0.0))
        data["supporting_evidence"] = supporting_evidence
        data["conflicting_evidence"] = conflicting_evidence
        data["support_chunks"] = support_chunks
        data["contradictions"] = contradictions
        data["assumptions"] = assumptions
        data["sources"] = normalized_sources
        data["source_count"] = len(normalized_sources)
        data["unique_domains"] = len({s.get("domain", "") for s in normalized_sources if s.get("domain")})
        data["context"] = context
        return data