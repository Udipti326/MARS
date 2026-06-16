from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

try:
    from langchain_tavily import TavilySearch
except Exception:
    TavilySearch = None

from backend.core.llm import get_llm
from backend.services.chat_memory_repository import ChatMemoryRepository
from backend.services.expedition_repository import ExpeditionRepository

CHAT_MODEL = os.getenv("CHAT_GROQ_MODEL", "llama-3.3-70b-versatile")
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.15"))
CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "450"))

MAX_SUMMARY_CHARS = int(os.getenv("CHAT_SUMMARY_CHARS", "1200"))
MAX_CLAIMS = int(os.getenv("CHAT_MAX_CLAIMS", "8"))
MAX_SOURCES = int(os.getenv("CHAT_MAX_SOURCES", "10"))
MAX_WEB_RESULTS = int(os.getenv("CHAT_MAX_WEB_RESULTS", "3"))
MAX_MEMORY_MESSAGES = int(os.getenv("CHAT_MAX_MEMORY_MESSAGES", "8"))

PROMPT_BUDGETS = [
    {"summary_chars": 1200, "claims": 8, "sources": 10, "web": 3, "memory": 8},
    {"summary_chars": 800, "claims": 6, "sources": 6, "web": 2, "memory": 6},
    {"summary_chars": 500, "claims": 4, "sources": 4, "web": 1, "memory": 4},
]

_SYSTEM_PROMPT = """You are a smart and insightful research chatbot for a saved expedition(expedition refers to a research performed by a user).

You have access to:
- saved expedition summary
- saved claims
- saved top sources
- recent chat memory
- web search results
- your own vast general knowledge

Rules:
- First use saved expedition context if relevant.
- If saved context is insufficient, use your own general knowledge and web results.
- For current or factual comparisons, use web results when available.
- Do not say "insufficient context" if you can answer using general ML knowledge and web context.
- Clearly mention when the answer is based on general knowledge/web rather than saved expedition data.
- Use citations like [S1], [C1], [W1].
- Return ONLY valid JSON.

JSON schema:
{
  "answer": "string",
  "confidence": 0.0,
  "citations": [
    {
      "label": "W1",
      "title": "string",
      "url": "string",
      "source_type": "web",
      "reason": "string"
    }
  ],
  "follow_up_questions": ["string", "string"]
}
"""


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _clip(text: Any, limit: int) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def _safe_parse_json(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)

    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


class ContextAwareChatService:
    def __init__(
        self,
        expedition_repo: ExpeditionRepository | None = None,
        chat_repo: ChatMemoryRepository | None = None,
        model_name: str | None = None,
    ):
        self.expedition_repo = expedition_repo or ExpeditionRepository()
        self.chat_repo = chat_repo or ChatMemoryRepository()
        self.model_name = model_name or CHAT_MODEL

        self.llm = get_llm(
            model_name=self.model_name,
            temperature=CHAT_TEMPERATURE,
            max_tokens=CHAT_MAX_TOKENS,
        )

        api_key = os.getenv("TAVILY_API_KEY", "").strip()
        self.web_search = None
        if TavilySearch and api_key:
            try:
                self.web_search = TavilySearch(
                    max_results=MAX_WEB_RESULTS,
                    include_raw_content=False,
                    tavily_api_key=api_key,
                )
            except Exception:
                self.web_search = None

    def _get_root_query(self, detail: Dict[str, Any]) -> str:
        expedition = detail.get("expedition") or {}
        return str(
            detail.get("root_query")
            or expedition.get("root_query")
            or detail.get("title")
            or expedition.get("title")
            or ""
        ).strip()

    def _get_title(self, detail: Dict[str, Any]) -> str:
        expedition = detail.get("expedition") or {}
        return str(
            detail.get("title")
            or expedition.get("title")
            or detail.get("root_query")
            or expedition.get("root_query")
            or "Expedition"
        ).strip()

    def _get_summary_text(self, detail: Dict[str, Any], limit: int) -> str:
        summary_obj = (
            detail.get("summary")
            or detail.get("summary_json")
            or (detail.get("expedition") or {}).get("summary_json")
            or {}
        )
        if isinstance(summary_obj, dict):
            for key in ("summary", "text", "answer", "content"):
                if summary_obj.get(key):
                    return _clip(summary_obj.get(key), limit)
            return _clip(json.dumps(summary_obj, ensure_ascii=False), limit)
        return _clip(summary_obj, limit)

    def _get_overall_text(self, detail: Dict[str, Any], limit: int) -> str:
        overall_obj = (
            detail.get("overall")
            or detail.get("overall_json")
            or (detail.get("expedition") or {}).get("overall_json")
            or {}
        )
        if isinstance(overall_obj, dict):
            parts = []
            for key in ("verdict", "label", "confidence", "summary", "text"):
                if overall_obj.get(key) not in (None, ""):
                    parts.append(f"{key}={overall_obj.get(key)}")
            if parts:
                return _clip(" | ".join(parts), limit)
            return _clip(json.dumps(overall_obj, ensure_ascii=False), limit)
        return _clip(overall_obj, limit)

    def _normalize_sources(self, sources: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        cleaned = []
        for i, src in enumerate(sources or [], start=1):
            if not isinstance(src, dict):
                continue
            title = str(src.get("title") or src.get("source_title") or f"Source {i}").strip()
            url = str(src.get("url") or src.get("source_url") or "").strip()
            domain = str(src.get("domain") or "").strip()
            source_type = str(src.get("source_type") or "").strip()
            rank_score = _safe_float(src.get("rank_score", 0.0), 0.0)
            content = _clip(src.get("content") or src.get("snippet") or "", 220)

            cleaned.append(
                {
                    "label": f"S{i}",
                    "title": title,
                    "url": url,
                    "domain": domain,
                    "source_type": source_type,
                    "rank_score": rank_score,
                    "snippet": content,
                }
            )

        cleaned.sort(key=lambda x: x["rank_score"], reverse=True)
        return cleaned[:limit]

    def _normalize_claims(self, claims: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        cleaned = []
        for i, claim in enumerate(claims or [], start=1):
            if not isinstance(claim, dict):
                continue

            text = str(
                claim.get("claim")
                or claim.get("claim_text")
                or claim.get("text")
                or ""
            ).strip()
            if not text:
                continue

            confidence = _safe_float(
                (claim.get("scores") or {}).get("confidence", claim.get("confidence", 0.0)),
                0.0,
            )
            verdict = str(
                (claim.get("judge") or {}).get("verdict")
                or claim.get("verdict")
                or ""
            ).strip()
            label = str((claim.get("scores") or {}).get("label") or claim.get("label") or "").strip()

            cleaned.append(
                {
                    "label": f"C{i}",
                    "text": _clip(text, 260),
                    "verdict": verdict,
                    "label_text": label,
                    "confidence": confidence,
                }
            )

        cleaned.sort(key=lambda x: x["confidence"], reverse=True)
        return cleaned[:limit]

    def _normalize_memory(self, messages: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        cleaned = []
        for i, msg in enumerate((messages or [])[-limit:], start=1):
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "").strip()
            content = _clip(msg.get("content") or "", 220)
            if not role or not content:
                continue
            cleaned.append({"label": f"M{i}", "role": role, "content": content})
        return cleaned

    def _normalize_web_results(self, results: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        cleaned = []
        for i, item in enumerate(results or [], start=1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or item.get("name") or f"Web result {i}").strip()
            url = str(item.get("url") or item.get("link") or "").strip()
            domain = str(item.get("domain") or "").strip()
            snippet = _clip(item.get("content") or item.get("snippet") or item.get("raw_content") or "", 220)
            cleaned.append(
                {
                    "label": f"W{i}",
                    "title": title,
                    "url": url,
                    "domain": domain,
                    "snippet": snippet,
                }
            )
        return cleaned[:limit]

    def _search_web(self, question: str, root_query: str, title: str) -> List[Dict[str, Any]]:
        if not self.web_search:
            return []

        query = " ".join(
            part for part in [root_query, title, question] if part and str(part).strip()
        )
        query = _clip(query, 220)

        try:
            raw = self.web_search.invoke({"query": query})
        except Exception:
            try:
                raw = self.web_search.invoke(query)
            except Exception:
                return []

        if isinstance(raw, dict):
            items = raw.get("results") or raw.get("data") or []
        elif isinstance(raw, list):
            items = raw
        else:
            items = []

        return self._normalize_web_results(items, MAX_WEB_RESULTS)

    def _build_context_text(
        self,
        detail: Dict[str, Any],
        question: str,
        memory_rows: List[Dict[str, Any]],
        web_results: List[Dict[str, Any]],
        limits: Dict[str, int],
    ) -> Dict[str, Any]:
        root_query = self._get_root_query(detail)
        title = self._get_title(detail)
        summary_text = self._get_summary_text(detail, limits["summary_chars"])
        overall_text = self._get_overall_text(detail, 240)

        sources = self._normalize_sources(detail.get("sources", []) or [], limits["sources"])
        claims = self._normalize_claims(detail.get("claims", []) or [], limits["claims"])
        memory = self._normalize_memory(memory_rows, limits["memory"])
        web = self._normalize_web_results(web_results, limits["web"])

        lines: List[str] = []
        lines.append(f"EXPEDITION TITLE: {title}")
        lines.append(f"ROOT QUERY: {root_query}")
        if summary_text:
            lines.append(f"SUMMARY: {summary_text}")
        if overall_text:
            lines.append(f"OVERALL: {overall_text}")

        if claims:
            lines.append("CLAIMS:")
            for claim in claims:
                lines.append(
                    f"{claim['label']} | verdict={claim['verdict'] or '—'} | "
                    f"confidence={claim['confidence']:.3f} | {claim['text']}"
                )

        if sources:
            lines.append("TOP SOURCES:")
            for src in sources:
                lines.append(
                    f"{src['label']} | {src['title']} | "
                    f"domain={src['domain'] or '—'} | type={src['source_type'] or '—'} | "
                    f"rank={src['rank_score']:.3f} | {src['snippet']} | url={src['url'] or '—'}"
                )

        if memory:
            lines.append("RECENT MEMORY:")
            for msg in memory:
                lines.append(f"{msg['label']} | {msg['role'].upper()} | {msg['content']}")

        if web:
            lines.append("WEB SEARCH:")
            for item in web:
                lines.append(
                    f"{item['label']} | {item['title']} | domain={item['domain'] or '—'} | "
                    f"{item['snippet']} | url={item['url'] or '—'}"
                )

        context_text = "\n".join(lines)
        return {
            "context_text": context_text,
            "root_query": root_query,
            "title": title,
            "summary_text": summary_text,
            "overall_text": overall_text,
            "sources": sources,
            "claims": claims,
            "memory": memory,
            "web": web,
        }

    def _parse_model_output(self, text: str) -> Dict[str, Any]:
        parsed = _safe_parse_json(text)
        if parsed:
            return parsed

        # Fallback if the model returns plain text.
        text = _clip(text, 2500)
        return {
            "answer": text,
            "confidence": 0.35,
            "citations": [],
            "follow_up_questions": [],
        }

    def _is_retriable_llm_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(
            token in msg
            for token in [
                "413",
                "request too large",
                "rate limit",
                "tokens per day",
                "tokens per minute",
                "tpm",
                "tpd",
            ]
        )

    def _invoke_model(self, question: str, context_payload: Dict[str, Any]) -> Dict[str, Any]:
        context_text = context_payload["context_text"]

        system = SystemMessage(content=_SYSTEM_PROMPT)
        user = HumanMessage(
            content=(
                f"Question: {_clip(question, 500)}\n\n"
                f"Use these labels when citing sources in the answer:\n"
                f"- Saved sources: [S1], [S2], ...\n"
                f"- Claims: [C1], [C2], ...\n"
                f"- Memory: [M1], [M2], ...\n"
                f"- Web: [W1], [W2], ...\n\n"
                f"Context:\n{context_text}\n\n"
                "Return only the JSON object described in the system instructions."
            )
        )

        response = self.llm.invoke([system, user])
        raw = response.content if hasattr(response, "content") else str(response)
        return self._parse_model_output(raw)

    def _fallback_answer(
        self,
        question: str,
        context_payload: Dict[str, Any],
        error: Exception | None = None,
    ) -> Dict[str, Any]:
        root_query = context_payload.get("root_query") or "the expedition topic"
        summary_text = context_payload.get("summary_text") or ""
        claims = context_payload.get("claims") or []
        sources = context_payload.get("sources") or []
        web = context_payload.get("web") or []

        claim_bits = []
        for claim in claims[:3]:
            claim_bits.append(
                f"{claim['label']}: {claim['text']} ({claim['verdict'] or 'no verdict'}, confidence {claim['confidence']:.2f})"
            )

        source_bits = []
        for src in sources[:3]:
            source_bits.append(f"{src['label']}: {src['title']}")

        web_bits = []
        for item in web[:2]:
            web_bits.append(f"{item['label']}: {item['title']}")

        parts = [
            f"I could not generate a full model answer right now.",
            f"Based on the saved context for '{root_query}', the expedition summary says: {summary_text[:240]}",
        ]
        if claim_bits:
            parts.append("Relevant claims: " + " | ".join(claim_bits))
        if source_bits:
            parts.append("Top sources: " + " | ".join(source_bits))
        if web_bits:
            parts.append("Recent web results: " + " | ".join(web_bits))
        if error is not None:
            parts.append(f"(Fallback used because: {str(error)[:180]})")

        citations = []
        for src in sources[:3]:
            citations.append(
                {
                    "label": src["label"],
                    "title": src["title"],
                    "url": src["url"],
                    "source_type": "saved_source",
                    "reason": "Saved expedition source",
                }
            )
        for item in web[:2]:
            citations.append(
                {
                    "label": item["label"],
                    "title": item["title"],
                    "url": item["url"],
                    "source_type": "web",
                    "reason": "Supplemental web search result",
                }
            )

        return {
            "answer": " ".join(parts),
            "confidence": 0.25,
            "citations": citations[:5],
            "follow_up_questions": [
                f"What are the strongest findings about {root_query}?",
                f"Which claim in {root_query} is least certain?",
            ],
        }

    def ask(self, expedition_id: str, message: str) -> Dict[str, Any]:
        question = str(message or "").strip()
        if not question:
            raise ValueError("message required")

        detail = self.expedition_repo.get_expedition_detail(expedition_id)
        prior_memory = self.chat_repo.list_messages(expedition_id, limit=MAX_MEMORY_MESSAGES)
        web_results = self._search_web(
            question=question,
            root_query=self._get_root_query(detail),
            title=self._get_title(detail),
        )

        # Save the user's message first so the chat stays persistent even if the model fails.
        self.chat_repo.append_message(
            expedition_id=expedition_id,
            role="user",
            content=question,
            metadata={
                "kind": "user_question",
                "model": self.model_name,
            },
        )

        last_error: Exception | None = None
        answer_payload: Dict[str, Any] = {}

        for limits in PROMPT_BUDGETS:
            try:
                context_payload = self._build_context_text(
                    detail=detail,
                    question=question,
                    memory_rows=prior_memory,
                    web_results=web_results,
                    limits=limits,
                )

                answer_payload = self._invoke_model(question, context_payload)

                # Ensure the payload is well-formed.
                if "answer" not in answer_payload:
                    answer_payload["answer"] = ""
                if "confidence" not in answer_payload:
                    answer_payload["confidence"] = 0.35
                if not isinstance(answer_payload.get("citations"), list):
                    answer_payload["citations"] = []
                if not isinstance(answer_payload.get("follow_up_questions"), list):
                    answer_payload["follow_up_questions"] = []

                # Persist assistant response.
                self.chat_repo.append_message(
                    expedition_id=expedition_id,
                    role="assistant",
                    content=str(answer_payload["answer"]),
                    metadata={
                        "kind": "assistant_answer",
                        "model": self.model_name,
                        "confidence": float(answer_payload.get("confidence", 0.0) or 0.0),
                        "citations": answer_payload.get("citations", []),
                        "used_sources": [s["label"] for s in context_payload["sources"]],
                        "used_claims": [c["label"] for c in context_payload["claims"]],
                        "used_web_results": [w["label"] for w in context_payload["web"]],
                    },
                )

                return {
                    "answer": str(answer_payload["answer"]),
                    "confidence": float(answer_payload.get("confidence", 0.0) or 0.0),
                    "citations": answer_payload.get("citations", []),
                    "follow_up_questions": answer_payload.get("follow_up_questions", []),
                    "web_results": web_results,
                    "memory": prior_memory,
                    "expedition": detail.get("expedition", {}),
                    "model": self.model_name,
                }

            except Exception as exc:
                last_error = exc
                if not self._is_retriable_llm_error(exc):
                    break

        # Fallback, so the UI still gets a useful answer even if Groq is temporarily limited.
        fallback_context = self._build_context_text(
            detail=detail,
            question=question,
            memory_rows=prior_memory,
            web_results=web_results,
            limits=PROMPT_BUDGETS[-1],
        )
        fallback_payload = self._fallback_answer(question, fallback_context, last_error)

        self.chat_repo.append_message(
            expedition_id=expedition_id,
            role="assistant",
            content=str(fallback_payload["answer"]),
            metadata={
                "kind": "assistant_answer_fallback",
                "model": self.model_name,
                "confidence": float(fallback_payload.get("confidence", 0.0) or 0.0),
                "citations": fallback_payload.get("citations", []),
                "error": str(last_error)[:300] if last_error else "",
            },
        )

        return {
            "answer": str(fallback_payload["answer"]),
            "confidence": float(fallback_payload.get("confidence", 0.0) or 0.0),
            "citations": fallback_payload.get("citations", []),
            "follow_up_questions": fallback_payload.get("follow_up_questions", []),
            "web_results": web_results,
            "memory": prior_memory,
            "expedition": detail.get("expedition", {}),
            "model": self.model_name,
            "fallback": True,
            "error": str(last_error) if last_error else "",
        }