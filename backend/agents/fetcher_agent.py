from __future__ import annotations

from typing import Any, Dict, List

from backend.agents.orchestrator_agent import OrchestratorAgent
from backend.services.source_ranker import SourceRanker
from backend.services.tool_executor_service import ToolExecutorService


class FetcherAgent:
    def __init__(self):
        self.orchestrator = OrchestratorAgent()
        self.executor = ToolExecutorService()

    def run(self, query: str, max_total_sources: int = 30) -> List[Dict[str, Any]]:
        plan = self.orchestrator.run(query)
        if not isinstance(plan, dict):
            plan = {}

        raw_sources = self.executor.execute_plan(plan, default_query=query)

        if not raw_sources:
            fallback_plan = {
                "tool_calls": [
                    {"tool": "tavily_search", "args": {"query": query}, "reason": "Fallback web search"},
                    {"tool": "wikipedia", "args": {"query": query}, "reason": "Fallback encyclopedic search"},
                    {"tool": "arxiv", "args": {"query": query}, "reason": "Fallback paper search"},
                    {"tool": "github_search", "args": {"query": query}, "reason": "Fallback implementation search"},
                ]
            }
            raw_sources = self.executor.execute_plan(fallback_plan, default_query=query)

        ranked = SourceRanker.rank(query, raw_sources)

        cleaned: List[Dict[str, Any]] = []
        seen = set()

        for src in ranked:
            url = str(src.get("url", "") or "").strip()
            title = str(src.get("title", "") or "").strip()
            content = str(src.get("content", "") or "").strip()

            if not content:
                continue

            # Prefer clickable, canonical links.
            if not url:
                continue

            key = url.lower()
            if key in seen:
                continue
            seen.add(key)

            extra = src.get("extra") or {}
            extra["root_query"] = query
            src["extra"] = extra

            cleaned.append(src)

            if len(cleaned) >= max_total_sources:
                break

        return cleaned