from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from backend.core.llm import get_llm
from backend.services.tools.tool_registry import ToolRegistry


# ---------------------------------------------------
# Structured Output Models
# ---------------------------------------------------

class ToolCall(BaseModel):
    tool: str
    args: Dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class OrchestratorPlan(BaseModel):
    tool_calls: List[ToolCall] = Field(default_factory=list)
    max_link_depth: int = 1
    notes: str = ""


# ---------------------------------------------------
# Orchestrator Agent
# ---------------------------------------------------

class OrchestratorAgent:

    def __init__(self, temperature: float = 0.0):

        self.llm = get_llm(temperature=temperature)

        self.registry = ToolRegistry()

        self.allowed_tools = set(
            self.registry.tools.keys()
        )

        # ✅ STRUCTURED OUTPUT
        self.structured_llm = self.llm.with_structured_output(
            OrchestratorPlan
        )

    # ---------------------------------------------------
    # Fallback
    # ---------------------------------------------------

    def _fallback_plan(self, query: str):

        tool_calls = []

        preferred = [
            "tavily_search",
            "wikipedia",
            "arxiv",
            "github_search",
        ]

        for tool_name in preferred:

            if tool_name in self.allowed_tools:

                tool_calls.append(
                    {
                        "tool": tool_name,
                        "args": {
                            "query": query
                        },
                        "reason": "Fallback retrieval tool."
                    }
                )

        return {
            "tool_calls": tool_calls,
            "max_link_depth": 1,
            "notes": "Fallback plan used.",
            "query": query,
        }

    # ---------------------------------------------------
    # Main
    # ---------------------------------------------------

    def run(self, query: str):

        catalog = self.registry.catalog_text()

        prompt = f"""
You are a research orchestrator.

Your job:
- choose the BEST tools
- prioritize:
    1. tavily_search
    2. wikipedia
    3. arxiv
    4. github_search
- use NIA only as supplementary

IMPORTANT:
- prefer multiple tools
- prefer 2-4 tools
- do NOT answer the query
- ONLY create a retrieval plan

Available tools:
{catalog}

User query:
{query}
"""

        try:

            result = self.structured_llm.invoke(prompt)

            if isinstance(result, OrchestratorPlan):

                plan = result.model_dump()

            else:
                plan = {}

        except Exception as exc:

            print("\n⚠️ ORCHESTRATOR FAILED:\n")
            print(exc)

            return self._fallback_plan(query)

        # ---------------------------------------------------
        # Validate tool calls
        # ---------------------------------------------------

        normalized_calls = []

        for item in plan.get("tool_calls", []):

            if isinstance(item, ToolCall):
                item = item.model_dump()

            if not isinstance(item, dict):
                continue

            tool_name = str(
                item.get("tool", "")
            ).strip()

            if tool_name not in self.allowed_tools:
                continue

            args = item.get("args", {})

            if not isinstance(args, dict):
                args = {
                    "query": query
                }

            normalized_calls.append(
                {
                    "tool": tool_name,
                    "args": args,
                    "reason": str(
                        item.get("reason", "")
                    ).strip(),
                }
            )

        # ---------------------------------------------------
        # Force core retrieval tools
        # ---------------------------------------------------

        existing_tools = {
            item["tool"]
            for item in normalized_calls
        }

        core_tools = [
            "tavily_search",
            "wikipedia",
        ]

        for tool_name in core_tools:

            if (
                tool_name in self.allowed_tools
                and tool_name not in existing_tools
            ):

                normalized_calls.append(
                    {
                        "tool": tool_name,
                        "args": {
                            "query": query
                        },
                        "reason": "Core retrieval tool automatically added."
                    }
                )

        technical_keywords = [
            "attention",
            "transformer",
            "machine learning",
            "deep learning",
            "ai",
            "neural",
            "nlp",
        ]

        if any(
            k in query.lower()
            for k in technical_keywords
        ):

            if (
                "arxiv" in self.allowed_tools
                and "arxiv" not in existing_tools
            ):

                normalized_calls.append(
                    {
                        "tool": "arxiv",
                        "args": {
                            "query": query
                        },
                        "reason": "Scientific search automatically added."
                    }
                )

        # ---------------------------------------------------
        # Final fallback
        # ---------------------------------------------------

        if not normalized_calls:

            return self._fallback_plan(query)

        return {
            "tool_calls": normalized_calls,
            "max_link_depth": 1,
            "notes": str(
                plan.get("notes", "")
            ),
            "query": query,
        }