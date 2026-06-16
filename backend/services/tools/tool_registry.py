from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List

from dotenv import load_dotenv

from backend.services.tools.structured_sources import (
    arxiv_search,
    github_search,
    tavily_search,
    wikipedia_search,
)


def load_project_env() -> str | None:
    candidates = [
        Path.cwd() / "backend" / ".env",
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
        Path(__file__).resolve().parents[3] / ".env",
    ]

    for path in candidates:
        if path.exists():
            load_dotenv(path, override=True)
            return str(path)

    return None


loaded_from = load_project_env()
print(f"[ToolRegistry] .env loaded from: {loaded_from or 'NOT FOUND'}")
print(f"[ToolRegistry] TAVILY_API_KEY present: {bool(os.getenv('TAVILY_API_KEY'))}")


class ToolRegistry:
    def __init__(self):
        self._tools = {}
        self._register_builtin_tools()

    def _register_builtin_tools(self):
        self.register("wikipedia", wikipedia_search)
        self.register("arxiv", arxiv_search)
        self.register("tavily_search", tavily_search)
        self.register("github_search", github_search)

    def register(self, name: str, tool: Any):
        self._tools[name] = tool

    def get(self, name: str):
        return self._tools.get(name)

    def get_tools(self) -> List[Any]:
        return list(self._tools.values())

    @property
    def tools(self):
        return self._tools

    def catalog_text(self) -> str:
        lines = []
        for name in self._tools.keys():
            desc = ""
            if name == "wikipedia":
                desc = "Structured Wikipedia search returning title, content, and url."
            elif name == "arxiv":
                desc = "Structured arXiv search returning title, content, and url."
            elif name == "tavily_search":
                desc = "Structured Tavily web search returning title, content, and url."
            elif name == "github_search":
                desc = "Structured GitHub repository search returning title, content, and url."
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)