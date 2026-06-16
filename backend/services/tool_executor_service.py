from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.documents import Document

from backend.services.tools.tool_registry import ToolRegistry
from backend.services.tools.normalizer import normalize_source


class ToolExecutorService:
    def __init__(self, registry: ToolRegistry | None = None):
        self.registry = registry or ToolRegistry()

    def _normalize_args(self, args: Any, default_query: str = "") -> str:
        if args is None:
            args = {}

        if isinstance(args, dict):
            q = str(args.get("query") or args.get("q") or default_query or "").strip()
            return q

        if isinstance(args, str):
            return args.strip() or default_query

        return default_query

    def _invoke_tool(self, tool: Any, tool_name: str, query_text: str):
        candidates: List[Any] = []
        if query_text:
            candidates.append(query_text)
            candidates.append({"query": query_text})
        candidates.append({})

        for candidate in candidates:
            try:
                if hasattr(tool, "invoke"):
                    return tool.invoke(candidate)
            except Exception:
                pass

            try:
                if hasattr(tool, "run"):
                    return tool.run(candidate)
            except Exception:
                pass

            try:
                if callable(tool):
                    return tool(candidate)
            except Exception:
                pass

        raise TypeError(f"Tool could not be invoked: {tool_name}")

    def _document_to_source(self, doc: Document, tool_name: str, tool_args: Any) -> Dict[str, Any]:
        md = doc.metadata or {}
        return normalize_source(
            title=str(md.get("title") or md.get("source") or tool_name),
            content=doc.page_content or "",
            url=str(md.get("source") or md.get("url") or ""),
            source_type=tool_name,
            domain=str(md.get("domain", "")),
            authors=md.get("authors", []),
            published_at=str(md.get("published_at", "")),
            extra={"tool_name": tool_name, "tool_args": tool_args, "raw_metadata": md},
        )

    def _dict_to_source(self, item: Dict[str, Any], tool_name: str, tool_args: Any) -> Dict[str, Any]:
        title = item.get("title") or item.get("name") or tool_name
        content = (
            item.get("content")
            or item.get("raw_content")
            or item.get("snippet")
            or item.get("summary")
            or ""
        )
        url = item.get("url") or item.get("link") or item.get("source") or ""

        return normalize_source(
            title=title,
            content=content,
            url=url,
            source_type=tool_name,
            domain=item.get("domain", ""),
            authors=item.get("authors", []),
            published_at=item.get("published_at", ""),
            extra={"tool_name": tool_name, "tool_args": tool_args, "raw_item": item},
        )

    def _to_source_items(self, raw_output: Any, tool_name: str, tool_args: Any) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []

        if raw_output is None:
            return items

        if isinstance(raw_output, Document):
            items.append(self._document_to_source(raw_output, tool_name, tool_args))
            return items

        if isinstance(raw_output, list):
            for entry in raw_output:
                if isinstance(entry, Document):
                    items.append(self._document_to_source(entry, tool_name, tool_args))
                elif isinstance(entry, dict):
                    items.append(self._dict_to_source(entry, tool_name, tool_args))
                else:
                    text = str(entry).strip()
                    if text:
                        items.append(
                            normalize_source(
                                title=tool_name,
                                content=text,
                                url="",
                                source_type=tool_name,
                                extra={"tool_name": tool_name, "tool_args": tool_args},
                            )
                        )
            return items

        if isinstance(raw_output, dict):
            if isinstance(raw_output.get("results"), list):
                for entry in raw_output["results"]:
                    if isinstance(entry, dict):
                        items.append(self._dict_to_source(entry, tool_name, tool_args))
                    else:
                        text = str(entry).strip()
                        if text:
                            items.append(
                                normalize_source(
                                    title=tool_name,
                                    content=text,
                                    url="",
                                    source_type=tool_name,
                                    extra={"tool_name": tool_name, "tool_args": tool_args},
                                )
                            )
                return items

            items.append(self._dict_to_source(raw_output, tool_name, tool_args))
            return items

        text = str(raw_output).strip()
        if text:
            items.append(
                normalize_source(
                    title=tool_name,
                    content=text,
                    url="",
                    source_type=tool_name,
                    extra={"tool_name": tool_name, "tool_args": tool_args},
                )
            )

        return items

    def execute_plan(self, plan: Dict[str, Any], default_query: str = "") -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        for call in plan.get("tool_calls", []):
            if not isinstance(call, dict):
                continue

            tool_name = str(call.get("tool", "")).strip()
            query_text = self._normalize_args(call.get("args", {}), default_query=default_query)

            tool = self.registry.get(tool_name)
            if tool is None:
                continue

            try:
                raw_output = self._invoke_tool(tool, tool_name, query_text)
            except Exception as exc:
                print(f"\nTOOL FAILED: {tool_name}")
                print(exc)
                continue

            results.extend(self._to_source_items(raw_output, tool_name, {"query": query_text}))

        return results