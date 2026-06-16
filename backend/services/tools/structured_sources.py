from __future__ import annotations

import os
import re
from typing import Any, Dict, List
from urllib.parse import quote

import arxiv
import wikipedia
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from backend.services.tools.github_tool import GitHubTool
from backend.services.tools.normalizer import normalize_source

URL_RE = re.compile(r"https?://[^\s\]\)\"'>]+", re.IGNORECASE)


def _first_url(text: str) -> str:
    if not text:
        return ""
    match = URL_RE.search(text)
    if not match:
        return ""
    return match.group(0).rstrip(".,);]>")


def _doc_to_source(doc: Document, tool_name: str, fallback_title: str = "") -> Dict[str, Any]:
    md = doc.metadata or {}
    title = str(md.get("title") or md.get("source") or fallback_title or tool_name).strip()
    url = str(md.get("source") or md.get("url") or md.get("entry_id") or "").strip()
    if not url:
        url = _first_url(doc.page_content or "")

    if tool_name == "wikipedia" and not url and title:
        url = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"

    if tool_name == "arxiv" and not url and title:
        arxiv_id = str(md.get("arxiv_id") or md.get("id") or "").strip()
        if arxiv_id:
            url = f"https://arxiv.org/abs/{arxiv_id}"

    return normalize_source(
        title=title,
        content=doc.page_content or "",
        url=url,
        source_type=tool_name,
        domain=str(md.get("domain", "") or ""),
        authors=md.get("authors", []) if isinstance(md.get("authors", []), list) else [],
        published_at=str(md.get("published_at", "") or ""),
        extra={"raw_metadata": md},
    )


def _dict_to_source(item: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
    title = str(item.get("title") or item.get("name") or tool_name).strip()
    content = str(
        item.get("content")
        or item.get("raw_content")
        or item.get("snippet")
        or item.get("summary")
        or ""
    ).strip()
    url = str(item.get("url") or item.get("link") or item.get("source") or "").strip()
    if not url:
        url = _first_url(content)

    return normalize_source(
        title=title,
        content=content,
        url=url,
        source_type=tool_name,
        domain=str(item.get("domain", "") or ""),
        authors=item.get("authors", []) if isinstance(item.get("authors", []), list) else [],
        published_at=str(item.get("published_at", "") or ""),
        extra={"raw_item": item},
    )


def _coerce_results(raw_output: Any, tool_name: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    if raw_output is None:
        return results

    if isinstance(raw_output, Document):
        return [_doc_to_source(raw_output, tool_name)]

    if isinstance(raw_output, list):
        for item in raw_output:
            if isinstance(item, Document):
                results.append(_doc_to_source(item, tool_name))
            elif isinstance(item, dict):
                results.append(_dict_to_source(item, tool_name))
            else:
                text = str(item).strip()
                if text:
                    results.append(
                        normalize_source(
                            title=tool_name,
                            content=text,
                            url=_first_url(text),
                            source_type=tool_name,
                        )
                    )
        return results

    if isinstance(raw_output, dict):
        if isinstance(raw_output.get("results"), list):
            for item in raw_output["results"]:
                if isinstance(item, dict):
                    results.append(_dict_to_source(item, tool_name))
                else:
                    text = str(item).strip()
                    if text:
                        results.append(
                            normalize_source(
                                title=tool_name,
                                content=text,
                                url=_first_url(text),
                                source_type=tool_name,
                            )
                        )
            return results

        results.append(_dict_to_source(raw_output, tool_name))
        return results

    text = str(raw_output).strip()
    if text:
        results.append(
            normalize_source(
                title=tool_name,
                content=text,
                url=_first_url(text),
                source_type=tool_name,
            )
        )

    return results


def wikipedia_search(query: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        titles = wikipedia.search(query, results=3)
    except Exception:
        titles = []

    for title in titles[:3]:
        try:
            page = wikipedia.page(title, auto_suggest=False, preload=False)
            summary = wikipedia.summary(title, sentences=4, auto_suggest=False, redirect=True)
            out.append(
                normalize_source(
                    title=page.title,
                    content=summary or getattr(page, "content", "")[:1600],
                    url=page.url or f"https://en.wikipedia.org/wiki/{quote(page.title.replace(' ', '_'))}",
                    source_type="wikipedia",
                    domain="en.wikipedia.org",
                    extra={
                        "pageid": getattr(page, "pageid", None),
                        "tool_name": "wikipedia",
                        "query": query,
                    },
                )
            )
        except Exception:
            continue

    return out


def arxiv_search(query: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    try:
        search = arxiv.Search(
            query=query,
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        client = arxiv.Client(page_size=5, num_retries=2, delay_seconds=1)

        for result in client.results(search):
            authors = []
            try:
                authors = [a.name for a in result.authors]
            except Exception:
                authors = []

            published_at = ""
            try:
                published_at = result.published.isoformat()
            except Exception:
                published_at = ""

            out.append(
                normalize_source(
                    title=result.title,
                    content=result.summary or "",
                    url=str(result.entry_id or result.pdf_url or "").strip(),
                    source_type="arxiv",
                    domain="arxiv.org",
                    authors=authors,
                    published_at=published_at,
                    extra={
                        "arxiv_id": getattr(result, "get_short_id", lambda: "")(),
                        "pdf_url": str(result.pdf_url or ""),
                        "tool_name": "arxiv",
                        "query": query,
                    },
                )
            )
    except Exception:
        return []

    return out


def tavily_search(query: str) -> List[Dict[str, Any]]:
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return []

    tool = TavilySearch(
        max_results=5,
        include_raw_content=True,
        tavily_api_key=api_key,
    )

    raw = None
    for candidate in ({"query": query}, query):
        try:
            raw = tool.invoke(candidate)
            break
        except Exception:
            continue

    return _coerce_results(raw, "tavily_search")


def github_search(query: str) -> List[Dict[str, Any]]:
    try:
        results = GitHubTool().search(query, max_results=5, fetch_readme=False)
    except Exception:
        return []

    normalized: List[Dict[str, Any]] = []
    for item in results:
        if isinstance(item, dict):
            normalized.append(_dict_to_source(item, "github_search"))
        else:
            text = str(item).strip()
            if text:
                normalized.append(
                    normalize_source(
                        title="github_search",
                        content=text,
                        url=_first_url(text),
                        source_type="github_search",
                    )
                )
    return normalized