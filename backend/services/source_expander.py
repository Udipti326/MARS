from __future__ import annotations

import re
from collections import deque
from typing import Any, Dict, List
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

from backend.services.tools.github_tool import GitHubTool
from backend.services.tools.youtube_tool import YouTubeTool
from backend.services.tools.normalizer import normalize_source
from backend.utils.text_cleaning import clean_text

URL_RE = re.compile(r"https?://[^\s\]\)\"'>]+", re.IGNORECASE)


def normalize_url(url: str) -> str:
    if not url:
        return ""
    try:
        parsed = urlparse(url.strip())
        parsed = parsed._replace(fragment="")
        return urlunparse(parsed).rstrip("/")
    except Exception:
        return url.strip().rstrip("/")


def source_key(item: Dict[str, Any]) -> str:
    url = normalize_url(str(item.get("url", "")))
    if url:
        return f"url::{url}"
    title = clean_text(str(item.get("title", "")), 120).lower()
    content = clean_text(str(item.get("content", "")), 200).lower()
    source_type = str(item.get("source_type", "")).lower()
    return f"no_url::{source_type}::{title}::{content}"


class SourceExpander:
    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.youtube = YouTubeTool()
        self.github = GitHubTool()

    def extract_urls_from_text(self, text: str) -> List[str]:
        if not text:
            return []
        urls = [normalize_url(u) for u in URL_RE.findall(text)]
        seen = set()
        out = []
        for url in urls:
            if not url or url in seen:
                continue
            seen.add(url)
            out.append(url)
        return out

    def _extract_links_from_html(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        links: List[str] = []
        for a in soup.find_all("a", href=True):
            href = a.get("href", "").strip()
            if not href or href.startswith(("mailto:", "javascript:", "#")):
                continue
            abs_url = normalize_url(urljoin(base_url, href))
            if abs_url.startswith(("http://", "https://")):
                links.append(abs_url)
        return list(dict.fromkeys(links))

    def _fetch_web_preview(self, url: str) -> Dict[str, Any] | None:
        try:
            resp = requests.get(
                url,
                timeout=self.timeout,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()
        except Exception:
            return None

        content_type = (resp.headers.get("Content-Type") or "").lower()

        if "html" not in content_type and "<html" not in resp.text.lower():
            return normalize_source(
                title=url,
                content=clean_text(resp.text, 1800),
                url=url,
                source_type="linked_web",
                domain=urlparse(url).netloc,
                extra={"kind": "linked_non_html", "found_links": []},
            )

        try:
            soup = BeautifulSoup(resp.text, "html.parser")
        except Exception:
            return None

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        title = soup.title.text.strip() if soup.title and soup.title.text else url

        paragraphs = []
        for node in soup.find_all(["p", "li", "h1", "h2", "h3", "article"]):
            txt = node.get_text(" ", strip=True)
            if txt:
                paragraphs.append(txt)

        content = clean_text(" ".join(paragraphs), 2200)
        found_links = self._extract_links_from_html(soup, url)

        return normalize_source(
            title=title,
            content=content,
            url=url,
            source_type="linked_web",
            domain=urlparse(url).netloc,
            extra={"kind": "linked_page", "found_links": found_links},
        )

    def _materialize_source(self, src: Dict[str, Any], depth: int, discovered_from: str) -> Dict[str, Any] | None:
        current_url = normalize_url(src.get("url", ""))
        source_type = src.get("source_type", "web")

        # Preserve sources even when they have no URL.
        if not current_url:
            title = str(src.get("title", "")).strip()
            content = str(src.get("content", "")).strip()
            if not title and not content:
                return None

            return normalize_source(
                title=title or source_type,
                content=content,
                url="",
                source_type=source_type,
                domain=str(src.get("domain", "")),
                authors=src.get("authors", []) if isinstance(src.get("authors", []), list) else [],
                published_at=str(src.get("published_at", "")),
                extra={**(src.get("extra") or {}), "discovered_from": discovered_from, "link_depth": depth},
            )

        parsed_domain = urlparse(current_url).netloc.lower()

        if source_type == "youtube":
            yt = self.youtube.fetch_by_url(current_url)
            if yt:
                yt["extra"] = {**(yt.get("extra") or {}), "discovered_from": discovered_from, "link_depth": depth}
                return yt

        if source_type == "github":
            gh = self.github.fetch_repository_by_url(current_url)
            if gh:
                gh["extra"] = {**(gh.get("extra") or {}), "discovered_from": discovered_from, "link_depth": depth}
                return gh

        if source_type == "arxiv" or "arxiv.org" in parsed_domain:
            enriched = normalize_source(
                title=src.get("title", current_url),
                content=src.get("content", ""),
                url=current_url,
                source_type="arxiv",
                domain="arxiv.org",
                authors=src.get("authors", []),
                published_at=src.get("published_at", ""),
                extra={**(src.get("extra") or {}), "discovered_from": discovered_from, "link_depth": depth},
            )
            if not enriched.get("content"):
                preview = self._fetch_web_preview(current_url)
                if preview:
                    preview["extra"] = {**(preview.get("extra") or {}), "discovered_from": discovered_from, "link_depth": depth}
                    return preview
            return enriched

        preview = self._fetch_web_preview(current_url)
        if preview:
            preview["extra"] = {**(preview.get("extra") or {}), "discovered_from": discovered_from, "link_depth": depth}
            return preview

        return normalize_source(
            title=src.get("title", current_url),
            content=src.get("content", ""),
            url=current_url,
            source_type=source_type,
            domain=src.get("domain", parsed_domain),
            authors=src.get("authors", []),
            published_at=src.get("published_at", ""),
            extra={**(src.get("extra") or {}), "discovered_from": discovered_from, "link_depth": depth},
        )

    def expand_sources(
        self,
        sources: List[Dict[str, Any]],
        max_depth: int = 1,
        max_total_sources: int = 30,
        max_links_per_source: int = 5,
    ) -> List[Dict[str, Any]]:
        queue = deque()
        for src in sources:
            queue.append((src, 0, "root"))

        visited = set()
        expanded: List[Dict[str, Any]] = []

        while queue and len(expanded) < max_total_sources:
            src, depth, discovered_from = queue.popleft()
            current_url = normalize_url(src.get("url", ""))

            key = source_key(src)
            if key in visited:
                continue
            visited.add(key)

            materialized = self._materialize_source(src, depth, discovered_from)
            if not materialized:
                continue

            expanded.append(materialized)

            if depth >= max_depth:
                continue

            extra = materialized.get("extra") or {}
            extra_links = extra.get("found_links", [])
            if not isinstance(extra_links, list):
                extra_links = []

            content_links = self.extract_urls_from_text(materialized.get("content", ""))
            candidate_links = list(dict.fromkeys(extra_links + content_links))

            for link in candidate_links[:max_links_per_source]:
                nlink = normalize_url(link)
                if not nlink or nlink in visited:
                    continue

                queue.append(
                    (
                        {
                            "title": nlink,
                            "content": "",
                            "url": nlink,
                            "source_type": "linked_web",
                            "domain": urlparse(nlink).netloc,
                            "authors": [],
                            "published_at": "",
                            "extra": {
                                "discovered_from": current_url,
                                "root_url": src.get("url", ""),
                                "root_query": (src.get("extra") or {}).get("root_query", ""),
                            },
                        },
                        depth + 1,
                        current_url,
                    )
                )

        unique: List[Dict[str, Any]] = []
        seen = set()
        for item in expanded:
            key = source_key(item)
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)

        return unique