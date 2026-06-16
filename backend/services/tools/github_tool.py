from __future__ import annotations

import base64
import os
from typing import Any, Dict, List

import requests
from langchain.tools import tool

from .base_tool import BaseTool
from .normalizer import normalize_source


class GitHubTool(BaseTool):
    SEARCH_URL = "https://api.github.com/search/repositories"
    CONTENT_URL = "https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    REPO_URL = "https://api.github.com/repos/{owner}/{repo}"

    def __init__(self, token: str | None = None, timeout: int = 20):
        self.token = token or os.getenv("GITHUB_TOKEN", "")
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def search(self, query: str, max_results: int = 5, fetch_readme: bool = True) -> List[Dict[str, Any]]:
        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": min(max_results, 10),
        }

        try:
            resp = requests.get(
                self.SEARCH_URL,
                params=params,
                headers=self._headers(),
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return []

        results: List[Dict[str, Any]] = []

        for repo in data.get("items", [])[:max_results]:
            owner = (repo.get("owner") or {}).get("login", "")
            repo_name = repo.get("name", "")
            full_name = repo.get("full_name", "")
            html_url = repo.get("html_url", "")
            description = repo.get("description") or ""
            language = repo.get("language") or ""
            stars = repo.get("stargazers_count", 0)
            updated_at = repo.get("updated_at", "")

            readme = self.fetch_readme(owner, repo_name) if fetch_readme and owner and repo_name else ""

            results.append(
                normalize_source(
                    title=full_name or repo_name,
                    content=readme or description,
                    url=html_url,
                    source_type="github",
                    domain="github.com",
                    authors=[owner] if owner else [],
                    published_at=updated_at,
                    extra={
                        "kind": "github_repository",
                        "language": language,
                        "stars": stars,
                        "repo_full_name": full_name,
                        "description": description,
                    },
                )
            )

        return results

    def fetch_readme(self, owner: str, repo: str, path: str = "README.md") -> str:
        url = self.CONTENT_URL.format(owner=owner, repo=repo, path=path)

        try:
            resp = requests.get(url, headers=self._headers(), timeout=self.timeout)
            if resp.status_code == 404:
                return ""
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return ""

        content = data.get("content", "")
        encoding = data.get("encoding", "")

        if content and encoding == "base64":
            try:
                return base64.b64decode(content).decode("utf-8", errors="ignore")
            except Exception:
                return ""
        return ""

    def fetch_repository_by_url(self, url: str) -> Dict[str, Any] | None:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if "github.com" not in (parsed.netloc or "").lower():
            return None

        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) < 2:
            return None

        owner, repo = parts[0], parts[1].removesuffix(".git")

        try:
            resp = requests.get(
                self.REPO_URL.format(owner=owner, repo=repo),
                headers=self._headers(),
                timeout=self.timeout,
            )
            resp.raise_for_status()
            repo_data = resp.json()
        except Exception:
            return None

        readme = self.fetch_readme(owner, repo)

        return normalize_source(
            title=repo_data.get("full_name") or repo,
            content=readme or (repo_data.get("description") or ""),
            url=repo_data.get("html_url", url),
            source_type="github",
            domain="github.com",
            authors=[owner] if owner else [],
            published_at=repo_data.get("updated_at", ""),
            extra={
                "kind": "github_repository",
                "language": repo_data.get("language", ""),
                "stars": repo_data.get("stargazers_count", 0),
                "repo_full_name": repo_data.get("full_name", ""),
                "description": repo_data.get("description", ""),
                "fetched_from_url": True,
            },
        )


@tool
def github_search(query: str) -> str:
    """
    Search GitHub repositories and return compact text.
    """
    tool = GitHubTool()
    results = tool.search(query, max_results=5, fetch_readme=False)
    if not results:
        return "No GitHub results found."

    lines = []
    for r in results:
        lines.append(f"{r['title']} | {r['url']} | {r.get('content', '')[:200]}")
    return "\n".join(lines)