from __future__ import annotations

from typing import Any, Dict, Optional
from urllib.parse import urlparse


def normalize_source(
    *,
    title: str,
    content: str,
    url: str,
    source_type: str,
    domain: Optional[str] = None,
    authors: Optional[list[str]] = None,
    published_at: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    parsed_domain = domain
    if not parsed_domain and url:
        try:
            parsed_domain = urlparse(url).netloc
        except Exception:
            parsed_domain = ""

    return {
        "title": title.strip() if title else "",
        "content": content.strip() if content else "",
        "url": url.strip() if url else "",
        "domain": parsed_domain or "",
        "source_type": source_type,
        "authors": authors or [],
        "published_at": published_at or "",
        "extra": extra or {},
    }