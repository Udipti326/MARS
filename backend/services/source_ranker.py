from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List


TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(TOKEN_RE.findall((text or "").lower()))


def _recency_score(published_at: str) -> float:
    if not published_at:
        return 0.6
    try:
        text = published_at.replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        year = dt.year
        current_year = datetime.utcnow().year

        if year >= current_year:
            return 1.0
        if year >= current_year - 1:
            return 0.9
        if year >= current_year - 3:
            return 0.8
        if year >= current_year - 5:
            return 0.7
        return 0.6
    except Exception:
        return 0.6


class SourceRanker:
    TYPE_WEIGHTS = {
        "arxiv": 1.0,
        "github": 0.93,
        "web": 0.82,
        "linked_web": 0.80,
        "youtube": 0.72,
    }

    @classmethod
    def score(cls, query: str, source: Dict[str, Any]) -> float:
        q = _tokenize(query)
        text = " ".join(
            [
                str(source.get("title", "")),
                str(source.get("content", "")),
                str(source.get("domain", "")),
            ]
        )
        s = _tokenize(text)

        overlap = len(q & s) / max(len(q), 1)
        type_weight = cls.TYPE_WEIGHTS.get(source.get("source_type", ""), 0.75)
        recency = _recency_score(str(source.get("published_at", "")))

        depth = 0
        try:
            depth = int((source.get("extra") or {}).get("link_depth", 0) or 0)
        except Exception:
            depth = 0

        depth_penalty = min(0.12, depth * 0.03)

        score = (0.45 * overlap) + (0.30 * type_weight) + (0.25 * recency) - depth_penalty
        return round(max(0.0, min(1.0, score)), 3)

    @classmethod
    def rank(cls, query: str, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ranked = []
        for src in sources:
            item = dict(src)
            item["rank_score"] = cls.score(query, item)
            ranked.append(item)

        return sorted(ranked, key=lambda x: x.get("rank_score", 0.0), reverse=True)