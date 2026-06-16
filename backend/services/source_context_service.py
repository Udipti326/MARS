from __future__ import annotations

from typing import Any, Dict, List

from backend.utils.text_cleaning import clean_text


def _safe_rank(source: Dict[str, Any]) -> float:
    try:
        return float(source.get("rank_score", 0.0) or 0.0)
    except Exception:
        return 0.0


class SourceContextService:
    @staticmethod
    def prepare_sources(
        sources: List[Dict[str, Any]],
        max_sources: int = 8,
    ) -> List[Dict[str, Any]]:
        valid: List[Dict[str, Any]] = []

        for src in sources or []:
            if not isinstance(src, dict):
                continue

            title = str(src.get("title", "") or "").strip()
            content = str(src.get("content", "") or "").strip()
            url = str(src.get("url", "") or "").strip()
            source_type = str(src.get("source_type", "") or "").strip()

            if not title and not content and not url:
                continue

            item = dict(src)
            item["title"] = title
            item["content"] = content
            item["url"] = url
            item["source_type"] = source_type
            item["rank_score"] = _safe_rank(item)
            valid.append(item)

        valid.sort(key=lambda x: x.get("rank_score", 0.0), reverse=True)
        return valid[:max_sources]

    @staticmethod
    def build_source_bundle(
        sources: List[Dict[str, Any]],
        max_sources: int = 8,
        max_chars_per_source: int = 900,
    ) -> str:
        prepared = SourceContextService.prepare_sources(sources, max_sources=max_sources)

        blocks: List[str] = []

        for i, src in enumerate(prepared, start=1):
            extra = src.get("extra") or {}
            source_ref = f"S{i}"

            title = clean_text(src.get("title", ""), 180)
            content = clean_text(src.get("content", ""), max_chars_per_source)
            url = src.get("url", "") or "n/a"
            source_type = src.get("source_type", "") or "unknown"
            domain = src.get("domain", "") or "n/a"
            rank_score = src.get("rank_score", 0.0)

            block = [
                f"[{source_ref}] {title}",
                f"CITATION: [{source_ref}]",
                f"URL: {url}",
                f"TYPE: {source_type}",
                f"DOMAIN: {domain}",
                f"RANK: {rank_score:.3f}",
            ]

            if src.get("authors"):
                if isinstance(src["authors"], list):
                    block.append(f"AUTHORS: {', '.join(map(str, src['authors']))}")

            if src.get("published_at"):
                block.append(f"PUBLISHED: {src['published_at']}")

            if extra.get("discovered_from"):
                block.append(f"DISCOVERED_FROM: {extra['discovered_from']}")

            if extra.get("link_depth") is not None:
                block.append(f"LINK_DEPTH: {extra.get('link_depth', 0)}")

            block.append("CONTENT:")
            block.append(content or "n/a")

            blocks.append("\n".join(block))

        return "\n\n".join(blocks)