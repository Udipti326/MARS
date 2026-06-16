from __future__ import annotations

import re
from typing import Any, Dict, List

from backend.services.source_context_service import SourceContextService
from backend.utils.text_cleaning import clean_text

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
PARA_SPLIT_RE = re.compile(r"\n{2,}")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "by",
    "is", "are", "was", "were", "be", "been", "being", "as", "at", "from", "that",
    "this", "it", "its", "into", "than", "then", "them", "they", "their", "there",
    "about", "over", "under", "between", "through", "while", "during", "after",
    "before", "can", "could", "should", "would", "may", "might", "must", "not",
}


def _tokenize(text: str) -> set[str]:
    return {
        tok
        for tok in TOKEN_RE.findall((text or "").lower())
        if tok and tok not in STOPWORDS
    }


def _normalize_text_key(text: str) -> str:
    return " ".join((text or "").lower().split()).strip()


def _split_into_passages(text: str) -> List[str]:
    cleaned = clean_text(text or "", max_length=6000, remove_urls=False)
    if not cleaned:
        return []

    paragraphs = [p.strip() for p in PARA_SPLIT_RE.split(cleaned) if p.strip()]
    if not paragraphs:
        paragraphs = [cleaned]

    passages: List[str] = []

    for para in paragraphs:
        if len(para) <= 280:
            passages.append(para)
            continue

        sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(para) if s.strip()]
        if not sentences:
            passages.append(para[:320])
            continue

        for i, sent in enumerate(sentences):
            chunk = sent
            if i + 1 < len(sentences) and len(chunk) < 200:
                chunk = f"{chunk} {sentences[i + 1]}"
            passages.append(chunk[:360])

    deduped = []
    seen = set()
    for p in passages:
        key = _normalize_text_key(p)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(p)

    return deduped


class DebateEvidenceService:
    """
    Builds a compact claim-specific evidence pack from the retrieved sources.
    """

    def __init__(self):
        self.source_type_boost = {
            "arxiv": 0.12,
            "wikipedia": 0.10,
            "github": 0.09,
            "linked_web": 0.08,
            "web": 0.08,
            "youtube": 0.06,
        }

    def _score_candidate(self, query: str, claim: str, text: str, source: Dict[str, Any]) -> float:
        text_lower = (text or "").lower()
        if not text_lower.strip():
            return 0.0
        if "tool execution failed" in text_lower:
            return 0.0

        q_tokens = _tokenize(query)
        c_tokens = _tokenize(claim)
        t_tokens = _tokenize(text)
        title_tokens = _tokenize(str(source.get("title", "")))

        q_overlap = len(q_tokens & t_tokens) / max(len(q_tokens), 1)
        c_overlap = len(c_tokens & t_tokens) / max(len(c_tokens), 1)
        title_overlap = len((q_tokens | c_tokens) & title_tokens) / max(len(title_tokens), 1) if title_tokens else 0.0

        rank_score = float(source.get("rank_score", 0.0) or 0.0)
        rank_bonus = min(0.12, max(0.0, rank_score) * 0.12)

        source_type = str(source.get("source_type", "")).lower()
        source_boost = self.source_type_boost.get(source_type, 0.05)

        exact_bonus = 0.18 if claim and claim.lower() in text_lower else 0.0
        length_bonus = 0.04 if 40 <= len(text) <= 450 else 0.0

        score = (
            0.48 * c_overlap +
            0.22 * q_overlap +
            0.10 * title_overlap +
            source_boost +
            rank_bonus +
            exact_bonus +
            length_bonus
        )

        return round(max(0.0, min(1.0, score)), 3)

    def _source_snippets(self, source: Dict[str, Any]) -> List[str]:
        title = clean_text(str(source.get("title", "")), max_length=220, remove_urls=False)
        content = str(source.get("content", "") or "").strip()
        passages = _split_into_passages(content)

        snippets: List[str] = []
        if title:
            snippets.append(title)
        snippets.extend(passages)

        deduped = []
        seen = set()
        for snip in snippets:
            key = _normalize_text_key(snip)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(snip)

        return deduped

    def build_claim_evidence_pack(
        self,
        query: str,
        claim: str,
        sources: List[Dict[str, Any]],
        source_bundle: str = "",
        top_k: int = 8,
        snippets_per_source: int = 4,
    ) -> Dict[str, Any]:
        ordered_sources = SourceContextService.prepare_sources(sources, max_sources=max(12, top_k * 2))

        candidates: List[Dict[str, Any]] = []

        for src_index, source in enumerate(ordered_sources, start=1):
            if not isinstance(source, dict):
                continue

            content = str(source.get("content", "") or "")
            title = str(source.get("title", "") or "")
            if "tool execution failed" in content.lower():
                continue
            if not title and not content.strip():
                continue

            snippets = self._source_snippets(source)[:snippets_per_source]
            source_ref = f"S{src_index}"

            for snippet_index, snippet in enumerate(snippets, start=1):
                score = self._score_candidate(query, claim, snippet, source)
                if score <= 0:
                    continue

                candidates.append(
                    {
                        "source_index": src_index,
                        "source_ref": source_ref,
                        "snippet_index": snippet_index,
                        "score": score,
                        "text": snippet,
                        "source_title": title or str(source.get("source_type", "source")),
                        "source_url": str(source.get("url", "") or ""),
                        "source_type": str(source.get("source_type", "") or ""),
                        "domain": str(source.get("domain", "") or ""),
                        "published_at": str(source.get("published_at", "") or ""),
                        "rank_score": float(source.get("rank_score", 0.0) or 0.0),
                    }
                )

        candidates.sort(key=lambda x: (x["score"], x["rank_score"]), reverse=True)

        selected: List[Dict[str, Any]] = []
        seen_text = set()
        for item in candidates:
            key = _normalize_text_key(item["text"])
            if not key or key in seen_text:
                continue
            seen_text.add(key)
            selected.append(item)
            if len(selected) >= top_k:
                break

        if not selected:
            for i, src in enumerate(ordered_sources[:top_k], start=1):
                title = clean_text(str(src.get("title", "")), max_length=180)
                content = clean_text(str(src.get("content", "")), max_length=220)
                snippet = title if title else content
                if not snippet:
                    continue
                selected.append(
                    {
                        "source_index": i,
                        "source_ref": f"S{i}",
                        "snippet_index": 1,
                        "score": 0.1,
                        "text": snippet,
                        "source_title": title or str(src.get("source_type", "source")),
                        "source_url": str(src.get("url", "") or ""),
                        "source_type": str(src.get("source_type", "") or ""),
                        "domain": str(src.get("domain", "") or ""),
                        "published_at": str(src.get("published_at", "") or ""),
                        "rank_score": float(src.get("rank_score", 0.0) or 0.0),
                    }
                )

        lines = [
            f"Query: {query}",
            f"Claim: {claim}",
            "",
            "Use only the evidence items below.",
            "If the evidence is weak or unrelated, say so explicitly.",
            "",
        ]

        for idx, item in enumerate(selected, start=1):
            ev_id = f"E{idx}"
            item["evidence_id"] = ev_id

            lines.append(
                f"[{ev_id} | {item['source_ref']}] score={item['score']:.3f} | "
                f"source={item['source_title']} | "
                f"type={item['source_type']} | "
                f"domain={item['domain'] or 'n/a'} | "
                f"url={item['source_url'] or 'n/a'}"
            )
            lines.append(f"Snippet: {item['text']}")
            lines.append("")

        if source_bundle:
            lines.append("Additional source bundle context:")
            lines.append(source_bundle[:1500])

        return {
            "query": query,
            "claim": claim,
            "selected_evidence": selected,
            "evidence_context": "\n".join(lines).strip(),
        }