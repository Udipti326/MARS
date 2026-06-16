from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.utils.text_cleaning import clean_text

_WORD_RE = re.compile(r"\w+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_RE = re.compile(r"\n{2,}")


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def _normalize_ws(text: str) -> str:
    return " ".join((text or "").split()).strip()


@dataclass
class SimpleChunk:
    text: str
    start_index: int
    end_index: int
    token_count: int


class SemanticChunkerService:
    """
    Chonkie-backed semantic chunker with safe fallback.

    Uses Chonkie when it can initialize cleanly.
    Falls back to a safe sentence-window chunker otherwise.
    """

    def __init__(
        self,
        embedding_model: str | None = None,
        threshold: float = 0.72,
        chunk_size: int = 512,
        similarity_window: int = 3,
        skip_window: int = 1,
        min_sentences_per_chunk: int = 1,
        min_characters_per_sentence: int = 24,
        overlap_sentences: int = 1,
        max_chars: int = 30000,
    ):
        self.embedding_model_name = embedding_model or "minishlab/potion-base-32M"
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.similarity_window = similarity_window
        self.skip_window = skip_window
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.min_characters_per_sentence = min_characters_per_sentence
        self.overlap_sentences = overlap_sentences
        self.max_chars = max_chars

        self._mode = "fallback"
        self._chunker = None

        try:
            from chonkie import Model2VecEmbeddings, SemanticChunker as ChonkieSemanticChunker
        except Exception:
            Model2VecEmbeddings = None
            ChonkieSemanticChunker = None

        if Model2VecEmbeddings is not None and ChonkieSemanticChunker is not None:
            try:
                embeddings = Model2VecEmbeddings()

                self._chunker = ChonkieSemanticChunker(
                    embedding_model=embeddings,
                    threshold=threshold,
                    chunk_size=chunk_size,
                    similarity_window=similarity_window,
                    skip_window=skip_window,
                    min_sentences_per_chunk=min_sentences_per_chunk,
                    min_characters_per_sentence=min_characters_per_sentence,
                )

                self._mode = "chonkie"
            except Exception:
                self._chunker = None
                self._mode = "fallback"

    def _prepare_text(self, text: str) -> str:
        text = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        if len(text) > self.max_chars:
            text = text[: self.max_chars]
        return text

    def _fallback_semantic_chunks(self, text: str):
        text = self._prepare_text(text)
        if not text:
            return []

        chunks = []
        paragraphs = [p.strip() for p in _PARAGRAPH_RE.split(text) if p.strip()]
        if not paragraphs:
            paragraphs = [text]

        cursor = 0
        for para in paragraphs:
            if _word_count(para) <= self.chunk_size:
                start = text.find(para, cursor)
                if start == -1:
                    start = cursor
                end = start + len(para)
                chunks.append(
                    {
                        "text": _normalize_ws(para),
                        "start_index": start,
                        "end_index": end,
                        "token_count": _word_count(para),
                    }
                )
                cursor = end
                continue

            sentences = [s.strip() for s in _SENTENCE_RE.split(para) if s.strip()]
            if not sentences:
                start = text.find(para, cursor)
                if start == -1:
                    start = cursor
                end = start + len(para)
                chunks.append(
                    {
                        "text": _normalize_ws(para),
                        "start_index": start,
                        "end_index": end,
                        "token_count": _word_count(para),
                    }
                )
                cursor = end
                continue

            window = []
            window_words = 0

            for sentence in sentences:
                s_words = _word_count(sentence)
                if window and window_words + s_words > self.chunk_size:
                    chunk_text = _normalize_ws(" ".join(window))
                    if chunk_text:
                        start = text.find(chunk_text, cursor)
                        if start == -1:
                            start = cursor
                        end = start + len(chunk_text)
                        chunks.append(
                            {
                                "text": chunk_text,
                                "start_index": start,
                                "end_index": end,
                                "token_count": _word_count(chunk_text),
                            }
                        )
                        cursor = end

                    if self.overlap_sentences > 0 and len(window) > self.overlap_sentences:
                        window = window[-self.overlap_sentences :]
                        window_words = _word_count(" ".join(window))
                    else:
                        window = []
                        window_words = 0

                window.append(sentence)
                window_words += s_words

            if window:
                chunk_text = _normalize_ws(" ".join(window))
                if chunk_text:
                    start = text.find(chunk_text, cursor)
                    if start == -1:
                        start = cursor
                    end = start + len(chunk_text)
                    chunks.append(
                        {
                            "text": chunk_text,
                            "start_index": start,
                            "end_index": end,
                            "token_count": _word_count(chunk_text),
                        }
                    )
                    cursor = end

        out = []
        seen = set()
        for c in chunks:
            key = _normalize_ws(c["text"]).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(c)
        return out

    def chunk_text(self, text: str):
        text = self._prepare_text(text)
        if not text:
            return []

        if self._mode == "chonkie" and self._chunker is not None:
            try:
                return self._chunker.chunk(text)
            except Exception:
                return self._fallback_semantic_chunks(text)

        return self._fallback_semantic_chunks(text)

    def chunk_source(
        self,
        source: Dict[str, Any],
        session_id: str,
        source_index: int,
        query: str = "",
    ) -> List[Dict[str, Any]]:
        title = _normalize_ws(str(source.get("title", "") or ""))
        content = str(source.get("content", "") or "").strip()
        url = str(source.get("url", "") or "").strip()
        source_type = str(source.get("source_type", "") or "").strip()
        domain = str(source.get("domain", "") or "").strip()
        published_at = str(source.get("published_at", "") or "").strip()

        if title and content:
            text_for_chunking = f"{title}\n\n{content}"
        elif title:
            text_for_chunking = title
        else:
            text_for_chunking = content

        raw_chunks = self.chunk_text(text_for_chunking)
        if not raw_chunks:
            return []

        source_id = re.sub(r"\W+", "", f"{session_id}_{source_index}_{url or title or content[:120]}")[:32] or f"src{source_index}"

        records: List[Dict[str, Any]] = []
        for chunk_index, chunk in enumerate(raw_chunks, start=1):
            if hasattr(chunk, "text"):
                chunk_text = _normalize_ws(str(chunk.text or ""))
                start_index = int(getattr(chunk, "start_index", -1) or -1)
                end_index = int(getattr(chunk, "end_index", -1) or -1)
                token_count = int(getattr(chunk, "token_count", 0) or _word_count(chunk_text))
            else:
                chunk_text = _normalize_ws(str(chunk or ""))
                start_index = int(chunk.get("start_index", -1)) if isinstance(chunk, dict) else -1
                end_index = int(chunk.get("end_index", -1)) if isinstance(chunk, dict) else -1
                token_count = int(chunk.get("token_count", _word_count(chunk_text))) if isinstance(chunk, dict) else _word_count(chunk_text)

            if not chunk_text:
                continue

            if token_count < 8 and len(raw_chunks) > 1:
                continue

            records.append(
                {
                    "chunk_id": f"{source_id}_{chunk_index}",
                    "session_id": session_id,
                    "query": query,
                    "source_index": source_index,
                    "source_id": source_id,
                    "source_title": title,
                    "source_url": url,
                    "source_type": source_type,
                    "domain": domain,
                    "published_at": published_at,
                    "rank_score": float(source.get("rank_score", 0.0) or 0.0),
                    "chunk_index": chunk_index,
                    "section_title": title,
                    "token_count": token_count,
                    "text": chunk_text,
                    "embedding_text": f"{title}. {chunk_text}" if title else chunk_text,
                    "start_index": start_index,
                    "end_index": end_index,
                    "chunker": {
                        "name": "chonkie.semantic.SemanticChunker",
                        "mode": self._mode,
                        "threshold": self.threshold,
                        "chunk_size": self.chunk_size,
                        "similarity_window": self.similarity_window,
                        "skip_window": self.skip_window,
                        "overlap_sentences": self.overlap_sentences,
                    },
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        return records

    def chunk_sources(self, sources: List[Dict[str, Any]], session_id: str, query: str = "") -> List[Dict[str, Any]]:
        all_chunks: List[Dict[str, Any]] = []
        for source_index, source in enumerate(sources or [], start=1):
            if not isinstance(source, dict):
                continue
            all_chunks.extend(self.chunk_source(source, session_id, source_index, query=query))
        return all_chunks