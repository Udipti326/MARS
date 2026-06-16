from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from backend.services.semantic_chunker import SemanticChunkerService
from backend.utils.text_cleaning import clean_text


def _normalize_text_key(text: str) -> str:
    return " ".join((text or "").lower().split()).strip()


def _stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _source_type_boost(source_type: str) -> float:
    st = (source_type or "").lower()
    return {
        "arxiv": 0.12,
        "github": 0.10,
        "wikipedia": 0.08,
        "linked_web": 0.07,
        "web": 0.07,
        "youtube": 0.05,
    }.get(st, 0.04)


class FaissVectorStore:
    """
    Persistent per-session FAISS index + metadata store.

    Files per session:
      backend/storage/faiss/<session_id>.index
      backend/storage/faiss/<session_id>.json
    """

    def __init__(
        self,
        storage_dir: str | Path = "backend/storage/faiss",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        lazy_load: bool = True,
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_model_name = embedding_model
        self.embedder = SentenceTransformer(
    embedding_model,
    device="cpu",
)

    def _index_path(self, session_id: str) -> Path:
        safe_id = re.sub(r"[^a-zA-Z0-9_\-]+", "_", session_id)
        return self.storage_dir / f"{safe_id}.index"

    def _meta_path(self, session_id: str) -> Path:
        safe_id = re.sub(r"[^a-zA-Z0-9_\-]+", "_", session_id)
        return self.storage_dir / f"{safe_id}.json"

    @staticmethod
    def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return vectors / norms

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        texts = [str(t or "").strip() for t in texts if str(t or "").strip()]

        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        vectors = self.embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
        )

        return np.asarray(vectors, dtype=np.float32)

    def build_session_index(self, session_id: str, chunks: List[Dict[str, Any]]) -> int:
        if not chunks:
            return 0

        texts = [str(c.get("embedding_text") or c.get("text") or "") for c in chunks]
        vectors = self.embed_texts(texts)
        if vectors.size == 0:
            return 0

        dim = int(vectors.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        faiss.write_index(index, str(self._index_path(session_id)))

        payload = {
            "embedding_model": self.embedding_model_name,
            "session_id": session_id,
            "chunks": chunks,
        }

        with open(self._meta_path(session_id), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        return len(chunks)

    def load_session(self, session_id: str):
        index_path = self._index_path(session_id)
        meta_path = self._meta_path(session_id)

        if not index_path.exists() or not meta_path.exists():
            return None, []

        index = faiss.read_index(str(index_path))

        with open(meta_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        chunks = payload.get("chunks", [])
        if not isinstance(chunks, list):
            chunks = []

        return index, chunks

    def search(
        self,
        session_id: str,
        query_text: str,
        top_k: int = 8,
        candidate_pool: int = 40,
        per_source_cap: int = 2,
    ) -> List[Dict[str, Any]]:
        index, chunks = self.load_session(session_id)
        if index is None or not chunks:
            return []

        query_vec = self.embed_texts([query_text])
        if query_vec.size == 0:
            return []

        pool = min(candidate_pool, len(chunks))
        scores, indices = index.search(query_vec, pool)

        ranked_candidates: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(chunks):
                continue
            item = dict(chunks[idx])
            item["retrieval_score"] = float(score)
            ranked_candidates.append(item)

        ranked_candidates.sort(
            key=lambda x: (
                x.get("retrieval_score", 0.0),
                x.get("rank_score", 0.0),
                x.get("token_count", 0),
            ),
            reverse=True,
        )

        selected: List[Dict[str, Any]] = []
        source_counts: Dict[str, int] = {}

        for item in ranked_candidates:
            source_id = str(item.get("source_id", "") or "")
            if not source_id:
                source_id = _stable_hash(
                    "|".join(
                        [
                            str(item.get("source_title", "")),
                            str(item.get("source_url", "")),
                            str(item.get("text", ""))[:100],
                        ]
                    )
                )

            if source_counts.get(source_id, 0) >= per_source_cap:
                continue

            source_counts[source_id] = source_counts.get(source_id, 0) + 1
            selected.append(item)

            if len(selected) >= top_k:
                break

        unique: List[Dict[str, Any]] = []
        seen = set()
        for item in selected:
            key = _normalize_text_key(str(item.get("text", "")))
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(item)

        return unique


class EvidenceRetrievalService:
    """
    Retrieval -> Chonkie chunking -> FastEmbed embeddings -> FAISS search -> top-K evidence.
    """

    def __init__(
        self,
        chunk_embedding_model: str = "minishlab/potion-base-32M",
        retrieval_embedding_model: str = "BAAI/bge-small-en-v1.5",
        storage_dir: str | Path = "backend/storage/faiss",
    ):
        self.chunker = SemanticChunkerService(embedding_model=chunk_embedding_model)
        self.store = FaissVectorStore(
            storage_dir=storage_dir,
            embedding_model=retrieval_embedding_model,
            lazy_load=True,
        )

    def ingest_sources(self, session_id: str, query: str, sources: List[Dict[str, Any]]) -> int:
        chunks = self.chunker.chunk_sources(sources, session_id=session_id, query=query)
        if not chunks:
            return 0
        return self.store.build_session_index(session_id, chunks)

    def build_claim_evidence_pack(
        self,
        session_id: str,
        query: str,
        claim: str,
        top_k: int = 8,
    ) -> Dict[str, Any]:
        query_text = clean_text(f"{query}. {claim}".strip(), max_length=1200)

        selected = self.store.search(
            session_id=session_id,
            query_text=query_text,
            top_k=top_k,
            candidate_pool=40,
            per_source_cap=2,
        )

        # Optional small heuristic boost after FAISS retrieval.
        for item in selected:
            retrieval_score = _safe_float(item.get("retrieval_score", 0.0))
            rank_score = _safe_float(item.get("rank_score", 0.0))
            source_type = str(item.get("source_type", "") or "")
            type_boost = _source_type_boost(source_type)

            final_score = 0.88 * retrieval_score + 0.08 * rank_score + type_boost
            text = str(item.get("text", "") or "").lower()

            if claim and claim.lower() in text:
                final_score += 0.08
            elif query and query.lower() in text:
                final_score += 0.03

            item["retrieval_score"] = float(max(0.0, min(1.0, final_score)))

        selected.sort(
            key=lambda x: (
                x.get("retrieval_score", 0.0),
                x.get("rank_score", 0.0),
                x.get("token_count", 0),
            ),
            reverse=True,
        )

        lines = [
            f"Query: {query}",
            f"Claim: {claim}",
            "",
            "Use only the evidence items below.",
            "If evidence is weak or unrelated, say so explicitly.",
            "",
        ]

        for idx, item in enumerate(selected, start=1):
            evidence_id = f"E{idx}"
            source_ref = f"S{int(item.get('source_index', idx) or idx)}"
            item["evidence_id"] = evidence_id
            item["source_ref"] = source_ref

            lines.append(
                f"[{evidence_id} | {source_ref}] "
                f"score={item.get('retrieval_score', 0.0):.3f} | "
                f"rank={float(item.get('rank_score', 0.0) or 0.0):.3f} | "
                f"source={item.get('source_title', 'source')} | "
                f"type={item.get('source_type', 'unknown')} | "
                f"url={item.get('source_url') or 'n/a'}"
            )
            lines.append(f"Snippet: {item.get('text', '')}")
            lines.append("")

        return {
            "query": query,
            "claim": claim,
            "selected_evidence": selected,
            "evidence_context": "\n".join(lines).strip(),
        }