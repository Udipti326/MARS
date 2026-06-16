from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class ChunkRepository:
    """
    Persistent chunk store in SQLite.
    Embeddings are optional now because retrieval is handled without torch.
    """

    def __init__(self, db_path: str | Path = "backend/storage/chunks.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    query TEXT,
                    source_id TEXT,
                    source_index INTEGER,
                    source_title TEXT,
                    source_url TEXT,
                    source_type TEXT,
                    domain TEXT,
                    published_at TEXT,
                    rank_score REAL,
                    chunk_index INTEGER,
                    section_title TEXT,
                    token_count INTEGER,
                    text TEXT,
                    metadata_json TEXT,
                    created_at TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_rank ON chunks(rank_score)")
            conn.commit()

    def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        if not chunks:
            return 0

        rows = []
        for chunk in chunks:
            rows.append(
                (
                    chunk["chunk_id"],
                    chunk["session_id"],
                    chunk.get("query", ""),
                    chunk.get("source_id", ""),
                    int(chunk.get("source_index", 0) or 0),
                    chunk.get("source_title", ""),
                    chunk.get("source_url", ""),
                    chunk.get("source_type", ""),
                    chunk.get("domain", ""),
                    chunk.get("published_at", ""),
                    float(chunk.get("rank_score", 0.0) or 0.0),
                    int(chunk.get("chunk_index", 0) or 0),
                    chunk.get("section_title", ""),
                    int(chunk.get("token_count", 0) or 0),
                    chunk.get("text", ""),
                    json.dumps(
                        {
                            "start_index": chunk.get("start_index", -1),
                            "end_index": chunk.get("end_index", -1),
                            "chunker": chunk.get("chunker", {}),
                            "embedding_text": chunk.get("embedding_text", ""),
                        },
                        ensure_ascii=False,
                    ),
                    chunk.get("created_at", ""),
                )
            )

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunks (
                    chunk_id, session_id, query, source_id, source_index,
                    source_title, source_url, source_type, domain, published_at,
                    rank_score, chunk_index, section_title, token_count,
                    text, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

        return len(rows)

    def load_chunks(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM chunks"
        params: tuple[Any, ...] = ()

        if session_id:
            query += " WHERE session_id = ?"
            params = (session_id,)

        query += " ORDER BY rank_score DESC, source_index ASC, chunk_index ASC"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        out: List[Dict[str, Any]] = []
        for row in rows:
            metadata = {}
            if row["metadata_json"]:
                try:
                    metadata = json.loads(row["metadata_json"])
                except Exception:
                    metadata = {}

            out.append(
                {
                    "chunk_id": row["chunk_id"],
                    "session_id": row["session_id"],
                    "query": row["query"],
                    "source_id": row["source_id"],
                    "source_index": row["source_index"],
                    "source_title": row["source_title"],
                    "source_url": row["source_url"],
                    "source_type": row["source_type"],
                    "domain": row["domain"],
                    "published_at": row["published_at"],
                    "rank_score": row["rank_score"],
                    "chunk_index": row["chunk_index"],
                    "section_title": row["section_title"],
                    "token_count": row["token_count"],
                    "text": row["text"],
                    "metadata_json": metadata,
                    "created_at": row["created_at"],
                }
            )

        return out

    def clear_session(self, session_id: str) -> int:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM chunks WHERE session_id = ?", (session_id,))
            conn.commit()
            return cur.rowcount

    def count_chunks(self, session_id: Optional[str] = None) -> int:
        query = "SELECT COUNT(*) AS c FROM chunks"
        params: tuple[Any, ...] = ()

        if session_id:
            query += " WHERE session_id = ?"
            params = (session_id,)

        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()

        return int(row["c"] if row else 0)