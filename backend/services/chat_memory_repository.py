from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor, Json


class ChatMemoryRepository:
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.getenv("DATABASE_URL", "").strip()
        if not self.database_url:
            raise ValueError("DATABASE_URL is missing")

    @contextmanager
    def _conn(self):
        conn = psycopg2.connect(self.database_url)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def append_message(
        self,
        expedition_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        metadata = metadata or {}

        with self._conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO chat_memory (expedition_id, role, content, metadata_json)
                    VALUES (%s, %s, %s, %s::jsonb)
                    RETURNING id, expedition_id, role, content, metadata_json, created_at
                    """,
                    (expedition_id, role, content, Json(metadata)),
                )
                row = dict(cur.fetchone())
                if isinstance(row.get("metadata_json"), str):
                    try:
                        row["metadata_json"] = json.loads(row["metadata_json"])
                    except Exception:
                        row["metadata_json"] = {}
                return row

    def list_messages(self, expedition_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if limit and limit > 0:
                    cur.execute(
                        """
                        SELECT id, expedition_id, role, content, metadata_json, created_at
                        FROM chat_memory
                        WHERE expedition_id = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                        """,
                        (expedition_id, limit),
                    )
                    rows = list(reversed(cur.fetchall()))
                else:
                    cur.execute(
                        """
                        SELECT id, expedition_id, role, content, metadata_json, created_at
                        FROM chat_memory
                        WHERE expedition_id = %s
                        ORDER BY created_at ASC
                        """,
                        (expedition_id,),
                    )
                    rows = cur.fetchall()

                out = []
                for r in rows:
                    row = dict(r)
                    if isinstance(row.get("metadata_json"), str):
                        try:
                            row["metadata_json"] = json.loads(row["metadata_json"])
                        except Exception:
                            row["metadata_json"] = {}
                    out.append(row)
                return out

    def delete_messages(self, expedition_id: str) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chat_memory WHERE expedition_id = %s", (expedition_id,))