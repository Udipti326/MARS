from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2.extras import Json, RealDictCursor


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _json_safe(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


class ExpeditionRepository:
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

    def get_or_create_user(self, email: str, display_name: str = "Local User") -> Dict[str, Any]:
        email = (email or "guest@mars.local").strip().lower()
        display_name = (display_name or "Local User").strip()

        with self._conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO users (email, display_name)
                    VALUES (%s, %s)
                    ON CONFLICT (email)
                    DO UPDATE SET display_name = COALESCE(EXCLUDED.display_name, users.display_name)
                    RETURNING id, email, display_name, created_at
                    """,
                    (email, display_name),
                )
                return dict(cur.fetchone())

    def save_expedition(
        self,
        user_email: str,
        expedition: Dict[str, Any],
        display_name: str = "Local User",
        expedition_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        user = self.get_or_create_user(user_email, display_name=display_name)
        user_id = user["id"]

        root_query = str(expedition.get("query") or expedition.get("root_query") or "").strip()
        title = str(expedition.get("title") or root_query[:120] or "Untitled Expedition").strip()
        summary_json = _json_safe(expedition.get("summary") or expedition.get("summary_json") or {})
        overall_json = _json_safe(expedition.get("overall") or expedition.get("overall_json") or {})
        status = str(expedition.get("status") or "completed").strip()

        sources = expedition.get("sources", []) or []
        claims = expedition.get("claims", []) or []
        messages = expedition.get("messages", []) or []

        with self._conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if expedition_id:
                    cur.execute(
                        """
                        UPDATE expeditions
                        SET user_id = %s,
                            title = %s,
                            root_query = %s,
                            summary_json = %s::jsonb,
                            overall_json = %s::jsonb,
                            status = %s,
                            updated_at = now(),
                            last_accessed_at = now()
                        WHERE id = %s
                        RETURNING id
                        """,
                        (
                            user_id,
                            title,
                            root_query,
                            Json(summary_json),
                            Json(overall_json),
                            status,
                            expedition_id,
                        ),
                    )
                    row = cur.fetchone()
                    if not row:
                        raise ValueError(f"Expedition not found: {expedition_id}")
                    final_id = str(row["id"])

                    cur.execute(
                        """
                        DELETE FROM expedition_evidence
                        WHERE expedition_claim_id IN (
                            SELECT id FROM expedition_claims WHERE expedition_id = %s
                        )
                        """,
                        (final_id,),
                    )
                    cur.execute("DELETE FROM expedition_claims WHERE expedition_id = %s", (final_id,))
                    cur.execute("DELETE FROM expedition_sources WHERE expedition_id = %s", (final_id,))
                else:
                    cur.execute(
                        """
                        INSERT INTO expeditions (
                            user_id, title, root_query, summary_json, overall_json, status, last_accessed_at
                        )
                        VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, now())
                        RETURNING id
                        """,
                        (
                            user_id,
                            title,
                            root_query,
                            Json(summary_json),
                            Json(overall_json),
                            status,
                        ),
                    )
                    final_id = str(cur.fetchone()["id"])

                for i, source in enumerate(sources or [], start=1):
                    if not isinstance(source, dict):
                        continue

                    cur.execute(
                        """
                        INSERT INTO expedition_sources (
                            expedition_id, source_index, title, url, domain, source_type,
                            rank_score, content, raw_json
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                        """,
                        (
                            final_id,
                            int(source.get("source_index", i) or i),
                            str(source.get("title", "") or ""),
                            str(source.get("url", "") or ""),
                            str(source.get("domain", "") or ""),
                            str(source.get("source_type", "") or ""),
                            _safe_float(source.get("rank_score", 0.0), 0.0),
                            str(source.get("content", "") or ""),
                            Json(_json_safe(source)),
                        ),
                    )

                for i, claim in enumerate(claims or [], start=1):
                    if not isinstance(claim, dict):
                        continue

                    judge = claim.get("judge", {}) or {}
                    scores = claim.get("scores", {}) or {}
                    factcheck = claim.get("factcheck", {}) or {}
                    support = claim.get("support", {}) or {}
                    skeptic = claim.get("skeptic", {}) or {}

                    claim_text = (
                        claim.get("claim")
                        or claim.get("claim_text")
                        or claim.get("text")
                        or claim.get("statement")
                        or ""
                    )

                    cur.execute(
                        """
                        INSERT INTO expedition_claims (
                            expedition_id, claim_index, claim_text, verdict, confidence, label,
                            scores_json, judge_json, factcheck_json, support_json, skeptic_json
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb)
                        RETURNING id
                        """,
                        (
                            final_id,
                            int(claim.get("claim_index", i) or i),
                            str(claim_text or ""),
                            str(judge.get("verdict") or claim.get("verdict") or ""),
                            _safe_float(scores.get("confidence", claim.get("confidence", 0.0)), 0.0),
                            str(scores.get("label") or claim.get("label") or ""),
                            Json(_json_safe(scores)),
                            Json(_json_safe(judge)),
                            Json(_json_safe(factcheck)),
                            Json(_json_safe(support)),
                            Json(_json_safe(skeptic)),
                        ),
                    )
                    claim_id = str(cur.fetchone()["id"])

                    evidence_pack = claim.get("evidence_pack", {}) or {}
                    selected_evidence = evidence_pack.get("selected_evidence", []) or []

                    for ev in selected_evidence:
                        if not isinstance(ev, dict):
                            continue
                        cur.execute(
                            """
                            INSERT INTO expedition_evidence (
                                expedition_claim_id, evidence_id, chunk_id, source_title, source_url,
                                text, retrieval_score, rerank_score, metadata_json
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                            """,
                            (
                                claim_id,
                                str(ev.get("evidence_id", "") or ""),
                                str(ev.get("chunk_id", "") or ""),
                                str(ev.get("source_title", "") or ""),
                                str(ev.get("source_url", "") or ""),
                                str(ev.get("text", "") or ""),
                                _safe_float(ev.get("retrieval_score", 0.0), 0.0),
                                _safe_float(ev.get("rerank_score", 0.0), 0.0),
                                Json(_json_safe(ev)),
                            ),
                        )

                for msg in messages or []:
                    if not isinstance(msg, dict):
                        continue
                    role = str(msg.get("role", "") or "").strip()
                    content = str(msg.get("content", "") or "").strip()
                    if not role or not content:
                        continue
                    cur.execute(
                        """
                        INSERT INTO expedition_messages (
                            expedition_id, role, content, metadata_json
                        )
                        VALUES (%s, %s, %s, %s::jsonb)
                        """,
                        (
                            final_id,
                            role,
                            content,
                            Json(_json_safe(msg.get("metadata_json") or msg.get("metadata") or {})),
                        ),
                    )

                return {"id": final_id, "user": user}

    def list_expeditions(self, user_email: str) -> List[Dict[str, Any]]:
        email = (user_email or "guest@mars.local").strip().lower()

        with self._conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT
                        e.id,
                        e.title,
                        e.root_query,
                        e.status,
                        e.created_at,
                        e.updated_at,
                        e.last_accessed_at,
                        COALESCE((SELECT COUNT(*) FROM expedition_sources s WHERE s.expedition_id = e.id), 0) AS source_count,
                        COALESCE((SELECT COUNT(*) FROM expedition_claims c WHERE c.expedition_id = e.id), 0) AS claim_count,
                        COALESCE((SELECT COUNT(*) FROM chat_memory m WHERE m.expedition_id = e.id), 0) AS message_count
                    FROM expeditions e
                    JOIN users u ON u.id = e.user_id
                    WHERE u.email = %s
                    ORDER BY e.updated_at DESC, e.created_at DESC
                    """,
                    (email,),
                )
                return [_json_safe(dict(r)) for r in cur.fetchall()]

    def get_expedition_detail(self, expedition_id: str) -> Dict[str, Any]:
        with self._conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT e.*, u.email AS user_email, u.display_name
                    FROM expeditions e
                    JOIN users u ON u.id = e.user_id
                    WHERE e.id = %s
                    """,
                    (expedition_id,),
                )
                expedition = cur.fetchone()
                if not expedition:
                    raise ValueError(f"Expedition not found: {expedition_id}")

                exp = dict(expedition)
                exp["summary_json"] = _json_safe(exp.get("summary_json"))
                exp["overall_json"] = _json_safe(exp.get("overall_json"))
                exp["summary"] = _json_safe(exp.get("summary_json"))
                exp["overall"] = _json_safe(exp.get("overall_json"))
                exp["display_name"] = exp.get("display_name") or ""
                exp["user_email"] = exp.get("user_email") or ""

                cur.execute(
                    "SELECT * FROM expedition_sources WHERE expedition_id = %s ORDER BY source_index ASC",
                    (expedition_id,),
                )
                sources = []
                for row in cur.fetchall():
                    source = dict(row)
                    source["rank_score"] = _safe_float(source.get("rank_score", 0.0), 0.0)
                    sources.append(source)

                cur.execute(
                    "SELECT * FROM expedition_claims WHERE expedition_id = %s ORDER BY claim_index ASC",
                    (expedition_id,),
                )
                claims = []
                for row in cur.fetchall():
                    claim = dict(row)

                    claim_text = (
                        claim.get("claim_text")
                        or claim.get("claim")
                        or claim.get("text")
                        or ""
                    )

                    claim["claim"] = claim_text
                    claim["claim_text"] = claim_text
                    claim["text"] = claim_text

                    claim["scores_json"] = _json_safe(claim.get("scores_json") or {})
                    claim["judge_json"] = _json_safe(claim.get("judge_json") or {})
                    claim["factcheck_json"] = _json_safe(claim.get("factcheck_json") or {})
                    claim["support_json"] = _json_safe(claim.get("support_json") or {})
                    claim["skeptic_json"] = _json_safe(claim.get("skeptic_json") or {})

                    claim["scores"] = _json_safe(claim.get("scores_json") or {})
                    claim["judge"] = _json_safe(claim.get("judge_json") or {})
                    claim["factcheck"] = _json_safe(claim.get("factcheck_json") or {})
                    claim["support"] = _json_safe(claim.get("support_json") or {})
                    claim["skeptic"] = _json_safe(claim.get("skeptic_json") or {})

                    claim["confidence"] = _safe_float(claim.get("confidence", 0.0), 0.0)

                    cur.execute(
                        "SELECT * FROM expedition_evidence WHERE expedition_claim_id = %s ORDER BY created_at ASC",
                        (claim["id"],),
                    )
                    evidence_rows = []
                    for ev in cur.fetchall():
                        evidence = dict(ev)
                        evidence["retrieval_score"] = _safe_float(evidence.get("retrieval_score", 0.0), 0.0)
                        evidence["rerank_score"] = _safe_float(evidence.get("rerank_score", 0.0), 0.0)
                        evidence_rows.append(evidence)

                    claim["evidence"] = evidence_rows
                    claim["evidence_pack"] = {"selected_evidence": evidence_rows}
                    claims.append(claim)

                cur.execute(
                    """
                    SELECT * FROM expedition_messages
                    WHERE expedition_id = %s
                    ORDER BY created_at ASC
                    """,
                    (expedition_id,),
                )
                messages = [_json_safe(dict(r)) for r in cur.fetchall()]

                return {
                    "expedition": _json_safe(exp),
                    "sources": _json_safe(sources),
                    "claims": _json_safe(claims),
                    "messages": _json_safe(messages),
                    "summary": _json_safe(exp.get("summary_json", {})),
                    "summary_json": _json_safe(exp.get("summary_json", {})),
                    "overall": _json_safe(exp.get("overall_json", {})),
                    "overall_json": _json_safe(exp.get("overall_json", {})),
                    "root_query": exp.get("root_query", ""),
                    "title": exp.get("title", ""),
                    "user_email": exp.get("user_email", ""),
                    "display_name": exp.get("display_name", ""),
                }

    def delete_expedition(self, expedition_id: str) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM expedition_messages WHERE expedition_id = %s", (expedition_id,))
                cur.execute("DELETE FROM chat_memory WHERE expedition_id = %s", (expedition_id,))
                cur.execute("DELETE FROM expeditions WHERE id = %s", (expedition_id,))