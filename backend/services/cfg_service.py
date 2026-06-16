# backend/services/cfg_service.py
from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Tuple

from neo4j import GraphDatabase

from backend.core.llm import get_llm
from backend.services.cfg_semantic_scorer import SemanticRelatednessScorer
from backend.services.expedition_repository import ExpeditionRepository

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "to", "of", "for", "in", "on", "at", "by",
    "with", "without", "from", "into", "over", "under", "about", "as", "is", "are", "was", "were", "be",
    "been", "being", "that", "this", "these", "those", "it", "its", "their", "his", "her", "our", "your",
    "we", "you", "they", "i", "me", "my", "mine", "not", "do", "does", "did", "done",
}

FOLLOWUP_TEMPLATES = [
    "examples",
    "applications",
    "limitations",
    "related concepts",
    "deeper analysis",
]

FOCUS_VERBS = (
    r"has|have|contains|contain|includes|include|is|are|was|were|represents|defines|describes|explains|"
    r"supports|improves|enables|models|outperforms|compares to|relates to|uses|used by|depends on"
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _entity_to_dict(entity: Any) -> Dict[str, Any]:
    if entity is None:
        return {}

    if isinstance(entity, Mapping):
        return _json_safe(dict(entity))

    try:
        return _json_safe(dict(entity.items()))
    except Exception:
        pass

    try:
        return _json_safe(dict(entity))
    except Exception:
        pass

    props = getattr(entity, "_properties", None)
    if props is not None:
        try:
            return _json_safe(dict(props))
        except Exception:
            pass

    return {"value": str(entity)}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _node_key(text: str) -> str:
    text = _normalize_text(text).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "item"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        key = _normalize_text(item).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _split_candidate_phrases(text: str) -> List[str]:
    text = _normalize_text(text)
    if not text:
        return []

    pieces = re.split(r"[,\.;:\n\|\/]+", text)
    out: List[str] = []

    for piece in pieces:
        piece = _normalize_text(piece)
        if not piece:
            continue

        m = re.match(
            rf"^(?:the|a|an)?\s*(.+?)\s+(?:{FOCUS_VERBS})\b",
            piece,
            flags=re.IGNORECASE,
        )
        if m:
            phrase = _normalize_text(m.group(1))
            if phrase:
                out.append(phrase)

        subpieces = re.split(
            r"\b(?:and|or|vs|versus|with|using|based on|through|along with)\b",
            piece,
            flags=re.IGNORECASE,
        )
        for sp in subpieces:
            sp = _normalize_text(sp)
            if not sp:
                continue

            words = [w for w in re.split(r"\s+", sp) if w]
            if 2 <= len(words) <= 7:
                candidate = " ".join(words).strip(" -_")
                if len(candidate) >= 3:
                    out.append(candidate)

    return _unique_preserve_order(out)


def _extract_concepts(detail: Dict[str, Any]) -> List[Dict[str, Any]]:
    claims = detail.get("claims", []) or []
    summary = detail.get("summary", {}) or {}

    candidates: List[Tuple[str, str]] = []

    root_query = str(detail.get("root_query", "") or "").strip()
    title = str(detail.get("title", "") or "").strip()

    if root_query:
        candidates.append((root_query, "query"))
    if title and title.lower() != root_query.lower():
        candidates.append((title, "title"))

    if isinstance(summary, dict):
        summary_text = summary.get("summary") or summary.get("text") or ""
        if summary_text:
            candidates.append((str(summary_text), "summary"))

    for claim in claims[:20]:
        if not isinstance(claim, dict):
            continue
        claim_text = (
            claim.get("claim_text")
            or claim.get("claim")
            or claim.get("text")
            or ""
        )
        if claim_text:
            candidates.append((str(claim_text), "claim"))

    seen = set()
    concepts: List[Dict[str, Any]] = []

    for text, kind in candidates:
        for phrase in _split_candidate_phrases(text):
            key = phrase.lower()
            if key in seen:
                continue
            seen.add(key)

            words = [w.lower() for w in re.split(r"\s+", phrase) if w]
            if not words:
                continue

            if all(w in STOPWORDS for w in words):
                continue
            if len(words) == 1 and len(words[0]) < 3:
                continue

            concepts.append(
                {
                    "name": phrase,
                    "kind": kind,
                    "weight": 1.0 if kind in {"query", "title"} else 0.75,
                }
            )

    return concepts


def _extract_claim_meta(claim: Dict[str, Any]) -> Dict[str, Any]:
    scores = claim.get("scores") or claim.get("scores_json") or {}
    judge = claim.get("judge") or claim.get("judge_json") or {}
    factcheck = claim.get("factcheck") or claim.get("factcheck_json") or {}
    support = claim.get("support") or claim.get("support_json") or {}
    skeptic = claim.get("skeptic") or claim.get("skeptic_json") or {}

    if not isinstance(scores, dict):
        scores = {}
    if not isinstance(judge, dict):
        judge = {}
    if not isinstance(factcheck, dict):
        factcheck = {}
    if not isinstance(support, dict):
        support = {}
    if not isinstance(skeptic, dict):
        skeptic = {}

    verdict = str(judge.get("verdict") or claim.get("verdict") or "").strip()
    label = str(scores.get("label") or claim.get("label") or "").strip()

    confidence = scores.get("confidence")
    if confidence is None:
        confidence = claim.get("confidence")
    if confidence is None:
        confidence = judge.get("confidence")
    confidence = _safe_float(confidence, 0.0)

    return {
        "verdict": verdict,
        "label": label,
        "confidence": confidence,
        "scores": _json_safe(scores),
        "judge": _json_safe(judge),
        "factcheck": _json_safe(factcheck),
        "support": _json_safe(support),
        "skeptic": _json_safe(skeptic),
    }


class CFGService:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687").strip() or "bolt://localhost:7687"
        self.user = os.getenv("NEO4J_USER", "neo4j").strip() or "neo4j"
        self.password = os.getenv("NEO4J_PASSWORD", "12345678").strip() or "12345678"
        self.llm = get_llm()
        self.expedition_repo = ExpeditionRepository()

        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

        artifact_dir = Path(__file__).resolve().parents[1] / "semantic_artifacts"
        try:
            self.scorer = SemanticRelatednessScorer(artifact_dir=artifact_dir)
        except TypeError:
            self.scorer = SemanticRelatednessScorer()

        self._ensure_schema()

    def close(self):
        if self.driver:
            self.driver.close()

    def _ensure_schema(self):
        cypher_statements = [
            "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
            "CREATE CONSTRAINT expedition_id IF NOT EXISTS FOR (e:Expedition) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT next_research_id IF NOT EXISTS FOR (n:NextResearch) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (c:Claim) REQUIRE c.id IS UNIQUE",
            "CREATE INDEX concept_kind IF NOT EXISTS FOR (c:Concept) ON (c.kind)",
            "CREATE INDEX concept_last_seen IF NOT EXISTS FOR (c:Concept) ON (c.last_seen_at)",
        ]
        with self.driver.session() as session:
            for stmt in cypher_statements:
                session.run(stmt)

    def _parse_json_obj(self, text: str) -> dict:
        """
        Best-effort JSON parsing for LLM output.
        Returns {} on failure instead of crashing.
        """
        raw = str(text or "").strip()
        if not raw:
            return {}

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            raw = match.group(0)

        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception as exc:
            print("[CFG LLM PARSE FAILED]", exc)
            print(raw)
            return {}

    def _llm_cfg_plan(self, root_query: str, summary: str, claims: list) -> dict:
        claim_texts = []
        for c in claims[:5]:
            if isinstance(c, dict):
                txt = c.get("claim_text") or c.get("claim") or c.get("text")
                if txt:
                    claim_texts.append(str(txt)[:180])

        prompt = f"""
Return ONLY valid JSON.
No markdown.
No explanation.
Every array item must be a quoted string.

topic: {root_query}
summary: {summary[:450]}
claims: {claim_texts}

Return exactly:
{{
  "topics": ["5 clean research topics"],
  "learn_next": ["5 next research directions"],
  "depth": [
    {{"name":"topic","depth_score":0.0,"reason":"short"}}
  ]
}}
"""
        resp = self.llm.invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        parsed = self._parse_json_obj(text)
        return parsed if isinstance(parsed, dict) else {}

    def _score_pair(self, a: str, b: str) -> float:
        try:
            return float(self.scorer.predict(str(a or ""), str(b or "")))
        except Exception as exc:
            print(f"[CFG SCORE ERROR] {a!r} vs {b!r}: {exc}")
            return 0.0

    def sync_from_detail(self, detail: Dict[str, Any]) -> Dict[str, Any]:
        expedition = detail.get("expedition", {}) or {}
        expedition_id = str(expedition.get("id") or "").strip()
        if not expedition_id:
            raise ValueError("expedition.id is missing")

        with self.driver.session() as session:
            session.execute_write(self._write_graph, detail)

        return self.get_graph(expedition_id)

    def _write_graph(self, tx, detail: Dict[str, Any]) -> Dict[str, Any]:
        expedition = detail.get("expedition") or {}
        expedition_id = str(expedition.get("id") or detail.get("id") or "").strip()
        if not expedition_id:
            raise ValueError("Missing expedition id")

        root_query = str(
            expedition.get("root_query")
            or detail.get("root_query")
            or detail.get("query")
            or expedition.get("title")
            or detail.get("title")
            or "Untitled expedition"
        ).strip()

        title = str(
            expedition.get("title")
            or detail.get("title")
            or root_query
        ).strip()

        summary_obj = (
            detail.get("summary")
            or detail.get("summary_json")
            or expedition.get("summary_json")
            or {}
        )

        if isinstance(summary_obj, dict):
            summary_text = (
                summary_obj.get("summary")
                or summary_obj.get("text")
                or summary_obj.get("answer")
                or json.dumps(summary_obj, default=str)[:800]
            )
        else:
            summary_text = str(summary_obj or "")

        claims = (
            detail.get("claims")
            or detail.get("claim_results")
            or detail.get("claims_json")
            or detail.get("expedition_claims")
            or []
        )

        if isinstance(claims, dict):
            claims = claims.get("claims") or claims.get("items") or claims.get("results") or []

        clean_claims: List[Dict[str, Any]] = []
        for i, c in enumerate((claims or [])[:8]):
            if isinstance(c, dict):
                text = (
                    c.get("claim_text")
                    or c.get("claim")
                    or c.get("text")
                    or c.get("statement")
                    or ""
                )
                meta = _extract_claim_meta(c)
                verdict = meta.get("verdict", "")
                confidence = meta.get("confidence", 0.0)
            else:
                text = str(c)
                verdict = ""
                confidence = 0.0

            text = _normalize_text(text)
            if text:
                clean_claims.append(
                    {
                        "id": f"{expedition_id}_claim_{i}",
                        "index": i,
                        "text": text,
                        "verdict": str(verdict or ""),
                        "confidence": round(_safe_float(confidence, 0.0), 4),
                    }
                )

        try:
            llm_plan = self._llm_cfg_plan(root_query, summary_text, clean_claims)
            raw_topics = llm_plan.get("topics", []) if isinstance(llm_plan, dict) else []
            raw_next = llm_plan.get("learn_next", []) if isinstance(llm_plan, dict) else []
        except Exception as e:
            print("[CFG LLM PLAN FAILED]", e)
            raw_topics = []
            raw_next = []

        bad_names = {"com", "org", "edu", "www", "http", "https", "source", "sources", "content", "title", "url"}

        def clean_name_list(items: Any, fallback: List[str], limit: int = 5) -> List[str]:
            names: List[str] = []
            seen_lower = set()

            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        name = item.get("name") or item.get("topic") or item.get("title") or ""
                    else:
                        name = item
                    name = _normalize_text(str(name))
                    if not name:
                        continue
                    low = name.lower()
                    if low in bad_names or len(name) < 3 or low in seen_lower:
                        continue
                    names.append(name)
                    seen_lower.add(low)
                    if len(names) >= limit:
                        break

            if len(names) < limit:
                for item in fallback:
                    name = _normalize_text(str(item))
                    low = name.lower()
                    if name and low not in bad_names and low not in seen_lower:
                        names.append(name)
                        seen_lower.add(low)
                    if len(names) >= limit:
                        break

            return names[:limit]

        fallback_concepts = _extract_concepts(
            {
                "root_query": root_query,
                "title": title,
                "summary": {"summary": summary_text},
                "claims": [{"claim_text": c["text"]} for c in clean_claims],
            }
        )
        fallback_topic_names = [c["name"] for c in fallback_concepts]
        fallback_topic_names.extend(
            [
                root_query,
                "Semantic relationship scoring",
                "Evidence-based claim verification",
                "Research source ranking",
                "Concept-level knowledge mapping",
            ]
        )

        topic_names = clean_name_list(raw_topics, fallback_topic_names, limit=5)

        fallback_next = [
            f"Advanced research on {root_query}",
            f"Open research gaps in {root_query}",
            f"Comparative studies related to {root_query}",
            f"Recent trends in {root_query}",
            f"Practical applications of {root_query}",
        ]
        next_names = clean_name_list(raw_next, fallback_next, limit=5)

        top_concepts: List[Dict[str, Any]] = []
        for name in topic_names:
            score = self._score_pair(root_query, name)
            top_concepts.append(
                {
                    "id": f"{expedition_id}_topic_{_node_key(name)}",
                    "name": name,
                    "score": round(score, 6),
                    "kind": "research_topic",
                }
            )

        next_to_learn: List[Dict[str, Any]] = []
        for name in next_names:
            score = self._score_pair(root_query, name)
            next_to_learn.append(
                {
                    "id": f"{expedition_id}_next_{_node_key(name)}",
                    "name": name,
                    "score": round(score, 6),
                    "reason": "Suggested next research direction based on the current expedition context",
                }
            )

        # Delete the previous CFG subgraph for this expedition.
        # We delete all nodes stamped with expedition_id, then the expedition node itself.
        tx.run(
            """
            MATCH (n)
            WHERE n.expedition_id = $expedition_id
            DETACH DELETE n
            """,
            expedition_id=expedition_id,
        )

        tx.run(
            """
            MATCH (e:Expedition {id: $expedition_id})
            DETACH DELETE e
            """,
            expedition_id=expedition_id,
        )

        tx.run(
            """
            MERGE (e:Expedition {id: $expedition_id})
            SET e.title = $title,
                e.root_query = $root_query,
                e.updated_at = datetime()
            """,
            expedition_id=expedition_id,
            title=title,
            root_query=root_query,
        )

        for concept in top_concepts:
            tx.run(
                """
                MATCH (e:Expedition {id: $expedition_id})
                MERGE (c:Concept {id: $concept_id})
                SET c.name = $name,
                    c.kind = $kind,
                    c.score = $score,
                    c.relevance_score = $score,
                    c.expedition_id = $expedition_id,
                    c.last_seen_at = datetime()
                MERGE (e)-[r:HAS_CONCEPT]->(c)
                SET r.weight = $score,
                    r.expedition_id = $expedition_id
                """,
                expedition_id=expedition_id,
                concept_id=concept["id"],
                name=concept["name"],
                kind=concept["kind"],
                score=concept["score"],
            )

        for i, a in enumerate(top_concepts):
            for b in top_concepts[i + 1:]:
                rel_score = round(self._score_pair(a["name"], b["name"]), 6)
                if rel_score <= 0:
                    continue

                tx.run(
                    """
                    MATCH (a:Concept {id: $a_id})
                    MATCH (b:Concept {id: $b_id})
                    MERGE (a)-[r:RELATED_TO {expedition_id: $expedition_id}]->(b)
                    SET r.weight = $weight
                    """,
                    a_id=a["id"],
                    b_id=b["id"],
                    expedition_id=expedition_id,
                    weight=rel_score,
                )

        for claim in clean_claims:
            tx.run(
                """
                MATCH (e:Expedition {id: $expedition_id})
                MERGE (cl:Claim {id: $claim_id})
                SET cl.text = $text,
                    cl.verdict = $verdict,
                    cl.confidence = $confidence,
                    cl.claim_index = $claim_index,
                    cl.expedition_id = $expedition_id
                MERGE (e)-[r:HAS_CLAIM]->(cl)
                SET r.weight = $confidence,
                    r.expedition_id = $expedition_id
                """,
                expedition_id=expedition_id,
                claim_id=claim["id"],
                text=claim["text"],
                verdict=claim["verdict"],
                confidence=claim["confidence"],
                claim_index=claim["index"],
            )

            claim_topic_scores: List[Tuple[Dict[str, Any], float]] = []
            for concept in top_concepts:
                rel_score = round(self._score_pair(claim["text"], concept["name"]), 6)
                claim_topic_scores.append((concept, rel_score))

            claim_topic_scores.sort(key=lambda x: x[1], reverse=True)

            for concept, rel_score in claim_topic_scores[:2]:
                if rel_score <= 0:
                    continue
                tx.run(
                    """
                    MATCH (cl:Claim {id: $claim_id})
                    MATCH (c:Concept {id: $concept_id})
                    MERGE (cl)-[r:CLAIM_TO_TOPIC {expedition_id: $expedition_id}]->(c)
                    SET r.weight = $weight
                    """,
                    claim_id=claim["id"],
                    concept_id=concept["id"],
                    weight=rel_score,
                    expedition_id=expedition_id,
                )

        for item in next_to_learn:
            tx.run(
                """
                MATCH (e:Expedition {id: $expedition_id})
                MERGE (n:NextResearch {id: $next_id})
                SET n.name = $name,
                    n.score = $score,
                    n.reason = $reason,
                    n.expedition_id = $expedition_id
                MERGE (e)-[r:NEXT_RESEARCH]->(n)
                SET r.weight = $score,
                    r.score = $score,
                    r.expedition_id = $expedition_id
                """,
                expedition_id=expedition_id,
                next_id=item["id"],
                name=item["name"],
                score=item["score"],
                reason=item["reason"],
            )

        return {"expedition_id": expedition_id, "topics": top_concepts, "next": next_to_learn}

    def get_graph(self, expedition_id: str) -> Dict[str, Any]:
        def _score_pair(a: str, b: str) -> float:
            try:
                return float(self.scorer.predict(str(a or ""), str(b or "")))
            except Exception:
                return 0.0

        def _exp_label(exp: Dict[str, Any]) -> str:
            return str(
                exp.get("title")
                or exp.get("root_query")
                or exp.get("id")
                or "Expedition"
            ).strip()

        def _exp_root(exp: Dict[str, Any]) -> str:
            return str(
                exp.get("root_query")
                or exp.get("title")
                or ""
            ).strip()

        def _concept_node_id(exp_id: str, name: str) -> str:
            return f"concept:{_node_key(exp_id)}:{_node_key(name)}"

        def _make_concept_node(
            exp_id: str,
            exp_title: str,
            exp_root: str,
            c: Dict[str, Any],
            r: Dict[str, Any],
        ) -> Dict[str, Any]:
            name = str(c.get("name") or "").strip()
            local_score = _safe_float(r.get("weight", c.get("score", 0.0)), 0.0)
            relevance_score = _score_pair(exp_root, name)
            return {
                "id": _concept_node_id(exp_id, name),
                "label": name,
                "type": "concept",
                "role": "topic",
                "parent_expedition_id": exp_id,
                "parent_expedition_title": exp_title,
                "parent_expedition_root_query": exp_root,
                "local_score": round(local_score, 6),
                "relevance_score": round(relevance_score, 6),
                "score": round(local_score, 6),
                "kind": c.get("kind", "concept"),
                "usage_count": int(c.get("usage_count", 0) or 0),
            }

        rebuilt_once = False

        while True:
            with self.driver.session() as session:
                current = session.run(
                    """
                    MATCH (e:Expedition {id: $expedition_id})
                    OPTIONAL MATCH (u:User)-[:OWNS]->(e)
                    RETURN e, u
                    """,
                    expedition_id=expedition_id,
                ).single()

                if not current or current["e"] is None:
                    return {
                        "expedition": {},
                        "user": {},
                        "nodes": [],
                        "links": [],
                        "learn_next": [],
                        "trends": [],
                        "forgotten_curve": [],
                        "related_expeditions": [],
                        "ready": False,
                        "message": "CFG not built yet",
                    }

                current_exp = _entity_to_dict(current["e"])
                current_user = _entity_to_dict(current["u"]) if current["u"] else {}

                current_id = str(current_exp.get("id") or expedition_id)
                current_title = _exp_label(current_exp)
                current_root = _exp_root(current_exp)

                current_concept_rows = session.run(
                    """
                    MATCH (e:Expedition {id: $expedition_id})-[r:HAS_CONCEPT]->(c:Concept)
                    RETURN c, r
                    ORDER BY coalesce(r.weight, 0) DESC, coalesce(c.usage_count, 0) DESC, c.name ASC
                    LIMIT 5
                    """,
                    expedition_id=expedition_id,
                ).data()

                # Auto-build if this expedition exists in PostgreSQL but has no CFG yet.
                if not current_concept_rows and not rebuilt_once:
                    try:
                        detail = self.expedition_repo.get_expedition_detail(expedition_id)
                        self.sync_from_detail(detail)
                        rebuilt_once = True
                        continue
                    except Exception:
                        pass

                current_concepts: List[Dict[str, Any]] = []
                for row in current_concept_rows:
                    c = _entity_to_dict(row["c"])
                    r = _entity_to_dict(row["r"])
                    label = str(c.get("name") or "").strip()
                    if not label:
                        continue
                    current_concepts.append(_make_concept_node(current_id, current_title, current_root, c, r))

                claim_rows = session.run(
                    """
                    MATCH (e:Expedition {id: $expedition_id})-[:HAS_CLAIM]->(cl:Claim)
                    RETURN cl
                    ORDER BY coalesce(cl.confidence, 0) DESC, cl.text ASC
                    LIMIT 5
                    """,
                    expedition_id=expedition_id,
                ).data()

                claims: List[Dict[str, Any]] = []
                for row in claim_rows:
                    cl = _entity_to_dict(row["cl"])
                    claim_text = str(cl.get("text") or "").strip()
                    if not claim_text:
                        continue
                    claim_id = str(cl.get("id") or f"{expedition_id}:claim:{_node_key(claim_text)}")
                    claims.append(
                        {
                            "id": f"claim:{_node_key(claim_id)}",
                            "label": claim_text,
                            "type": "claim",
                            "role": "claim",
                            "claim_id": claim_id,
                            "parent_expedition_id": current_id,
                            "parent_expedition_title": current_title,
                            "verdict": cl.get("verdict", ""),
                            "confidence": _safe_float(cl.get("confidence", 0.0), 0.0),
                            "score": _safe_float(cl.get("confidence", 0.0), 0.0),
                        }
                    )

                # Pull all other saved expeditions for the same user and show ONLY their parent nodes.
                current_user_email = str(
                    current_user.get("email")
                    or current_exp.get("user_email")
                    or "guest@mars.local"
                ).strip().lower()

                all_expeditions = self.expedition_repo.list_expeditions(current_user_email)
                related_expeditions: List[Dict[str, Any]] = []

                for exp in all_expeditions:
                    other_id = str(exp.get("id") or "").strip()
                    if not other_id or other_id == current_id:
                        continue

                    other_title = _exp_label(exp)
                    other_root = _exp_root(exp)

                    similarity_to_current = _score_pair(current_root, other_root)

                    related_expeditions.append(
                        {
                            "id": other_id,
                            "label": other_title,
                            "type": "expedition",
                            "role": "related",
                            "root_query": other_root,
                            "similarity_to_current": round(similarity_to_current, 6),
                            "score": round(similarity_to_current, 6),
                        }
                    )

                related_expeditions.sort(key=lambda x: x["similarity_to_current"], reverse=True)

                nodes: List[Dict[str, Any]] = []
                links: List[Dict[str, Any]] = []
                node_ids = set()

                current_expedition_node = {
                    "id": current_id,
                    "label": current_title,
                    "type": "expedition",
                    "role": "current",
                    "root_query": current_root,
                    "score": 1.0,
                }
                nodes.append(current_expedition_node)
                node_ids.add(current_expedition_node["id"])

                if current_user:
                    user_id = str(current_user.get("id") or current_user.get("email") or "user")
                    user_node = {
                        "id": user_id,
                        "label": current_user.get("display_name") or current_user.get("email") or "User",
                        "type": "user",
                        "role": "owner",
                    }
                    nodes.append(user_node)
                    node_ids.add(user_node["id"])
                    links.append(
                        {
                            "source": user_id,
                            "target": current_id,
                            "type": "OWNS",
                            "weight": 1.0,
                        }
                    )

                current_concept_lookup: Dict[str, Dict[str, Any]] = {}
                for concept in current_concepts:
                    nodes.append(concept)
                    node_ids.add(concept["id"])
                    current_concept_lookup[concept["label"]] = concept
                    links.append(
                        {
                            "source": current_id,
                            "target": concept["id"],
                            "type": "HAS_CONCEPT",
                            "weight": float(concept["score"] or 0.0),
                        }
                    )

                current_claim_lookup: Dict[str, Dict[str, Any]] = {}
                for claim in claims:
                    nodes.append(claim)
                    node_ids.add(claim["id"])
                    current_claim_lookup[claim["label"]] = claim
                    links.append(
                        {
                            "source": current_id,
                            "target": claim["id"],
                            "type": "HAS_CLAIM",
                            "weight": float(claim["confidence"] or 0.0),
                        }
                    )

                # Claims connect to current expedition's top 5 topics.
                claim_topic_counts: Dict[str, int] = {c["label"]: 0 for c in current_concepts}
                for claim in claims:
                    scored_topics: List[Tuple[Dict[str, Any], float]] = []
                    for concept in current_concepts:
                        rel_score = _score_pair(claim["label"], concept["label"])
                        scored_topics.append((concept, rel_score))

                    scored_topics.sort(key=lambda x: x[1], reverse=True)
                    for concept, rel_score in scored_topics[:2]:
                        if rel_score < 0.35:
                            continue
                        links.append(
                            {
                                "source": claim["id"],
                                "target": concept["id"],
                                "type": "CLAIM_TO_TOPIC",
                                "weight": round(rel_score, 6),
                            }
                        )
                        claim_topic_counts[concept["label"]] = claim_topic_counts.get(concept["label"], 0) + 1

                # Other expeditions appear as parent nodes only.
                for rel in related_expeditions:
                    rel_node = dict(rel)
                    nodes.append(rel_node)
                    node_ids.add(rel_node["id"])
                    links.append(
                        {
                            "source": current_id,
                            "target": rel_node["id"],
                            "type": "RELATED_RESEARCH",
                            "weight": rel_node["similarity_to_current"],
                        }
                    )

                # Topic depth / research consistency for the current expedition.
                trends = []
                max_claims = max(1, len(claims))
                max_related = max(1, len(related_expeditions))

                for concept in current_concepts:
                    related_links = sum(
                        1
                        for link in links
                        if link.get("type") == "RELATED_TOPIC"
                        and (
                            str(link.get("source")) == concept["id"]
                            or str(link.get("target")) == concept["id"]
                        )
                    )
                    claims_connected = int(claim_topic_counts.get(concept["label"], 0) or 0)
                    local_score = float(concept["score"] or 0.0)
                    depth_score = round(
                        (0.65 * local_score)
                        + (0.25 * (claims_connected / max_claims))
                        + (0.10 * (related_links / max_related)),
                        6,
                    )
                    trends.append(
                        {
                            "name": concept["label"],
                            "depth": depth_score,
                            "claims": claims_connected,
                            "relations": related_links,
                            "topic_score": round(local_score, 6),
                            "relevance_score": round(local_score, 6),
                        }
                    )

                trends.sort(key=lambda x: (x["depth"], x["claims"], x["relations"]), reverse=True)
                forgotten_curve = self._forgotten_curve(current_concepts)

                node_id_set = {n["id"] for n in nodes}
                links = [
                    link for link in links
                    if str(link.get("source")) in node_id_set and str(link.get("target")) in node_id_set
                ]

                return {
                    "expedition": _json_safe(current_exp),
                    "user": _json_safe(current_user),
                    "nodes": _json_safe(nodes),
                    "links": _json_safe(links),
                    "learn_next": _json_safe(
                        [
                            {
                                "name": rel["label"],
                                "score": rel["similarity_to_current"],
                            }
                            for rel in related_expeditions[:5]
                        ]
                    ),
                    "trends": _json_safe(trends[:5]),
                    "forgotten_curve": _json_safe(forgotten_curve),
                    "related_expeditions": _json_safe(related_expeditions),
                    "ready": True,
                    "message": "CFG ready",
                }

    def delete_expedition_graph(self, expedition_id: str) -> None:
        expedition_id = str(expedition_id or "").strip()
        if not expedition_id:
            return
        with self.driver.session() as session:
            session.run(
                """
                MATCH (n)
                WHERE n.expedition_id = $expedition_id
                DETACH DELETE n
                """,
                expedition_id=expedition_id,
            )
            session.run(
                """
                MATCH (e:Expedition {id: $expedition_id})
                DETACH DELETE e
                """,
                expedition_id=expedition_id,
            )

    def _forgotten_curve(self, concepts: List[Dict[str, Any]], half_life_days: int = 7) -> List[Dict[str, Any]]:
        base_activity = 1.0
        if concepts:
            base_activity = min(1.0, max(0.35, len(concepts) / 12.0))

        curve = []
        for day in [0, 1, 2, 3, 5, 7, 10, 14, 21, 30]:
            retention = base_activity * math.exp(-day / float(half_life_days))
            curve.append({"day": day, "retention": round(retention, 4)})
        return curve