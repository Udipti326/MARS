from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.prompts import PromptTemplate

from backend.core.llm import get_llm
from backend.utils.parser import safe_json_parse


def _normalize_edges(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []

    for edge in edges:
        if not isinstance(edge, dict):
            continue

        prerequisite = str(edge.get("prerequisite") or edge.get("from") or edge.get("source") or "").strip()
        dependent = str(edge.get("dependent") or edge.get("to") or edge.get("target") or "").strip()
        if not prerequisite or not dependent:
            continue

        relation_type = str(edge.get("relation_type") or "prerequisite").strip().lower()
        if relation_type not in {"prerequisite", "related", "advanced_to"}:
            relation_type = "prerequisite"

        key = (prerequisite.lower(), dependent.lower(), relation_type)
        if key in seen:
            continue
        seen.add(key)

        out.append(
            {
                "prerequisite": prerequisite,
                "dependent": dependent,
                "relation_type": relation_type,
                "confidence": round(max(0.0, min(1.0, float(edge.get("confidence", 0.7) or 0.7))), 3),
                "reason": str(edge.get("reason") or edge.get("notes") or "").strip(),
            }
        )

    return out


def _normalize_nodes(concepts: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    seen = set()
    for concept in concepts or []:
        if not isinstance(concept, dict):
            continue
        name = str(concept.get("normalized_name") or concept.get("name") or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
    return out


class DependencyAgent:
    """
    Infers prerequisite and related-topic dependencies from extracted concepts.

    Use this after ConceptExtractorAgent. It returns graph-ready edges and a roadmap.
    """

    def __init__(self, temperature: float = 0.0):
        self.llm = get_llm(temperature=temperature)

    def run(
        self,
        concepts: List[Dict[str, Any]],
        topic: str = "",
        existing_nodes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        existing_nodes = existing_nodes or []
        concept_names = _normalize_nodes(concepts)

        prompt = PromptTemplate.from_template(
            """
You are a dependency inference agent for a learning roadmap system.

Return ONLY valid JSON.
No markdown, no explanation, no code fences.

Topic:
{topic}

Concepts:
{concepts}

Existing knowledge nodes:
{existing_nodes}

Your job:
- infer prerequisite relationships
- infer related concepts
- create a simple learning roadmap
- identify missing prerequisites if obvious

Return this JSON schema:
{{
  "topic": "{topic}",
  "dependencies": [
    {{
      "prerequisite": "...",
      "dependent": "...",
      "relation_type": "prerequisite|related|advanced_to",
      "confidence": 0.0,
      "reason": "..."
    }}
  ],
  "roadmap": [
    "...",
    "..."
  ],
  "missing_prerequisites": ["..."],
  "notes": "short explanation"
}}
"""
        )

        response = (prompt | self.llm).invoke(
            {
                "topic": topic.strip() or "unspecified",
                "concepts": "\n".join(f"- {x}" for x in concept_names) or "- none",
                "existing_nodes": "\n".join(f"- {x}" for x in existing_nodes) or "- none",
            }
        )

        data = safe_json_parse(response.content if hasattr(response, "content") else str(response))
        if not isinstance(data, dict):
            raise ValueError("DependencyAgent output must be a JSON object")

        dependencies = data.get("dependencies", [])
        if not isinstance(dependencies, list):
            dependencies = []

        roadmap = data.get("roadmap", [])
        if not isinstance(roadmap, list):
            roadmap = []

        missing_prerequisites = data.get("missing_prerequisites", [])
        if not isinstance(missing_prerequisites, list):
            missing_prerequisites = []

        data["topic"] = topic.strip() or data.get("topic", "unspecified")
        data["concepts"] = concept_names
        data["existing_nodes"] = existing_nodes
        data["dependencies"] = _normalize_edges(dependencies)
        data["roadmap"] = [str(x).strip() for x in roadmap if str(x).strip()]
        data["missing_prerequisites"] = [str(x).strip() for x in missing_prerequisites if str(x).strip()]
        return data