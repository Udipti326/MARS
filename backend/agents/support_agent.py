from __future__ import annotations

from langchain_core.prompts import PromptTemplate

from backend.core.llm import get_llm
from backend.utils.parser import safe_json_parse


class SupportAgent:
    def __init__(self, temperature: float = 0.0):
        self.llm = get_llm(temperature=temperature)

    def run(self, claim: str, source_bundle: str, query: str = ""):
        prompt = PromptTemplate.from_template(
            """
You are a support agent.

Your job is to find evidence that SUPPORTS the claim using the source bundle.
Return ONLY valid JSON.
No markdown. No explanations.

Query:
{query}

Claim:
{claim}

Source Bundle:
{context}

Return this JSON:
{{
  "support_points": ["..."],
  "support_chunks": [
    {{
      "text": "...",
      "similarity": 0.0
    }}
  ]
}}
"""
        )

        response = (prompt | self.llm).invoke(
            {
                "query": query or "unspecified",
                "claim": claim,
                "context": source_bundle,
            }
        )

        output_text = response.content if hasattr(response, "content") else str(response)
        return safe_json_parse(output_text)