from __future__ import annotations

from langchain_core.prompts import PromptTemplate

from backend.core.llm import get_llm
from backend.utils.parser import safe_json_parse


class SummarizerAgent:
    def __init__(self, temperature: float = 0.0):
        self.llm = get_llm(temperature=temperature)

    def run(self, query: str, source_bundle: str):
        prompt = PromptTemplate.from_template(
            """
You are a research summarizer.

You are given a research query and a structured source bundle.

IMPORTANT:
- Focus only on information relevant to the query.
- Ignore unrelated content.
- Return ONLY valid JSON.
- No markdown. No explanations.

Query:
{query}

Source Bundle:
{context}

Return this JSON:
{{
  "summary": "short summary of the relevant information",
  "claims": ["claim 1", "claim 2", "claim 3"]
}}
"""
        )

        response = (prompt | self.llm).invoke(
            {
                "query": query,
                "context": source_bundle,
            }
        )

        output_text = response.content if hasattr(response, "content") else str(response)
        return safe_json_parse(output_text)