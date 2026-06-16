from langchain_core.prompts import PromptTemplate

from backend.core.llm import get_llm
from backend.utils.parser import safe_json_parse

llm = get_llm()


class JudgeAgent:

    def run(
        self,
        claim,
        support_data,
        skeptic_data,
        evidence_context="",
        query="",
        **kwargs,
    ):

        prompt = PromptTemplate.from_template(
            """
You are the judge agent in a multi-agent research system.

Return ONLY valid JSON.

Original Query:
{query}

Claim:
{claim}

Support Evidence:
{support}

Skeptic Analysis:
{skeptic}

Additional Evidence Context:
{evidence_context}

Return JSON:

{{
  "verdict": "True | Likely True | Mixed | Uncertain | False",
  "reasoning": "...",
  "confidence": 0.0
}}
"""
        )

        chain = prompt | llm

        response = chain.invoke(
            {
                "query": query,
                "claim": claim,
                "support": support_data,
                "skeptic": skeptic_data,
                "evidence_context": evidence_context,
            }
        )

        return safe_json_parse(response.content)