from langchain.agents import initialize_agent, AgentType
from backend.core.llm import get_llm
from backend.services.tools.tool_registry import ToolRegistry


class ToolAgent:

    def __init__(self):
        self.llm = get_llm()
        self.tools = ToolRegistry().get_tools()

        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True
        )

    def run(self, query: str):
        return self.agent.run(query)