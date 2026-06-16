from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper


class ArxivTool:

    def __init__(self):
        self.wrapper = ArxivAPIWrapper(
            top_k_results=3,
            doc_content_chars_max=1500
        )
        self.tool = ArxivQueryRun(api_wrapper=self.wrapper)

    def get_tool(self):
        return self.tool