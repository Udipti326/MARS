from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


class WikiTool:

    def __init__(self):
        self.wrapper = WikipediaAPIWrapper(
            top_k_results=2,
            doc_content_chars_max=1000
        )
        self.tool = WikipediaQueryRun(api_wrapper=self.wrapper)

    def get_tool(self):
        return self.tool