from langchain_tavily import TavilySearch, TavilyExtract


class TavilyTools:

    def __init__(self):
        self.search = TavilySearch(
            max_results=5,
            topic="general",
            include_raw_content=True
        )

        self.extract = TavilyExtract(
            extract_depth="basic"
        )

    def search_tool(self):
        return self.search

    def extract_tool(self):
        return self.extract