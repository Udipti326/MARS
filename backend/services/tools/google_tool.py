import requests
from bs4 import BeautifulSoup
from .base_tool import BaseTool
from .normalizer import normalize_source


class GoogleTool(BaseTool):

    def search(self, query, max_results=5):
        url = "https://duckduckgo.com/html/"
        params = {"q": query}

        try:
            res = requests.post(url, data=params, timeout=10)
            soup = BeautifulSoup(res.text, "html.parser")
        except:
            return []

        results = []

        for result in soup.select(".result")[:max_results]:
            title_el = result.select_one(".result__title")
            snippet_el = result.select_one(".result__snippet")
            link_el = result.select_one("a")

            if not title_el or not link_el:
                continue

            results.append(
                normalize_source(
                    title=title_el.get_text(),
                    content=snippet_el.get_text() if snippet_el else "",
                    url=link_el["href"],
                    source_type="web"
                )
            )

        return results