from yt_dlp import YoutubeDL
from .base_tool import BaseTool
from .normalizer import normalize_source


class YouTubeTool(BaseTool):

    def search(self, query, max_results=5):
        ydl_opts = {
            "quiet": True,
            "skip_download": True
        }

        results = []

        try:
            with YoutubeDL(ydl_opts) as ydl:
                search_query = f"ytsearch{max_results}:{query}"
                info = ydl.extract_info(search_query, download=False)

                for entry in info.get("entries", []):
                    results.append(
                        normalize_source(
                            title=entry.get("title", ""),
                            content=entry.get("description", ""),
                            url=entry.get("webpage_url", ""),
                            source_type="youtube",
                            domain="youtube.com",
                            authors=[entry.get("channel", "")]
                        )
                    )
        except:
            return []

        return results