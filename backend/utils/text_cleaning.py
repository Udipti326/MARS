# backend/utils/text_cleaning.py

from __future__ import annotations

import re
from html import unescape


WHITESPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://\S+")
HTML_TAG_RE = re.compile(r"<[^>]+>")


def clean_text(
    text: str,
    max_length: int = 2000,
    remove_urls: bool = False,
) -> str:
    """
    Cleans noisy text for LLM consumption.

    Features:
    - removes HTML tags
    - normalizes whitespace
    - optionally removes URLs
    - truncates long text safely
    """

    if not text:
        return ""

    if not isinstance(text, str):
        text = str(text)

    # HTML decode
    text = unescape(text)

    # remove html tags
    text = HTML_TAG_RE.sub(" ", text)

    # optionally remove urls
    if remove_urls:
        text = URL_RE.sub(" ", text)

    # normalize whitespace
    text = WHITESPACE_RE.sub(" ", text)

    text = text.strip()

    # truncate
    if max_length and len(text) > max_length:
        text = text[:max_length].rsplit(" ", 1)[0] + "..."

    return text