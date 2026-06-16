from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq


def _load_env() -> None:
    candidates = [
        Path.cwd() / "backend" / ".env",
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
        Path(__file__).resolve().parents[1] / ".env",
    ]
    for path in candidates:
        if path.exists():
            load_dotenv(path, override=True)
            break


_load_env()


def _coerce_int(value: Optional[int | str], default: int) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def _coerce_float(value: Optional[float | str], default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


@lru_cache(maxsize=32)
def get_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    streaming: bool = False,
):
    """
    Shared Groq chat model factory.

    Existing code can keep calling get_llm().
    Chat-specific code can call get_llm(model_name="llama-3.3-70b-versatile").
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing")

    model = (model_name or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")).strip()
    temp = _coerce_float(
        temperature,
        _coerce_float(os.getenv("GROQ_TEMPERATURE", "0.2"), 0.2),
    )
    max_tok = _coerce_int(
        max_tokens,
        _coerce_int(os.getenv("GROQ_MAX_TOKENS", "1024"), 1024),
    )
    top_p_val = _coerce_float(
        top_p,
        _coerce_float(os.getenv("GROQ_TOP_P", "1.0"), 1.0),
    )

    kwargs = {
        "groq_api_key": api_key,
        "temperature": temp,
        "streaming": streaming,
    }
    if max_tok > 0:
        kwargs["max_tokens"] = max_tok
    if top_p_val is not None:
        kwargs["top_p"] = top_p_val

    # Different langchain_groq versions accept either `model` or `model_name`.
    try:
        return ChatGroq(model=model, **kwargs)
    except TypeError:
        return ChatGroq(model_name=model, **kwargs)