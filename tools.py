# tools.py

import datetime
from typing import Dict, Any
import textwrap

import requests

from config import MODEL_NAME
from llm_client import call_llm
from schemas import (
    GetTimeArgs,
    GetWeatherArgs,
    FetchUrlArgs,
    SummarizeTextArgs,
)


# === Existing simple tools ===


def tool_get_time(args: GetTimeArgs) -> str:
    """
    Return the current time in a given city (stubbed; does not use real time zones).

    Args:
        args: GetTimeArgs Pydantic model with:
            - city: str
    """
    city = args.city
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    return f"The current time in {city} is {now} UTC (stubbed response)."


def tool_get_weather(args: GetWeatherArgs) -> str:
    """
    Return the current weather for a given city (stubbed).

    Args:
        args: GetWeatherArgs Pydantic model with:
            - city: str
    """
    city = args.city
    # This is deliberately stubbed; you can later integrate a real weather API.
    return f"The weather in {city} is SuNnY and 22°C (stubbed response)."


# === Web + summarisation tools ===


def tool_fetch_url(args: FetchUrlArgs) -> str:
    """
    Fetch the raw text/HTML from a URL.

    Args:
        args: FetchUrlArgs Pydantic model with:
            - url: str

    This uses requests.get with a short timeout and returns the first
    few KB of content to avoid huge payloads.
    """
    url = args.url
    if not url:
        return "Error: 'url' argument is required."

    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
    except Exception as e:
        return f"Error fetching URL '{url}': {e!r}"

    # Truncate to avoid overwhelming the LLM in subsequent steps
    content = resp.text
    max_chars = 4000
    if len(content) > max_chars:
        content = content[:max_chars] + "\n...[truncated]"

    return f"Fetched content from {url}:\n\n{content}"


def tool_summarize_text(args: SummarizeTextArgs) -> str:
    """
    Summarise arbitrary text into a small number of bullet points,
    highlighting any risks.

    Args:
        args: SummarizeTextArgs Pydantic model with:
            - text: str
            - bullets: int (default 3)

    This tool uses the same LLM (MODEL_NAME) via call_llm.
    """
    text = args.text
    bullets = args.bullets

    if not text:
        return "Error: 'text' argument is required for summarize_text."

    prompt = textwrap.dedent(
        f"""
        You are a concise summarisation assistant.

        Summarise the following text into {bullets} bullet points.
        Additionally, highlight any risks or concerns that appear in the text.

        Text:
        {text}
        """
    ).strip()

    summary = call_llm(prompt)
    return f"Summary ({bullets} bullet points, model={MODEL_NAME}):\n\n{summary}"


# === Tool registry ===
# Note: args_model holds the Pydantic schema; fn expects an instance of that model.

TOOLS: Dict[str, Dict[str, Any]] = {
    "get_time": {
        "fn": tool_get_time,
        "args_model": GetTimeArgs,
        "description": "Return the current time in a specified city (stubbed).",
    },
    "get_weather": {
        "fn": tool_get_weather,
        "args_model": GetWeatherArgs,
        "description": "Return stubbed weather information for a specified city.",
    },
    "fetch_url": {
        "fn": tool_fetch_url,
        "args_model": FetchUrlArgs,
        "description": "Fetch raw content from a given URL (HTML/text, truncated).",
    },
    "summarize_text": {
        "fn": tool_summarize_text,
        "args_model": SummarizeTextArgs,
        "description": "Summarise arbitrary text into bullet points and highlight risks using the LLM.",
    },
}