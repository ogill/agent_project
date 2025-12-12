# tools.py

import datetime
import textwrap
from typing import Any, Dict

import requests

from config import MODEL_NAME
from llm_client import call_llm
from schemas import (
    AlwaysFailArgs,
    FetchUrlArgs,
    GetTimeArgs,
    GetWeatherArgs,
    SummarizeTextArgs,
)


# ---------------------------------------------------------------------
# Test tool for replanning
# ---------------------------------------------------------------------

def tool_always_fail(args: AlwaysFailArgs) -> str:
    raise RuntimeError(f"Intentional failure triggered: {args.reason}")


# ---------------------------------------------------------------------
# Simple stub tools
# ---------------------------------------------------------------------

def tool_get_time(args: GetTimeArgs) -> str:
    """
    Return the current time in a given city (stubbed; uses UTC, not real time zones).
    """
    city = args.city
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    return f"The current time in {city} is {now} UTC (stubbed response)."


def tool_get_weather(args: GetWeatherArgs) -> str:
    """
    Return the current weather for a given city (stubbed).
    """
    city = args.city
    return f"The weather in {city} is SuNnY and 22°C (stubbed response)."


# ---------------------------------------------------------------------
# Web + summarisation tools
# ---------------------------------------------------------------------

def tool_fetch_url(args: FetchUrlArgs) -> str:
    """
    Fetch the raw text/HTML from a URL.
    Returns up to ~4k chars to avoid huge payloads.
    """
    url = args.url
    if not url:
        return "Error: 'url' argument is required."

    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
    except Exception as e:
        return f"Error fetching URL '{url}': {e!r}"

    content = resp.text
    max_chars = 4000
    if len(content) > max_chars:
        content = content[:max_chars] + "\n...[truncated]"

    return f"Fetched content from {url}:\n\n{content}"


def tool_summarize_text(args: SummarizeTextArgs) -> str:
    """
    Summarise text into N bullet points, highlighting risks.
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


# ---------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------

TOOLS: Dict[str, Dict[str, Any]] = {
    "always_fail": {
        "fn": tool_always_fail,
        "description": "Always fails intentionally to test dynamic replanning.",
        "args_model": AlwaysFailArgs,
    },
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