# tools.py
#
# Tool registry for the agent (Pydantic v2 arg validation).
# NOTE: fetch_url is now a HARD-FAIL tool (raises on error) to enable replanning.

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict

import requests
from pydantic import BaseModel, Field

from llm_client import call_llm


# ---------------------------------------------------------------------
# Args models (validated in agent._call_tool via model_validate)
# ---------------------------------------------------------------------


class AlwaysFailArgs(BaseModel):
    reason: str = Field(default="forced failure for replanning test")


class GetTimeArgs(BaseModel):
    city: str


class GetWeatherArgs(BaseModel):
    city: str


class FetchUrlArgs(BaseModel):
    url: str


class SummarizeTextArgs(BaseModel):
    text: str
    bullets: int = Field(default=3, ge=1, le=10)


# ---------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------


def tool_always_fail(args: AlwaysFailArgs) -> str:
    # Hard fail by design
    raise RuntimeError(f"Intentional failure triggered: {args.reason}")


def tool_get_time(args: GetTimeArgs) -> str:
    # Stubbed (UTC). You can later swap for a real timezone lookup.
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"The current time in {args.city} is {now_utc} (stubbed response)."


def tool_get_weather(args: GetWeatherArgs) -> str:
    # Stubbed
    return f"Weather in {args.city}: 12°C, light cloud (stubbed response)."


def tool_fetch_url(args: FetchUrlArgs) -> str:
    """
    Fetch the raw text/HTML from a URL.
    HARD-FAIL behaviour:
      - raises RuntimeError on network errors / non-2xx HTTP responses
      - this allows the agent to trigger replanning
    Returns up to ~4k chars to avoid huge payloads.
    """
    url = (args.url or "").strip()
    if not url:
        raise ValueError("'url' argument is required.")

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        # HARD FAIL (important for replanning)
        raise RuntimeError(f"Failed to fetch URL '{url}': {e!r}") from e

    content = resp.text or ""
    max_chars = 4000
    if len(content) > max_chars:
        content = content[:max_chars] + "\n...[truncated]"

    return f"Fetched content from {url}:\n\n{content}"


def tool_summarize_text(args: SummarizeTextArgs) -> str:
    """
    Summarise arbitrary text into bullet points and highlight risks using the LLM.
    Kept simple and deterministic in format.
    """
    bullets = args.bullets
    text = args.text.strip()

    prompt = (
        "You are a concise summarizer.\n\n"
        f"Summarize the following text into exactly {bullets} bullet points.\n"
        "Then add a final line starting with 'Risks:' listing any risks/concerns, or 'Risks: none'.\n\n"
        "Text:\n"
        f"{text}\n"
    )

    out = call_llm(prompt).strip()
    return f"Summary ({bullets} bullet points, model={__import__('config').MODEL_NAME}):\n\n{out}"


# ---------------------------------------------------------------------
# Tool registry (consumed by planner + agent)
# ---------------------------------------------------------------------

TOOLS: Dict[str, Dict[str, Any]] = {
    "always_fail": {
        "description": "Always fails intentionally to test dynamic replanning.",
        "args_model": AlwaysFailArgs,
        "fn": tool_always_fail,
    },
    "get_time": {
        "description": "Return the current time in a specified city (stubbed).",
        "args_model": GetTimeArgs,
        "fn": tool_get_time,
    },
    "get_weather": {
        "description": "Return stubbed weather information for a specified city.",
        "args_model": GetWeatherArgs,
        "fn": tool_get_weather,
    },
    "fetch_url": {
        "description": "Fetch raw content from a given URL (HTML/text, truncated). (Hard-fail on errors)",
        "args_model": FetchUrlArgs,
        "fn": tool_fetch_url,
    },
    "summarize_text": {
        "description": "Summarise arbitrary text into bullet points and highlight risks using the LLM.",
        "args_model": SummarizeTextArgs,
        "fn": tool_summarize_text,
    },
}