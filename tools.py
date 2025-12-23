# tools.py

from __future__ import annotations

import datetime
import urllib.request
from typing import Any, Dict

from pydantic import BaseModel, Field

from config import MCP_ENABLED, MCP_SERVERS
from mcp.provider import McpProvider
from schemas import (
    AlwaysFailArgs,
    FetchUrlArgs,
    GetTimeArgs,
    GetWeatherArgs,
)


def _fetch_url(url: str) -> str:
    """
    Fetch raw text from a URL. Hard-fails on network/DNS/HTTP errors.
    Returns truncated content to keep prompts small.
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "agent_project/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read()
        text = raw.decode("utf-8", errors="replace")
        return text[:4000]
    except Exception as e:
        raise RuntimeError(f"Failed to fetch URL '{url}': {e!r}") from e


def always_fail(reason: str = "forced failure for replanning test") -> str:
    raise RuntimeError(reason)


def soft_fail(reason: str = "soft failure for replanning test", retryable: bool = False) -> Dict[str, Any]:
    """
    Soft-failure tool: does NOT raise.
    Returns a structured failure payload so the Agent can trigger replanning
    without relying on exceptions.
    """
    return {
        "ok": False,
        "status": "failed",
        "reason": reason,
        "retryable": retryable,
    }


def get_time(city: str) -> str:
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"Time in {city}: {now} (stubbed)"


def get_weather(city: str) -> str:
    return f"Weather in {city}: 18C, clear (stubbed)"


def fetch_url(url: str) -> str:
    return _fetch_url(url)


# --- Inline schema for soft_fail (keeps change isolated; no need to edit schemas.py yet) ---
class SoftFailArgs(BaseModel):
    reason: str = Field(default="soft failure for replanning test")
    retryable: bool = Field(default=False)


TOOLS: Dict[str, Dict[str, Any]] = {
    "always_fail": {
        "description": "Always fails intentionally to test dynamic replanning.",
        "args_model": AlwaysFailArgs,
        "fn": lambda **kwargs: always_fail(**AlwaysFailArgs(**kwargs).model_dump()),
    },
    "soft_fail": {
        "description": "Returns a structured failure payload (no exception) to test soft-failure replanning.",
        "args_model": SoftFailArgs,
        "fn": lambda **kwargs: soft_fail(**SoftFailArgs(**kwargs).model_dump()),
    },
    "get_time": {
        "description": "Return the current time in a specified city (stubbed).",
        "args_model": GetTimeArgs,
        "fn": lambda **kwargs: get_time(**GetTimeArgs(**kwargs).model_dump()),
    },
    "get_weather": {
        "description": "Return stubbed weather information for a specified city.",
        "args_model": GetWeatherArgs,
        "fn": lambda **kwargs: get_weather(**GetWeatherArgs(**kwargs).model_dump()),
    },
    "fetch_url": {
        "description": "Fetch raw content from a given URL (HTML/text, truncated). (Hard-fail on errors)",
        "args_model": FetchUrlArgs,
        "fn": lambda **kwargs: fetch_url(**FetchUrlArgs(**kwargs).model_dump()),
    },
}

# --- MCP tools (Stage 7) ---
if MCP_ENABLED:
    _mcp_provider = McpProvider.from_config(enabled=True, servers_cfg=MCP_SERVERS)
    if _mcp_provider is not None:
        TOOLS.update(_mcp_provider.get_tools_dict())