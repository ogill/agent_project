# llm_client.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import requests

try:
    from config import OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_TIMEOUT_S
except Exception:
    # Sensible defaults if config.py is incomplete during bring-up
    OLLAMA_HOST = "http://localhost:11434"
    OLLAMA_MODEL = "qwen2.5:7b-instruct"
    OLLAMA_TIMEOUT_S = 120


@dataclass
class LLMConfig:
    host: str = OLLAMA_HOST
    model: str = OLLAMA_MODEL
    timeout_s: int = OLLAMA_TIMEOUT_S


def call_llm(prompt: str, *, system: Optional[str] = None, cfg: Optional[LLMConfig] = None) -> str:
    """
    Minimal Ollama client.
    Returns raw model text (no streaming).
    """
    cfg = cfg or LLMConfig()
    url = cfg.host.rstrip("/") + "/api/generate"

    payload = {
        "model": cfg.model,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system

    try:
        resp = requests.post(url, json=payload, timeout=cfg.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        return (data.get("response") or "").strip()
    except Exception as e:
        # Keep error surface clean and actionable
        raise RuntimeError(f"call_llm failed (host={cfg.host}, model={cfg.model}): {e}") from e