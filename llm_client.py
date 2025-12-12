# llm_client.py
# Minimal hardened Ollama client + JSON helper (for planning/replanning)

import json
from typing import Any, Dict

import requests
from config import OLLAMA_URL, MODEL_NAME, LLM_TIMEOUT_SECONDS


def call_llm(prompt: str) -> str:
    """
    Send a prompt to the local model via Ollama and return raw text.
    """
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}

    resp = requests.post(OLLAMA_URL, json=payload, timeout=LLM_TIMEOUT_SECONDS)
    resp.raise_for_status()
    data = resp.json()

    text = str(data.get("response", "")).strip()
    if not text:
        raise RuntimeError(f"Ollama returned empty response. Full JSON: {data}")
    return text


def call_llm_json(prompt: str) -> Any:
    """
    Call the LLM and parse a JSON response (planning/replanning).
    - Strips ```json fences if present
    - Extracts first JSON object/array if extra text slips in
    """
    raw = call_llm(prompt)

    text = raw.strip()

    # Strip markdown fences
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            inner = text[first_nl + 1 :]
            fence_end = inner.rfind("```")
            if fence_end != -1:
                text = inner[:fence_end].strip()

    # Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extract first {..} or [..]
    start_obj = text.find("{")
    start_arr = text.find("[")
    starts = [i for i in [start_obj, start_arr] if i != -1]
    if not starts:
        raise RuntimeError(f"LLM did not return JSON. Raw:\n{raw}")

    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    if end <= start:
        raise RuntimeError(f"LLM did not return JSON. Raw:\n{raw}")

    candidate = text[start : end + 1]
    return json.loads(candidate)