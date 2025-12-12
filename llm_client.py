# llm.py
# This isolates the LLM call so your agent logic stays clean.

import requests
from config import OLLAMA_URL, MODEL_NAME


def call_llm(prompt: str) -> str:
    """
    Send a prompt to the local model via Ollama and return the raw text response.

    This version is hardened:
    - Raises clear errors if the HTTP call fails.
    - Raises clear errors if JSON parsing fails.
    - Raises clear errors if the 'response' field is missing or empty.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    except requests.RequestException as e:
        raise RuntimeError(
            f"Error calling Ollama at {OLLAMA_URL} with model '{MODEL_NAME}': {e!r}"
        )

    # HTTP-level errors (non-2xx)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        # Include body for easier debugging
        raise RuntimeError(
            f"Ollama returned HTTP {resp.status_code}.\n"
            f"Response text:\n{resp.text}"
        ) from e

    # Try to parse JSON
    try:
        data = resp.json()
    except ValueError as e:
        raise RuntimeError(
            "Ollama response was not valid JSON.\n"
            f"Raw response text:\n{resp.text}"
        ) from e

    # Happy path: expect a 'response' field (Ollama's standard)
    raw = data.get("response")
    if raw is None:
        # No 'response' field – show the full JSON so you can see what's going on
        raise RuntimeError(
            "Ollama JSON had no 'response' field.\n"
            f"Full JSON payload:\n{data}"
        )

    text = str(raw).strip()
    if not text:
        # Empty string – again, show full JSON
        raise RuntimeError(
            "Ollama returned an empty 'response' string.\n"
            f"Full JSON payload:\n{data}"
        )

    return text