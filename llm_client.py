# llm_client.py
# Hardened Ollama client + JSON helper (for planning/replanning)

import json
import time
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import OLLAMA_URL, MODEL_NAME, LLM_TIMEOUT_SECONDS


# -----------------------------
# Session + retries (keepalive)
# -----------------------------

_SESSION: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is not None:
        return _SESSION

    s = requests.Session()

    # Retries: ONLY for transient transport/server errors
    retry = Retry(
        total=2,                # small: prevents long retry loops
        connect=2,
        read=0,                 # don't auto-retry "read" by urllib3; we handle timeouts explicitly
        status=2,
        backoff_factor=0.4,     # 0.4s, 0.8s ...
        status_forcelist=(502, 503, 504),
        allowed_methods=("POST",),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    s.mount("http://", adapter)
    s.mount("https://", adapter)

    _SESSION = s
    return s


def _timeouts() -> tuple[float, float]:
    """
    Separate connect + read timeouts.

    - connect timeout: quick failure if Ollama is down (default 5s)
    - read timeout: how long we allow the model to think (LLM_TIMEOUT_SECONDS)
    """
    connect_timeout = 5.0
    read_timeout = float(LLM_TIMEOUT_SECONDS)
    return (connect_timeout, read_timeout)


def call_llm(prompt: str) -> str:
    """
    Send a prompt to the local model via Ollama and return raw text.
    """
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    session = _get_session()
    timeout = _timeouts()

    # A tiny manual retry on ReadTimeout/ConnectionError. This catches:
    # - Ollama restarting
    # - brief local socket hiccups
    # BUT avoids long loops (max 1 retry).
    last_err: Optional[Exception] = None
    for attempt in range(2):
        try:
            resp = session.post(OLLAMA_URL, json=payload, timeout=timeout)

            # If Ollama returns non-200, include body snippet (useful debugging)
            if resp.status_code >= 400:
                body = (resp.text or "").strip()
                snippet = body[:800] + ("..." if len(body) > 800 else "")
                raise RuntimeError(
                    f"Ollama HTTP {resp.status_code}. Body (truncated): {snippet}"
                )

            data = resp.json()
            text = str(data.get("response", "")).strip()
            if not text:
                raise RuntimeError(f"Ollama returned empty response. Full JSON: {data}")
            return text

        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            # transient-ish: quick retry
            time.sleep(0.25 * (attempt + 1))
            continue

        except requests.exceptions.ReadTimeout as e:
            # Model took too long. This is usually NOT recoverable by retrying immediately.
            # We still retry once in case the socket timed out but Ollama is fine.
            last_err = e
            if attempt == 0:
                time.sleep(0.25)
                continue
            raise RuntimeError(
                f"Ollama read timed out after {timeout[1]}s (model='{MODEL_NAME}'). "
                f"Consider lowering prompt size or increasing LLM_TIMEOUT_SECONDS."
            ) from e

        except Exception as e:
            # Anything else: fail fast
            raise

    # Should never get here
    raise RuntimeError(f"Ollama call failed after retries: {last_err!r}")


def call_llm_json(prompt: str) -> Any:
    """
    Call the LLM and parse a JSON response (planning/replanning).
    - Strips ```json fences if present
    - Extracts first JSON object/array if extra text slips in
    """
    raw = call_llm(prompt)
    text = raw.strip()

    # Strip markdown fences
    if "```" in text:
        first = text.find("```")
        last = text.rfind("```")
        if first != -1 and last != -1 and last > first:
            inner = text[first + 3 : last].lstrip()
            if inner.lower().startswith("json"):
                nl = inner.find("\n")
                if nl != -1:
                    inner = inner[nl + 1 :]
            text = inner.strip()

    # Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extract first {..} or [..]
    start_obj = text.find("{")
    start_arr = text.find("[")
    starts = [i for i in (start_obj, start_arr) if i != -1]
    if not starts:
        raise RuntimeError(f"LLM did not return JSON. Raw:\n{raw}")

    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    if end <= start:
        raise RuntimeError(f"LLM did not return JSON. Raw:\n{raw}")

    candidate = text[start : end + 1]
    return json.loads(candidate)