from __future__ import annotations

from typing import List
import requests
import config


class OllamaEmbedder:
    """
    Compute embeddings in-app via Ollama.

    Primary:   POST /api/embed        { "model": "...", "input": "..." }
    Fallback:  POST /api/embeddings   { "model": "...", "prompt": "..." }
    """

    def __init__(self) -> None:
        self.base_url = getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.model = getattr(config, "EMBEDDINGS_MODEL", "nomic-embed-text")

    def embed(self, text: str) -> List[float]:
        text = (text or "").strip()
        if not text:
            raise ValueError("Cannot embed empty text.")

        # Prefer newer endpoint
        try:
            return self._embed_via_api_embed(text)
        except requests.HTTPError as e:
            # If /api/embed isn't supported, fall back to /api/embeddings
            if e.response is not None and e.response.status_code == 404:
                return self._embed_via_api_embeddings(text)
            raise
        except (ValueError, KeyError, TypeError):
            # If payload shape is weird, try fallback once (older servers / proxies)
            return self._embed_via_api_embeddings(text)

    def _embed_via_api_embed(self, text: str) -> List[float]:
        url = f"{self.base_url}/api/embed"
        payload = {"model": self.model, "input": text}

        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()

        # Expected: { "embeddings": [[...]] }
        embs = data.get("embeddings")
        if not isinstance(embs, list) or not embs or not isinstance(embs[0], list) or not embs[0]:
            raise ValueError(f"Ollama /api/embed returned unexpected payload: {data}")

        return [float(x) for x in embs[0]]

    def _embed_via_api_embeddings(self, text: str) -> List[float]:
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model, "prompt": text}

        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()

        # Expected: { "embedding": [...] }
        vec = data.get("embedding")
        if not isinstance(vec, list) or not vec:
            raise ValueError(f"Ollama /api/embeddings returned unexpected payload: {data}")

        return [float(x) for x in vec]