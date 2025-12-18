from __future__ import annotations

import config
from .base import SemanticStore


def make_semantic_store() -> SemanticStore | None:
    backend = getattr(config, "SEMANTIC_BACKEND", "none")
    backend = str(backend).lower().strip()

    if backend in {"none", "off", ""}:
        return None

    if backend == "chroma":
        try:
            from .chroma_store import ChromaStore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "SEMANTIC_BACKEND is set to 'chroma' but chroma dependencies "
                "could not be imported. Is chromadb installed?"
            ) from e

        return ChromaStore(collection="episodes")

    raise ValueError(
        f"Unknown SEMANTIC_BACKEND: {backend!r}. "
        "Valid values are: 'none', 'off', 'chroma'."
    )