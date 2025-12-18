# chroma_store.py

from __future__ import annotations

import json
from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings

import config
from .base import SemanticHit, SemanticStore


class ChromaStore(SemanticStore):
    """
    Persistent Chroma-backed vector store.
    Stores: id, embedding vector, text, metadata

    Notes:
    - Chroma metadata values must be primitive types (str/int/float/bool/None).
      Lists/dicts will raise, so we sanitize them (JSON-stringify).
    - Chroma returns distances (smaller = more similar). We convert to a loose similarity score.
    - We keep the raw distance on the hit metadata as '_distance' so callers can debug/filter.
    - Debug printing is controlled by config.SEMANTIC_DEBUG (default False).
    """

    def __init__(self, *, collection: str = "episodes") -> None:
        chroma_dir = getattr(config, "CHROMA_DIR", None)
        if chroma_dir is None:
            raise ValueError("CHROMA_DIR is not set in config.")

        path = str(chroma_dir)

        self.client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.col = self.client.get_or_create_collection(name=collection)

    # --------------------------- metadata sanitization ---------------------------

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Chroma requires metadata values to be: str, int, float, bool, or None.
        Convert lists/dicts to JSON strings and unknown objects to str().
        """
        out: Dict[str, Any] = {}

        for k, v in (metadata or {}).items():
            key = str(k)

            # Fast path: already valid primitives
            if v is None or isinstance(v, (str, int, float, bool)):
                out[key] = v
                continue

            # Lists/tuples/sets -> JSON string
            if isinstance(v, (list, tuple, set)):
                try:
                    out[key] = json.dumps(list(v), ensure_ascii=False)
                except Exception:
                    out[key] = str(list(v))
                continue

            # Dict -> JSON string (sanitize nested values by best-effort JSON encoding)
            if isinstance(v, dict):
                try:
                    out[key] = json.dumps(v, ensure_ascii=False, default=str)
                except Exception:
                    out[key] = str(v)
                continue

            # Anything else -> string
            out[key] = str(v)

        return out

    # --------------------------- store API ---------------------------

    def upsert(self, *, id: str, vector: List[float], text: str, metadata: Dict[str, Any]) -> None:
        safe_meta = self._sanitize_metadata(metadata)

        self.col.upsert(
            ids=[id],
            embeddings=[vector],
            documents=[text],
            metadatas=[safe_meta],
        )

    def query(self, *, vector: List[float], k: int) -> List[SemanticHit]:
        res = self.col.query(
            query_embeddings=[vector],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        hits: List[SemanticHit] = []
        n = min(len(ids), len(docs), len(dists)) if ids else 0

        for i in range(n):
            dist = float(dists[i]) if i < len(dists) else 999.0
            score = 1.0 / (1.0 + dist)

            meta: Dict[str, Any] = {}
            if i < len(metas) and metas[i]:
                meta = dict(metas[i])

            # Preserve raw distance for downstream filtering/debugging
            meta["_distance"] = dist

            hits.append(
                SemanticHit(
                    id=str(ids[i]),
                    score=score,
                    text=str(docs[i]) if i < len(docs) else "",
                    metadata=meta,
                )
            )

        hits.sort(key=lambda h: h.score, reverse=True)

        if bool(getattr(config, "SEMANTIC_DEBUG", False)):
            print(f"[CHROMA DEBUG] Hits: {hits}")

        return hits