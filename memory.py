from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set

import config

# Optional Stage 6 semantic memory imports (safe if present)
try:
    from semantic_memory import make_semantic_store
    from semantic_memory.embedder import OllamaEmbedder
except Exception:  # pragma: no cover
    make_semantic_store = None  # type: ignore
    OllamaEmbedder = None  # type: ignore


@dataclass
class Episode:
    id: str
    timestamp: float
    user: str
    assistant: str


class EpisodeStore:
    """
    Episodic memory = append-only log (ground truth).
    Semantic memory (Stage 6) = vector index used to retrieve relevant episode IDs.

    Key behaviour:
    - On retrieval we FILTER episodes containing failures unless the user is explicitly
      asking about failures/debugging (prevents "old failure cause" bleed-through).
    """

    _FAILED_TOOL_RE = re.compile(r"tool\s+`?([a-zA-Z0-9_\-]+)`?\s+failed", re.IGNORECASE)

    def __init__(self) -> None:
        self.memory_dir = self._get_memory_dir()
        self.episodes_path = self._get_episodes_path()

        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # Stage 6 (optional): semantic index
        self.semantic_store = make_semantic_store() if make_semantic_store else None
        self.embedder = OllamaEmbedder() if OllamaEmbedder else None

    # ---------------- public API ----------------

    def append(self, user: str, assistant: str) -> None:
        ep = Episode(
            id=self._new_episode_id(),
            timestamp=time.time(),
            user=user,
            assistant=assistant,
        )

        self._append_episode_jsonl(ep)

        # Best-effort semantic upsert (never crash agent/tests)
        if self.semantic_store is not None and self.embedder is not None:
            try:
                text = self._episode_text(ep)
                vec = self.embedder.embed(text)

                meta = self._semantic_metadata_for_episode(ep)
                self.semantic_store.upsert(id=ep.id, vector=vec, text=text, metadata=meta)
            except Exception as e:
                print(f"[MEMORY DEBUG] Semantic upsert failed (ignored): {e}")

    def build_context(self, user_input: str, *, max_recent: int | None = None) -> str:
        episodes = self._load_episodes()
        if not episodes:
            return ""

        if max_recent is None:
            max_recent = int(getattr(config, "MAX_RECENT_EPISODES", 6))

        recent = episodes[-max_recent:]

        # Should we allow failure episodes in memory context?
        allow_failure_hits = bool(getattr(config, "SEMANTIC_INCLUDE_FAILURE_HITS", False))
        wants_debug = self._user_wants_debug(user_input)
        filter_failures = (not allow_failure_hits) and (not wants_debug)

        # Stage 6 semantic retrieval (optional)
        semantic_ids: List[str] = []
        top_k = int(getattr(config, "SEMANTIC_TOP_K", 5))
        min_score = float(getattr(config, "SEMANTIC_MIN_SCORE", 0.0))

        if self.semantic_store is not None and self.embedder is not None:
            try:
                qvec = self.embedder.embed(user_input)
                hits = self.semantic_store.query(vector=qvec, k=top_k)

                # score threshold
                hits = [h for h in hits if h.score >= min_score]

                # failure-hit filtering (prevents “wrong old failure reason” contamination)
                if filter_failures:
                    hits = self._filter_semantic_hits(hits)

                if bool(getattr(config, "SEMANTIC_DEBUG", False)):
                    print(f"[MEMORY DEBUG] Filtered semantic hit ids: {[h.id for h in hits]}")

                semantic_ids = [h.id for h in hits]
            except Exception as e:
                print(f"[MEMORY DEBUG] Semantic query failed (ignored): {e}")

        id_to_ep = {ep.id: ep for ep in episodes}

        ordered: List[Episode] = []
        seen: Set[str] = set()

        # semantic first (relevance)
        for eid in semantic_ids:
            ep = id_to_ep.get(eid)
            if ep and ep.id not in seen:
                if (not filter_failures) or (not self._episode_looks_like_failure(ep)):
                    ordered.append(ep)
                    seen.add(ep.id)

        # then recent (recency) — IMPORTANT: filter failures here too
        for ep in recent:
            if ep.id in seen:
                continue
            if filter_failures and self._episode_looks_like_failure(ep):
                continue
            ordered.append(ep)
            seen.add(ep.id)

        # Render
        lines: List[str] = []
        for ep in ordered:
            lines.append(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ep.timestamp))}]")
            lines.append(f"User: {ep.user}")
            lines.append(f"Assistant: {ep.assistant}")
            lines.append("")

        return "\n".join(lines).strip()

    # ---------------- filtering / heuristics ----------------

    def _user_wants_debug(self, user_input: str) -> bool:
        q = (user_input or "").lower()
        return any(
            kw in q
            for kw in [
                "failed", "failure", "error", "exception", "why did", "what happened",
                "http", "status code", "404", "timeout", "dns", "url", "fetch",
            ]
        )

    def _filter_semantic_hits(self, hits: List[Any]) -> List[Any]:
        """
        Remove semantic hits that are primarily about failures.
        Works even with older indexed entries (no metadata) by falling back to text heuristics.
        """
        filtered: List[Any] = []
        for h in hits:
            meta = getattr(h, "metadata", {}) or {}
            text = (getattr(h, "text", "") or "").lower()

            has_failure_meta = bool(meta.get("has_failure", False))
            has_failure_text = (" tool " in text and " failed" in text) or text.startswith("i couldn’t complete")

            if has_failure_meta or has_failure_text:
                continue

            filtered.append(h)
        return filtered

    def _episode_looks_like_failure(self, ep: Episode) -> bool:
        a = (ep.assistant or "")
        lower = a.lower()

        # Tool failure string your agent generates
        if "tool" in lower and "failed" in lower:
            return True

        # Fallback answer prefix
        if lower.startswith("i couldn’t complete"):
            return True

        # Explicit tool name detection
        if self._FAILED_TOOL_RE.search(a):
            return True

        return False

    def _semantic_metadata_for_episode(self, ep: Episode) -> Dict[str, Any]:
        assistant = (ep.assistant or "")
        lower = assistant.lower()

        failed_tools = sorted(set(self._FAILED_TOOL_RE.findall(assistant)))
        has_failure = False

        if failed_tools:
            has_failure = True
        elif (" tool " in lower and " failed" in lower) or lower.startswith("i couldn’t complete"):
            has_failure = True

        return {
            "timestamp": float(ep.timestamp),
            "has_failure": bool(has_failure),
            "failed_tools": ", ".join(failed_tools),  # IMPORTANT: Chroma metadata must be scalar
        }

    # ---------------- internals ----------------

    def _get_memory_dir(self) -> Path:
        md = getattr(config, "MEMORY_DIR", Path(".memory"))
        return md if isinstance(md, Path) else Path(md)

    def _get_episodes_path(self) -> Path:
        ep = getattr(config, "EPISODES_PATH", None)
        if ep is not None:
            return ep if isinstance(ep, Path) else Path(ep)

        md = self._get_memory_dir()
        fname = getattr(config, "EPISODES_FILE", "episodes.jsonl")
        return md / str(fname)

    def _new_episode_id(self) -> str:
        return f"ep_{int(time.time() * 1000)}"

    def _append_episode_jsonl(self, ep: Episode) -> None:
        record = {"id": ep.id, "timestamp": ep.timestamp, "user": ep.user, "assistant": ep.assistant}
        self.episodes_path.parent.mkdir(parents=True, exist_ok=True)
        with self.episodes_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_episodes(self) -> List[Episode]:
        if not self.episodes_path.exists():
            return []
        out: List[Episode] = []
        with self.episodes_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    out.append(
                        Episode(
                            id=str(obj.get("id", "")),
                            timestamp=float(obj.get("timestamp", 0.0)),
                            user=str(obj.get("user", "")),
                            assistant=str(obj.get("assistant", "")),
                        )
                    )
                except Exception:
                    continue
        return out

    def _episode_text(self, ep: Episode) -> str:
        return f"User: {ep.user}\nAssistant: {ep.assistant}"