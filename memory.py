# memory.py

from __future__ import annotations
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List

import config


@dataclass
class Episode:
    ts: str
    user: str
    assistant: str


class EpisodeStore:
    """
    Episodic memory persisted to JSONL:
    one JSON object per line: {ts, user, assistant}

    IMPORTANT: Reads config dynamically so tests can monkeypatch config.MEMORY_DIR / EPISODES_FILE.
    """

    def __init__(self) -> None:
        os.makedirs(config.MEMORY_DIR, exist_ok=True)
        self.path = os.path.join(config.MEMORY_DIR, config.EPISODES_FILE)

    def append(self, user: str, assistant: str) -> None:
        ep = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "user": user,
            "assistant": assistant,
        }
        # Ensure directory exists even if config was monkeypatched after init in some workflows
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    def load_all(self) -> List[Episode]:
        if not os.path.exists(self.path):
            return []
        out: List[Episode] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    out.append(Episode(**obj))
                except Exception:
                    continue
        return out

    def load_recent(self, n: int | None = None) -> List[Episode]:
        if n is None:
            n = config.MAX_RECENT_EPISODES
        eps = self.load_all()
        return eps[-n:] if n > 0 else []

    def _score(self, query: str, ep: Episode) -> int:
        q = set(query.lower().split())
        t = set((ep.user + " " + ep.assistant).lower().split())
        return len(q.intersection(t))

    def find_relevant(self, query: str, k: int | None = None) -> List[Episode]:
        if k is None:
            k = config.MAX_RELEVANT_EPISODES
        eps = self.load_all()
        scored = [(self._score(query, ep), ep) for ep in eps]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for score, ep in scored if score > 0][:k]

    def build_context(self, user_input: str) -> str:
        recent = self.load_recent(config.MAX_RECENT_EPISODES)
        relevant = self.find_relevant(user_input, config.MAX_RELEVANT_EPISODES)

        # avoid duplicates (same ts)
        seen = set()
        merged: List[Episode] = []
        for ep in relevant + recent:
            if ep.ts in seen:
                continue
            seen.add(ep.ts)
            merged.append(ep)

        if not merged:
            return ""

        blocks = []
        limit = config.MAX_RECENT_EPISODES + config.MAX_RELEVANT_EPISODES
        for ep in merged[-limit:]:
            blocks.append(f"Human: {ep.user}\nAssistant: {ep.assistant}")
        return "\n---\n".join(blocks)