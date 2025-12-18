from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol


@dataclass
class SemanticHit:
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]


class SemanticStore(Protocol):
    def upsert(self, *, id: str, vector: List[float], text: str, metadata: Dict[str, Any]) -> None: ...
    def query(self, *, vector: List[float], k: int) -> List[SemanticHit]: ...