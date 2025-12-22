from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Artifact:
    key: str
    value: Any
    producer: str  # work_item_id
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunContext:
    """
    Orchestrator-owned, append-only context for a single run.
    Agents never mutate this directly.
    """
    artifacts: Dict[str, Artifact] = field(default_factory=dict)

    def add_artifact(self, artifact: Artifact) -> None:
        if artifact.key in self.artifacts:
            raise ValueError(f"Artifact already exists: {artifact.key}")
        self.artifacts[artifact.key] = artifact

    def get(self, key: str) -> Optional[Artifact]:
        return self.artifacts.get(key)

    def snapshot(self) -> Dict[str, Any]:
        """
        Safe, read-only view injected into agent input.
        """
        return {
            k: {
                "value": a.value,
                "producer": a.producer,
                "metadata": a.metadata,
            }
            for k, a in self.artifacts.items()
        }