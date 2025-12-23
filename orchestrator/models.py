from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class WorkItem:
    id: str
    assigned_agent: str
    goal: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    expected_output: Dict[str, Any] = field(default_factory=dict)

    # Stage 8.3.2: explicit artifact dependencies
    depends_on: List[str] = field(default_factory=list)