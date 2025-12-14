from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PlanStep:
    id: str
    description: str
    tool: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    requires: List[str] = field(default_factory=list)


@dataclass
class Plan:
    goal: str
    steps: List[PlanStep]