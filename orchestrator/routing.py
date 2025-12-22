from __future__ import annotations

from typing import Any, Dict, List

from orchestrator.models import WorkItem
from orchestrator.role_names import GENERALIST, RESEARCHER, REVIEWER


def build_work_items_for_template(template: str, goal: str, context: Dict[str, Any]) -> List[WorkItem]:
    """
    Deterministic decomposition: no LLM calls, no heuristics beyond template selection.
    """
    if template == "single":
        return [
            WorkItem(id="work-001", assigned_agent=GENERALIST, goal=goal, inputs=context),
        ]

    if template == "design_review":
        return [
            WorkItem(
                id="task-001",
                assigned_agent=RESEARCHER,
                goal=f"Extract key requirements and constraints from: {goal}",
                inputs=context,
            ),
            WorkItem(
                id="task-002",
                assigned_agent=GENERALIST,
                goal=f"Propose an implementation approach for: {goal}",
                inputs=context,
            ),
            WorkItem(
                id="task-003",
                assigned_agent=REVIEWER,
                goal=f"Review the proposed approach for risks, gaps, and a minimal test plan for: {goal}",
                inputs=context,
            ),
        ]

    raise ValueError(f"Unknown template: {template!r}")