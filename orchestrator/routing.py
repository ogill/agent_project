from __future__ import annotations

from typing import Any, Dict, List

from orchestrator.models import WorkItem
from orchestrator.role_names import GENERALIST, RESEARCHER, REVIEWER


def build_work_items_for_template(
    template: str,
    goal: str,
    context: Dict[str, Any],
) -> List[WorkItem]:
    """
    Deterministic decomposition:
    - No LLM calls
    - No heuristics beyond explicit template selection
    - Produces a fixed WorkItem graph
    """
    template = template.strip().lower()

    if template == "single":
        return [
            WorkItem(
                id="work-001",
                assigned_agent=GENERALIST,
                goal=goal,
                inputs=context,
                depends_on=[],
            ),
        ]

    if template == "design_review":
        return [
            WorkItem(
                id="task-001",
                assigned_agent=RESEARCHER,
                goal=f"Extract key requirements and constraints from: {goal}",
                inputs=context,
                depends_on=[],
            ),
            WorkItem(
                id="task-002",
                assigned_agent=GENERALIST,
                goal=f"Propose an implementation approach for: {goal}",
                inputs=context,
                depends_on=["task-001.output"],
            ),
            WorkItem(
                id="task-003",
                assigned_agent=REVIEWER,
                goal=(
                    f"Review the proposed approach for risks, gaps, and a minimal "
                    f"test plan for: {goal}"
                ),
                inputs=context,
                depends_on=["task-001.output", "task-002.output"],
            ),
        ]

    # ─────────────────────────────────────────────────────────────
    # Stage 8.5: Agent-to-Agent interaction (Draft → Review → Revise)
    # ─────────────────────────────────────────────────────────────
    if template == "draft_review_revise":
        return [
            WorkItem(
                id="draft",
                assigned_agent=GENERALIST,
                goal=f"Draft an initial answer for: {goal}",
                inputs=context,
                depends_on=[],
            ),
            WorkItem(
                id="review",
                assigned_agent=REVIEWER,
                goal=(
                    "Critically review the draft for correctness, clarity, "
                    "missing steps, and risks. Provide concise feedback."
                ),
                inputs={},
                depends_on=["draft.output"],
            ),
            WorkItem(
                id="revise",
                assigned_agent=GENERALIST,
                goal=(
                    "Revise the original draft using the review feedback. "
                    "Return the improved final answer only."
                ),
                inputs={},
                depends_on=["draft.output", "review.output"],
            ),
        ]

    raise ValueError(f"Unknown template: {template!r}")