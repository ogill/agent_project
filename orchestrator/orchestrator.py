from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from orchestrator.models import WorkItem
from orchestrator.roles import RoleRegistry
from orchestrator.routing import build_work_items_for_template
from orchestrator.context import RunContext, Artifact

@dataclass
class OrchestratorPolicy:
    max_work_items: int = 10


class Orchestrator:
    """
    Stage 8.2 Orchestrator:
    - 8.2: single WorkItem -> single agent call (parity path)
    - 8.2.1: multiple WorkItems -> sequential agent calls + deterministic merge
    - 8.2.2: role indirection via RoleRegistry (no behavior change)
    """

    def __init__(
        self,
        role_registry: RoleRegistry,
        policy: Optional[OrchestratorPolicy] = None,
    ) -> None:
        self.role_registry = role_registry
        self.policy = policy or OrchestratorPolicy()

    def run(self, goal: str, context: Optional[Dict[str, Any]] = None) -> str:
        context = context or {}

        work_item = WorkItem(
            id="work-001",
            assigned_agent="generalist",
            goal=goal,
            inputs=context,
            expected_output={},
        )

        return self.run_work_items([work_item])

    def run_work_items(self, work_items: List[WorkItem]) -> str:
        run_context = RunContext()
        if len(work_items) > self.policy.max_work_items:
            raise ValueError(
                f"Too many WorkItems: {len(work_items)} (max {self.policy.max_work_items})"
            )

        results: Dict[str, str] = {}

        for wi in work_items:
            agent = self.role_registry.get_agent(wi.assigned_agent)

            # Only inject what this WorkItem explicitly depends on
            selected = {}
            if wi.depends_on:
                selected = run_context.snapshot_selected(wi.depends_on)

            user_input = _compose_user_input(wi.goal, wi.inputs, selected)
            output = _run_stage7_agent(agent, user_input)

            artifact = Artifact(
                key=f"{wi.id}.output",
                value=output,
                producer=wi.id,
                metadata={
                    "role": wi.assigned_agent,
                },
            )

            run_context.add_artifact(artifact)
            results[wi.id] = output

        return _merge_results_deterministically(work_items, results)

    def run_template(self, template: str, goal: str, context: Optional[Dict[str, Any]] = None) -> str:
            """
            Stage 8.2.3: deterministic multi-role routing via templates.
            """
            context = context or {}
            work_items = build_work_items_for_template(template=template, goal=goal, context=context)
            return self.run_work_items(work_items)




def _compose_user_input(
        goal: str,
        context: Dict[str, Any],
        selected_artifacts: Dict[str, Any],
    ) -> str:
        parts = [goal]

        if context:
            parts.append(f"Initial context (JSON): {context}")

        if selected_artifacts:
            parts.append("Shared context artifacts:\n" + f"{selected_artifacts}")

        return "\n\n".join(parts)


def _run_stage7_agent(agent: Any, user_input: str) -> str:
    if hasattr(agent, "run"):
        return agent.run(user_input)
    raise AttributeError("Agent does not expose .run(user_input: str)")


def _merge_results_deterministically(work_items: List[WorkItem], results: Dict[str, str]) -> str:
    # Stage 7 parity guarantee for single WorkItem:
    if len(work_items) == 1:
        wi = work_items[0]
        return results.get(wi.id, "")

    lines: List[str] = []
    for wi in work_items:
        lines.append(f"[{wi.id}] {wi.assigned_agent}: {wi.goal}")
        lines.append(results.get(wi.id, "").rstrip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"

