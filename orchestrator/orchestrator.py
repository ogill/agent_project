from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from orchestrator.models import WorkItem


@dataclass
class OrchestratorPolicy:
    max_work_items: int = 10


class Orchestrator:
    """
    Stage 8.2 Orchestrator (scaffold):
    - Stage 8.2: single WorkItem -> single agent call (parity path)
    - Stage 8.2.1: multiple WorkItems -> sequential agent calls + deterministic merge
    """

    def __init__(
        self,
        agent_registry: Mapping[str, Any],
        policy: Optional[OrchestratorPolicy] = None,
    ) -> None:
        self.agent_registry = agent_registry
        self.policy = policy or OrchestratorPolicy()

    def run(self, goal: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Parity path: 1 goal -> 1 WorkItem -> 1 Stage-7 Agent.run(user_input).
        This must remain unchanged so Stage 7 behavior is preserved.
        """
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
        """
        Stage 8.2.1: run N WorkItems sequentially, then merge deterministically.
        """
        if len(work_items) > self.policy.max_work_items:
            raise ValueError(
                f"Too many WorkItems: {len(work_items)} (max {self.policy.max_work_items})"
            )

        results: Dict[str, str] = {}

        for wi in work_items:
            agent = self.agent_registry.get(wi.assigned_agent)
            if agent is None:
                raise KeyError(f"No agent registered for role: {wi.assigned_agent!r}")

            user_input = _compose_user_input(wi.goal, wi.inputs)
            output = _run_stage7_agent(agent, user_input)
            results[wi.id] = output

        return _merge_results_deterministically(work_items, results)


def _compose_user_input(goal: str, context: Dict[str, Any]) -> str:
    """
    Deterministic composition so we can accept structured context
    while still calling Stage-7 Agent.run(user_input: str).
    """
    if not context:
        return goal
    return f"{goal}\n\nContext (JSON): {context}"


def _run_stage7_agent(agent: Any, user_input: str) -> str:
    if hasattr(agent, "run"):
        return agent.run(user_input)
    raise AttributeError("Agent registry entry does not expose .run(user_input: str)")


def _merge_results_deterministically(work_items: List[WorkItem], results: Dict[str, str]) -> str:
    """
    Deterministic merge policy (Stage 8.2.1):
    - If only one WorkItem, return output unchanged (Stage 7 parity guarantee)
    - Otherwise, preserve order and emit stable combined output with boundaries
    """
    if len(work_items) == 1:
        wi = work_items[0]
        return results.get(wi.id, "")

    lines: List[str] = []
    for wi in work_items:
        lines.append(f"[{wi.id}] {wi.assigned_agent}: {wi.goal}")
        lines.append(results.get(wi.id, "").rstrip())
        lines.append("")  # blank line between tasks

    return "\n".join(lines).rstrip() + "\n"