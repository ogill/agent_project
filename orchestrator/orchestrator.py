from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from orchestrator.models import WorkItem


@dataclass
class OrchestratorPolicy:
    max_work_items: int = 1


class Orchestrator:
    """
    Stage 8.2 Orchestrator (scaffold):
    - emits exactly one WorkItem
    - routes to a single role ("generalist")
    - calls the Stage-7 Agent.run(user_input: str) unchanged
    """

    def __init__(
        self,
        agent_registry: Mapping[str, Any],
        policy: Optional[OrchestratorPolicy] = None,
    ) -> None:
        self.agent_registry = agent_registry
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

        agent = self.agent_registry.get(work_item.assigned_agent)
        if agent is None:
            raise KeyError(f"No agent registered for role: {work_item.assigned_agent!r}")

        user_input = _compose_user_input(work_item.goal, work_item.inputs)
        return _run_stage7_agent(agent, user_input)


def _compose_user_input(goal: str, context: Dict[str, Any]) -> str:
    """
    Deterministic composition so we can preserve Stage 7 behavior while still
    allowing the Orchestrator API to accept structured context.
    """
    if not context:
        return goal

    # Keep this stable and minimal; no "reasoning" here.
    return f"{goal}\n\nContext (JSON): {context}"


def _run_stage7_agent(agent: Any, user_input: str) -> str:
    if hasattr(agent, "run"):
        return agent.run(user_input)  # Stage 7 API
    raise AttributeError("Agent registry entry does not expose .run(user_input: str)")