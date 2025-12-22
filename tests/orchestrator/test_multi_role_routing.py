from __future__ import annotations

from agent import Agent
from orchestrator.orchestrator import Orchestrator
from orchestrator.role_names import EXECUTOR, GENERALIST, RESEARCHER, REVIEWER
from orchestrator.roles import RoleRegistry, RoleSpec


def test_run_template_routes_to_multiple_roles_and_merges():
    agent = Agent()

    roles = RoleRegistry(
        {
            GENERALIST: RoleSpec(name=GENERALIST, agent=agent),
            RESEARCHER: RoleSpec(name=RESEARCHER, agent=agent),
            REVIEWER: RoleSpec(name=REVIEWER, agent=agent),
            EXECUTOR: RoleSpec(name=EXECUTOR, agent=agent),
        }
    )

    orch = Orchestrator(role_registry=roles)

    combined = orch.run_template(
        template="design_review",
        goal="Design Stage 8 multi-agent orchestration on top of Stage 7",
        context={},
    )

    # Proves multiple role labels appear in merged output
    assert RESEARCHER in combined
    assert GENERALIST in combined
    assert REVIEWER in combined
    assert "task-001" in combined
    assert "task-002" in combined
    assert "task-003" in combined