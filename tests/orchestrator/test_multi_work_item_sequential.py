from __future__ import annotations

from agent import Agent
from orchestrator.models import WorkItem
from orchestrator.orchestrator import Orchestrator
from orchestrator.roles import RoleRegistry, RoleSpec


def test_multi_work_item_sequential_merge_contains_all_outputs():
    agent = Agent()
    roles = RoleRegistry(
        {
            "generalist": RoleSpec(name="generalist", agent=agent),
        }
    )
    orch = Orchestrator(role_registry=roles)

    work_items = [
        WorkItem(
            id="task-001",
            assigned_agent="generalist",
            goal="Return exactly the string: A",
            inputs={},
        ),
        WorkItem(
            id="task-002",
            assigned_agent="generalist",
            goal="Return exactly the string: B",
            inputs={},
        ),
    ]

    combined = orch.run_work_items(work_items)

    assert "task-001" in combined
    assert "task-002" in combined

    # Keep these assertions soft because your agent may add extra text around A/B
    assert "A" in combined
    assert "B" in combined