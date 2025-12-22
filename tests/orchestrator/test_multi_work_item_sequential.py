from __future__ import annotations

from agent import Agent
from orchestrator.models import WorkItem
from orchestrator.orchestrator import Orchestrator


def test_multi_work_item_sequential_merge_contains_all_outputs():
    agent = Agent()
    orch = Orchestrator(agent_registry={"generalist": agent})

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
    assert "\nA" in combined or "A\n" in combined
    assert "\nB" in combined or "B\n" in combined