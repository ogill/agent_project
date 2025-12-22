from __future__ import annotations

from orchestrator.context import RunContext
from orchestrator.models import WorkItem
from orchestrator.orchestrator import Orchestrator
from orchestrator.roles import RoleRegistry, RoleSpec


class FakeAgent:
    def __init__(self, name: str):
        self.name = name

    def run(self, user_input: str) -> str:
        if "Shared context artifacts" in user_input:
            return f"{self.name} saw context"
        return f"{self.name} initial"


def test_downstream_work_item_sees_upstream_artifact():
    agent = FakeAgent("agent")

    orch = Orchestrator(
        role_registry=RoleRegistry(
            {"generalist": RoleSpec("generalist", agent)}
        )
    )

    work_items = [
        WorkItem(id="w1", assigned_agent="generalist", goal="Step one", inputs={}),
        WorkItem(id="w2", assigned_agent="generalist", goal="Step two", inputs={}),
    ]

    out = orch.run_work_items(work_items)

    assert "w1" in out
    assert "w2" in out
    assert "saw context" in out