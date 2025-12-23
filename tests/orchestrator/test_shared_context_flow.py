from __future__ import annotations

from orchestrator.context import RunContext
from orchestrator.models import WorkItem
from orchestrator.orchestrator import Orchestrator
from orchestrator.roles import RoleRegistry, RoleSpec


class FakeAgent:
    def __init__(self, name: str):
        self.name = name

    def run(self, user_input: str) -> str:
        if "Shared context artifacts" not in user_input:
            return f"{self.name} initial"

        saw_w1 = "w1.output" in user_input
        saw_w2 = "w2.output" in user_input
        return f"saw_w1={saw_w1} saw_w2={saw_w2}"


def test_downstream_work_item_sees_upstream_artifact():
    agent = FakeAgent("agent")

    orch = Orchestrator(
        role_registry=RoleRegistry(
            {"generalist": RoleSpec("generalist", agent)}
        )
    )

    work_items = [
        WorkItem(id="w1", assigned_agent="generalist", goal="Step one", inputs={}, depends_on=[]),
        WorkItem(id="w2", assigned_agent="generalist", goal="Step two", inputs={}, depends_on=["w1.output"]),
    ]
    out = orch.run_work_items(work_items)

    assert "saw_w1=True" in out
    assert "saw_w2=False" in out