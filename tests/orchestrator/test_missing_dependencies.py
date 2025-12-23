import pytest
from orchestrator.models import WorkItem
from orchestrator.orchestrator import Orchestrator
from orchestrator.roles import RoleRegistry, RoleSpec

class FakeAgent:
    def run(self, user_input: str) -> str:
        return "ok"

def test_missing_dependency_raises_keyerror():
    orch = Orchestrator(role_registry=RoleRegistry({"generalist": RoleSpec("generalist", FakeAgent())}))

    items = [
        WorkItem(id="w1", assigned_agent="generalist", goal="one", inputs={}, depends_on=["does.not.exist"]),
    ]

    with pytest.raises(KeyError):
        orch.run_work_items(items)