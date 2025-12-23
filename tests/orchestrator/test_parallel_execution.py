from __future__ import annotations

import time

from orchestrator.models import WorkItem
from orchestrator.orchestrator import Orchestrator, OrchestratorPolicy
from orchestrator.roles import RoleRegistry, RoleSpec


class SleepyAgent:
    def __init__(self, seconds: float, out: str) -> None:
        self.seconds = seconds
        self.out = out

    def run(self, user_input: str) -> str:
        time.sleep(self.seconds)
        return self.out


def test_parallel_wave_finishes_faster_than_sequential():
    roles = RoleRegistry(
        {
            "a": RoleSpec("a", SleepyAgent(0.5, "A")),
            "b": RoleSpec("b", SleepyAgent(0.5, "B")),
        }
    )
    orch = Orchestrator(
        role_registry=roles,
        policy=OrchestratorPolicy(max_concurrency=2, enable_parallel=True, per_item_timeout_s=5.0),
    )

    items = [
        WorkItem(id="w1", assigned_agent="a", goal="one", inputs={}, depends_on=[]),
        WorkItem(id="w2", assigned_agent="b", goal="two", inputs={}, depends_on=[]),
    ]

    t0 = time.time()
    out = orch.run_work_items(items)
    elapsed = time.time() - t0

    assert "A" in out
    assert "B" in out

    # Sequential would be ~1.0s; parallel should be closer to ~0.5s.
    assert elapsed < 0.9


def test_dependency_forces_sequential_execution():
    roles = RoleRegistry(
        {
            "a": RoleSpec("a", SleepyAgent(0.5, "A")),
            "b": RoleSpec("b", SleepyAgent(0.5, "B")),
        }
    )
    orch = Orchestrator(
        role_registry=roles,
        policy=OrchestratorPolicy(max_concurrency=2, enable_parallel=True, per_item_timeout_s=5.0),
    )

    items = [
        WorkItem(id="w1", assigned_agent="a", goal="one", inputs={}, depends_on=[]),
        WorkItem(id="w2", assigned_agent="b", goal="two", inputs={}, depends_on=["w1.output"]),
    ]

    t0 = time.time()
    out = orch.run_work_items(items)
    elapsed = time.time() - t0

    assert "A" in out
    assert "B" in out

    # With dependency, should be roughly sequential (~1.0s). Allow some slack.
    assert elapsed > 0.9