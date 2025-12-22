from __future__ import annotations

from agent import Agent
from orchestrator.orchestrator import Orchestrator


class _FakeAgent:
    def __init__(self, result: str) -> None:
        self._result = result

    def run(self, user_input: str) -> str:
        return self._result


def test_parity_single_work_item_pass_through_is_exact():
    # Deterministic parity guarantee: orchestrator must return agent output unchanged
    agent = _FakeAgent("OK")
    orch = Orchestrator(agent_registry={"generalist": agent})

    out = orch.run("Return exactly the string: OK")

    assert out == "OK"


def test_single_work_item_real_agent_smoke_contract():
    # Integration smoke: real LLM may vary, so test contract not byte-for-byte equality
    user_input = "Return exactly the string: OK"

    agent = Agent()
    orch = Orchestrator(agent_registry={"generalist": agent})

    out = orch.run(user_input)

    assert isinstance(out, str)
    assert "OK" in out