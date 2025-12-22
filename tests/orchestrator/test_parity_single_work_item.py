from __future__ import annotations

from agent import Agent
from orchestrator.orchestrator import Orchestrator


def test_parity_single_work_item_exact_match():
    # Choose something deterministic for your system.
    # If your LLM is non-deterministic, switch this to "shape" checks instead of equality.
    user_input = "Return exactly the string: OK"

    agent = Agent()

    direct = agent.run(user_input)

    orch = Orchestrator(agent_registry={"generalist": agent})
    orchestrated = orch.run(user_input)

    assert direct == orchestrated