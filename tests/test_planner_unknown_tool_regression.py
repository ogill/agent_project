# tests/test_planner_unknown_tool_regression.py

from __future__ import annotations

import planner as planner_module
from planner import Planner


def test_planner_coerces_unknown_tool_to_none_and_does_not_raise(monkeypatch):
    """
    Regression:
    - If the LLM invents an unknown tool, Planner must NOT raise.
    - The unknown tool must be coerced to tool=None (so Agent won't execute it).
    - compose_answer must exist and be the final step.
    - compose_answer.requires must not include itself.
    """

    def fake_call_llm(_prompt: str) -> str:
        return """
        {
          "goal": "test unknown tool",
          "steps": [
            {
              "id": "do_unknown",
              "description": "Try an invented tool",
              "tool": "test_system",
              "args": {},
              "requires": []
            },
            {
              "id": "compose_answer",
              "description": "Finish",
              "tool": null,
              "args": null,
              "requires": ["do_unknown", "compose_answer"]
            }
          ]
        }
        """

    monkeypatch.setattr(planner_module, "call_llm", fake_call_llm)

    p = Planner()
    plan = p.generate_plan(
        user_input="Anything",
        tools_spec="Available tools:\n- get_time: stub\n",
        is_replan=False,
    )

    assert len(plan.steps) >= 1
    assert plan.steps[-1].id == "compose_answer"
    assert plan.steps[-1].tool is None
    assert "compose_answer" not in (plan.steps[-1].requires or [])

    # unknown tool step must be coerced to tool=None (not executable)
    unknown = next((s for s in plan.steps if s.id == "do_unknown"), None)
    assert unknown is not None
    assert unknown.tool is None