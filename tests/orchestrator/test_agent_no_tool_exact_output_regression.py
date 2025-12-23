from __future__ import annotations

from agent import Agent


class _FakePlannerComposeOnly:
    def generate_plan(self, *args, **kwargs):
        # Minimal object matching the shape Agent expects: plan.steps list with .tool attr
        class _Step:
            def __init__(self):
                self.id = "compose_answer"
                self.description = "compose"
                self.tool = None
                self.args = None
                self.requires = ["nonsense"]  # simulate bad requires from model

        class _Plan:
            def __init__(self):
                self.goal = "whatever"
                self.steps = [_Step()]

        return _Plan()


class _NoOpEpisodeStore:
    def append(self, *_args, **_kwargs):
        return None

    def build_context(self, *_args, **_kwargs):
        return ""


def test_agent_compose_only_exact_output_returns_literal_without_llm():
    agent = Agent()
    agent.planner = _FakePlannerComposeOnly()
    agent.memory = _NoOpEpisodeStore()

    out = agent.run("Return exactly the string: OK")

    assert out == "OK"


def test_agent_compose_only_output_exact_returns_literal_without_llm():
    agent = Agent()
    agent.planner = _FakePlannerComposeOnly()
    agent.memory = _NoOpEpisodeStore()

    out = agent.run("Do not use any tools. Output exactly: OK")

    assert out == "OK"