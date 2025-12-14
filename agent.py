# agent.py

from __future__ import annotations

from typing import Any, Dict, List, Set

from planner import Planner
from plan_types import PlanStep
from tools import TOOLS
from memory import EpisodeStore


class Agent:
    """
    Agent = control flow:
    - ask Planner for a plan
    - execute tool steps
    - if a tool fails: capture failure + observations, then replan
    - always end with a user-facing final answer
    """

    def __init__(self, max_replans: int = 2) -> None:
        self.planner = Planner()
        self.memory = EpisodeStore()
        self.max_replans = max_replans

    def run(self, user_input: str) -> str:
        observations: Dict[str, Any] = {}
        failed_tools: Set[str] = set()
        observations_text = ""
        failure_text = ""

        # Initial plan
        plan = self.planner.generate_plan(
            user_input=user_input,
            observations_text="",
            failure_text="",
            is_replan=False,
        )

        for attempt in range(self.max_replans + 1):
            try:
                # Execute tool steps
                for step in plan.steps:
                    if step.tool:
                        if step.tool in failed_tools:
                            raise RuntimeError(
                                f"Tool '{step.tool}' was already observed failing earlier; refusing to call it again."
                            )
                        observations[step.id] = self._execute_tool(step)

                # Compose final answer
                try:
                    final_answer = self._compose_answer(user_input, observations)
                except Exception as e:
                    # IMPORTANT: composing the final response failing should not trigger replanning loops.
                    observations_text = self._format_observations(observations, failed_tools)
                    failure_text = f"Final answer composition failed: {e}"
                    final_answer = self._fallback_answer(user_input, observations_text, failure_text)

                self.memory.append(user_input, final_answer)
                return final_answer

            except Exception as e:
                # This block is ONLY for tool / execution failures.
                failure_text = str(e)

                msg = str(e)
                if msg.startswith("Tool '") and "' failed:" in msg:
                    tool_name = msg.split("Tool '", 1)[1].split("'", 1)[0]
                    failed_tools.add(tool_name)

                observations_text = self._format_observations(observations, failed_tools)

                if attempt >= self.max_replans:
                    final_answer = self._fallback_answer(user_input, observations_text, failure_text)
                    self.memory.append(user_input, final_answer)
                    return final_answer

                # Replan
                plan = self.planner.generate_plan(
                    user_input=user_input,
                    observations_text=observations_text,
                    failure_text=failure_text,
                    is_replan=True,
                )

        final_answer = self._fallback_answer(user_input, observations_text, failure_text)
        self.memory.append(user_input, final_answer)
        return final_answer

    def _execute_tool(self, step: PlanStep) -> Any:
        tool = TOOLS.get(step.tool)
        if not tool:
            raise ValueError(f"Unknown tool: {step.tool}")

        fn = tool.get("fn") or tool.get("func")
        if fn is None:
            raise KeyError(
                f"Tool '{step.tool}' has no callable. Expected 'fn' or 'func'. "
                f"Available keys: {list(tool.keys())}"
            )

        args = step.args or {}

        try:
            return fn(**args)
        except TypeError:
            return fn(args)
        except Exception as e:
            raise RuntimeError(f"Tool '{step.tool}' failed: {e}") from e

    def _compose_answer(self, user_input: str, observations: Dict[str, Any]) -> str:
        from llm_client import call_llm

        memory_context = self.memory.build_context(user_input)
        obs_text = self._format_observations(observations, failed_tools=set())

        prompt_parts: List[str] = []
        if memory_context:
            prompt_parts.append("Relevant prior context (episodic memory):\n" + memory_context)

        prompt_parts.append("User request:\n" + user_input)

        prompt_parts.append("Observations (ground truth):\n" + (obs_text if obs_text.strip() else "(none)"))

        prompt_parts.append(
            "Write a clear, user-facing response. If tools failed, explain why in plain English."
        )

        prompt = "\n\n---\n\n".join(prompt_parts)
        return call_llm(prompt)

    def _fallback_answer(self, user_input: str, observations_text: str, failure_text: str) -> str:
        lines = [
            "I couldn’t complete the request because a required tool call failed.",
            "",
            f"Request: {user_input}",
        ]
        if observations_text.strip():
            lines.append("")
            lines.append("What I observed:")
            lines.append(observations_text)
        if failure_text.strip():
            lines.append("")
            lines.append("Failure:")
            lines.append(failure_text)
        return "\n".join(lines)

    def _format_observations(self, observations: Dict[str, Any], failed_tools: Set[str]) -> str:
        lines: List[str] = []
        if failed_tools:
            lines.append("Failed tools (do not retry): " + ", ".join(sorted(failed_tools)))
        for step_id, value in observations.items():
            lines.append(f"- {step_id}: {value}")
        return "\n".join(lines)