# agent.py

from __future__ import annotations

import re
from typing import Any, Dict, List, Set

from planner import Planner
from plan_types import PlanStep
from tools import TOOLS
from memory import EpisodeStore


class Agent:
    """
    Agent = control flow:
    - ask Planner for a plan
    - execute tool steps deterministically
    - if a tool fails (hard or soft): capture failure + observations, then replan
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
        last_failure_text = ""  # tracks latest failure in THIS run

        soft_failure_retryable: Dict[str, bool] = {}

        def _is_soft_failure(result: Any) -> bool:
            if not isinstance(result, dict):
                return False
            if result.get("ok") is False:
                return True
            status = str(result.get("status", "")).lower().strip()
            return status in {"error", "failed", "failure"}

        def _soft_failure_reason(result: Dict[str, Any]) -> str:
            return str(
                result.get("error")
                or result.get("reason")
                or result.get("message")
                or "unspecified"
            )

        def _retryable(result: Dict[str, Any]) -> bool:
            return bool(result.get("retryable", False))

        # Initial plan
        plan = self.planner.generate_plan(
            user_input=user_input,
            observations_text="",
            failure_text="",
            is_replan=False,
        )

        for attempt in range(self.max_replans + 1):
            try:
                tools_ran = False

                # Execute tool steps
                for step in plan.steps:
                    if not step.tool:
                        continue

                    tools_ran = True

                    if step.tool in failed_tools:
                        raise RuntimeError(
                            f"Tool '{step.tool}' was already observed failing earlier; refusing to call it again."
                        )

                    result = self._execute_tool(step)
                    observations[step.id] = result

                    # Soft failure = tool returned structured failure payload (no exception)
                    if _is_soft_failure(result):
                        reason = _soft_failure_reason(result)
                        retryable = _retryable(result)

                        soft_failure_retryable[step.tool] = retryable

                        if not retryable:
                            failed_tools.add(step.tool)

                        raise RuntimeError(f"Tool '{step.tool}' soft failed: {reason}")

                # If NO tools ran, do NOT ask the LLM to "interpret missing observations".
                # First, handle "exact output" style requests deterministically.
                if not tools_ran:
                    exact = self._extract_exact_output_request(user_input)
                    if exact is not None:
                        self.memory.append(user_input, exact)
                        return exact

                # Compose final answer (do NOT trigger replanning if this fails)
                try:
                    compose_failure_text = last_failure_text if failed_tools else ""
                    final_answer = self._compose_answer(
                        user_input=user_input,
                        observations=observations,
                        failed_tools=failed_tools,
                        failure_text=compose_failure_text,
                        tools_ran=tools_ran,
                    )
                except Exception as e:
                    observations_text = self._format_observations(observations, failed_tools)
                    failure_text = f"Final answer composition failed: {e}"
                    final_answer = self._fallback_answer(user_input, observations_text, failure_text)

                self.memory.append(user_input, final_answer)
                return final_answer

            except Exception as e:
                # ONLY tool / execution failures arrive here
                failure_text = str(e)
                last_failure_text = failure_text

                tool_name = None
                msg = failure_text
                hard_fail = False
                soft_fail = False

                if msg.startswith("Tool '") and "' failed:" in msg:
                    tool_name = msg.split("Tool '", 1)[1].split("'", 1)[0]
                    hard_fail = True
                elif msg.startswith("Tool '") and "' soft failed:" in msg:
                    tool_name = msg.split("Tool '", 1)[1].split("'", 1)[0]
                    soft_fail = True

                if tool_name:
                    if hard_fail:
                        failed_tools.add(tool_name)
                    elif soft_fail:
                        retryable = soft_failure_retryable.get(tool_name, False)
                        if not retryable:
                            failed_tools.add(tool_name)

                observations_text = self._format_observations(observations, failed_tools)

                if attempt >= self.max_replans:
                    final_answer = self._fallback_answer(user_input, observations_text, failure_text)
                    self.memory.append(user_input, final_answer)
                    return final_answer

                # Replan
                try:
                    plan = self.planner.generate_plan(
                        user_input=user_input,
                        observations_text=observations_text,
                        failure_text=failure_text,
                        is_replan=True,
                        forbidden_tools=failed_tools,
                    )
                except TypeError:
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
                f"Tool '{step.tool}' has no callable. Available keys: {list(tool.keys())}"
            )

        args = step.args or {}

        try:
            return fn(**args)
        except Exception as e:
            raise RuntimeError(f"Tool '{step.tool}' failed: {e}") from e

    def _extract_exact_output_request(self, user_input: str) -> str | None:
        """
        Deterministic helper for prompts like:
          - "Return exactly the string: OK"
          - "Output exactly: OK"
          - "Do not use any tools. Output exactly: OK"
        Returns the requested literal (e.g. "OK") or None if not matched.
        """
        s = (user_input or "").strip()

        # Return exactly the string: X
        m = re.search(r"(?i)\breturn exactly the string:\s*(.+)\s*$", s)
        if m:
            return m.group(1).strip().strip('"').strip("'")

        # Output exactly: X
        m = re.search(r"(?i)\boutput exactly:\s*(.+)\s*$", s)
        if m:
            return m.group(1).strip().strip('"').strip("'")

        return None

    def _compose_answer(
        self,
        *,
        user_input: str,
        observations: Dict[str, Any],
        failed_tools: Set[str] | None = None,
        failure_text: str = "",
        tools_ran: bool = False,
    ) -> str:
        from llm_client import call_llm

        failed_tools = failed_tools or set()

        prompt_parts: List[str] = []

        # Only inject episodic memory when there was no failure in this run
        # AND the run wasn't a tool-driven failure scenario.
        if not failed_tools and not failure_text.strip():
            memory_context = self.memory.build_context(user_input)
            if memory_context:
                prompt_parts.append(
                    "Relevant prior context (episodic memory):\n"
                    + memory_context
                    + "\n\nIMPORTANT: Use prior context ONLY if it is clearly about the SAME request. "
                      "If it seems unrelated, ignore it completely and do not mention it."
                )

        prompt_parts.append("User request:\n" + user_input)

        # IMPORTANT: Only mention tools/observations if tools actually ran OR a tool failed.
        if tools_ran or failed_tools or failure_text.strip():
            obs_text = self._format_observations(observations, failed_tools)
            prompt_parts.append(
                "Observations (ground truth from tools in THIS run):\n"
                + (obs_text if obs_text.strip() else "(none)")
            )

        if failure_text.strip():
            prompt_parts.append(
                "Failure encountered in THIS run (ground truth):\n"
                + failure_text.strip()
            )

        # If this was a no-tool run, be explicit: answer directly; do not invent tools.
        if not tools_ran and not failed_tools and not failure_text.strip():
            prompt_parts.append(
                "Instructions:\n"
                "- No tools were run for this request.\n"
                "- Answer the user directly from the request.\n"
                "- Do NOT mention tools, observations, or failures.\n"
                "- If the user asked for an exact literal output, output exactly that."
            )
        else:
            prompt_parts.append(
                "Write a clear, user-facing response.\n"
                "- If any tool failed, you MUST explicitly say: 'The tool <tool_name> failed' (use the word 'failed').\n"
                "- Explain the failure cause ONLY using the Failure text above and/or the Observations above.\n"
                "- Do NOT invent HTTP status codes or error causes that are not explicitly present.\n"
                "- Do NOT mention unrelated past failures or prior conversations unless the user explicitly asks."
            )

        prompt = "\n\n".join(prompt_parts)
        return call_llm(prompt)

    def _fallback_answer(self, user_input: str, observations_text: str, failure_text: str) -> str:
        lines = [
            "I couldn’t complete the request because a required tool call failed.",
            "",
            f"Request: {user_input}",
        ]
        if observations_text.strip():
            lines.extend(["", "What I observed:", observations_text])
        if failure_text.strip():
            lines.extend(["", "Failure:", failure_text])
        return "\n".join(lines)

    def _format_observations(self, observations: Dict[str, Any], failed_tools: Set[str]) -> str:
        lines: List[str] = []
        if failed_tools:
            lines.append("Failed tools (do not retry): " + ", ".join(sorted(failed_tools)))
        for step_id, value in observations.items():
            lines.append(f"- {step_id}: {value}")
        return "\n".join(lines)