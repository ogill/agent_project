# agent.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
import json

from config import DEBUG_AGENT, DEBUG_AGENT_PROMPTS, MAX_REACT_STEPS, PROMPT_MODE
from llm_client import call_llm
from tools import TOOLS
from prompts import get_executor_prompt


SYSTEM_PROMPT = get_executor_prompt(PROMPT_MODE)


def build_step_aware_prompt(
    user_input: str,
    observations: List[str],
    plan_step: "PlanStep",
    current_step_index: int,
    total_steps: int,
) -> str:
    """
    Build an executor prompt aware of the current plan step.

    IMPORTANT:
    - In this version, the executor does NOT ask the LLM to create tool calls.
    - Tool steps are executed by the system directly (using plan_step.tool + plan_step.args).
    - The LLM is used only for non-tool steps (especially compose_answer).
    """
    prompt = SYSTEM_PROMPT + "\n\n"
    prompt += (
        f"You are executing step {current_step_index + 1} of {total_steps} in a predefined plan.\n"
        f"Step id: {plan_step.id}\n"
        f"Step description: {plan_step.description}\n"
        f"Required tool for this step (if any): {plan_step.tool}\n"
        "Do NOT change the overall plan.\n\n"
    )

    prompt += "User: " + (user_input or "").strip() + "\n"

    if observations:
        prompt += "\nObservations so far (ground truth):\n"
        for idx, obs in enumerate(observations, start=1):
            prompt += f"{idx}) {obs}\n"

    prompt += "\nAssistant:"
    return prompt


def parse_react_response(text: str) -> Dict[str, Optional[str]]:
    """
    Parse a response into Thought / Action / Final Answer.
    We keep this for compatibility with your prompt format,
    but we do NOT rely on Action for tool calls anymore.
    """
    thought = action = final_answer = None
    lines = text.splitlines()
    final_lines = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        lower = line.lower()

        if lower.startswith("thought:") and thought is None:
            thought = line[len("thought:") :].strip()

        elif lower.startswith("action:") and action is None:
            action = line[len("action:") :].strip()

        elif lower.startswith("final answer:") and final_lines is None:
            final_lines = []
            rest = line[len("final answer:") :].strip()
            if rest:
                final_lines.append(rest)
            for j in range(i + 1, len(lines)):
                final_lines.append(lines[j])
            break

        i += 1

    if final_lines is not None:
        final_answer = "\n".join(final_lines).strip() or None

    return {"thought": thought, "action": action, "final_answer": final_answer}


def _is_placeholder_final_answer(text: Optional[str]) -> bool:
    if not text:
        return True
    t = text.strip().lower()

    placeholders = {
        "(waiting for observations)",
        "(waiting for tool result)",
        "(waiting for tool results)",
        "(waiting for results)",
        "waiting for observations",
        "waiting for tool result",
        "waiting for tool results",
        "waiting for results",
    }
    if t in placeholders:
        return True

    if "waiting for" in t and len(t) <= 80:
        return True

    return False


def _stable_args_key(args: Dict[str, Any]) -> str:
    try:
        return json.dumps(args, sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(args)


@dataclass
class PlanStep:
    id: str
    description: str
    tool: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    requires: List[str] = field(default_factory=list)


@dataclass
class Plan:
    goal: str
    steps: List[PlanStep] = field(default_factory=list)


@dataclass
class AgentState:
    user_input: str
    observations: List[str] = field(default_factory=list)

    tools_used: Dict[str, int] = field(default_factory=dict)
    last_tool_name: Optional[str] = None
    last_tool_args: Optional[Dict[str, Any]] = None
    last_tool_result: Optional[str] = None
    last_llm_response: Optional[str] = None

    plan: Optional[Plan] = None
    current_step: int = 0
    replan_count: int = 0

    failed_tool_calls: Set[Tuple[str, str]] = field(default_factory=set)


class SimpleAgent:
    """
    Planner + Executor with Dynamic Replanning.

    Key change vs your previous version:
    - Tool steps are executed directly using the plan (step.tool + step.args).
    - The LLM is used only to write the final answer (compose) and other non-tool steps.
    """

    def __init__(self, planner: Optional["Planner"] = None) -> None:
        if planner is None:
            from planner import Planner

            planner = Planner()
        self.planner = planner
        self.max_replans = 2

        self.allow_retry_tools = {"fetch_url"}            # transient
        self.intentional_failure_tools = {"always_fail"}  # by design

    def _debug_replan(self, reason: str, step: PlanStep, observations: List[str]) -> None:
        if DEBUG_AGENT:
            print(
                f"[REPLAN] Triggered replanning | reason={reason} | "
                f"failed_step={step.id} | observations={len(observations)}"
            )

    def handle_turn(self, user_input: str) -> str:
        state = AgentState(user_input=user_input)
        state.plan = self.planner.generate_plan(user_input, tools_spec="")
        state.current_step = 0
        return self._execute_plan(state)

    def _already_failed(self, state: AgentState, tool_name: str, args: Dict[str, Any]) -> bool:
        return (tool_name, _stable_args_key(args)) in state.failed_tool_calls

    def _mark_failed(self, state: AgentState, tool_name: str, args: Dict[str, Any]) -> None:
        state.failed_tool_calls.add((tool_name, _stable_args_key(args)))

    def _execute_plan(self, state: AgentState) -> str:
        if not state.plan:
            return "No plan available."

        react_step = 0

        while state.current_step < len(state.plan.steps) and react_step < MAX_REACT_STEPS:
            react_step += 1
            step = state.plan.steps[state.current_step]

            # ============================================================
            # TOOL STEP: execute tool directly (no LLM call needed)
            # ============================================================
            if step.tool is not None:
                tool_name = step.tool
                args = step.args or {}

                # Safety: tool must exist
                if tool_name not in TOOLS:
                    return self._replan_or_exit(state, step, f"Unknown tool in plan: '{tool_name}'")

                # Prevent pointless repeats of the exact same failing call,
                # but DO NOT block intentional failure tools like always_fail.
                if (
                    tool_name not in self.intentional_failure_tools
                    and tool_name not in self.allow_retry_tools
                    and self._already_failed(state, tool_name, args)
                ):
                    return self._replan_or_exit(
                        state,
                        step,
                        f"Tool '{tool_name}' previously failed with the same args; refusing to retry.",
                    )

                result, failed = self._call_tool(tool_name, args)

                obs = (
                    f"The tool '{tool_name}' was called with arguments {json.dumps(args)} "
                    f"and returned this result: {result}"
                )
                state.observations.append(obs)

                state.tools_used[tool_name] = state.tools_used.get(tool_name, 0) + 1
                state.last_tool_name = tool_name
                state.last_tool_args = args
                state.last_tool_result = result

                if failed:
                    # If the failure is intentional (e.g., always_fail), we do NOT replan.
                    # We keep the observation and continue so compose_answer can explain it.
                    if tool_name in self.intentional_failure_tools:
                        state.current_step += 1
                        continue

                    self._mark_failed(state, tool_name, args)
                    return self._replan_or_exit(state, step, result)

                state.current_step += 1
                continue

            # ============================================================
            # NON-TOOL STEP
            # - Only call LLM for compose_answer
            # - Skip LLM for intermediate non-tool steps (deterministic)
            # ============================================================
            if step.id != "compose_answer":
                state.observations.append(f"Non-tool step '{step.id}' skipped (deterministic).")
                state.current_step += 1
                continue

            # compose_answer: ask LLM to write the final response
            prompt = build_step_aware_prompt(
                state.user_input,
                state.observations,
                step,
                state.current_step,
                len(state.plan.steps),
            )

            if DEBUG_AGENT_PROMPTS:
                print("[DEBUG PROMPT]\n", prompt, "\n")

            response = call_llm(prompt)
            state.last_llm_response = response
            parts = parse_react_response(response)
            final_answer = parts["final_answer"]

            if _is_placeholder_final_answer(final_answer):
                return self._replan_or_exit(
                    state,
                    step,
                    "compose_answer produced no usable final answer (placeholder/empty).",
                )

            return final_answer  # type: ignore[return-value]

        # Fallback: if we exit loop without returning (e.g., no compose_answer, or step limit hit)
        return state.last_llm_response or (
            "I was unable to complete the request.\n"
            + ("Observations:\n" + "\n".join(state.observations[-10:]) if state.observations else "")
        )

    def _call_tool(self, tool_name: str, args: Dict[str, Any]) -> Tuple[str, bool]:
        tool = TOOLS[tool_name]
        try:
            validated = tool["args_model"].model_validate(args)
            return str(tool["fn"](validated)), False
        except Exception as e:
            return f"Tool '{tool_name}' failed: {e!r}", True

    def _replan_or_exit(self, state: AgentState, step: PlanStep, reason: str) -> str:
        if state.replan_count >= self.max_replans:
            return reason

        state.replan_count += 1
        self._debug_replan(reason, step, state.observations)

        new_plan = self.planner.generate_plan(
            state.user_input,
            tools_spec="",
            observations_text="\n".join(state.observations),
            failure_text=reason,
            is_replan=True,
        )

        state.plan = new_plan
        state.current_step = 0
        return self._execute_plan(state)