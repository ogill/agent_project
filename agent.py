from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
import json

from config import DEBUG_AGENT, DEBUG_AGENT_PROMPTS, MAX_REACT_STEPS
from llm_client import call_llm
from tools import TOOLS
from prompts import get_react_system_prompt

SYSTEM_PROMPT = get_react_system_prompt()


def build_step_aware_prompt(
    user_input: str,
    observations: List[str],
    plan_step: "PlanStep",
    current_step_index: int,
    total_steps: int,
) -> str:
    """
    Build a ReAct prompt that is aware of the current plan step.
    """
    prompt = SYSTEM_PROMPT + "\n\n"
    prompt += (
        f"You are executing step {current_step_index + 1} of {total_steps} "
        f"in a predefined plan.\n"
        f"Step id: {plan_step.id}\n"
        f"Step description: {plan_step.description}\n"
        f"Required tool for this step (if any): {plan_step.tool}\n"
        "Do NOT change the overall plan. Only work on this step.\n\n"
    )

    prompt += "User: " + user_input + "\n"

    if observations:
        prompt += "\nObservations so far:\n"
        for idx, obs in enumerate(observations, start=1):
            prompt += f"{idx}) {obs}\n"

    prompt += "\nAssistant:"
    return prompt


def parse_react_response(text: str) -> Dict[str, Optional[str]]:
    """
    Parse a ReAct-style response into Thought / Action / Final Answer.
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


def maybe_parse_tool_call(action_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse CALL_TOOL:tool_name({...})
    """
    if not action_text or not action_text.startswith("CALL_TOOL:"):
        return None

    rest = action_text[len("CALL_TOOL:") :].strip()
    open_i = rest.find("(")
    close_i = rest.rfind(")")

    if open_i == -1 or close_i == -1:
        return None

    tool = rest[:open_i].strip()
    args_str = rest[open_i + 1 : close_i].strip()

    if not tool:
        return None

    try:
        args = json.loads(args_str) if args_str else {}
    except json.JSONDecodeError:
        return None

    return {"tool_name": tool, "args": args}


def _is_placeholder_final_answer(text: Optional[str]) -> bool:
    """
    Detect common "placeholder" answers that should NOT be accepted as completion.
    """
    if not text:
        return True
    t = text.strip().lower()

    # Common placeholders your runs show
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

    # Also treat very short "waiting" style outputs as placeholders
    if "waiting for" in t and len(t) <= 80:
        return True

    return False


def _stable_args_key(args: Dict[str, Any]) -> str:
    """
    Stable JSON key for args dict (for retry detection).
    """
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
    step: int = 0

    tools_used: Dict[str, int] = field(default_factory=dict)
    last_tool_name: Optional[str] = None
    last_tool_args: Optional[Dict[str, Any]] = None
    last_tool_result: Optional[str] = None
    last_llm_response: Optional[str] = None

    plan: Optional[Plan] = None
    current_step: int = 0
    replan_count: int = 0

    # Track tool failures to avoid infinite loops / pointless retries
    failed_tool_calls: Set[Tuple[str, str]] = field(default_factory=set)


class SimpleAgent:
    """
    Planner + ReAct Executor with Dynamic Replanning.
    """

    def __init__(self, planner: Optional["Planner"] = None) -> None:
        if planner is None:
            from planner import Planner

            planner = Planner()
        self.planner = planner
        self.max_replans = 2

        # Tools that are allowed to fail and/or be retried (intentional or transient)
        self.allow_retry_tools = {"fetch_url"}  # network can be transient
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
        key = (tool_name, _stable_args_key(args))
        return key in state.failed_tool_calls

    def _mark_failed(self, state: AgentState, tool_name: str, args: Dict[str, Any]) -> None:
        key = (tool_name, _stable_args_key(args))
        state.failed_tool_calls.add(key)

    def _execute_plan(self, state: AgentState) -> str:
        if not state.plan:
            return "No plan available."

        react_step = 0

        while state.current_step < len(state.plan.steps) and react_step < MAX_REACT_STEPS:
            react_step += 1
            step = state.plan.steps[state.current_step]

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

            action = parts["action"]
            final_answer = parts["final_answer"]

            action_text = None
            if action and action.strip().upper() != "NONE":
                action_text = action.strip()

            # === Non-tool steps ===
            if step.tool is None:
                # Must not call tools in non-tool steps
                if action_text:
                    return self._replan_or_exit(state, step, f"Unexpected tool call: {action_text}")

                # Placeholder / empty final answer is NOT acceptable, especially for compose_answer
                if _is_placeholder_final_answer(final_answer):
                    return self._replan_or_exit(
                        state,
                        step,
                        f"Non-tool step '{step.id}' produced no usable final answer (placeholder/empty).",
                    )

                # Final step must return the composed answer
                if step.id == "compose_answer":
                    return final_answer  # type: ignore[return-value]

                # Intermediate non-tool step: store as observation, then continue
                state.observations.append(f"Non-tool step '{step.id}' result: {final_answer}")
                state.current_step += 1
                continue

            # === Tool steps ===
            if not action_text:
                return self._replan_or_exit(state, step, "No tool call produced.")

            tool_call = maybe_parse_tool_call(action_text)
            if not tool_call:
                return self._replan_or_exit(state, step, f"Invalid tool call: {action_text}")

            tool_name = tool_call["tool_name"]
            args = tool_call["args"]

            if tool_name != step.tool:
                return self._replan_or_exit(
                    state,
                    step,
                    f"Wrong tool called: expected {step.tool}, got {tool_name}",
                )

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

            # Track usage / last call metadata
            state.tools_used[tool_name] = state.tools_used.get(tool_name, 0) + 1
            state.last_tool_name = tool_name
            state.last_tool_args = args
            state.last_tool_result = result

            if failed:
                self._mark_failed(state, tool_name, args)
                return self._replan_or_exit(state, step, result)

            state.current_step += 1

        return state.last_llm_response or "I was unable to complete the request."

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