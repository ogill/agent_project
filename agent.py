# agent.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
import re

from config import DEBUG_AGENT, DEBUG_AGENT_PROMPTS, MAX_REACT_STEPS
from llm_client import call_llm
from tools import TOOLS  # TOOLS is a dict: { tool_name: {"fn": callable, "description": str, "args_model": PydanticModel}, ... }
from prompts import get_react_system_prompt

SYSTEM_PROMPT = get_react_system_prompt()


def build_react_prompt(user_input: str, observations: List[str]) -> str:
    """
    (Legacy helper, not used in Stage 3.3 but kept for reference.)
    """
    prompt = SYSTEM_PROMPT + "\n\nUser: " + user_input + "\n"

    if observations:
        prompt += "\nObservations so far:\n"
        for idx, obs in enumerate(observations, start=1):
            prompt += f"{idx}) {obs}\n"

    prompt += "\nAssistant:"
    return prompt


def build_step_aware_prompt(
    user_input: str,
    observations: List[str],
    plan_step: "PlanStep",
    current_step_index: int,
    total_steps: int,
) -> str:
    """
    Stage 3.3:
    Build a ReAct prompt that is aware of the current plan step.

    This adds a small "plan context" section before the normal
    user/observations, so the LLM knows:
    - which step of the plan it's executing
    - what the step is supposed to do
    - which tool (if any) is expected
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
    Parse a ReAct-style LLM response into:
    - Thought: ...
    - Action: ...
    - Final Answer: ...

    This version supports multi-line Final Answer:
    everything after the 'Final Answer:' line is captured.
    """
    thought: Optional[str] = None
    action: Optional[str] = None
    final_answer: Optional[str] = None

    lines = text.splitlines()
    final_answer_lines: Optional[List[str]] = None

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        lower = stripped.lower()

        if lower.startswith("thought:") and thought is None:
            thought = stripped[len("thought:") :].strip()

        elif lower.startswith("action:") and action is None:
            action = stripped[len("action:") :].strip()

        elif lower.startswith("final answer:") and final_answer_lines is None:
            after = stripped[len("final answer:") :].lstrip()
            final_answer_lines = []
            if after:
                final_answer_lines.append(after)

            for j in range(i + 1, len(lines)):
                final_answer_lines.append(lines[j].rstrip())
            break

        i += 1

    if final_answer_lines is not None:
        joined = "\n".join(final_answer_lines).strip()
        final_answer = joined if joined else None

    return {
        "thought": thought,
        "action": action,
        "final_answer": final_answer,
    }


def maybe_parse_tool_call(action_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse an Action line of the form:

        CALL_TOOL:get_weather({"city": "London"})

    Returns:
        {"tool_name": "get_weather", "args": {"city": "London"}}
    or None if it doesn't match the expected pattern.
    """
    if not action_text:
        return None

    action_text = action_text.strip()

    # Must start with CALL_TOOL:
    if not action_text.startswith("CALL_TOOL:"):
        return None

    # Strip prefix
    rest = action_text[len("CALL_TOOL:"):].strip()  # e.g. 'get_time({"city": "London"})'

    # Find the first '(' and the last ')'
    open_idx = rest.find("(")
    close_idx = rest.rfind(")")

    if open_idx == -1 or close_idx == -1 or close_idx < open_idx:
        return None

    tool_name = rest[:open_idx].strip()
    args_str = rest[open_idx + 1 : close_idx].strip()

    if not tool_name:
        return None

    # Handle empty args
    if not args_str or args_str == "{}":
        args: Dict[str, Any] = {}
    else:
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            return None

    return {
        "tool_name": tool_name,
        "args": args,
    }


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
    """
    Container for everything the agent "knows" during a single turn.
    """
    user_input: str
    observations: List[str] = field(default_factory=list)
    step: int = 0

    # Optional debug / introspection fields
    tools_used: Dict[str, int] = field(default_factory=dict)
    last_tool_name: Optional[str] = None
    last_tool_args: Optional[Dict[str, Any]] = None
    last_tool_result: Optional[str] = None
    last_llm_response: Optional[str] = None

    # Planning fields
    plan: Optional[Plan] = None
    current_step: int = 0


class SimpleAgent:
    """
    Production-ish Hybrid Planner + ReAct Agent:

    - First, generate a high-level Plan with the Planner.
      (Planner always ends with a 'compose_answer' step, tool=None.)
    - Store the Plan on AgentState (state.plan, state.current_step).
    - For each PlanStep:
        * Build a step-aware ReAct prompt.
        * Let the LLM either call a tool or, on non-tool steps (esp. compose_answer),
          produce the final user-facing answer.
    - Tools are never returned directly to the user; only compose_answer
      returns the final answer.
    """

    def __init__(self, planner: Optional["Planner"] = None) -> None:
        """
        Allow dependency injection of a Planner for testing, but
        default to a real Planner instance if none is provided.

        Planner is imported lazily to avoid circular imports.
        """
        if planner is None:
            from planner import Planner  # local import to avoid circular import
            planner = Planner()
        self.planner = planner

    def handle_turn(self, user_input: str) -> str:
        # 1) Initialise AgentState
        state = AgentState(user_input=user_input)

        # 2) Generate a Plan for this user request
        plan = self.planner.generate_plan(user_input, tools_spec="")
        state.plan = plan
        state.current_step = 0

        total_plan_steps = len(plan.steps)

        if DEBUG_AGENT:
            print(f"[DEBUG] Generated plan with {total_plan_steps} steps.")
            for ps in plan.steps:
                print(
                    f"[DEBUG]  - Step '{ps.id}': {ps.description} "
                    f"(tool={ps.tool}, requires={ps.requires})"
                )

        # 3) Execute each PlanStep using a ReAct cycle
        react_step = 0  # counts total ReAct iterations (safety with MAX_REACT_STEPS)

        while (
            state.plan is not None
            and state.current_step < total_plan_steps
            and react_step < MAX_REACT_STEPS
        ):
            react_step += 1
            state.step = react_step

            current_plan_step = state.plan.steps[state.current_step]

            if DEBUG_AGENT:
                print(
                    f"\n[DEBUG] ===== ReAct step {state.step} "
                    f"(plan step {state.current_step + 1}/{total_plan_steps}) ====="
                )

            # 3a) Build a step-aware prompt
            prompt = build_step_aware_prompt(
                state.user_input,
                state.observations,
                current_plan_step,
                state.current_step,
                total_plan_steps,
            )

            if DEBUG_AGENT_PROMPTS:
                print("[DEBUG PROMPT] Prompt sent to LLM:\n")
                print(prompt)
                print("[DEBUG PROMPT] End prompt\n")

            # 3b) Call LLM
            llm_response = call_llm(prompt)
            state.last_llm_response = llm_response

            if DEBUG_AGENT:
                print("[DEBUG] LLM raw response:")
                print(llm_response)
                print("[DEBUG] --- LLM Parsed Sections ---")
                debug_parts = parse_react_response(llm_response)
                print(f"[DEBUG] Thought:\n  {debug_parts.get('thought')}")
                print(f"[DEBUG] Action:\n  {debug_parts.get('action')}")
                print(f"[DEBUG] Final Answer:\n  {debug_parts.get('final_answer')}")
                print("[DEBUG] ----------------------------")

            # 3c) Parse ReAct structure
            react_parts = parse_react_response(llm_response)
            if DEBUG_AGENT:
                print("[DEBUG] --- Parsed ReAct Sections ---")
                print(f"[DEBUG] Thought:\n  {react_parts.get('thought')}")
                print(f"[DEBUG] Action:\n  {react_parts.get('action')}")
                print(f"[DEBUG] Final Answer:\n  {react_parts.get('final_answer')}")
                print("[DEBUG] ------------------------------")

            raw_action = react_parts.get("action")
            final_answer = react_parts.get("final_answer")

            # Normalise Action: treat 'NONE' (any case, extra spaces) as no action
            if raw_action and raw_action.strip().upper() == "NONE":
                action_text = None
            else:
                action_text = raw_action

            # 3d) If there is no action (or Action=NONE) and we have a final answer → we're done
            if not action_text and final_answer:
                if DEBUG_AGENT:
                    print(
                        "[DEBUG] No Action (or Action=NONE), but Final Answer found. "
                        "Returning Final Answer."
                    )
                return final_answer

            # 3e) If there is no action and no final answer → fallback with raw response
            if not action_text:
                if DEBUG_AGENT:
                    print(
                        "[DEBUG] No Action or Final Answer found, "
                        "returning raw LLM response."
                    )
                return llm_response

            if DEBUG_AGENT:
                print("[DEBUG] Action text:", repr(action_text))

            # 3f) Parse tool call
            tool_call = maybe_parse_tool_call(action_text)
            if tool_call is None:
                if DEBUG_AGENT:
                    print(
                        "[DEBUG] Action line did not contain a valid CALL_TOOL, "
                        "returning raw LLM response."
                    )
                return llm_response

            if DEBUG_AGENT:
                print("[DEBUG] Parsed CALL_TOOL:", tool_call)

            tool_name = tool_call["tool_name"]
            args = tool_call["args"]

            tool_def = TOOLS.get(tool_name)
            if tool_def is None:
                err = (
                    f"The model requested an unknown tool '{tool_name}'. "
                    f"Raw response was: {llm_response}"
                )
                if DEBUG_AGENT:
                    print("[DEBUG]", err)
                return err

            try:
                tool_fn = tool_def["fn"]
                args_model = tool_def["args_model"]
                try:
                    validated_args = args_model.model_validate(args)
                except Exception as e:
                    return f"Tool '{tool_name}' argument validation error: {e}"

                tool_result = tool_fn(validated_args)
            except Exception as e:
                tool_result = f"Tool '{tool_name}' failed with error: {e!r}"

            if DEBUG_AGENT:
                print(f"[DEBUG] Tool '{tool_name}' called with args:", args)
                print(f"[DEBUG] Tool '{tool_name}' result:", tool_result)

            # 3g) Add an observation describing this tool call
            obs = (
                f"The tool '{tool_name}' was called with arguments {json.dumps(args)} "
                f"and returned this result: {tool_result}"
            )
            state.observations.append(obs)

            # Update debug fields on state
            state.last_tool_name = tool_name
            state.last_tool_args = args
            state.last_tool_result = tool_result
            state.tools_used[tool_name] = state.tools_used.get(tool_name, 0) + 1

            # 3h) Mark this plan step as complete and move to the next one
            state.current_step += 1

        # 4) If we exit the loop without a clear Final Answer, return last raw response
        if DEBUG_AGENT:
            print(
                "[DEBUG] Reached the end of the plan or MAX_REACT_STEPS "
                "without a clear Final Answer. "
                "Returning last LLM response."
            )
        return state.last_llm_response or "I was unable to produce an answer."