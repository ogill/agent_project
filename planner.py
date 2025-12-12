# planner.py

import json
from typing import Any, Dict, List

from llm_client import call_llm
from prompts import PLANNER_SYSTEM_PROMPT
from tools import TOOLS
from agent import Plan, PlanStep


class Planner:
    """
    Planner: given a user request, ask the LLM to produce a structured Plan.

    Production-style behaviour:
    - It inspects the TOOLS registry.
    - It passes a human-readable list of tools and their JSON schemas.
    - It instructs the LLM to ONLY use those tools by name.
    - It encourages a final 'compose_answer' step (tool=null) that combines results.
    """

    def generate_plan(self, user_input: str, tools_spec: str = "") -> Plan:
        """
        Create a high-level Plan for the user's request.

        - user_input: the original user query
        - tools_spec: optional override; if empty, it is generated from TOOLS.

        Returns:
            Plan instance populated with PlanSteps.
        """
        if not tools_spec:
            tools_spec = self._build_tools_spec()

        prompt = self._build_planner_prompt(user_input, tools_spec)

        # Debug: show the full planner prompt
        print("\n" + "=" * 80)
        print("[PLANNER DEBUG] FULL PROMPT SENT TO LLM:\n")
        print(prompt)
        print("=" * 80 + "\n")

        raw = call_llm(prompt)

        # Debug: show raw LLM response
        print("\n" + "=" * 80)
        print("[PLANNER DEBUG] RAW RESPONSE FROM LLM:\n")
        print(raw)
        print("=" * 80 + "\n")

        # NEW GUARD: handle completely empty / whitespace-only responses
        if not raw or not raw.strip():
            raise ValueError(
                f"Planner LLM returned an empty response. Raw output was: {raw!r}"
            )

        plan_json = self._parse_json(raw)

        # Debug: show parsed plan summary
        self._debug_plan_summary(plan_json)

        return self._to_plan(plan_json)

    def _build_tools_spec(self) -> str:
        """
        Build a human-readable description of available tools and their schemas
        from the TOOLS registry.

        TOOLS is expected to be:
            {
              tool_name: {
                "fn": callable,
                "description": str,
                "args_model": PydanticModel,
                ...
              },
              ...
            }
        """
        lines: List[str] = ["Available tools:"]

        for name, info in TOOLS.items():
            desc = info.get("description", "")
            args_model = info.get("args_model")
            schema = (
                args_model.model_json_schema()
                if args_model is not None
                else {"description": "No schema available."}
            )

            lines.append(f"- {name}: {desc}")
            lines.append(f"  Args schema: {schema}")

        return "\n".join(lines)

    def _build_planner_prompt(self, user_input: str, tools_spec: str) -> str:
        """
        Build the full prompt string for the planner LLM call.
        """
        return (
            PLANNER_SYSTEM_PROMPT
            + "\n\n"
            + tools_spec
            + "\n\nUser request:\n"
            + user_input
            + "\n\nReturn ONLY the Plan JSON now."
        )

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        """
        Parse the raw LLM output into a dict that _to_plan can understand.

        Supports:
        - Pure JSON object with "goal" and "steps".
        - JSON object wrapped in markdown fences.
        - A bare JSON array of tool calls (we wrap this into a Plan ourselves).

        On failure, raises a ValueError with the original raw output included.
        """
        text = raw.strip()

        # 0) Strip markdown code fences if present
        if text.startswith("```"):
            # Try to remove ```json\n...\n``` wrapper
            first_newline = text.find("\n")
            if first_newline != -1:
                # content after the first line (``` or ```json)
                inner = text[first_newline + 1 :]
                fence_end = inner.rfind("```")
                if fence_end != -1:
                    text = inner[:fence_end].strip()

        # 1) Try direct JSON
        try:
            data = json.loads(text)
            return self._normalise_plan_json(data)
        except json.JSONDecodeError:
            pass

        # 2) Try to extract the first top-level JSON object or array
        start_obj = text.find("{")
        start_arr = text.find("[")

        candidates = [i for i in [start_obj, start_arr] if i != -1]
        if candidates:
            start = min(candidates)
            # Try object/array up to last } or ]
            end_obj = text.rfind("}")
            end_arr = text.rfind("]")

            end = max(end_obj, end_arr)
            if end != -1 and end > start:
                candidate = text[start : end + 1]
                try:
                    data = json.loads(candidate)
                    return self._normalise_plan_json(data)
                except json.JSONDecodeError:
                    pass

        # 3) If we still can't parse, surface a clear error
        raise ValueError(
            "Planner output was not valid JSON in any supported format:\n"
            f"{raw}"
        )

    def _normalise_plan_json(self, data: Any) -> Dict[str, Any]:
        """
        Normalise different planner shapes into a dict with 'goal' and 'steps'.

        Supported inputs:
        - dict with "steps" (already in the right shape).
        - list of tool-call dicts: [{"tool": "...", "args": {...}}, ...]
          -> we wrap into a Plan with autogenerated goal and compose_answer.
        """
        # Case 1: Already the expected shape
        if isinstance(data, dict) and "steps" in data:
            return data

        # Case 2: Bare list of tool calls
        if isinstance(data, list):
            steps = []
            for idx, item in enumerate(data):
                if not isinstance(item, dict) or "tool" not in item:
                    continue  # skip malformed entries

                tool = item.get("tool")
                args = item.get("args", {})
                step_id = f"step_{idx+1}"
                desc = f"Call tool '{tool}' with args {args}"

                steps.append(
                    {
                        "id": step_id,
                        "description": desc,
                        "tool": tool,
                        "args": args,
                        "requires": [],
                    }
                )

            # Append a compose_answer step that depends on all others
            compose_step = {
                "id": "compose_answer",
                "description": "Compose a single coherent answer using all previous tool results.",
                "tool": None,
                "args": None,
                "requires": [s["id"] for s in steps],
            }
            steps.append(compose_step)

            return {
                "goal": "Autogenerated plan from bare tool-call list.",
                "steps": steps,
            }

        # Anything else is unsupported
        raise ValueError(f"Unsupported planner JSON shape: {data!r}")

    def _to_plan(self, data: Dict[str, Any]) -> Plan:
        """
        Convert the JSON dict into a Plan dataclass instance.
        """
        steps_data = data.get("steps", [])
        steps: List[PlanStep] = []

        for s in steps_data:
            step = PlanStep(
                id=s["id"],
                description=s["description"],
                tool=s.get("tool"),
                args=s.get("args"),
                requires=s.get("requires", []),
            )
            steps.append(step)

        return Plan(
            goal=data.get("goal", ""),
            steps=steps,
        )

    def _debug_plan_summary(self, data: Dict[str, Any]) -> None:
        """
        Print a human-readable summary of the parsed plan.
        """
        goal = data.get("goal", "")
        steps = data.get("steps", [])

        print("[PLANNER DEBUG] Parsed plan summary:")
        print(f"  Goal: '{goal}'")
        print(f"  Steps: {len(steps)}")
        for s in steps:
            print(
                f"    - id='{s.get('id')}', "
                f"tool='{s.get('tool')}', "
                f"desc='{s.get('description')}', "
                f"requires={s.get('requires', [])}"
            )
        print()