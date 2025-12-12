# planner.py

import json
from typing import Any, Dict, List

from llm_client import call_llm
from prompts import PLANNER_SYSTEM_PROMPT, PLANNER_REPLAN_SUFFIX
from tools import TOOLS
from agent import Plan, PlanStep


class Planner:
    """
    Planner: given a user request (+ optional context), ask the LLM to produce a structured Plan.

    - Tool-aware (includes tool list + JSON schemas)
    - Enforces a final compose_answer step (tool=null)
    - Supports replanning with observations + failure context
    - Robust to non-JSON LLM outputs via a JSON-repair pass
    - Normalizes plans so every step ALWAYS has: id, description, tool, args, requires
    """

    MAX_JSON_REPAIR_ATTEMPTS = 2

    def generate_plan(
        self,
        user_input: str,
        tools_spec: str = "",
        *,
        observations_text: str = "",
        failure_text: str = "",
        is_replan: bool = False,
    ) -> Plan:
        if not tools_spec:
            tools_spec = self._build_tools_spec()

        prompt = self._build_planner_prompt(
            user_input=user_input,
            tools_spec=tools_spec,
            observations_text=observations_text,
            failure_text=failure_text,
            is_replan=is_replan,
        )

        print("\n" + "=" * 80)
        print("[PLANNER DEBUG] FULL PROMPT SENT TO LLM:\n")
        print(prompt)
        print("=" * 80 + "\n")

        raw = call_llm(prompt)

        print("\n" + "=" * 80)
        print("[PLANNER DEBUG] RAW RESPONSE FROM LLM:\n")
        print(raw)
        print("=" * 80 + "\n")

        plan_json = self._parse_or_repair_json(raw, original_prompt=prompt)
        self._debug_plan_summary(plan_json)
        return self._to_plan(plan_json)

    def _build_tools_spec(self) -> str:
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

    def _build_planner_prompt(
        self,
        *,
        user_input: str,
        tools_spec: str,
        observations_text: str,
        failure_text: str,
        is_replan: bool,
    ) -> str:
        base = (
            PLANNER_SYSTEM_PROMPT
            + "\n\n"
            + tools_spec
            + "\n\nUser request:\n"
            + user_input
        )

        if is_replan:
            base += (
                "\n\n"
                + PLANNER_REPLAN_SUFFIX
                + "\n\nObservations so far:\n"
                + (observations_text.strip() or "(none)")
                + "\n\nFailure / blocker encountered:\n"
                + (failure_text.strip() or "(none)")
            )

        base += "\n\nReturn ONLY the Plan JSON now."
        return base

    # -------------------------------------------------------------------------
    # JSON parsing + repair
    # -------------------------------------------------------------------------

    def _parse_or_repair_json(self, raw: str, *, original_prompt: str) -> Dict[str, Any]:
        """
        Try to parse JSON. If it fails, attempt a JSON-repair pass by asking the LLM
        to output valid JSON ONLY.
        """
        last_err: Exception | None = None
        text = raw

        for attempt in range(self.MAX_JSON_REPAIR_ATTEMPTS + 1):
            try:
                return self._parse_json(text)
            except Exception as e:
                last_err = e
                if attempt >= self.MAX_JSON_REPAIR_ATTEMPTS:
                    break

                # Repair attempt
                repair_prompt = self._build_json_repair_prompt(bad_output=text, original_prompt=original_prompt)
                print("\n" + "=" * 80)
                print(f"[PLANNER DEBUG] JSON REPAIR ATTEMPT {attempt+1} PROMPT SENT TO LLM:\n")
                print(repair_prompt)
                print("=" * 80 + "\n")

                text = call_llm(repair_prompt)

                print("\n" + "=" * 80)
                print(f"[PLANNER DEBUG] JSON REPAIR ATTEMPT {attempt+1} RAW RESPONSE FROM LLM:\n")
                print(text)
                print("=" * 80 + "\n")

        raise ValueError(f"Planner output was not valid JSON after repair attempts. Last error: {last_err}\n\nRaw:\n{raw}")

    def _build_json_repair_prompt(self, *, bad_output: str, original_prompt: str) -> str:
        """
        A strict prompt to coerce the model into returning only valid JSON matching our schema.
        """
        return (
            "You are a JSON repair utility.\n"
            "You MUST return VALID JSON ONLY.\n"
            "No prose, no markdown, no code fences.\n\n"
            "The JSON MUST match this schema exactly:\n"
            "{\n"
            '  "goal": "<string>",\n'
            '  "steps": [\n'
            "    {\n"
            '      "id": "<short identifier for this step>",\n'
            '      "description": "<natural language description>",\n'
            '      "tool": "<tool name or null>",\n'
            '      "args": { ... } or null,\n'
            '      "requires": ["<ids>"]\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Constraints:\n"
            "- Every step MUST include: id, description, tool, args, requires.\n"
            '- The FINAL step MUST have id="compose_answer", tool=null, args=null.\n'
            '- compose_answer.requires MUST include every prior step id.\n'
            "- Tools must be one of the tools listed in the original prompt.\n\n"
            "Original planner instructions (for context):\n"
            + original_prompt
            + "\n\nBad model output to repair:\n"
            + bad_output
            + "\n\nReturn ONLY the repaired Plan JSON now."
        )

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        """
        Parse the raw LLM output into a dict with 'goal' and 'steps'.

        Supports:
        - Pure JSON object with {goal, steps}
        - JSON object wrapped in ``` fences (even if there is text around it)
        - Extract-first-object strategy for messy outputs
        - If it returns a bare array of tool calls, we normalise it
        """
        if raw is None:
            raise ValueError("Planner output was empty (None).")

        text = raw.strip()
        if not text:
            raise ValueError("Planner output was empty string.")

        # 0) Strip outer ``` fences if present
        if "```" in text:
            first = text.find("```")
            last = text.rfind("```")
            if first != -1 and last != -1 and last > first:
                inner = text[first + 3 : last].lstrip()
                if inner.lower().startswith("json"):
                    nl = inner.find("\n")
                    if nl != -1:
                        inner = inner[nl + 1 :]
                text = inner.strip()

        # 1) Direct parse
        try:
            data = json.loads(text)
            return self._normalise_plan_json(data)
        except json.JSONDecodeError:
            pass

        # 2) Extract first {...} or [...]
        start_obj = text.find("{")
        start_arr = text.find("[")
        starts = [i for i in [start_obj, start_arr] if i != -1]
        if starts:
            start = min(starts)
            end_obj = text.rfind("}")
            end_arr = text.rfind("]")
            end = max(end_obj, end_arr)
            if end != -1 and end > start:
                candidate = text[start : end + 1]
                data = json.loads(candidate)
                return self._normalise_plan_json(data)

        raise ValueError(f"Planner output was not valid JSON in any supported format:\n{raw}")

    # -------------------------------------------------------------------------
    # Normalisation / hardening
    # -------------------------------------------------------------------------

    def _normalise_plan_json(self, data: Any) -> Dict[str, Any]:
        # Case 1: expected shape
        if isinstance(data, dict) and "steps" in data:
            plan = data
        # Case 2: bare list of tool calls -> wrap
        elif isinstance(data, list):
            steps: List[Dict[str, Any]] = []
            for idx, item in enumerate(data):
                if not isinstance(item, dict) or "tool" not in item:
                    continue
                tool = item.get("tool")
                args = item.get("args", {})
                step_id = f"step_{idx+1}"
                steps.append(
                    {
                        "id": step_id,
                        "description": f"Call tool '{tool}'.",
                        "tool": tool,
                        "args": args,
                        "requires": [],
                    }
                )
            plan = {"goal": "Autogenerated plan from tool-call list.", "steps": steps}
        else:
            raise ValueError(f"Unsupported planner JSON shape: {type(data)}")

        plan = self._ensure_compose_answer(plan)
        plan = self._ensure_step_fields(plan)
        self._validate_tools(plan)
        return plan

    def _ensure_compose_answer(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        steps: List[Dict[str, Any]] = plan.get("steps", [])
        if not isinstance(steps, list):
            raise ValueError("Planner JSON 'steps' must be a list.")

        # Ensure goal exists
        if "goal" not in plan or not isinstance(plan.get("goal"), str):
            plan["goal"] = str(plan.get("goal") or "").strip()

        # Check if compose_answer already exists
        has_compose = any(isinstance(s, dict) and s.get("id") == "compose_answer" for s in steps)
        if not has_compose:
            requires = [s.get("id") for s in steps if isinstance(s, dict) and s.get("id")]
            steps.append(
                {
                    "id": "compose_answer",
                    "description": "Read all previous step results and produce a single coherent answer for the user.",
                    "tool": None,
                    "args": None,
                    "requires": requires,
                }
            )
            plan["steps"] = steps
            return plan

        # If compose exists, force it to be last and force tool/args null
        # (This avoids models putting compose_answer in the middle.)
        compose = None
        rest = []
        for s in steps:
            if isinstance(s, dict) and s.get("id") == "compose_answer":
                compose = s
            else:
                rest.append(s)

        if compose is None:
            plan["steps"] = steps
            return plan

        compose["tool"] = None
        compose["args"] = None
        compose.setdefault(
            "description",
            "Read all previous step results and produce a single coherent answer for the user.",
        )

        # Make compose depend on all non-compose steps
        compose["requires"] = [s.get("id") for s in rest if isinstance(s, dict) and s.get("id")]

        plan["steps"] = rest + [compose]
        return plan

    def _ensure_step_fields(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Guarantee every step contains the keys expected by the runtime.
        This prevents KeyError crashes like: 'description'
        """
        steps: List[Dict[str, Any]] = plan.get("steps", [])
        fixed: List[Dict[str, Any]] = []

        for idx, s in enumerate(steps):
            if not isinstance(s, dict):
                s = {}

            step_id = s.get("id") or f"step_{idx+1}"
            tool = s.get("tool", None)

            # description: always required
            desc = s.get("description")
            if not isinstance(desc, str) or not desc.strip():
                if step_id == "compose_answer":
                    desc = "Read all previous step results and produce a single coherent answer for the user."
                elif tool:
                    desc = f"Call tool '{tool}'."
                else:
                    desc = "Perform a non-tool reasoning step."

            # requires: always a list
            req = s.get("requires", [])
            if req is None:
                req = []
            if not isinstance(req, list):
                req = [req]

            # args: ensure null for tool=null; otherwise ensure dict or null
            args = s.get("args", None)
            if tool is None:
                args = None
            else:
                if args is None:
                    args = {}
                if not isinstance(args, dict):
                    # last resort: wrap
                    args = {"value": args}

            fixed.append(
                {
                    "id": step_id,
                    "description": desc,
                    "tool": tool,
                    "args": args,
                    "requires": req,
                }
            )

        plan["steps"] = fixed
        return plan

    def _validate_tools(self, plan: Dict[str, Any]) -> None:
        """
        Fail early if planner invents tools.
        """
        allowed = set(TOOLS.keys())
        for s in plan.get("steps", []):
            tool = s.get("tool")
            if tool is None:
                continue
            if tool not in allowed:
                raise ValueError(f"Planner invented unknown tool '{tool}'. Allowed: {sorted(allowed)}")

    # -------------------------------------------------------------------------
    # Conversion / debug
    # -------------------------------------------------------------------------

    def _to_plan(self, data: Dict[str, Any]) -> Plan:
        steps_data = data.get("steps", [])
        steps: List[PlanStep] = []

        for s in steps_data:
            step = PlanStep(
                id=s["id"],
                description=s.get("description") or "",
                tool=s.get("tool"),
                args=s.get("args"),
                requires=s.get("requires", []),
            )
            steps.append(step)

        return Plan(goal=data.get("goal", ""), steps=steps)

    def _debug_plan_summary(self, data: Dict[str, Any]) -> None:
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