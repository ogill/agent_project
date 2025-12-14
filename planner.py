# planner.py
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List

from llm_client import call_llm
from tools import TOOLS
from plan_types import Plan, PlanStep
from config import PROMPT_MODE
from prompts import get_planner_prompt

DEFAULT_REPLAN_SUFFIX = """
REPLANNING MODE:

You are replanning because the current plan could not be completed.

You will be given:
- Observations so far (tool outputs already collected)
- A specific Failure / blocker encountered

Your task:
- Produce a NEW plan (same JSON schema as before) that still achieves the user's request.
- Use ONLY the listed tools. Do NOT invent tools.
- Avoid repeating already completed work if Observations already contain the needed info.

CRITICAL RULE:
- You MUST NOT include any tool step that already failed, unless the failure reason explicitly states it was transient.
- If a tool failure is already observed, you must reason from the observation instead of calling the tool again.
"""


class Planner:
    """
    Planner: ask the LLM to produce a structured Plan (JSON).

    Stabilisations:
    - Deterministic local sanitisation for common "almost JSON" model bugs (e.g. `"https"://`)
    - Safer LLM JSON repair prompt (embeds bad output as escaped JSON string)
    - Normalises steps (id/description/tool/args/requires)
    - Enforces compose_answer final step and dependencies
    - Replanning is deterministic: we do NOT trust the model plan structure in replan mode
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
        replan_suffix: str | None = None,
    ) -> Plan:
        if not tools_spec:
            tools_spec = self._build_tools_spec()

        prompt = self._build_planner_prompt(
            user_input=user_input,
            tools_spec=tools_spec,
            observations_text=observations_text,
            failure_text=failure_text,
            is_replan=is_replan,
            replan_suffix=replan_suffix or DEFAULT_REPLAN_SUFFIX,
        )

        print("\n" + "=" * 80)
        print("[PLANNER DEBUG] FULL PROMPT SENT TO LLM:\n " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(prompt)
        print("=" * 80 + "\n")

        raw = call_llm(prompt)

        print("\n" + "=" * 80)
        print("[PLANNER DEBUG] RAW RESPONSE FROM LLM:\n " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(raw)
        print("=" * 80 + "\n")

        # Parse/repair JSON to keep logging + transparency
        plan_json = self._parse_or_repair_json(
            raw,
            original_prompt=prompt,
            observations_text=observations_text,
            failure_text=failure_text,
            is_replan=is_replan,
        )

        # IMPORTANT: in replanning mode, we do NOT trust the model plan structure.
        # We deterministically build the plan from observed failure/observations.
        if is_replan:
            plan_json = self._deterministic_replan_plan(
                user_input=user_input,
                observations_text=observations_text,
                failure_text=failure_text,
            )

        self._debug_plan_summary(plan_json)
        return self._to_plan(plan_json)

    # --------------------------- prompt construction ---------------------------

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
        replan_suffix: str,
    ) -> str:
        base = (
            get_planner_prompt(PROMPT_MODE)
            + "\n\nIMPORTANT:\n"
            + "- Do NOT create intermediate non-tool steps (tool=null).\n"
            + "- The ONLY non-tool step allowed is the FINAL 'compose_answer' step.\n"
            + "- If you need to 'use observations', do it inside compose_answer, not as separate steps.\n\n"
            + tools_spec
            + "\n\nUser request:\n"
            + user_input
        )

        if is_replan:
            base += (
                "\n\n"
                + replan_suffix.strip()
                + "\n\nObservations so far:\n"
                + (observations_text.strip() or "(none)")
                + "\n\nFailure / blocker encountered:\n"
                + (failure_text.strip() or "(none)")
            )

        base += "\n\nReturn ONLY the Plan JSON now."
        return base

    # --------------------------- deterministic replanning ---------------------------

    def _deterministic_replan_plan(
        self,
        *,
        user_input: str,
        observations_text: str,
        failure_text: str,
    ) -> Dict[str, Any]:
        """
        Deterministic replanning: never retry failed tools automatically.
        Always summarise the failure+observations, then compose final answer.
        """
        payload_parts: List[str] = []
        if failure_text.strip():
            payload_parts.append(failure_text.strip())
        if observations_text.strip():
            payload_parts.append("Observations:\n" + observations_text.strip())

        payload = "\n\n".join(payload_parts).strip()

        # If somehow empty, still provide a safe structure
        if not payload:
            payload = (
                "A previous tool call failed, but no failure text or observations were provided. "
                "Explain that the agent could not complete the request due to a prior tool error."
            )

        return {
            "goal": f"Respond to the user request using observed failures/observations: {user_input}",
            "steps": [
                {
                    "id": "summarize_failure",
                    "description": "Summarise the failure and observations in a compact form.",
                    "tool": "summarize_text",
                    "args": {"text": payload, "bullets": 3},
                    "requires": [],
                },
                {
                    "id": "compose_answer",
                    "description": "Read all previous step results and produce one coherent answer.",
                    "tool": None,
                    "args": None,
                    "requires": ["summarize_failure"],
                },
            ],
        }

    # --------------------------- JSON parsing + repair ---------------------------

    def _parse_or_repair_json(
        self,
        raw: str,
        *,
        original_prompt: str,
        observations_text: str,
        failure_text: str,
        is_replan: bool,
    ) -> Dict[str, Any]:
        last_err: Exception | None = None
        text = raw or ""

        text = self._sanitize_common_model_json_bugs(text)

        for attempt in range(self.MAX_JSON_REPAIR_ATTEMPTS + 1):
            try:
                return self._parse_json(
                    text,
                    observations_text=observations_text,
                    failure_text=failure_text,
                    is_replan=is_replan,
                )
            except Exception as e:
                last_err = e
                if attempt >= self.MAX_JSON_REPAIR_ATTEMPTS:
                    break

                repair_prompt = self._build_json_repair_prompt(
                    bad_output=text,
                    original_prompt=original_prompt,
                )

                print("\n" + "=" * 80)
                print(f"[PLANNER DEBUG] JSON REPAIR ATTEMPT {attempt+1} PROMPT SENT TO LLM:\n")
                print(repair_prompt)
                print("=" * 80 + "\n")

                text = call_llm(repair_prompt)
                text = self._sanitize_common_model_json_bugs(text)

                print("\n" + "=" * 80)
                print(f"[PLANNER DEBUG] JSON REPAIR ATTEMPT {attempt+1} RAW RESPONSE FROM LLM:\n")
                print(text)
                print("=" * 80 + "\n")

        raise ValueError(
            "Planner output was not valid JSON after repair attempts. "
            f"Last error: {last_err}\n\nRaw:\n{raw}"
        )

    def _sanitize_common_model_json_bugs(self, s: str) -> str:
        if not s:
            return s

        s = s.replace('"https"://', "https://")
        s = s.replace('"http"://', "http://")
        s = s.replace('from "https"://', "from https://")
        s = s.replace('from "http"://', "from http://")

        s = s.replace("“", '"').replace("”", '"').replace("’", "'")
        return s

    def _parse_json(
        self,
        raw: str,
        *,
        observations_text: str,
        failure_text: str,
        is_replan: bool,
    ) -> Dict[str, Any]:
        if raw is None:
            raise ValueError("Planner output was empty (None).")

        text = raw.strip()
        if not text:
            raise ValueError("Planner output was empty string.")

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

        try:
            data = json.loads(text)
            return self._normalise_plan_json(
                data,
                observations_text=observations_text,
                failure_text=failure_text,
                is_replan=is_replan,
            )
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            candidate = self._sanitize_common_model_json_bugs(candidate)
            data = json.loads(candidate)
            return self._normalise_plan_json(
                data,
                observations_text=observations_text,
                failure_text=failure_text,
                is_replan=is_replan,
            )

        raise ValueError(f"Planner output was not valid JSON:\n{raw}")

    def _build_json_repair_prompt(self, *, bad_output: str, original_prompt: str) -> str:
        bad_as_json_string = json.dumps(bad_output)
        allowed_tools = ", ".join(sorted(TOOLS.keys()))

        return (
            "You are a JSON repair utility.\n"
            "You MUST return VALID JSON ONLY.\n"
            "No prose, no markdown, no code fences.\n\n"
            "Return JSON matching this schema exactly:\n"
            '{\n  "goal": "<string>",\n  "steps": [\n'
            '    {\n      "id": "<string>",\n      "description": "<string>",\n'
            '      "tool": "<tool name or null>",\n      "args": { ... } or null,\n'
            '      "requires": ["<ids>"]\n    }\n  ]\n}\n\n'
            "Constraints:\n"
            "- FINAL step must be id='compose_answer' with tool=null and args=null.\n"
            "- compose_answer.requires must include every prior step id.\n"
            f"- tool names must be one of: {allowed_tools}\n\n"
            "The prior model output is provided as an escaped JSON string below.\n"
            "Parse it, fix it, and output ONLY the repaired plan JSON.\n\n"
            f"BAD_OUTPUT_JSON_STRING: {bad_as_json_string}\n"
        )

    # --------------------------- normalisation / guardrails ---------------------------

    def _normalise_plan_json(
        self,
        data: Any,
        *,
        observations_text: str,
        failure_text: str,
        is_replan: bool,
    ) -> Dict[str, Any]:
        if not isinstance(data, dict) or "steps" not in data:
            raise ValueError(f"Unsupported planner JSON shape: {type(data)}")

        plan: Dict[str, Any] = data

        plan = self._ensure_step_fields(plan)
        plan = self._ensure_compose_answer(plan)
        plan = self._prune_non_tool_steps(plan)
        plan = self._sanitize_requires(plan)
        self._validate_tools(plan)

        return plan

    def _ensure_step_fields(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        out: List[Dict[str, Any]] = []
        for i, s in enumerate(plan.get("steps", [])):
            if not isinstance(s, dict):
                continue

            tool = s.get("tool", None)
            args = s.get("args", {})

            # args must be dict for tool steps
            if tool is not None and not isinstance(args, dict):
                args = {}

            req = s.get("requires", [])
            if not isinstance(req, list):
                req = [req]

            out.append(
                {
                    "id": s.get("id", f"step_{i+1}"),
                    "description": s.get("description", ""),
                    "tool": tool,
                    "args": None if tool is None else args,
                    "requires": req,
                }
            )

        plan["steps"] = out
        return plan

    def _ensure_compose_answer(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        steps = plan.get("steps", [])
        if not any(s.get("id") == "compose_answer" for s in steps):
            steps.append(
                {
                    "id": "compose_answer",
                    "description": "Reads all previous step results and produces one coherent answer.",
                    "tool": None,
                    "args": None,
                    "requires": [],
                }
            )
        plan["steps"] = steps
        return plan

    def _prune_non_tool_steps(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        steps = plan.get("steps", [])
        tools = [s for s in steps if s.get("tool") is not None]
        compose = next((s for s in steps if s.get("id") == "compose_answer"), None)

        if compose is None:
            compose = {
                "id": "compose_answer",
                "description": "Reads all previous step results and produces one coherent answer.",
                "tool": None,
                "args": None,
                "requires": [],
            }

        plan["steps"] = tools + [compose]
        return plan

    def _sanitize_requires(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        steps = plan.get("steps", [])
        ids = {s.get("id") for s in steps}

        for s in steps:
            req = s.get("requires", [])
            if not isinstance(req, list):
                req = [req]
            s["requires"] = [r for r in req if r in ids]

        # Force compose_answer to depend on ALL prior tool steps
        compose = next((s for s in steps if s.get("id") == "compose_answer"), None)
        if compose is not None:
            prior_ids = [s.get("id") for s in steps if s.get("id") != "compose_answer"]
            compose["requires"] = prior_ids

        return plan

    def _validate_tools(self, plan: Dict[str, Any]) -> None:
        allowed = set(TOOLS.keys())
        for s in plan.get("steps", []):
            tool = s.get("tool")
            if tool and tool not in allowed:
                raise ValueError(f"Planner invented unknown tool '{tool}'.")

    # --------------------------- conversions / debug ---------------------------

    def _to_plan(self, data: Dict[str, Any]) -> Plan:
        return Plan(
            goal=data.get("goal", ""),
            steps=[
                PlanStep(
                    id=s["id"],
                    description=s.get("description", ""),
                    tool=s.get("tool"),
                    args=s.get("args"),
                    requires=s.get("requires", []),
                )
                for s in data.get("steps", [])
            ],
        )

    def _debug_plan_summary(self, data: Dict[str, Any]) -> None:
        print("[PLANNER DEBUG] Parsed plan summary:")
        print(f"  Goal: {data.get('goal')}")
        for s in data.get("steps", []):
            print(f"    - {s.get('id')} | tool={s.get('tool')} | requires={s.get('requires')}")
        print()