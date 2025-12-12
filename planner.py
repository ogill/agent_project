# planner.py

import json
import re
from typing import Any, Dict, List, Optional
from datetime import datetime

from llm_client import call_llm
from tools import TOOLS
from agent import Plan, PlanStep
from config import PROMPT_MODE
from prompts import get_planner_prompt, PLANNER_REPLAN_SUFFIX


class Planner:
    """
    Planner: given a user request (+ optional context), ask the LLM to produce a structured Plan.

    - Tool-aware (includes tool list + JSON schemas)
    - Enforces a final compose_answer step (tool=null)
    - Supports replanning with observations + failure context
    - Robust to non-JSON LLM outputs via:
        1) deterministic local JSON repairs
        2) optional LLM JSON-repair pass
    - Normalizes plans so every step ALWAYS has: id, description, tool, args, requires
    - Prunes intermediate non-tool steps (tool=null) so executor only needs LLM for compose_answer
    """

    MAX_JSON_REPAIR_ATTEMPTS = 2  # LLM repair attempts (after local repairs)

    # ---------------------------------------------------------------------
    # Public entry
    # ---------------------------------------------------------------------

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
        print("[PLANNER DEBUG] FULL PROMPT SENT TO LLM:\n " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(prompt)
        print("=" * 80 + "\n")

        raw = call_llm(prompt)

        print("\n" + "=" * 80)
        print("[PLANNER DEBUG] RAW RESPONSE FROM LLM:\n " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(raw)
        print("=" * 80 + "\n")

        plan_json = self._parse_or_repair_json(
            raw,
            original_prompt=prompt,
            observations_text=observations_text,
            failure_text=failure_text,
            is_replan=is_replan,
        )

        self._debug_plan_summary(plan_json)
        return self._to_plan(plan_json)

    # ---------------------------------------------------------------------
    # Prompt construction
    # ---------------------------------------------------------------------

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
                + PLANNER_REPLAN_SUFFIX
                + "\n\nObservations so far:\n"
                + (observations_text.strip() or "(none)")
                + "\n\nFailure / blocker encountered:\n"
                + (failure_text.strip() or "(none)")
            )

        base += "\n\nReturn ONLY the Plan JSON now."
        return base

    # ---------------------------------------------------------------------
    # JSON parsing + repair
    # ---------------------------------------------------------------------

    def _parse_or_repair_json(
        self,
        raw: str,
        *,
        original_prompt: str,
        observations_text: str,
        failure_text: str,
        is_replan: bool,
    ) -> Dict[str, Any]:
        """
        Strategy:
        1) Deterministically extract candidate JSON and apply local fixes.
        2) Try json.loads.
        3) If still failing, do up to MAX_JSON_REPAIR_ATTEMPTS LLM repair calls,
           but *always* wrap bad output as an escaped JSON string in the repair prompt,
           then run local fixes again before parsing.
        """
        last_err: Optional[Exception] = None
        text = raw or ""

        # First pass: local extraction + fixes (no LLM)
        text = self._extract_json_candidate(text)
        text = self._local_json_fixes(text)

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
                    bad_output=text, original_prompt=original_prompt
                )

                print("\n" + "=" * 80)
                print(f"[PLANNER DEBUG] JSON REPAIR ATTEMPT {attempt+1} PROMPT SENT TO LLM:\n")
                print(repair_prompt)
                print("=" * 80 + "\n")

                text = call_llm(repair_prompt) or ""

                print("\n" + "=" * 80)
                print(f"[PLANNER DEBUG] JSON REPAIR ATTEMPT {attempt+1} RAW RESPONSE FROM LLM:\n")
                print(text)
                print("=" * 80 + "\n")

                # Always re-extract + locally fix after LLM repair
                text = self._extract_json_candidate(text)
                text = self._local_json_fixes(text)

        raise ValueError(
            f"Planner output was not valid JSON after repair attempts. "
            f"Last error: {last_err}\n\nRaw:\n{raw}"
        )

    def _build_json_repair_prompt(self, *, bad_output: str, original_prompt: str) -> str:
        """
        IMPORTANT: embed bad_output as a JSON string (escaped) so it can't break this prompt.
        """
        bad_output_escaped = json.dumps(bad_output)

        return (
            "You are a JSON repair utility.\n"
            "You MUST return VALID JSON ONLY.\n"
            "No prose, no markdown, no code fences.\n\n"
            "Return a JSON object matching this schema exactly:\n"
            "{\n"
            '  "goal": "<string>",\n'
            '  "steps": [\n'
            "    {\n"
            '      "id": "<short identifier>",\n'
            '      "description": "<string>",\n'
            '      "tool": "<tool name or null>",\n'
            '      "args": { ... } or null,\n'
            '      "requires": ["<ids>"]\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Constraints:\n"
            "- Every step MUST include: id, description, tool, args, requires.\n"
            '- The FINAL step MUST have id="compose_answer", tool=null, args=null.\n'
            "- Do NOT include any non-tool steps (tool=null) except compose_answer.\n"
            "- compose_answer.requires MUST include every prior step id.\n"
            "- Tools must be one of the tools listed in the original prompt.\n\n"
            "Original planner instructions (for context):\n"
            + original_prompt
            + "\n\nBad model output to repair (as an escaped string):\n"
            + bad_output_escaped
            + "\n\nReturn ONLY the repaired Plan JSON now."
        )

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

        # Strip ``` fences if any slipped through
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

        # One more deterministic cleanup before loads
        text = self._extract_json_candidate(text)
        text = self._local_json_fixes(text)

        data = json.loads(text)
        return self._normalise_plan_json(
            data,
            observations_text=observations_text,
            failure_text=failure_text,
            is_replan=is_replan,
        )

    # ---------------------------------------------------------------------
    # Deterministic JSON hardening
    # ---------------------------------------------------------------------

    def _extract_json_candidate(self, text: str) -> str:
        """
        Extract the first JSON object/array from a messy blob.
        If already clean, returns as-is.
        """
        if not text:
            return text
        t = text.strip()

        # Fast path
        if (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]")):
            return t

        # Find first { or [
        start_obj = t.find("{")
        start_arr = t.find("[")
        starts = [i for i in (start_obj, start_arr) if i != -1]
        if not starts:
            return t

        start = min(starts)
        end = max(t.rfind("}"), t.rfind("]"))
        if end == -1 or end <= start:
            return t

        return t[start : end + 1].strip()

    def _local_json_fixes(self, text: str) -> str:
        """
        Fix common LLM JSON issues deterministically.
        - URL corruption inside quoted strings: "https":// -> https://
        - Unquoted keys like: bullets: 3  -> "bullets": 3
        """
        if not text:
            return text

        t = text

        # Fix URL corruption everywhere (safe even if it doesn't occur)
        t = t.replace('"https"://', "https://")
        t = t.replace('"http"://', "http://")

        # Quote bare keys like: { bullets: 3 } or , bullets: 3
        # Only applies when the key is not already quoted.
        t = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)', r'\1"\2"\3', t)

        return t

    # ---------------------------------------------------------------------
    # Normalisation / guardrails
    # ---------------------------------------------------------------------

    def _normalise_plan_json(
        self,
        data: Any,
        *,
        observations_text: str,
        failure_text: str,
        is_replan: bool,
    ) -> Dict[str, Any]:
        if isinstance(data, dict) and "steps" in data:
            plan = data
        elif isinstance(data, list):
            steps = []
            for i, item in enumerate(data):
                if isinstance(item, dict) and "tool" in item:
                    steps.append(
                        {
                            "id": f"step_{i+1}",
                            "description": f"Call tool '{item.get('tool')}'.",
                            "tool": item.get("tool"),
                            "args": item.get("args", {}),
                            "requires": [],
                        }
                    )
            plan = {"goal": "Autogenerated plan.", "steps": steps}
        else:
            raise ValueError(f"Unsupported planner JSON shape: {type(data)}")

        plan = self._ensure_compose_answer(plan)
        plan = self._ensure_step_fields(plan)
        plan = self._sanitize_requires(plan)
        plan = self._prune_non_tool_steps(plan)

        plan = self._ensure_replan_has_summary_step(
            plan=plan,
            observations_text=observations_text,
            failure_text=failure_text,
            is_replan=is_replan,
        )

        self._validate_tools(plan)
        return plan

    def _ensure_replan_has_summary_step(
        self,
        *,
        plan: Dict[str, Any],
        observations_text: str,
        failure_text: str,
        is_replan: bool,
    ) -> Dict[str, Any]:
        if not is_replan:
            return plan

        steps = plan.get("steps", [])
        has_tool = any(s.get("tool") for s in steps if isinstance(s, dict))
        if has_tool:
            return plan

        payload = (failure_text or "").strip()
        obs = (observations_text or "").strip()
        if obs:
            payload = (payload + "\n\nObservations:\n" + obs).strip()

        if not payload:
            return plan

        plan["steps"] = [
            {
                "id": "summarize_failure",
                "description": "Summarise the failure and observations.",
                "tool": "summarize_text",
                "args": {"text": payload, "bullets": 3},
                "requires": [],
            },
            {
                "id": "compose_answer",
                "description": "Produce final answer from summary.",
                "tool": None,
                "args": None,
                "requires": ["summarize_failure"],
            },
        ]
        return plan

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------

    def _sanitize_requires(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        steps = plan.get("steps", [])
        ids = {s.get("id") for s in steps if isinstance(s, dict) and s.get("id")}

        for s in steps:
            req = s.get("requires", [])
            if req is None:
                req = []
            if not isinstance(req, list):
                req = [req]
            s["requires"] = [r for r in req if isinstance(r, str) and r in ids]

        for i, s in enumerate(steps):
            if s.get("id") == "compose_answer":
                s["requires"] = [x.get("id") for x in steps[:i] if isinstance(x, dict) and x.get("id")]

        return plan

    def _prune_non_tool_steps(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        steps = plan.get("steps", [])
        tools = [s for s in steps if isinstance(s, dict) and s.get("tool") is not None]
        compose = next((s for s in steps if isinstance(s, dict) and s.get("id") == "compose_answer"), None)

        if compose is None:
            compose = {
                "id": "compose_answer",
                "description": "Produce final answer.",
                "tool": None,
                "args": None,
                "requires": [],
            }

        compose["tool"] = None
        compose["args"] = None

        plan["steps"] = tools + [compose]
        return plan

    def _ensure_compose_answer(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        steps = plan.get("steps", [])
        if not isinstance(steps, list):
            steps = []

        if not any(isinstance(s, dict) and s.get("id") == "compose_answer" for s in steps):
            steps.append(
                {
                    "id": "compose_answer",
                    "description": "Produce final answer.",
                    "tool": None,
                    "args": None,
                    "requires": [s.get("id") for s in steps if isinstance(s, dict) and s.get("id")],
                }
            )

        plan["steps"] = steps
        return plan

    def _ensure_step_fields(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        out = []
        for i, s in enumerate(plan.get("steps", [])):
            if not isinstance(s, dict):
                continue
            tool = s.get("tool")
            out.append(
                {
                    "id": s.get("id", f"step_{i+1}"),
                    "description": s.get("description", ""),
                    "tool": tool,
                    "args": None if tool is None else (s.get("args") if isinstance(s.get("args"), dict) else {}),
                    "requires": s.get("requires", []) if isinstance(s.get("requires", []), list) else [],
                }
            )
        plan["steps"] = out
        return plan

    def _validate_tools(self, plan: Dict[str, Any]) -> None:
        allowed = set(TOOLS.keys())
        for s in plan.get("steps", []):
            if not isinstance(s, dict):
                continue
            tool = s.get("tool")
            if tool and tool not in allowed:
                raise ValueError(f"Planner invented unknown tool '{tool}'.")

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
                if isinstance(s, dict)
            ],
        )

    def _debug_plan_summary(self, data: Dict[str, Any]) -> None:
        print("[PLANNER DEBUG] Parsed plan summary:")
        print(f"  Goal: {data.get('goal')}")
        for s in data.get("steps", []):
            print(f"    - {s.get('id')} | tool={s.get('tool')} | requires={s.get('requires')}")
        print()