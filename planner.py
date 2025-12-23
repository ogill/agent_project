# planner.py

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Set

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

    Hardening:
    - Deterministic sanitisation for common "almost JSON" bugs
    - JSON repair prompt (embeds bad output as escaped JSON string)
    - Normalises steps (id/description/tool/args/requires)
    - Coerces UNKNOWN tools to tool=None (never raises for invented tool names)
    - Enforces compose_answer final step and dependencies
    - Enforces "no intermediate tool=None steps" by pruning them
    - Replanning can forbid tools; fallback for replan violations
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
        forbidden_tools: Set[str] | None = None,
    ) -> Plan:
        forbidden_tools = forbidden_tools or set()

        # Deterministic shortcut 1: explicit memory WRITE requests should not call tools.
        if (not is_replan) and self._is_memory_only_request(user_input):
            plan_json = {
                "goal": f"Persist the user-provided fact and acknowledge it: {user_input}",
                "steps": [
                    {
                        "id": "compose_answer",
                        "description": "Acknowledge the user's fact and confirm it has been remembered. Do not call tools.",
                        "tool": None,
                        "args": None,
                        "requires": [],
                    }
                ],
            }
            self._debug_plan_summary(plan_json)
            return self._to_plan(plan_json)

        # Deterministic shortcut 2: likely memory recall -> no tools
        if (not is_replan) and self._is_likely_memory_recall_request(user_input):
            plan_json = {
                "goal": f"Answer the user's question using available context/memory: {user_input}",
                "steps": [
                    {
                        "id": "compose_answer",
                        "description": "Answer using the conversation context/memory. Do not call tools.",
                        "tool": None,
                        "args": None,
                        "requires": [],
                    }
                ],
            }
            self._debug_plan_summary(plan_json)
            return self._to_plan(plan_json)

        if not tools_spec:
            tools_spec = self._build_tools_spec()

        prompt = self._build_planner_prompt(
            user_input=user_input,
            tools_spec=tools_spec,
            observations_text=observations_text,
            failure_text=failure_text,
            is_replan=is_replan,
            replan_suffix=replan_suffix or DEFAULT_REPLAN_SUFFIX,
            forbidden_tools=forbidden_tools,
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

        # Enforce forbidden tools (esp. during replans). If violated during replan, fallback safely.
        try:
            self._enforce_forbidden_tools(plan_json, forbidden_tools=forbidden_tools)
        except Exception as e:
            if is_replan:
                print(f"[PLANNER DEBUG] Model replan violated forbidden tools, falling back. Reason: {e}")
                plan_json = self._deterministic_replan_plan(
                    user_input=user_input,
                    observations_text=observations_text,
                    failure_text=failure_text,
                )
            else:
                raise

        self._debug_plan_summary(plan_json)
        return self._to_plan(plan_json)

    # --------------------------- heuristics ---------------------------

    def _is_memory_only_request(self, user_input: str) -> bool:
        s = (user_input or "").lower().strip()
        return (
            s.startswith("remember:") or
            s.startswith("remember ") or
            s.startswith("note:") or
            s.startswith("note ") or
            "add this to memory" in s or
            "store this" in s or
            "save this" in s or
            "keep this" in s
        )

    def _is_likely_memory_recall_request(self, user_input: str) -> bool:
        s = (user_input or "").lower().strip()

        tool_intent_markers = [
            "time is it",
            "what time",
            "weather",
            "forecast",
            "fetch ",
            "fetch the contents",
            "open ",
            "http://",
            "https://",
            "url",
        ]
        if any(m in s for m in tool_intent_markers):
            return False

        recall_markers = [
            "what is my ",
            "what's my ",
            "what do i like",
            "what did i say",
            "what did i tell you",
            "do you remember",
            "what is my favourite",
            "what's my favourite",
            "which bike do i like",
            "what bike do i like",
        ]
        return any(m in s for m in recall_markers)

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
        forbidden_tools: Set[str] | None = None,
    ) -> str:
        forbidden_tools = forbidden_tools or set()

        base = (
            get_planner_prompt(PROMPT_MODE)
            + "\n\nIMPORTANT:\n"
            + "- Do NOT create intermediate non-tool steps (tool=null).\n"
            + "- The ONLY non-tool step allowed is the FINAL 'compose_answer' step.\n"
            + "- If you need to 'use observations', do it inside compose_answer, not as separate steps.\n"
            + "- Do NOT use JSON references like `$ref` or any symbolic placeholders in tool args; args MUST be concrete JSON values.\n\n"
            + tools_spec
            + "\n\nUser request:\n"
            + user_input
        )

        if is_replan:
            base += (
                "\n\n"
                + replan_suffix.strip()
                + "\n\nIMPORTANT REPLAN CONSTRAINT REMINDER:\n"
                + "- Do NOT add any non-tool steps (tool=null). The ONLY allowed non-tool step is the FINAL 'compose_answer'.\n"
                + "- Do NOT invent step IDs in requires; requires must reference real step ids in your plan.\n"
            )

            if forbidden_tools:
                base += (
                    "\n\nFORBIDDEN TOOLS (do not call these in your new plan):\n"
                    + ", ".join(sorted(forbidden_tools))
                    + "\n"
                )

            base += (
                "\n\nObservations so far:\n"
                + (observations_text.strip() or "(none)")
                + "\n\nFailure / blocker encountered:\n"
                + (failure_text.strip() or "(none)")
            )

        base += "\n\nReturn ONLY the Plan JSON now."
        return base

    # --------------------------- deterministic replanning (fallback) ---------------------------

    def _deterministic_replan_plan(
        self,
        *,
        user_input: str,
        observations_text: str,
        failure_text: str,
    ) -> Dict[str, Any]:
        _ = observations_text
        _ = failure_text

        return {
            "goal": f"Answer the user using available failure/observations without calling tools: {user_input}",
            "steps": [
                {
                    "id": "compose_answer",
                    "description": "Explain what happened using the failure/observations already provided. Do not call tools.",
                    "tool": None,
                    "args": None,
                    "requires": [],
                }
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
        text = (raw or "")
        text = self._sanitize_common_model_json_bugs(text)

        # allow MCP tools at runtime
        from tools import TOOLS as _TOOLS
        allowed_tool_names = sorted(_TOOLS.keys())

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
                    allowed_tools=allowed_tool_names,
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

    def _contains_ref(self, obj: Any) -> bool:
        if isinstance(obj, dict):
            if "$ref" in obj:
                return True
            return any(self._contains_ref(v) for v in obj.values())
        if isinstance(obj, list):
            return any(self._contains_ref(v) for v in obj)
        if isinstance(obj, str):
            s = obj.lower()
            if "$ref" in s:
                return True
            if "steps/" in s and "result" in s:
                return True
        return False

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

        # strip code fences if present
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

        # parse
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

        # salvage JSON between braces
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

    def _build_json_repair_prompt(self, *, bad_output: str, original_prompt: str, allowed_tools: list[str]) -> str:
        _ = original_prompt
        bad_as_json_string = json.dumps(bad_output)
        allowed_tools_str = ", ".join(allowed_tools)

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
            f"- tool names must be one of: {allowed_tools_str}\n"
            "- Do NOT use $ref/symbolic placeholders in tool args; args MUST be concrete JSON values.\n\n"
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
        _ = observations_text
        _ = failure_text
        _ = is_replan

        if not isinstance(data, dict) or "steps" not in data:
            raise ValueError(f"Unsupported planner JSON shape: {type(data)}")

        plan: Dict[str, Any] = data

        plan = self._ensure_step_fields(plan)
        plan = self._coerce_unknown_tools_to_none(plan)   # <-- KEY FIX (no raise)
        plan = self._ensure_compose_answer(plan)
        plan = self._prune_intermediate_non_tool_steps(plan)  # <-- KEY FIX (prune)
        plan = self._sanitize_requires(plan)
        self._validate_tools(plan)  # validates only remaining tool steps

        return plan

    def _ensure_step_fields(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        out: List[Dict[str, Any]] = []
        for i, s in enumerate(plan.get("steps", [])):
            if not isinstance(s, dict):
                continue

            tool = s.get("tool", None)
            args = s.get("args", None)

            # Normalise requires
            req = s.get("requires", [])
            if not isinstance(req, list):
                req = [req]

            # If tool is None => args must be None
            if tool is None:
                args = None
            else:
                # tool step => args must be dict
                if not isinstance(args, dict):
                    args = {}

            out.append(
                {
                    "id": s.get("id", f"step_{i+1}"),
                    "description": s.get("description", ""),
                    "tool": tool,
                    "args": args,
                    "requires": req,
                }
            )

        plan["steps"] = out
        return plan

    def _coerce_unknown_tools_to_none(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        If the LLM invents a tool, do NOT raise.
        Coerce that step to tool=None and args=None, but keep the step (marked),
        so tests/diagnostics can see that the model attempted an unknown tool.
        """
        allowed = set(TOOLS.keys())
        for s in plan.get("steps", []):
            tool = s.get("tool", None)
            if tool is None:
                s["args"] = None
                continue
            if tool not in allowed:
                s["tool"] = None
                s["args"] = None
                s["_coerced_unknown_tool"] = True  # <-- key
        return plan

    def _ensure_compose_answer(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        steps = plan.get("steps", [])
        if not any(isinstance(s, dict) and s.get("id") == "compose_answer" for s in steps):
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

    def _prune_intermediate_non_tool_steps(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enforce contract:
        - compose_answer is final and tool=None
        - prune intermediate tool=None steps EXCEPT those that were coerced from unknown tools
        (kept for visibility; Agent still won't execute them because tool=None).
        """
        steps = plan.get("steps", [])

        tool_steps: List[Dict[str, Any]] = []
        coerced_unknown_steps: List[Dict[str, Any]] = []

        for s in steps:
            if s.get("id") == "compose_answer":
                continue
            if s.get("tool") is not None:
                tool_steps.append(s)
            elif s.get("_coerced_unknown_tool") is True:
                # keep it (non-executable), so the regression test can find it
                coerced_unknown_steps.append(s)
            # else: prune plain non-tool intermediate steps

        compose = next((s for s in steps if s.get("id") == "compose_answer"), None)
        if compose is None:
            compose = {
                "id": "compose_answer",
                "description": "Reads all previous step results and produces one coherent answer.",
                "tool": None,
                "args": None,
                "requires": [],
            }
        else:
            compose["tool"] = None
            compose["args"] = None

        plan["steps"] = tool_steps + coerced_unknown_steps + [compose]
        return plan

    def _sanitize_requires(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        steps = plan.get("steps", [])
        ids = {s.get("id") for s in steps if isinstance(s, dict)}

        for s in steps:
            req = s.get("requires", [])
            if not isinstance(req, list):
                req = []
            s["requires"] = [r for r in req if isinstance(r, str) and r in ids and r != s.get("id")]

        compose = next((s for s in steps if s.get("id") == "compose_answer"), None)
        if compose is not None:
            # compose depends on ALL prior tool steps (and never itself)
            compose["requires"] = [
                s.get("id") for s in steps
                if s.get("id") != "compose_answer" and s.get("tool") is not None
            ]

        return plan

    def _validate_tools(self, plan: Dict[str, Any]) -> None:
        allowed = set(TOOLS.keys())

        for s in plan.get("steps", []):
            tool = s.get("tool")

            if tool is None:
                # compose_answer only
                continue

            if tool not in allowed:
                # Should not happen due to coercion, but keep defensive
                raise ValueError(f"Planner invented unknown tool '{tool}'.")

            args = s.get("args")
            if not isinstance(args, dict):
                raise ValueError(f"Tool args must be an object for tool '{tool}', got {type(args).__name__}.")

            if self._contains_ref(args):
                raise ValueError("Planner used $ref/symbolic placeholder in tool args; args must be concrete JSON values.")

    def _enforce_forbidden_tools(self, plan: Dict[str, Any], *, forbidden_tools: Set[str]) -> None:
        if not forbidden_tools:
            return
        violating: Set[str] = set()
        for s in plan.get("steps", []):
            tool = s.get("tool")
            if tool and tool in forbidden_tools:
                violating.add(tool)
        if violating:
            raise ValueError(f"Plan attempted to call forbidden tools: {', '.join(sorted(violating))}")

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