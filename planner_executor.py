# planner_executor.py

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from plan_types import Plan, PlanStep
from tools import TOOLS


@dataclass
class ExecutionResult:
    """
    Result of executing a Plan's tool steps (compose_answer is not executed here).
    """
    plan: Plan
    observations: Dict[str, Any]          # step_id -> tool output (or structured failure payload)
    tool_calls: List[str]                 # step_ids executed, in order
    replans_used: int
    last_failure_text: str | None = None


class PlannerExecutor:
    """
    Executes tool steps in a Plan using TOOLS.

    Notes:
    - Does NOT call the LLM to "compose_answer".
    - Supports replanning if you pass a Planner into execute_with_replanning().
    - Supports numeric step ids like "0" / "1" by normalizing to "step_1" / "step_2"
      (and rewriting `requires`) to keep downstream debug + orchestration cleaner.
    """

    def __init__(
        self,
        *,
        max_replans: int = 2,
        max_steps: int = 25,
        observation_max_chars: int = 8000,
        trace: bool = False,
    ) -> None:
        self.max_replans = max_replans
        self.max_steps = max_steps
        self.observation_max_chars = observation_max_chars
        self.trace = trace

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def execute_plan(self, plan: Plan, *, observations: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """
        Execute tool steps from the given plan (no replanning).
        Raises on tool exceptions unless the tool returns a structured soft-failure payload.
        """
        # Normalize once per execution so step ids and requires are stable.
        plan, id_map = self._normalize_plan_ids(plan)

        observations = observations or {}
        observations = self._remap_observations(observations, id_map)

        tool_calls: List[str] = []

        tool_steps = [s for s in plan.steps if s.tool is not None and s.id != "compose_answer"]

        if len(tool_steps) > self.max_steps:
            raise RuntimeError(f"Plan has too many tool steps ({len(tool_steps)} > {self.max_steps}).")

        ordered = self._order_steps(tool_steps)

        for step in ordered:
            # Skip already-executed steps (lets us carry observations across replans)
            if step.id in observations:
                continue

            if self.trace:
                print(f"[EXEC] run: {step.id} tool={step.tool} requires={step.requires} args={step.args}")

            out = self._run_tool_step(step)
            observations[step.id] = out
            tool_calls.append(step.id)

            # Soft failure payload convention
            if self._is_soft_failure(out):
                reason = self._soft_failure_reason(out)
                raise SoftToolFailure(step_id=step.id, tool_name=step.tool or "", reason=reason, payload=out)

        return ExecutionResult(
            plan=plan,
            observations=observations,
            tool_calls=tool_calls,
            replans_used=0,
            last_failure_text=None,
        )

    def execute_with_replanning(
        self,
        *,
        user_input: str,
        initial_plan: Plan,
        planner: Any,  # your Planner instance
    ) -> ExecutionResult:
        """
        Execute with replanning:
        - Executes tool steps.
        - On failure, asks planner.generate_plan(... is_replan=True ...) for a new plan.
        - Carries forward existing observations.
        """
        # We'll keep observations keyed by the *normalized* ids from the latest plan,
        # and remap when new plans arrive.
        observations: Dict[str, Any] = {}
        tool_calls: List[str] = []
        replans_used = 0

        current_plan = initial_plan
        forbidden_tools: Set[str] = set()
        last_failure_text: str | None = None

        while True:
            try:
                # execute_plan() handles normalization + observation remap for us
                res = self.execute_plan(current_plan, observations=observations)

                # Persist normalized plan + observations for next loop
                current_plan = res.plan
                observations = res.observations
                tool_calls.extend(res.tool_calls)

                return ExecutionResult(
                    plan=current_plan,
                    observations=observations,
                    tool_calls=tool_calls,
                    replans_used=replans_used,
                    last_failure_text=last_failure_text,
                )

            except SoftToolFailure as e:
                replans_used += 1
                last_failure_text = f"Soft failure in step '{e.step_id}' tool '{e.tool_name}': {e.reason}"
                if self.trace:
                    print(f"[EXEC] {last_failure_text}")

                retryable = bool(e.payload.get("retryable", False)) if isinstance(e.payload, dict) else False
                if not retryable and e.tool_name:
                    forbidden_tools.add(e.tool_name)

                if replans_used > self.max_replans:
                    raise RuntimeError(
                        f"Exceeded max replans ({self.max_replans}). Last failure: {last_failure_text}"
                    ) from e

                current_plan = self._replan(
                    planner=planner,
                    user_input=user_input,
                    observations=observations,
                    failure_text=last_failure_text,
                    forbidden_tools=forbidden_tools,
                )
                continue

            except Exception as e:
                replans_used += 1
                last_failure_text = f"Hard failure during tool execution: {e!r}"
                if self.trace:
                    print(f"[EXEC] {last_failure_text}")

                if replans_used > self.max_replans:
                    raise RuntimeError(
                        f"Exceeded max replans ({self.max_replans}). Last failure: {last_failure_text}"
                    ) from e

                current_plan = self._replan(
                    planner=planner,
                    user_input=user_input,
                    observations=observations,
                    failure_text=last_failure_text,
                    forbidden_tools=forbidden_tools,
                )
                continue

    # ---------------------------------------------------------------------
    # ID normalization (fixes numeric ids like "0")
    # ---------------------------------------------------------------------

    def _normalize_plan_ids(self, plan: Plan) -> tuple[Plan, Dict[str, str]]:
        """
        If the model returns numeric-ish step ids ("0", "1", "2") we normalize them to
        "step_1", "step_2", ... and rewrite requires accordingly.

        Returns:
          (new_plan, id_map) where id_map maps old_id -> new_id.
        """
        steps = list(plan.steps or [])

        # Build stable mapping for "numeric" ids only (excluding compose_answer)
        id_map: Dict[str, str] = {}

        # Detect numeric ids
        numeric_ids: List[str] = []
        for s in steps:
            sid = str(s.id)
            if sid == "compose_answer":
                continue
            if sid.isdigit():
                numeric_ids.append(sid)

        if not numeric_ids:
            return plan, {}

        # Create mapping in numeric order to keep it deterministic
        for old in sorted(numeric_ids, key=lambda x: int(x)):
            id_map[old] = f"step_{int(old) + 1}"  # "0" -> "step_1"

        # Rewrite steps
        new_steps: List[PlanStep] = []
        for s in steps:
            old_id = str(s.id)
            new_id = id_map.get(old_id, old_id)

            reqs = s.requires or []
            new_reqs = [id_map.get(str(r), str(r)) for r in reqs]

            new_steps.append(
                PlanStep(
                    id=new_id,
                    description=s.description,
                    tool=s.tool,
                    args=s.args,
                    requires=new_reqs,
                )
            )

        # Ensure compose_answer.requires references the rewritten ids (if present)
        # We rely on planner.py sanitize usually, but we keep it consistent here too.
        for s in new_steps:
            if s.id == "compose_answer":
                s.requires = [id_map.get(str(r), str(r)) for r in (s.requires or [])]

        if self.trace:
            print(f"[EXEC] normalized ids: {id_map}")

        return Plan(goal=plan.goal, steps=new_steps), id_map

    def _remap_observations(self, observations: Dict[str, Any], id_map: Dict[str, str]) -> Dict[str, Any]:
        """
        If we normalized step ids, remap any existing observations keys to match.
        This allows replans/continuations to keep working even if earlier execution
        stored observations under numeric ids.
        """
        if not id_map:
            return observations

        remapped: Dict[str, Any] = {}
        for k, v in observations.items():
            nk = id_map.get(str(k), str(k))
            # If both exist, prefer the normalized key
            remapped[nk] = v
        return remapped

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _run_tool_step(self, step: PlanStep) -> Any:
        tool_name = step.tool
        if not tool_name:
            return None

        if tool_name not in TOOLS:
            raise RuntimeError(f"Unknown tool '{tool_name}' in step '{step.id}'.")

        fn = TOOLS[tool_name].get("fn")
        if not callable(fn):
            raise RuntimeError(f"Tool '{tool_name}' has no callable 'fn'.")

        args = step.args or {}
        if not isinstance(args, dict):
            raise RuntimeError(f"Tool args for '{tool_name}' must be a dict, got {type(args).__name__}.")

        return fn(**args)

    def _order_steps(self, steps: List[PlanStep]) -> List[PlanStep]:
        """
        Topological sort using `requires` where requires refer to step ids.
        """
        by_id: Dict[str, PlanStep] = {s.id: s for s in steps}
        deps: Dict[str, Set[str]] = {s.id: set([r for r in (s.requires or []) if r in by_id]) for s in steps}

        ordered: List[PlanStep] = []
        ready = sorted([sid for sid, req in deps.items() if not req])

        while ready:
            sid = ready.pop(0)
            ordered.append(by_id[sid])

            # remove sid from remaining deps
            for other_id in list(deps.keys()):
                if sid in deps[other_id]:
                    deps[other_id].remove(sid)
                    if not deps[other_id]:
                        ready.append(other_id)
                        ready.sort()

            deps.pop(sid, None)

        if deps:
            cycle = ", ".join(sorted(deps.keys()))
            raise RuntimeError(f"Plan contains a dependency cycle or unresolved deps among: {cycle}")

        return ordered

    def _is_soft_failure(self, out: Any) -> bool:
        """
        Soft failure convention used by your tools.py:
          {"ok": False, "status": "failed", "reason": "...", "retryable": bool}
        """
        if not isinstance(out, dict):
            return False
        if out.get("ok") is False:
            return True
        if str(out.get("status", "")).lower().strip() in {"failed", "failure", "error"}:
            return True
        return False

    def _soft_failure_reason(self, out: Any) -> str:
        if isinstance(out, dict):
            return str(out.get("reason") or out.get("error") or out.get("message") or "soft failure")
        return "soft failure"

    def _replan(
        self,
        *,
        planner: Any,
        user_input: str,
        observations: Dict[str, Any],
        failure_text: str,
        forbidden_tools: Set[str],
    ) -> Plan:
        observations_text = self._format_observations(observations)

        # Your Planner supports these kwargs.
        new_plan = planner.generate_plan(
            user_input,
            observations_text=observations_text,
            failure_text=failure_text,
            is_replan=True,
            forbidden_tools=set(forbidden_tools),
        )

        if self.trace:
            print("[EXEC] replanned. new goal:", getattr(new_plan, "goal", ""))
        return new_plan

    def _format_observations(self, observations: Dict[str, Any]) -> str:
        """
        Produce a compact text view for replanning prompts.
        """
        parts: List[str] = []
        for k in sorted(observations.keys()):
            v = observations[k]
            try:
                s = json.dumps(v, ensure_ascii=False)
            except Exception:
                s = repr(v)
            if len(s) > self.observation_max_chars:
                s = s[: self.observation_max_chars] + "...(truncated)"
            parts.append(f"{k}: {s}")
        return "\n".join(parts) if parts else "(none)"


class SoftToolFailure(RuntimeError):
    def __init__(self, *, step_id: str, tool_name: str, reason: str, payload: Any) -> None:
        super().__init__(reason)
        self.step_id = step_id
        self.tool_name = tool_name
        self.reason = reason
        self.payload = payload