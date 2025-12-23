from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from orchestrator.context import Artifact, RunContext
from orchestrator.models import WorkItem
from orchestrator.roles import RoleRegistry
from orchestrator.routing import build_work_items_for_template


@dataclass
class OrchestratorPolicy:
    max_work_items: int = 10
    max_concurrency: int = 4
    per_item_timeout_s: float = 15.0
    enable_parallel: bool = True


class Orchestrator:
    """
    Stage 8 Orchestrator:
    - 8.2: single WorkItem -> single agent call (parity path)
    - 8.2.1: multiple WorkItems -> sequential agent calls + deterministic merge
    - 8.2.2: role indirection via RoleRegistry (no behavior change)
    - 8.2.3: deterministic routing templates -> multi-role WorkItems
    - 8.3.1: RunContext + structured artifacts
    - 8.3.2: explicit dependencies (depends_on) + selective artifact injection
    - 8.4: parallel execution in dependency-safe "waves" (bounded concurrency + timeouts)
    """

    def __init__(
        self,
        role_registry: RoleRegistry,
        policy: Optional[OrchestratorPolicy] = None,
    ) -> None:
        self.role_registry = role_registry
        self.policy = policy or OrchestratorPolicy()

    def run(self, goal: str, context: Optional[Dict[str, Any]] = None) -> str:
        context = context or {}

        work_item = WorkItem(
            id="work-001",
            assigned_agent="generalist",
            goal=goal,
            inputs=context,
            expected_output={},
        )

        return self.run_work_items([work_item])

    def run_template(self, template: str, goal: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Stage 8.2.3: deterministic multi-role routing via templates.
        """
        context = context or {}
        work_items = build_work_items_for_template(template=template, goal=goal, context=context)
        return self.run_work_items(work_items)

    def run_work_items(self, work_items: List[WorkItem]) -> str:
        if len(work_items) > self.policy.max_work_items:
            raise ValueError(
                f"Too many WorkItems: {len(work_items)} (max {self.policy.max_work_items})"
            )

        run_context = RunContext()
        results: Dict[str, str] = {}

        # Execute WorkItems in dependency-safe "waves"
        remaining = list(work_items)

        while remaining:
            ready = _select_ready_work_items(remaining, run_context)

            if not ready:
                missing_detail: Dict[str, List[str]] = {}
                for wi in remaining:
                    deps = getattr(wi, "depends_on", []) or []
                    missing = [d for d in deps if d not in run_context.artifacts]
                    if missing:
                        missing_detail[wi.id] = missing

                # Preserve 8.3.2 semantics: missing deps are a KeyError
                if missing_detail:
                    raise KeyError(f"Missing required artifacts: {missing_detail}")

                # Otherwise, it's a cycle (deps exist but cannot be satisfied due to ordering)
                raise RuntimeError("No runnable WorkItems found. Possible dependency cycle.")

            # If only one item is ready, or parallel disabled: run sequentially
            if (not self.policy.enable_parallel) or (len(ready) == 1):
                wi = ready[0]
                out = _run_one_sync(wi, self.role_registry, run_context)
                _record_output(wi, out, run_context, results)
            else:
                # Run a wave in parallel
                outs = _run_wave_parallel(ready, self.role_registry, run_context, self.policy)

                # Record outputs in deterministic order (the order they appear in `ready`)
                for wi in ready:
                    out = outs[wi.id]
                    _record_output(wi, out, run_context, results)

            # Remove completed items
            done_ids = {wi.id for wi in ready}
            remaining = [wi for wi in remaining if wi.id not in done_ids]

        return _merge_results_deterministically(work_items, results)


def _compose_user_input(goal: str, context: Dict[str, Any], selected_artifacts: Dict[str, Any]) -> str:
    parts: List[str] = [goal]

    if context:
        parts.append(f"Initial context (JSON): {context}")

    if selected_artifacts:
        parts.append("Shared context artifacts:\n" + f"{selected_artifacts}")

    return "\n\n".join(parts)


def _run_stage7_agent(agent: Any, user_input: str) -> str:
    if hasattr(agent, "run"):
        return agent.run(user_input)
    raise AttributeError("Agent does not expose .run(user_input: str)")


def _select_ready_work_items(remaining: List[WorkItem], run_context: RunContext) -> List[WorkItem]:
    """
    A WorkItem is ready if all its depends_on keys exist in run_context.artifacts.
    """
    ready: List[WorkItem] = []
    for wi in remaining:
        deps = getattr(wi, "depends_on", []) or []
        if all(dep in run_context.artifacts for dep in deps):
            ready.append(wi)
    return ready


def _run_one_sync(wi: WorkItem, role_registry: RoleRegistry, run_context: RunContext) -> str:
    agent = role_registry.get_agent(wi.assigned_agent)

    selected: Dict[str, Any] = {}
    if getattr(wi, "depends_on", None):
        # Raises KeyError if missing deps -> good, fail fast
        selected = run_context.snapshot_selected(wi.depends_on)

    user_input = _compose_user_input(wi.goal, wi.inputs, selected)
    return _run_stage7_agent(agent, user_input)


async def _run_one_async(
    wi: WorkItem,
    role_registry: RoleRegistry,
    run_context: RunContext,
    policy: OrchestratorPolicy,
    sem: asyncio.Semaphore,
) -> tuple[str, str]:
    async with sem:
        agent = role_registry.get_agent(wi.assigned_agent)

        selected: Dict[str, Any] = {}
        if getattr(wi, "depends_on", None):
            selected = run_context.snapshot_selected(wi.depends_on)

        user_input = _compose_user_input(wi.goal, wi.inputs, selected)

        async def _call() -> str:
            return await asyncio.to_thread(_run_stage7_agent, agent, user_input)

        out = await asyncio.wait_for(_call(), timeout=policy.per_item_timeout_s)
        return wi.id, out


def _run_wave_parallel(
    wave: List[WorkItem],
    role_registry: RoleRegistry,
    run_context: RunContext,
    policy: OrchestratorPolicy,
) -> Dict[str, str]:
    """
    Runs a wave concurrently with bounded concurrency + per-item timeout.
    Returns {work_item_id: output}.
    """
    async def _runner() -> Dict[str, str]:
        sem = asyncio.Semaphore(policy.max_concurrency)
        tasks = [
            asyncio.create_task(_run_one_async(wi, role_registry, run_context, policy, sem))
            for wi in wave
        ]
        pairs = await asyncio.gather(*tasks)
        return {k: v for (k, v) in pairs}

    return asyncio.run(_runner())


def _record_output(wi: WorkItem, output: str, run_context: RunContext, results: Dict[str, str]) -> None:
    artifact = Artifact(
        key=f"{wi.id}.output",
        value=output,
        producer=wi.id,
        metadata={"role": wi.assigned_agent},
    )
    run_context.add_artifact(artifact)
    results[wi.id] = output


def _merge_results_deterministically(work_items: List[WorkItem], results: Dict[str, str]) -> str:
    # Stage 7 parity guarantee for single WorkItem:
    if len(work_items) == 1:
        wi = work_items[0]
        return results.get(wi.id, "")

    lines: List[str] = []
    for wi in work_items:
        lines.append(f"[{wi.id}] {wi.assigned_agent}: {wi.goal}")
        lines.append(results.get(wi.id, "").rstrip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"