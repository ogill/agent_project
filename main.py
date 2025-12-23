# main.py
from __future__ import annotations

from typing import Any, Dict, List

from agent import Agent
from config import MODEL_NAME
from orchestrator.role_agent import RoleAgent

# --- Stage 8 Orchestrator wiring ---
from orchestrator.role_names import GENERALIST, RESEARCHER, REVIEWER
from orchestrator.roles import RoleRegistry
from orchestrator.orchestrator import Orchestrator, OrchestratorPolicy
from orchestrator.routing import build_work_items_for_template

HELP = """Commands:
  :help                         Show this help
  :multi                        Enter multiline mode (finish with :end)

  :trace off|summary|debug       Control planner verbosity (Stage 7 planner)
  :plan <text>                   Generate and print planner plan only (no execution)
  :explain <text>                Generate and explain planner plan (no execution)
  :plan                          Set next input intent to 'plan' (works with :multi)
  :explain                       Set next input intent to 'explain' (works with :multi)

  :orch on|off                   Toggle orchestrator execution mode (Stage 8)
  :template <name>               Set orchestrator template (single|design_review|draft_review_revise)
  :orchplan                      Next input prints orchestrator plan only (works with :multi)

  exit | quit                    Exit
"""


def _describe_orch_template(template: str, goal: str, context: Dict[str, Any] | None = None) -> str:
    """
    Human-friendly view of the deterministic WorkItem graph for the selected template.
    (No LLM calls. No execution.)
    """
    context = context or {}
    work_items = build_work_items_for_template(template=template, goal=goal, context=context)

    lines: List[str] = []
    lines.append("Orchestrator plan:")
    lines.append(f"- Template: {template}")
    lines.append(f"- WorkItems: {len(work_items)}")
    lines.append("- What will happen:")
    for i, wi in enumerate(work_items, 1):
        deps = getattr(wi, "depends_on", []) or []
        lines.append(f"  {i}) {wi.id} role={wi.assigned_agent} deps={deps}")
        lines.append(f"     goal: {wi.goal}")
    lines.append("- Then: merge WorkItem outputs deterministically.")
    return "\n".join(lines)


def main() -> None:
    # Stage 7 agent (single-agent path)
    stage7_agent = Agent()

    # --- Multi-agent wiring (Stage 8): distinct agents per role via RoleAgent ---
    roles = {
        GENERALIST: RoleAgent(role=GENERALIST, base_agent=Agent()),
        RESEARCHER: RoleAgent(role=RESEARCHER, base_agent=Agent()),
        REVIEWER: RoleAgent(role=REVIEWER, base_agent=Agent()),
    }

    # RoleRegistry signature varies depending on your implementation; support both.
    try:
        registry = RoleRegistry(roles=roles)  # preferred: keyword
    except TypeError:
        registry = RoleRegistry(roles)  # fallback: positional

    orch = Orchestrator(
        role_registry=registry,
        policy=OrchestratorPolicy(
            max_work_items=10,
            max_concurrency=4,
            per_item_timeout_s=15.0,
            enable_parallel=True,
        ),
    )

    print(f"ReAct Agent via Ollama ({MODEL_NAME})")
    print("Type 'exit' or 'quit' to stop.")
    print("Use ':multi' to enter multiline mode, ':end' to submit.")
    print("Type ':help' for commands.\n")

    multiline_mode = False
    multiline_buffer: list[str] = []

    next_intent: str | None = None  # "plan" | "explain" | "orchplan" | None
    orch_enabled = False
    orch_template = "single"

    def _handle_intent(text: str) -> None:
        nonlocal next_intent, orch_template
        text = (text or "").strip()
        if not text:
            print("(empty input ignored)\n")
            next_intent = None
            return

        try:
            if next_intent in {"plan", "explain"}:
                plan = stage7_agent.planner.generate_plan(text)
                if next_intent == "plan":
                    stage7_agent.planner.print_plan_summary(plan)
                    print()
                else:
                    print(stage7_agent.planner.explain_plan(plan))
                    print()
            elif next_intent == "orchplan":
                print(_describe_orch_template(template=orch_template, goal=text, context={}))
                print()
            else:
                print("(unknown intent)\n")
        except Exception as e:
            print(f"\n[ERROR] {e}\n")
        finally:
            next_intent = None

    def _run_query(text: str) -> None:
        nonlocal orch_enabled, orch_template
        text = (text or "").strip()
        if not text:
            print("(empty input ignored)\n")
            return

        try:
            if orch_enabled:
                out = orch.run_template(template=orch_template, goal=text, context={})
                print(f"\n{out}\n")
            else:
                out = stage7_agent.run(text)
                print(f"\n{out}\n")
        except Exception as e:
            print(f"\n[ERROR] {e}\n")

    while True:
        try:
            line = input("Human (multi) > " if multiline_mode else "Human > ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        stripped = line.strip()

        if stripped.lower() in {"exit", "quit", ":exit", ":quit"}:
            print("Exiting.")
            break

        # MULTILINE MODE
        if multiline_mode:
            # Allow commands even in multiline mode
            if stripped in {":help", "help"}:
                print(HELP)
                continue

            if stripped.startswith(":trace"):
                parts = stripped.split(maxsplit=1)
                level = (parts[1].strip().lower() if len(parts) == 2 else "")
                if level not in {"off", "summary", "debug"}:
                    print("Usage: :trace off|summary|debug\n")
                    continue
                stage7_agent.planner.set_trace_level(level)
                print(f"(trace set to '{level}')\n")
                continue

            if stripped.startswith(":orch "):
                parts = stripped.split(maxsplit=1)
                val = (parts[1].strip().lower() if len(parts) == 2 else "")
                if val not in {"on", "off"}:
                    print("Usage: :orch on|off\n")
                    continue
                orch_enabled = (val == "on")
                print(f"(orchestrator set to '{val}', template='{orch_template}')\n")
                continue

            if stripped.startswith(":template "):
                _, name = stripped.split(" ", 1)
                name = name.strip()
                if name not in {"single", "design_review", "draft_review_revise"}:
                    print("Usage: :template single|design_review|draft_review_revise\n")
                    continue
                orch_template = name
                print(f"(template set to '{orch_template}')\n")
                continue

            if stripped == ":plan":
                next_intent = "plan"
                print("(next input intent: plan)\n")
                continue

            if stripped == ":explain":
                next_intent = "explain"
                print("(next input intent: explain)\n")
                continue

            if stripped == ":orchplan":
                next_intent = "orchplan"
                print(f"(next input intent: orchplan using template '{orch_template}')\n")
                continue

            if stripped == ":end":
                text = "\n".join(multiline_buffer).strip()
                multiline_mode = False
                multiline_buffer = []

                if next_intent in {"plan", "explain", "orchplan"}:
                    _handle_intent(text)
                    continue

                _run_query(text)
                continue

            multiline_buffer.append(line)
            continue

        # SINGLE-LINE MODE
        if not stripped:
            continue

        if stripped in {":help", "help"}:
            print(HELP)
            continue

        if stripped.startswith(":trace"):
            parts = stripped.split(maxsplit=1)
            level = (parts[1].strip().lower() if len(parts) == 2 else "")
            if level not in {"off", "summary", "debug"}:
                print("Usage: :trace off|summary|debug\n")
                continue
            stage7_agent.planner.set_trace_level(level)
            print(f"(trace set to '{level}')\n")
            continue

        if stripped == ":plan":
            next_intent = "plan"
            print("(next input intent: plan)\n")
            continue

        if stripped == ":explain":
            next_intent = "explain"
            print("(next input intent: explain)\n")
            continue

        if stripped == ":orchplan":
            next_intent = "orchplan"
            print(f"(next input intent: orchplan using template '{orch_template}')\n")
            continue

        if stripped.startswith(":plan "):
            next_intent = "plan"
            _handle_intent(stripped.split(" ", 1)[1])
            continue

        if stripped.startswith(":explain "):
            next_intent = "explain"
            _handle_intent(stripped.split(" ", 1)[1])
            continue

        if stripped.startswith(":orch "):
            parts = stripped.split(maxsplit=1)
            val = (parts[1].strip().lower() if len(parts) == 2 else "")
            if val not in {"on", "off"}:
                print("Usage: :orch on|off\n")
                continue
            orch_enabled = (val == "on")
            print(f"(orchestrator set to '{val}', template='{orch_template}')\n")
            continue

        if stripped.startswith(":template "):
            _, name = stripped.split(" ", 1)
            name = name.strip()
            if name not in {"single", "design_review", "draft_review_revise"}:
                print("Usage: :template single|design_review|draft_review_revise\n")
                continue
            orch_template = name
            print(f"(template set to '{orch_template}')\n")
            continue

        if stripped == ":multi":
            multiline_mode = True
            multiline_buffer = []
            print("(multiline mode — type ':end' on its own line to submit)")
            continue

        if next_intent in {"plan", "explain", "orchplan"}:
            _handle_intent(stripped)
            continue

        # Normal execution
        _run_query(stripped)


if __name__ == "__main__":
    main()