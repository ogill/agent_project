# main.py

from __future__ import annotations

from agent import Agent
from config import MODEL_NAME

from orchestrator.orchestrator import Orchestrator
from orchestrator.roles import RoleRegistry, RoleSpec
from orchestrator.role_names import GENERALIST, RESEARCHER, REVIEWER


def _build_stage8_orchestrator() -> Orchestrator:
    """
    Stage 8 real-run harness:
    - Uses real Stage 7 Agent instances as role workers.
    - Orchestrator remains deterministic; only agents/Planner call the LLM.
    """
    roles = RoleRegistry(
        {
            GENERALIST: RoleSpec(name=GENERALIST, agent=Agent()),
            RESEARCHER: RoleSpec(name=RESEARCHER, agent=Agent()),
            REVIEWER: RoleSpec(name=REVIEWER, agent=Agent()),
        }
    )
    return Orchestrator(role_registry=roles)


def main() -> None:
    agent = Agent()
    orch = _build_stage8_orchestrator()

    print(f"ReAct Agent via Ollama ({MODEL_NAME})")
    print("Type 'exit' or 'quit' to stop.")
    print("Stage 8 templates: use '@t <template> <goal>' (e.g. '@t draft_review_revise ...').\n")

    while True:
        user_input = input("Human > ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        # Stage 8 template run
        if user_input.lower().startswith("@t "):
            # Format: @t <template> <goal...>
            parts = user_input.split(" ", 2)
            if len(parts) < 3:
                print("\nAgent: Usage: @t <template> <goal>\n")
                continue
            _, template, goal = parts
            answer = orch.run_template(template=template, goal=goal, context={})
            print(f"\nAgent (orchestrated/{template}): {answer}\n")
            continue

        # Default: Stage 7 direct agent run
        answer = agent.run(user_input)
        print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    main()