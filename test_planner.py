# agent_project/test_planner.py

from planner import Planner


def main() -> None:
    user_input = (
        "Summarize the content of https://example.com into three bullet points, "
        "and highlight any key risks mentioned."
    )

    # For now we can keep tools_spec empty or add a simple description string.
    tools_spec = ""

    planner = Planner()
    plan = planner.generate_plan(user_input, tools_spec)

    print("=== Generated Plan ===")
    print(f"Goal: {plan.goal}")
    print("Steps:")
    for step in plan.steps:
        print(
            f"- [{step.id}] {step.description} "
            f"(tool={step.tool}, requires={step.requires})"
        )


if __name__ == "__main__":
    main()