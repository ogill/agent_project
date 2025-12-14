# main.py

from __future__ import annotations

from agent import Agent
from config import MODEL_NAME


def main() -> None:
    agent = Agent()
    print(f"ReAct Agent via Ollama ({MODEL_NAME})")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("Human > ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        answer = agent.run(user_input)
        print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    main()