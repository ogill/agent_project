# main.py

from __future__ import annotations

from agent import Agent
from config import MODEL_NAME


def main() -> None:
    agent = Agent()

    print(f"ReAct Agent via Ollama ({MODEL_NAME})")
    print("Type 'exit' or 'quit' to stop.")
    print("Use ':multi' to enter multiline mode, ':end' to submit.\n")

    multiline_mode = False
    multiline_buffer: list[str] = []

    while True:
        try:
            if multiline_mode:
                line = input("Human (multi) > ")
            else:
                line = input("Human > ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        stripped = line.strip()

        # --------------------------------------------------
        # GLOBAL EXIT — works in all modes
        # --------------------------------------------------
        if stripped.lower() in {"exit", "quit", ":exit", ":quit"}:
            print("Exiting.")
            break

        # --------------------------------------------------
        # ENTER MULTILINE MODE
        # --------------------------------------------------
        if not multiline_mode and stripped == ":multi":
            multiline_mode = True
            multiline_buffer = []
            print("(multiline mode — type ':end' on its own line to submit)")
            continue

        # --------------------------------------------------
        # MULTILINE MODE HANDLING
        # --------------------------------------------------
        if multiline_mode:
            if stripped == ":end":
                # Submit multiline input
                user_input = "\n".join(multiline_buffer).strip()
                multiline_mode = False
                multiline_buffer = []

                if not user_input:
                    print("(empty input ignored)\n")
                    continue

                try:
                    answer = agent.run(user_input)
                    print(f"\nAgent: {answer}\n")
                except Exception as e:
                    print(f"\n[ERROR] {e}\n")

                continue

            # Still collecting multiline input
            multiline_buffer.append(line)
            continue

        # --------------------------------------------------
        # SINGLE-LINE MODE
        # --------------------------------------------------
        if not stripped:
            continue

        try:
            answer = agent.run(stripped)
            print(f"\nAgent: {answer}\n")
        except Exception as e:
            print(f"\n[ERROR] {e}\n")


if __name__ == "__main__":
    main()