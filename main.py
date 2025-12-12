# main.py


"""

You have built a local multi-step Agentic AI system with a clean pipeline: 
    a Planner LLM first converts any user request into a structured multi-step Plan (JSON), 
    specifying which tools to call and in what order; then a ReAct Executor runs each step, 
    sending structured prompts to the LLM that enforce Thought → Action → Final Answer reasoning, 
    calling only the required tool for that step; Tools (time, weather, fetch_url, summarize_text) 
    are real Python functions the agent can invoke; each tool result becomes an Observation fed back 
    into the next ReAct step; the final “compose_answer” step produces a coherent Final Answer using 
    all accumulated observations. The system is fully deterministic, debuggable, and production-patterned, 
    with strict tool schemas, explicit plans, and complete prompt/response transparency.

"""







# main.py

from agent import SimpleAgent
from io_utils import get_user_input, print_agent_response
from config import MODEL_NAME


def main() -> None:
    print(f"ReAct Agent via Ollama ({MODEL_NAME})")
    print("Type 'exit' or 'quit' to stop.\n")

    agent = SimpleAgent()

    while True:
        try:
            user_input = get_user_input()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input or not user_input.strip():
            continue

        if user_input.strip().lower() in {"exit", "quit"}:
            break

        try:
            answer = agent.handle_turn(user_input.strip())
        except Exception as e:
            print(f"[ERROR] Agent failed: {e!r}")
            continue

        print_agent_response(answer)


if __name__ == "__main__":
    main()