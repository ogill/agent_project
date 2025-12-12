# io_utils.py

HUMAN_LABEL = "Human"
AGENT_LABEL = "Agent"


def get_user_input() -> str:
    """
    Get a line of input from the human user.
    This is the *only* place where the CLI prompt string lives.
    """
    return input(f"{HUMAN_LABEL} > ").strip()


def print_agent_response(text: str) -> None:
    """
    Print the agent's response in a consistent, future-proof way.
    """
    print(f"{AGENT_LABEL}: {text}\n")