from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RoleAgent:
    """
    Lightweight adapter that makes the same underlying Stage-7 Agent behave differently per role
    by prefixing role instructions into the user_input.
    """
    role: str
    base_agent: Any

    def run(self, user_input: str) -> str:
        prefix = _role_prefix(self.role)
        return self.base_agent.run(f"{prefix}\n\n{user_input}")


def _role_prefix(role: str) -> str:
    role = (role or "").lower().strip()

    if role == "researcher":
        return (
            "ROLE: RESEARCHER\n"
            "Task: Extract requirements/constraints, ask clarifying questions, list assumptions.\n"
            "Output format:\n"
            "- Requirements\n- Constraints\n- Open questions\n- Assumptions"
        )

    if role == "reviewer":
        return (
            "ROLE: REVIEWER\n"
            "Task: Critically review the proposed approach. Be strict.\n"
            "Output format:\n"
            "- Risks\n- Gaps\n- Edge cases\n- Minimal test plan\n- Recommendation"
        )

    # default: generalist
    return (
        "ROLE: GENERALIST\n"
        "Task: Provide a practical implementation-oriented answer.\n"
        "Output format:\n"
        "- When agents make sense\n- When they don’t\n- Warning signs\n- Phased adoption approach"
    )
    