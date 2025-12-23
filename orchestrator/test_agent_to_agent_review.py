from __future__ import annotations

from orchestrator.orchestrator import Orchestrator
from orchestrator.roles import RoleRegistry, RoleSpec
from orchestrator.role_names import GENERALIST, REVIEWER


class DraftThenReviseAgent:
    """
    Single agent instance used for both draft + revise steps.
    Distinguish behavior by looking at the goal prefix in the user_input.
    """

    def run(self, user_input: str) -> str:
        if "Draft an initial answer" in user_input:
            return "DRAFT_v1"
        if "Revise the original draft" in user_input:
            # In a real system you'd rewrite, but for test we just emit a new token.
            return "FINAL_v2"
        return "UNKNOWN"


class ReviewAgent:
    def run(self, user_input: str) -> str:
        # We expect shared context to be injected as a dict string that includes the artifact key.
        saw_draft = "draft.output" in user_input
        return f"REVIEW saw_draft={saw_draft}"


def test_draft_review_revise_template_propagates_artifacts():
    roles = RoleRegistry(
        {
            GENERALIST: RoleSpec(name=GENERALIST, agent=DraftThenReviseAgent()),
            REVIEWER: RoleSpec(name=REVIEWER, agent=ReviewAgent()),
        }
    )

    orch = Orchestrator(role_registry=roles)

    out = orch.run_template(
        template="draft_review_revise",
        goal="Return exactly the string: OK",
        context={},
    )

    # Review must see draft artifact injected
    assert "REVIEW saw_draft=True" in out

    # Revise must run and produce the final output
    assert "FINAL_v2" in out