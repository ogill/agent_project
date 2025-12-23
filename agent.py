# agent.py

from __future__ import annotations

from typing import Any, Dict

from config import MODEL_NAME
from planner import Planner
from planner_executor import PlannerExecutor
from memory import EpisodeStore
from plan_types import Plan


class Agent:
    """
    Agent = control flow:
    - ask Planner for a plan
    - execute tool steps deterministically (via PlannerExecutor)
    - replan on hard/soft tool failures
    - always end with a user-facing final answer
    """

    def __init__(self, max_replans: int = 2) -> None:
        self.planner = Planner()
        self.executor = PlannerExecutor(
            max_replans=max_replans,
            trace=False,
        )
        self.memory = EpisodeStore()

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> str:
        print(f"[AGENT] stage7.react model={MODEL_NAME}")

        # 1) Initial plan
        plan = self.planner.generate_plan(
            user_input=user_input,
            observations_text="",
            failure_text="",
            is_replan=False,
        )

        # 2) Execute plan with replanning support
        exec_result = self.executor.execute_with_replanning(
            user_input=user_input,
            initial_plan=plan,
            planner=self.planner,
        )

        # 3) Compose final answer (LLM)
        final_answer = self._compose_answer(
            user_input=user_input,
            observations=exec_result.observations,
            last_failure_text=exec_result.last_failure_text,
        )

        # 4) Persist episodic memory
        self.memory.append(user_input, final_answer)
        return final_answer

    # ------------------------------------------------------------------
    # Final answer composition
    # ------------------------------------------------------------------

    def _compose_answer(
        self,
        *,
        user_input: str,
        observations: Dict[str, Any],
        last_failure_text: str | None,
    ) -> str:
        """
        Final LLM call that produces the user-facing answer.
        """
        from llm_client import call_llm

        has_failure = bool((last_failure_text or "").strip())
        prompt_parts: list[str] = []

        print(f"[AGENT] stage7.react compose_answer model={MODEL_NAME}")

        # --------------------------------------------------
        # Failure-aware guardrails
        # --------------------------------------------------
        if has_failure:
            prompt_parts.append(
                "A failure occurred during tool execution.\n"
                "Rules:\n"
                "- Explain the failure ONLY using the provided failure text and observations.\n"
                "- Do NOT invent tools, error codes, or causes.\n"
            )

            prompt_parts.append(
                "Failure text:\n"
                + last_failure_text
            )
        else:
            prompt_parts.append(
                "Rules:\n"
                "- Answer directly from the user request and observations.\n"
                "- Do NOT mention tools, orchestration, retries, or planning.\n"
            )

        # --------------------------------------------------
        # Episodic memory (only when clean run)
        # --------------------------------------------------
        if not has_failure:
            memory_context = self.memory.build_context(user_input)
            if memory_context:
                prompt_parts.append(
                    "Relevant prior context (episodic memory):\n"
                    + memory_context
                    + "\n\nIMPORTANT: Use prior context ONLY if it is clearly relevant."
                )

        # --------------------------------------------------
        # User request
        # --------------------------------------------------
        prompt_parts.append("User request:\n" + user_input)

        # --------------------------------------------------
        # Observations (ground truth)
        # --------------------------------------------------
        if observations:
            obs_lines = []
            for step_id, value in observations.items():
                obs_lines.append(f"- {step_id}: {value}")
            prompt_parts.append(
                "Observations (ground truth from tools):\n"
                + "\n".join(obs_lines)
            )

        # --------------------------------------------------
        # Final instructions
        # --------------------------------------------------
        if has_failure:
            prompt_parts.append(
                "Instructions:\n"
                "- Clearly explain what failed and what could still be concluded.\n"
                "- Be factual and explicit about limitations.\n"
            )
        else:
            prompt_parts.append(
                "Instructions:\n"
                "- Produce a clear, complete answer for the user.\n"
            )

        prompt = "\n\n".join(prompt_parts)
        return call_llm(prompt)