# Stage 8 Multi-Agent Orchestration

## Purpose

Stage 8 introduces a Multi-Agent Orchestrator that coordinates multiple Stage-7 agents to solve a single user goal, while preserving Stage 7 behavior exactly (strict JSON planning, deterministic execution loop, guardrails, and MCP tool treatment).

## Stage 7 invariants that must remain true

The following are non-negotiable across all Stage 8 work:

- The Planner produces strict JSON plans.
- The Agent executes plans with tool calls and supports replanning.
- Guardrails remain enforced:
  - No $ref in tool args
  - Forbidden tool enforcement
  - Deterministic replan fallback
- MCP tools are treated as external, stateless capabilities.
- Stage 8 must not require changing Stage 7 prompts or Stage 7 agent control flow.

## Conceptual model

- Agent: “Given a task, plan and execute it correctly using tools and replanning.”
- Orchestrator: “Who should do what, in what order, and how do we merge results?”

Stage 8 adds coordination around agents, not new reasoning inside agents.

## Architecture

### Components

- Orchestrator (new): pure coordination layer
- Agent Pool (existing Stage-7 Agents): same agent behavior, role-based configuration only
- Tool Registry (existing): local tools + MCP tools, with Stage-7 enforcement

### WorkItem contract

A WorkItem is the unit of orchestration.

Minimal shape:

{
  "id": "task-001",
  "assigned_agent": "generalist",
  "goal": "Return constraints and approach",
  "inputs": {},
  "expected_output": {}
}

Notes:
- assigned_agent selects a role configuration, not a different code path.
- inputs are structured context; the Orchestrator must still call Stage-7 Agent.run(user_input: str).

### Orchestrator state

Run-level state is owned by the Orchestrator, not by agents:

- run_id
- user_goal + run_context
- work_queue (pending WorkItems)
- work_results (outputs keyed by WorkItem id)
- trace_log (audit events)

Agent state remains per-run / per-WorkItem as in Stage 7.

## Stage 8.1 scope

Design-only:
- Orchestrator responsibilities and boundaries
- WorkItem contract, routing rules, state model
- Deterministic failure handling and merge policy
- Regression plan to prove Stage 7 parity

No code changes.

## Stage 8.2 scope

Introduce an Orchestrator module that:
- creates a single WorkItem
- routes it to a single agent role (generalist)
- calls Stage-7 Agent.run(user_input) unchanged
- returns the agent output unchanged

Acceptance criteria:
- Stage 7 behavior is preserved.
- A parity test demonstrates Agent.run(user_input) equals Orchestrator.run(user_input) for the same input.

## Stage 8.2.1 scope

Extend Orchestrator from 1 WorkItem to N WorkItems (sequential), with deterministic merge.

Implementation constraints:
- Orchestrator remains code-first (no LLM calls inside orchestrator).
- WorkItem execution is sequential (no parallelism yet).
- Merge must be deterministic and stable across runs.

Deterministic merge policy (8.2.1):
- Preserve WorkItem order.
- Emit a combined output with explicit boundaries per WorkItem.

Acceptance criteria:
- The single-WorkItem parity test remains unchanged and passing.
- A new multi-WorkItem test proves that outputs from all WorkItems appear in the final merged response.

## Stage 8 stages (single sentence each)

- Stage 8.1: Define the Orchestrator, WorkItem contract, agent roles, routing rules, state model, and deterministic failure/merge policies without changing code.
- Stage 8.2: Implement a minimal Orchestrator that can run a single WorkItem through an unchanged Stage-7 agent and prove parity via tests.
- Stage 8.2.1: Extend Orchestrator to run multiple WorkItems sequentially and merge outputs deterministically, while keeping Stage-7 parity intact.
- Stage 8.3: Add structured shared context (facts and artifacts) passed into WorkItems so agents consume consistent inputs without shared mutable state.
- Stage 8.4: Add safe parallel execution for independent WorkItems with concurrency limits, timeouts, and deterministic merge rules.
- Stage 8.5: Add critique/verification patterns (reviewer agent, contradiction checks, quality gates) without weakening tool policies or deterministic fallback guarantees.
- Stage 8.6: Add optional orchestration memory (run summaries, outcomes) separate from MCP stateless tooling and separate from per-agent episodic state.
- Stage 8.7: Hardening and observability: traces, reproducibility, policy auditing, and regression tests proving Stage-7 parity under orchestration.

## Regression requirements

Before merging any Stage 8 code:
- Existing Stage 7 tests pass unchanged.
- Parity test remains: Stage 7 direct Agent.run(user_input) equals Stage 8 Orchestrator.run(user_input) for the same input (within defined tolerance).
- Add targeted tests for multi-WorkItem orchestration behavior, merge stability, and role routing.