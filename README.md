Local Agent Project — Architecture & Stage Progression

This project implements a single-user, local agent system built incrementally to explore agent architectures, planning, tool execution, memory, and protocol-based extensibility.

The system is designed with clear stage boundaries, where each stage introduces new capability without refactoring or invalidating prior decisions.
Earlier stages are treated as production-stable once completed.

⸻

High-Level Architecture

At its core, the system follows a Plan → Execute → Observe → Replan → Compose loop:
	•	Planner: Generates a structured JSON plan using a strict schema
	•	Executor: Executes tool calls sequentially
	•	Observations: Captures tool outputs and failures
	•	Replanning: Deterministically triggered on failure
	•	Composer: Produces the final user-facing response
	•	Memory (Stages 5–6): Episodic + semantic memory injected conditionally

The agent is local-first, runs entirely on the developer machine, and uses Ollama for LLM inference and embeddings.

⸻

Stage       Overview                 
Stage                Focus                            Status
Stage 0              Environment & scaffolding        ✅ Complete
Stage 1              Minimal agent loop               ✅ Complete
Stage 2              Tool calling                     ✅ Complete
Stage 3              Planner-based execution          ✅ Complete
Stage 4              Replanning & failure handling    ✅ Complete
Stage 5              Episodic memory                  ✅ Complete
Stage 6              Semantic memory                  ✅ Complete (locked)
Stage 7              MCP integration                  ✅ Complete
Stage 8              Multi-agent orchestration        ⏳ Future


Stage 0 — Environment & Foundations

Goal: Establish a clean, reproducible local development environment.

Key decisions:
	•	Python virtual environment (.venv)
	•	Explicit dependency bootstrap script
	•	Local execution only (no cloud services)
	•	Ollama used as the LLM runtime

This stage focused purely on developer ergonomics and repeatability.

⸻

Stage 1 — Minimal Agent Loop

Goal: Create the smallest viable agent loop.

Capabilities:
	•	Accept user input
	•	Send prompt to LLM
	•	Return model output directly

No tools, no memory, no planning — just a baseline loop to build upon.

⸻

Stage 2 — Tool Calling

Goal: Introduce structured tool execution.

Key features:
	•	Tools defined as:
	•	description
	•	args_model (Pydantic)
	•	fn callable
	•	LLM instructed to choose tools
	•	Deterministic execution of chosen tool

This stage established the tool abstraction that all later stages depend on.

⸻

Stage 3 — Planner-Based Execution

Goal: Separate reasoning from execution.

Major change:
	•	Introduced a Planner that outputs a strict JSON plan:

  {
  "goal": "...",
  "steps": [
    { "id", "description", "tool", "args", "requires" }
  ]
}

lanner guardrails:
	•	JSON-only output
	•	Fixed schema
	•	Explicit compose_answer final step
	•	No intermediate non-tool steps

This stage made the system inspectable, debuggable, and deterministic.

⸻

Stage 4 — Replanning & Failure Handling

Goal: Make the agent robust to tool failures.

Capabilities added:
	•	Hard failures (exceptions)
	•	Soft failures (structured error payloads)
	•	Forbidden-tool enforcement during replans
	•	Deterministic fallback plan when replanning violates constraints

Replanning rules:
	•	Failed tools cannot be retried unless explicitly marked transient
	•	Observations must be used instead of repeating work
	•	Final fallback never calls tools

This stage established operational safety.

⸻

Stage 5 — Episodic Memory

Goal: Add short-term memory without changing agent control flow.

Implementation:
	•	Append-only JSONL log of episodes
	•	Each episode includes:
	•	User input
	•	Plan
	•	Tool calls
	•	Observations
	•	Final answer

Memory usage:
	•	Last N recent episodes injected conditionally
	•	Never blindly appended to prompts
	•	Memory-aware planner shortcuts for:
	•	Explicit memory writes
	•	Likely memory recall questions

⸻

Stage 6 — Semantic Memory (LOCKED)

Goal: Enable semantic retrieval across past interactions.

Implementation details:
	•	ChromaDB as vector store
	•	Embeddings computed in-app via Ollama (nomic-embed-text)
	•	Stored data model:
{ "id", "text", "metadata" }



Retrieval strategy:
	•	Top-K semantic hits (score-thresholded)
	•	Plus last N recent episodes
	•	Failure-related episodes excluded by default
	•	Failure memory included only for explicit debugging queries

Non-goals (explicitly locked):
	•	No state memory / user profile
	•	No database swaps
	•	No memory refactors
	•	No cloud services

Stage 6 is treated as production-stable and immutable.

⸻

Stage 7 — MCP Integration (Model Context Protocol)

Goal: Integrate external tools via MCP without changing the agent core.

Key Principle

MCP is a tool provider, not an agent controller.

What Was Added
	•	MCP HTTP client
	•	MCP registry (tool discovery & name routing)
	•	MCP provider (thin adapter)
	•	Runtime injection of MCP tools into existing TOOLS
	•	Reference MCP server (mcp_server_math) running as a separate process
	•	Automated smoke test validating MCP discovery

What Was NOT Changed
	•	Planner logic
	•	Agent control flow
	•	Memory system
	•	Replanning semantics

Architecture
Agent
 ├─ Planner (unchanged)
 ├─ Tool Registry
 │    ├─ Local tools
 │    └─ MCP tools (discovered at runtime)
 ├─ Executor
 │    ├─ Local execution
 │    └─ MCP HTTP calls
 └─ Memory (Stages 5–6)

 MCP Server
	•	Runs independently via Uvicorn / FastAPI
	•	Exposes:
	•	GET /mcp/tools
	•	POST /mcp/invoke
	•	Tools execute inside the server process
	•	Demonstrated with math tools (add_numbers, subtract_numbers)

Guardrails
	•	MCP tools treated identically to local tools
	•	Planner validation enforces:
	•	Known tool names only
	•	No $ref or symbolic placeholders in tool args
	•	Replanning respects forbidden MCP tools on failure

Validation
	•	test_mcp_smoke.py asserts MCP tools are discoverable
	•	Requires MCP server running
	•	Confirms Stage 7 integration contract

⸻

How to Run

1. Start MCP Server
uvicorn mcp_server_math.server:app --port 8080

2. Run Agent
python main.py

3. Run Tests
pytest -q

Current State (After Stage 7)
	•	Single-user local agent
	•	Planner-driven execution
	•	Robust replanning
	•	Episodic + semantic memory
	•	External tool integration via MCP
	•	Clean, test-validated architecture
	•	main branch represents Stage 7 complete

⸻

Stage 8 (Future)

Planned, but not implemented:
	•	Multi-agent orchestration
	•	Agent-to-agent communication via MCP
	•	Shared MCP servers across agents
	•	Role-specialised agents

Stage 7 deliberately sets the foundation for this without premature abstraction.

⸻

Design Philosophy
	•	Incremental stages
	•	No refactors of completed stages
	•	Thin adapters over deep abstraction
	•	Local-first, inspectable systems
	•	Explicit non-goals

This project prioritises architectural clarity over framework churn.

   

  
