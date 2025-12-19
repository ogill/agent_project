# config.py
from __future__ import annotations

from pathlib import Path

PROMPT_MODE = "strict"  # kept for compatibility with prompts.get_planner_prompt()
MODEL_NAME = "qwen2.5:7b-instruct"
MAX_REPLANS = 2

# ---------------------------
# Episodic memory (Stage 5)
# ---------------------------
MEMORY_ENABLED = True

# Use Path objects (important for Stage 6)
MEMORY_DIR = Path(".memory")
EPISODES_PATH = MEMORY_DIR / "episodes.jsonl"

# How much memory to inject each turn
MAX_RECENT_EPISODES = 6
MAX_RELEVANT_EPISODES = 2

# ---------------------------
# Semantic memory (Stage 6)
# ---------------------------
CHROMA_DIR = MEMORY_DIR / "chroma"

SEMANTIC_BACKEND = "chroma"   # chroma | none (mcp/lancedb later)
SEMANTIC_TOP_K = 5
SEMANTIC_MIN_SCORE = 0.15     # similarity threshold (0..1)

# ---------------------------
# Embeddings (computed in-app)
# ---------------------------
EMBEDDINGS_PROVIDER = "ollama"   # ollama | local (future)
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDINGS_MODEL = "nomic-embed-text"

SEMANTIC_DEBUG = False
SEMANTIC_INCLUDE_FAILURE_HITS = False

# ---------------------------
# MCP integration (Stage 7)
# ---------------------------

# Master switch — MUST remain False until MCP is explicitly enabled
MCP_ENABLED = False

# MCP servers exposed as tools to the agent
# Each server is a remote MCP endpoint (HTTP)
MCP_SERVERS = [
    # Example (disabled by default):
    # {
    #     "alias": "math",
    #     "endpoint": "http://localhost:8080/mcp",
    #     "timeout_ms": 5000,
    # }
]