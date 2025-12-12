# config.py

# Local Ollama endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"

# Default model
MODEL_NAME = "deepseek-r1:8b"

# Enable or disable debug logging for the agent
DEBUG_AGENT = True  # set to False to turn off debug logs

# If True, also print the full prompts sent to the LLM (can be noisy)
DEBUG_AGENT_PROMPTS = True

# Maximum number of ReAct reasoning steps per user turn
MAX_REACT_STEPS = 5

# Planner debug
DEBUG_PLANNER = True

# -------------------------
# Stage 4: Dynamic replanning knobs
# -------------------------

# Maximum times the agent is allowed to replan in a single turn
MAX_REPLANS = 2

# Increase time budget because replanning can add extra LLM calls
# (Optional but recommended)
LLM_TIMEOUT_SECONDS = 300