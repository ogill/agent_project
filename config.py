# config.py

PROMPT_MODE = "strict"  # kept for compatibility with prompts.get_planner_prompt()

MODEL_NAME = "qwen2.5:7b-instruct"

MAX_REPLANS = 2

# Memory
MEMORY_ENABLED = True
MEMORY_DIR = ".memory"
EPISODES_FILE = "episodes.jsonl"

# How much memory to inject each turn
MAX_RECENT_EPISODES = 6
MAX_RELEVANT_EPISODES = 2