"""Frozen constants from DECISIONS.md, in one place.

Experiment variables (topology, reward_mode, ...) do NOT live here; they go in
the per-condition config (Phase 1). These values stay fixed across conditions.
"""
import os

# Bump on ANY change to prompt text in swarm/agents/llm_agent.py. Part of config_hash().
PROMPT_VERSION = "v4"

# ---- Serving ----
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-14B-Instruct-AWQ")
TEMPERATURE = 0.7
MAX_TOKENS = 400
REQUEST_TIMEOUT_S = 120
JSON_RETRIES = 1  # retry once on bad JSON, then fall back to `pass`

# ---- World ----
N_AGENTS = 6
N_SECTIONS = 6

# ---- Resources ----
POOL_START = 30
POOL_REGEN = 10
POOL_CAP = 40
AGENT_START_CREDITS = 10
UPKEEP = 1
WRITE_COST = 3
MESSAGE_COST = 0
MAX_CLAIM = 5

# ---- Rounds / termination ----
MAX_ROUNDS_CEILING = 100
STALL_ROUNDS = 10

# ---- Memory ----
MESSAGE_HISTORY_ROUNDS = 3
PRIVATE_NOTE_MAX_WORDS = 100

# ---- Paths ----
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "logs")
CONFIG_DIR = os.path.join(ROOT, "configs")


def snapshot() -> dict:
    """All frozen constants that affect run behaviour (excludes serving endpoint and paths).

    Included in ExperimentConfig.config_hash() so runs never mix across settings or
    prompt versions.
    """
    exclude = {"VLLM_BASE_URL", "VLLM_API_KEY", "ROOT", "LOG_DIR", "CONFIG_DIR"}
    return {
        k: v for k, v in globals().items()
        if k.isupper() and k not in exclude and isinstance(v, (int, float, str, bool))
    }
