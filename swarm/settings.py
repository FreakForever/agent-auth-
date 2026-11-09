"""Frozen constants from DECISIONS.md, in one place.

Experiment variables (topology, reward_mode, ...) do NOT live here; they go in
the per-condition config (Phase 1). These values stay fixed across conditions.
"""
import os

# Bump on any change to World mechanics (termination, credits, grading) that the config
# doesn't capture. Part of config_hash(), so old logs are never resumed under new rules.
ENGINE_VERSION = "e2"  # e2: stall back to document-only after pilot v7
# Bump on ANY change to prompt text in swarm/agents/llm_agent.py. Part of config_hash().
PROMPT_VERSION = "v6"  # v5: write-affordability line; v6: it counts upkeep, solo prompt has no comms actions
# Phase 8 compromised-agent mode (liar prompt + target selection). Added to config_hash ONLY
# when compromised_agents is non-empty, so normal configs keep their hashes.
COMPROMISE_VERSION = "c1"
# Phase 9 secret-passphrase task (swarm/secret_task.py). In the hash only when task_mode != normal.
SECRET_TASK_VERSION = "s2"  # s2: mixed public/secret cards, one question per round, score paragraph

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
    exclude = {"VLLM_BASE_URL", "VLLM_API_KEY", "ROOT", "LOG_DIR", "CONFIG_DIR", "COMPROMISE_VERSION",
               "SECRET_TASK_VERSION"}
    return {
        k: v for k, v in globals().items()
        if k.isupper() and k not in exclude and isinstance(v, (int, float, str, bool))
    }
