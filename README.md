# Swarm Hack: decentralized agent network simulator

Common simulator + measurement and logging engine for multi-agent LLM experiments.
Frozen design choices are in `DECISIONS.md`.

## Layout
```
swarm/
  settings.py      frozen constants (Phase 0)
  config.py        experiment config + action + log schemas (Phase 1)
  world.py         world engine, no LLM (Phase 2)
  logger.py        JSONL writer (Phase 2)
  agents/bots.py   scripted bots for testing (Phase 2)
  agents/llm_agent.py  LLM agent (Phase 4)
  metrics.py       Gini, SWF, SI, leadership, info-sharing (Phase 3)
  diagnose.py      failure causes, model-error rate (Phase 3)
  run.py           one condition x N seeds, one process (Phase 6)
  launch.py        runs each condition in a fresh subprocess (Phase 6)
  dashboard.py     Streamlit (Phase 7)
configs/           one YAML per condition
logs/              one JSONL per run
scripts/           checks and utilities
tests/             pytest
```

## Setup
```
pip install -r requirements.txt
export VLLM_BASE_URL=http://localhost:8000/v1
export MODEL_NAME=<exact id your vLLM server reports>
python scripts/check_vllm.py
```
