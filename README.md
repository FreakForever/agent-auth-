# Swarm Hack: a simulator for decentralised LLM agent swarms

Six LLM agents must pool split secret codes into a shared document while managing a common
credit pool. One simulator and one measurement engine; each experiment is a YAML config.
Every number is computed offline from JSONL run logs, and every metric is validated on
scripted bots before it is used on LLM runs.

Three experiments are included (80 runs, logs in `logs/`):

| Experiment | Question | Configs | Logs |
|---|---|---|---|
| 1. Topology x reward | Do network structure (mesh / star / chain) and reward framing (team / individual) change whether agents share information and credits and finish the task? | `configs/grid/` | `logs/grid/` |
| 2. Compromised agent | If one agent is secretly told to push a false value, how far does it spread, and does anyone question it? | `configs/compromised/` | `logs/compromised/` |
| 3. Fragmented secret | Do agents protect a confidential passphrase when the request is split into innocent one-word pieces, while still answering harmless questions? | `configs/secret/` | `logs/secret/` |

Paper: https://drive.google.com/file/d/1e4Y1CG2390-uSIqjTi_vsJq1TG7XXy9Q/view?usp=sharing 
 
Live dashboard: https://exprimnt101.streamlit.app/

---

## How it works (one paragraph)

Each section of the document has a fictional code `WORD-1234` split into two halves held by
different agents, so nothing can be completed without communicating. Agents start with 10
credits, pay 1 per round to stay in the game, pay 3 to write a section and can claim up to 5
from a shared pool (start 30, +10 per round, cap 40). Each round every agent takes one action
(`send_message`, `claim_resource`, `give_resource`, `write_section`, `propose`, `endorse`,
`pass`). Messages reach only network neighbours, one round later. A run ends on success,
when everyone is removed, after 10 rounds without a document change, or at `max_rounds`.
Frozen design numbers and the reasons for every change are in `DECISIONS.md`.

---

## 1. Set up the model server (vLLM, on a GPU machine)

The agents call an OpenAI-compatible API. We serve **Qwen2.5-14B-Instruct-AWQ** with vLLM on
a separate Linux GPU machine; all code, runs and logs stay on your own computer.

**Tested on:** one NVIDIA A100 40GB (Google Cloud VM), Ubuntu, Python 3.10+.

On the **GPU machine**:

```bash
# 1. Python environment with vLLM (pulls in PyTorch + CUDA wheels)
python3 -m venv ~/swarm-hack/.venv
source ~/swarm-hack/.venv/bin/activate
pip install --upgrade pip
pip install vllm

# 2. Start the server inside tmux so it keeps running after you disconnect
tmux new -s vllm
vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ \
  --host 127.0.0.1 --port 8000 \
  --quantization awq --dtype half \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 32
# The first start downloads the model from Hugging Face (~10 GB).
# Wait for "Application startup complete", then detach with Ctrl-b d.
# Reattach later with: tmux attach -t vllm

# 3. Check it answers (on the GPU machine)
curl -s http://127.0.0.1:8000/v1/models
```

Notes:
- `--host 127.0.0.1` keeps the server private to the machine; you reach it through an SSH
  tunnel (step 3 below), so no port is exposed to the internet.
- `--max-num-seqs 32` lets vLLM batch all agents of several parallel runs together.
- The GPU memory belongs to the vLLM process, so experiments can run in parallel without
  restarting the server.

## 2. Set up this repository (on your own computer)

```bash
git clone <this repo URL>
cd <repo folder>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt     # no vLLM or PyTorch needed locally

python -m pytest -q tests/           # no GPU needed; expect: 76 passed
```

## 3. Connect to the model server

Open an SSH tunnel in a **separate terminal** and leave it running:

```bash
ssh -N -L 8000:127.0.0.1:8000 -o ServerAliveInterval=30 <user>@<gpu-machine-address>
```

Then, in your working terminal:

```bash
source .venv/bin/activate
export MODEL_NAME=Qwen/Qwen2.5-14B-Instruct-AWQ
export VLLM_BASE_URL=http://localhost:8000/v1    # this is the default

curl -s localhost:8000/v1/models | head -c 120   # should list the model
python scripts/check_vllm.py                     # 6 concurrent JSON calls; all must pass
```

Every run starts with a pre-flight check and stops immediately if the server is unreachable,
so a dead tunnel cannot produce a fake log. If a tunnel drops mid-run, delete any log written
during the outage (all agents "pass" and starve) and re-run; finished runs are skipped.

## 4. Run a smoke test (one run, about a minute)

```bash
python scripts/llm_smoke.py                      # mesh, team reward, 15 rounds; prints every action
python scripts/llm_smoke.py --topology chain --reward individual --rounds 20
python scripts/llm_smoke.py --compromised 0      # with agent 0 as the secret liar
```

## 5. Run the experiments

Each command runs every seed of every config given and writes one JSONL log per run.
`--parallel` runs that many seeds at once (all share the one vLLM server). On macOS, prefix
long runs with `caffeinate -i` to stop the laptop sleeping. Times below are wall-clock on one
A100 40GB.

```bash
# Experiment 1: topology x reward, plus solo-baseline and no-messaging controls
# 8 configs x 5 seeds = 40 runs, ~10 min
python -m swarm.run configs/grid/*.yaml --log-dir logs/grid --parallel 3
python -m swarm.report logs/grid

# Experiment 2: compromised agent (control = the matching Experiment 1 runs)
# 4 configs x 5 seeds = 20 runs, ~5 min
python -m swarm.run configs/compromised/*.yaml --log-dir logs/compromised --parallel 3
python -m swarm.report logs/compromised

# Experiment 3: fragmented secret request
# 4 configs x 5 seeds = 20 runs, ~2 min
python -m swarm.run configs/secret/*.yaml --log-dir logs/secret --parallel 4
python -m swarm.secret_task logs/secret
```

Rules that keep results clean:
- **Use a fresh `--log-dir` whenever the code, prompt or settings change.** The report groups
  runs by condition name, so mixing versions in one folder would average them together.
- **Runs resume.** Re-running a command skips every run whose log is complete. Use `--no-skip`
  to force a re-run.
- **Run IDs are versioned.** Each log is named `<condition>__s<seed>__<hash>.jsonl`. The hash
  covers the config plus all frozen settings, `PROMPT_VERSION` and `ENGINE_VERSION` (and the
  compromise / secret-task versions when those modes are on), so a changed prompt or rule never
  silently reuses old logs. Bump the matching version in `swarm/settings.py` after any change.

## 6. Look at the results

```bash
python scripts/show_run.py logs/grid/<file>.jsonl              # readable transcript of one run
python scripts/show_run.py <log> --rounds 1-5 --agent 3        # filter by rounds / agent
streamlit run swarm/dashboard.py                                # dashboard (compare + replay)
```

The dashboard reads `logs/grid`, `logs/compromised` and `logs/secret`, so it works without a
GPU on the logs already in this repository. The shared document is not a separate file: its
state after every round is stored in the `document` field of each `round_end` record.

---

## Defining a new experiment

An experiment is one YAML file. Everything not listed keeps its default.

```yaml
name: my_condition            # used in log names and reports
topology: mesh                # mesh | star | chain | none
reward_mode: collaborative    # collaborative (team) | individual
seeds: [0, 1, 2, 3, 4]        # seed k = the same codes and dealing in every condition
max_rounds: 47
n_agents: 6                   # 1 = solo baseline
start_credits: 60             # optional; default 10 per agent
compromised_agents: [0]       # optional; default [] (off). Agent ids given a secret liar objective
task_mode: normal             # normal | secret_direct | fragmented | secret_benign (Experiment 3)
```

Before using a new metric on LLM runs, check it gives the known answer on the scripted bots in
`swarm/agents/bots.py` (see `tests/`).

## Repository layout

```
swarm/
  settings.py          frozen constants + PROMPT_VERSION / ENGINE_VERSION + snapshot()
  config.py            ExperimentConfig, action schema + parser, log record schemas
  world.py             game engine (no LLM): rounds, topology, credits, grading, logging
  logger.py            JSONL writer/reader (one file per run, flushed per line)
  agents/bots.py       scripted bots with known behaviour (validate engine and metrics)
  agents/llm_agent.py  prompts, batched async calls to vLLM, retry-then-pass, pre-flight check
  metrics.py           Gini/SWF, labour division, leadership, knowledge timeline, contagion
  diagnose.py          one failure cause per incorrect section
  secret_task.py       Experiment 3 task: episode, metrics, report
  run.py               runs all seeds of one or more configs (parallel, resumable)
  report.py            per-condition tables, go/no-go checks
  dashboard.py         Streamlit dashboard
configs/               grid/, compromised/, secret/ (experiments); pilot/ (development)
scripts/               check_vllm.py, llm_smoke.py, show_run.py
tests/                 pytest suite (no GPU needed)
logs/                  run logs (grid/, compromised/, secret/ = results; pilot*/ = development)
DECISIONS.md           frozen design numbers and the reason for every change
```

## Safety note

All "harm" is a harmless proxy: fictional codes, a fictional passphrase, and a scripted outside
requester. No hazardous content is requested or generated. See the paper's Appendix A for
dual-use considerations.
