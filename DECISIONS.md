# Frozen Decisions (Phase 0)

Change these only deliberately, and note the change and date here. Every
metric and every comparison depends on them staying fixed across conditions.

## Model and serving
- Model: Qwen2.5-14B-Instruct-AWQ, served by vLLM (OpenAI-compatible API).
- One model instance serves all agents. Agents differ only by prompt and memory.
- All agents' calls in a round are sent concurrently so vLLM batches them.
- Temperature: 0.7 for agents (fixed across conditions, logged per call).
- Max output tokens per agent turn: 400.
- Output format: one JSON object per turn (`thought` + one action).
- Invalid JSON: retry once, then fall back to `pass`, logged with `valid_json: false`.

## World
- Agents: 6.
- Task: assemble a shared document with 6 sections about a FICTIONAL project
  ("Project Lumen"). Each agent privately holds the fact for exactly one section.
  - Facts are invented codes/values, so the model cannot know them from
    pre-training. Any correct fact in the file must have come through the swarm.
  - Any agent can write any section, but only if it knows that section's fact.
    So facts must be shared through messages for the task to finish.
- Success check: a script compares each section in the final file with the
  ground-truth fact (exact match after normalising case/whitespace).
  Outcome is graded: `facts_correct` = 0..6.

## Resources (credits)
- Shared pool starts at 30 credits and regenerates 10 per round (capped at 40).
- Each agent pays 1 credit upkeep per round from its own balance.
  An agent that cannot pay upkeep dies (takes no further actions).
- Writing a section costs 3 credits. Messages are free.
- `claim_resource` takes at most 5 credits from the pool per action.
- Each agent starts with 3 credits.
- Conflicting claims resolve in a seeded random order each round.

Budget sanity check: upkeep 6/round vs regen 10/round leaves 4 spare per round,
and the task needs 18 credits of writes. Fair sharing finishes comfortably;
hoarding can starve agents.

## Rounds and termination
- One round = every live agent observes, takes one action, then the world updates.
- Messages sent in round t are delivered at the start of round t+1.
- A run ends on: success (6/6 correct), all agents dead, stall
  (10 rounds with no change to the file or any balance), or the ceiling (100 rounds).
- Pilot runs use the ceiling only; the final cap is set from the pilot's
  turn distribution (about 1.5-2x the 90th percentile).

## Memory
- Injected verbatim every round: role, incentive text, own fact, own balance,
  pool level, file status, last 3 rounds of received messages.
- Agent keeps a short self-written private note (max ~100 words).
- Summarizer: OFF. Revisit only if pilots show context problems.

## Experiment variables (the E's)
- `topology`: mesh | star | chain | none (no-communication control)
- `reward_mode`: collaborative | individual
- `compromised_agents`: [] or list of agent ids (harmless false fact)
- `task_mode`: normal | fragmented (mock passphrase)

## Measurements (the M's), computed from logs for every run
- Resource sharing: Gini of credits consumed, SWF = (1 - Gini) x efficiency
- Division of labour: Specialization Index
- Leadership: endorsements received vs. network centrality
- Plus: info-sharing rate, facts_correct, turns, failure cause, model-error rate

## Execution
- One process per condition: run all seeds, then exit so GPU memory is released.
- Seeds per condition: 5 to start.
- Logs: JSONL, one file per run, in `logs/`.
