# Findings log

Source material for the writeup, organised by experiment. Each finding gives the result, the
evidence (log file + round, quotable), and how strong it is.

**Status legend.** `grid` = shown in a full experiment (5 seeds per condition; no significance
tests yet). `pilot` = seen in pilot runs only (n=1-5, a hint). Pilot logs are development
history: each folder used a different prompt or engine version, so never pool them with
experiment results. Cite them only to explain a design change, or as an illustrative excerpt.

**Wording rule for the writeup.** With 5 seeds per condition, differences *between conditions*
are **trends**: report what was observed, with the numbers and n, never as proven causes.
- Write: "indiv_mesh scored higher in our runs (2.4 vs 0.8 sections, n=5 each)."
- Not: "individual reward makes teams perform better."
- Example check: indiv_mesh vs collab_mesh correct sections, permutation test p ≈ 0.09
  (unpaired) / 0.25 (paired by seed): not significant.

Patterns pooled across many agents or runs (e.g. "57 of 70 agents who died never claimed",
"0 proposals in 30 team runs", "baseline 5/5 vs teams 0/30") can be stated directly.

| Folder | Prompt | Engine | Use |
|---|---|---|---|
| `logs/grid/` | v6 | e2 | **Experiment 1 results** (40 runs) |
| `logs/pilot_v8/` | v6 | e2 | final pilot, same code as the grid |
| `logs/pilot_v7/` | v6 | mixed | mixes two engine versions; don't cite |
| `logs/pilot/`, `pilot_v5/`, `pilot_v6/`, `logs/smoke__s0__8914ddfbdf.jsonl` | v4-v6 | e1 | development history only |

Show a transcript: `python scripts/show_run.py <log>`. Recompute tables: `python -m swarm.report logs/grid`.

---

# Experiment 1: Network topology x reward framing (Phase 6 grid)

**Question.** How do communication structure (who can message whom) and incentive framing
(team vs individual reward) change how LLM agents share information and resources?

**Design.** 6 agents (Qwen2.5-14B-Instruct-AWQ, temperature 0.7), shared 6-section document.
Each section's code is split into two halves held by different agents, so every section needs
communication. Prompt v6, engine e2, `max_rounds: 47`, stall = 10 rounds with no document change.

| Condition | Topology | Reward | Config |
|---|---|---|---|
| collab_mesh / indiv_mesh | everyone talks to everyone | team / individual | `configs/grid/{collab,indiv}_mesh.yaml` |
| collab_star / indiv_star | agent 0 is the hub | team / individual | `configs/grid/{collab,indiv}_star.yaml` |
| collab_chain / indiv_chain | 0-1-2-3-4-5 line | team / individual | `configs/grid/{collab,indiv}_chain.yaml` |
| none (control) | no messaging | team | `configs/grid/none_control.yaml` |
| baseline_single (control) | 1 agent, all 12 halves, 60 credits | team | `configs/grid/baseline_single.yaml` |

5 seeds each (0-4), 40 runs. Seeds fix the codes and the dealing, so seed *k* is the same
puzzle in every condition.

## Results table

Means over 5 runs. "Assemblable" = sections whose two halves both reached one live agent.

| Condition | Correct /6 | Assemblable /6 | Rounds (median) | Deaths | Total claimed | Never claimed /6 | gini_claimed |
|---|---|---|---|---|---|---|---|
| baseline_single | **6.0** | 6.0 | 6 | 0 | 0 | n/a | n/a |
| collab_mesh | 0.8 | 5.2 | 10 | 1.8 | 38 | 3.8 | 0.69 |
| indiv_mesh | **2.4** | 5.8 | 17 | 2.6 | 115 | 1.8 | 0.44 |
| collab_star | 0.8 | 3.8 | 12 | 3.0 | 55 | 3.6 | 0.73 |
| indiv_star | 0.6 | 3.0 | 14 | 2.6 | 87 | 2.8 | 0.65 |
| collab_chain | 0.6 | 3.2 | 14 | 2.6 | 70 | 3.4 | 0.67 |
| indiv_chain | 0.4 | 2.6 | 12 | 1.4 | 59 | 2.6 | 0.60 |
| none | 0.0 | 0.0 | 10 | 0 | 88 | 1.6 | 0.49 |

Outcomes: baseline 5/5 success. Team conditions: 29 stalled, 1 all agents dead (collab_star),
0 successes, 0 hit the round cap (longest run 27 rounds).

## Controls  `grid`

- **The model can do the task alone:** baseline 5/5 solved in exactly 6 rounds, 30/30 writes correct.
- **Valid output:** model error rate (invalid JSON) 0.0% in all 40 runs.
- **No guessing:** no-messaging control scored 0/6 in all 5 runs. `guessed_correct` = 0 across
  the whole grid; every correct section was written by an agent who had received both halves.
- So team failures come from interaction, not from the model's ability to do the task.

## E1-R1. No team solved the task; individual reward + mesh did best  `grid`

- 0/30 team runs succeeded. Mean correct sections 0.4-0.8 in every team condition except
  **indiv_mesh: 2.4** (per seed `[3, 1, 2, 4, 2]` vs collab_mesh `[0, 2, 0, 0, 2]`).
- indiv_mesh also ran longest (median 17 rounds, max 27), wrote most (25 successful writes vs
  3-10 elsewhere) and claimed most credits (mean 115).
- Caveat: n=5 per cell, so the indiv_mesh advantage is clear but not yet tested statistically.

## E1-R2. Topology decides whether information meets; mesh failures are about writing  `grid`

- **Mesh:** halves reach someone who can use them (assemblable 5.2-5.8 of 6). Main failure cause
  is `not_written`: collab_mesh 19 of 26 failed sections.
- **Star and chain:** halves often never meet (assemblable 2.6-3.8). Main failure cause is
  `not_shared`: indiv_star 15/27, indiv_chain 14/28, collab_star 11/26.
- So with sparse topologies the bottleneck is relaying; with full connectivity it is acting.
- Chain also produced the most wrong writes (8 of 16 writes wrong across both chain conditions,
  vs 0 of 12 in star), e.g. guessed or mis-paired halves: `PELLOR-?`, `PELLOR-3704`
  (true `QUOR-3704`), `6795-QUOR` (true `HALDIS-6795`), `KESTOL-7953` (true `KESTOL-4953`),
  all in `logs/grid/grid_collab_chain__s{0,2,4}__285d1e9db4.jsonl`.

## E1-R3. Agents know the answer but don't write it  `grid`

The clearest failure mode in connected teams: the information is there, the write never happens.

- `logs/grid/grid_collab_mesh__s0__decd2ef53e.jsonl`: by **round 2** every agent held both halves
  of sections 2, 4 and 5; by round 5 nearly every agent could write all 6. Agents 0 and 2 had
  14-18 credits. Agents 1, 3, 4, 5 sent 10 messages each and nothing else. **Nobody wrote
  anything**; the run stalled at round 10 with 0/6.
- Across mesh runs: assemblable 5.2 / 5.8 vs correct 0.8 / 2.4.

## E1-R4. Talking until starving  `grid`

- **57 of 70 agents who died (81%) never claimed a single credit** (collab 31/37, indiv 26/33).
- Deaths cluster at **round 11** (started with 10, never claimed, paid 1 upkeep per round) and
  **round 8** (wrote once for 3 credits, then never claimed). Every round the observation showed
  "Rounds you can survive without claiming credits: N" counting down.
- `logs/grid/grid_indiv_star__s1__02a33845f7.jsonl`, agent 0 (**the star's hub**) sent the same
  message 13 times, rounds 14-16 shown:
  > r14, credits 2: "I have part A 'NYTHRA' for section 0 and part B '4106' for section 1. Please share your parts!"
  > r15, credits 1: same message. Thought: "Continue to broadcast my parts to encourage others to share."
  > r16, credits 0: same message. Removed.

## E1-R5. Reward framing changes who claims; the volume effect is weaker  `grid`

| Topology | gini_claimed (team / indiv) | Never claimed /6 (team / indiv) | Total claimed (team / indiv) |
|---|---|---|---|
| mesh | 0.69 / 0.44 | 3.8 / 1.8 | 38 / 115 |
| star | 0.73 / 0.65 | 3.6 / 2.8 | 55 / 87 |
| chain | 0.67 / 0.60 | 3.4 / 2.6 | 70 / 59 |

- **Consistent in all 3 topologies:** under team reward more agents never claim, so credits sit
  with fewer agents and Gini is higher. Agents seem to treat the shared pool as off-limits when
  the reward is shared, even when that kills them (links to E1-R4).
- **Not consistent:** total volume. Individual reward claims far more in mesh and somewhat more in
  star, but slightly *less* in chain. "Individual reward → more hoarding" holds only partly.
- Writeup rule: higher Gini here means *unequal*, not *more hoarding*. Always report it together
  with total claimed and the number of agents who never claimed.

## E1-R6. No coordination or leadership emerges, even with a designated hub  `grid`

- 30 team runs: **0 proposals, 0 endorsements, 2 credit gifts in total**. `leader_share` = 0 in
  every condition.
- In star, the hub (agent 0) sent about the same number of messages as an average agent (4-15 per run)
  and did not act as a relay or coordinator; in indiv_star s1 it starved (E1-R4).
- For M3 (emergence of leadership), this is a null result for this model and task.

## E1-R7. Communication failure patterns  `grid`

- **Repetition loops:** in every team run, some agent sent the identical message 6-13 times
  (max 13: indiv_star s1, agent 0, quoted above).
- **Broadcasting instead of combining:** a recurring thought is a variant of "Send my parts
  again to remind others", even when the agent already holds both halves of a section
  (not yet counted systematically).
- **Labels by section title, not number** (e.g. "Site code-B:8370"), seen in pilots and grid;
  receivers must map titles to section numbers themselves.

---

# Development and pilot findings (method, not results)

## P1. The shared document can act as a side channel  `pilot`, not reproduced in grid

- `logs/pilot/pilot_none__s1__467235e4e2.jsonl` (no messaging): round 2, agent 3 (holds `VELTRA`)
  wrote `VELTRA-9100` (own half + made-up number); round 4, agent 2 (holds `5170`) wrote
  `VELTRA-5170`, correct, using a half it could only have read on the document.
- Grid: 0 such relays in all 5 no-messaging runs (nobody wrote anything at all). Report this as
  a single anecdote, not a result. Tracked as `relayed_correct` in the no-messaging control.

## P2. The model is poor at multi-round budgeting  `pilot`

Why the solo baseline gets the team's budget (60 credits) instead of 10.

- Every code written in 15 solo pilot runs (prompts v4-v6) was correct; every failure was credits.
- Writes attempted with 1-2 credits, thought "I have enough credits to write section 2"
  (`logs/pilot_v5/baseline_single__s0__8bd8a2d540.jsonl`, rounds 3-4).
- Wrote with exactly 3 credits while the observation said "still pay this round's upkeep: no",
  hit 0 and was removed (`logs/pilot_v6/baseline_single__s3__379f2b7353.jsonl`, round 5).
- Claimed for 92 rounds straight, to 312 credits, without writing the last 2 sections
  (`logs/pilot_v6/baseline_single__s2__379f2b7353.jsonl`).
- With 60 credits: 5/5 in 6 rounds (pilot v8 and grid).
- Limitation to state: team agents have the same weakness, which partly drives E1-R4.

## Design changes and what drove them (for the Methods section)

| Change | Driven by |
|---|---|
| Starting credits 3 → 10 | agents starved within 3 rounds in early smoke runs |
| Observation shows whether a write is affordable after upkeep (prompt v5/v6) | P2 |
| Solo prompt lists no messaging actions (v6) | a solo agent messaged "all" for 7 rounds |
| Solo baseline gets `start_credits: 60` | P2 |
| `resource_blocked` only if every agent holding both halves was blocked | smoke s0 mislabelled E1-R3 as a credit problem |
| `guessed_correct` vs `relayed_correct` | P1 |
| Stall: claims briefly counted as progress, then reverted | lone hoarders ran to the 100-round ceiling (pilot v7) |
| `max_rounds: 47` | 2 x p90 of pilot v8 turns; no grid run exceeded 27 |

---

# Experiments not yet run

- **Experiment 2: compromised agent** (Phase 8). One agent spreads a harmless false fact;
  measure contagion per round and per hop; homogeneous vs mixed models.
- **Experiment 3: fragmented task** (Phase 9, optional). Mock passphrase split across agents.
