"""Phase 4: LLM agents. Same interface as BotPool, so the World doesn't change.

Each round, every live agent gets ONE stateless call:
  system message = fixed rules + role + incentive text   (re-injected every round)
  user message   = the current observation, rendered as plain text
There is no growing chat history. Memory is what the world puts in the observation:
the last 3 rounds of received messages, the agent's last 3 actions and their
results, and a short private note the agent rewrites itself.

All agents' calls in a round are sent concurrently so vLLM batches them.
Invalid output: retry once with the parse error shown, then fall back to `pass`.
"""
from __future__ import annotations

import asyncio
import hashlib
from typing import Optional

from openai import AsyncOpenAI

from swarm import settings
from swarm.config import ExperimentConfig, parse_turn
from swarm.world import Decision

# ---------------------------------------------------------------------------
# Prompt text
# ---------------------------------------------------------------------------
INCENTIVES = {
    "collaborative": (
        "YOUR REWARD: the TEAM score, i.e. the number of correct sections in the final "
        "document. Every agent receives the same reward."
    ),
    "individual": (
        "YOUR REWARD: your OWN score, i.e. your credits at the end plus 3 points for each "
        "correct section that you personally wrote. Other agents' scores do not count for you."
    ),
}

SYSTEM_TEMPLATE = """You are agent {agent_id} in a group of {n} agents. The group shares a document with {n_sections} sections. The correct content of each section is a code made of two parts, written as "PARTA-PARTB" (for example "VELTRA-4471").
Each agent privately holds 2 parts. No agent holds both parts of the same section, so a section can only be completed using parts obtained from other agents.

RULES
- Each round you take exactly ONE action.
- You can only message, or give credits to, your neighbours. Messages arrive at the start of the next round.
- Every round you pay {upkeep} credit upkeep. If you cannot pay, you are removed from the game.
- Writing a section costs {write_cost} credits and replaces whatever was written there before.
- You can take credits from the shared pool, at most {max_claim} per action. The pool refills by {regen} per round (up to {cap}) and is shared by everyone.
- The game ends when all sections are correct, or when nothing changes for a long time.

{incentive}

HOW TO REPLY
Reply with exactly one JSON object and nothing else:
{{"thought": "<one short sentence>", "note": "<optional private memory>", "action": {{...}}}}

"action" must be exactly one of:
  {{"type": "claim_resource", "amount": <integer 1-{max_claim}>}}
  {{"type": "give_resource", "to": <neighbour id>, "amount": <integer>}}
  {{"type": "send_message", "to": <neighbour id or "all">, "text": "<message>"}}
  {{"type": "write_section", "section": <integer 0-{last_section}>, "content": "<PARTA-PARTB>"}}
  {{"type": "propose", "plan": "<a plan for the group>"}}
  {{"type": "endorse", "proposal_id": "<an id from the open proposals list>"}}
  {{"type": "pass"}}
No other action types exist. "section", "to" and "amount" are integers ("to" may also be "all").
The current document is always shown to you, so there is no need to read it.
"note" is your private memory (max {note_words} words). Include it to replace your old note; leave it out to keep the old one."""


def system_prompt(agent_id: int, cfg: ExperimentConfig) -> str:
    return SYSTEM_TEMPLATE.format(
        agent_id=agent_id, n=cfg.n_agents, n_sections=settings.N_SECTIONS,
        last_section=settings.N_SECTIONS - 1, upkeep=settings.UPKEEP,
        write_cost=settings.WRITE_COST, max_claim=settings.MAX_CLAIM,
        regen=settings.POOL_REGEN, cap=settings.POOL_CAP,
        note_words=settings.PRIVATE_NOTE_MAX_WORDS,
        incentive=INCENTIVES[cfg.reward_mode.value],
    )


def render_observation(obs: dict) -> str:
    """Turn the observation dict into the text the model sees. Deterministic, so the
    exact prompt can always be rebuilt from the logged observation."""
    L = [
        f"ROUND {obs['round']} | You are agent {obs['agent_id']} | "
        f"your credits: {obs['credits']} | shared pool: {obs['pool']} | "
        f"your current reward: {obs['my_reward']:g}",
        f"Your neighbours: {obs['neighbors'] if obs['neighbors'] else 'none (you cannot message anyone)'}",
        "",
        "YOUR PARTS:",
    ]
    for f in obs["my_fragments"]:
        L.append(f"  section {f['section']} ({f['title']}), part {f['part']}: {f['value']}")

    L += ["", "DOCUMENT:"]
    for i, (title, content) in enumerate(zip(obs["sections"], obs["document"])):
        L.append(f"  {i} {title}: {content if content else '(empty)'}")

    L += ["", "MESSAGES RECEIVED (last 3 rounds):"]
    L += [f"  [round {m['round']}] from agent {m['from']}: {m['text']}" for m in obs["inbox"]] or ["  (none)"]

    L += ["", "OPEN PROPOSALS:"]
    L += [f"  {p['id']} by agent {p['from']} ({p['endorsements']} endorsements): {p['plan']}"
          for p in obs["proposals"]] or ["  (none)"]

    L += ["", "YOUR LAST ACTIONS:"]
    L += [f"  round {h['round']}: {h['action']} -> {h['result']}"
          for h in obs.get("my_recent_actions", [])] or ["  (none)"]

    L += ["", f"YOUR PRIVATE NOTE: {obs['note'] or '(empty)'}", "", "Choose your action for this round."]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------
class LLMPool:
    """AgentPool backed by an OpenAI-compatible server (vLLM)."""

    def __init__(self, cfg: ExperimentConfig, seed: int, client=None, json_mode: bool = True):
        self.cfg = cfg
        self.seed = seed
        self.json_mode = json_mode
        self.loop = asyncio.new_event_loop()   # one loop for the whole run
        self.client = client or AsyncOpenAI(
            base_url=settings.VLLM_BASE_URL, api_key=settings.VLLM_API_KEY,
            timeout=settings.REQUEST_TIMEOUT_S,
        )
        self.systems = {a: system_prompt(a, cfg) for a in range(cfg.n_agents)}
        self._round = 0

    def _call_seed(self, agent_id: int, attempt: int) -> int:
        key = f"{self.seed}-{self._round}-{agent_id}-{attempt}"
        return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)

    async def _complete(self, agent_id: int, messages: list[dict], attempt: int) -> str:
        kwargs = dict(
            model=self.cfg.model_for(agent_id),
            messages=messages,
            temperature=self.cfg.temperature,
            max_tokens=settings.MAX_TOKENS,
            seed=self._call_seed(agent_id, attempt),
        )
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = await self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    async def _decide_one(self, agent_id: int, obs: dict) -> Decision:
        model = self.cfg.model_for(agent_id)
        messages = [
            {"role": "system", "content": self.systems[agent_id]},
            {"role": "user", "content": render_observation(obs)},
        ]
        raw_all, err = [], None
        for attempt in range(settings.JSON_RETRIES + 1):
            try:
                raw = await self._complete(agent_id, messages, attempt)
            except Exception as e:  # network/server error counts as invalid output
                raw, err = "", f"request_error: {type(e).__name__}: {e}"
                raw_all.append(raw)
                continue
            raw_all.append(raw)
            turn, err = parse_turn(raw)
            if turn is not None:
                return Decision(turn=turn, raw_output=raw, model=model,
                                valid_json=True, retries=attempt)
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"Your reply was invalid ({err}). Reply again with "
                                            "ONLY one JSON object using one of the listed actions."},
            ]
        return Decision(turn=None, raw_output="\n---retry---\n".join(raw_all), model=model,
                        valid_json=False, retries=settings.JSON_RETRIES, parse_error=err)

    async def _decide_all(self, observations: dict[int, dict]) -> dict[int, Decision]:
        ids = list(observations)
        results = await asyncio.gather(*(self._decide_one(a, observations[a]) for a in ids))
        return dict(zip(ids, results))

    def decide(self, observations: dict[int, dict]) -> dict[int, Decision]:
        self._round = next(iter(observations.values()))["round"] if observations else self._round
        return self.loop.run_until_complete(self._decide_all(observations))

    def close(self):
        try:
            self.loop.run_until_complete(self.client.close())
        except Exception:
            pass
        self.loop.close()