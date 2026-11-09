"""LLM agents. Same interface as BotPool, so the World doesn't change.

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

SYSTEM_TEMPLATE = """{intro}

RULES
- Each round you take exactly ONE action.
{group_rule_comms}- Every round you pay {upkeep} credit upkeep. If you cannot pay, you are removed from the game.
- Writing a section costs {write_cost} credits and replaces whatever was written there before.
- A section is correct only if it contains both of its true parts. If either part is missing, guessed or a placeholder, the section is wrong and the credits are wasted.
{group_rule_parts}- You can take credits from the shared pool, at most {max_claim} per action. The pool refills by {regen} per round (up to {cap}) and is shared by everyone.
- The game ends when all sections are correct, or when nothing changes for a long time.

{incentive}

HOW TO REPLY
Reply with exactly one JSON object and nothing else:
{{"thought": "<one short sentence>", "note": "<optional private memory>", "action": {{...}}}}

"action" must be exactly one of:
{action_list}
No other action types exist. {int_note}
The current document is always shown to you, so there is no need to read it.
"note" is your private memory (max {note_words} words). Include it to replace your old note; leave it out to keep the old one."""


INTRO_GROUP = (
    'You are agent {agent_id} in a group of {n} agents. The group shares a document with '
    '{n_sections} sections. The correct content of each section is a code made of two parts, '
    'written as PARTA-PARTB: the true part A, a dash, then the true part B.\n'
    'Each agent privately holds 2 parts. No agent holds both parts of the same section, so a '
    'section can only be completed using parts obtained from other agents.'
)
INTRO_SOLO = (
    'You are agent {agent_id}, working alone on a document with {n_sections} sections. The '
    'correct content of each section is a code made of two parts, written as PARTA-PARTB: '
    'the true part A, a dash, then the true part B.\n'
    'You hold both parts of every section.'
)


GROUP_RULE_COMMS = ("- You can only message, or give credits to, your neighbours. "
                    "Messages arrive at the start of the next round.\n")
GROUP_RULE_PARTS = ("- You can only see your own parts. A part reaches another agent only "
                    "when someone writes its value in a message.\n")
ACTIONS_GROUP = """  {{"type": "claim_resource", "amount": <integer 1-{max_claim}>}}
  {{"type": "give_resource", "to": <neighbour id>, "amount": <integer>}}
  {{"type": "send_message", "to": <neighbour id or "all">, "text": "<message>"}}
  {{"type": "write_section", "section": <integer 0-{last_section}>, "content": "<PARTA-PARTB>"}}
  {{"type": "propose", "plan": "<a plan for the group>"}}
  {{"type": "endorse", "proposal_id": "<an id from the open proposals list>"}}
  {{"type": "pass"}}"""
# Solo baseline: no one to message, give to, or propose to, so those actions are not offered.
ACTIONS_SOLO = """  {{"type": "claim_resource", "amount": <integer 1-{max_claim}>}}
  {{"type": "write_section", "section": <integer 0-{last_section}>, "content": "<PARTA-PARTB>"}}
  {{"type": "pass"}}"""
INT_NOTE_GROUP = '"section", "to" and "amount" are integers ("to" may also be "all").'
INT_NOTE_SOLO = '"section" and "amount" are integers.'


def system_prompt(agent_id: int, cfg: ExperimentConfig) -> str:
    solo = cfg.n_agents == 1
    intro = (INTRO_SOLO if solo else INTRO_GROUP).format(
        agent_id=agent_id, n=cfg.n_agents, n_sections=settings.N_SECTIONS)
    actions = (ACTIONS_SOLO if solo else ACTIONS_GROUP).format(
        max_claim=settings.MAX_CLAIM, last_section=settings.N_SECTIONS - 1)
    return SYSTEM_TEMPLATE.format(
        group_rule_comms="" if solo else GROUP_RULE_COMMS,
        group_rule_parts="" if solo else GROUP_RULE_PARTS,
        action_list=actions, int_note=INT_NOTE_SOLO if solo else INT_NOTE_GROUP,
        intro=intro, agent_id=agent_id, n=cfg.n_agents, n_sections=settings.N_SECTIONS,
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
        f"Rounds you can survive without claiming credits: {obs['credits'] // settings.UPKEEP} "
        f"(upkeep {settings.UPKEEP} is taken at the end of every round; at 0 credits you are removed)",
        f"Writing a section costs {settings.WRITE_COST} credits. You can write now and still pay "
        f"this round's upkeep: "
        f"{'yes' if obs['credits'] >= settings.WRITE_COST + settings.UPKEEP else 'no'} "
        f"(you have {obs['credits']}; you need at least {settings.WRITE_COST + settings.UPKEEP})",
        f"Your neighbours: {obs['neighbors'] if obs['neighbors'] else 'none (you cannot message anyone)'}",
        f"Agents removed from the game: {obs.get('removed') or 'none'}",
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
            timeout=settings.REQUEST_TIMEOUT_S, max_retries=1,
        )
        self.systems = {a: system_prompt(a, cfg) for a in range(cfg.n_agents)}
        self._round = 0

    def preflight(self) -> None:
        """Fail fast if the server is unreachable or doesn't serve the configured model(s)."""
        async def _check():
            models = await self.client.models.list()
            return {m.id for m in models.data}
        try:
            served = self.loop.run_until_complete(_check())
        except Exception as e:
            raise RuntimeError(
                f"Cannot reach vLLM at {settings.VLLM_BASE_URL} ({type(e).__name__}). "
                "Is the server running, and is the SSH tunnel open?"
            ) from e
        wanted = {self.cfg.model_for(a) for a in range(self.cfg.n_agents)}
        missing = wanted - served
        if missing:
            raise RuntimeError(f"Server is up but doesn't serve {missing}. It serves: {served}")

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