"""Scripted bots: agents with KNOWN behaviour, used to test the engine and metrics.

If the metrics can't tell a hoarder from a sharer here, no LLM result is trustworthy.
Bots speak the same Action schema as LLM agents and plug into the same World.

Message format bots use for facts:  "FRAG <section> <A|B> <value>; FRAG ..."
"""
from __future__ import annotations

import json
import random
import re

from swarm import settings
from swarm.config import AgentTurn
from swarm.world import Decision

_FRAG_RE = re.compile(r"FRAG (\d+) ([AB]) ([A-Za-z0-9]+)")


def _turn(action: dict, thought: str = "") -> AgentTurn:
    return AgentTurn.model_validate({"thought": thought, "action": action})


class Bot:
    kind = "bot"

    def __init__(self, agent_id: int, seed: int = 0):
        self.id = agent_id
        self.rng = random.Random(seed * 1000 + agent_id)

    def act(self, obs: dict) -> AgentTurn:
        raise NotImplementedError

    def decide(self, obs: dict) -> Decision:
        turn = self.act(obs)
        return Decision(turn=turn, raw_output=turn.model_dump_json(), model=f"bot:{self.kind}")


class SharerBot(Bot):
    """Cooperative: broadcasts and relays fragments, claims only what it needs,
    writes sections once it has both halves, endorses visible proposals."""

    kind = "sharer"

    def __init__(self, agent_id, seed=0):
        super().__init__(agent_id, seed)
        self.known: dict[tuple[int, str], str] = {}   # (section, part) -> value
        self.sent: set[tuple[int, str]] = set()
        self.endorsed: set[str] = set()

    def _learn(self, obs):
        for f in obs["my_fragments"]:
            self.known[(f["section"], f["part"])] = f["value"]
        for m in obs["inbox"]:
            for s, p, v in _FRAG_RE.findall(m["text"]):
                self.known[(int(s), p)] = v

    def _writable(self, obs):
        mine = {f["section"] for f in obs["my_fragments"]}
        out = []
        for s in range(settings.N_SECTIONS):
            a, b = self.known.get((s, "A")), self.known.get((s, "B"))
            if a and b:
                doc = (obs["document"][s] or "").lower()
                if not (a.lower() in doc and b.lower() in doc):
                    out.append((s, f"{a}-{b}"))
        out.sort(key=lambda x: x[0] not in mine)  # own sections first
        return out

    def act(self, obs):
        self._learn(obs)
        need = settings.WRITE_COST + settings.UPKEEP

        if obs["credits"] <= settings.UPKEEP and obs["pool"] > 0:
            return _turn({"type": "claim_resource", "amount": 2}, "need upkeep")

        unsent = [k for k in self.known if k not in self.sent]
        if unsent and obs["neighbors"]:
            self.sent.update(unsent)
            text = "; ".join(f"FRAG {s} {p} {self.known[(s, p)]}" for s, p in sorted(unsent))
            return _turn({"type": "send_message", "to": "all", "text": text}, "share fragments")

        writable = self._writable(obs)
        if writable:
            if obs["credits"] >= need:
                s, content = writable[0]
                return _turn({"type": "write_section", "section": s, "content": content}, "write")
            if obs["pool"] > 0:
                # claim enough to still have `need` after this round's upkeep
                amt = min(settings.MAX_CLAIM, need + settings.UPKEEP - obs["credits"])
                return _turn({"type": "claim_resource", "amount": max(1, amt)}, "save for write")

        for p in obs["proposals"]:
            if p["from"] != self.id and p["id"] not in self.endorsed:
                self.endorsed.add(p["id"])
                return _turn({"type": "endorse", "proposal_id": p["id"]}, "endorse")

        return _turn({"type": "pass"}, "nothing to do")


class ProposerBot(SharerBot):
    """A sharer that also proposes plans every few rounds (for leadership tests)."""

    kind = "proposer"

    def act(self, obs):
        if obs["round"] % 4 == 1 and obs["neighbors"]:
            self._learn(obs)
            return _turn({"type": "propose", "plan": f"Plan from {self.id}: share all fragments, "
                                                       "then write own sections."}, "lead")
        return super().act(obs)


class HoarderBot(Bot):
    """Selfish: claims the maximum every round, never shares, never gives."""

    kind = "hoarder"

    def act(self, obs):
        if obs["pool"] > 0:
            return _turn({"type": "claim_resource", "amount": settings.MAX_CLAIM}, "take all")
        return _turn({"type": "pass"}, "pool empty")


class RandomBot(Bot):
    """Uniformly random valid action (baseline for low specialization)."""

    kind = "random"

    def act(self, obs):
        r = self.rng.random()
        nbrs = obs["neighbors"]
        if obs["credits"] <= settings.UPKEEP and obs["pool"] > 0:
            r = 0.0  
        if r < 0.2:
            return _turn({"type": "claim_resource", "amount": self.rng.randint(1, settings.MAX_CLAIM)})
        if r < 0.4 and nbrs:
            return _turn({"type": "send_message", "to": self.rng.choice(nbrs), "text": "hello"})
        if r < 0.55 and nbrs and obs["credits"] > 2:
            return _turn({"type": "give_resource", "to": self.rng.choice(nbrs), "amount": 1})
        if r < 0.7 and obs["credits"] >= settings.WRITE_COST:
            return _turn({"type": "write_section", "section": self.rng.randrange(settings.N_SECTIONS),
                          "content": "random text"})
        if r < 0.85 and nbrs:
            return _turn({"type": "propose", "plan": "random plan"})
        return _turn({"type": "pass"})


class SpecialistBot(Bot):
    """Always does the same kind of action (baseline for high specialization)."""

    kind = "specialist"

    def __init__(self, agent_id, seed=0, action="claim_resource"):
        super().__init__(agent_id, seed)
        self.action = action

    def act(self, obs):
        if self.action == "claim_resource":
            return _turn({"type": "claim_resource", "amount": 1})
        if self.action == "send_message" and obs["neighbors"]:
            return _turn({"type": "send_message", "to": "all", "text": "status ok"})
        return _turn({"type": "pass"})


class BrokenBot(Bot):
    """Emits invalid output, to test the invalid-JSON path."""

    kind = "broken"

    def decide(self, obs):
        return Decision(turn=None, raw_output="I think I will claim some credits!",
                        model="bot:broken", valid_json=False, retries=1,
                        parse_error="no_json_object")


BOT_TYPES = {
    "sharer": SharerBot, "proposer": ProposerBot, "hoarder": HoarderBot,
    "random": RandomBot, "specialist": SpecialistBot, "broken": BrokenBot,
}


class BotPool:
    """AgentPool made of bots. `kinds` is one bot name per agent."""

    def __init__(self, kinds: list[str], seed: int = 0):
        self.bots = {i: BOT_TYPES[k](i, seed) for i, k in enumerate(kinds)}

    def decide(self, observations: dict[int, dict]) -> dict[int, Decision]:
        return {a: self.bots[a].decide(o) for a, o in observations.items()}