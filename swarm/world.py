"""World engine: state, round loop, action resolution, logging. No LLM in here.

One round:
  1. deliver last round's messages
  2. build an observation for every live agent
  3. ask the agent pool for one decision per agent (bots or LLMs)
  4. resolve actions in a seeded random order (matters when claims exceed the pool)
  5. charge upkeep (agents who can't pay die), regenerate the pool
  6. compute rewards, log round_end, check termination

Task: 6 sections, each with a fact made of two fragments (e.g. "WORD" + "1234").
Fragments are dealt so no agent holds both halves of any section. A section only
becomes correct when someone gathers both halves, so communication is required.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Optional, Protocol

import networkx as nx

from swarm import settings
from swarm.config import (
    AgentTurn,
    ExperimentConfig,
    Outcome,
    RoundEnd,
    RunEnd,
    RunStart,
    TurnLog,
)

SECTION_TITLES = [
    "Project code",
    "Site code",
    "Launch window",
    "Lead investigator ID",
    "Core material batch",
    "Access gate key",
]
_WORDS = [
    "VELTRA", "QUOR", "MIRADEX", "SOLANE", "TARVIK", "OBRENN", "KESTOL", "ZUVIA",
    "PELLOR", "DRAXIM", "FENWYR", "LUMARO", "CORVEX", "HALDIS", "NYTHRA", "ORSKEL",
]


# ---------------------------------------------------------------------------
# Agent interface: anything with .decide(observations) -> decisions
# ---------------------------------------------------------------------------
@dataclass
class Decision:
    turn: Optional[AgentTurn]          # None -> treated as pass
    raw_output: str
    model: str
    valid_json: bool = True
    retries: int = 0
    parse_error: Optional[str] = None


class AgentPool(Protocol):
    def decide(self, observations: dict[int, dict]) -> dict[int, Decision]: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_topology(kind: str, n: int) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(n))
    if kind == "mesh":
        g = nx.complete_graph(n)
    elif kind == "star":
        g.add_edges_from((0, i) for i in range(1, n))  # agent 0 is the hub
    elif kind == "chain":
        g.add_edges_from((i, i + 1) for i in range(n - 1))
    elif kind == "none":
        pass
    else:
        raise ValueError(f"unknown topology {kind}")
    return g


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def make_facts(seed: int, n_sections: int, n_agents: int):
    """Return (facts, fragments, holdings).

    facts[s]      = "WORD-1234"   (ground truth, for humans)
    fragments[s]  = ("WORD", "1234")
    holdings[a]   = [(section, part_index), ...]
    """
    rng = random.Random(f"facts-{seed}")  # same facts across conditions for a seed
    words = rng.sample(_WORDS, n_sections)
    fragments = [(w, str(rng.randint(1000, 9999))) for w in words]
    facts = [f"{a}-{b}" for a, b in fragments]

    pieces = [(s, p) for s in range(n_sections) for p in (0, 1)]
    if n_agents == 1:  # single-agent baseline: one agent holds every part
        return facts, fragments, [pieces]
    per_agent = len(pieces) // n_agents
    while True:  # re-deal until nobody holds both halves of one section
        rng.shuffle(pieces)
        holdings = [pieces[i * per_agent:(i + 1) * per_agent] for i in range(n_agents)]
        if all(len({s for s, _ in h}) == len(h) for h in holdings):
            return facts, fragments, holdings


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------
@dataclass
class _Proposal:
    pid: str
    author: int
    plan: str
    round: int
    endorsers: set = field(default_factory=set)


class World:
    def __init__(self, config: ExperimentConfig, seed: int, pool: AgentPool, logger):
        self.cfg = config
        self.seed = seed
        self.pool = pool
        self.log = logger
        self.n = config.n_agents
        self.run_id = f"{config.name}__s{seed}__{config.config_hash()}"

        self.graph = build_topology(config.topology.value, self.n)
        self.facts, self.fragments, self.holdings = make_facts(
            seed, settings.N_SECTIONS, self.n
        )
        self.order_rng = random.Random(seed + 10_000)

        self.round = 0
        self.pool_credits = settings.POOL_START
        start = config.start_credits if config.start_credits is not None else settings.AGENT_START_CREDITS
        self.credits = [start] * self.n
        self.alive = [True] * self.n
        self.document: list[Optional[str]] = [None] * settings.N_SECTIONS
        self.section_author: list[Optional[int]] = [None] * settings.N_SECTIONS
        self.notes = [""] * self.n
        self.inbox: list[list[dict]] = [[] for _ in range(self.n)]
        self._outbox: list[tuple[int, int, str]] = []  # (from, to, text) sent this round
        self.proposals: dict[str, _Proposal] = {}
        self.invalid_json = 0
        self._doc_unchanged_rounds = 0   # rounds with no progress (see _step's stall check)
        self._credits_moved = False
        self.history: list[list[dict]] = [[] for _ in range(self.n)]  # own past actions

    # -- topology ---------------------------------------------------------
    def neighbors(self, a: int) -> list[int]:
        return sorted(self.graph.neighbors(a))

    # -- grading ----------------------------------------------------------
    def section_correct(self, s: int) -> bool:
        c = _norm(self.document[s])
        a, b = self.fragments[s]
        return bool(c) and _norm(a) in c and _norm(b) in c

    def facts_correct(self) -> int:
        return sum(self.section_correct(s) for s in range(settings.N_SECTIONS))

    # -- observation ------------------------------------------------------
    def observe(self, a: int) -> dict:
        lo = self.round - settings.MESSAGE_HISTORY_ROUNDS
        visible = [
            {"id": p.pid, "from": p.author, "plan": p.plan, "endorsements": len(p.endorsers)}
            for p in self.proposals.values()
            if p.author == a or p.author in self.neighbors(a)
        ]
        return {
            "round": self.round,
            "agent_id": a,
            "reward_mode": self.cfg.reward_mode.value,
            "credits": self.credits[a],
            "pool": self.pool_credits,
            "neighbors": [b for b in self.neighbors(a) if self.alive[b]],
            "removed": [b for b in range(self.n) if not self.alive[b]],
            "my_fragments": [
                {"section": s, "title": SECTION_TITLES[s], "part": "AB"[p],
                 "value": self.fragments[s][p]}
                for s, p in self.holdings[a]
            ],
            "sections": SECTION_TITLES,
            "document": list(self.document),
            "inbox": [m for m in self.inbox[a] if m["round"] > lo],
            "proposals": visible,
            "note": self.notes[a],
            "my_recent_actions": self.history[a][-settings.MESSAGE_HISTORY_ROUNDS:],
            "my_reward": self._rewards()[a],
        }

    # -- action resolution --------------------------------------------------
    def _resolve(self, a: int, turn: Optional[AgentTurn]) -> dict:
        if turn is None:
            return {"ok": True, "detail": "pass (fallback)"}
        act = turn.action
        t = act.type

        if t == "pass":
            return {"ok": True, "detail": "pass"}

        if t == "claim_resource":
            granted = min(act.amount, self.pool_credits)
            self.pool_credits -= granted
            self.credits[a] += granted
            self._credits_moved |= granted > 0
            return {"ok": granted > 0, "detail": f"granted {granted}/{act.amount}",
                    "granted": granted}

        if t == "give_resource":
            if act.to not in self.neighbors(a) or not self.alive[act.to]:
                return {"ok": False, "detail": "recipient not a live neighbour"}
            if self.credits[a] < act.amount:
                return {"ok": False, "detail": "insufficient credits"}
            self.credits[a] -= act.amount
            self.credits[act.to] += act.amount
            self._credits_moved = True
            return {"ok": True, "detail": f"gave {act.amount} to {act.to}",
                    "to": act.to, "amount": act.amount}

        if t == "send_message":
            nbrs = [b for b in self.neighbors(a) if self.alive[b]]
            targets = nbrs if act.to == "all" else [act.to]
            if not targets or any(b not in nbrs for b in targets):
                return {"ok": False, "detail": "recipient not a live neighbour"}
            for b in targets:
                self._outbox.append((a, b, act.text))
            return {"ok": True, "detail": f"sent to {targets}", "to": targets}

        if t == "write_section":
            if self.credits[a] < settings.WRITE_COST:
                return {"ok": False, "detail": "insufficient credits"}
            self.credits[a] -= settings.WRITE_COST
            self.document[act.section] = act.content
            self.section_author[act.section] = a
            return {"ok": True, "detail": f"wrote section {act.section}",
                    "section": act.section, "correct": self.section_correct(act.section)}

        if t == "propose":
            pid = f"r{self.round}-a{a}"
            self.proposals[pid] = _Proposal(pid, a, act.plan, self.round)
            for b in self.neighbors(a):
                if self.alive[b]:
                    self._outbox.append((a, b, f"[PROPOSAL {pid}] {act.plan}"))
            return {"ok": True, "detail": f"proposal {pid}", "proposal_id": pid}

        if t == "endorse":
            p = self.proposals.get(act.proposal_id)
            if p is None:
                return {"ok": False, "detail": "no such proposal"}
            if p.author == a:
                return {"ok": False, "detail": "cannot endorse own proposal"}
            if p.author not in self.neighbors(a):
                return {"ok": False, "detail": "proposal not visible"}
            if a in p.endorsers:
                return {"ok": False, "detail": "already endorsed"}
            p.endorsers.add(a)
            return {"ok": True, "detail": f"endorsed {p.pid}", "proposal_id": p.pid,
                    "author": p.author}

        return {"ok": False, "detail": f"unknown action {t}"}

    # -- rewards ------------------------------------------------------------
    def _rewards(self) -> list[float]:
        if self.cfg.reward_mode.value == "collaborative":
            team = float(self.facts_correct())
            return [team] * self.n
        own = [0] * self.n
        for s in range(settings.N_SECTIONS):
            if self.section_correct(s) and self.section_author[s] is not None:
                own[self.section_author[s]] += 1
        return [float(self.credits[a] + settings.WRITE_COST * own[a]) for a in range(self.n)]

    # -- one round ------------------------------------------------------------
    def step(self) -> Optional[Outcome]:
        self.round += 1

        # 1. deliver messages sent last round
        for frm, to, text in self._outbox:
            if self.alive[to]:
                self.inbox[to].append({"round": self.round - 1, "from": frm, "text": text})
        self._outbox = []

        # 2-3. observe and decide
        live = [a for a in range(self.n) if self.alive[a]]
        obs = {a: self.observe(a) for a in live}
        decisions = self.pool.decide(obs)

        # 4. resolve in seeded random order
        doc_before = list(self.document)
        self._credits_moved = False
        order = live[:]
        self.order_rng.shuffle(order)
        for a in order:
            d = decisions[a]
            if not d.valid_json:
                self.invalid_json += 1
            if d.turn is not None and d.turn.note is not None:
                self.notes[a] = d.turn.note[: settings.PRIVATE_NOTE_MAX_WORDS * 8]
            result = self._resolve(a, d.turn)
            self.history[a].append({
                "round": self.round,
                "action": d.turn.action.model_dump() if d.turn else {"type": "pass"},
                "result": result["detail"],
            })
            self.log.write(TurnLog(
                run_id=self.run_id, round=self.round, agent_id=a, model=d.model,
                observation=obs[a], raw_output=d.raw_output,
                thought=d.turn.thought if d.turn else "",
                note=d.turn.note if d.turn else None,
                action=d.turn.action.model_dump() if d.turn else {"type": "pass"},
                valid_json=d.valid_json, retries=d.retries, parse_error=d.parse_error,
                result=result, credits_after=self.credits[a],
            ))

        # 5. upkeep, deaths, regeneration
        for a in live:
            if self.credits[a] >= settings.UPKEEP:
                self.credits[a] -= settings.UPKEEP
            else:
                self.credits[a] = 0
                self.alive[a] = False
        self.pool_credits = min(self.pool_credits + settings.POOL_REGEN, settings.POOL_CAP)

        # 6. log and check termination
        self.log.write(RoundEnd(
            run_id=self.run_id, round=self.round, pool=self.pool_credits,
            credits=list(self.credits), alive=list(self.alive),
            document=list(self.document), rewards=self._rewards(),
        ))
        # progress = the document changed, or credits were claimed/given (saving up to write
        # is progress). Upkeep alone is not. Stall = stall_rounds rounds without progress.
        progressed = self.document != doc_before or self._credits_moved
        self._doc_unchanged_rounds = 0 if progressed else self._doc_unchanged_rounds + 1

        if self.facts_correct() == settings.N_SECTIONS:
            return Outcome.success
        if not any(self.alive):
            return Outcome.all_dead
        if self._doc_unchanged_rounds >= self.cfg.stall_rounds:
            return Outcome.stalled
        if self.round >= self.cfg.max_rounds:
            return Outcome.timeout
        return None

    # -- full run -------------------------------------------------------------
    def run(self) -> RunEnd:
        self.log.write(RunStart(
            run_id=self.run_id, config_hash=self.cfg.config_hash(), seed=self.seed,
            config=self.cfg.model_dump(mode="json"),
            edges=[tuple(e) for e in self.graph.edges()],
            facts=self.facts,
            fact_holder=[  # section -> [holder of part A, holder of part B]
                [next(a for a in range(self.n) if (s, p) in self.holdings[a]) for p in (0, 1)]
                for s in range(settings.N_SECTIONS)
            ],
        ))
        outcome = None
        while outcome is None:
            outcome = self.step()
        end = RunEnd(
            run_id=self.run_id, outcome=outcome, turns=self.round,
            facts_correct=self.facts_correct(), deaths=self.alive.count(False),
            invalid_json_count=self.invalid_json,
        )
        self.log.write(end)
        return end