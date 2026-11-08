"""Phase 1 schemas: experiment config, agent actions, and log records.

Everything else in the project reads or writes these. Treat them as frozen.
"""
from __future__ import annotations

import hashlib
import json
import time
from enum import Enum
from typing import Annotated, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, TypeAdapter, ValidationError, model_validator

from swarm import settings


# ---------------------------------------------------------------------------
# Experiment config (the E's)
# ---------------------------------------------------------------------------
class Topology(str, Enum):
    mesh = "mesh"      # everyone talks to everyone
    star = "star"      # agent 0 is the hub
    chain = "chain"    # 0-1-2-3-4-5
    none = "none"      # no communication (control)


class RewardMode(str, Enum):
    collaborative = "collaborative"
    individual = "individual"


class TaskMode(str, Enum):
    normal = "normal"
    fragmented = "fragmented"


class ExperimentConfig(BaseModel):
    name: str
    topology: Topology = Topology.mesh
    reward_mode: RewardMode = RewardMode.collaborative
    compromised_agents: list[int] = Field(default_factory=list)
    task_mode: TaskMode = TaskMode.normal
    model_mix: Optional[list[str]] = None  # one model id per agent; None = all MODEL_NAME
    n_agents: int = settings.N_AGENTS
    max_rounds: int = settings.MAX_ROUNDS_CEILING
    stall_rounds: int = settings.STALL_ROUNDS
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    temperature: float = settings.TEMPERATURE

    @model_validator(mode="after")
    def _check(self):
        bad = [a for a in self.compromised_agents if not 0 <= a < self.n_agents]
        if bad:
            raise ValueError(f"compromised_agents out of range: {bad}")
        if self.model_mix is not None and len(self.model_mix) != self.n_agents:
            raise ValueError("model_mix must have exactly n_agents entries")
        return self

    def model_for(self, agent_id: int) -> str:
        return self.model_mix[agent_id] if self.model_mix else settings.MODEL_NAME

    def config_hash(self) -> str:
        """Hash of everything except seeds, so all seeds of a condition share it."""
        d = self.model_dump(mode="json", exclude={"seeds"})
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:10]

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))


# ---------------------------------------------------------------------------
# Actions: the ONLY things an agent can do. Discriminated on "type".
# ---------------------------------------------------------------------------
class ClaimResource(BaseModel):
    type: Literal["claim_resource"]
    amount: int = Field(ge=1, le=settings.MAX_CLAIM)


class GiveResource(BaseModel):
    type: Literal["give_resource"]
    to: int
    amount: int = Field(ge=1)


class SendMessage(BaseModel):
    type: Literal["send_message"]
    to: Union[int, Literal["all"]]  # "all" = all neighbours in the topology
    text: str = Field(min_length=1, max_length=500)


class WriteSection(BaseModel):
    type: Literal["write_section"]
    section: int = Field(ge=0, lt=settings.N_SECTIONS)
    content: str = Field(min_length=1, max_length=200)


class Propose(BaseModel):
    type: Literal["propose"]
    plan: str = Field(min_length=1, max_length=500)


class Endorse(BaseModel):
    type: Literal["endorse"]
    proposal_id: str


class Pass(BaseModel):
    type: Literal["pass"]


Action = Annotated[
    Union[ClaimResource, GiveResource, SendMessage, WriteSection, Propose, Endorse, Pass],
    Field(discriminator="type"),
]


class AgentTurn(BaseModel):
    """What the model must return each round."""
    thought: str = ""
    note: Optional[str] = None  # optional rewrite of the agent's private note
    action: Action


_turn_adapter = TypeAdapter(AgentTurn)


def parse_turn(text: str) -> tuple[Optional[AgentTurn], Optional[str]]:
    """Parse raw model output. Returns (turn, None) or (None, error_message)."""
    text = (text or "").strip()
    # tolerate ```json fences
    if text.startswith("```"):
        text = text.strip("`")
        text = text[4:] if text.lower().startswith("json") else text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None, "no_json_object"
    try:
        return _turn_adapter.validate_json(text[start : end + 1]), None
    except ValidationError as e:
        return None, f"schema_error: {e.errors()[0]['msg']}"


# ---------------------------------------------------------------------------
# Log records. One JSONL file per run; every line has an "event" field.
# ---------------------------------------------------------------------------
class Outcome(str, Enum):
    success = "success"
    all_dead = "all_dead"
    stalled = "stalled"
    timeout = "timeout"


class RunStart(BaseModel):
    event: Literal["run_start"] = "run_start"
    run_id: str
    config_hash: str
    seed: int
    config: dict
    edges: list[tuple[int, int]]      # topology, so metrics can compute centrality
    facts: list[str]                  # ground truth, section i -> fact
    fact_holder: list[int]            # section i -> agent who knows it
    ts: float = Field(default_factory=time.time)


class TurnLog(BaseModel):
    event: Literal["turn"] = "turn"
    run_id: str
    round: int
    agent_id: int
    model: str
    observation: dict                 # exactly what the agent was shown
    raw_output: str
    thought: str = ""
    note: Optional[str] = None
    action: dict                      # parsed action (fallback {"type": "pass"})
    valid_json: bool
    retries: int = 0
    parse_error: Optional[str] = None
    result: dict                      # {"ok": bool, "detail": str, ...}
    credits_after: int
    ts: float = Field(default_factory=time.time)


class RoundEnd(BaseModel):
    event: Literal["round_end"] = "round_end"
    run_id: str
    round: int
    pool: int
    credits: list[int]                # per agent
    alive: list[bool]
    document: list[Optional[str]]     # section i -> current content
    rewards: list[float]              # per agent, under the run's reward_mode
    ts: float = Field(default_factory=time.time)


class RunEnd(BaseModel):
    event: Literal["run_end"] = "run_end"
    run_id: str
    outcome: Outcome
    turns: int                        # rounds completed
    facts_correct: int
    deaths: int
    invalid_json_count: int
    ts: float = Field(default_factory=time.time)


LogRecord = Annotated[
    Union[RunStart, TurnLog, RoundEnd, RunEnd], Field(discriminator="event")
]
_record_adapter = TypeAdapter(LogRecord)


def parse_record(line: str):
    return _record_adapter.validate_json(line)