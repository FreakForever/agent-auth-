"""Phase 2 checks: the engine behaves correctly with bots of known behaviour."""
import json

from swarm import settings
from swarm.agents.bots import BotPool
from swarm.config import ExperimentConfig, Outcome
from swarm.logger import JsonlLogger, NullLogger, read_log
from swarm.world import World, make_facts


def run(kinds, topology="mesh", seed=0, reward="collaborative", logger=None, **kw):
    cfg = ExperimentConfig(name="t", topology=topology, reward_mode=reward, **kw)
    w = World(cfg, seed, BotPool(kinds, seed), logger or NullLogger())
    return w, w.run()


def test_fragments_dealt_fairly():
    for seed in range(20):
        facts, frags, holdings = make_facts(seed, 6, 6)
        assert all(len(h) == 2 for h in holdings)
        assert all(len({s for s, _ in h}) == 2 for h in holdings)  # never both halves
        assert sorted(p for h in holdings for p in h) == [(s, p) for s in range(6) for p in (0, 1)]


def test_sharers_succeed_in_mesh():
    w, end = run(["sharer"] * 6)
    assert end.outcome == Outcome.success and end.facts_correct == 6 and end.deaths == 0


def test_sharers_succeed_in_chain_but_slower():
    _, mesh = run(["sharer"] * 6, "mesh")
    _, chain = run(["sharer"] * 6, "chain")
    assert chain.outcome == Outcome.success
    assert chain.turns > mesh.turns   # info must hop along the chain


def test_no_communication_cannot_succeed():
    _, end = run(["sharer"] * 6, "none")
    assert end.facts_correct == 0 and end.outcome != Outcome.success


def test_hoarders_never_finish():
    _, end = run(["hoarder"] * 6)
    assert end.facts_correct == 0 and end.outcome == Outcome.stalled


def test_hoarders_get_richer_than_sharers():
    w, _ = run(["hoarder"] * 3 + ["sharer"] * 3, max_rounds=20, stall_rounds=99)
    assert sum(w.credits[:3]) > 3 * sum(w.credits[3:])


def test_starvation_kills():
    # specialists claim 1/round == upkeep, so they survive; broken bots never act, so they starve
    w, end = run(["specialist"] * 6, max_rounds=20)
    # stall_rounds raised so the stall check (10) doesn't end the run before 10-credit agents die
    w2, end2 = run(["broken"] * 6, max_rounds=20, stall_rounds=99)
    assert end.deaths == 0
    assert end2.outcome == Outcome.all_dead and end2.deaths == 6
    assert end2.invalid_json_count > 0


def test_messages_arrive_next_round_only_to_neighbours():
    cfg = ExperimentConfig(name="t", topology="chain")
    w = World(cfg, 0, BotPool(["sharer"] * 6), NullLogger())
    w.step()   # round 1: everyone broadcasts
    assert all(not box for box in w.inbox)            # nothing delivered yet
    w.step()   # round 2: delivery happens at start
    assert {m["from"] for m in w.inbox[0]} == {1}      # agent 0 only hears agent 1
    assert {m["from"] for m in w.inbox[3]} == {2, 4}


def test_endorsements_follow_topology():
    w, _ = run(["proposer"] + ["sharer"] * 5, "star", max_rounds=12, stall_rounds=99)
    hub = [p for p in w.proposals.values() if p.author == 0]
    assert hub and max(len(p.endorsers) for p in hub) >= 3


def test_same_seed_same_log(tmp_path):
    def strip(path):
        out = []
        for r in read_log(path):
            d = r.model_dump(mode="json")
            d.pop("ts", None)
            out.append(json.dumps(d, sort_keys=True))
        return out

    for name in ("a", "b"):
        with JsonlLogger(str(tmp_path / f"{name}.jsonl")) as lg:
            run(["random"] * 3 + ["sharer"] * 3, seed=7, logger=lg)
    assert strip(tmp_path / "a.jsonl") == strip(tmp_path / "b.jsonl")


def test_log_has_every_turn(tmp_path):
    p = str(tmp_path / "r.jsonl")
    with JsonlLogger(p) as lg:
        _, end = run(["sharer"] * 6, logger=lg)
    recs = list(read_log(p))
    assert recs[0].event == "run_start" and recs[-1].event == "run_end"
    turns = [r for r in recs if r.event == "turn"]
    rounds = [r for r in recs if r.event == "round_end"]
    assert len(rounds) == end.turns
    assert len(turns) == 6 * end.turns   # nobody died

def test_claiming_alone_does_not_prevent_stall():
    # specialists claim 1/round and never write: claiming is not progress, so the run
    # stalls after stall_rounds instead of running to max_rounds (pilot v7 regression)
    _, end = run(["specialist"] * 6, max_rounds=50)
    assert end.outcome == Outcome.stalled and end.turns == settings.STALL_ROUNDS


def test_idle_agents_still_stall():
    # agents that only message: no document change, no credits moved -> stall at stall_rounds
    from swarm.agents.bots import SpecialistBot
    pool = BotPool(["specialist"] * 6)
    pool.bots = {i: SpecialistBot(i, action="send_message") for i in range(6)}
    cfg = ExperimentConfig(name="t", max_rounds=30)
    end = World(cfg, 0, pool, NullLogger()).run()
    assert end.outcome == Outcome.stalled and end.turns == settings.STALL_ROUNDS
