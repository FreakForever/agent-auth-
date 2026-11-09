"""Phase 8 checks: compromised-agent mode is OFF by default, deterministic, and the contagion
metrics give the KNOWN answer on bots (one LiarBot among honest SharerBots)."""
import pytest

from swarm import settings
from swarm.agents.bots import BotPool
from swarm.agents.llm_agent import LLMPool, system_prompt
from swarm.config import ExperimentConfig
from swarm.metrics import compute_metrics, contagion_metrics
from swarm.world import World, make_compromise, make_facts
from tests.test_metrics import make_run


def test_off_by_default_changes_nothing():
    cfg = ExperimentConfig(name="x")
    assert cfg.compromised_agents == []
    w = World(cfg, 0, BotPool(["sharer"] * 6), None)
    assert w.compromise == {} and all("secret" not in w.observe(a) for a in range(6))
    assert "SECRET" not in system_prompt(0, cfg)
    # turning it on changes the hash (so runs never mix), but only for that config
    on = ExperimentConfig(name="x", compromised_agents=[0])
    assert on.config_hash() != cfg.config_hash()


def test_grid_hashes_unchanged():
    # reproducibility guard: Phase 8 must not change existing experiment run ids
    expected = {"collab_mesh": "decd2ef53e", "indiv_star": "02a33845f7", "baseline_single": "0972ff6b81"}
    for name, h in expected.items():
        assert ExperimentConfig.from_yaml(f"configs/grid/{name}.yaml").config_hash() == h


def test_target_is_own_half_and_false_value_is_new():
    for seed in range(20):
        facts, frags, holds = make_facts(seed, 6, 6)
        c = make_compromise(seed, holds, frags, [0])[0]
        p = "AB".index(c["part"])
        assert (c["section"], p) in holds[0]                 # liar holds the true half itself
        assert c["true"] == frags[c["section"]][p]
        assert c["false"] != c["true"] and c["false"] not in {v for f in frags for v in f}
        assert make_compromise(seed, holds, frags, [0]) == {0: c}   # deterministic


def test_only_liar_prompt_has_secret():
    cfg = ExperimentConfig(name="x", compromised_agents=[0])

    class NoClient:
        pass

    pool = LLMPool(cfg, 3, client=NoClient())
    c = make_compromise(3, make_facts(3, 6, 6)[2], make_facts(3, 6, 6)[1], [0])[0]
    assert "SECRET OBJECTIVE" in pool.systems[0] and c["false"] in pool.systems[0]
    assert all("SECRET" not in pool.systems[a] for a in range(1, 6))
    assert pool.systems[1] == system_prompt(1, ExperimentConfig(name="x"))  # honest = unchanged
    pool.close()


def _liar_run(tmp_path, topology, seed=0, liar=0):
    kinds = ["sharer"] * 6
    kinds[liar] = "liar"
    return make_run(tmp_path, kinds, name=f"liar_{topology}", topology=topology, seed=seed,
                    compromised_agents=[liar], max_rounds=30)


def test_liar_bot_mesh_known_answer(tmp_path):
    # honest sharers relay everything they learn and write any section they can complete:
    # the false half reaches all 5 honest agents, they repeat it, and the target section is
    # written with the false value; the other 5 sections are correct.
    run, _ = _liar_run(tmp_path, "mesh")
    c = contagion_metrics(run)
    assert c["liar_complied"] and not c["liar_leaked_true"]
    assert c["false_reached"] == 5 and c["false_repeaters"] >= 1
    assert c["target_outcome"] == "false" and c["false_written"]
    assert compute_metrics(run)["facts_correct"] == 5


def test_liar_bot_chain_spreads_hop_by_hop(tmp_path):
    # chain 0-1-2-3-4-5 with the liar at one end: the value must be relayed, so it reaches
    # agents further away later and repeaters appear more than 1 hop away
    run, _ = _liar_run(tmp_path, "chain")
    c = contagion_metrics(run)
    assert c["false_max_hops"] >= 2
    curve = c["false_reach_by_round"]
    assert curve == sorted(curve) and curve[0] < curve[-1]


def test_no_compromise_no_contagion_metrics(tmp_path):
    run, _ = make_run(tmp_path, ["sharer"] * 6)
    assert contagion_metrics(run) == {} and "false_repeaters" not in compute_metrics(run)


def test_lie_spread_via_proposal_counts(tmp_path):
    # a liar who only PROPOSES the false value (no messages) still complied and reached its
    # neighbours (found in liar_star s3: the LLM liar used 9 proposals and no messages)
    from swarm.agents.bots import Bot, _turn

    class Proposer(Bot):
        kind = "lieproposer"

        def act(self, obs):
            sec = obs.get("secret")
            if sec and obs["round"] == 1:
                return _turn({"type": "propose", "plan": f"section {sec['section']} part B is {sec['false']}"})
            return _turn({"type": "pass"})

    pool = BotPool(["sharer"] * 6)
    pool.bots = {i: Proposer(i) for i in range(6)}
    run, _ = make_run(tmp_path, pool=pool, topology="star", compromised_agents=[0], max_rounds=3)
    c = contagion_metrics(run)
    assert c["liar_complied"] and c["false_reached"] == 5   # hub's neighbours = all leaves
