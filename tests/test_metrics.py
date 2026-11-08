"""Phase 3 checks: metrics give the KNOWN answer on bots with known behaviour."""
import json

import pytest

from swarm import settings
from swarm.agents.bots import BotPool, SharerBot, SpecialistBot, _turn
from swarm.config import ExperimentConfig
from swarm.diagnose import diagnose
from swarm.logger import JsonlLogger, read_log_dicts
from swarm.metrics import Run, compute_metrics, gini, round_series, summarize_dir
from swarm.world import World


def make_run(tmp_path, kinds=None, pool=None, name="t", topology="mesh", seed=0, **kw):
    cfg = ExperimentConfig(name=name, topology=topology, **kw)
    pool = pool or BotPool(kinds, seed)
    path = str(tmp_path / f"{name}_{topology}_{seed}.jsonl")
    with JsonlLogger(path) as lg:
        World(cfg, seed, pool, lg).run()
    return Run.load(path), path


# ---------------- building blocks ----------------
def test_gini_extremes():
    assert gini([5, 5, 5, 5]) == 0
    assert gini([0, 0, 0, 12]) == pytest.approx(0.75)   # (n-1)/n
    assert gini([0, 0, 0, 0]) == 0


# ---------------- resource sharing ----------------
def test_hoarders_raise_gini(tmp_path):
    fair, _ = make_run(tmp_path, ["sharer"] * 6, name="fair")
    mixed, _ = make_run(tmp_path, ["hoarder"] * 3 + ["sharer"] * 3, name="mixed",
                        max_rounds=20, stall_rounds=99)
    mf, mm = compute_metrics(fair), compute_metrics(mixed)
    assert mm["gini_claimed"] > mf["gini_claimed"] + 0.2
    assert mf["swf"] > mm["swf"]


# ---------------- division of labour ----------------
def test_specialists_vs_random(tmp_path):
    spec, _ = make_run(tmp_path, ["specialist"] * 6, name="spec", max_rounds=15)
    rand, _ = make_run(tmp_path, ["random"] * 6, name="rand", max_rounds=30, stall_rounds=99)
    s, r = compute_metrics(spec)["specialization"], compute_metrics(rand)["specialization"]
    assert s > 0.95 and r < 0.5 and s - r > 0.5


def test_differentiation_needs_different_roles(tmp_path):
    same, _ = make_run(tmp_path, ["specialist"] * 6, name="same", max_rounds=15)
    pool = BotPool(["specialist"] * 6)
    for i in range(3, 6):
        pool.bots[i] = SpecialistBot(i, action="send_message")
    split, _ = make_run(tmp_path, pool=pool, name="split", max_rounds=15)
    assert compute_metrics(same)["differentiation"] < 0.01
    assert compute_metrics(split)["differentiation"] > 0.4


# ---------------- leadership ----------------
def test_star_hub_becomes_leader(tmp_path):
    run, _ = make_run(tmp_path, ["proposer"] + ["sharer"] * 5, topology="star",
                      max_rounds=12, stall_rounds=99)
    m = compute_metrics(run)
    assert m["leader"] == 0
    assert m["leader_share"] == 1.0
    assert m["centrality_endorsement_corr"] > 0.5


def test_mesh_centrality_corr_undefined(tmp_path):
    run, _ = make_run(tmp_path, ["proposer"] + ["sharer"] * 5, max_rounds=8, stall_rounds=99)
    assert compute_metrics(run)["centrality_endorsement_corr"] is None  # equal centrality


# ---------------- information ----------------
def test_info_sharing(tmp_path):
    good, _ = make_run(tmp_path, ["sharer"] * 6, name="good")
    bad, _ = make_run(tmp_path, ["hoarder"] * 6, name="bad")
    alone, _ = make_run(tmp_path, ["sharer"] * 6, name="alone", topology="none")
    assert compute_metrics(good)["info_sharing_rate"] == 1.0
    assert compute_metrics(good)["sections_assemblable"] == 6
    assert compute_metrics(bad)["info_sharing_rate"] == 0.0
    assert compute_metrics(alone)["sections_assemblable"] == 0


# ---------------- diagnosis ----------------
def test_diagnose_not_shared(tmp_path):
    run, _ = make_run(tmp_path, ["hoarder"] * 6)
    d = diagnose(run)
    assert set(d["section_causes"].values()) == {"not_shared"}


def test_diagnose_success_has_no_causes(tmp_path):
    run, _ = make_run(tmp_path, ["sharer"] * 6)
    assert diagnose(run)["primary_cause"] == "none"


def test_diagnose_resource_blocked(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "POOL_START", 0)
    monkeypatch.setattr(settings, "POOL_REGEN", 0)
    run, _ = make_run(tmp_path, ["sharer"] * 6)
    d = diagnose(run)
    assert d["primary_cause"] == "resource_blocked"


def test_diagnose_not_written(tmp_path):
    # one round only: everyone broadcasts, nobody has time to write
    run, _ = make_run(tmp_path, ["sharer"] * 6, max_rounds=1)
    assert set(diagnose(run)["section_causes"].values()) == {"not_written"}


def test_diagnose_misreported(tmp_path):
    class Garbler(SharerBot):
        kind = "garbler"

        def _writable(self, obs):
            return [(s, c.replace("-", " ")[:-1] + "X") for s, c in super()._writable(obs)]

    pool = BotPool(["sharer"] * 6)
    pool.bots = {i: Garbler(i) for i in range(6)}
    run, _ = make_run(tmp_path, pool=pool, max_rounds=12, stall_rounds=99)
    assert diagnose(run)["primary_cause"] == "misreported"


def test_diagnose_overwritten(tmp_path):
    _, path = make_run(tmp_path, ["sharer"] * 6)
    recs = read_log_dicts(path)
    last_round = max(i for i, r in enumerate(recs) if r["event"] == "round_end")
    vandal = dict(next(r for r in recs if r["event"] == "turn"))
    vandal.update(action={"type": "write_section", "section": 0, "content": "vandalised"},
                  result={"ok": True, "section": 0, "correct": False})
    recs.insert(last_round, vandal)
    recs[last_round + 1]["document"][0] = "vandalised"
    d = diagnose(Run(recs))
    assert d["section_causes"] == {0: "overwritten"}


def test_model_error_flag(tmp_path):
    run, _ = make_run(tmp_path, ["broken"] * 3 + ["sharer"] * 3, max_rounds=10)
    d = diagnose(run)
    assert d["high_model_error"] and d["model_error_rate"] > 0.2


# ---------------- tables ----------------
def test_summary_and_series(tmp_path):
    make_run(tmp_path, ["sharer"] * 6, name="a")
    make_run(tmp_path, ["hoarder"] * 6, name="b")
    df = summarize_dir(str(tmp_path))
    assert len(df) == 2 and {"gini_claimed", "primary_cause", "specialization"} <= set(df.columns)
    run, _ = make_run(tmp_path, ["sharer"] * 6, name="c")
    s = round_series(run)
    assert len(s) == 6 * run.end["turns"]