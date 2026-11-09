"""Phase 5 checks: baseline support, runner, resume, report (bots only, no GPU)."""
from swarm.agents.bots import BotPool
from swarm.config import ExperimentConfig
from swarm.metrics import summarize_dir
from swarm.report import checks, condition_table, recommend_cap
from swarm.run import is_complete, run_condition
from swarm.world import make_facts


def bots(kind):
    return lambda cfg, seed: BotPool([kind] * cfg.n_agents, seed)


def test_single_agent_holds_everything():
    facts, frags, holdings = make_facts(0, 6, 1)
    assert len(holdings) == 1 and len(holdings[0]) == 12


def test_single_sharer_solves_baseline(tmp_path):
    cfg = ExperimentConfig.from_yaml("configs/pilot/baseline_single.yaml")
    paths = run_condition(cfg, str(tmp_path), pool_factory=bots("sharer"), verbose=False)
    df = summarize_dir(str(tmp_path))
    assert len(paths) == len(cfg.seeds) and (df.facts_correct == 6).all()


def test_parallel_and_resume(tmp_path):
    cfg = ExperimentConfig(name="p", seeds=[0, 1, 2, 3])
    first = run_condition(cfg, str(tmp_path), parallel=4, pool_factory=bots("sharer"), verbose=False)
    assert len(first) == 4 and all(is_complete(p) for p in first)
    again = run_condition(cfg, str(tmp_path), parallel=4, pool_factory=bots("sharer"), verbose=False)
    assert again == []   # everything skipped


def test_incomplete_log_is_rerun(tmp_path):
    cfg = ExperimentConfig(name="p", seeds=[0])
    (path,) = run_condition(cfg, str(tmp_path), pool_factory=bots("sharer"), verbose=False)
    lines = open(path).read().splitlines()[:-1]          # drop run_end = simulated crash
    open(path, "w").write("\n".join(lines) + "\n")
    assert not is_complete(path)
    assert len(run_condition(cfg, str(tmp_path), pool_factory=bots("sharer"), verbose=False)) == 1


def test_report_and_cap(tmp_path):
    for name, topo in [("pilot_mesh", "mesh"), ("pilot_chain", "chain"), ("pilot_none", "none")]:
        run_condition(ExperimentConfig(name=name, topology=topo, seeds=[0, 1, 2]), str(tmp_path),
                      pool_factory=bots("sharer"), verbose=False)
    df = summarize_dir(str(tmp_path))
    table = condition_table(df)
    assert set(table.index) == {"pilot_mesh", "pilot_chain", "pilot_none"}
    cap = recommend_cap(df)
    assert cap["cap"] >= 20 and cap["basis"] == "successful runs"
    msgs = checks(df)
    assert any(m.startswith("OK") and "no-messaging" in m for m in msgs)
    assert "none:" in table.loc["pilot_mesh", "causes"]