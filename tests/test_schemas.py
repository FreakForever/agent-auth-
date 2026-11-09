import json
import pytest
from swarm.config import (
    ExperimentConfig, parse_turn, parse_record, RunStart, TurnLog, RoundEnd, RunEnd, Outcome,
)


def test_config_loads_and_hash_ignores_seeds():
    c = ExperimentConfig.from_yaml("configs/pilot/collab_mesh.yaml")
    c2 = c.model_copy(update={"seeds": [9]})
    assert c.config_hash() == c2.config_hash()
    c3 = ExperimentConfig.model_validate({**c.model_dump(), "reward_mode": "individual"})
    assert c.config_hash() != c3.config_hash()


def test_config_hash_changes_with_prompt_version_and_settings(monkeypatch):
    from swarm import settings
    c = ExperimentConfig.from_yaml("configs/pilot/collab_mesh.yaml")
    h = c.config_hash()
    assert settings.snapshot()["PROMPT_VERSION"] == settings.PROMPT_VERSION
    monkeypatch.setattr(settings, "PROMPT_VERSION", "v-test")
    assert c.config_hash() != h
    monkeypatch.undo()
    assert c.config_hash() == h
    monkeypatch.setattr(settings, "AGENT_START_CREDITS", 999)
    assert c.config_hash() != h
    monkeypatch.undo()
    monkeypatch.setattr(settings, "ENGINE_VERSION", "e-test")
    assert c.config_hash() != h


def test_snapshot_excludes_endpoint_and_paths():
    from swarm import settings
    snap = settings.snapshot()
    for k in ("VLLM_BASE_URL", "VLLM_API_KEY", "ROOT", "LOG_DIR", "CONFIG_DIR"):
        assert k not in snap
    assert snap["AGENT_START_CREDITS"] == 10


def test_config_rejects_bad_compromised_id():
    with pytest.raises(ValueError):
        ExperimentConfig(name="x", compromised_agents=[7])


@pytest.mark.parametrize("raw,typ", [
    ('{"thought":"t","action":{"type":"claim_resource","amount":3}}', "claim_resource"),
    ('{"thought":"t","action":{"type":"send_message","to":"all","text":"hi"}}', "send_message"),
    ('```json\n{"thought":"t","action":{"type":"pass"}}\n```', "pass"),
    ('Sure! {"thought":"t","action":{"type":"write_section","section":2,"content":"17 March"}}', "write_section"),
])
def test_parse_turn_ok(raw, typ):
    turn, err = parse_turn(raw)
    assert err is None and turn.action.type == typ


@pytest.mark.parametrize("raw", [
    "I will claim credits",                                            # no JSON
    '{"thought":"t","action":{"type":"claim_resource","amount":50}}',  # over MAX_CLAIM
    '{"thought":"t","action":{"type":"teleport"}}',                    # unknown action
])
def test_parse_turn_rejects(raw):
    turn, err = parse_turn(raw)
    assert turn is None and err


def test_log_records_roundtrip():
    recs = [
        RunStart(run_id="r", config_hash="h", seed=0, config={}, edges=[(0, 1)],
                 facts=["a"] * 6, fact_holder=[[0, 1]] * 6),
        TurnLog(run_id="r", round=1, agent_id=0, model="m", observation={}, raw_output="{}",
                action={"type": "pass"}, valid_json=False, result={"ok": True}, credits_after=2),
        RoundEnd(run_id="r", round=1, pool=34, credits=[2] * 6, alive=[True] * 6,
                 document=[None] * 6, rewards=[0.0] * 6),
        RunEnd(run_id="r", outcome=Outcome.success, turns=9, facts_correct=6, deaths=0,
               invalid_json_count=1),
    ]
    for r in recs:
        back = parse_record(r.model_dump_json())
        assert type(back) is type(r)

def test_start_credits_defaults_and_override():
    from swarm import settings
    from swarm.agents.bots import BotPool
    from swarm.logger import NullLogger
    from swarm.world import World
    assert ExperimentConfig(name="x").start_credits is None
    w = World(ExperimentConfig(name="x"), 0, BotPool(["sharer"] * 6), NullLogger())
    assert w.credits == [settings.AGENT_START_CREDITS] * 6
    base = ExperimentConfig.from_yaml("configs/pilot/baseline_single.yaml")
    w = World(base, 0, BotPool(["sharer"]), NullLogger())
    assert w.credits == [60]
    with pytest.raises(ValueError):
        ExperimentConfig(name="x", start_credits=-1)
