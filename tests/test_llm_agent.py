"""Phase 4 checks with a FAKE model client, so they run without a GPU."""
import asyncio
import json

from swarm.agents.llm_agent import LLMPool, render_observation, system_prompt
from swarm.config import ExperimentConfig
from swarm.logger import JsonlLogger
from swarm.metrics import Run
from swarm.world import World


class _Msg:
    def __init__(self, content): self.content = content


class _Choice:
    def __init__(self, content): self.message = _Msg(content)


class _Resp:
    def __init__(self, content): self.choices = [_Choice(content)]


class FakeClient:
    """Returns scripted replies. `script(agent_id, n_calls_for_agent, kwargs) -> str`."""

    def __init__(self, script):
        self.script = script
        self.calls = []
        self.in_flight = 0
        self.max_in_flight = 0
        outer = self

        class _Completions:
            async def create(self, **kw):
                outer.in_flight += 1
                outer.max_in_flight = max(outer.max_in_flight, outer.in_flight)
                await asyncio.sleep(0.01)
                outer.in_flight -= 1
                aid = int(kw["messages"][0]["content"].split("You are agent ")[1].split()[0])
                n = sum(1 for c in outer.calls if c[0] == aid)
                outer.calls.append((aid, kw))
                return _Resp(outer.script(aid, n, kw))

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()

    async def close(self):
        pass


CFG = ExperimentConfig(name="llmtest", max_rounds=3, stall_rounds=99)


def test_prompt_contains_rules_and_incentive():
    collab = system_prompt(2, CFG)
    indiv = system_prompt(2, ExperimentConfig.model_validate({**CFG.model_dump(), "reward_mode": "individual"}))
    assert "You are agent 2" in collab and "TEAM score" in collab
    assert "OWN score" in indiv and "TEAM score" not in indiv
    assert '"type": "write_section"' in collab


def test_render_observation_shows_state():
    pool = FakeClient(lambda *a: '{"action":{"type":"pass"}}')
    w = World(CFG, 0, LLMPool(CFG, 0, client=pool), None)
    text = render_observation(w.observe(0))
    for frag in w.observe(0)["my_fragments"]:
        assert frag["value"] in text
    assert "DOCUMENT:" in text and "(empty)" in text and "MESSAGES RECEIVED" in text


def test_valid_reply_and_concurrency(tmp_path):
    client = FakeClient(lambda aid, n, kw: json.dumps(
        {"thought": "t", "action": {"type": "send_message", "to": "all", "text": f"hi from {aid}"}}))
    pool = LLMPool(CFG, 0, client=client)
    p = str(tmp_path / "r.jsonl")
    with JsonlLogger(p) as lg:
        end = World(CFG, 0, pool, lg).run()
    pool.close()
    assert end.invalid_json_count == 0
    assert client.max_in_flight == 6            # all agents called at once each round
    assert len(client.calls) == 6 * end.turns   # one call per agent per round, no retries


def test_retry_then_success():
    client = FakeClient(lambda aid, n, kw:
                        "I will pass" if n == 0 else '{"thought":"ok","action":{"type":"pass"}}')
    pool = LLMPool(CFG, 0, client=client)
    w = World(CFG, 0, pool, None)
    d = pool.decide({0: w.observe(0)})[0]
    pool.close()
    assert d.valid_json and d.retries == 1 and d.turn.action.type == "pass"
    retry_msgs = client.calls[1][1]["messages"]
    assert retry_msgs[-1]["role"] == "user" and "invalid" in retry_msgs[-1]["content"]


def test_fallback_to_pass_after_two_failures():
    client = FakeClient(lambda aid, n, kw: '{"thought":"x","action":{"type":"read_document"}}')
    pool = LLMPool(CFG, 0, client=client)
    w = World(CFG, 0, pool, None)
    d = pool.decide({0: w.observe(0)})[0]
    pool.close()
    assert not d.valid_json and d.turn is None and "schema_error" in d.parse_error
    assert d.raw_output.count("read_document") == 2


def test_request_error_is_contained():
    client = FakeClient(lambda *a: (_ for _ in ()).throw(RuntimeError("server down")))
    pool = LLMPool(CFG, 0, client=client)
    w = World(CFG, 0, pool, None)
    d = pool.decide({0: w.observe(0)})[0]
    pool.close()
    assert not d.valid_json and "request_error" in d.parse_error


def test_llm_pool_drives_a_full_run_with_metrics(tmp_path):
    """A scripted 'LLM' that shares then writes, through the real prompt/parse path."""
    def script(aid, n, kw):
        user = kw["messages"][1]["content"]
        parts = [l.split(": ")[-1] for l in user.splitlines() if ", part " in l]
        if n == 0:
            return json.dumps({"thought": "share", "action": {"type": "send_message", "to": "all",
                                                             "text": "my parts: " + " ".join(parts)}})
        return json.dumps({"thought": "wait", "action": {"type": "claim_resource", "amount": 2}})

    client = FakeClient(script)
    pool = LLMPool(CFG, 0, client=client)
    p = str(tmp_path / "r.jsonl")
    with JsonlLogger(p) as lg:
        World(CFG, 0, pool, lg).run()
    pool.close()
    from swarm.metrics import compute_metrics
    m = compute_metrics(Run.load(p))
    assert m["info_sharing_rate"] == 1.0 and m["model_error_rate"] == 0