"""One short real LLM run through the full pipeline.

Usage:
    python scripts/llm_smoke.py                       # 6 agents, mesh, collaborative, 15 rounds
    python scripts/llm_smoke.py --topology chain --reward individual --rounds 20
    python scripts/llm_smoke.py --no-json-mode

Prints every action as it happens, then the metrics and failure diagnosis.
The log is written to logs/smoke__....jsonl so you can read it afterwards.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swarm import settings  # noqa: E402
from swarm.agents.llm_agent import LLMPool  # noqa: E402
from swarm.config import ExperimentConfig  # noqa: E402
from swarm.diagnose import diagnose  # noqa: E402
from swarm.logger import JsonlLogger  # noqa: E402
from swarm.metrics import Run, compute_metrics  # noqa: E402
from swarm.world import World  # noqa: E402


class PrintingLogger(JsonlLogger):
    """Writes the log AND prints a one-line summary of each turn."""

    def __init__(self, path):
        super().__init__(path)
        self.t0 = time.perf_counter()

    def write(self, record):
        super().write(record)
        if record.event == "turn":
            a = record.action
            body = {k: v for k, v in a.items() if k != "type"}
            flag = "" if record.valid_json else "  [INVALID -> pass]"
            print(f"  r{record.round:>2} agent {record.agent_id}: {a['type']:<15} {body}"
                  f"  => {record.result.get('detail')}{flag}")
        elif record.event == "round_end":
            doc = ["#" if c else "." for c in record.document]
            print(f"  -- end r{record.round}: pool={record.pool} credits={record.credits} "
                  f"doc=[{''.join(doc)}]  ({time.perf_counter() - self.t0:.1f}s)\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--topology", default="mesh", choices=["mesh", "star", "chain", "none"])
    p.add_argument("--reward", default="collaborative", choices=["collaborative", "individual"])
    p.add_argument("--rounds", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-json-mode", action="store_true")
    a = p.parse_args()

    cfg = ExperimentConfig(name="smoke", topology=a.topology, reward_mode=a.reward,
                           max_rounds=a.rounds, seeds=[a.seed])
    pool = LLMPool(cfg, a.seed, json_mode=not a.no_json_mode)
    pool.preflight()
    world = World(cfg, a.seed, pool, None)
    path = os.path.join(settings.LOG_DIR, f"{world.run_id}.jsonl")
    print(f"model={settings.MODEL_NAME}  run={world.run_id}")
    print(f"ground truth: {world.facts}\n")

    with PrintingLogger(path) as lg:
        world.log = lg
        end = world.run()
    pool.close()

    run = Run.load(path)
    m, d = compute_metrics(run), diagnose(run)
    print("=" * 70)
    print(f"outcome={end.outcome.value} turns={end.turns} facts_correct={end.facts_correct}/6 "
          f"deaths={end.deaths} invalid_json={end.invalid_json_count}")
    print("final document:", run.rounds[-1]["document"])
    for k in ["gini_claimed", "swf", "specialization", "differentiation", "info_sharing_rate",
              "sections_assemblable", "leader", "model_error_rate", "failed_action_rate"]:
        print(f"  {k:<22} {m[k]}")
    print(f"  section causes         {d['section_causes']}")
    print(f"  unsupported writes     {d['unsupported_writes']}")
    print(f"\nlog: {path}")


if __name__ == "__main__":
    main()