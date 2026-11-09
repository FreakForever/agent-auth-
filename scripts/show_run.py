"""Read one run log like a transcript: who thought what, did what, and what happened.

    python scripts/show_run.py logs/pilot/collab_mesh__s0__abc123.jsonl
    python scripts/show_run.py <log> --rounds 1-5        # only some rounds
    python scripts/show_run.py <log> --agent 3           # only one agent
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swarm.diagnose import diagnose  # noqa: E402
from swarm.metrics import Run  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log")
    p.add_argument("--rounds", default=None, help="e.g. 1-5")
    p.add_argument("--agent", type=int, default=None)
    a = p.parse_args()

    run = Run.load(a.log)
    lo, hi = (map(int, a.rounds.split("-")) if a.rounds else (0, 10 ** 9))
    s = run.start
    print(f"RUN {s['run_id']}  topology={run.config['topology']}  reward={run.config['reward_mode']}")
    print("edges:", s["edges"])
    for i, (fact, (ha, hb)) in enumerate(zip(s["facts"], run.holder)):
        print(f"  section {i}: {fact:<14} part A held by {ha}, part B held by {hb}")

    rounds = {r["round"]: r for r in run.rounds}
    by_round = {}
    for t in run.turns:
        by_round.setdefault(t["round"], []).append(t)
    for rnd in sorted(by_round):
        if not lo <= rnd <= hi:
            continue
        print(f"\n--- ROUND {rnd} ---")
        for t in by_round[rnd]:
            if a.agent is not None and t["agent_id"] != a.agent:
                continue
            act = {k: v for k, v in t["action"].items() if k != "type"}
            bad = "" if t["valid_json"] else f"  !! INVALID ({t['parse_error']})"
            print(f" agent {t['agent_id']}  credits_after={t['credits_after']}{bad}")
            if t["thought"]:
                print(f"    thought: {t['thought']}")
            print(f"    action : {t['action']['type']} {act}")
            print(f"    result : {t['result'].get('detail')}")
            if t.get("note"):
                print(f"    note   : {t['note']}")
        r = rounds.get(rnd)
        if r:
            print(f"  end: pool={r['pool']} credits={r['credits']} alive={r['alive']}")
            print(f"       document={r['document']}")

    if run.end:
        e = run.end
        print(f"\nOUTCOME {e['outcome']}  turns={e['turns']}  facts_correct={e['facts_correct']}/6 "
              f"deaths={e['deaths']}  invalid_json={e['invalid_json_count']}")
    d = diagnose(run)
    print("causes:", d["section_causes"], " unsupported writes:", d["unsupported_writes"])


if __name__ == "__main__":
    main()