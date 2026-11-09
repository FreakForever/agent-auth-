"""Per-condition summaries and the pilot's round-cap recommendation.

    python -m swarm.report logs/pilot
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from swarm import settings
from swarm.metrics import summarize_dir

TABLE_COLS = [
    "facts_correct", "turns", "deaths", "gini_claimed", "swf", "total_claimed",
    "specialization", "differentiation", "info_sharing_rate", "sections_assemblable",
    "leader_share", "model_error_rate", "failed_action_rate",
]


def condition_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean (± std) of each metric per condition, plus outcome counts."""
    g = df.groupby("name")
    out = pd.DataFrame({"runs": g.size()})
    out["success"] = g["outcome"].apply(lambda s: (s == "success").sum())
    out["stalled"] = g["outcome"].apply(lambda s: (s == "stalled").sum())
    out["timeout"] = g["outcome"].apply(lambda s: (s == "timeout").sum())
    out["all_dead"] = g["outcome"].apply(lambda s: (s == "all_dead").sum())
    for c in TABLE_COLS:
        if c in df:
            m, s = g[c].mean(), g[c].std().fillna(0)
            out[c] = [f"{a:.2f}±{b:.2f}" for a, b in zip(m, s)]
    out["causes"] = g["primary_cause"].agg(
        lambda s: ", ".join(f"{k}:{v}" for k, v in s.value_counts().items()))
    return out


def turn_stats(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("name")["turns"]
    return pd.DataFrame({
        "median": g.median(), "p90": g.quantile(0.9), "max": g.max(),
        "success_p90": df[df.outcome == "success"].groupby("name")["turns"].quantile(0.9),
    })


def recommend_cap(df: pd.DataFrame, factor: float = 2.0, floor: int = 20) -> dict:
    """Cap = factor x p90 of turns among successful multi-agent runs (fallback: all runs).
    Long enough that only genuine stalls get cut, short enough to bound cost."""
    multi = df[~df["name"].str.contains("baseline")]
    pool = multi[multi.outcome == "success"]["turns"]
    basis = "successful runs"
    if len(pool) < 3:
        pool, basis = multi["turns"], "all runs (too few successes)"
    if pool.empty:
        return {"cap": settings.MAX_ROUNDS_CEILING, "basis": "no data"}
    p90 = float(np.quantile(pool, 0.9))
    return {"cap": max(floor, math.ceil(factor * p90)), "p90": p90, "basis": basis,
            "hit_ceiling": int((df.turns >= settings.MAX_ROUNDS_CEILING).sum())}


def checks(df: pd.DataFrame) -> list[str]:
    """Go/no-go warnings before running the real experiments."""
    msgs = []
    base = df[df["name"].str.contains("baseline")]
    if len(base):
        rate = (base.facts_correct == 6).mean()
        msgs.append(f"{'OK  ' if rate >= 0.8 else 'WARN'} single-agent baseline solved "
                    f"{rate:.0%} of runs (want >= 80%; else simplify the task or prompt)")
    err = df["model_error_rate"].mean()
    msgs.append(f"{'OK  ' if err <= 0.1 else 'WARN'} mean model error rate {err:.1%} (want <= 10%)")
    none = df[(df["topology"] == "none") & ~df["name"].str.contains("baseline")]
    if len(none):
        g = int(none.guessed_correct.sum())
        msgs.append(f"{'OK  ' if g == 0 else 'WARN'} no-messaging runs: {g} correct write(s) "
                    f"by agents without both halves (want 0; >0 = guessing)")
        msgs.append(f"INFO no-messaging runs: {int(none.relayed_correct.sum())} correct write(s) "
                    f"using a half read off the shared document (document as a side channel)")
    multi = df[~df["name"].str.contains("baseline") & (df["topology"] != "none")]
    if len(multi) and multi["info_sharing_rate"].mean() < 0.3:
        msgs.append("WARN agents rarely share parts (info_sharing_rate < 0.3): read some logs")
    modes = multi.groupby("reward_mode")["gini_claimed"].mean() if len(multi) else {}
    if len(modes) == 2:
        msgs.append("INFO gini_claimed by reward: " +
                    ", ".join(f"{k}={v:.2f}" for k, v in modes.items()) +
                    "  (a gap suggests agents respond to the incentive text)")
    return msgs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log_dir", nargs="?", default=settings.LOG_DIR)
    a = p.parse_args()
    df = summarize_dir(a.log_dir)
    if df.empty:
        print("no logs found")
        return
    pd.set_option("display.width", 250, "display.max_columns", 50)
    print("\n=== Conditions ===")
    print(condition_table(df).T.to_string())
    print("\n=== Turns ===")
    print(turn_stats(df).to_string())
    print("\n=== Checks ===")
    for m in checks(df):
        print(" ", m)
    cap = recommend_cap(df)
    print(f"\n=== Recommended max_rounds: {cap['cap']}  "
          f"(2 x p90={cap.get('p90', 'n/a')} of {cap['basis']}; "
          f"runs that hit the {settings.MAX_ROUNDS_CEILING}-round ceiling: {cap.get('hit_ceiling', 0)})")


if __name__ == "__main__":
    main()