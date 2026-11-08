"""Phase 3: measurements (the M's), computed offline from one run's JSONL log.

Nothing here talks to the model or the world. Every number is derived from the
log, so any result can be re-checked later by re-running this file.

Metrics per run
  Resource sharing   gini_claimed, gini_final, swf
  Division of labour specialization (within agent), differentiation (between agents)
  Leadership         endorsements per agent, leader, leader_share, centrality correlation
  Information        info_sharing_rate, sections_assemblable
  Outcome            outcome, turns, facts_correct, deaths
  Model quality      model_error_rate, failed_action_rate
"""
from __future__ import annotations

import glob
import math
import os
import re
from collections import Counter, defaultdict
from typing import Optional

import networkx as nx
import numpy as np

from swarm import settings
from swarm.logger import read_log_dicts

ACTION_TYPES = [
    "claim_resource", "give_resource", "send_message", "write_section",
    "propose", "endorse", "pass",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
class Run:
    """A parsed run log, split by event type."""

    def __init__(self, records: list[dict]):
        self.start = next(r for r in records if r["event"] == "run_start")
        self.turns = [r for r in records if r["event"] == "turn"]
        self.rounds = [r for r in records if r["event"] == "round_end"]
        self.end = next((r for r in records if r["event"] == "run_end"), None)
        self.n = self.start["config"]["n_agents"]
        self.config = self.start["config"]
        self.fragments = [tuple(f.split("-", 1)) for f in self.start["facts"]]
        self.holder = self.start["fact_holder"]          # section -> [holder A, holder B]

    @classmethod
    def load(cls, path: str) -> "Run":
        return cls(read_log_dicts(path))


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def gini(values) -> float:
    """0 = perfectly equal, (n-1)/n = one agent has everything."""
    x = np.asarray(values, dtype=float)
    if x.size == 0 or x.sum() == 0:
        return 0.0
    diff = np.abs(x[:, None] - x[None, :]).sum()
    return float(diff / (2 * x.size ** 2 * x.mean()))


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _spearman(a, b) -> Optional[float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return None  # undefined when one side is constant (e.g. mesh: equal centrality)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def _mentions(text: str, value: str) -> bool:
    return re.search(rf"\b{re.escape(value.lower())}\b", (text or "").lower()) is not None


def knowledge(run: Run) -> list[set[str]]:
    """Fragment values each agent knew by the end: its own, plus any value that
    appeared in a successful message sent to it."""
    known = [set() for _ in range(run.n)]
    for s, (ha, hb) in enumerate(run.holder):
        known[ha].add(run.fragments[s][0])
        known[hb].add(run.fragments[s][1])
    all_values = [v for pair in run.fragments for v in pair]
    for t in run.turns:
        a = t["action"]
        if a["type"] == "send_message" and t["result"].get("ok"):
            for v in all_values:
                if _mentions(a["text"], v):
                    for to in t["result"]["to"]:
                        known[to].add(v)
    return known


def shared_values(run: Run) -> set[str]:
    """Fragment values that left their holder in at least one successful message."""
    out = set()
    for t in run.turns:
        a = t["action"]
        if a["type"] == "send_message" and t["result"].get("ok"):
            for pair in run.fragments:
                for v in pair:
                    if _mentions(a["text"], v):
                        out.add(v)
    return out


# ---------------------------------------------------------------------------
# Metric groups
# ---------------------------------------------------------------------------
def resource_metrics(run: Run, facts_correct: int) -> dict:
    claimed = [0] * run.n
    given, received = [0] * run.n, [0] * run.n
    for t in run.turns:
        a, r = t["action"], t["result"]
        if a["type"] == "claim_resource":
            claimed[t["agent_id"]] += r.get("granted", 0)
        elif a["type"] == "give_resource" and r.get("ok"):
            given[t["agent_id"]] += r["amount"]
            received[r["to"]] += r["amount"]
    final = run.rounds[-1]["credits"] if run.rounds else [0] * run.n
    g = gini(claimed)
    progress = facts_correct / settings.N_SECTIONS
    return {
        "claimed_per_agent": claimed,
        "total_claimed": sum(claimed),
        "gini_claimed": g,
        "gini_final": gini(final),
        "gives": sum(given),
        "swf": (1 - g) * progress,   # fairness x task progress, both in [0, 1]
    }


def labour_metrics(run: Run) -> dict:
    """specialization: how concentrated each agent's actions are (1 = one action type only).
    differentiation: how different agents' action mixes are from each other
    (0 = everyone does the same mix, 1 = completely disjoint roles)."""
    counts = np.zeros((run.n, len(ACTION_TYPES)))
    for t in run.turns:
        counts[t["agent_id"], ACTION_TYPES.index(t["action"]["type"])] += 1
    active = counts.sum(1) > 0
    probs = counts[active] / counts[active].sum(1, keepdims=True)
    k = math.log(len(ACTION_TYPES))
    spec = [1 - _entropy(p) / k for p in probs]

    # mean pairwise Jensen-Shannon divergence, normalised to [0, 1]
    js = []
    for i in range(len(probs)):
        for j in range(i + 1, len(probs)):
            m = (probs[i] + probs[j]) / 2
            js.append((_entropy(m) - (_entropy(probs[i]) + _entropy(probs[j])) / 2) / math.log(2))
    return {
        "specialization": float(np.mean(spec)) if spec else 0.0,
        "specialization_per_agent": [round(s, 3) for s in spec],
        "differentiation": float(np.mean(js)) if js else 0.0,
        "dominant_action": [ACTION_TYPES[int(row.argmax())] if row.sum() else None for row in counts],
    }


def leadership_metrics(run: Run) -> dict:
    endorsements = [0] * run.n
    proposals = [0] * run.n
    for t in run.turns:
        a, r = t["action"], t["result"]
        if a["type"] == "endorse" and r.get("ok"):
            endorsements[r["author"]] += 1
        elif a["type"] == "propose" and r.get("ok"):
            proposals[t["agent_id"]] += 1
    g = nx.Graph()
    g.add_nodes_from(range(run.n))
    g.add_edges_from(run.start["edges"])
    centrality = [nx.degree_centrality(g)[i] for i in range(run.n)]
    total = sum(endorsements)
    return {
        "proposals_per_agent": proposals,
        "endorsements_per_agent": endorsements,
        "leader": int(np.argmax(endorsements)) if total else None,
        "leader_share": max(endorsements) / total if total else 0.0,
        "centrality": [round(c, 3) for c in centrality],
        "centrality_endorsement_corr": _spearman(centrality, endorsements),
    }


def info_metrics(run: Run) -> dict:
    shared = shared_values(run)
    all_values = [v for pair in run.fragments for v in pair]
    known = knowledge(run)
    assemblable = sum(
        any(a in k and b in k for k in known) for a, b in run.fragments
    )
    return {
        "info_sharing_rate": len(shared) / len(all_values),
        "sections_assemblable": assemblable,   # some agent had both halves
        "messages_sent": sum(t["action"]["type"] == "send_message" and t["result"].get("ok")
                             for t in run.turns),
    }


def quality_metrics(run: Run) -> dict:
    n = len(run.turns) or 1
    invalid = sum(not t["valid_json"] for t in run.turns)
    failed = sum(t["valid_json"] and not t["result"].get("ok", True) for t in run.turns)
    return {"model_error_rate": invalid / n, "failed_action_rate": failed / n}


# ---------------------------------------------------------------------------
# Per run / per directory
# ---------------------------------------------------------------------------
def compute_metrics(run: Run) -> dict:
    end = run.end or {}
    facts_correct = end.get("facts_correct", 0)
    out = {
        "run_id": run.start["run_id"],
        "name": run.config["name"],
        "config_hash": run.start["config_hash"],
        "seed": run.start["seed"],
        "topology": run.config["topology"],
        "reward_mode": run.config["reward_mode"],
        "compromised": bool(run.config["compromised_agents"]),
        "outcome": end.get("outcome", "incomplete"),
        "turns": end.get("turns", len(run.rounds)),
        "facts_correct": facts_correct,
        "deaths": end.get("deaths", 0),
    }
    out.update(resource_metrics(run, facts_correct))
    out.update(labour_metrics(run))
    out.update(leadership_metrics(run))
    out.update(info_metrics(run))
    out.update(quality_metrics(run))
    return out


def round_series(run: Run):
    """Long-format table for per-round plots: one row per (round, agent)."""
    import pandas as pd

    rows = []
    for r in run.rounds:
        g = gini(r["credits"])
        for a in range(run.n):
            rows.append({"round": r["round"], "agent": a, "credits": r["credits"][a],
                         "alive": r["alive"][a], "reward": r["rewards"][a],
                         "pool": r["pool"], "gini_credits": g,
                         "sections_filled": sum(c is not None for c in r["document"])})
    return pd.DataFrame(rows)


def summarize_dir(log_dir: str = settings.LOG_DIR):
    """One row per run with all scalar metrics plus failure diagnosis."""
    import pandas as pd

    from swarm.diagnose import diagnose

    rows = []
    for path in sorted(glob.glob(os.path.join(log_dir, "*.jsonl"))):
        run = Run.load(path)
        m = compute_metrics(run)
        d = diagnose(run)
        m["primary_cause"] = d["primary_cause"]
        m["high_model_error"] = d["high_model_error"]
        rows.append({k: v for k, v in m.items() if not isinstance(v, list)})
    return pd.DataFrame(rows)