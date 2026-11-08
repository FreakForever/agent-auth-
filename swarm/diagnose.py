"""Phase 3: why did a run not reach 6/6? Trace each missing section through the log.

Each incorrect section gets ONE cause, checked in this order:
  overwritten       the section was correct at some point, then replaced
  not_shared        some half of it never left its holder in any message
  not_combined      both halves were shared, but no single agent ever had both
  misreported       an agent who had both halves wrote it, but wrote it wrong
  resource_blocked  an agent had both halves but died or lacked credits to write
  not_written       an agent had both halves and could have written, but didn't

The first three are interaction failures, misreported is usually a model failure,
resource_blocked is a resource failure. The run also gets a model-error flag:
if too many turns were invalid JSON, its interaction results are suspect.
"""
from __future__ import annotations

from collections import Counter

from swarm import settings
from swarm.metrics import Run, knowledge, shared_values

HIGH_MODEL_ERROR = 0.2  # flag runs where >20% of turns were invalid output


def _was_correct(run: Run, s: int) -> bool:
    return any(
        t["action"]["type"] == "write_section" and t["result"].get("section") == s
        and t["result"].get("correct")
        for t in run.turns
    )


def diagnose(run: Run) -> dict:
    final_doc = run.rounds[-1]["document"] if run.rounds else [None] * settings.N_SECTIONS
    known = knowledge(run)
    shared = shared_values(run)

    knowers = {
        s: [a for a in range(run.n) if run.fragments[s][0] in known[a] and run.fragments[s][1] in known[a]]
        for s in range(settings.N_SECTIONS)
    }
    died = set()
    if run.rounds:
        died = {a for a in range(run.n) if not run.rounds[-1]["alive"][a]}
    broke = {  # agents who ever failed a write or claim for lack of credits/pool
        t["agent_id"] for t in run.turns
        if (t["action"]["type"] == "write_section" and not t["result"].get("ok"))
        or (t["action"]["type"] == "claim_resource" and t["result"].get("granted", 1) == 0)
    }

    causes = {}
    for s in range(settings.N_SECTIONS):
        a_val, b_val = run.fragments[s]
        content = (final_doc[s] or "").lower()
        if a_val.lower() in content and b_val.lower() in content:
            continue  # correct

        if _was_correct(run, s):
            causes[s] = "overwritten"
            continue
        if not knowers[s]:
            causes[s] = "not_shared" if (a_val not in shared or b_val not in shared) else "not_combined"
            continue
        writes = [t for t in run.turns
                  if t["action"]["type"] == "write_section" and t["result"].get("section") == s]
        if writes and writes[-1]["agent_id"] in knowers[s]:
            causes[s] = "misreported"
        elif any(a in died or a in broke for a in knowers[s]):
            causes[s] = "resource_blocked"
        else:
            causes[s] = "not_written"

    n_turns = len(run.turns) or 1
    model_error_rate = sum(not t["valid_json"] for t in run.turns) / n_turns
    unsupported = sum(  # writes by agents who did not have both halves (guessing)
        t["action"]["type"] == "write_section" and t["result"].get("ok")
        and t["agent_id"] not in knowers[t["action"]["section"]]
        for t in run.turns
    )
    counts = Counter(causes.values())
    return {
        "section_causes": causes,
        "cause_counts": dict(counts),
        "primary_cause": counts.most_common(1)[0][0] if counts else "none",
        "model_error_rate": model_error_rate,
        "high_model_error": model_error_rate > HIGH_MODEL_ERROR,
        "unsupported_writes": unsupported,
    }