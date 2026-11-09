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
from swarm.metrics import Run, knew_both, knowledge, knowledge_timeline, shared_values

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
    timeline = knowledge_timeline(run)
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
        if writes and knew_both(timeline, writes[-1]["agent_id"], run.fragments[s], writes[-1]["round"]):
            causes[s] = "misreported"   # the writer had both halves at that moment, and still got it wrong
        elif all(a in died or a in broke for a in knowers[s]):
            # only if EVERY agent who could have written it was blocked; if any knower was
            # alive and never short of credits, the failure is behavioural (not_written)
            causes[s] = "resource_blocked"
        else:
            causes[s] = "not_written"

    n_turns = len(run.turns) or 1
    model_error_rate = sum(not t["valid_json"] for t in run.turns) / n_turns
    unsupported = sum(  # writes by agents who did not have both halves AT THAT ROUND (guessing)
        t["action"]["type"] == "write_section" and t["result"].get("ok")
        and not knew_both(timeline, t["agent_id"], run.fragments[t["action"]["section"]], t["round"])
        for t in run.turns
    )
    # correct writes split by where the code came from:
    #   guessed_correct: writer did not have both halves at that round (a lucky guess; should be 0)
    #   relayed_correct: writer had both halves but did not start with both, so it learned at
    #                    least one from others (messages or the shared document)
    guessed_correct = relayed_correct = 0
    for t in run.turns:
        if t["action"]["type"] != "write_section" or not t["result"].get("correct"):
            continue
        s = t["action"]["section"]
        if not knew_both(timeline, t["agent_id"], run.fragments[s], t["round"]):
            guessed_correct += 1
        elif t["agent_id"] not in run.holder[s] or len(set(run.holder[s])) > 1:
            relayed_correct += 1
    counts = Counter(causes.values())
    return {
        "section_causes": causes,
        "cause_counts": dict(counts),
        "primary_cause": counts.most_common(1)[0][0] if counts else "none",
        "model_error_rate": model_error_rate,
        "high_model_error": model_error_rate > HIGH_MODEL_ERROR,
        "unsupported_writes": unsupported,
        "guessed_correct": guessed_correct,
        "relayed_correct": relayed_correct,
    }