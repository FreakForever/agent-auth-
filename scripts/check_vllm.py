"""Phase 0 exit check: can the served model return valid JSON for 6 agents at once?

Usage:
    python scripts/check_vllm.py            # 6 concurrent calls, JSON mode on
    python scripts/check_vllm.py --no-json-mode
    python scripts/check_vllm.py --n 12     # stress test

Pass criteria: every response parses as JSON and contains "thought" and "action".
"""
import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import AsyncOpenAI  # noqa: E402

from swarm import settings  # noqa: E402

SYSTEM = (
    "You are agent {aid} in a team of 6 building a shared document about Project Lumen. "
    "You privately know this fact for section {sec}: '{fact}'. "
    "Respond with ONLY a JSON object of the form "
    '{{"thought": "<one short sentence>", "action": {{"type": "<send_message|write_section|pass>", ...}}}}. '
    'For send_message include "to" (agent id) and "text". For write_section include "section" and "content".'
)
USER = "Round 1. Credits: 3. Pool: 30. No messages yet. Choose one action."

FACTS = ["LUM-4471", "Saffron Hall", "17 March", "Dr. Ilse Varga", "cobalt-9", "Gate 12"]


async def one_call(client, aid, json_mode):
    kwargs = dict(
        model=settings.MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM.format(aid=aid, sec=aid, fact=FACTS[aid])},
            {"role": "user", "content": USER},
        ],
        temperature=settings.TEMPERATURE,
        max_tokens=settings.MAX_TOKENS,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    t0 = time.perf_counter()
    resp = await client.chat.completions.create(**kwargs)
    dt = time.perf_counter() - t0
    text = resp.choices[0].message.content or ""
    try:
        obj = json.loads(text)
        ok = isinstance(obj, dict) and "thought" in obj and "action" in obj
    except json.JSONDecodeError:
        obj, ok = None, False
    return aid, ok, dt, resp.usage, text


async def main(n, json_mode):
    client = AsyncOpenAI(
        base_url=settings.VLLM_BASE_URL,
        api_key=settings.VLLM_API_KEY,
        timeout=settings.REQUEST_TIMEOUT_S,
    )
    print(f"Server: {settings.VLLM_BASE_URL}  model: {settings.MODEL_NAME}  json_mode: {json_mode}")
    models = await client.models.list()
    print("Served models:", [m.id for m in models.data])

    t0 = time.perf_counter()
    results = await asyncio.gather(*(one_call(client, i % 6, json_mode) for i in range(n)))
    wall = time.perf_counter() - t0

    n_ok = 0
    for aid, ok, dt, usage, text in results:
        n_ok += ok
        tag = "OK  " if ok else "FAIL"
        toks = f"{usage.prompt_tokens}+{usage.completion_tokens} tok" if usage else ""
        print(f"[{tag}] agent {aid}  {dt:5.2f}s  {toks}  {text[:120]!r}")

    print(f"\n{n_ok}/{n} valid JSON. Wall time for the batch: {wall:.2f}s")
    print("PASS" if n_ok == n else "FAIL: inspect the outputs above")
    return 0 if n_ok == n else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--no-json-mode", action="store_true")
    a = p.parse_args()
    sys.exit(asyncio.run(main(a.n, not a.no_json_mode)))
