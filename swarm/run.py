"""Run one condition (one YAML config) for all its seeds.

    python -m swarm.run configs/pilot/collab_mesh.yaml --log-dir logs/pilot --parallel 3

- Each seed is one run, written to <log-dir>/<run_id>.jsonl
- Seeds can run in parallel threads. All of them talk to the same vLLM server,
  which batches their requests, so 3 parallel runs ~ 18 agents per GPU batch.
- Finished runs (log ends with run_end) are skipped, so a crashed launch can resume.
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from swarm import settings
from swarm.config import ExperimentConfig, TaskMode
from swarm.logger import JsonlLogger
from swarm.world import World


def run_id_for(cfg: ExperimentConfig, seed: int) -> str:
    return f"{cfg.name}__s{seed}__{cfg.config_hash()}"


def is_complete(path: str) -> bool:
    if not os.path.exists(path):
        return False
    with open(path, "rb") as f:
        try:
            f.seek(-2000, os.SEEK_END)
        except OSError:
            f.seek(0)
        tail = f.read().decode("utf-8", "ignore").strip().splitlines()
    return bool(tail) and '"event":"run_end"' in tail[-1].replace(" ", "")


def _default_pool(cfg, seed):
    from swarm.agents.llm_agent import LLMPool  # imported lazily: bots don't need openai
    pool = LLMPool(cfg, seed)
    pool.preflight()
    return pool


def run_one(cfg: ExperimentConfig, seed: int, log_dir: str,
            pool_factory: Optional[Callable] = None):
    if cfg.task_mode != TaskMode.normal:   # Phase 9 secret-passphrase task
        from swarm.secret_task import SecretEpisode, secret_pool
        pool = (pool_factory or secret_pool)(cfg, seed)
        world = SecretEpisode(cfg, seed, pool, None)
    else:
        pool = (pool_factory or _default_pool)(cfg, seed)
        world = World(cfg, seed, pool, None)
    path = os.path.join(log_dir, f"{world.run_id}.jsonl")
    try:
        with JsonlLogger(path) as lg:
            world.log = lg
            end = world.run()
    finally:
        if hasattr(pool, "close"):
            pool.close()
    return path, end


def run_condition(cfg: ExperimentConfig, log_dir: str = settings.LOG_DIR, parallel: int = 1,
                  pool_factory: Optional[Callable] = None, skip_existing: bool = True,
                  verbose: bool = True) -> list[str]:
    os.makedirs(log_dir, exist_ok=True)
    todo = []
    for seed in cfg.seeds:
        path = os.path.join(log_dir, f"{run_id_for(cfg, seed)}.jsonl")
        if skip_existing and is_complete(path):
            if verbose:
                print(f"[skip] {os.path.basename(path)} already complete")
            continue
        todo.append(seed)

    paths = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
        futs = {ex.submit(run_one, cfg, s, log_dir, pool_factory): s for s in todo}
        for fut in as_completed(futs):
            seed = futs[fut]
            try:
                path, end = fut.result()
                paths.append(path)
                if verbose:
                    print(f"[done] {cfg.name} seed={seed} outcome={end.outcome.value} "
                          f"turns={end.turns} facts={end.facts_correct}/6 "
                          f"invalid={end.invalid_json_count}  ({time.perf_counter() - t0:.0f}s)")
            except Exception as e:  # one bad run shouldn't kill the condition
                print(f"[FAIL] {cfg.name} seed={seed}: {type(e).__name__}: {e}")
    return paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("configs", nargs="+", help="one or more YAML config files")
    p.add_argument("--log-dir", default=settings.LOG_DIR)
    p.add_argument("--parallel", type=int, default=1, help="seeds run concurrently")
    p.add_argument("--no-skip", action="store_true", help="re-run even if a log is complete")
    a = p.parse_args()
    for path in a.configs:
        cfg = ExperimentConfig.from_yaml(path)
        print(f"=== {cfg.name}  ({path})  seeds={cfg.seeds}  hash={cfg.config_hash()}")
        run_condition(cfg, a.log_dir, a.parallel, skip_existing=not a.no_skip)


if __name__ == "__main__":
    main()