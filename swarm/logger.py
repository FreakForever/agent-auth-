"""JSONL logger: one file per run, one validated record per line.

Every line is a Pydantic record from swarm/config.py, so anything that reads
the logs later (metrics, dashboard) can trust the shape of the data.
"""
from __future__ import annotations

import json
import os
from typing import Iterator

from pydantic import BaseModel

from swarm.config import parse_record


class JsonlLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._f = open(path, "w", encoding="utf-8")

    def write(self, record: BaseModel) -> None:
        self._f.write(record.model_dump_json() + "\n")
        self._f.flush()  # a crashed run still leaves a readable partial log

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class NullLogger:
    """Drop-in logger that writes nothing (handy in quick tests)."""

    def write(self, record: BaseModel) -> None:
        pass

    def close(self) -> None:
        pass


def read_log(path: str) -> Iterator:
    """Yield validated records from a JSONL log."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield parse_record(line)


def read_log_dicts(path: str) -> list[dict]:
    """Raw dicts, for pandas."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]