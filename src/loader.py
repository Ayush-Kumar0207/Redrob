"""
Candidate Loader for the Redrob AI Candidate Ranking System.

Loads candidates from JSONL files (plain or gzipped).
Handles streaming for memory efficiency and progress reporting.
"""

from __future__ import annotations
import gzip
import json
import time
from pathlib import Path
from typing import Generator, List


def load_candidates(filepath: str) -> List[dict]:
    """
    Load all candidates from a JSONL file into memory.

    Supports JSON arrays, plain .jsonl, and gzipped .jsonl.gz files.
    For 100K candidates (~465MB), this uses ~1-2GB RAM which
    is well within the 16GB budget.

    Args:
        filepath: Path to candidates.jsonl or candidates.jsonl.gz

    Returns:
        List of candidate dicts
    """
    path = Path(filepath)
    candidates = []
    start = time.time()

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        candidates = payload if isinstance(payload, list) else [payload]
        elapsed = time.time() - start
        print(f"  Loaded {len(candidates):,} candidates total ({elapsed:.1f}s)")
        return candidates

    if path.suffix == ".gz":
        opener = gzip.open(path, "rt", encoding="utf-8")
    else:
        opener = open(path, "r", encoding="utf-8")

    with opener as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
                candidates.append(candidate)
            except json.JSONDecodeError as e:
                print(f"  [WARN] Skipping malformed JSON at line {i + 1}: {e}")

            if (i + 1) % 10000 == 0:
                elapsed = time.time() - start
                print(f"  Loaded {i + 1:,} candidates ({elapsed:.1f}s)")

    elapsed = time.time() - start
    print(f"  Loaded {len(candidates):,} candidates total ({elapsed:.1f}s)")
    return candidates


def stream_candidates(filepath: str) -> Generator[dict, None, None]:
    """
    Stream candidates one at a time from a JSONL file.
    Use this if memory is a concern.
    """
    path = Path(filepath)

    if path.suffix == ".gz":
        opener = gzip.open(path, "rt", encoding="utf-8")
    else:
        opener = open(path, "r", encoding="utf-8")

    with opener as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
