#!/usr/bin/env python3
"""Audit a Redrob submission without pretending to know hidden ground truth."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from src.behavioral import compute_behavioral_modifier
from src.integrity_scorer import compute_integrity_score
from src.template_fingerprinter import fingerprint_career
from validate_submission import validate_submission


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default="submission.csv")
    parser.add_argument("--candidates", required=True)
    args = parser.parse_args()

    errors = validate_submission(args.submission)
    if errors:
        print("FORMAT: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    with open(args.submission, encoding="utf-8", newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: int(row["rank"]))
    wanted = {row["candidate_id"] for row in rows}

    candidates = {}
    with open(args.candidates, encoding="utf-8") as handle:
        for line in handle:
            candidate = json.loads(line)
            if candidate["candidate_id"] in wanted:
                candidates[candidate["candidate_id"]] = candidate

    missing = sorted(wanted - candidates.keys())
    if missing:
        print(f"CANDIDATE IDS: FAIL ({len(missing)} IDs not in dataset)")
        return 1

    ranked = [candidates[row["candidate_id"]] for row in rows]
    print("FORMAT: PASS")
    for cutoff in (10, 50, 100):
        tier_counts = Counter(
            fingerprint_career(candidate.get("career_history", []))[0]
            for candidate in ranked[:cutoff]
        )
        print(f"TOP {cutoff} TEMPLATE TIERS: {dict(sorted(tier_counts.items(), reverse=True))}")

    honeypots = []
    unavailable = []
    for row, candidate in zip(rows, ranked):
        integrity, is_honeypot, reasons = compute_integrity_score(candidate)
        behavior, details = compute_behavioral_modifier(candidate.get("redrob_signals", {}))
        if is_honeypot:
            honeypots.append((row["rank"], row["candidate_id"], integrity, reasons))
        if behavior <= 0.74:
            unavailable.append((row["rank"], row["candidate_id"], behavior, details["days_since_active"]))

    reasonings = [row["reasoning"].strip() for row in rows]
    print(f"HONEYPOTS IN TOP 100: {len(honeypots)}")
    print(f"VERY LOW AVAILABILITY IN TOP 100: {len(unavailable)}")
    print(f"UNIQUE REASONINGS: {len(set(reasonings))}/100")
    print(f"AVERAGE REASONING LENGTH: {sum(map(len, reasonings)) / len(reasonings):.1f} chars")

    if honeypots:
        for item in honeypots:
            print(f"- honeypot rank={item[0]} id={item[1]} reasons={item[3]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
