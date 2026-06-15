#!/usr/bin/env python3
"""
Redrob AI — Intelligent Candidate Discovery & Ranking System
=============================================================

Main entry point for the ranking pipeline.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Architecture:
    1. Load candidates from JSONL
    2. Detect and flag honeypot candidates
    3. Score each candidate across 10 dimensions:
       - Template Fingerprint
       - Title & Career Fit
       - Career-JD Semantic Similarity
       - Career Trajectory
       - Assessment Scores
       - Experience Band
       - Skills Match
       - Location & Logistics
       - Skills-Career Cross-Validation
       - Education
    4. Apply behavioral signal modifier (0.4-1.3x)
    5. Apply penalty adjustments (keyword stuffing, title-skill mismatch)
    6. Rank, select top 100, generate reasoning
    7. Output submission CSV

Constraints:
    - Runtime: ≤ 5 minutes wall-clock
    - Memory: ≤ 16 GB RAM
    - Compute: CPU only
    - Network: None (fully offline)
"""

import argparse
import os
import sys
import time

from src.loader import load_candidates
from src.ranker import rank_candidates, write_submission_csv


def main():
    parser = argparse.ArgumentParser(
        description="Redrob AI Candidate Ranking System — "
                    "Ranks candidates for the Senior AI Engineer JD"
    )
    parser.add_argument(
        "--candidates",
        type=str,
        required=True,
        help="Path to candidates.jsonl or candidates.jsonl.gz",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./submission.csv",
        help="Output path for the submission CSV (default: ./submission.csv)",
    )
    parser.add_argument(
        "--semantic-mode",
        choices=("fast", "tfidf", "embeddings"),
        default="fast",
        help=(
            "Semantic scorer: fast structured evidence (default), sklearn TF-IDF, "
            "or optional pre-computed embeddings"
        ),
    )
    args = parser.parse_args()
    os.environ["REDROB_SEMANTIC_MODE"] = args.semantic_mode

    print("=" * 70)
    print("  Redrob AI — Intelligent Candidate Ranking System")
    print("  Senior AI Engineer — Founding Team")
    print("=" * 70)

    overall_start = time.time()

    # --- Step 1: Load candidates ---
    print(f"\n[1/3] Loading candidates from {args.candidates}...")
    candidates = load_candidates(args.candidates)

    if not candidates:
        print("ERROR: No candidates loaded. Check the file path.")
        sys.exit(1)

    # --- Step 2: Score and rank ---
    print(f"\n[2/3] Scoring and ranking {len(candidates):,} candidates...")
    results = rank_candidates(candidates)

    # --- Step 3: Write output ---
    print(f"\n[3/3] Writing submission to {args.out}...")
    write_submission_csv(results, args.out)

    # --- Summary ---
    overall_elapsed = time.time() - overall_start
    print(f"\n{'=' * 70}")
    print(f"  COMPLETE — Total runtime: {overall_elapsed:.1f}s")
    print(f"  Candidates processed: {len(candidates):,}")
    print(f"  Top 100 written to: {args.out}")
    print(f"{'=' * 70}")

    # Print top 5 for quick inspection
    print(f"\n  Top 5 candidates:")
    for r in results[:5]:
        cid = r["candidate_id"]
        score = r["score"]
        details = r["details"]
        profile = details["candidate"]["profile"]
        print(
            f"    #{r['rank']} {cid}: "
            f"{profile.get('current_title', '?')} at "
            f"{profile.get('current_company', '?')} "
            f"({profile.get('years_of_experience', 0):.1f}y) — "
            f"score={score:.4f}"
        )

    if overall_elapsed > 300:
        print(f"\n  [!] WARNING: Runtime ({overall_elapsed:.1f}s) exceeds 5-minute limit!")
    else:
        print(f"\n  [OK] Runtime ({overall_elapsed:.1f}s) within 5-minute budget")


if __name__ == "__main__":
    main()
