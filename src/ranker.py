"""
Final Ranker for the Redrob AI Candidate Ranking System.

Combines core scoring, behavioral modifiers, and penalty adjustments
to produce the final top-100 ranked list.
"""

from __future__ import annotations
import csv
import os
import time
from typing import List, Tuple, Dict

from src.scorer import score_candidate
from src.behavioral import compute_behavioral_modifier
from src.integrity_scorer import compute_integrity_score
from src.reasoning import generate_reasoning


def rank_candidates(candidates: List[dict], use_cross_encoder: bool | None = None) -> List[Dict]:
    """
    Score and rank all candidates, returning the top 100.

    Pipeline:
      1. Calibrated multi-dimensional scoring (10 dimensions)
      2. Behavioral modifier
      3. Integrity scoring (timeline/skill impossibility)
      4. Penalty adjustments (salary, stuffing, traps)
      5. Final score = core × behavioral × integrity × penalties
      6. Optional local Cross-Encoder re-ranking
      7. Sort, select top 100

    Returns:
        List of top 100 candidate results (dicts with ranking data)
    """
    if use_cross_encoder is None:
        use_cross_encoder = os.getenv("REDROB_ENABLE_CROSS_ENCODER", "0").lower() in {
            "1", "true", "yes", "on",
        }

    print(f"\n[Ranker] Scoring {len(candidates):,} candidates...")
    start = time.time()

    scored: List[Tuple[float, dict]] = []
    honeypot_count = 0

    for i, candidate in enumerate(candidates):
        cid = candidate.get("candidate_id", "UNKNOWN")

        # --- Step 1: Integrity scoring (replaces binary honeypot) ---
        integrity_score, is_honeypot, integrity_reasons = compute_integrity_score(
            candidate
        )
        if is_honeypot:
            honeypot_count += 1

        # --- Step 2: Core scoring ---
        score_details = score_candidate(candidate)
        core_score = score_details["composite_score"]

        # --- Step 3: Behavioral modifier ---
        signals = candidate.get("redrob_signals", {})
        behavioral_mod, behavioral_details = compute_behavioral_modifier(signals)

        # --- Step 4: Penalty adjustments ---
        penalty = _compute_penalties(candidate, score_details)

        # --- Step 5: Consistency boost ---
        # Candidates scoring well across ALL dimensions are qualitatively
        # better than those who are great at one thing but weak at others.
        consistency_boost = _compute_consistency_boost(score_details)

        # --- Step 6: Final score ---
        # Integrity score acts as a soft multiplier (0.0-1.0)
        final_score = core_score * behavioral_mod * integrity_score * penalty * consistency_boost

        scored.append((final_score, {
            "candidate": candidate,
            "candidate_id": cid,
            "final_score": final_score,
            "core_score": core_score,
            "behavioral_modifier": behavioral_mod,
            "integrity_score": integrity_score,
            "penalty": penalty,
            "is_honeypot": is_honeypot,
            "integrity_reasons": integrity_reasons,
            "score_details": score_details,
            "behavioral_details": behavioral_details,
        }))

        if (i + 1) % 10000 == 0:
            elapsed = time.time() - start
            print(f"  Scored {i + 1:,} candidates ({elapsed:.1f}s)")

    elapsed = time.time() - start
    print(f"  Scored all {len(candidates):,} candidates ({elapsed:.1f}s)")
    print(f"  Honeypots detected: {honeypot_count}")

    # --- Stage 1: Sort and select top 150 ---
    scored.sort(key=lambda x: (-x[0], x[1]["candidate_id"]))  # Desc score, asc ID for tiebreak
    top_150 = scored[:150]
    
    # --- Stage 2: Cross-Encoder Re-ranking ---
    if use_cross_encoder:
        print(f"\n[Ranker] Stage 2: Re-ranking top 150 with Cross-Encoder...")
        stage2_start = time.time()
        try:
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

            from sentence_transformers import CrossEncoder
            # Use a lightweight cross-encoder model. With offline env vars set,
            # this only works when the model is already cached locally.
            cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)

            # Build JD string for comparison
            jd_text = (
                "Senior AI Engineer founding team. Ranking retrieval matching systems "
                "embeddings NLP Vector DBs Qdrant Pinecone product company experience"
            )

            cross_inputs = []
            for _, data in top_150:
                cand_text = _build_cross_encoder_candidate_text(data["candidate"])
                cross_inputs.append((jd_text, cand_text))

            # Predict cross-scores
            cross_scores = cross_encoder.predict(cross_inputs)

            # Normalize cross-scores to 0-1
            min_c = min(cross_scores)
            max_c = max(cross_scores)
            if max_c > min_c:
                norm_cross_scores = [(s - min_c) / (max_c - min_c) for s in cross_scores]
            else:
                norm_cross_scores = [0.5 for _ in cross_scores]

            # Combine scores
            for idx, (stage1_score, data) in enumerate(top_150):
                cross_s = norm_cross_scores[idx]
                final_stage2_score = (0.75 * stage1_score) + (0.25 * cross_s)
                data["cross_encoder_score"] = round(cross_s, 4)
                data["final_score"] = final_stage2_score  # update to new final score
                top_150[idx] = (final_stage2_score, data)

            print(f"  Cross-Encoder finished in {time.time() - stage2_start:.1f}s")
        except Exception as e:
            print(f"  [!] Stage 2 failed, falling back to Stage 1. Error: {e}")
    else:
        print(
            "\n[Ranker] Stage 2: Cross-Encoder disabled "
            "(set REDROB_ENABLE_CROSS_ENCODER=1 to use a cached local model)."
        )
    
    # Re-sort after Stage 2
    top_150.sort(key=lambda x: (-x[0], x[1]["candidate_id"]))
    top_100 = top_150[:100]

    # --- Step 7: Generate reasoning for top 100 ---
    print(f"\n[Ranker] Generating reasoning for top 100...")
    results = []
    for rank, (score, data) in enumerate(top_100, 1):
        reasoning = generate_reasoning(
            candidate=data["candidate"],
            rank=rank,
            score_details=data["score_details"],
            behavioral_details=data["behavioral_details"],
            is_honeypot=data["is_honeypot"],
        )
        if "cross_encoder_score" in data:
            reasoning += f" [Stage 2 Cross-Score: {data['cross_encoder_score']}]"
            
        results.append({
            "candidate_id": data["candidate_id"],
            "rank": rank,
            "score": round(score, 4),
            "reasoning": reasoning,
            "details": data,
        })

    # Check honeypot rate in top 100
    hp_in_top = sum(1 for r in results if r["details"]["is_honeypot"])
    print(f"  Honeypots in top 100: {hp_in_top} ({hp_in_top}%)")
    if hp_in_top > 10:
        print("  [!] WARNING: Honeypot rate > 10%!")

    total_elapsed = time.time() - start
    print(f"\n[Ranker] Complete in {total_elapsed:.1f}s")

    return results


def _build_cross_encoder_candidate_text(candidate: dict) -> str:
    """Build the candidate text used by optional Stage 2 re-ranking."""
    profile = candidate.get("profile", {})
    parts = [
        profile.get("current_title", ""),
        profile.get("headline", ""),
        profile.get("summary", ""),
    ]

    skills = [
        s.get("name", "") if isinstance(s, dict) else str(s)
        for s in candidate.get("skills", [])
    ]
    if skills:
        parts.append("Skills: " + ", ".join(skills))

    for role in candidate.get("career_history", []):
        role_bits = []
        if role.get("title"):
            role_bits.append(role["title"])
        if role.get("company"):
            role_bits.append("at " + role["company"])
        if role.get("industry"):
            role_bits.append("(" + role["industry"] + ")")
        role_text = " ".join(role_bits)
        if role.get("description"):
            role_text += ": " + role["description"]
        if role_text:
            parts.append(role_text)

    return " ".join(part for part in parts if part)[:4000]


def _ensemble_composite(score_details: dict) -> float:
    """
    Backward-compatible alias for the calibrated composite.

    Older development scripts imported this helper. The production ranker now
    uses the single auditable weight vector in ``src.scorer.WEIGHTS`` directly;
    averaging several linear weightings is still just another linear weighting.
    """
    return float(score_details.get("composite_score", 0.0))


def _compute_penalties(candidate: dict, score_details: dict) -> float:
    """
    Compute penalty/bonus multiplier for edge cases.
    Returns a value in [0.3, 1.05].
    """
    penalty = 1.0
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    # --- Salary band alignment (JD: 35-55 LPA) ---
    salary = signals.get("expected_salary_range_inr_lpa", {})
    yoe = profile.get("years_of_experience", 0)
    min_salary = salary.get("min", 0)
    max_salary = salary.get("max", 0)

    # Salary is a secondary logistics signal, not a relevance shortcut.
    if min_salary >= 25 and max_salary <= 80:
        penalty *= 1.02
    elif min_salary >= 15 and max_salary <= 60:
        penalty *= 1.01

    # Absurd salary for experience level
    if yoe < 3 and max_salary > 80:
        penalty *= 0.85
    elif yoe < 5 and max_salary > 150:
        penalty *= 0.85

    # --- Industry alignment ---
    industry = profile.get("current_industry", "").lower()
    # Product-company industries that align with JD
    ai_product_industries = {
        "ai/ml", "saas", "e-commerce", "fintech", "edtech",
        "adtech", "healthtech ai", "conversational ai",
    }
    neutral_industries = {
        "software", "food delivery", "transportation",
        "insurance tech",
    }
    nonfit_industries = {
        "manufacturing", "conglomerate", "paper products",
    }

    if any(ind in industry for ind in ai_product_industries):
        penalty *= 1.02
    elif any(ind in industry for ind in nonfit_industries):
        penalty *= 0.90  # Manufacturing/conglomerate penalty
    elif industry == "it services":
        # IT services = consulting. JD explicitly says consulting-only is a disqualifier.
        # But having product experience can redeem them (checked elsewhere)
        penalty *= 0.90

    # --- Keyword stuffing penalty ---
    skills_details = score_details.get("skills_details", {})
    stuffing_risk = skills_details.get("stuffing_risk", 0)
    if stuffing_risk > 0.5:
        penalty *= 0.7
    elif stuffing_risk > 0.3:
        penalty *= 0.85

    # --- Title vs skills mismatch trap ---
    title_details = score_details.get("title_details", {})
    title_tier = title_details.get("title_tier", 3)
    core_count = skills_details.get("category_counts", {}).get("core_ai", 0)

    if title_tier <= 1 and core_count >= 5:
        penalty *= 0.4

    return max(0.3, min(1.05, penalty))


def _compute_consistency_boost(score_details: dict) -> float:
    """
    Reward candidates who score well across MULTIPLE dimensions.

    A candidate who scores 0.7+ on 6+ dimensions is qualitatively
    better than one who scores 0.95 on 2 dimensions and 0.3 on 4.
    The weighted sum alone doesn't capture this.

    Returns a multiplier in [0.95, 1.15].
    """
    # Core dimension scores
    dims = [
        score_details.get("template_score", 0),
        score_details.get("title_career_score", 0),
        score_details.get("skills_score", 0),
        score_details.get("career_jd_score", 0),
        score_details.get("cross_val_score", 0),
        score_details.get("assessment_score", 0),
        score_details.get("experience_score", 0),
        score_details.get("trajectory_score", 0),
    ]

    # Count dimensions scoring above thresholds
    above_07 = sum(1 for d in dims if d >= 0.7)
    above_05 = sum(1 for d in dims if d >= 0.5)
    below_03 = sum(1 for d in dims if d < 0.3)

    boost = 1.0

    # Multi-dimensional excellence boost
    if above_07 >= 6:
        boost = 1.12  # Exceptional across the board
    elif above_07 >= 5:
        boost = 1.08
    elif above_07 >= 4:
        boost = 1.04

    # Penalty for having any very weak dimension
    if below_03 >= 2:
        boost *= 0.95  # Multiple critical weaknesses

    return boost


def write_submission_csv(results: List[Dict], output_path: str):
    """
    Write the top-100 results to a CSV file matching the submission spec.

    Requirements from validate_submission.py:
    - Header: candidate_id,rank,score,reasoning
    - 100 data rows
    - Scores must be non-increasing
    - Ties must have ascending candidate_id
    - candidate_id format: CAND_XXXXXXX
    """
    # Normalize scores to 0.2-0.998 range
    scores = [r["score"] for r in results]
    max_score = max(scores) if scores else 1.0
    min_score = min(scores) if scores else 0.0
    score_range = max_score - min_score if max_score != min_score else 1.0

    # Build normalized scores
    normalized_results = []
    for r in results:
        norm = 0.2 + 0.798 * (r["score"] - min_score) / score_range
        normalized_results.append({
            "candidate_id": r["candidate_id"],
            "rank": r["rank"],
            "score": round(norm, 4),
            "reasoning": r["reasoning"],
        })

    # Enforce strictly non-increasing scores with proper tie-breaking.
    # When scores are equal after rounding, candidate_id must be ascending.
    # Strategy: sort by (-score, candidate_id), then assign strictly
    # decreasing scores using 0.0001 decrements for ties.

    # First: stable sort by (-score, candidate_id)
    normalized_results.sort(key=lambda x: (-x["score"], x["candidate_id"]))

    # Re-assign ranks after sort
    for idx, r in enumerate(normalized_results):
        r["rank"] = idx + 1

    # Ensure strictly decreasing scores (no ties in output)
    for i in range(1, len(normalized_results)):
        if normalized_results[i]["score"] >= normalized_results[i - 1]["score"]:
            normalized_results[i]["score"] = round(
                normalized_results[i - 1]["score"] - 0.0001, 4
            )

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for r in normalized_results:
            # Clean reasoning for CSV
            reasoning = r["reasoning"].replace('"', "'")

            writer.writerow([
                r["candidate_id"],
                r["rank"],
                f"{r['score']:.4f}",
                reasoning,
            ])

    print(f"\n[Output] Submission written to {output_path}")
    print(f"  Top candidate: {results[0]['candidate_id']} (score: {results[0]['score']:.4f})")
    print(f"  Bottom candidate: {results[-1]['candidate_id']} (score: {results[-1]['score']:.4f})")
