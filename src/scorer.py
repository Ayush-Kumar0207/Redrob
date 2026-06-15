"""
Multi-Dimensional Scorer for the Redrob AI Candidate Ranking System.

Computes a weighted composite score across 10 dimensions:
  1.  Title & Career Fit (0.13)
  2.  Skills Match (0.05)       -- REDUCED per JD warning
  3.  Career-JD Semantic Sim (0.10)  -- keyword + headline + summary
  4.  Skills-Career Cross-Val (0.05) -- detect keyword stuffers
  5.  Assessment Scores (0.08)  -- HARD EVIDENCE from platform tests
  6.  Experience Band (0.07)
  7.  Education & Background (0.03)
  8.  Location & Logistics (0.04)
  9.  Career Trajectory (0.09)
  10. Template Fingerprint (0.36) -- HIGHEST: synthetic relevance prior
"""

from __future__ import annotations
import os
from typing import Dict, List, Tuple

from src.skills_taxonomy import (
    compute_skill_relevance,
    is_cv_speech_only,
)
from src.company_classifier import analyze_career_companies
from src.title_matcher import (
    match_title,
    compute_career_title_trajectory,
)
from src.text_analyzer import analyze_career_descriptions, compute_fast_jd_evidence
from src.tfidf_scorer import (
    compute_career_jd_similarity,
    compute_skills_career_cross_validation,
    compute_assessment_score,
)
from src.template_fingerprinter import (
    fingerprint_career,
    template_tier_to_score,
)
from src.jd_parser import get_jd

# Dimension weights — 10 dimensions, sum = 1.0
# Template fingerprint has the HIGHEST weight because it directly
# maps to the synthetic data generation model's quality tiers.
WEIGHTS = {
    "template":         0.36,
    "title_career":     0.13,
    "career_jd_sim":    0.10,
    "trajectory":       0.09,
    "assessment":       0.08,
    "experience":       0.07,
    "skills":           0.05,
    "cross_validation": 0.05,
    "location":         0.04,
    "education":        0.03,
}

JD = get_jd()


def score_candidate(candidate: dict) -> Dict:
    """
    Score a single candidate across all 10 dimensions.

    Returns a dict with:
      - dimension scores (0-1 each)
      - composite_score (weighted sum, 0-1)
      - analysis details for reasoning generation
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])
    signals = candidate.get("redrob_signals", {})

    # Pre-compute shared analyses ONCE (performance optimization)
    company_analysis = analyze_career_companies(career)
    title_trajectory = compute_career_title_trajectory(career)
    desc_analysis = analyze_career_descriptions(career)

    # --- 1. Template Fingerprint (HIGHEST WEIGHT) ---
    template_tier, template_match = fingerprint_career(career)
    template_score = template_tier_to_score(template_tier)

    # --- 2. Title & Career Fit ---
    title_career_score, title_details = _score_title_career(
        profile, career, skills, company_analysis, title_trajectory, desc_analysis
    )

    # --- 3. Skills Match ---
    skills_score, skills_details = _score_skills(skills, signals)

    # --- 4. Career-JD Semantic Similarity ---
    # The deterministic lexical scorer is the reproducible default. Optional
    # pre-computed embeddings are used only when explicitly enabled and all
    # required artifacts are present.
    semantic_mode = os.getenv("REDROB_SEMANTIC_MODE", "fast").lower()
    embedding_paths = (
        "candidates_embeddings.npy",
        "candidates_ids.json",
        "jd_embedding.npy",
    )
    use_embeddings = (
        semantic_mode == "embeddings"
        and all(os.path.exists(path) for path in embedding_paths)
    )
    if use_embeddings:
        from src.embedding_scorer import compute_semantic_similarity

        career_jd_score, career_jd_details = compute_semantic_similarity(
            candidate.get("candidate_id")
        )
    elif semantic_mode == "tfidf":
        career_jd_score, career_jd_details = compute_career_jd_similarity(career, profile)
    else:
        career_jd_score, career_jd_details = compute_fast_jd_evidence(profile, desc_analysis)

    # --- 5. Skills-Career Cross-Validation ---
    cross_val_score, cross_val_details = compute_skills_career_cross_validation(
        skills, career
    )

    # --- 6. Assessment Scores (HARD EVIDENCE) ---
    assessment_score, assessment_details = compute_assessment_score(
        signals, skills
    )

    # --- 7. Experience Band ---
    experience_score, exp_details = _score_experience(profile)

    # --- 8. Education ---
    education_score, edu_details = _score_education(education)

    # --- 9. Location & Logistics ---
    location_score, loc_details = _score_location(profile, signals)

    # --- 10. Career Trajectory ---
    trajectory_score, traj_details = _score_trajectory(
        career, profile, company_analysis, title_trajectory
    )

    # --- Composite ---
    composite = (
        WEIGHTS["template"] * template_score
        + WEIGHTS["title_career"] * title_career_score
        + WEIGHTS["skills"] * skills_score
        + WEIGHTS["career_jd_sim"] * career_jd_score
        + WEIGHTS["cross_validation"] * cross_val_score
        + WEIGHTS["assessment"] * assessment_score
        + WEIGHTS["experience"] * experience_score
        + WEIGHTS["education"] * education_score
        + WEIGHTS["location"] * location_score
        + WEIGHTS["trajectory"] * trajectory_score
    )

    return {
        "composite_score": round(composite, 6),
        "template_score": round(template_score, 4),
        "template_tier": template_tier,
        "title_career_score": round(title_career_score, 4),
        "skills_score": round(skills_score, 4),
        "career_jd_score": round(career_jd_score, 4),
        "cross_val_score": round(cross_val_score, 4),
        "assessment_score": round(assessment_score, 4),
        "experience_score": round(experience_score, 4),
        "education_score": round(education_score, 4),
        "location_score": round(location_score, 4),
        "trajectory_score": round(trajectory_score, 4),
        # Detailed analysis for reasoning
        "template_details": {"tier": template_tier, "match": template_match},
        "title_details": title_details,
        "skills_details": skills_details,
        "career_jd_details": career_jd_details,
        "cross_val_details": cross_val_details,
        "assessment_details": assessment_details,
        "experience_details": exp_details,
        "education_details": edu_details,
        "location_details": loc_details,
        "trajectory_details": traj_details,
    }


# ===================================================================
# Dimension 1: Title & Career Fit (Weight: 0.30)
# ===================================================================

def _score_title_career(
    profile: dict, career: list, skills: list,
    company_analysis: dict, title_trajectory: dict, desc_analysis: dict,
) -> Tuple[float, dict]:
    """
    Score based on current title, career history at product companies,
    and evidence of ranking/search/retrieval work.
    """
    # Title tier (1-5)
    current_title = profile.get("current_title", "")
    title_tier, title_label = match_title(current_title)

    # Title score mapping (tier -> base score)
    tier_scores = {5: 1.0, 4: 0.75, 3: 0.50, 2: 0.25, 1: 0.05}
    title_base = tier_scores.get(title_tier, 0.25)

    # Boost if best-ever tier is higher than current
    if title_trajectory["best_tier"] > title_tier:
        title_base = max(
            title_base,
            (title_base + tier_scores.get(title_trajectory["best_tier"], 0)) / 2
        )

    # Company type modifier
    company_modifier = 1.0
    if company_analysis["consulting_only"]:
        # JD explicitly rejects consulting-only careers
        company_modifier = 0.3
    elif company_analysis["has_product_experience"]:
        # Bonus for product company experience
        product_ratio = company_analysis["product_months"] / max(
            company_analysis["total_months"], 1
        )
        company_modifier = 1.0 + (product_ratio * 0.2)  # Up to 1.2
    elif company_analysis["consulting_ratio"] > 0.7:
        company_modifier = 0.6

    # Description evidence modifier (very important per JD)
    desc_modifier = 1.0

    # Ranking/search experience is the strongest signal
    if desc_analysis["ranking_search_score"] > 0.3:
        desc_modifier += 0.3
    if desc_analysis["embeddings_nlp_score"] > 0.3:
        desc_modifier += 0.2
    if desc_analysis["production_score"] > 0.3:
        desc_modifier += 0.15
    if desc_analysis["ml_systems_score"] > 0.3:
        desc_modifier += 0.1

    # Penalize research-only and management-only
    if desc_analysis["research_only_score"] > 0.5 and not desc_analysis["is_production_focused"]:
        desc_modifier *= 0.6
    if desc_analysis["management_only_score"] > 0.5 and not desc_analysis["is_hands_on"]:
        desc_modifier *= 0.7

    # CV/Speech-only penalty (per JD anti-pattern)
    if is_cv_speech_only(skills):
        desc_modifier *= 0.5

    # Combine
    score = title_base * min(company_modifier, 1.5) * min(desc_modifier, 2.0)
    score = min(1.0, score)  # Cap at 1.0

    details = {
        "title_tier": title_tier,
        "title_label": title_label,
        "current_title": current_title,
        "company_analysis": company_analysis,
        "desc_analysis": {
            "production_score": desc_analysis["production_score"],
            "ranking_search_score": desc_analysis["ranking_search_score"],
            "embeddings_nlp_score": desc_analysis["embeddings_nlp_score"],
            "is_hands_on": desc_analysis["is_hands_on"],
            "ranking_evidence": desc_analysis["ranking_evidence"],
            "production_evidence": desc_analysis["production_evidence"],
        },
        "title_trajectory": title_trajectory,
    }

    return score, details


# ===================================================================
# Dimension 2: Skills Match (Weight: 0.25)
# ===================================================================

def _score_skills(skills: list, signals: dict) -> Tuple[float, dict]:
    """
    Score skills relevance with anti-keyword-stuffing checks.
    """
    assessment_scores = signals.get("skill_assessment_scores", {})

    skill_analysis = compute_skill_relevance(skills, assessment_scores)

    # Core AI skills are most important
    core_score = skill_analysis["core_ai_score"]  # 0-1
    supporting_score = skill_analysis["supporting_ai_score"]  # 0-1
    engineering_score = skill_analysis["engineering_score"]  # 0-1

    # Trust modifier
    trust = skill_analysis["skill_trust_score"]  # 0-1
    stuffing_risk = skill_analysis["keyword_stuffing_risk"]  # 0-1

    # Weighted skill score
    raw_skill_score = (
        0.55 * core_score
        + 0.25 * supporting_score
        + 0.20 * engineering_score
    )

    # Apply trust and anti-stuffing
    trust_modifier = 0.5 + (trust * 0.5)  # Range: 0.5-1.0
    stuffing_penalty = 1.0 - (stuffing_risk * 0.5)  # Range: 0.5-1.0

    score = raw_skill_score * trust_modifier * stuffing_penalty
    score = min(1.0, score)

    # Check for assessment score validation
    # If candidate claimed skills are validated by assessments → bonus
    if assessment_scores:
        avg_assessment = sum(assessment_scores.values()) / len(assessment_scores)
        if avg_assessment >= 60:
            score = min(1.0, score * 1.1)
        elif avg_assessment < 30:
            score *= 0.85  # Low assessment scores → reduce trust

    details = {
        "core_ai_score": core_score,
        "supporting_ai_score": supporting_score,
        "engineering_score": engineering_score,
        "trust_score": trust,
        "stuffing_risk": stuffing_risk,
        "matched_core": skill_analysis["matched_core_skills"],
        "matched_supporting": skill_analysis["matched_supporting_skills"],
        "category_counts": skill_analysis["category_counts"],
    }

    return round(score, 4), details


# ===================================================================
# Dimension 3: Experience Band (Weight: 0.15)
# ===================================================================

def _score_experience(profile: dict) -> Tuple[float, dict]:
    """
    Score based on years of experience vs JD ideal range.
    """
    yoe = profile.get("years_of_experience", 0)
    exp = JD["experience"]

    if exp["ideal_min"] <= yoe <= exp["ideal_max"]:
        score = 1.0
    elif exp["acceptable_min"] <= yoe < exp["ideal_min"]:
        # Slightly below ideal
        score = 0.7 + 0.3 * (yoe - exp["acceptable_min"]) / (
            exp["ideal_min"] - exp["acceptable_min"]
        )
    elif exp["ideal_max"] < yoe <= exp["acceptable_max"]:
        # Slightly above ideal
        score = 1.0 - 0.3 * (yoe - exp["ideal_max"]) / (
            exp["acceptable_max"] - exp["ideal_max"]
        )
    elif exp["hard_min"] <= yoe < exp["acceptable_min"]:
        score = 0.4
    elif yoe > exp["acceptable_max"]:
        # Over-experienced — still valuable but may be overqualified
        score = max(0.15, 0.7 - 0.05 * (yoe - exp["acceptable_max"]))
    else:
        score = 0.1

    details = {
        "years_of_experience": yoe,
        "ideal_range": f"{exp['ideal_min']}-{exp['ideal_max']}",
        "fit": "ideal" if exp["ideal_min"] <= yoe <= exp["ideal_max"]
               else "acceptable" if exp["acceptable_min"] <= yoe <= exp["acceptable_max"]
               else "outside",
    }

    return round(score, 4), details


# ===================================================================
# Dimension 4: Education & Background (Weight: 0.10)
# ===================================================================

# Relevant fields of study
_AI_ML_FIELDS = {
    "computer science", "cs", "artificial intelligence", "ai",
    "machine learning", "ml", "data science",
    "information technology", "it",
    "statistics", "applied mathematics", "mathematics",
    "computational linguistics", "nlp",
    "information systems",
}
_ENGINEERING_FIELDS = {
    "software engineering", "computer engineering",
    "electrical engineering", "electronics",
    "ece", "eee", "information science",
}
_NON_RELEVANT_FIELDS = {
    "mechanical engineering", "civil engineering",
    "chemical engineering", "industrial engineering",
    "architecture", "agriculture",
    "commerce", "business administration",
    "marketing", "finance", "accounting",
    "humanities", "arts", "sociology",
}

# Institution tier weights
_TIER_SCORES = {
    "tier_1": 1.0,   # IITs, IISc, top global
    "tier_2": 0.75,  # NITs, BITS, IIITs, good international
    "tier_3": 0.50,  # Decent private universities
    "tier_4": 0.30,  # Unknown / local colleges
    "unknown": 0.40,
}

# Degree level weights
_DEGREE_SCORES = {
    "ph.d": 0.9,  # PhD can indicate research focus, not always best
    "phd": 0.9,
    "m.tech": 0.85,
    "m.s": 0.8,
    "m.sc": 0.75,
    "m.e.": 0.8,
    "mba": 0.5,  # Not directly relevant
    "b.tech": 0.7,
    "b.e.": 0.7,
    "b.sc": 0.55,
    "b.s": 0.6,
    "b.com": 0.3,
    "b.a": 0.3,
    "diploma": 0.3,
}


def _score_education(education: list) -> Tuple[float, dict]:
    """Score based on educational background."""
    if not education:
        return 0.3, {"note": "No education listed"}

    best_score = 0.0
    best_entry = None

    for edu in education:
        field = edu.get("field_of_study", "").lower()
        degree = edu.get("degree", "").lower()
        tier = edu.get("tier", "unknown")
        institution = edu.get("institution", "")

        # Field relevance
        if field in _AI_ML_FIELDS or any(f in field for f in _AI_ML_FIELDS):
            field_score = 1.0
        elif field in _ENGINEERING_FIELDS or any(f in field for f in _ENGINEERING_FIELDS):
            field_score = 0.7
        elif field in _NON_RELEVANT_FIELDS or any(f in field for f in _NON_RELEVANT_FIELDS):
            field_score = 0.25
        else:
            field_score = 0.4

        # Degree level
        degree_score = 0.5  # default
        for deg_key, deg_val in _DEGREE_SCORES.items():
            if deg_key in degree:
                degree_score = deg_val
                break

        # Institution tier
        tier_score = _TIER_SCORES.get(tier, 0.4)

        # Combined (field is most important, then degree, then tier)
        edu_score = 0.45 * field_score + 0.30 * degree_score + 0.25 * tier_score

        if edu_score > best_score:
            best_score = edu_score
            best_entry = {
                "institution": institution,
                "degree": edu.get("degree", ""),
                "field": edu.get("field_of_study", ""),
                "tier": tier,
            }

    return round(best_score, 4), {"best_education": best_entry}


# ===================================================================
# Dimension 5: Location & Logistics (Weight: 0.10)
# ===================================================================

def _score_location(profile: dict, signals: dict) -> Tuple[float, dict]:
    """Score based on location, notice period, and work mode."""
    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    notice_days = signals.get("notice_period_days", 90)
    work_mode = signals.get("preferred_work_mode", "onsite")
    willing_relocate = signals.get("willing_to_relocate", False)

    jd_loc = JD["location"]

    # --- Location score ---
    if country == "india":
        # Check if in preferred cities
        in_preferred = any(
            city in location for city in jd_loc["preferred_cities"]
        )
        if in_preferred:
            loc_score = 1.0
        elif willing_relocate:
            loc_score = 0.8
        else:
            loc_score = 0.6
    else:
        # Outside India
        if willing_relocate:
            loc_score = 0.4
        else:
            loc_score = 0.2

    # --- Notice period score ---
    if notice_days <= 30:
        notice_score = 1.0
    elif notice_days <= 60:
        notice_score = 0.75
    elif notice_days <= 90:
        notice_score = 0.5
    else:
        notice_score = 0.25

    # --- Work mode compatibility ---
    # JD is Hybrid — flexible cadence
    work_mode_scores = {
        "hybrid": 1.0,
        "flexible": 1.0,
        "onsite": 0.85,
        "remote": 0.7,
    }
    work_score = work_mode_scores.get(work_mode, 0.7)

    # Combined location score
    score = 0.50 * loc_score + 0.35 * notice_score + 0.15 * work_score

    details = {
        "location": profile.get("location", ""),
        "country": country,
        "in_india": country == "india",
        "in_preferred_city": country == "india" and any(
            city in location for city in jd_loc["preferred_cities"]
        ),
        "notice_period_days": notice_days,
        "work_mode": work_mode,
        "willing_to_relocate": willing_relocate,
    }

    return round(score, 4), details


# ===================================================================
# Dimension 6: Career Trajectory (Weight: 0.10)
# ===================================================================

def _score_trajectory(
    career: list, profile: dict,
    company_analysis: dict, title_trajectory: dict,
) -> Tuple[float, dict]:
    """
    Score career trajectory for coherence, stability, and direction.
    """
    if not career:
        return 0.2, {"note": "No career history"}

    # Average tenure
    tenures = [r.get("duration_months", 0) for r in career]
    avg_tenure = sum(tenures) / max(len(tenures), 1)

    # Job hopping penalty (JD: "someone who plans to be here 3+ years")
    if avg_tenure < 12 and len(career) >= 3:
        stability_score = 0.3
    elif avg_tenure < 18 and len(career) >= 3:
        stability_score = 0.5
    elif avg_tenure < 24:
        stability_score = 0.7
    else:
        stability_score = 1.0

    # Career coherence (uses pre-computed title_trajectory)
    coherence_score = 0.5  # default
    if title_trajectory["has_ai_role_history"]:
        coherence_score = 0.9
    if title_trajectory["is_progressing_toward_ai"]:
        coherence_score = min(1.0, coherence_score + 0.1)

    # Recent coding (JD: "This role writes code")
    coding_score = 1.0 if title_trajectory["recent_is_coding"] else 0.5

    # Company diversity (uses pre-computed company_analysis)
    diversity_score = 0.5
    if company_analysis["has_product_experience"]:
        diversity_score = 0.8
    if company_analysis["consulting_only"]:
        diversity_score = 0.3

    # Combined
    score = (
        0.30 * stability_score
        + 0.25 * coherence_score
        + 0.25 * coding_score
        + 0.20 * diversity_score
    )

    details = {
        "avg_tenure_months": round(avg_tenure, 1),
        "num_roles": len(career),
        "stability": "stable" if avg_tenure >= 24 else "moderate" if avg_tenure >= 18 else "job_hopper",
        "has_ai_history": title_trajectory["has_ai_role_history"],
        "recent_is_coding": title_trajectory["recent_is_coding"],
        "title_hopping": title_trajectory["title_hopping"],
    }

    return round(score, 4), details
