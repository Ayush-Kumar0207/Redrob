"""
Reasoning Generator for the Redrob AI Candidate Ranking System (v2).

Generates specific, honest, 1-2 sentence reasonings per candidate.
Per spec: "Plain-language reasoning that demonstrates you actually
understood the candidate's profile will rank highly."

Stage 4 evaluation checks:
  1. Specific facts from profile (years, title, skills, signal values)
  2. JD connection (specific requirements, not generic praise)
  3. Honest concerns (acknowledge gaps)
  4. No hallucination (only reference actual profile data)
  5. Variation (not templated)
  6. Rank consistency (tone matches rank)
"""

from __future__ import annotations
from typing import Dict


def generate_reasoning(
    candidate: dict,
    rank: int,
    score_details: dict,
    behavioral_details: dict,
    is_honeypot: bool = False,
) -> str:
    """
    Generate a specific, honest reasoning for a ranked candidate.
    """
    if is_honeypot:
        return _honeypot_reasoning(candidate)

    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    title = profile.get("current_title", "Unknown")
    company = profile.get("current_company", "Unknown")
    yoe = profile.get("years_of_experience", 0)
    location = profile.get("location", "Unknown")
    country = profile.get("country", "Unknown")

    # Extract key details from all scoring dimensions
    title_details = score_details.get("title_details", {})
    skills_details = score_details.get("skills_details", {})
    exp_details = score_details.get("experience_details", {})
    loc_details = score_details.get("location_details", {})
    traj_details = score_details.get("trajectory_details", {})
    desc_analysis = title_details.get("desc_analysis", {})
    career_jd = score_details.get("career_jd_details", {})
    cross_val = score_details.get("cross_val_details", {})
    assessment = score_details.get("assessment_details", {})

    # Build reasoning parts
    parts = []

    # --- Core identity + JD-specific match ---
    parts.append(_identity_and_jd_match(
        title, company, yoe, rank, title_details, career_jd, desc_analysis
    ))

    # --- Evidence from skills, behavior, assessments ---
    evidence = _evidence_statement(
        skills_details, desc_analysis, signals, behavioral_details,
        cross_val, career_jd, rank
    )
    if evidence:
        parts.append(evidence)

    # --- Concerns (earlier for lower ranks) ---
    concerns = _concerns(
        rank, title_details, skills_details, exp_details,
        loc_details, traj_details, signals, behavioral_details,
        cross_val
    )
    if concerns:
        parts.append(concerns)

    reasoning = "; ".join(parts) + "."

    # Keep to 1-2 sentences
    if len(reasoning) > 420:
        reasoning = reasoning[:417] + "..."

    return reasoning


def _identity_and_jd_match(
    title: str, company: str, yoe: float, rank: int,
    title_details: dict, career_jd: dict, desc_analysis: dict,
) -> str:
    """Generate the primary identity + JD-match statement."""
    tier = title_details.get("title_tier", 1)
    trajectory = title_details.get("title_trajectory", {})
    company_analysis = title_details.get("company_analysis", {})

    # Base identity
    base = f"{title} at {company} with {yoe:.1f} years experience"

    # JD-specific matching based on what the JD actually asks for
    jd_matches = []

    # Check career description evidence (most important per JD)
    if desc_analysis.get("ranking_search_score", 0) > 0.3:
        evidence = desc_analysis.get("ranking_evidence", [])
        if evidence:
            jd_matches.append(f"career shows {evidence[0]} work matching JD's ranking/retrieval focus")
    elif desc_analysis.get("embeddings_nlp_score", 0) > 0.3:
        jd_matches.append("career includes embeddings/NLP work per JD requirements")
    elif desc_analysis.get("production_score", 0) > 0.3:
        jd_matches.append("has production ML deployment experience per JD")

    # Check career-JD semantic similarity
    # Works for both TF-IDF (v3) and MiniLM embedding scorer
    sem_level = career_jd.get("semantic_match_level", "")
    high_terms = career_jd.get("high_value_matches", [])
    if sem_level in ("exceptional", "strong") and not jd_matches:
        jd_matches.append("strong semantic match to JD's ranking/retrieval mandate")
    elif len(high_terms) >= 3 and not jd_matches:
        jd_matches.append(
            f"career descriptions reference {', '.join(high_terms[:3])} "
            f"aligning with JD's search/ranking mandate"
        )

    # Product company experience (JD explicitly wants this)
    if company_analysis.get("has_product_experience"):
        if not company_analysis.get("consulting_only"):
            jd_matches.append("product company background as JD prefers")

    if jd_matches:
        return f"{base}, {jd_matches[0]}"
    elif tier >= 4:
        return f"{base}, title aligns with the Senior AI Engineer role"
    else:
        return base


def _evidence_statement(
    skills_details: dict, desc_analysis: dict,
    signals: dict, behavioral: dict,
    cross_val: dict, career_jd: dict, rank: int,
) -> str:
    """Generate supporting evidence statement."""
    parts = []

    # Core skill match — reference actual JD-required skills
    core_skills = skills_details.get("matched_core", [])
    if core_skills:
        parts.append(f"core skills: {', '.join(core_skills[:3])}")

    # Cross-validation result
    val_ratio = cross_val.get("validation_ratio", 0)
    if val_ratio >= 0.6:
        parts.append("skills validated by career descriptions")

    # Behavioral - response rate (JD's top behavioral signal)
    response_rate = signals.get("recruiter_response_rate", 0)
    if response_rate >= 0.7:
        parts.append(f"{response_rate:.0%} recruiter response rate")
    elif response_rate < 0.2:
        parts.append(f"only {response_rate:.0%} recruiter response rate (JD flags this)")

    # GitHub (relevant for "writes code" JD requirement)
    github_score = signals.get("github_activity_score", -1)
    if github_score >= 50:
        parts.append(f"GitHub score {github_score:.0f}")

    # Assessment scores — hard evidence the spec values highly
    assessments = signals.get("skill_assessment_scores", {})
    if assessments:
        high_scores = {k: v for k, v in assessments.items() if v >= 60}
        if high_scores:
            # Show top 2 assessment scores for credibility
            sorted_scores = sorted(high_scores.items(), key=lambda x: -x[1])
            if len(sorted_scores) >= 2:
                parts.append(
                    f"platform assessments: {sorted_scores[0][0]} {sorted_scores[0][1]:.0f}/100, "
                    f"{sorted_scores[1][0]} {sorted_scores[1][1]:.0f}/100"
                )
            else:
                parts.append(f"scored {sorted_scores[0][1]:.0f}/100 on {sorted_scores[0][0]}")

    # For top-10, include production signals from career
    if rank <= 10:
        prod_count = career_jd.get("production_signals", 0)
        if prod_count >= 3:
            parts.append("multiple production deployment references in career")

    return "; ".join(parts[:2]) if parts else ""


def _concerns(
    rank: int,
    title_details: dict, skills_details: dict,
    exp_details: dict, loc_details: dict,
    traj_details: dict, signals: dict,
    behavioral: dict, cross_val: dict,
) -> str:
    """Generate honest concerns — JD-connected and specific."""
    concerns = []

    tier = title_details.get("title_tier", 1)
    yoe = exp_details.get("years_of_experience", 0)
    exp_fit = exp_details.get("fit", "outside")
    notice = signals.get("notice_period_days", 90)
    in_india = loc_details.get("in_india", True)
    response_rate = signals.get("recruiter_response_rate", 0.5)
    stability = traj_details.get("stability", "moderate")
    company_analysis = title_details.get("company_analysis", {})

    # Title concern (JD: "Marketing Manager is not a fit")
    if tier <= 2:
        label = title_details.get("title_label", "non-technical")
        concerns.append(f"role is {label}, not core AI/ML per JD requirements")

    # Experience (JD: "5-9 years")
    if exp_fit == "outside":
        if yoe < 4:
            concerns.append(f"{yoe:.1f}y experience below JD's 5-9y range")
        elif yoe > 12:
            concerns.append(f"{yoe:.1f}y may be overqualified for founding-team role")

    # Location (JD: "Pune/Noida preferred")
    if not in_india:
        concerns.append(f"based outside India ({loc_details.get('country', '?')})")

    # Notice (JD: "sub-30 preferred, can buy out up to 30")
    if notice > 90:
        concerns.append(f"{notice}-day notice period (JD prefers sub-30)")

    # Response rate (JD: explicitly calls this out)
    if response_rate < 0.15:
        concerns.append(f"very low response rate ({response_rate:.0%})")

    # Stability (JD: "plans to be here 3+ years")
    if stability == "job_hopper":
        concerns.append("frequent role changes (JD seeks 3+ year tenure)")

    # Consulting-only (JD: explicit disqualifier)
    if company_analysis.get("consulting_only"):
        concerns.append("career entirely at consulting firms (JD disqualifier)")

    # Top-ranked candidates can still have an honest operational trade-off.
    if rank <= 10:
        critical = []
        if not in_india:
            critical.append(f"based outside India ({loc_details.get('country', '?')})")
        if notice > 90:
            critical.append(f"{notice}-day notice period")
        if response_rate < 0.15:
            critical.append(f"very low response rate ({response_rate:.0%})")
        if company_analysis.get("consulting_only"):
            critical.append("consulting-only career")
        return f"trade-off: {critical[0]}" if critical else ""

    # Skills validation
    val_ratio = cross_val.get("validation_ratio", 0)
    if val_ratio < 0.25 and cross_val.get("checked", 0) >= 3:
        concerns.append("claimed skills not backed by career evidence")

    if concerns:
        concern_str = ", ".join(concerns[:2])
        if rank <= 30:
            return f"minor concern: {concern_str}"
        elif rank <= 60:
            return f"concerns: {concern_str}"
        else:
            return f"notable gaps: {concern_str}"
    return ""


def _honeypot_reasoning(candidate: dict) -> str:
    """Reasoning for honeypot candidates."""
    profile = candidate.get("profile", {})
    return (
        f"Profile flagged as inconsistent -- "
        f"{profile.get('current_title', 'Unknown')} at "
        f"{profile.get('current_company', 'Unknown')} with "
        f"contradictory career timeline and skill duration data."
    )
