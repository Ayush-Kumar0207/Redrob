"""
Honeypot Detector for the Redrob AI Candidate Ranking System.

Detects candidates with subtly impossible profiles.
Per the spec: "~80 honeypot candidates with subtly impossible profiles
(e.g., 8 years of experience at a company founded 3 years ago;
'expert' proficiency in 10 skills with 0 years used)"

Submissions with honeypot rate > 10% in top 100 are disqualified.
"""

from __future__ import annotations
from datetime import datetime, date
from typing import List, Tuple


def detect_honeypot(candidate: dict) -> Tuple[bool, List[str]]:
    """
    Detect if a candidate profile is a honeypot.

    Returns:
        (is_honeypot: bool, reasons: List[str])
        A candidate is flagged as honeypot if they have 2+ red flags.
    """
    red_flags: List[str] = []

    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])
    signals = candidate.get("redrob_signals", {})

    # ---------------------------------------------------------------
    # 1. Career timeline impossibilities
    # ---------------------------------------------------------------
    claimed_yoe = profile.get("years_of_experience", 0)

    # Sum of career durations vs claimed experience
    total_career_months = sum(
        role.get("duration_months", 0) for role in career
    )
    total_career_years = total_career_months / 12.0

    # Major mismatch: claimed experience vs actual career sum
    if claimed_yoe > 0 and total_career_years > 0:
        ratio = total_career_years / claimed_yoe
        if ratio > 2.5:
            red_flags.append(
                f"Career duration sum ({total_career_years:.1f}y) "
                f">> claimed experience ({claimed_yoe}y)"
            )
        elif ratio < 0.3 and claimed_yoe > 3:
            red_flags.append(
                f"Career duration sum ({total_career_years:.1f}y) "
                f"<< claimed experience ({claimed_yoe}y)"
            )

    # Check individual role date inconsistencies
    for role in career:
        start_str = role.get("start_date", "")
        end_str = role.get("end_date")
        duration = role.get("duration_months", 0)
        is_current = role.get("is_current", False)

        if start_str and end_str:
            try:
                start_date = _parse_date(start_str)
                end_date = _parse_date(end_str)

                if end_date < start_date:
                    red_flags.append(
                        f"Role end_date before start_date: "
                        f"{role.get('company', '?')}"
                    )

                # Check duration vs actual dates
                actual_months = (
                    (end_date.year - start_date.year) * 12
                    + (end_date.month - start_date.month)
                )
                if duration > 0 and actual_months > 0:
                    if abs(duration - actual_months) > max(12, actual_months * 0.5):
                        red_flags.append(
                            f"Duration mismatch at {role.get('company', '?')}: "
                            f"claimed {duration}mo vs dates suggest {actual_months}mo"
                        )
            except (ValueError, TypeError):
                pass

        # Current role with end_date set
        if is_current and end_str is not None:
            red_flags.append(
                f"Current role with end_date set: {role.get('company', '?')}"
            )

    # ---------------------------------------------------------------
    # 2. Skill impossibilities
    # ---------------------------------------------------------------
    expert_skills_with_zero_duration = 0
    expert_skills_with_very_low_duration = 0
    total_expert_skills = 0

    for skill in skills:
        proficiency = skill.get("proficiency", "")
        duration = skill.get("duration_months", 0)
        endorsements = skill.get("endorsements", 0)

        if proficiency in ("expert", "advanced"):
            total_expert_skills += 1
            if duration == 0:
                expert_skills_with_zero_duration += 1
            elif duration < 3:
                expert_skills_with_very_low_duration += 1

    # Many expert skills with 0 duration
    if expert_skills_with_zero_duration >= 5:
        red_flags.append(
            f"{expert_skills_with_zero_duration} expert/advanced skills "
            f"with 0 months duration"
        )

    # Too many expert claims with very low duration
    if expert_skills_with_very_low_duration >= 4:
        red_flags.append(
            f"{expert_skills_with_very_low_duration} expert/advanced skills "
            f"with < 3 months duration"
        )

    # Unreasonably many expert-level skills for experience
    if total_expert_skills >= 10 and claimed_yoe < 5:
        red_flags.append(
            f"{total_expert_skills} expert/advanced skills "
            f"with only {claimed_yoe}y experience"
        )

    # ---------------------------------------------------------------
    # 3. Education impossibilities
    # ---------------------------------------------------------------
    for edu in education:
        start_year = edu.get("start_year", 0)
        end_year = edu.get("end_year", 0)

        if start_year > 0 and end_year > 0:
            duration = end_year - start_year
            degree = edu.get("degree", "").lower()

            # PhD in 1 year or bachelor in 1 year
            if "ph.d" in degree or "phd" in degree:
                if duration < 2:
                    red_flags.append(
                        f"PhD completed in {duration} year(s)"
                    )
            elif "m." in degree or "master" in degree:
                if duration < 1:
                    red_flags.append(
                        f"Masters completed in {duration} year(s)"
                    )

            # Education start after career start
            if career and start_year > 0:
                earliest_career = None
                for role in career:
                    s = role.get("start_date", "")
                    if s:
                        try:
                            d = _parse_date(s)
                            if earliest_career is None or d < earliest_career:
                                earliest_career = d
                        except (ValueError, TypeError):
                            pass

    # ---------------------------------------------------------------
    # 4. Signal impossibilities
    # ---------------------------------------------------------------
    assessment_scores = signals.get("skill_assessment_scores", {})
    for skill_name, score in assessment_scores.items():
        if score > 100 or score < 0:
            red_flags.append(
                f"Impossible assessment score: {skill_name}={score}"
            )

    # Profile completeness > 100
    completeness = signals.get("profile_completeness_score", 0)
    if completeness > 100:
        red_flags.append(f"Profile completeness > 100: {completeness}")

    # Response rate > 1
    response_rate = signals.get("recruiter_response_rate", 0)
    if response_rate > 1.0:
        red_flags.append(f"Response rate > 1.0: {response_rate}")

    # Interview completion rate > 1
    interview_rate = signals.get("interview_completion_rate", 0)
    if interview_rate > 1.0:
        red_flags.append(f"Interview completion rate > 1.0: {interview_rate}")

    # ---------------------------------------------------------------
    # 5. Cross-field contradictions
    # ---------------------------------------------------------------
    # Title is very different from career description content
    current_title = profile.get("current_title", "").lower()
    current_industry = profile.get("current_industry", "").lower()

    # Marketing Manager at IT Services with all AI skills
    non_tech_titles = [
        "marketing manager", "hr manager", "accountant",
        "sales executive", "customer support", "graphic designer",
        "content writer", "operations manager",
        "civil engineer", "mechanical engineer",
    ]

    is_non_tech_title = any(t in current_title for t in non_tech_titles)

    # Check if career descriptions contradict the title/industry heavily
    if career:
        first_desc = career[0].get("description", "").lower()
        # If current title is "Marketing Manager" but description talks about
        # completely different work, that's a contradition (common in synthetic data)
        title_keywords = set(current_title.split())
        desc_mentions_title_domain = any(
            kw in first_desc for kw in title_keywords if len(kw) > 3
        )

    # ---------------------------------------------------------------
    # Final determination
    # ---------------------------------------------------------------
    # A candidate is flagged as honeypot if they have 2+ red flags
    is_honeypot = len(red_flags) >= 2

    return is_honeypot, red_flags


def _parse_date(date_str: str) -> date:
    """Parse a date string in YYYY-MM-DD format."""
    return datetime.strptime(date_str, "%Y-%m-%d").date()
