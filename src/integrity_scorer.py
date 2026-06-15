"""
Cross-field integrity checks for synthetic honeypot profiles.

The challenge contains a small number of deliberately impossible profiles.
Checks here target strong contradictions only; ordinary overlaps, rounding,
and minor date noise are not penalized.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import List, Tuple


REFERENCE_DATE = date(2026, 5, 24)
_SUMMARY_YOE = re.compile(
    r"with\s+(\d+(?:\.\d+)?)\s+years?\s+of\s+(?:hands-on\s+)?experience",
    re.IGNORECASE,
)


def _parse_date(raw: str | None) -> date | None:
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def _month_delta(start: date, end: date) -> int:
    return (end.year - start.year) * 12 + end.month - start.month


def compute_integrity_score(candidate: dict) -> Tuple[float, bool, List[str]]:
    """
    Return (integrity_score, is_honeypot, reasons).

    A single decisive contradiction is enough to flag a honeypot. Softer
    anomalies accumulate but do not punish otherwise coherent profiles.
    """
    penalty = 0.0
    decisive = False
    reasons: List[str] = []

    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    claimed_yoe = float(profile.get("years_of_experience", 0))

    starts: list[date] = []
    total_months = 0
    for role in career:
        start = _parse_date(role.get("start_date"))
        end = _parse_date(role.get("end_date"))
        is_current = bool(role.get("is_current", False))
        duration = int(role.get("duration_months", 0))
        total_months += duration

        if start:
            starts.append(start)
        effective_end = end or (REFERENCE_DATE if is_current else None)

        if start and effective_end:
            if effective_end < start:
                penalty += 0.8
                decisive = True
                reasons.append(f"End date precedes start date at {role.get('company', '?')}")
                continue

            actual_months = _month_delta(start, effective_end)
            gap = abs(duration - actual_months)
            if gap > 12:
                penalty += 0.7
                decisive = True
                reasons.append(
                    f"Role duration at {role.get('company', '?')} differs from its dates by {gap} months"
                )
            elif gap > 6:
                penalty += 0.45
                decisive = True
                reasons.append(
                    f"Role duration at {role.get('company', '?')} differs from its dates by {gap} months"
                )

        if is_current and end is not None:
            penalty += 0.5
            decisive = True
            reasons.append(f"Current role at {role.get('company', '?')} has an end date")

    # The generated summaries normally repeat the profile YoE. A large
    # disagreement is a strong cross-field contradiction and catches injected
    # YoE honeypots without penalizing ordinary career overlap.
    summary = str(profile.get("summary", ""))
    summary_match = _SUMMARY_YOE.search(summary)
    if summary_match:
        summary_yoe = float(summary_match.group(1))
        if abs(summary_yoe - claimed_yoe) > 1.5:
            penalty += 0.75
            decisive = True
            reasons.append(
                f"Summary states {summary_yoe:.1f} years but profile claims {claimed_yoe:.1f}"
            )

    if starts and claimed_yoe > 0:
        span_months = _month_delta(min(starts), REFERENCE_DATE)
        claimed_months = claimed_yoe * 12

        if claimed_months - span_months > 30:
            penalty += 0.65
            decisive = True
            reasons.append(
                f"Claims {claimed_yoe:.1f} years but career dates span only {span_months / 12:.1f}"
            )

        # A large sum-vs-claim gap can be caused by overlapping roles, so only
        # treat it as decisive when it is substantial both absolutely and
        # proportionally.
        excess = total_months - claimed_months
        if excess > 30 and total_months > claimed_months * 1.35:
            penalty += 0.55
            decisive = True
            reasons.append(
                f"Career durations total {total_months} months versus {claimed_yoe:.1f} claimed years"
            )

    expert_zero = sum(
        1
        for skill in skills
        if skill.get("proficiency") == "expert"
        and int(skill.get("duration_months", 0)) == 0
    )
    if expert_zero >= 5:
        penalty += 0.75
        decisive = True
        reasons.append(f"{expert_zero} expert skills claim zero months of use")

    # Schema-breaking signal values are always decisive.
    impossible_signal_checks = (
        (float(signals.get("profile_completeness_score", 0)) > 100, "Profile completeness exceeds 100"),
        (float(signals.get("recruiter_response_rate", 0)) > 1, "Recruiter response rate exceeds 1"),
        (float(signals.get("interview_completion_rate", 0)) > 1, "Interview completion rate exceeds 1"),
    )
    for invalid, reason in impossible_signal_checks:
        if invalid:
            penalty += 0.8
            decisive = True
            reasons.append(reason)

    for name, score in signals.get("skill_assessment_scores", {}).items():
        if float(score) < 0 or float(score) > 100:
            penalty += 0.8
            decisive = True
            reasons.append(f"Assessment score outside 0-100: {name}={score}")
            break

    integrity_score = max(0.0, 1.0 - penalty)
    is_honeypot = decisive and integrity_score <= 0.55
    return round(integrity_score, 4), is_honeypot, reasons
