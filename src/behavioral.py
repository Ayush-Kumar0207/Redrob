"""
Behavioral availability scoring for the Redrob candidate ranker.

The behavioral signals answer a different question from profile fit:
"Can a recruiter realistically engage and hire this candidate now?"
Each signal is used once, combined as a weighted evidence score, and then
converted to a deliberately modest multiplier so availability cannot erase
clear differences in technical relevance.
"""

from __future__ import annotations

from datetime import date, datetime
REFERENCE_DATE = date(2026, 5, 24)


def _piecewise(value: float, points: list[tuple[float, float]]) -> float:
    """Linearly interpolate a value across sorted (x, score) points."""
    if value <= points[0][0]:
        return points[0][1]
    if value >= points[-1][0]:
        return points[-1][1]

    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= value <= x1:
            ratio = (value - x0) / (x1 - x0)
            return y0 + ratio * (y1 - y0)
    return 0.0


def _days_since_active(signals: dict) -> int | None:
    raw = signals.get("last_active_date")
    if not raw:
        return None
    try:
        active = datetime.strptime(raw, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None
    return max(0, (REFERENCE_DATE - active).days)


def compute_behavioral_modifier(signals: dict) -> tuple[float, dict]:
    """
    Return an availability multiplier and auditable signal details.

    Normal candidates land near 1.0. Strong, reachable candidates can receive
    up to 1.20x; clearly unavailable candidates can fall as low as 0.55x.
    """
    days_since = _days_since_active(signals)
    response_rate = float(signals.get("recruiter_response_rate", 0.5))
    response_hours = float(signals.get("avg_response_time_hours", 48))
    interview_rate = float(signals.get("interview_completion_rate", 0.5))
    offer_rate = float(signals.get("offer_acceptance_rate", -1))
    notice_days = float(signals.get("notice_period_days", 90))
    completeness = float(signals.get("profile_completeness_score", 50))
    github = float(signals.get("github_activity_score", -1))
    open_to_work = bool(signals.get("open_to_work_flag", False))

    saved = float(signals.get("saved_by_recruiters_30d", 0))
    views = float(signals.get("profile_views_received_30d", 0))
    appearances = float(signals.get("search_appearance_30d", 0))
    applications = float(signals.get("applications_submitted_30d", 0))
    connections = float(signals.get("connection_count", 0))
    endorsements = float(signals.get("endorsements_received", 0))
    verified_count = sum(
        bool(signals.get(key, False))
        for key in ("verified_email", "verified_phone", "linkedin_connected")
    )

    recency_score = 0.0 if days_since is None else _piecewise(
        days_since,
        [(0, 1.0), (7, 0.9), (30, 0.65), (90, 0.15), (180, -0.45), (365, -1.0)],
    )
    response_score = _piecewise(
        response_rate,
        [(0.0, -1.0), (0.05, -0.9), (0.2, -0.55), (0.4, -0.05),
         (0.6, 0.35), (0.8, 0.75), (1.0, 1.0)],
    )
    speed_score = _piecewise(
        response_hours,
        [(0, 1.0), (12, 0.8), (24, 0.5), (72, 0.0), (168, -0.65), (336, -1.0)],
    )
    interview_score = _piecewise(
        interview_rate,
        [(0.0, -1.0), (0.4, -0.35), (0.6, 0.1), (0.8, 0.65), (1.0, 1.0)],
    )
    notice_score = _piecewise(
        notice_days,
        [(0, 1.0), (30, 0.7), (60, 0.15), (90, -0.35), (180, -1.0)],
    )
    completeness_score = _piecewise(
        completeness,
        [(0, -1.0), (40, -0.3), (60, 0.0), (85, 0.6), (100, 1.0)],
    )
    github_score = 0.0 if github < 0 else _piecewise(
        github,
        [(0, -0.2), (20, 0.15), (50, 0.6), (100, 1.0)],
    )
    offer_score = 0.0 if offer_rate < 0 else _piecewise(
        offer_rate,
        [(0.0, -0.8), (0.3, -0.45), (0.5, 0.0), (0.8, 0.65), (1.0, 1.0)],
    )

    # Market interest combines three correlated signals once.
    interest_score = (
        0.50 * _piecewise(saved, [(0, -0.2), (3, 0.0), (10, 0.45), (30, 1.0)])
        + 0.30 * _piecewise(views, [(0, -0.2), (10, 0.0), (50, 0.5), (200, 1.0)])
        + 0.20 * _piecewise(appearances, [(0, -0.2), (20, 0.0), (100, 0.5), (500, 1.0)])
    )
    job_seeking_score = _piecewise(applications, [(0, 0.0), (1, 0.2), (5, 0.6), (20, 1.0)])
    verification_score = {0: -0.5, 1: 0.0, 2: 0.45, 3: 0.8}[verified_count]
    social_proof_score = (
        0.5 * _piecewise(connections, [(0, -0.1), (50, 0.0), (250, 0.4), (1000, 1.0)])
        + 0.5 * _piecewise(endorsements, [(0, -0.1), (10, 0.0), (50, 0.5), (150, 1.0)])
    )

    mode = str(signals.get("preferred_work_mode", "")).lower()
    mode_score = {"hybrid": 0.8, "flexible": 0.8, "onsite": 0.45, "remote": 0.0}.get(mode, 0.0)
    if signals.get("willing_to_relocate", False):
        mode_score = min(1.0, mode_score + 0.2)

    components = {
        "response_rate": (0.22, response_score),
        "recency": (0.15, recency_score),
        "open_to_work": (0.09, 1.0 if open_to_work else -0.35),
        "interview_reliability": (0.08, interview_score),
        "notice": (0.07, notice_score),
        "response_speed": (0.06, speed_score),
        "github": (0.06, github_score),
        "market_interest": (0.07, interest_score),
        "offer_history": (0.04, offer_score),
        "profile_completeness": (0.04, completeness_score),
        "job_seeking": (0.03, job_seeking_score),
        "verification": (0.03, verification_score),
        "work_mode_fit": (0.03, mode_score),
        "social_proof": (0.03, social_proof_score),
    }
    evidence_score = sum(weight * score for weight, score in components.values())
    modifier = 1.0 + 0.24 * evidence_score

    # Explicit availability gates from the JD. Positive vanity signals such as
    # profile views must not rescue someone who is plainly unreachable.
    if response_rate < 0.10 and (days_since or 0) > 150:
        modifier = min(modifier, 0.62)
    elif response_rate < 0.20 and (days_since or 0) > 120:
        modifier = min(modifier, 0.74)
    elif response_rate < 0.20 and not open_to_work:
        modifier = min(modifier, 0.82)

    modifier = max(0.55, min(1.20, modifier))
    details = {
        "availability_score": round(evidence_score, 4),
        "days_since_active": days_since,
        "response_rate_value": response_rate,
        "open_to_work": open_to_work,
        "notice_days": int(notice_days),
        "components": {name: round(score, 4) for name, (_, score) in components.items()},
    }
    return round(modifier, 4), details
