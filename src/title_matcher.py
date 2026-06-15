"""
Title Matcher for the Redrob AI Candidate Ranking System.

Maps job titles to relevance tiers for the Senior AI Engineer JD.
The JD warns: "A candidate who has all the AI keywords listed as skills
but whose title is 'Marketing Manager' is not a fit."
"""

from __future__ import annotations
import re
from typing import Tuple

# ---------------------------------------------------------------------------
# Title Tier Definitions
# ---------------------------------------------------------------------------
# Tier 5 (Perfect fit): Titles that directly match the JD role
# Tier 4 (Strong fit): Adjacent AI/ML roles with production focus
# Tier 3 (Moderate): Software engineers who could have relevant experience
# Tier 2 (Weak): Technical roles that are tangentially related
# Tier 1 (Non-fit): Non-technical roles — trap candidates per JD warning

# Patterns are checked in order; first match wins.
# Each tuple: (compiled_regex_pattern, tier, label)

_TIER_PATTERNS: list[Tuple[re.Pattern, int, str]] = []


def _add(pattern: str, tier: int, label: str):
    _TIER_PATTERNS.append((re.compile(pattern, re.IGNORECASE), tier, label))


# --- Tier 5: Perfect Fit ---
_add(r"\b(?:senior\s+)?ai\s+engineer\b", 5, "AI Engineer")
_add(r"\b(?:senior\s+)?ml\s+engineer\b", 5, "ML Engineer")
_add(r"\b(?:senior\s+)?machine\s+learning\s+engineer\b", 5, "ML Engineer")
_add(r"\b(?:senior\s+)?applied\s+(?:ml|ai|machine\s+learning)\b", 5, "Applied ML")
_add(r"\b(?:senior\s+)?nlp\s+engineer\b", 5, "NLP Engineer")
_add(r"\b(?:senior\s+)?search\s+engineer\b", 5, "Search Engineer")
_add(r"\b(?:senior\s+)?ranking\s+engineer\b", 5, "Ranking Engineer")
_add(r"\b(?:senior\s+)?retrieval\s+engineer\b", 5, "Retrieval Engineer")
_add(r"\b(?:senior\s+)?recommendation(?:s)?\s+engineer\b", 5, "RecSys Engineer")
_add(r"\bai/ml\s+engineer\b", 5, "AI/ML Engineer")
_add(r"\bml/ai\s+engineer\b", 5, "ML/AI Engineer")
_add(r"\bdeep\s+learning\s+engineer\b", 5, "Deep Learning Engineer")
_add(r"\bllm\s+engineer\b", 5, "LLM Engineer")

# --- Tier 4: Strong Fit ---
_add(r"\b(?:senior\s+)?data\s+scientist\b", 4, "Data Scientist")
_add(r"\b(?:senior\s+)?research\s+engineer\b", 4, "Research Engineer")
_add(r"\b(?:senior\s+)?research\s+scientist\b", 4, "Research Scientist")
_add(r"\b(?:senior\s+)?(?:applied\s+)?scientist\b", 4, "Applied Scientist")
_add(r"\bmlops\s+engineer\b", 4, "MLOps Engineer")
_add(r"\b(?:senior\s+)?platform\s+engineer\b", 4, "Platform Engineer")
_add(r"\b(?:senior\s+)?backend\s+engineer\b", 4, "Backend Engineer")
_add(r"\b(?:senior\s+)?software\s+engineer.*(?:ml|ai|data|search)\b", 4, "SWE-ML")
_add(r"\bdata\s+engineer\b", 4, "Data Engineer")
_add(r"\banalytics\s+engineer\b", 4, "Analytics Engineer")

# --- Tier 3: Moderate Fit ---
_add(r"\b(?:senior\s+)?software\s+engineer\b", 3, "Software Engineer")
_add(r"\b(?:senior\s+)?(?:full\s*stack|fullstack)\s+(?:engineer|developer)\b", 3, "Full Stack Engineer")
_add(r"\b(?:senior\s+)?(?:software|application)\s+developer\b", 3, "Software Developer")
_add(r"\b(?:senior\s+)?devops\s+engineer\b", 3, "DevOps Engineer")
_add(r"\b(?:senior\s+)?cloud\s+engineer\b", 3, "Cloud Engineer")
_add(r"\b(?:senior\s+)?site\s+reliability\s+engineer\b", 3, "SRE")
_add(r"\bsre\b", 3, "SRE")
_add(r"\b(?:senior\s+)?(?:python|java|golang)\s+developer\b", 3, "Developer")
_add(r"\btechnical\s+lead\b", 3, "Technical Lead")
_add(r"\btech\s+lead\b", 3, "Tech Lead")
_add(r"\bstaff\s+engineer\b", 3, "Staff Engineer")
_add(r"\bprincipal\s+engineer\b", 3, "Principal Engineer")
_add(r"\b(?:junior\s+)?ml\s+engineer\b", 3, "Junior ML Engineer")

# --- Tier 2: Weak Fit ---
_add(r"\b(?:senior\s+)?(?:product|engineering)\s+manager\b", 2, "Manager")
_add(r"\b(?:vp|vice\s+president)\s+(?:of\s+)?engineering\b", 2, "VP Engineering")
_add(r"\bcto\b", 2, "CTO")
_add(r"\b(?:senior\s+)?(?:solutions?|technical)\s+architect\b", 2, "Architect")
_add(r"\b(?:senior\s+)?consultant\b", 2, "Consultant")
_add(r"\b(?:senior\s+)?(?:qa|test|quality)\s+engineer\b", 2, "QA Engineer")
_add(r"\b(?:senior\s+)?(?:frontend|front-end|ui)\s+(?:engineer|developer)\b", 2, "Frontend Engineer")
_add(r"\bproject\s+manager\b", 2, "Project Manager")
_add(r"\bscrum\s+master\b", 2, "Scrum Master")
_add(r"\b(?:business|data)\s+analyst\b", 2, "Analyst")

# --- Tier 1: Non-Fit (Traps per JD) ---
_add(r"\bmarketing\s+manager\b", 1, "Marketing Manager")
_add(r"\bhr\s+manager\b", 1, "HR Manager")
_add(r"\bhuman\s+resources?\b", 1, "HR")
_add(r"\baccountant\b", 1, "Accountant")
_add(r"\bsales\s+(?:executive|manager|representative)\b", 1, "Sales")
_add(r"\bcustomer\s+(?:support|service|success)\b", 1, "Customer Support")
_add(r"\bgraphic\s+designer\b", 1, "Graphic Designer")
_add(r"\bcontent\s+writer\b", 1, "Content Writer")
_add(r"\bcopywriter\b", 1, "Copywriter")
_add(r"\b(?:civil|mechanical|electrical|chemical)\s+engineer\b", 1, "Non-SW Engineer")
_add(r"\boperations\s+manager\b", 1, "Operations Manager")
_add(r"\bteacher\b", 1, "Teacher")
_add(r"\blecturer\b", 1, "Lecturer")
_add(r"\bfinancial\s+analyst\b", 1, "Financial Analyst")
_add(r"\badministrative\b", 1, "Administrative")
_add(r"\bsupply\s+chain\b", 1, "Supply Chain")
_add(r"\bmanufacturing\b", 1, "Manufacturing")


def match_title(title: str) -> Tuple[int, str]:
    """
    Match a job title to a relevance tier.

    Returns:
        (tier, label) where tier is 1-5 and label is a human-readable category.
        Tier 5 = perfect fit, Tier 1 = non-fit.
    """
    if not title:
        return 1, "Unknown"

    for pattern, tier, label in _TIER_PATTERNS:
        if pattern.search(title):
            return tier, label

    # Fallback: try to infer from keywords
    title_lower = title.lower()

    # Check for AI/ML keywords in unmatched titles
    if any(kw in title_lower for kw in ["ai", "ml", "machine learning", "deep learning"]):
        return 4, "AI-Related"
    if any(kw in title_lower for kw in ["data", "analytics"]):
        return 3, "Data-Related"
    if any(kw in title_lower for kw in ["engineer", "developer", "programmer"]):
        return 3, "Engineer"
    if any(kw in title_lower for kw in ["manager", "director", "head"]):
        return 2, "Manager"
    if any(kw in title_lower for kw in ["intern", "trainee", "fresher"]):
        return 1, "Junior/Intern"

    return 2, "Other"


def compute_career_title_trajectory(career_history: list[dict]) -> dict:
    """
    Analyze the trajectory of titles across a candidate's career.

    Returns:
        Dict with:
        - current_tier: tier of current/latest role
        - current_label: label of current role
        - best_tier: highest tier achieved in career
        - best_label: label of best-tier role
        - has_ai_role_history: had an AI/ML title at any point
        - is_progressing_toward_ai: trajectory shows movement to AI
        - recent_is_coding: current role likely involves writing code
        - title_hopping: bool, frequent title changes suggesting instability
    """
    if not career_history:
        return {
            "current_tier": 1, "current_label": "Unknown",
            "best_tier": 1, "best_label": "Unknown",
            "has_ai_role_history": False,
            "is_progressing_toward_ai": False,
            "recent_is_coding": False,
            "title_hopping": False,
        }

    tiers_and_labels = []
    for role in career_history:
        title = role.get("title", "")
        tier, label = match_title(title)
        is_current = role.get("is_current", False)
        tiers_and_labels.append((tier, label, is_current, role))

    # Current role
    current_roles = [t for t in tiers_and_labels if t[2]]
    if current_roles:
        current_tier, current_label = current_roles[0][0], current_roles[0][1]
    else:
        # Use first entry (most recent)
        current_tier, current_label = tiers_and_labels[0][0], tiers_and_labels[0][1]

    # Best tier
    best_entry = max(tiers_and_labels, key=lambda x: x[0])
    best_tier, best_label = best_entry[0], best_entry[1]

    # Has AI role history
    has_ai = any(t >= 4 for t, _, _, _ in tiers_and_labels)

    # Progressing toward AI: later roles have higher tiers
    is_progressing = False
    if len(tiers_and_labels) >= 2:
        # Career history is usually ordered most recent first
        recent_avg = sum(t for t, _, _, _ in tiers_and_labels[:2]) / 2
        old_avg = sum(t for t, _, _, _ in tiers_and_labels[-2:]) / 2
        is_progressing = recent_avg > old_avg

    # Recent role is coding (not pure management)
    coding_tiers = {3, 4, 5}  # Engineers write code
    recent_is_coding = current_tier in coding_tiers

    # Title hopping: many different titles in short tenure
    unique_titles = set(label for _, label, _, _ in tiers_and_labels)
    avg_tenure = sum(
        r.get("duration_months", 0) for _, _, _, r in tiers_and_labels
    ) / max(len(tiers_and_labels), 1)
    title_hopping = len(unique_titles) >= 3 and avg_tenure < 18

    return {
        "current_tier": current_tier,
        "current_label": current_label,
        "best_tier": best_tier,
        "best_label": best_label,
        "has_ai_role_history": has_ai,
        "is_progressing_toward_ai": is_progressing,
        "recent_is_coding": recent_is_coding,
        "title_hopping": title_hopping,
    }
