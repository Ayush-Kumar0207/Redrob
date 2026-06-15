import unittest

from src.behavioral import compute_behavioral_modifier
from src.integrity_scorer import compute_integrity_score
from src.scorer import WEIGHTS
from src.template_fingerprinter import template_tier_to_score
from src.text_analyzer import compute_fast_jd_evidence


def _signals(**overrides):
    values = {
        "last_active_date": "2026-05-20",
        "open_to_work_flag": True,
        "recruiter_response_rate": 0.8,
        "avg_response_time_hours": 12,
        "interview_completion_rate": 0.9,
        "offer_acceptance_rate": 0.8,
        "notice_period_days": 30,
        "profile_completeness_score": 90,
        "github_activity_score": 70,
        "saved_by_recruiters_30d": 12,
        "profile_views_received_30d": 80,
        "search_appearance_30d": 150,
        "applications_submitted_30d": 5,
        "connection_count": 300,
        "endorsements_received": 60,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": True,
        "preferred_work_mode": "hybrid",
        "willing_to_relocate": True,
        "skill_assessment_scores": {},
    }
    values.update(overrides)
    return values


def _candidate():
    return {
        "candidate_id": "CAND_0000001",
        "profile": {
            "summary": "Machine learning engineer with 5.0 years of experience shipping systems.",
            "years_of_experience": 5.0,
        },
        "career_history": [
            {
                "company": "Example Product",
                "start_date": "2021-05-24",
                "end_date": None,
                "duration_months": 60,
                "is_current": True,
            }
        ],
        "skills": [],
        "redrob_signals": _signals(),
    }


class BehavioralTests(unittest.TestCase):
    def test_reachable_candidate_beats_unavailable_candidate(self):
        reachable, _ = compute_behavioral_modifier(_signals())
        unavailable, _ = compute_behavioral_modifier(
            _signals(
                last_active_date="2025-10-01",
                open_to_work_flag=False,
                recruiter_response_rate=0.05,
                avg_response_time_hours=240,
            )
        )
        self.assertGreater(reachable, unavailable)
        self.assertLessEqual(reachable, 1.20)
        self.assertLessEqual(unavailable, 0.62)


class IntegrityTests(unittest.TestCase):
    def test_coherent_current_role_is_not_penalized(self):
        score, honeypot, reasons = compute_integrity_score(_candidate())
        self.assertEqual(score, 1.0)
        self.assertFalse(honeypot)
        self.assertEqual(reasons, [])

    def test_summary_profile_experience_contradiction_is_honeypot(self):
        candidate = _candidate()
        candidate["profile"]["years_of_experience"] = 15.0
        score, honeypot, reasons = compute_integrity_score(candidate)
        self.assertTrue(honeypot)
        self.assertLess(score, 0.55)
        self.assertTrue(any("Summary states" in reason for reason in reasons))


class ScoringTests(unittest.TestCase):
    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(WEIGHTS.values()), 1.0)

    def test_template_prior_is_strictly_ordered(self):
        ordered = [template_tier_to_score(tier) for tier in [5, 4, 3, 2, 1, 0]]
        self.assertEqual(ordered, sorted(ordered, reverse=True))
        self.assertEqual(len(ordered), len(set(ordered)))

    def test_fast_jd_evidence_rewards_production_ranking(self):
        strong, _ = compute_fast_jd_evidence(
            {"summary": "Shipped ranking and retrieval systems with offline evaluation."},
            {
                "ranking_search_score": 1.0,
                "embeddings_nlp_score": 0.8,
                "production_score": 1.0,
                "ml_systems_score": 0.8,
                "is_hands_on": True,
                "ranking_evidence": ["ranking system"],
                "embeddings_evidence": ["embeddings"],
                "production_evidence": ["production"],
            },
        )
        weak, _ = compute_fast_jd_evidence(
            {"summary": "General business operations."},
            {
                "ranking_search_score": 0.0,
                "embeddings_nlp_score": 0.0,
                "production_score": 0.0,
                "ml_systems_score": 0.0,
                "is_hands_on": False,
                "ranking_evidence": [],
                "embeddings_evidence": [],
                "production_evidence": [],
            },
        )
        self.assertGreater(strong, weak)


if __name__ == "__main__":
    unittest.main()
