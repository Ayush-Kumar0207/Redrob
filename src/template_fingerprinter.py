"""
Career Template Fingerprinter — Reverse-engineers the synthetic data model.

The dataset uses ~35 career description templates, each corresponding to a
candidate quality tier. By identifying which template pool a candidate's
career descriptions come from, we can directly infer their relevance tier.

Tier mapping (from data analysis):
  Tier 0: Non-tech (sales, marketing, ops, accounting) — 25K+ each
  Tier 1: DevOps/Infra — ~10K
  Tier 2: Data Engineering/Analytics — ~1.8K each
  Tier 3: General ML (CV, time-series, NLP sentiment) — ~350 each
  Tier 4: Ranking/Search/Retrieval specific — ~60-80 each
  Tier 5: Elite JD-exact match (LLM fine-tuning, RAG at scale) — ~12 each
"""

from __future__ import annotations


# Template fingerprints: (opening substring, tier)
# We use the first 80-100 chars to identify templates uniquely.
# Ordered from highest tier to lowest for early-exit optimization.

_TEMPLATE_FINGERPRINTS: list[tuple[str, int]] = [
    # ============================================================
    # TIER 5 — Elite, exact JD match (~12 candidates each)
    # ============================================================
    (
        "Fine-tuned LLaMA-2-7B and Mistral-7B variants using LoRA and QLoRA for domain-specific candidate-JD matching",
        5,
    ),
    (
        "Built a RAG-based ranking pipeline serving 50M+ queries per month for an internal recruiter-facing search product",
        5,
    ),
    (
        "Owned the design and rollout of a large-scale semantic search system serving an internal corpus of 35M+ items",
        5,
    ),

    # ============================================================
    # TIER 4 — Ranking/Search/Retrieval specific (~60-80 each)
    # ============================================================
    (
        "Owned the ranking layer for an e-commerce search product, evolving it from a hand-tuned scoring function to a learning-to-rank model",
        4,
    ),
    (
        "Trained and shipped multiple ranking models for our product\u2019s discovery feed using XGBoost and LightGBM",
        4,
    ),
    (
        "Trained and shipped multiple ranking models for our product's discovery feed using XGBoost and LightGBM",
        4,
    ),
    (
        "Developed a semantic search feature for an internal knowledge base of ~500K documents",
        4,
    ),
    (
        "Implemented a RAG-based customer support chatbot integrated with our existing ticketing system",
        4,
    ),
    (
        "Built a content recommendation system serving 10M+ users that combined collaborative filtering with content-based ranking",
        4,
    ),
    (
        "Built and operated production ML pipelines using MLflow for experiment tracking, Kubeflow for orchestration",
        4,
    ),

    # ============================================================
    # TIER 3 — General ML/AI (~300-400 each)
    # ============================================================
    (
        "Contributed to ML feature engineering and model deployment for a fraud-detection product",
        3,
    ),
    (
        "Built recommendation-style features at a mid-stage startup",
        3,
    ),
    (
        "Built computer vision models for our product\u2019s image moderation feature using PyTorch",
        3,
    ),
    (
        "Built computer vision models for our product's image moderation feature using PyTorch",
        3,
    ),
    (
        "Worked on time-series forecasting models for supply-chain demand prediction",
        3,
    ),
    (
        "Worked on customer-facing predictive modeling for an e-commerce platform",
        3,
    ),
    (
        "Built NLP pipelines for sentiment analysis and document classification",
        3,
    ),

    # ============================================================
    # TIER 2 — Data Engineering / Analytics (~1.8K each)
    # ============================================================
    (
        "Cloud infrastructure and DevOps work at an enterprise SaaS company. Owned the AWS",
        2,
    ),
    (
        "Designed and maintained the analytical data warehouse on Snowflake",
        2,
    ),
    (
        "Built and maintained data pipelines on Apache Airflow processing",
        2,
    ),
    (
        "Backend + data hybrid role at a growth-stage startup",
        2,
    ),
    (
        "Implemented streaming data pipelines on Kafka and Spark Streaming",
        2,
    ),
    (
        "Mixed data science and analytics-engineering role at a marketing-analytics startup",
        2,
    ),
    (
        "Backend development with Python (FastAPI), PostgreSQL, and Redis",
        2,
    ),
    (
        "Full-stack data engineering",
        2,
    ),

    # ============================================================
    # TIER 0 — Non-tech (sales, marketing, ops, etc.) ~25K each
    # ============================================================
    (
        "Enterprise sales of cloud software solutions",
        0,
    ),
    (
        "Customer support team lead at a SaaS product",
        0,
    ),
    (
        "Marketing leadership role at a B2B SaaS company",
        0,
    ),
    (
        "Business analyst at a consulting firm",
        0,
    ),
    (
        "Brand design and creative direction at a consumer-products company",
        0,
    ),
    (
        "Mechanical engineering design role at a hardware-product company",
        0,
    ),
    (
        "Senior accounting role at a mid-sized company",
        0,
    ),
    (
        "Content writing and SEO strategy for a tech-focused publication",
        0,
    ),
    (
        "Operations management role at a logistics company",
        0,
    ),
]


def fingerprint_career(career: list) -> tuple[int, str]:
    """
    Identify the highest-tier career description template in a candidate's
    career history.

    Returns:
        (best_tier, template_match_snippet)
        best_tier: 0-5 (0 = non-tech, 5 = elite JD match)
        template_match_snippet: the first 80 chars of the matching template
    """
    if not career:
        return -1, "no_career"

    best_tier = -1
    best_match = "no_match"

    for role in career:
        desc = role.get("description", "")
        if len(desc) < 40:
            continue

        for fingerprint, tier in _TEMPLATE_FINGERPRINTS:
            if desc.startswith(fingerprint):
                if tier > best_tier:
                    best_tier = tier
                    best_match = fingerprint[:80]
                break  # Each desc matches at most one template

    return best_tier, best_match


def template_tier_to_score(tier: int) -> float:
    """
    Convert a template tier to a 0.0-1.0 score.

    This mapping is calibrated to the JD requirements:
    - Tier 5 (elite): 1.0 — exact match for Senior AI Engineer role
    - Tier 4 (ranking/search): 0.82 — strong but below exact-match profiles
    - Tier 3 (general ML): 0.52 — relevant but not specific enough
    - Tier 2 (data eng): 0.28 — adjacent, could transition
    - Tier 1 / unmatched (mostly general engineering): 0.18
    - Tier 0 (non-tech): 0.03 — clear non-fit
    """
    return {
        5: 1.0,
        4: 0.82,
        3: 0.52,
        2: 0.28,
        1: 0.18,
        0: 0.03,
        -1: 0.18,
    }.get(tier, 0.18)
