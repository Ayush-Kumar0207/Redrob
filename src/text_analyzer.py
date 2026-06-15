"""
Text Analyzer for the Redrob AI Candidate Ranking System.

OPTIMIZED: All regex patterns pre-compiled at module load time.
Uses simple string matching where possible for performance.
Target: 100K candidates in < 60 seconds.
"""

from __future__ import annotations
import re
from typing import List

# ---------------------------------------------------------------------------
# PRE-COMPILED keyword patterns for career description analysis
# Using simple lowercase string matching for most cases (much faster than regex)
# ---------------------------------------------------------------------------

# Production/deployment experience (strong positive)
PRODUCTION_TERMS = [
    "production", "deployed", "shipped", "real users", "at scale",
    "scalable", "latency", "throughput", "sla",
    "a/b test", "ab test", "online experiment",
    "monitoring", "alerting", "on-call",
    "release", "rollout", "roll out",
    "ci/cd", "continuous integration", "continuous delivery",
    "end-to-end", "end to end",
    "live system", "live service", "live traffic",
    "million users", "millions of users", "million requests",
    "user-facing", "user facing",
]

# Ranking/search/retrieval experience (highest value for JD)
RANKING_SEARCH_TERMS = [
    "ranking system", "ranking model", "ranking algorithm", "ranking pipeline",
    "ranking engine",
    "search system", "search engine", "search infrastructure", "search pipeline",
    "search ranking", "search quality", "search relevance",
    "retrieval system", "retrieval pipeline", "retrieval model", "retrieval quality",
    "recommendation system", "recommendation engine", "recommendation pipeline",
    "recommendation model", "recommendation algorithm",
    "recsys",
    "information retrieval",
    "semantic search",
    "vector search", "vector database", "vector index", "vector store",
    "vector similarity",
    "hybrid search", "hybrid retrieval",
    "re-ranking", "reranking",
    "learning to rank", "learning-to-rank", "ltr",
    "ndcg", "mrr",
    "candidate matching", "candidate ranking", "candidate scoring",
    "talent matching", "talent intelligence",
    "matching system", "matching algorithm", "matching engine",
    "collaborative filtering",
    "content-based filtering", "content based filtering",
]

# Embeddings/NLP work
EMBEDDINGS_NLP_TERMS = [
    "embedding", "embeddings",
    "sentence-transformer", "sentence transformer",
    "bert", "gpt", "transformer",
    "fine-tun", "fine tun", "finetuning",
    "llm", "large language model",
    "nlp", "natural language processing",
    "text classification", "text mining", "text generation",
    "text processing",
    "ner", "named entity",
    "sentiment analysis",
    "prompt engineering",
    "rag", "retrieval-augmented", "retrieval augmented",
    "word2vec", "fasttext",
    "tokeniz",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "elasticsearch", "opensearch",
    "lora", "qlora", "peft",
]

# ML/AI system work (broader)
ML_SYSTEM_TERMS = [
    "machine learning", "deep learning",
    "neural network",
    "model training", "model serving", "model deployment",
    "model inference", "model optimization",
    "feature engineering", "feature store", "feature pipeline",
    "data pipeline",
    "mlops", "ml pipeline", "ml infrastructure", "ml platform",
    "pytorch", "tensorflow",
    "scikit-learn", "sklearn", "xgboost",
    "hugging face", "huggingface",
    "model evaluation", "model validation", "model testing",
    "hyperparameter", "cross-validation", "cross validation",
]

# Research-only indicators (potential negative per JD)
RESEARCH_ONLY_TERMS = [
    "published paper", "published research",
    "academic lab", "academic research", "academic institution",
    "phd research",
    "journal paper", "journal publication",
    "conference paper", "conference presentation",
    "theoretical",
    "proof of concept", "proof-of-concept",
    "prototype only",
]

# Management-only indicators (potential negative per JD)
MANAGEMENT_TERMS = [
    "managed a team", "managed team",
    "led a team", "led team",
    "people management",
    "stakeholder management",
    "budget management", "budget planning",
    "headcount",
    "executive communication",
    "slide-craft", "slidecraft",
    "strategic planning", "strategy planning",
    "business development", "business strategy",
    "kpi",
    "operational efficiency", "operational excellence",
]


def _count_term_matches(text_lower: str, terms: List[str]) -> int:
    """Count how many terms appear in the lowercase text. Simple string 'in' check."""
    count = 0
    for term in terms:
        if term in text_lower:
            count += 1
    return count


def _find_term_matches(text_lower: str, terms: List[str]) -> List[str]:
    """Return all matching terms found in the lowercase text."""
    return [term for term in terms if term in text_lower]


def analyze_career_descriptions(career_history: List[dict]) -> dict:
    """
    Analyze all career descriptions for a candidate.
    OPTIMIZED: uses simple string matching instead of regex.
    """
    if not career_history:
        return _empty_result()

    # Concatenate all descriptions
    parts = []
    for role in career_history:
        desc = role.get("description", "")
        if desc:
            parts.append(desc)

    if not parts:
        return _empty_result()

    # Single lowercase conversion for all matching
    full_text = " ".join(parts).lower()

    # Count matches in each category
    production_count = _count_term_matches(full_text, PRODUCTION_TERMS)
    ranking_count = _count_term_matches(full_text, RANKING_SEARCH_TERMS)
    embeddings_count = _count_term_matches(full_text, EMBEDDINGS_NLP_TERMS)
    ml_count = _count_term_matches(full_text, ML_SYSTEM_TERMS)
    research_count = _count_term_matches(full_text, RESEARCH_ONLY_TERMS)
    management_count = _count_term_matches(full_text, MANAGEMENT_TERMS)

    # Find specific evidence (for reasoning generation)
    production_evidence = _find_term_matches(full_text, PRODUCTION_TERMS)[:5]
    ranking_evidence = _find_term_matches(full_text, RANKING_SEARCH_TERMS)[:5]
    embeddings_evidence = _find_term_matches(full_text, EMBEDDINGS_NLP_TERMS)[:5]

    # Normalize to 0-1 (diminishing returns after threshold)
    production_score = min(1.0, production_count / 5.0)
    ranking_score = min(1.0, ranking_count / 3.0)
    embeddings_score = min(1.0, embeddings_count / 4.0)
    ml_score = min(1.0, ml_count / 4.0)
    research_score = min(1.0, research_count / 3.0)
    management_score = min(1.0, management_count / 4.0)

    # Derived signals
    technical_signal = production_score + ranking_score + embeddings_score + ml_score
    management_signal = management_score

    is_hands_on = technical_signal > management_signal
    is_production_focused = production_score > 0.3

    # If both research AND production, reduce research penalty
    if production_score > 0.3 and research_score > 0:
        research_score *= 0.3

    return {
        "production_score": round(production_score, 4),
        "ranking_search_score": round(ranking_score, 4),
        "embeddings_nlp_score": round(embeddings_score, 4),
        "ml_systems_score": round(ml_score, 4),
        "research_only_score": round(research_score, 4),
        "management_only_score": round(management_score, 4),
        "production_evidence": production_evidence,
        "ranking_evidence": ranking_evidence,
        "embeddings_evidence": embeddings_evidence,
        "is_hands_on": is_hands_on,
        "is_production_focused": is_production_focused,
    }


def compute_fast_jd_evidence(profile: dict, desc_analysis: dict) -> tuple[float, dict]:
    """
    Compute a fast, deterministic JD-evidence score from parsed career text.

    This is the production default because it is reproducible without large
    artifacts and avoids fitting or transforming one TF-IDF vector at a time.
    """
    profile_text = " ".join(
        [
            str(profile.get("current_title", "")),
            str(profile.get("headline", "")),
            str(profile.get("summary", "")),
        ]
    ).lower()
    profile_terms = [
        "ranking", "retrieval", "search", "recommendation", "matching",
        "embedding", "vector", "ndcg", "mrr", "a/b", "evaluation",
        "production", "product company", "end-to-end", "shipped",
    ]
    profile_matches = [term for term in profile_terms if term in profile_text]
    profile_score = min(1.0, len(profile_matches) / 6.0)

    ranking = float(desc_analysis.get("ranking_search_score", 0))
    embeddings = float(desc_analysis.get("embeddings_nlp_score", 0))
    production = float(desc_analysis.get("production_score", 0))
    ml_systems = float(desc_analysis.get("ml_systems_score", 0))
    hands_on = 1.0 if desc_analysis.get("is_hands_on", False) else 0.0

    score = (
        0.32 * ranking
        + 0.20 * embeddings
        + 0.20 * production
        + 0.10 * ml_systems
        + 0.10 * profile_score
        + 0.08 * hands_on
    )
    if ranking >= 0.6 and production >= 0.4:
        score += 0.08
    score = max(0.0, min(1.0, score))

    evidence = list(
        dict.fromkeys(
            desc_analysis.get("ranking_evidence", [])
            + desc_analysis.get("embeddings_evidence", [])
            + profile_matches
        )
    )
    level = (
        "exceptional" if score >= 0.8
        else "strong" if score >= 0.6
        else "moderate" if score >= 0.35
        else "weak"
    )
    return round(score, 4), {
        "method": "fast_structured_evidence_v1",
        "semantic_match_level": level,
        "high_value_matches": evidence[:5],
        "production_signals": len(desc_analysis.get("production_evidence", [])),
        "profile_matches": profile_matches[:5],
    }


def _empty_result() -> dict:
    return {
        "production_score": 0.0,
        "ranking_search_score": 0.0,
        "embeddings_nlp_score": 0.0,
        "ml_systems_score": 0.0,
        "research_only_score": 0.0,
        "management_only_score": 0.0,
        "production_evidence": [],
        "ranking_evidence": [],
        "embeddings_evidence": [],
        "is_hands_on": False,
        "is_production_focused": False,
    }
