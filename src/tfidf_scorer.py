"""
TF-IDF Career-JD Similarity Scorer (v3 — sklearn upgrade).

Uses sklearn TfidfVectorizer for REAL cosine similarity against the JD,
supplemented by high-value keyword bonuses for domain-specific terms.

This is used as the fallback when pre-computed MiniLM embeddings are
not available (see embedding_scorer.py for the primary scorer).

Also contains:
  - Skills-Career cross-validation (keyword stuffing detector)
  - Assessment score evaluator (hard evidence from platform tests)
"""

from __future__ import annotations
from typing import List, Dict, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
import numpy as np

# ---- Module-level JD vectorizer (fitted ONCE at import time) ----

# Condensed JD text for TF-IDF matching — focuses on what the role actually needs
_JD_TEXT = """
Senior AI Engineer founding team. Ranking retrieval matching systems embeddings
vector databases hybrid search. Production experience sentence-transformers
OpenAI embeddings BGE E5. Vector databases Pinecone Weaviate Qdrant Milvus
OpenSearch Elasticsearch FAISS. Python code quality. Evaluation frameworks
NDCG MRR MAP A/B testing. LLM fine-tuning LoRA QLoRA PEFT.
Learning-to-rank models XGBoost neural. Recommendation systems collaborative
filtering. NLP natural language processing transformer BERT GPT.
Embeddings retrieval ranking search recommendation semantic reranking.
Production deployment scale latency pipeline serving inference real-time.
Product company startup early-stage. Ship build architect end-to-end.
Recruiter candidate matching talent intelligence platform.
Deep learning neural networks PyTorch TensorFlow. MLOps MLflow Kubeflow
Docker Kubernetes. Distributed systems large-scale inference optimization.
"""

_TFIDF_VECTORIZER = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2),      # Capture bigrams like "vector search", "deep learning"
    min_df=1,
    sublinear_tf=True,       # Use log(1 + tf) for better scaling
)

# Fit on the JD and transform it
_JD_VEC = _TFIDF_VECTORIZER.fit_transform([_JD_TEXT])


# Keywords that indicate relevant career work (weighted by importance)
_HIGH_VALUE_TERMS = {
    "ranking", "retrieval", "search", "recommendation", "embeddings",
    "vector", "semantic", "reranking", "re-ranking", "ndcg", "mrr",
    "relevance", "index", "indexing", "query", "candidate matching",
    "information retrieval", "learning to rank", "l2r", "bm25",
    "elasticsearch", "opensearch", "solr", "faiss", "pinecone",
    "weaviate", "qdrant", "milvus", "chromadb",
}

_MEDIUM_VALUE_TERMS = {
    "nlp", "natural language", "transformer", "bert", "gpt",
    "fine-tuning", "fine-tune", "lora", "qlora", "peft",
    "production", "deployed", "deployment", "scale", "latency",
    "pipeline", "serving", "inference", "model serving",
    "a/b test", "evaluation", "metrics", "benchmark",
    "embedding", "sentence-transformer", "encoder",
}

# Headline/summary terms that signal JD fit
_HEADLINE_JD_TERMS = {
    "ai engineer", "ml engineer", "machine learning", "nlp",
    "data scientist", "applied ml", "recommendation", "search",
    "ranking", "retrieval", "deep learning", "natural language",
    "applied ai", "senior ai", "staff ml", "principal",
    "research engineer", "applied scientist",
}

# Headline terms that signal NON-fit
_HEADLINE_NONFIT_TERMS = {
    "marketing", "accountant", "hr manager", "operations",
    "sales", "customer support", "graphic designer",
    "content writer", "civil engineer", "mechanical engineer",
    "business analyst", "project manager", "frontend",
    "web developer", "ui/ux", "product manager",
}

# Production / system-building language
_PRODUCTION_SIGNALS = [
    "deployed", "production", "scale", "users", "traffic",
    "latency", "throughput", "million", "serving", "real-time",
    "a/b test", "experiment", "metrics", "monitoring",
    "sla", "uptime", "reliability", "incident",
]

_SYSTEM_SIGNALS = [
    "built", "designed", "architected", "implemented", "shipped",
    "owned", "led the", "end-to-end", "from scratch", "ground up",
    "developed", "created", "launched", "drove",
]

# JD-relevant skill names for assessment score checking
_JD_RELEVANT_ASSESSMENTS = {
    "nlp", "machine learning", "deep learning", "embeddings",
    "information retrieval", "semantic search", "recommendation systems",
    "pytorch", "tensorflow", "transformers", "ranking systems",
    "search", "learning to rank", "llms", "fine-tuning llms",
    "natural language processing", "python",
    "vector search", "elasticsearch", "faiss",
}


def _build_candidate_text(career: list, profile: dict) -> str:
    """Build comprehensive text from all candidate sources."""
    parts = []

    # Career descriptions (deepest signal)
    for role in career:
        desc = role.get("description", "")
        if desc:
            parts.append(desc)

    # Headline and summary
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")
    if headline:
        parts.append(headline)
    if summary:
        parts.append(summary)

    return " ".join(parts)


def compute_career_jd_similarity(
    career: list,
    profile: dict = None,
) -> Tuple[float, dict]:
    """
    Compute TF-IDF cosine similarity between candidate text and the JD,
    supplemented by high-value keyword bonuses.

    Returns:
        (similarity_score, details_dict)
        similarity_score: 0.0 to 1.0
    """
    if profile is None:
        profile = {}

    # Build candidate text
    candidate_text = _build_candidate_text(career, profile)

    if len(candidate_text.strip()) < 20:
        return 0.0, {"note": "Insufficient text data"}

    # --- sklearn TF-IDF cosine similarity ---
    try:
        cand_vec = _TFIDF_VECTORIZER.transform([candidate_text])
        tfidf_sim = float(sklearn_cosine(_JD_VEC, cand_vec)[0][0])
    except Exception:
        tfidf_sim = 0.0

    # Scale TF-IDF similarity: typical range is 0.0-0.5 for text
    # Map [0.0, 0.4] -> [0.0, 1.0] with soft cap
    tfidf_score = min(1.0, tfidf_sim / 0.35)

    # --- High-value keyword bonus (supplements TF-IDF) ---
    career_text = " ".join(
        role.get("description", "") for role in career
    ).lower()
    headline = profile.get("headline", "").lower()
    summary = profile.get("summary", "").lower()

    high_matches = [t for t in _HIGH_VALUE_TERMS if t in career_text]
    medium_matches = [t for t in _MEDIUM_VALUE_TERMS if t in career_text]

    keyword_bonus = min(0.20, len(high_matches) * 0.025 + len(medium_matches) * 0.012)

    # --- Headline JD-fit signal ---
    headline_fit = any(t in headline for t in _HEADLINE_JD_TERMS)
    headline_nonfit = any(t in headline for t in _HEADLINE_NONFIT_TERMS)
    headline_bonus = 0.0
    if headline_fit:
        headline_bonus = 0.05
    if headline_nonfit:
        headline_bonus = -0.05

    # --- Production signals bonus ---
    prod_count = sum(1 for s in _PRODUCTION_SIGNALS if s in career_text)
    production_bonus = min(0.10, prod_count * 0.012)

    # --- System-building signals bonus ---
    system_count = sum(1 for s in _SYSTEM_SIGNALS if s in career_text)
    system_bonus = min(0.06, system_count * 0.010)

    # --- Combined score ---
    # TF-IDF provides the base (70%), keyword bonuses add precision (30%)
    score = 0.70 * tfidf_score + keyword_bonus + headline_bonus + production_bonus + system_bonus
    score = max(0.0, min(1.0, score))

    details = {
        "tfidf_cosine_sim": round(tfidf_sim, 4),
        "tfidf_score": round(tfidf_score, 4),
        "high_value_matches": high_matches[:5],
        "medium_value_matches": medium_matches[:5],
        "high_count": len(high_matches),
        "medium_count": len(medium_matches),
        "production_signals": prod_count,
        "system_signals": system_count,
        "headline_fit": headline_fit,
        "headline_nonfit": headline_nonfit,
        "method": "sklearn_tfidf_v3",
    }

    return round(score, 4), details


def compute_skills_career_cross_validation(
    skills: list, career: list
) -> Tuple[float, dict]:
    """
    Cross-validate claimed skills against career descriptions.
    Detects keyword stuffers whose skills don't match their actual work.

    Returns:
        (validation_score, details)
        validation_score: 0.0 (all unvalidated) to 1.0 (all validated)
    """
    if not skills or not career:
        return 0.5, {"note": "Insufficient data for cross-validation"}

    # Build career text
    career_text = " ".join(
        role.get("description", "") for role in career
    ).lower()

    if len(career_text) < 50:
        return 0.5, {"note": "Career descriptions too short"}

    # Core AI/ML skills that should appear in career descriptions if genuine
    core_skills_to_check = {
        "pytorch": ["pytorch", "torch"],
        "tensorflow": ["tensorflow", "tf."],
        "nlp": ["nlp", "natural language", "text processing", "language model"],
        "deep learning": ["deep learning", "neural network", "cnn", "rnn", "lstm"],
        "machine learning": ["machine learning", "ml model", "training", "classifier"],
        "recommendation systems": ["recommendation", "recommender", "collaborative filtering"],
        "semantic search": ["semantic search", "semantic", "vector search"],
        "embeddings": ["embedding", "vector representation", "sentence-transformer"],
        "elasticsearch": ["elasticsearch", "elastic", "opensearch"],
        "faiss": ["faiss", "vector index", "approximate nearest"],
        "langchain": ["langchain", "chain", "agent"],
        "rag": ["rag", "retrieval augmented", "retrieval-augmented"],
        "llms": ["llm", "large language model", "gpt", "language model"],
        "fine-tuning llms": ["fine-tun", "finetuning", "lora", "qlora", "peft"],
        "information retrieval": ["information retrieval", "ir ", "retrieval"],
        "ranking systems": ["ranking", "ranker", "learning to rank", "l2r"],
        "pinecone": ["pinecone"],
        "weaviate": ["weaviate"],
        "qdrant": ["qdrant"],
        "milvus": ["milvus"],
        "mlflow": ["mlflow", "experiment tracking"],
        "kubeflow": ["kubeflow", "ml pipeline"],
        "docker": ["docker", "container"],
        "kubernetes": ["kubernetes", "k8s"],
        "scikit-learn": ["scikit", "sklearn", "random forest", "logistic regression"],
        "prompt engineering": ["prompt", "prompting"],
        "sentence transformers": ["sentence-transformer", "sentence transformer", "sbert"],
        "learning to rank": ["learning to rank", "l2r", "lambdamart", "listwise"],
        "search": ["search engine", "search system", "search infrastructure", "query"],
    }

    validated = 0
    unvalidated = 0
    checked = 0
    validated_skills = []
    unvalidated_skills = []

    for skill in skills:
        name = skill.get("name", "").lower()
        prof = skill.get("proficiency", "")

        if name in core_skills_to_check:
            checked += 1
            search_terms = core_skills_to_check[name]
            found = any(t in career_text for t in search_terms)
            if found:
                validated += 1
                validated_skills.append(name)
            else:
                unvalidated += 1
                unvalidated_skills.append(name)

    if checked == 0:
        return 0.5, {"note": "No checkable core skills"}

    validation_ratio = validated / checked

    if validation_ratio >= 0.6:
        score = 0.8 + (validation_ratio - 0.6) * 0.5
    elif validation_ratio >= 0.3:
        score = 0.5 + (validation_ratio - 0.3) * 1.0
    else:
        score = validation_ratio * 1.67

    details = {
        "checked": checked,
        "validated": validated,
        "unvalidated": unvalidated,
        "validation_ratio": round(validation_ratio, 2),
        "validated_skills": validated_skills[:5],
        "unvalidated_skills": unvalidated_skills[:5],
    }

    return round(min(1.0, score), 4), details


def compute_assessment_score(
    signals: dict, skills: list
) -> Tuple[float, dict]:
    """
    Score based on Redrob skill_assessment_scores — HARD EVIDENCE.
    Assessment scores are verified platform tests, far more trustworthy
    than self-reported proficiency levels.

    Returns:
        (assessment_fit_score, details)
    """
    assessments = signals.get("skill_assessment_scores", {})
    if not assessments:
        return 0.5, {"note": "No assessments taken"}

    # Check JD-relevant assessments
    relevant_scores = []
    all_scores = []
    relevant_names = []

    for skill_name, score in assessments.items():
        norm = skill_name.lower()
        all_scores.append(score)
        if norm in _JD_RELEVANT_ASSESSMENTS:
            relevant_scores.append(score)
            relevant_names.append(f"{skill_name}:{score:.0f}")

    if not relevant_scores:
        # They took assessments but none are JD-relevant
        avg_all = sum(all_scores) / len(all_scores)
        # Generic competence signal
        return round(min(1.0, avg_all / 100.0 * 0.6), 4), {
            "note": "No JD-relevant assessments",
            "avg_all": round(avg_all, 1),
        }

    # Score based on JD-relevant assessment performance
    avg_relevant = sum(relevant_scores) / len(relevant_scores)
    max_relevant = max(relevant_scores)

    # Scoring: 80+ avg = excellent, 60+ = good, 40+ = moderate, <40 = weak
    if avg_relevant >= 80:
        score = 0.95
    elif avg_relevant >= 65:
        score = 0.7 + (avg_relevant - 65) / 60  # 0.7-0.95
    elif avg_relevant >= 50:
        score = 0.5 + (avg_relevant - 50) / 75  # 0.5-0.7
    elif avg_relevant >= 35:
        score = 0.3 + (avg_relevant - 35) / 75  # 0.3-0.5
    else:
        score = avg_relevant / 100.0  # 0.0-0.35

    # Bonus for high max score (proves they CAN score well)
    if max_relevant >= 85:
        score = min(1.0, score + 0.05)

    details = {
        "relevant_count": len(relevant_scores),
        "avg_relevant": round(avg_relevant, 1),
        "max_relevant": round(max_relevant, 1),
        "relevant_names": relevant_names[:5],
        "total_assessments": len(all_scores),
    }

    return round(min(1.0, score), 4), details
