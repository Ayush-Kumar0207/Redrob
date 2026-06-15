"""
JD Parser for the Redrob AI Candidate Ranking System.

Structured representation of the Senior AI Engineer JD requirements.
This is hardcoded rather than dynamically parsed, because:
  1. The JD is fixed for this challenge
  2. Hardcoding ensures precision in matching
  3. No runtime parsing overhead
"""

from __future__ import annotations

# The structured JD as a config dict.
# Every scoring component references this.
JD = {
    "title": "Senior AI Engineer — Founding Team",
    "company": "Redrob AI",
    "company_stage": "Series A",
    "product_domain": "AI-native talent intelligence platform",

    # --- Experience ---
    "experience": {
        "ideal_min": 5,
        "ideal_max": 9,
        "acceptable_min": 4,
        "acceptable_max": 12,
        "hard_min": 2,  # Below this is almost certainly not a fit
        "note": "Some people hit senior judgment at 4; some never at 15",
    },

    # --- Location ---
    "location": {
        "office_cities": ["pune", "noida"],
        "preferred_cities": [
            "pune", "noida", "delhi", "new delhi", "delhi ncr",
            "gurgaon", "gurugram", "ghaziabad", "faridabad",
            "hyderabad", "mumbai", "bangalore", "bengaluru",
            "chennai",
        ],
        "country": "india",
        "work_mode": "hybrid",  # Flexible cadence
        "relocation_note": "Open to relocation from Tier-1 Indian cities",
        "international_note": "Case-by-case, no visa sponsorship",
    },

    # --- Notice Period ---
    "notice_period": {
        "ideal_max_days": 30,
        "buyout_max_days": 30,
        "note": "30+ day candidates still in scope but bar is higher",
    },

    # --- Required Skills (must-have) ---
    "required_skills": {
        "embeddings_retrieval": {
            "description": "Production experience with embeddings-based retrieval",
            "keywords": [
                "embeddings", "sentence-transformers", "sentence transformers",
                "bge", "e5", "openai embeddings", "vector embeddings",
                "word2vec", "doc2vec", "fasttext",
                "semantic search", "dense retrieval",
            ],
        },
        "vector_databases": {
            "description": "Production experience with vector DBs / hybrid search",
            "keywords": [
                "pinecone", "weaviate", "qdrant", "milvus", "faiss",
                "elasticsearch", "opensearch", "vespa",
                "vector database", "vector search", "hybrid search",
                "annoy", "scann",
            ],
        },
        "python": {
            "description": "Strong Python, code quality matters",
            "keywords": ["python"],
        },
        "evaluation_frameworks": {
            "description": "Ranking system evaluation (NDCG, MRR, MAP, A/B testing)",
            "keywords": [
                "ndcg", "mrr", "map", "a/b testing", "ab testing",
                "offline evaluation", "online evaluation",
                "ranking evaluation", "model evaluation",
                "precision", "recall",
            ],
        },
    },

    # --- Nice-to-have Skills ---
    "nice_to_have_skills": {
        "llm_finetuning": {
            "description": "LLM fine-tuning (LoRA, QLoRA, PEFT)",
            "keywords": ["lora", "qlora", "peft", "fine-tuning", "fine tuning",
                         "adapter tuning", "instruction tuning"],
        },
        "learning_to_rank": {
            "description": "Learning-to-rank models (XGBoost-based or neural)",
            "keywords": ["learning to rank", "ltr", "xgboost", "lightgbm",
                         "lambdamart", "ranknet"],
        },
        "hr_tech": {
            "description": "HR-tech, recruiting tech, marketplace products",
            "keywords": ["hr tech", "recruiting", "talent", "marketplace",
                         "hiring", "ats", "applicant tracking"],
        },
        "distributed_systems": {
            "description": "Distributed systems or large-scale inference",
            "keywords": ["distributed systems", "large scale", "inference optimization",
                         "kubernetes", "docker", "microservices"],
        },
        "open_source": {
            "description": "Open-source contributions in AI/ML",
            "keywords": ["open source", "open-source", "github", "contributor"],
        },
    },

    # --- Explicit Anti-Patterns (from JD "Things we do NOT want") ---
    "anti_patterns": {
        "title_chasers": "Career of Senior→Staff→Principal every 1.5 years",
        "framework_enthusiasts": "GitHub full of LangChain tutorials, no systems thinking",
        "consulting_only": "Entire career at TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini",
        "cv_speech_only": "Primary expertise is CV/speech/robotics without NLP/IR",
        "no_external_validation": "5+ years on closed-source without papers/talks/OSS",
        "research_only": "Pure research without production deployment",
        "not_coding_recently": "Hasn't written production code in 18+ months",
    },

    # --- Ideal Candidate Profile (from JD "How to read between the lines") ---
    "ideal_profile": {
        "total_experience": "6-8 years",
        "ai_experience": "4-5 years in applied ML/AI at product companies",
        "shipped_system": "End-to-end ranking, search, or recommendation system",
        "strong_opinions": "Hybrid vs dense retrieval, offline vs online eval, fine-tune vs prompt",
        "location": "In or willing to relocate to Noida or Pune",
        "platform_active": "Active on Redrob platform",
    },
}


def get_jd() -> dict:
    """Return the structured JD configuration."""
    return JD


def get_jd_embedding_text() -> str:
    """Return focused JD text for optional offline embedding pre-computation."""
    return """
    Senior AI Engineer, founding team at a Series A talent intelligence product.
    Own production ranking, retrieval, search, recommendation, and candidate-job
    matching systems. Must have shipped embeddings-based retrieval, vector
    databases or hybrid search, strong Python, and rigorous ranking evaluation
    using NDCG, MRR, MAP, offline benchmarks, and online A/B tests. Ideal
    candidate has 5-9 years of experience, recent hands-on production coding,
    product-company experience, and a shipper mindset. Strong positives include
    learning to rank, LLM fine-tuning, distributed systems, HR-tech or
    marketplace experience, and open-source work. Penalize pure research,
    consulting-only careers, framework tutorials without systems depth,
    CV/speech-only expertise, title chasing, and stale or unreachable profiles.
    Pune or Noida preferred; relocation within India is acceptable.
    """.strip()
