"""
Skills Taxonomy for the Redrob AI Candidate Ranking System.

Categorizes skills by relevance to the Senior AI Engineer JD,
provides synonym mapping, and detects keyword-stuffing patterns.
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple

# ---------------------------------------------------------------------------
# Skill Categories — relevance to "Senior AI Engineer, Founding Team" JD
# ---------------------------------------------------------------------------

# CORE_AI_SKILLS: The JD explicitly requires production experience with these.
# These are the highest-signal skills for ranking.
CORE_AI_SKILLS: Set[str] = {
    # Embeddings & Retrieval
    "embeddings", "sentence-transformers", "sentence transformers",
    "vector embeddings", "bge", "e5", "openai embeddings",
    "word2vec", "doc2vec", "fasttext", "sbert",
    "text embeddings", "embedding models",
    # Vector Databases
    "pinecone", "weaviate", "qdrant", "milvus", "faiss",
    "elasticsearch", "opensearch", "vespa", "annoy", "scann",
    "chromadb", "chroma", "pgvector", "vector search",
    "vector database", "vector store", "haystack",
    # NLP & IR
    "nlp", "natural language processing", "information retrieval",
    "text mining", "text classification", "named entity recognition",
    "ner", "sentiment analysis", "text generation",
    "semantic search", "search engines", "search ranking",
    "ranking systems", "learning to rank", "search",
    "ir", "reranking", "re-ranking",
    # Core ML
    "machine learning", "deep learning", "neural networks",
    "transformers", "bert", "gpt", "llm", "llms",
    "large language models", "language models",
    "attention mechanism", "seq2seq",
    # Frameworks
    "pytorch", "tensorflow", "jax", "hugging face", "huggingface",
    "transformers library",
    # RAG & LLM Systems
    "rag", "retrieval augmented generation",
    "prompt engineering", "langchain", "llamaindex", "llama index",
    "llama-index",
    # Fine-tuning
    "fine-tuning", "fine tuning", "finetuning",
    "fine-tuning llms", "lora", "qlora", "peft",
    "adapter tuning", "instruction tuning",
    # Evaluation
    "ndcg", "mrr", "map", "a/b testing", "ab testing",
    "model evaluation", "ranking evaluation",
    "precision", "recall", "f1",
    # MLOps & Serving
    "mlflow", "weights & biases", "wandb", "bentoml",
    "torchserve", "triton", "onnx", "tensorrt",
    "model serving", "model deployment", "kubeflow",
    # Recommendation Systems
    "recommendation systems", "collaborative filtering",
    "content-based filtering", "matrix factorization",
    "recommender systems", "recommendations",
}

# SUPPORTING_AI_SKILLS: Relevant and helpful, but not core JD requirements.
SUPPORTING_AI_SKILLS: Set[str] = {
    # Python ecosystem
    "python", "numpy", "pandas", "scipy", "matplotlib",
    "scikit-learn", "sklearn", "jupyter",
    # Data Engineering (adjacent)
    "spark", "pyspark", "apache spark",
    "airflow", "apache airflow", "luigi",
    "dbt", "data pipelines", "etl",
    "feature engineering", "feature store",
    "data warehousing", "data modeling",
    # Statistical & Classical ML
    "statistical modeling", "statistics",
    "xgboost", "lightgbm", "catboost", "random forest",
    "gradient boosting", "regression", "classification",
    "clustering", "dimensionality reduction",
    # Data Science
    "data science", "data analysis", "data visualization",
    "exploratory data analysis",
    # Databases (for data work)
    "sql", "postgresql", "mysql", "mongodb", "redis",
    "snowflake", "bigquery", "redshift", "databricks",
    # Cloud ML
    "sagemaker", "vertex ai", "azure ml",
    # Misc ML
    "computer vision", "image classification", "object detection",
    "gans", "generative ai", "diffusion models",
    "speech recognition", "tts", "asr",
    "reinforcement learning", "time series",
    "anomaly detection",
}

# ENGINEERING_SKILLS: Technical but not AI-specific. Show engineering competence.
ENGINEERING_SKILLS: Set[str] = {
    # Infrastructure
    "docker", "kubernetes", "k8s", "terraform", "ansible",
    "ci/cd", "jenkins", "github actions",
    # Cloud
    "aws", "gcp", "azure", "cloud computing",
    "ec2", "s3", "lambda", "ecs", "eks",
    # Backend
    "java", "golang", "go", "rust", "c++", "c#", "scala",
    "flask", "django", "fastapi",
    "rest api", "graphql", "grpc", "api design",
    "microservices", "system design",
    # Data Streaming
    "kafka", "apache kafka", "rabbitmq",
    "apache beam", "apache flink", "kinesis",
    # DevOps
    "linux", "bash", "shell scripting",
    "monitoring", "prometheus", "grafana",
    "logging", "observability",
    # Version Control
    "git", "github",
}

# FRONTEND_SKILLS: Not relevant to the AI Engineer role but not negative.
FRONTEND_SKILLS: Set[str] = {
    "react", "angular", "vue", "svelte",
    "javascript", "typescript", "node.js", "nodejs",
    "html", "css", "tailwind", "bootstrap",
    "redux", "webpack", "vite", "next.js", "nextjs",
    "figma", "ui/ux", "responsive design",
}

# NON_TECHNICAL_SKILLS: Negative signal if these dominate the skill profile.
NON_TECHNICAL_SKILLS: Set[str] = {
    "project management", "scrum", "agile", "kanban",
    "jira", "confluence", "trello",
    "marketing", "seo", "sem", "content writing",
    "copywriting", "social media",
    "accounting", "finance", "bookkeeping", "tally",
    "excel", "powerpoint", "word", "google sheets",
    "photoshop", "illustrator", "canva",
    "six sigma", "lean", "sap", "erp",
    "sales", "crm", "salesforce",
    "hr", "recruitment", "talent acquisition",
    "supply chain", "logistics", "inventory",
    "mechanical design", "solidworks", "autocad", "catia",
    "civil engineering", "structural analysis",
    "electrical engineering", "plc", "scada",
}


def _normalize(skill_name: str) -> str:
    """Normalize a skill name for matching."""
    return skill_name.lower().strip()


def categorize_skill(skill_name: str) -> str:
    """
    Return the category for a given skill name.
    Categories: 'core_ai', 'supporting_ai', 'engineering',
                'frontend', 'non_technical', 'unknown'
    """
    norm = _normalize(skill_name)
    if norm in CORE_AI_SKILLS:
        return "core_ai"
    if norm in SUPPORTING_AI_SKILLS:
        return "supporting_ai"
    if norm in ENGINEERING_SKILLS:
        return "engineering"
    if norm in FRONTEND_SKILLS:
        return "frontend"
    if norm in NON_TECHNICAL_SKILLS:
        return "non_technical"
    return "unknown"


def compute_skill_relevance(
    skills: List[dict],
    assessment_scores: Dict[str, float] | None = None,
) -> dict:
    """
    Compute a comprehensive skill relevance profile for a candidate.

    Returns a dict with:
      - core_ai_score: weighted score for core AI skills (0-1)
      - supporting_ai_score: weighted score for supporting skills (0-1)
      - engineering_score: score for engineering skills
      - skill_trust_score: how much we trust the claimed skills (0-1)
      - keyword_stuffing_risk: risk of keyword stuffing (0-1)
      - category_counts: dict of category -> count
      - matched_core_skills: list of matched core skill names
      - matched_supporting_skills: list of matched supporting skill names
    """
    if not skills:
        return {
            "core_ai_score": 0.0,
            "supporting_ai_score": 0.0,
            "engineering_score": 0.0,
            "skill_trust_score": 0.0,
            "keyword_stuffing_risk": 0.0,
            "category_counts": {},
            "matched_core_skills": [],
            "matched_supporting_skills": [],
        }

    assessment_scores = assessment_scores or {}
    category_counts = {
        "core_ai": 0, "supporting_ai": 0, "engineering": 0,
        "frontend": 0, "non_technical": 0, "unknown": 0,
    }
    matched_core: List[str] = []
    matched_supporting: List[str] = []

    core_ai_total = 0.0
    supporting_ai_total = 0.0
    engineering_total = 0.0

    # Keyword-stuffing detection signals
    expert_with_low_duration = 0
    expert_with_zero_endorsements = 0
    total_expert_claims = 0
    total_trust_signals = 0.0
    trust_count = 0

    for skill in skills:
        name = skill.get("name", "")
        proficiency = skill.get("proficiency", "beginner")
        endorsements = skill.get("endorsements", 0)
        duration_months = skill.get("duration_months", 0)

        category = categorize_skill(name)
        category_counts[category] += 1

        # Proficiency weight
        prof_weight = {
            "expert": 1.0, "advanced": 0.75,
            "intermediate": 0.5, "beginner": 0.25,
        }.get(proficiency, 0.25)

        # Trust multiplier per skill
        trust = _compute_skill_trust(
            proficiency, endorsements, duration_months,
            assessment_scores.get(name),
        )
        total_trust_signals += trust
        trust_count += 1

        # Keyword-stuffing markers
        if proficiency in ("expert", "advanced"):
            total_expert_claims += 1
            if duration_months < 6:
                expert_with_low_duration += 1
            if endorsements == 0:
                expert_with_zero_endorsements += 1

        # Weighted contribution by category
        weighted = prof_weight * trust
        if category == "core_ai":
            core_ai_total += weighted
            matched_core.append(name)
        elif category == "supporting_ai":
            supporting_ai_total += weighted
            matched_supporting.append(name)
        elif category == "engineering":
            engineering_total += weighted

    # Normalize scores (cap at 1.0)
    # A candidate with 4+ strong core AI skills should approach 1.0
    core_ai_score = min(1.0, core_ai_total / 4.0)
    supporting_ai_score = min(1.0, supporting_ai_total / 3.0)
    engineering_score = min(1.0, engineering_total / 3.0)

    # Overall skill trust
    avg_trust = total_trust_signals / max(trust_count, 1)

    # Keyword stuffing risk
    stuffing_risk = _compute_stuffing_risk(
        total_expert_claims, expert_with_low_duration,
        expert_with_zero_endorsements, category_counts,
    )

    return {
        "core_ai_score": round(core_ai_score, 4),
        "supporting_ai_score": round(supporting_ai_score, 4),
        "engineering_score": round(engineering_score, 4),
        "skill_trust_score": round(avg_trust, 4),
        "keyword_stuffing_risk": round(stuffing_risk, 4),
        "category_counts": category_counts,
        "matched_core_skills": matched_core,
        "matched_supporting_skills": matched_supporting,
    }


def _compute_skill_trust(
    proficiency: str,
    endorsements: int,
    duration_months: int,
    assessment_score: float | None,
) -> float:
    """
    Compute a trust multiplier (0.0-1.0) for a single skill claim.
    High trust = proficiency backed by endorsements, duration, and assessment.
    Low trust = high proficiency claim with no backing evidence.
    """
    trust = 0.5  # baseline

    # Endorsement signal
    if endorsements >= 20:
        trust += 0.2
    elif endorsements >= 5:
        trust += 0.1
    elif endorsements == 0:
        trust -= 0.1

    # Duration signal
    if duration_months >= 24:
        trust += 0.2
    elif duration_months >= 12:
        trust += 0.1
    elif duration_months < 6 and proficiency in ("expert", "advanced"):
        trust -= 0.2  # Suspicious: expert in < 6 months

    # Assessment score signal (if available)
    if assessment_score is not None:
        if assessment_score >= 70:
            trust += 0.15
        elif assessment_score >= 50:
            trust += 0.05
        elif assessment_score < 30:
            trust -= 0.15  # Low assessment contradicts proficiency

    return max(0.0, min(1.0, trust))


def _compute_stuffing_risk(
    total_expert: int,
    expert_low_duration: int,
    expert_zero_endorsements: int,
    category_counts: dict,
) -> float:
    """
    Compute keyword-stuffing risk score (0.0-1.0).
    High risk = many expert claims with no backing evidence.
    """
    risk = 0.0

    if total_expert > 0:
        # High ratio of expert claims with low duration
        if expert_low_duration / max(total_expert, 1) > 0.5:
            risk += 0.3
        # High ratio of expert claims with zero endorsements
        if expert_zero_endorsements / max(total_expert, 1) > 0.5:
            risk += 0.3

    # Many core AI skills from a non-technical title
    # (This is checked at the scorer level, not here)

    # Too many expert/advanced claims overall
    if total_expert > 8:
        risk += 0.2
    elif total_expert > 5:
        risk += 0.1

    return min(1.0, risk)


def get_primary_expertise_domain(category_counts: dict) -> str:
    """
    Determine the candidate's primary expertise domain
    based on skill category distribution.
    """
    if not category_counts:
        return "unknown"

    # Exclude 'unknown' from determination
    relevant = {
        k: v for k, v in category_counts.items()
        if k != "unknown" and v > 0
    }
    if not relevant:
        return "unknown"

    return max(relevant, key=relevant.get)


def is_cv_speech_only(skills: List[dict]) -> bool:
    """
    Check if a candidate's AI skills are primarily
    Computer Vision or Speech (without NLP/IR).
    Per JD: "People whose primary expertise is computer vision,
    speech, or robotics without significant NLP/IR exposure"
    """
    cv_speech_skills = {
        "computer vision", "image classification", "object detection",
        "gans", "image segmentation", "image processing",
        "speech recognition", "tts", "asr", "speech synthesis",
        "robotics", "ros", "slam",
        "diffusion models",  # if only this and no NLP
    }
    nlp_ir_skills = {
        "nlp", "natural language processing", "information retrieval",
        "text mining", "text classification", "named entity recognition",
        "ner", "sentiment analysis", "search", "ranking",
        "embeddings", "semantic search", "llm", "llms",
        "transformers", "bert", "gpt", "language models",
        "fine-tuning llms", "rag", "prompt engineering",
    }

    has_cv_speech = False
    has_nlp_ir = False

    for skill in skills:
        norm = _normalize(skill.get("name", ""))
        if norm in cv_speech_skills:
            has_cv_speech = True
        if norm in nlp_ir_skills:
            has_nlp_ir = True

    return has_cv_speech and not has_nlp_ir
