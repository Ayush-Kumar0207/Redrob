import json
from src.company_classifier import analyze_career_companies
from src.integrity_scorer import compute_integrity_score
from src.loader import load_candidates
from src.tfidf_scorer import compute_career_jd_similarity
from src.title_matcher import match_title

def load_validation_candidates():
    with open("validation_candidates.json", "r") as f:
        return json.load(f)

def save_validation_candidates(cands):
    with open("validation_candidates.json", "w", encoding='utf-8') as f:
        json.dump(cands, f, indent=2)

def auto_label(candidate: dict) -> int:
    """Assign a 1-5 label based on strict JD rules."""
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    # This is a heuristic bootstrap label, not a substitute for manual review.
    integrity, is_honeypot, _ = compute_integrity_score(candidate)
    if is_honeypot or integrity < 0.5:
        return 1

    yoe = profile.get("years_of_experience", 0)
    title_tier, _ = match_title(profile.get("current_title", ""))
    sim_score, _ = compute_career_jd_similarity(career, profile)
    company_analysis = analyze_career_companies(career)

    skill_names = [
        s.get("name", "") if isinstance(s, dict) else str(s)
        for s in skills
    ]
    skills_lower = [s.lower() for s in skill_names]
    core_terms = (
        "nlp", "embedding", "vector", "retrieval", "ranking",
        "semantic search", "llm", "rag", "sentence transformer",
        "information retrieval",
    )
    has_core_ai = any(
        any(term in skill for term in core_terms)
        for skill in skills_lower
    )

    if company_analysis["consulting_only"]:
        return 1
    if title_tier <= 2:
        return 2
    if yoe < 3 or yoe > 15:
        return 2
    if not has_core_ai:
        return 2

    points = 0
    if 5 <= yoe <= 9:
        points += 1
    if sim_score >= 0.3:
        points += 1
    if sim_score >= 0.5:
        points += 1
    if title_tier >= 4:
        points += 1
    if company_analysis["has_product_experience"]:
        points += 1

    if points >= 4:
        return 5
    if points >= 2:
        return 4
    return 3

if __name__ == "__main__":
    # Load the 100K to get full data for these 50
    print("Loading 100K candidates to get full career histories...")
    all_cands = load_candidates("dataset/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl")
    cand_map = {c["candidate_id"]: c for c in all_cands}
    
    val_cands = load_validation_candidates()
    
    labeled_count = 0
    for vc in val_cands:
        cid = vc["candidate_id"]
        full_cand = cand_map.get(cid)
        if not full_cand:
            vc["human_tier_label"] = 1
            continue

        vc["human_tier_label"] = auto_label(full_cand)
        labeled_count += 1
        
    save_validation_candidates(val_cands)
    print(f"Successfully auto-labeled {labeled_count} candidates.")
