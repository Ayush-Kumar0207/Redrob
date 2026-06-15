"""
Small sensitivity check for dimension weights.

This script loads the manually labeled validation set and performs a grid search
to compare NDCG@10 on heuristic proxy labels. These labels are not hidden
ground truth and the result must not be presented as a real leaderboard metric.
"""

import json
import itertools
import math
import numpy as np
from src.scorer import score_candidate
from src.ranker import _compute_penalties, compute_behavioral_modifier, _compute_consistency_boost

DATA_PATH = r"dataset/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
VALIDATION_PATH = "validation_candidates.json"

def load_validation_set():
    with open(VALIDATION_PATH, "r") as f:
        labels = json.load(f)
        
    cids_to_labels = {}
    for entry in labels:
        if entry.get("human_tier_label") is not None:
            cids_to_labels[entry["candidate_id"]] = entry["human_tier_label"]
            
    if not cids_to_labels:
        print("No human labels found. Please fill 'human_tier_label' in validation_candidates.json")
        return None
        
    print(f"Loaded {len(cids_to_labels)} labeled candidates.")
    return cids_to_labels

def load_candidates(target_cids):
    candidates = {}
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            cid = c["candidate_id"]
            if cid in target_cids:
                candidates[cid] = c
                if len(candidates) == len(target_cids):
                    break
    return candidates

def dcg_at_k(relevances, k=10):
    relevances = relevances[:k]
    return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(relevances))

def ndcg_at_k(predicted_relevances, ideal_relevances, k=10):
    dcg = dcg_at_k(predicted_relevances, k)
    idcg = dcg_at_k(sorted(ideal_relevances, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0

def generate_weight_combinations():
    # Base weights we start with
    base_weights = {
        "template": 0.36,
        "title_career": 0.13,
        "career_jd_sim": 0.10,
        "trajectory": 0.09,
        "assessment": 0.08,
        "experience": 0.07,
        "skills": 0.05,
        "location": 0.04,
        "cross_validation": 0.05,
        "education": 0.03,
    }
    
    # Generate variations around the base
    # For speed, we just do a small grid
    variations = []
    
    # 1. Base
    variations.append(base_weights)
    
    # 2. Template heavy
    w2 = base_weights.copy()
    w2["template"] = 0.42
    w2["title_career"] = 0.10
    w2["career_jd_sim"] = 0.07
    variations.append(w2)
    
    # 3. Semantic Sim heavy
    w3 = base_weights.copy()
    w3["template"] = 0.28
    w3["career_jd_sim"] = 0.18
    variations.append(w3)
    
    # ... In a real scenario, use itertools.product to search a grid ...
    # This is a stub for the full grid search
    return variations

def evaluate_weights(candidates, labels, weights):
    # This evaluates NDCG@10 given a set of weights
    import src.scorer
    
    # Temporarily override weights
    original_weights = src.scorer.WEIGHTS.copy()
    src.scorer.WEIGHTS.update(weights)
    
    scored = []
    for cid, c in candidates.items():
        score_data = score_candidate(c)
        core_score = score_data["composite_score"]
        
        signals = c.get("redrob_signals", {})
        behavioral_mod, _ = compute_behavioral_modifier(signals)
        penalty = _compute_penalties(c, score_data)
        consistency_boost = _compute_consistency_boost(score_data)
        
        final_score = core_score * behavioral_mod * penalty * consistency_boost
        scored.append((final_score, cid))
        
    # Restore original weights
    src.scorer.WEIGHTS = original_weights
    
    scored.sort(key=lambda x: -x[0])
    
    predicted_relevances = []
    for score, cid in scored:
        predicted_relevances.append(labels[cid])
        
    return ndcg_at_k(predicted_relevances, list(labels.values()), k=10)

def run_grid_search():
    labels = load_validation_set()
    if not labels:
        return
        
    candidates = load_candidates(set(labels.keys()))
    
    variations = generate_weight_combinations()
    print(f"Testing {len(variations)} weight combinations...")
    
    best_ndcg = -1.0
    best_weights = None
    
    for i, w in enumerate(variations):
        ndcg = evaluate_weights(candidates, labels, w)
        print(f"Config {i+1}: NDCG@10 = {ndcg:.4f}")
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_weights = w
            
    print(f"\nBest proxy-label NDCG@10: {best_ndcg:.4f}")
    print("Best weights:")
    print(json.dumps(best_weights, indent=2))

    

if __name__ == "__main__":
    run_grid_search()
