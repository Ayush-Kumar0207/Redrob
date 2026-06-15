import json
import math
from src.ranker import rank_candidates
from src.loader import load_candidates

def compute_dcg(relevances):
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))

def main():
    # Load the labeled validation candidates
    with open("validation_candidates.json", "r") as f:
        val_cands = json.load(f)
        
    # We need full candidate details for the pipeline
    # The validation_candidates.json only has minimal info, so we need to merge with the raw data
    print("Loading 100K candidates for full profiles...")
    all_cands = load_candidates("dataset/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl")
    cand_map = {c["candidate_id"]: c for c in all_cands}
    
    # Get labels
    labels = {vc["candidate_id"]: vc.get("human_tier_label", 1) for vc in val_cands}
    
    # Build full candidate objects for the 50 validation ones
    eval_cands = []
    for cid in labels.keys():
        if cid in cand_map:
            eval_cands.append(cand_map[cid])
            
    print(f"Running production ranker on {len(eval_cands)} proxy-labeled candidates...")
    
    # Run the same default pipeline used for submission generation.
    results = rank_candidates(eval_cands)
    
    # Calculate NDCG@10
    k = 10
    top_k_results = results[:k]
    
    # Get relevances of our top 10
    system_relevances = [labels.get(r["candidate_id"], 1) for r in top_k_results]
    
    # Calculate ideal ranking (sort all 50 by label descending)
    ideal_sorted_cids = sorted(labels.keys(), key=lambda x: labels[x], reverse=True)
    ideal_relevances = [labels[cid] for cid in ideal_sorted_cids[:k]]
    
    dcg = compute_dcg(system_relevances)
    idcg = compute_dcg(ideal_relevances)
    
    ndcg = dcg / idcg if idcg > 0 else 0
    
    print("\n" + "="*50)
    print(f"Proxy-label NDCG@10 (not hidden ground truth): {ndcg:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
