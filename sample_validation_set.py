"""
Sample 50 candidates for manual validation labeling.

To ensure a balanced dataset, we sample candidates across different
template tiers (0 to 5).
"""

import json
import random

DATA_PATH = r"dataset/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
OUT_PATH = "validation_candidates.json"

from src.template_fingerprinter import fingerprint_career

def sample_candidates():
    # Group candidates by tier
    tiers = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], -1: []}
    
    print("Scanning candidates...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            c = json.loads(line)
            career = c.get("career_history", [])
            tier, _ = fingerprint_career(career)
            tiers[tier].append(c)
            
            if i % 20000 == 0 and i > 0:
                print(f"  Scanned {i:,}")
                
    # Sample target: 50 total
    # Distribute: 
    # Tier 5: 10
    # Tier 4: 15
    # Tier 3: 10
    # Tier 2: 5
    # Tier 1: 5
    # Tier 0: 5
    
    samples = []
    
    # We might not have 10 tier-5 candidates, so take min
    n_t5 = min(10, len(tiers[5]))
    samples.extend(random.sample(tiers[5], n_t5))
    
    n_t4 = min(15, len(tiers[4]))
    samples.extend(random.sample(tiers[4], n_t4))
    
    n_t3 = min(10, len(tiers[3]))
    samples.extend(random.sample(tiers[3], n_t3))
    
    n_t2 = min(5, len(tiers[2]))
    samples.extend(random.sample(tiers[2], n_t2))
    n_t1 = min(5, len(tiers[1]))
    samples.extend(random.sample(tiers[1], n_t1))
    n_t0 = min(5, len(tiers[0]))
    samples.extend(random.sample(tiers[0], n_t0))
    
    # Fill remaining to reach 50 from tier 4/5/3 if needed
    while len(samples) < 50:
        if tiers[4]:
            samples.append(random.choice(tiers[4]))
        else:
            samples.append(random.choice(tiers[3]))
        
    random.shuffle(samples)
    
    # Keep only the top 50
    samples = samples[:50]
    
    # Save a clean JSON for labeling
    out_data = []
    for c in samples:
        p = c["profile"]
        s = c["redrob_signals"]
        out_data.append({
            "candidate_id": c["candidate_id"],
            "title": p.get("current_title"),
            "company": p.get("current_company"),
            "yoe": p.get("years_of_experience"),
            "salary_range": s.get("expected_salary_range_inr_lpa"),
            "location": p.get("location"),
            "skills": [sk["name"] for sk in c.get("skills", [])],
            # We want the human label (0 to 5)
            "human_tier_label": None,
            "notes": ""
        })
        
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
        
    print(f"Saved {len(samples)} candidates to {OUT_PATH} for manual labeling.")


if __name__ == "__main__":
    sample_candidates()
