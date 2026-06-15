"""
Pre-compute semantic embeddings for the Redrob AI Challenge.

Uses sentence-transformers/all-MiniLM-L6-v2 to compute embeddings
for all 100K candidates and the JD. Saves the results as .npy files
to be loaded instantly during the ranking phase.

This falls under the "pre-computation loophole" allowed by the spec.
"""

import json
import time
import os
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please run: pip install sentence-transformers")
    exit(1)

from src.jd_parser import get_jd_embedding_text

MODEL_NAME = "all-MiniLM-L6-v2"
DATA_PATH = r"dataset/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
OUT_JD = "jd_embedding.npy"
OUT_CANDS = "candidates_embeddings.npy"
OUT_CIDS = "candidates_ids.json"


def build_candidate_text(candidate: dict) -> str:
    """Build a comprehensive text representation of a candidate."""
    p = candidate.get("profile", {})
    parts = []
    
    # Headline and summary
    if p.get("headline"):
        parts.append(p["headline"])
    if p.get("summary"):
        parts.append(p["summary"])
        
    # Career history descriptions
    for role in candidate.get("career_history", []):
        if role.get("description"):
            parts.append(role["description"])
            
    # Add core AI skills for semantic matching
    skills = [s["name"] for s in candidate.get("skills", [])]
    if skills:
        parts.append("Skills: " + ", ".join(skills))
        
    return " ".join(parts)


def precompute():
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    
    # 1. Embed JD
    print("Embedding JD...")
    jd_text = get_jd_embedding_text()
    jd_emb = model.encode([jd_text], convert_to_numpy=True, show_progress_bar=False)
    np.save(OUT_JD, jd_emb)
    
    # 2. Embed Candidates
    print(f"Loading candidates from {DATA_PATH}...")
    texts = []
    cids = []
    
    start = time.time()
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            c = json.loads(line)
            cids.append(c["candidate_id"])
            texts.append(build_candidate_text(c))
            
            if (i + 1) % 10000 == 0:
                print(f"  Loaded {i + 1:,} candidates...")
                
    print(f"Embedding {len(texts):,} candidates (this will take 15-30 mins)...")
    
    # Encode in batches to manage memory
    batch_size = 512
    cand_embs = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    
    print("Saving to disk...")
    np.save(OUT_CANDS, cand_embs)
    with open(OUT_CIDS, "w") as f:
        json.dump(cids, f)
        
    elapsed = time.time() - start
    print(f"DONE in {elapsed:.1f}s")
    print(f"Saved: {OUT_JD}, {OUT_CANDS}, {OUT_CIDS}")


if __name__ == "__main__":
    precompute()
