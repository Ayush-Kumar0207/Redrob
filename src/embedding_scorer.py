"""
Semantic Similarity Scorer using Pre-Computed MiniLM Embeddings.

This replaces the keyword-based TF-IDF matcher. It loads the pre-computed
embeddings for the JD and all candidates into memory ONCE, and performs
O(1) lookups and cosine similarity computations during ranking.
"""

from __future__ import annotations
import os
import json
import numpy as np

# Cache for loaded embeddings to avoid reloading on every candidate
_EMBEDDINGS_CACHE = None
_JD_EMB = None
_CID_TO_IDX = None

def _load_embeddings():
    """Load the pre-computed embeddings into memory."""
    global _EMBEDDINGS_CACHE, _JD_EMB, _CID_TO_IDX
    
    # Paths (relative to project root)
    cands_path = "candidates_embeddings.npy"
    cids_path = "candidates_ids.json"
    jd_path = "jd_embedding.npy"
    
    if not (os.path.exists(cands_path) and os.path.exists(cids_path) and os.path.exists(jd_path)):
        return False
        
    if _EMBEDDINGS_CACHE is None:
        print("[EmbeddingScorer] Loading pre-computed MiniLM embeddings...")
        _EMBEDDINGS_CACHE = np.load(cands_path)
        _JD_EMB = np.load(jd_path)
        
        with open(cids_path, "r") as f:
            cids = json.load(f)
        _CID_TO_IDX = {cid: i for i, cid in enumerate(cids)}
        print(f"[EmbeddingScorer] Loaded {_EMBEDDINGS_CACHE.shape[0]} embeddings.")
        
    return True


def compute_semantic_similarity(candidate_id: str) -> tuple[float, dict]:
    """
    Compute true semantic similarity using pre-computed sentence embeddings.
    
    Returns:
        (score: float in [0.0, 1.0], details: dict)
    """
    # Attempt to load embeddings if not already loaded
    if not _load_embeddings():
        return 0.4, {"error": "Embeddings not found on disk"}
        
    idx = _CID_TO_IDX.get(candidate_id)
    if idx is None:
        return 0.4, {"error": "Candidate ID not found in embeddings"}
        
    cand_emb = _EMBEDDINGS_CACHE[idx:idx+1]
    jd_emb = _JD_EMB
    
    # Compute cosine similarity
    # Both embeddings are from all-MiniLM-L6-v2 and are usually normalized, 
    # but we compute standard cosine sim just in case.
    dot = np.dot(cand_emb, jd_emb.T)[0][0]
    norm_cand = np.linalg.norm(cand_emb)
    norm_jd = np.linalg.norm(jd_emb)
    
    if norm_cand == 0 or norm_jd == 0:
        sim = 0.0
    else:
        sim = dot / (norm_cand * norm_jd)
        
    # Semantic similarity range scaling
    # MiniLM cosine similarities typically range from 0.0 to 0.8 for text.
    # A score > 0.6 is very good, < 0.2 is poor.
    # We map [0.2, 0.7] -> [0.0, 1.0]
    
    scaled_score = (sim - 0.2) / 0.5
    scaled_score = max(0.0, min(1.0, scaled_score))
    
    # Add a nonlinear boost for exceptionally high similarity (e.g., > 0.65)
    if sim > 0.65:
        scaled_score = min(1.0, scaled_score * 1.1)
        
    details = {
        "raw_cosine_similarity": float(sim),
        "semantic_match_level": "exceptional" if sim > 0.65 else "strong" if sim > 0.5 else "moderate" if sim > 0.3 else "weak",
        "used_precomputed_minilm": True
    }
    
    return round(scaled_score, 4), details
