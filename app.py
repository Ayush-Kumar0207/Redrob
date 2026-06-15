import gradio as gr
import json
import pandas as pd
import time
import os
import tempfile
from typing import List, Dict
import gzip

from src.ranker import rank_candidates

def load_candidates(filepath: str) -> List[Dict]:
    """Load candidates from JSON or JSONL (with optional gzip)."""
    candidates = []
    
    # Handle gzipped files
    open_fn = gzip.open if filepath.endswith('.gz') else open
    mode = 'rt' if filepath.endswith('.gz') else 'r'
    
    try:
        with open_fn(filepath, mode, encoding='utf-8') as f:
            if filepath.endswith('.jsonl') or filepath.endswith('.jsonl.gz'):
                for line in f:
                    line = line.strip()
                    if line:
                        candidates.append(json.loads(line))
            elif filepath.endswith('.json'):
                content = f.read()
                data = json.loads(content)
                if isinstance(data, list):
                    candidates = data
                else:
                    candidates = [data]
            else:
                raise ValueError("Unsupported file format. Please upload .json, .jsonl, or .jsonl.gz")
    except Exception as e:
        raise gr.Error(f"Error loading file: {str(e)}")
        
    return candidates

def run_ranking(file_obj) -> tuple[pd.DataFrame, str]:
    """Run the ranking pipeline and return results."""
    if file_obj is None:
        # Use default sample candidates if no file uploaded
        if os.path.exists("dataset/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/sample_candidates.json"):
            filepath = "dataset/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/sample_candidates.json"
        elif os.path.exists("sample_candidates.json"):
            filepath = "sample_candidates.json"
        else:
            raise gr.Error("No file uploaded and sample_candidates.json not found.")
    else:
        filepath = file_obj.name
        
    gr.Info(f"Loading candidates from {os.path.basename(filepath)}...")
    start_load = time.time()
    candidates = load_candidates(filepath)
    load_time = time.time() - start_load
    
    if not candidates:
        raise gr.Error("No candidates found in the file.")
        
    gr.Info(f"Loaded {len(candidates):,} candidates. Starting ranking pipeline...")
    start_rank = time.time()
    
    # Limit to 5000 candidates for the interactive sandbox if a huge file is uploaded
    # to keep the UX responsive. The real full pipeline runs offline.
    is_truncated = False
    if len(candidates) > 5000:
        candidates = candidates[:5000]
        is_truncated = True
        
    # Run the ranker
    results = rank_candidates(candidates)
    rank_time = time.time() - start_rank
    
    # Format for display
    df_data = []
    for r in results:
        df_data.append({
            "Rank": r["rank"],
            "Candidate ID": r["candidate_id"],
            "Score": f"{r['score']:.4f}",
            "Title": r["details"]["candidate"].get("profile", {}).get("current_title", ""),
            "Company": r["details"]["candidate"].get("profile", {}).get("current_company", ""),
            "Reasoning": r["reasoning"]
        })
        
    df = pd.DataFrame(df_data)
    
    # Generate summary text
    status_msg = f"✅ Processing complete!\n"
    if is_truncated:
        status_msg += f"⚠️ Note: Input was truncated to 5,000 candidates for the web sandbox to ensure fast response times.\n"
    status_msg += f"- Candidates processed: {len(candidates):,}\n"
    status_msg += f"- Ranking time: {rank_time:.2f} seconds\n"
    status_msg += f"- Semantic mode: {os.getenv('REDROB_SEMANTIC_MODE', 'fast')}"
    
    return df, status_msg

# --- Gradio UI ---
with gr.Blocks(title="Redrob AI Candidate Ranker (Sandbox)") as demo:
    gr.Markdown("# 🤖 Redrob AI Candidate Ranker — V7 Sandbox")
    gr.Markdown("""
    This is the interactive sandbox for evaluating the Redrob AI Ranker. 
    It runs the complete **10-dimension calibrated ranking pipeline**.
    
    Upload a `.json` or `.jsonl` candidate file, or click 'Run Ranking' to use the default 50-candidate sample.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload Candidates (.json, .jsonl, .jsonl.gz)",
                file_types=[".json", ".jsonl", ".gz"]
            )
            run_btn = gr.Button("🚀 Run Ranking", variant="primary")
            status_text = gr.Textbox(label="Status", interactive=False, lines=5)
            
        with gr.Column(scale=3):
            results_table = gr.Dataframe(
                label="Top Candidates (Ranked)",
                headers=["Rank", "Candidate ID", "Score", "Title", "Company", "Reasoning"],
                wrap=True,
                interactive=False
            )
            
    run_btn.click(
        fn=run_ranking,
        inputs=[file_input],
        outputs=[results_table, status_text]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
