import gradio as gr
import html
import json
import os
import random
from typing import List, Dict

# File paths
LABELS_FILE = "human_labels.json"
DATASET_PATH = "dataset/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"

def load_all_candidates() -> List[Dict]:
    from src.loader import load_candidates
    print(f"Loading candidates from {DATASET_PATH}...")
    cands = load_candidates(DATASET_PATH)
    return cands

def load_labels() -> dict:
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_labels(labels_dict: dict):
    with open(LABELS_FILE, "w") as f:
        json.dump(labels_dict, f, indent=2)

def escape(value) -> str:
    return html.escape(str(value)) if value is not None else ""

def truncate(text: str, max_chars: int = 420) -> str:
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."

def format_progress(labels_dict: dict) -> str:
    return f"**Progress:** {len(labels_dict)} / {len(EVAL_POOL)} candidates labeled."

# Global State
CANDIDATES = load_all_candidates()
random.seed(42) # Deterministic sample to ensure we get good overlap if restarted
EVAL_POOL = random.sample(CANDIDATES, min(500, len(CANDIDATES)))

def get_next_candidate(current_labels: dict) -> Dict:
    # Find first candidate in EVAL_POOL not in current_labels
    for cand in EVAL_POOL:
        if cand["candidate_id"] not in current_labels:
            return cand
    return None

def format_candidate_html(cand: Dict) -> str:
    if not cand:
        return "<h2 style='text-align: center; color: #10b981;'>All done. You've labeled the entire evaluation pool.</h2>"
        
    prof = cand.get("profile", {})
    signals = cand.get("redrob_signals", {})
    title = prof.get("current_title", "Unknown Title")
    company = prof.get("current_company", "Unknown Company")
    yoe = prof.get("years_of_experience", 0)
    headline = prof.get("headline", "")
    summary = truncate(prof.get("summary", ""), 360)
    location = ", ".join(
        part for part in [prof.get("location", ""), prof.get("country", "")]
        if part
    )
    salary = signals.get("expected_salary_range_inr_lpa", {})
    salary_text = "Not listed"
    if salary.get("min") is not None and salary.get("max") is not None:
        salary_text = f"{salary.get('min')} - {salary.get('max')} LPA"
    assessments = signals.get("skill_assessment_scores", {})
    assessment_text = (
        ", ".join(f"{escape(k)}: {escape(v)}" for k, v in assessments.items())
        if assessments else "No assessments"
    )
    
    html = f"""
    <div style="background: linear-gradient(145deg, #1f2937, #111827); border-radius: 12px; padding: 24px; color: #f3f4f6; box-shadow: 0 10px 25px rgba(0,0,0,0.5); border: 1px solid #374151;">
        <div style="border-bottom: 2px solid #3b82f6; padding-bottom: 12px; margin-bottom: 20px;">
            <h1 style="color: #60a5fa; margin: 0; font-size: 28px;">{escape(title)}</h1>
            <h3 style="color: #9ca3af; margin: 5px 0 0 0;">at {escape(company)} - {escape(yoe)} Years Experience</h3>
            <p style="color: #6b7280; font-size: 12px; margin: 5px 0 0 0;">ID: {escape(cand['candidate_id'])}</p>
        </div>
    """

    if headline or summary:
        html += "<div style='margin-bottom: 20px;'>"
        if headline:
            html += f"<p style='color: #dbeafe; font-weight: 700; margin: 0 0 8px 0;'>{escape(headline)}</p>"
        if summary:
            html += f"<p style='color: #cbd5e1; line-height: 1.5; margin: 0;'>{escape(summary)}</p>"
        html += "</div>"

    html += f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin-bottom: 20px;">
            <div style="background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 10px;">
                <div style="color: #94a3b8; font-size: 12px;">Location</div>
                <div style="color: #f8fafc; font-weight: 700;">{escape(location or "Unknown")}</div>
            </div>
            <div style="background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 10px;">
                <div style="color: #94a3b8; font-size: 12px;">Notice / Work Mode</div>
                <div style="color: #f8fafc; font-weight: 700;">{escape(signals.get("notice_period_days", "NA"))} days / {escape(signals.get("preferred_work_mode", "NA"))}</div>
            </div>
            <div style="background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 10px;">
                <div style="color: #94a3b8; font-size: 12px;">Expected Salary</div>
                <div style="color: #f8fafc; font-weight: 700;">{escape(salary_text)}</div>
            </div>
            <div style="background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 10px;">
                <div style="color: #94a3b8; font-size: 12px;">Recruiter Response</div>
                <div style="color: #f8fafc; font-weight: 700;">{escape(signals.get("recruiter_response_rate", "NA"))}</div>
            </div>
            <div style="background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 10px;">
                <div style="color: #94a3b8; font-size: 12px;">GitHub Activity</div>
                <div style="color: #f8fafc; font-weight: 700;">{escape(signals.get("github_activity_score", "NA"))}</div>
            </div>
            <div style="background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 10px;">
                <div style="color: #94a3b8; font-size: 12px;">Assessments</div>
                <div style="color: #f8fafc; font-weight: 700;">{assessment_text}</div>
            </div>
        </div>
    """
    
    # Skills
    skills = cand.get("skills", [])
    skill_names = [s.get("name", "") if isinstance(s, dict) else s for s in skills]
    if skill_names:
        html += "<h3 style='color: #e5e7eb;'>Core Skills</h3><div style='display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px;'>"
        for s in skill_names[:15]: # Show top 15
            html += f"<span style='background: #374151; color: #93c5fd; padding: 4px 10px; border-radius: 999px; font-size: 13px; font-weight: 500;'>{escape(s)}</span>"
        html += "</div>"
        
    # Career History
    career = cand.get("career_history", [])
    if career:
        html += "<h3 style='color: #e5e7eb;'>Career History</h3><ul style='list-style-type: none; padding-left: 0;'>"
        for role in career[:5]: # Show last 5
            c_title = role.get("title", "")
            c_company = role.get("company", "")
            industry = role.get("industry", "")
            dur = role.get("duration_months", 0)
            dur_str = f"{dur//12}y {dur%12}m" if dur > 0 else "Unknown duration"
            html += f"<li style='margin-bottom: 10px; border-left: 3px solid #4b5563; padding-left: 12px;'>"
            html += f"<strong style='color: #d1d5db;'>{escape(c_title)}</strong> at <span style='color: #9ca3af;'>{escape(c_company)}</span> "
            html += f"<span style='color: #6b7280; font-size: 13px;'>({dur_str})</span>"
            if industry:
                html += f"<div style='color: #94a3b8; font-size: 13px; margin-top: 2px;'>Industry: {escape(industry)}</div>"
            if role.get("description"):
                html += f"<p style='color: #cbd5e1; line-height: 1.45; margin: 6px 0 0 0;'>{escape(truncate(role.get('description'), 460))}</p>"
            html += f"</li>"
        html += "</ul>"
        
    html += "</div>"
    return html

def build_ui():
    with gr.Blocks() as demo:
        
        # State
        labels_state = gr.State(load_labels())
        current_cand_state = gr.State(None)
        last_label_state = gr.State(None)
        
        gr.Markdown("""
        # 🎯 Gamified Human Labeling (Senior AI Engineer)
        Rate candidates exactly how you would as a recruiter. Check role descriptions, product vs consulting history, seniority, NLP/ranking evidence, and logistics before choosing.
        * **1 - Poor Fit**: Consulting only, wrong skills, junior.
        * **3 - Okay Fit**: Missing key NLP/ranking skills, or slightly outside experience band.
        * **4 - Strong Fit**: Great skills, good experience, solid product company.
        * **5 - Exceptional Fit**: Absolute perfect fit.
        """, elem_classes="text-white")
        
        progress_text = gr.Markdown("Loading...", elem_classes="text-white")
        
        # Candidate Card
        card = gr.HTML(value="Loading...")

        with gr.Row():
            notes_input = gr.Textbox(
                label="Reviewer notes",
                placeholder="Optional: why this label? red flags? uncertainty?",
                lines=2,
            )
            confidence_input = gr.Radio(
                ["High", "Medium", "Low"],
                value="High",
                label="Confidence",
            )
            undo_btn = gr.Button("Undo Last Label")
        
        with gr.Row():
            btn_1 = gr.Button("1 - Poor Fit ❌", elem_classes="label-btn btn-poor")
            btn_3 = gr.Button("3 - Okay Fit 😐", elem_classes="label-btn btn-okay")
            btn_4 = gr.Button("4 - Strong Fit 👍", elem_classes="label-btn btn-strong")
            btn_5 = gr.Button("5 - Exceptional Fit 🚀", elem_classes="label-btn btn-exceptional")
            
        def update_ui(labels_dict):
            cand = get_next_candidate(labels_dict)
            html = format_candidate_html(cand)
            prog = format_progress(labels_dict)
            return cand, html, prog
            
        def on_rate(rating, cand, labels_dict, notes, confidence):
            if not cand:
                return cand, format_candidate_html(cand), format_progress(labels_dict), labels_dict, None, "", "High"
                
            cid = cand["candidate_id"]
            previous_label = labels_dict.get(cid)
            labels_dict[cid] = {
                "rating": rating,
                "confidence": confidence,
                "notes": (notes or "").strip(),
            }
            save_labels(labels_dict)
            
            # Load next
            next_cand = get_next_candidate(labels_dict)
            html = format_candidate_html(next_cand)
            prog = format_progress(labels_dict)
            last_label = {
                "candidate": cand,
                "candidate_id": cid,
                "previous_label": previous_label,
            }
            
            return next_cand, html, prog, labels_dict, last_label, "", "High"

        def undo_last(current_cand, labels_dict, last_label):
            if not last_label:
                return current_cand, format_candidate_html(current_cand), format_progress(labels_dict), labels_dict, None, "", "High"

            cid = last_label["candidate_id"]
            previous_label = last_label.get("previous_label")
            if previous_label is None:
                labels_dict.pop(cid, None)
            else:
                labels_dict[cid] = previous_label
            save_labels(labels_dict)

            restored_cand = last_label["candidate"]
            return restored_cand, format_candidate_html(restored_cand), format_progress(labels_dict), labels_dict, None, "", "High"

        # Load initial
        demo.load(
            fn=update_ui,
            inputs=[labels_state],
            outputs=[current_cand_state, card, progress_text]
        )
        
        # Bind buttons
        label_outputs = [current_cand_state, card, progress_text, labels_state, last_label_state, notes_input, confidence_input]
        label_inputs = [current_cand_state, labels_state, notes_input, confidence_input]
        btn_1.click(fn=lambda c, l, n, conf: on_rate(1, c, l, n, conf), inputs=label_inputs, outputs=label_outputs)
        btn_3.click(fn=lambda c, l, n, conf: on_rate(3, c, l, n, conf), inputs=label_inputs, outputs=label_outputs)
        btn_4.click(fn=lambda c, l, n, conf: on_rate(4, c, l, n, conf), inputs=label_inputs, outputs=label_outputs)
        btn_5.click(fn=lambda c, l, n, conf: on_rate(5, c, l, n, conf), inputs=label_inputs, outputs=label_outputs)
        undo_btn.click(fn=undo_last, inputs=[current_cand_state, labels_state, last_label_state], outputs=label_outputs)

    return demo

if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0", 
        server_port=int(os.getenv("REDROB_ANNOTATION_PORT", "7860")),
        share=False,
        theme=gr.themes.Base(),
        css="""
        .gradio-container { background-color: #0f172a; }
        .label-btn { font-size: 18px !important; font-weight: bold !important; padding: 15px !important; border-radius: 8px !important; transition: all 0.2s !important; }
        .label-btn:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
        .btn-poor { background-color: #ef4444 !important; color: white !important; border: none !important; }
        .btn-okay { background-color: #f59e0b !important; color: white !important; border: none !important; }
        .btn-strong { background-color: #3b82f6 !important; color: white !important; border: none !important; }
        .btn-exceptional { background-color: #10b981 !important; color: white !important; border: none !important; }
        """
    )
