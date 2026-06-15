"""V5 sorted top-10 with template tiers."""
import csv, json

ranks = {}
with open('submission.csv', 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        ranks[row['candidate_id']] = {
            'rank': int(row['rank']),
            'score': row['score'],
            'reasoning': row['reasoning']
        }

data_path = r'dataset/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl'
top10 = {}
with open(data_path, 'r', encoding='utf-8') as f:
    for line in f:
        c = json.loads(line)
        cid = c['candidate_id']
        if cid in ranks and ranks[cid]['rank'] <= 15:
            top10[ranks[cid]['rank']] = c

# Template fingerprinting
from src.template_fingerprinter import fingerprint_career

for rank in sorted(top10.keys()):
    c = top10[rank]
    p = c['profile']
    s = c['redrob_signals']
    sal = s.get('expected_salary_range_inr_lpa', {})
    tier, match = fingerprint_career(c.get('career_history', []))
    
    print(f"#{rank} {c['candidate_id']} (score={ranks[c['candidate_id']]['score']})")
    print(f"  {p.get('current_title')} at {p.get('current_company')} ({p.get('current_industry')})")
    print(f"  TEMPLATE TIER: {tier}")
    print(f"  YoE={p.get('years_of_experience')}  Loc={p.get('location')}  Salary={sal.get('min',0):.0f}-{sal.get('max',0):.0f} LPA")
    print(f"  Notice={s.get('notice_period_days')}d  Mode={s.get('preferred_work_mode')}  Open={s.get('open_to_work_flag')}")
    print(f"  Response={s.get('recruiter_response_rate',0):.0%}  OfferAccept={s.get('offer_acceptance_rate',-1)}")
    print(f"  Assessments: {s.get('skill_assessment_scores', {})}")
    print()
