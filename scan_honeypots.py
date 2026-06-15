"""Calibrate honeypot thresholds - find the ~80 TRUE honeypots."""
import json
from datetime import datetime
from collections import Counter

f = open(r'dataset/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl','r',encoding='utf-8')

all_scores = []  # (candidate_id, impossibility_score, flags)

for line in f:
    c = json.loads(line)
    score = 0.0
    flags = []
    profile = c.get('profile', {})
    career = c.get('career_history', [])
    skills = c.get('skills', [])
    signals = c.get('redrob_signals', {})
    yoe = profile.get('years_of_experience', 0)
    
    # --- Career sum vs actual span ratio ---
    total_career = sum(r.get('duration_months', 0) for r in career)
    if career:
        dates = []
        for r in career:
            sd = r.get('start_date', '')
            if sd:
                try: dates.append(datetime.strptime(sd, '%Y-%m-%d'))
                except: pass
            ed = r.get('end_date')
            if ed:
                try: dates.append(datetime.strptime(ed, '%Y-%m-%d'))
                except: pass
        if dates:
            earliest = min(dates)
            latest = max(dates)
            span = max(1, (latest.year - earliest.year)*12 + (latest.month - earliest.month))
            ratio = total_career / span
            if ratio > 2.5:
                score += (ratio - 2.5) * 2
                flags.append(f'career_ratio={ratio:.1f}')
            elif ratio > 2.0:
                score += (ratio - 2.0) * 1
                flags.append(f'career_ratio={ratio:.1f}')
    
    # --- Skill duration exceeding experience ---
    skill_violations = 0
    worst_skill_ratio = 0
    for s in skills:
        sdur = s.get('duration_months', 0)
        if yoe > 0 and sdur > (yoe * 12) + 6 and sdur > 24:
            excess = (sdur - yoe * 12) / max(1, yoe * 12)
            worst_skill_ratio = max(worst_skill_ratio, excess)
            skill_violations += 1
    if worst_skill_ratio > 0.3:
        score += worst_skill_ratio * 2
        flags.append(f'skill_excess={worst_skill_ratio:.1f} ({skill_violations} skills)')
    
    # --- Expert skills with 0 duration ---
    expert_zero = sum(1 for s in skills 
                      if s.get('proficiency') == 'expert' and s.get('duration_months', 0) == 0)
    if expert_zero >= 5:
        score += (expert_zero - 4) * 1.5
        flags.append(f'expert_zero={expert_zero}')
    
    # --- YoE exceeds career span ---
    if career:
        starts = []
        for r in career:
            sd = r.get('start_date', '')
            if sd:
                try: starts.append(datetime.strptime(sd, '%Y-%m-%d'))
                except: pass
        if starts:
            earliest_start = min(starts)
            actual_yoe = (2026 - earliest_start.year) + (6 - earliest_start.month)/12
            if yoe > actual_yoe + 4 and yoe / max(0.1, actual_yoe) > 2.0:
                score += (yoe - actual_yoe) * 0.5
                flags.append(f'yoe_gap={yoe:.1f} vs {actual_yoe:.1f}')
    
    # --- Impossible signal values ---
    if signals.get('profile_completeness_score', 0) > 100:
        score += 3
        flags.append('completeness>100')
    if signals.get('recruiter_response_rate', 0) > 1.0:
        score += 3
        flags.append('response>1')
    if signals.get('interview_completion_rate', 0) > 1.0:
        score += 3
        flags.append('interview>1')
    
    # --- End before start ---
    for r in career:
        sd = r.get('start_date', '')
        ed = r.get('end_date')
        if sd and ed and ed < sd:
            score += 5
            flags.append('end<start')
    
    # --- Duration at role exceeds time since start ---
    for r in career:
        dur = r.get('duration_months', 0)
        sd = r.get('start_date', '')
        if sd and dur > 0:
            try:
                start = datetime.strptime(sd, '%Y-%m-%d')
                months_possible = (2026 - start.year)*12 + (6 - start.month)
                if dur > months_possible + 3:
                    excess = dur - months_possible
                    score += excess * 0.3
                    flags.append(f'dur_exceeds_time: {dur}mo but only {months_possible}mo possible')
            except:
                pass
    
    all_scores.append((c['candidate_id'], round(score, 2), flags))

f.close()

all_scores.sort(key=lambda x: -x[1])

# Show distribution
print("=== Impossibility Score Distribution ===")
thresholds = [0, 0.5, 1, 2, 3, 4, 5, 7, 10, 15, 20]
for i, t in enumerate(thresholds):
    count = sum(1 for _, s, _ in all_scores if s >= t)
    print(f"  Score >= {t:4.0f}: {count:6d} candidates")

print(f"\n=== Top 100 most impossible profiles ===")
for cid, s, fl in all_scores[:100]:
    print(f"  {cid}: score={s:6.2f} flags={fl}")

print(f"\n=== Around the ~80 honeypot mark ===")
for cid, s, fl in all_scores[70:90]:
    print(f"  {cid}: score={s:6.2f} flags={fl}")
