# Redrob AI Candidate Ranker

A deterministic, CPU-only candidate ranking system for the IndiaRuns / Redrob
Data & AI Challenge. It ranks 100,000 profiles for the **Senior AI Engineer -
Founding Team** role and produces a recruiter-readable top-100 CSV.

- **Team:** The Indic Protocol
- **Participant ID:** `686b95d376ce14837de12b88`
- **Repository:** <https://github.com/Ayush-Kumar0207/Redrob>

## Verified Snapshot

Measured on the full released pool on Windows 11 / Python 3.12:

| Check | Result |
|---|---:|
| Candidates ranked | 100,000 |
| Native CPU runtime | 84.6 seconds; deterministic repeat: 82.3 seconds |
| Docker reproduction runtime | 162.8 seconds |
| Network calls during ranking | 0 |
| GPU required | No |
| Detected decisive integrity contradictions | 57 |
| Detected honeypots in submitted top 100 | 0 |
| Very-low-availability profiles in top 100 | 0 |
| Unique reasoning strings | 100 / 100 |
| Exact-match template tier in top 10 | 10 / 10 |
| Deterministic CSV SHA-256 | `5D3502CE69682631A433D3F7FD0E8A710CD01003B4E7FFE2B9648A3B8529B6A1` |

The template-tier counts are a **challenge-specific diagnostic**, not hidden
ground truth. There is no public leaderboard or official local label set.

## Reproduce

```bash
python -m pip install -r requirements-ranking.txt
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
python validate_submission.py submission.csv
python audit_submission.py --submission submission.csv --candidates ./candidates.jsonl
python -m unittest discover -s tests -v
```

The default command uses the fast deterministic semantic-evidence scorer. It
does not require model downloads, ignored embedding files, a GPU, or network
access.

### Docker

```bash
docker build -t redrob-ranker .
docker run --rm -v "$PWD:/data" redrob-ranker \
  --candidates /data/candidates.jsonl --out /data/submission.csv
```

The final `indic-protocol-redrob:submission` image was built from a clean
`python:3.12-slim` base and completed the full 100,000-profile run in 162.8
seconds. Its output passed the official validator and exactly matched the native
submission SHA-256.

### Small-Sample Demo

`app.py` is a Gradio sandbox that accepts JSON, JSONL, or gzipped JSONL. A
50-profile `sample_candidates.json` is included so the demo works without the
private full dataset.

## Ranking Design

The ranker first estimates **technical relevance**, then adjusts for whether the
profile is credible and realistically hireable.

```text
candidate
  -> profile/career parsing
  -> 10-dimensional calibrated fit score
  -> behavioral availability modifier
  -> integrity and anti-stuffing checks
  -> deterministic ranking
  -> evidence-grounded reasoning
```

### Calibrated Fit Score

| Dimension | Weight | Purpose |
|---|---:|---|
| Synthetic template prior | 0.36 | Separates exact ranking/retrieval careers from adjacent and trap templates |
| Title and career fit | 0.13 | Rewards hands-on AI/search roles and product-company experience |
| Career-JD evidence | 0.10 | Fast structured evidence for ranking, retrieval, production, and embeddings |
| Career trajectory | 0.09 | Recent coding, stability, and movement toward applied AI |
| Assessments | 0.08 | Hard platform evidence on JD-relevant skills |
| Experience band | 0.07 | Soft preference for the JD's 5-9 year range |
| Skills match | 0.05 | Trusted core skills, with anti-stuffing checks |
| Skills-career validation | 0.05 | Checks whether claims appear in career evidence |
| Location and logistics | 0.04 | India location, notice period, and work-mode fit |
| Education | 0.03 | Relevant field, degree, and institution tier |

The template prior is intentionally dominant because the released synthetic
pool contains repeated career-description families with a clear relevance
ladder. It is not presented as a general-purpose production feature.

### Behavioral Availability

Behavioral signals are used once in a weighted score and converted to a modest
`0.55x-1.20x` modifier. Recruiter response rate, recent activity, open-to-work,
interview reliability, notice period, and response speed carry the most weight.
Explicit gates prevent profile views or connection counts from rescuing someone
who is plainly unreachable.

### Integrity Checks

The integrity scorer targets decisive cross-field contradictions:

- summary experience contradicts profile experience
- role duration contradicts its start/end dates
- claimed experience is impossible given the career span
- many expert skills claim zero months of use
- signal or assessment values break their documented ranges

Minor timeline noise and normal overlapping roles are not penalized. This
avoids the broad false positives produced by naive duration-sum checks.

### Semantic Modes

```bash
# Default: fast, deterministic, no artifacts
python rank.py --semantic-mode fast --candidates ./candidates.jsonl --out submission.csv

# Slower sklearn comparison mode
python rank.py --semantic-mode tfidf --candidates ./candidates.jsonl --out submission.csv

# Optional pre-computed MiniLM artifacts
python precompute_embeddings.py
python rank.py --semantic-mode embeddings --candidates ./candidates.jsonl --out submission.csv
```

Only the default `fast` mode produced the submitted CSV. Optional embeddings are
for experimentation and are not required for Stage 3 reproduction.

## Evaluation Discipline

The challenge does not provide labels or a live leaderboard. Local
`validation_candidates.json` labels are heuristic proxies and are explicitly
treated as sensitivity checks, not real NDCG evidence. Submission readiness is
therefore evaluated through:

- official format validation
- full-pool runtime reproduction
- deterministic output
- top-N profile inspection
- honeypot and availability audits
- reasoning specificity and uniqueness
- focused unit tests for high-risk scoring behavior

## Repository Map

```text
rank.py                     production CLI
src/scorer.py               calibrated 10-dimensional fit score
src/ranker.py               final modifiers, ranking, and CSV output
src/behavioral.py           non-duplicated availability scoring
src/integrity_scorer.py     decisive honeypot checks
src/reasoning.py            evidence-grounded reasoning
audit_submission.py         top-N quality and integrity audit
validate_submission.py      official format validator
app.py                      Gradio small-sample sandbox
tests/test_core.py          focused scoring and integrity tests
Dockerfile                  CPU-only reproduction image
annotation_app.py           optional manual-labeling interface
auto_label.py               proxy-label bootstrap utility
eval_stage2.py              proxy-label sensitivity evaluation
grid_search.py              proxy-label weight sensitivity search
scan_honeypots.py           exploratory integrity-threshold audit
inspect_submission.py       recruiter-style shortlist inspection
SUBMISSION_CHECKLIST.md     remaining portal and publishing tasks
```

Final deliverables are written to `submission.csv`,
`686b95d376ce14837de12b88.csv`,
`outputs/redrob_ai_ranker_submission_deck.pptx`, and
`output/pdf/redrob_ai_ranker_submission_deck.pdf`.

The editable PPTX follows the organizer-provided 11-slide Idea Submission
Template. The verified upload PDF contains 11 pages and is 1.64 MB, below the
Hack2Skill portal's 5 MB limit. Upload `686b95d376ce14837de12b88.csv` as the
ranked output file and `output/pdf/redrob_ai_ranker_submission_deck.pdf` as the
deck PDF.

## Important Limitations

- Hidden-ground-truth performance cannot be known before organizers score it.
- Template fingerprinting is intentionally challenge-specific and would be
  replaced by learned relevance features in a real production system.
- The public reproducibility URL points to this repository's Docker workflow;
  the ranking system itself remains fully offline during execution.

## License

MIT
