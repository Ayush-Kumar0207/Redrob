# Submission Checklist

## Must Complete Before Upload

- [ ] Rename `submission.csv` to your registered participant ID, for example `team_xxx.csv`.
- [ ] Replace every placeholder in `submission_metadata.yaml` with real team, GitHub, and sandbox details.
- [ ] Declare all AI tools used, including ChatGPT/Codex.
- [ ] Create and push a reachable GitHub repository.
- [ ] Start Docker Desktop, build the image, and run a full container reproduction.
- [ ] Publish a working sandbox or public Docker image link.
- [ ] Export the final presentation deck to PDF and upload it with the ranked CSV.

## Reproduction Gate

```bash
python -m pip install -r requirements-ranking.txt
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
python validate_submission.py submission.csv
python audit_submission.py --submission submission.csv --candidates ./candidates.jsonl
python -m unittest discover -s tests -v
```

Expected full-pool runtime on the development machine: under 5 minutes, CPU-only,
with no network calls. The default `fast` semantic mode is deterministic and does
not require the ignored embedding artifacts.

## Manual Review Gate

- [ ] Read at least the top 20 profiles and verify the rank ordering.
- [ ] Spot-check at least 10 random reasoning strings against source profiles.
- [ ] Confirm there are zero detected honeypots in the top 100.
- [ ] Confirm the final Git history reflects real iteration and is not a single code dump.
- [ ] Rehearse a 5-minute architecture walkthrough and explain every scoring signal.
