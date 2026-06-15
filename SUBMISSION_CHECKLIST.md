# Submission Checklist

## Must Complete Before Upload

- [x] Create `686b95d376ce14837de12b88.csv` using the registered participant ID.
- [x] Replace every placeholder in `submission_metadata.yaml` with verified team and repository details.
- [x] Declare all AI tools used, including ChatGPT/Codex.
- [x] Create a reachable public GitHub repository.
- [x] Build the Docker image and run a full container reproduction.
- [x] Publish a public Docker reproduction workflow through the GitHub repository.
- [x] Rebuild the deck inside the organizer-provided 11-slide PPT template.
- [x] Export and visually verify the 11-page final PDF.
- [x] Confirm the final PDF is below the portal's 5 MB limit (1.64 MB).
- [x] Revalidate the participant-ID CSV with the official validator and audit.
- [ ] Upload the participant-ID CSV and final PDF to the Hack2Skill portal.

## Portal Upload Files

- GitHub URL: `https://github.com/Ayush-Kumar0207/Redrob`
- Deck PDF: `output/pdf/redrob_ai_ranker_submission_deck.pdf`
- Ranked output: `686b95d376ce14837de12b88.csv`

The ranked output is a CSV, which satisfies the portal's accepted CSV/XLSX
format.

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
