# Job kit

One command for the repetitive part of a small-business forecast or lead-scoring job. Jon still decides what the job is, which columns are fair, and whether the draft is something he will send.

## Five-minute quickstart

```bash
pip install -r jobkit/requirements.txt
python -m jobkit run sales.csv --task forecast --target sales --date month --output runs/acme
python -m jobkit run leads.csv --task classify --target bought --output runs/acme-leads
```

Open `runs/acme/REPORT.md`. Edit every line that starts with `REVIEW`. Attach the chart and the CSV. Send the note yourself. The templates in `jobkit/templates/` are drafts for the emails around the job. Nothing in this kit sends mail.

Read `JOB_CHECKLIST.md` for the judgment steps and a time box for each one.

## What the command does

- Guesses dates, numbers, categories, id columns, and columns that might be the answer leaking in.
- Writes down every cleanup change: duplicates, bad dates, blank values, outliers that were kept.
- Fits simple baselines, then Holt-Winters, a small SARIMA, and gradient boosting for forecasts, or logistic regression and gradient boosting for scoring.
- Chooses the forecast method on an earlier time window, and the scoring model with stratified cross-validation. The final numbers are from a later held-out slice.
- Writes a chart, a CSV, `REPORT.md`, and `REPORT.html`. It says a model is better only when the holdout error or ranking beats the baseline.

## Limits

- A forecast is one series. Extra rows on the same date are added together.
- SARIMA is one small seasonal model, not a search over many orders.
- There is no PDF export.
- The kit does not know the client's costs, and it does not know the real meaning of a column. Those stay in the checklist.
