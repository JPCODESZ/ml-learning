# Lead-scoring sample

Who should a small team call first? This sample uses a public bank telemarketing dataset. The target is whether the person subscribed to a term deposit. It is lead scoring, not a telco cancellation model. The familiar IBM telco churn file does not have a clear permissive license, so it is not used here.

## Data and license

Source: Moro, S., Rita, P., and Cortez, P. (2014). Bank Marketing. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306

Direct file: https://archive.ics.uci.edu/static/public/222/data.csv

Dataset page: https://archive.ics.uci.edu/dataset/222/bank+marketing

License: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode). You may share and adapt the data for any purpose if you give credit.

Introductory paper: Moro, S., Cortez, P., and Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. Decision Support Systems. https://doi.org/10.1016/j.dss.2014.03.001

## How to run

```bash
pip install -r fiverr-samples/churn-lead-scoring/requirements.txt
python3 fiverr-samples/churn-lead-scoring/score_leads.py
```

The script downloads the CSV into `data/` (not committed), writes `figures/lead_scores.png`, and writes `output/metrics.json`.

Call duration is dropped before modeling. The baseline is "call at random" (ROC-AUC 0.50) plus a one-rule list: call only people whose previous campaign succeeded. Logistic regression and gradient boosting are compared with 5-fold cross-validation and a 20% holdout. The call-list cutoff is chosen on the training people. See `REPORT.md` for the measured results.
