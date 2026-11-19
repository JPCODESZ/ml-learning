# Sales forecast sample

A monthly forecast for U.S. clothing stores, written so a shop owner can see the test. This is a portfolio sample. It is not a forecast for one private business, and it is not financial advice.

## Data and license

Source: U.S. Census Bureau, Monthly Retail Trade Survey, "Retail and Food Services Sales Excel (1992–present)".

File: [mrtssales92-present.xlsx](https://www.census.gov/retail/mrts/www/mrtssales92-present.xlsx)

Page: [Monthly Retail Trade - Sales Report](https://www.census.gov/retail/sales.html)

U.S. Census Bureau data are U.S. government works and are in the public domain. The series used here is **Clothing and clothing access. stores**, not seasonally adjusted, in millions of dollars.

## How to run

```bash
pip install -r fiverr-samples/sales-forecast/requirements.txt
python3 fiverr-samples/sales-forecast/forecast_sales.py
```

The script downloads the workbook into `data/` (that folder is not committed), writes `figures/forecast_holdout.png`, and writes `output/metrics.json`.

The last 24 months are the holdout. The 24 months before that are only for choosing a model. The candidates are a seasonal-naive baseline, residual and year-over-year gradient boosting, Holt-Winters, and SARIMA. On the run saved in this folder, Holt-Winters with multiplicative seasonality was the best alternative on the validation window and then beat the seasonal baseline on the holdout (MAE 993.2 vs 1,174.5 million dollars; MAPE 3.84% vs 4.75%). See `REPORT.md`.
