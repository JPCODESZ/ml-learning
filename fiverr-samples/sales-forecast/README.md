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

The holdout is the last 24 months. The baseline is seasonal naive (same month last year). The comparison model is gradient boosting with lags and calendar features. On the run saved in this folder, the baseline was more accurate. See `REPORT.md`.
