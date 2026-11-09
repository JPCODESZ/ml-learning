# Clothing-store sales: what the last two years say

This is a sample forecast for a clothing retailer. The numbers are not from one shop's till. They are U.S. clothing and clothing-accessories store sales, in millions of dollars, published by the Census Bureau. The pattern is the one a clothing shop knows: a quiet January and a much bigger December. The same test is what we would run on your own monthly totals.

## What we checked in the file

The Census workbook is one sheet per year, from January 1992 through August 2026 (416 months). We kept only the not-seasonally-adjusted clothing row.

- 35 seasonally adjusted copies of the same series were left out. Those already have the seasonal swing removed, so they would make a seasonal forecast look pointless.
- 34 yearly TOTAL columns were skipped. A total is not a month.
- No month failed to parse, no sales figure was blank or non-numeric, no month was duplicated, and no month was missing between the start and the end.

## How we chose the model

We hid the last 24 months, September 2024 through August 2026. Those months were not used to pick a method.

On the 24 months before that, September 2022 through August 2024, we compared:

- copying the same month from the year before (the seasonal baseline)
- gradient boosting of the change from last year
- scaling last year by the most recent year-over-year change
- Holt-Winters (several seasonal settings)
- a few seasonal ARIMA models

Copying last year was the most accurate on that earlier window: average miss **482.2 million dollars**, or **2.11%**. The closest alternative was Holt-Winters with a multiplicative seasonal pattern: average miss **651.4 million dollars**, or **2.68%**. That is the method we froze and carried forward. The other alternatives were less accurate on that window, so they were not given a second look on the hidden months.

## How accurate on the hidden months

Each forecast is for the next month only, using sales that would already be known.

On September 2024 through August 2026:

- Seasonal baseline: average miss of **1,174.5 million dollars**, which is **4.75%** of that month's sales.
- Holt-Winters: average miss of **993.2 million dollars**, which is **3.84%**.

Holt-Winters beat the baseline on both measures. It still overshot the biggest December. The gain is a smaller miss in a typical month, not a perfect holiday forecast.

## What to do next

Use Holt-Winters as the working next-month forecast, and keep last year's number beside it. It won the hidden two years, but it lost the two years we used to choose it, so this is one good check rather than a permanent trophy.

When we do this on your shop, send a clean monthly total (one row per month, no yearly totals mixed in). We will repeat the same split: choose on an earlier window, then score the choice once on months it has not seen.
