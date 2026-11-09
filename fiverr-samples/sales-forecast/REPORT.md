# Clothing-store sales: what the last two years say

This is a sample forecast for a clothing retailer. The numbers are not from one shop's till. They are U.S. clothing and clothing-accessories store sales, in millions of dollars, published by the Census Bureau. The pattern is the one a clothing shop knows: a quiet January and a much bigger December. The same test is what we would run on your own monthly totals.

## What we checked in the file

The Census workbook is one sheet per year, from January 1992 through August 2026 (416 months). We kept only the not-seasonally-adjusted clothing row.

- 35 seasonally adjusted copies of the same series were left out. Those already have the seasonal swing removed, so they would make a seasonal forecast look pointless.
- 34 yearly TOTAL columns were skipped. A total is not a month.
- No month failed to parse, no sales figure was blank or non-numeric, no month was duplicated, and no month was missing between the start and the end.

## How we tested it

We hid the last 24 months, September 2024 through August 2026, and forecast one month at a time. Each forecast was allowed to use sales that would already have been known, including the same month last year. It was not allowed to see the month it was predicting.

The simple baseline copies that month from the previous year. The other model is gradient boosting, using recent months, last year, and the calendar.

## How accurate

On those 24 months:

- Seasonal baseline: average miss of **1,174.5 million dollars**, which is **4.75%** of that month's sales.
- Gradient boosting: average miss of **1,698.2 million dollars**, which is **6.52%**.

The fancier model did **not** beat the baseline. It followed the shape of the year, but it dulled the December peak and added error in other months. Copying last year was the more accurate forecast.

## What to do next

Use last year's same month as the working forecast, and update it every month. Do not pay for a more complicated model until it beats that simple rule on a fresh stretch of months you held back.

When we do this on your shop, send a clean monthly total (one row per month, no yearly totals mixed in). We will repeat this exact test. If your Decembers move around more than the national series, a model may earn its place. On this public series, it did not.
