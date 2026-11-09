"""Monthly sales forecast for U.S. clothing stores.

The series is national, not one shop's till. It is the public series a
clothing retailer would recognize: busy Decembers, quieter Januarys.
The same steps apply to a shop's own monthly export.
"""

import json
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"
OUT_DIR = HERE / "output"
SOURCE_URL = "https://www.census.gov/retail/mrts/www/mrtssales92-present.xlsx"
WORKBOOK = DATA_DIR / "mrtssales92-present.xlsx"
SERIES_NAME = "Clothing and clothing access. stores"
HOLDOUT_MONTHS = 24

MONTH_NUMBERS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def download_workbook():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if WORKBOOK.exists() and WORKBOOK.stat().st_size > 0:
        print(f"Using saved workbook {WORKBOOK.name}")
        return
    print("Downloading the Census retail workbook...")
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "ml-learning-fiverr-sample/1.0 (educational portfolio)"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        WORKBOOK.write_bytes(response.read())


def month_headers(row, year):
    """Turn 'Jan. 2024' and 'May 2024' into real month-start dates."""
    dates = []
    for value in row:
        if not isinstance(value, str):
            dates.append(None)
            continue
        token = value.replace(".", " ").split()
        if not token:
            dates.append(None)
            continue
        month = MONTH_NUMBERS.get(token[0][:3].lower())
        if month is None:
            dates.append(None)
            continue
        dates.append(pd.Timestamp(year=int(year), month=month, day=1))
    return dates


def series_from_sheet(frame, year):
    labels = frame.iloc[:, 1].fillna("").astype(str).str.strip()
    not_adjusted = labels[labels.str.upper() == "NOT ADJUSTED"]
    adjusted = labels[labels.str.upper().str.startswith("ADJUSTED")]
    if not_adjusted.empty or adjusted.empty:
        return []
    start = int(not_adjusted.index[0])
    end = int(adjusted.index[0])
    matches = labels[(labels == SERIES_NAME) & (labels.index > start) & (labels.index < end)]
    if matches.empty:
        return []
    row_index = int(matches.index[0])
    # The month names sit a few rows above the numbers, on the same columns.
    header_row = None
    for candidate in range(start, -1, -1):
        dates = month_headers(frame.iloc[candidate, :], year)
        if sum(item is not None for item in dates) >= 6:
            header_row = dates
            break
    if header_row is None:
        return []
    points = []
    for column, date in enumerate(header_row):
        if date is None:
            continue
        points.append((date, frame.iat[row_index, column]))
    return points


def load_sales():
    download_workbook()
    workbook = pd.ExcelFile(WORKBOOK)
    rows = []
    notes = {
        "sheets_read": 0,
        "adjusted_values_left_out": 0,
        "total_columns_left_out": 0,
    }
    for sheet in workbook.sheet_names:
        if not str(sheet).isdigit():
            continue
        notes["sheets_read"] += 1
        frame = pd.read_excel(workbook, sheet, header=None)
        labels = frame.iloc[:, 1].fillna("").astype(str).str.strip()
        adjusted_rows = labels[labels.str.upper().str.startswith("ADJUSTED")]
        if not adjusted_rows.empty:
            adjusted_block = frame.iloc[int(adjusted_rows.index[0]) :]
            clothing = adjusted_block.iloc[:, 1].fillna("").astype(str).str.strip() == SERIES_NAME
            notes["adjusted_values_left_out"] += int(clothing.sum())
        header_hits = frame.iloc[4].astype(str).str.upper().eq("TOTAL").sum() if frame.shape[0] > 4 else 0
        notes["total_columns_left_out"] += int(header_hits)
        rows.extend(series_from_sheet(frame, int(sheet)))
    raw = pd.DataFrame(rows, columns=["month", "sales_raw"])
    raw["sales_million"] = pd.to_numeric(raw["sales_raw"], errors="coerce")
    raw.attrs["notes"] = notes
    return raw


def clean_sales(raw):
    """Show the cleanup instead of hiding it: blanks, duplicates, broken dates."""
    report = {
        "rows_read": int(len(raw)),
        "unparseable_dates": int(raw["month"].isna().sum()),
        "non_numeric_sales": int(raw["sales_million"].isna().sum()),
        "sheets_read": int(raw.attrs.get("notes", {}).get("sheets_read", 0)),
        "adjusted_values_left_out": int(raw.attrs.get("notes", {}).get("adjusted_values_left_out", 0)),
        "total_columns_left_out": int(raw.attrs.get("notes", {}).get("total_columns_left_out", 0)),
    }
    cleaned = raw.dropna(subset=["month"]).copy()
    cleaned = cleaned.dropna(subset=["sales_million"])
    cleaned["month"] = pd.to_datetime(cleaned["month"])
    duplicate_count = int(cleaned["month"].duplicated().sum())
    report["duplicate_months_removed"] = duplicate_count
    cleaned = cleaned.sort_values("month").drop_duplicates("month", keep="last")
    full_index = pd.date_range(cleaned["month"].min(), cleaned["month"].max(), freq="MS")
    missing_months = full_index.difference(pd.DatetimeIndex(cleaned["month"]))
    report["missing_months"] = [item.strftime("%Y-%m") for item in missing_months]
    report["months_kept"] = int(len(cleaned))
    report["start"] = cleaned["month"].min().strftime("%Y-%m")
    report["end"] = cleaned["month"].max().strftime("%Y-%m")
    return cleaned.reset_index(drop=True), report


def add_features(sales):
    frame = sales.set_index("month").sort_index().copy()
    frame["lag_1"] = frame["sales_million"].shift(1)
    frame["lag_2"] = frame["sales_million"].shift(2)
    frame["lag_3"] = frame["sales_million"].shift(3)
    frame["lag_12"] = frame["sales_million"].shift(12)
    frame["roll_mean_3"] = frame["sales_million"].shift(1).rolling(3).mean()
    frame["roll_mean_12"] = frame["sales_million"].shift(1).rolling(12).mean()
    frame["month_number"] = frame.index.month
    frame["month_sin"] = np.sin(2 * np.pi * frame["month_number"] / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame["month_number"] / 12)
    frame["trend"] = np.arange(len(frame))
    return frame


FEATURE_COLUMNS = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_12",
    "roll_mean_3",
    "roll_mean_12",
    "month_sin",
    "month_cos",
    "trend",
]


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    mae = float(np.mean(np.abs(error)))
    mape = float(np.mean(np.abs(error) / np.abs(actual)) * 100)
    return {"mae_million": mae, "mape_percent": mape}


def main():
    raw = load_sales()
    cleaned, cleanup = clean_sales(raw)
    featured = add_features(cleaned).dropna(subset=FEATURE_COLUMNS + ["sales_million"])
    if len(featured) <= HOLDOUT_MONTHS + 24:
        raise SystemExit("Not enough history for a 24-month holdout.")

    train = featured.iloc[:-HOLDOUT_MONTHS]
    test = featured.iloc[-HOLDOUT_MONTHS:]
    # Seasonal naive: this month last year. That number is known before the month begins.
    seasonal = test["lag_12"].to_numpy()
    model = HistGradientBoostingRegressor(
        max_depth=4,
        learning_rate=0.08,
        max_iter=400,
        l2_regularization=0.1,
        random_state=0,
    )
    model.fit(train[FEATURE_COLUMNS], train["sales_million"])
    # One month ahead, using only sales that would already be known.
    boosted = model.predict(test[FEATURE_COLUMNS])

    baseline = metrics(test["sales_million"], seasonal)
    better = metrics(test["sales_million"], boosted)
    result = {
        "series": SERIES_NAME,
        "unit": "millions of U.S. dollars",
        "adjustment": "not seasonally adjusted",
        "holdout_months": HOLDOUT_MONTHS,
        "holdout_start": test.index.min().strftime("%Y-%m"),
        "holdout_end": test.index.max().strftime("%Y-%m"),
        "cleanup": cleanup,
        "baseline_seasonal_naive": baseline,
        "gradient_boosting": better,
        "boosting_beats_baseline_on_mae": better["mae_million"] < baseline["mae_million"],
        "boosting_beats_baseline_on_mape": better["mape_percent"] < baseline["mape_percent"],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    history = featured.iloc[-36:]
    figure, axis = plt.subplots(figsize=(10.5, 5.4))
    axis.plot(history.index, history["sales_million"], color="#1f2933", linewidth=2.2, label="Actual sales")
    axis.plot(test.index, seasonal, color="#7b8794", linewidth=2, linestyle="--", label="Seasonal naive")
    axis.plot(test.index, boosted, color="#c2410c", linewidth=2.2, label="Gradient boosting")
    axis.axvspan(test.index.min(), test.index.max(), color="#c2410c", alpha=0.08, label="Held-out months")
    axis.set_title("U.S. clothing stores: next-month sales forecasts")
    axis.set_ylabel("Sales (millions of dollars)")
    axis.set_xlabel("")
    axis.legend(frameon=False, loc="upper left")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#e6e8eb")
    figure.text(
        0.01,
        0.01,
        "Source: U.S. Census Bureau, Monthly Retail Trade Survey, not seasonally adjusted. Public domain.",
        fontsize=8,
        color="#52606d",
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    chart = FIG_DIR / "forecast_holdout.png"
    figure.savefig(chart, dpi=140)
    plt.close(figure)

    print(json.dumps(result, indent=2))
    print(f"Chart: {chart}")


if __name__ == "__main__":
    main()
