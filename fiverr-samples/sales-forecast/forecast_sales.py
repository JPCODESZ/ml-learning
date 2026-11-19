"""Monthly sales forecast for U.S. clothing stores.

The series is national, not one shop's till. It is the public series a
clothing retailer would recognize: busy Decembers, quieter Januarys.
The same steps apply to a shop's own monthly export.
"""

import json
import warnings
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"
OUT_DIR = HERE / "output"
SOURCE_URL = "https://www.census.gov/retail/mrts/www/mrtssales92-present.xlsx"
WORKBOOK = DATA_DIR / "mrtssales92-present.xlsx"
SERIES_NAME = "Clothing and clothing access. stores"
# The last 24 months are the holdout. The 24 months before that are for
# choosing a model. Nothing in the choice looks at the holdout.
HOLDOUT_MONTHS = 24
VALIDATION_MONTHS = 24

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
    frame["lag_13"] = frame["sales_million"].shift(13)
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
    "lag_13",
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


def residual_columns():
    # Last year's sales are added back after the model runs, so they are not a feature.
    return [column for column in FEATURE_COLUMNS if column != "lag_12"]


def fit_residual_boost(train, window, max_depth, learning_rate):
    """Predict this year minus last year, then add last year back."""
    model = HistGradientBoostingRegressor(
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=300,
        l2_regularization=1.0,
        random_state=0,
    )
    change = train["sales_million"] - train["lag_12"]
    model.fit(train[residual_columns()], change)
    return window["lag_12"].to_numpy() + model.predict(window[residual_columns()])


def fit_yoy_boost(train, window):
    """Predict the percent change from last year, then scale last year's sales."""
    model = HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.05,
        max_iter=300,
        l2_regularization=1.0,
        random_state=0,
    )
    growth = train["sales_million"] / train["lag_12"] - 1.0
    model.fit(train[residual_columns()], growth)
    predicted_growth = model.predict(window[residual_columns()])
    return window["lag_12"].to_numpy() * (1.0 + predicted_growth)


def recent_growth_scale(window):
    """Last year, scaled by how this month's predecessor moved versus a year earlier."""
    return (window["lag_12"] * window["lag_1"] / window["lag_13"]).to_numpy()


def one_step_holt_winters(series, window_index, trend, seasonal, damped):
    """Refit on everything before each month, then forecast that one month."""
    preds = []
    for stamp in window_index:
        history = series.loc[:stamp].iloc[:-1].asfreq("MS")
        model = ExponentialSmoothing(
            history,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=12,
            damped_trend=bool(damped) if trend else False,
            initialization_method="estimated",
        )
        fitted = model.fit(optimized=True)
        value = float(np.asarray(fitted.forecast(1))[0])
        preds.append(value)
    return np.asarray(preds, dtype=float)


def one_step_sarima(series, window_index, order, seasonal_order):
    """Estimate once on the past, then step forward one month at a time."""
    first = window_index[0]
    history = series.loc[:first].iloc[:-1].asfreq("MS")
    fitted = SARIMAX(
        history,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=75)
    preds = []
    for stamp in window_index:
        value = float(np.asarray(fitted.forecast(1))[0])
        preds.append(value)
        observed = series.loc[[stamp]].asfreq("MS")
        fitted = fitted.append(observed, refit=False)
    return np.asarray(preds, dtype=float)


def score_window(actual, predicted):
    predicted = np.asarray(predicted, dtype=float)
    if len(predicted) != len(actual) or not np.isfinite(predicted).all():
        raise ValueError("Forecast was missing or not a finite number.")
    return metrics(actual, predicted)


def challengers():
    """Models we are allowed to choose among. The holdout is not an input."""
    options = [
        ("residual_boost_depth2", "residual boosting", lambda train, window, series: fit_residual_boost(train, window, 2, 0.05)),
        ("residual_boost_depth3", "residual boosting", lambda train, window, series: fit_residual_boost(train, window, 3, 0.05)),
        ("residual_boost_depth4", "residual boosting", lambda train, window, series: fit_residual_boost(train, window, 4, 0.08)),
        ("yoy_boost", "year-over-year boosting", lambda train, window, series: fit_yoy_boost(train, window)),
        ("seasonal_scaled_by_recent_growth", "year-over-year adjustment", lambda train, window, series: recent_growth_scale(window)),
        ("holt_winters_add", "Holt-Winters", lambda train, window, series: one_step_holt_winters(series, window.index, "add", "add", False)),
        ("holt_winters_mul_season", "Holt-Winters", lambda train, window, series: one_step_holt_winters(series, window.index, "add", "mul", False)),
        ("holt_winters_damped", "Holt-Winters", lambda train, window, series: one_step_holt_winters(series, window.index, "add", "add", True)),
        ("holt_winters_season_only", "Holt-Winters", lambda train, window, series: one_step_holt_winters(series, window.index, None, "add", False)),
        ("sarima_airline", "SARIMA", lambda train, window, series: one_step_sarima(series, window.index, (0, 1, 1), (0, 1, 1, 12))),
        ("sarima_110_011", "SARIMA", lambda train, window, series: one_step_sarima(series, window.index, (1, 1, 0), (0, 1, 1, 12))),
        ("sarima_111_110", "SARIMA", lambda train, window, series: one_step_sarima(series, window.index, (1, 1, 1), (1, 1, 0, 12))),
    ]
    return options


def run_candidates(train, window, series):
    scored = []
    for name, family, predictor in challengers():
        print(f"trying {name}...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                predicted = predictor(train, window, series)
            numbers = score_window(window["sales_million"], predicted)
            row = {"name": name, "family": family, **numbers}
            print(f"validation {name}: MAE {numbers['mae_million']:.2f} MAPE {numbers['mape_percent']:.2f}")
        except Exception as exc:
            row = {"name": name, "family": family, "error": f"{type(exc).__name__}: {exc}"}
            print(f"skipped {name}: {row['error']}")
        scored.append(row)
    return scored


def main():
    raw = load_sales()
    cleaned, cleanup = clean_sales(raw)
    featured = add_features(cleaned).dropna(subset=FEATURE_COLUMNS + ["sales_million"])
    needed = HOLDOUT_MONTHS + VALIDATION_MONTHS + 24
    if len(featured) <= needed:
        raise SystemExit("Not enough history for a validation window and a 24-month holdout.")

    holdout = featured.iloc[-HOLDOUT_MONTHS:]
    before_holdout = featured.iloc[:-HOLDOUT_MONTHS]
    validation = before_holdout.iloc[-VALIDATION_MONTHS:]
    fit_for_choice = before_holdout.iloc[:-VALIDATION_MONTHS]
    sales = cleaned.set_index("month")["sales_million"].sort_index().asfreq("MS")

    print(
        "Choosing a model on "
        f"{validation.index.min():%Y-%m} to {validation.index.max():%Y-%m}. "
        "The holdout is not used for this choice."
    )
    naive_validation = score_window(validation["sales_million"], validation["lag_12"])
    print(
        "validation seasonal_naive: "
        f"MAE {naive_validation['mae_million']:.2f} MAPE {naive_validation['mape_percent']:.2f}"
    )
    leaderboard = run_candidates(fit_for_choice, validation, sales)
    usable = [row for row in leaderboard if "mae_million" in row]
    if not usable:
        raise SystemExit("Every challenger failed on the validation window.")
    # Lowest validation error wins. Ties break toward the lower percent error.
    selected = min(usable, key=lambda row: (row["mae_million"], row["mape_percent"]))
    selected_predictor = dict((name, fn) for name, _family, fn in challengers())[selected["name"]]
    print(f"Selected on validation: {selected['name']}")

    # Refit using every month before the holdout, then score the holdout once.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selected_holdout = selected_predictor(before_holdout, holdout, sales)
    baseline_holdout = score_window(holdout["sales_million"], holdout["lag_12"])
    selected_holdout_score = score_window(holdout["sales_million"], selected_holdout)

    result = {
        "series": SERIES_NAME,
        "unit": "millions of U.S. dollars",
        "adjustment": "not seasonally adjusted",
        "validation_months": VALIDATION_MONTHS,
        "validation_start": validation.index.min().strftime("%Y-%m"),
        "validation_end": validation.index.max().strftime("%Y-%m"),
        "holdout_months": HOLDOUT_MONTHS,
        "holdout_start": holdout.index.min().strftime("%Y-%m"),
        "holdout_end": holdout.index.max().strftime("%Y-%m"),
        "cleanup": cleanup,
        "validation_seasonal_naive": naive_validation,
        "validation_leaderboard": leaderboard,
        "selected_model": selected["name"],
        "selected_family": selected["family"],
        "baseline_seasonal_naive": baseline_holdout,
        "selected_model_holdout": selected_holdout_score,
        "selected_beats_baseline_on_mae": selected_holdout_score["mae_million"] < baseline_holdout["mae_million"],
        "selected_beats_baseline_on_mape": selected_holdout_score["mape_percent"] < baseline_holdout["mape_percent"],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    history = featured.iloc[-(HOLDOUT_MONTHS + 12):]
    figure, axis = plt.subplots(figsize=(10.5, 5.4))
    axis.plot(history.index, history["sales_million"], color="#1f2933", linewidth=2.2, label="Actual sales")
    axis.plot(holdout.index, holdout["lag_12"], color="#7b8794", linewidth=2, linestyle="--", label="Seasonal naive")
    axis.plot(
        holdout.index,
        selected_holdout,
        color="#c2410c",
        linewidth=2.2,
        label=selected["family"],
    )
    axis.axvspan(holdout.index.min(), holdout.index.max(), color="#c2410c", alpha=0.08, label="Held-out months")
    axis.set_title("U.S. clothing stores: next-month sales forecasts")
    axis.set_ylabel("Sales (millions of dollars)")
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
