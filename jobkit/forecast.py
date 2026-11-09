"""Next-period forecast from a client sales file."""

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


def _frequency(dates):
    gaps = dates.sort_values().diff().dt.days.dropna()
    typical = float(gaps.median()) if len(gaps) else 30
    if typical <= 2:
        return "D", 7, "daily"
    if typical <= 9:
        return "W-MON", 52, "weekly"
    return "MS", 12, "monthly"


def build_series(frame, target, date_column, log):
    dated = frame[[date_column, target]].dropna().sort_values(date_column)
    freq, season, label = _frequency(dated[date_column])
    dated["bucket"] = dated[date_column].dt.to_period(freq[0] if freq != "W-MON" else "W").dt.to_timestamp()
    if freq == "W-MON":
        dated["bucket"] = dated[date_column].dt.to_period("W").dt.start_time
    duplicates = int(dated["bucket"].duplicated().sum())
    if duplicates:
        log.add(
            "added up rows that share a period",
            f"{duplicates} extra rows were summed into the {label} total",
        )
    series = dated.groupby("bucket")[target].sum().sort_index()
    series.index = pd.DatetimeIndex(series.index)
    full = series.asfreq("W-MON" if freq == "W-MON" else freq)
    missing = int(full.isna().sum())
    if missing:
        filled = full.interpolate(limit=2)
        still_missing = int(filled.isna().sum())
        used = missing - still_missing
        if used:
            log.add("filled short gaps", f"{used} missing {label} periods were filled with a straight line")
        if still_missing:
            log.add("left long gaps blank", f"{still_missing} periods were dropped rather than invented")
        full = filled.dropna()
    else:
        log.add("checked the calendar", f"no missing {label} periods")
    q1, q3 = full.quantile([0.25, 0.75])
    span = q3 - q1
    if span > 0:
        outliers = int(((full < q1 - 3 * span) | (full > q3 + 3 * span)).sum())
        log.add(
            "flagged outliers",
            f"{outliers} periods sit far from the middle. They were kept, because a real spike is not an error by itself.",
        )
    full.index.freq = pd.tseries.frequencies.to_offset("W-MON" if freq == "W-MON" else freq)
    return full.astype(float), season, label


def _features(series, season):
    frame = pd.DataFrame(index=series.index)
    frame["lag_1"] = series.shift(1)
    frame["lag_season"] = series.shift(season)
    frame["roll_3"] = series.shift(1).rolling(3).mean()
    frame["calendar"] = series.index.month if season == 12 else series.index.dayofyear
    frame["trend"] = np.arange(len(series))
    return frame


def _metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    return {
        "mae": float(np.mean(np.abs(error))),
        "mape": float(np.mean(np.abs(error) / np.clip(np.abs(actual), 1e-9, None)) * 100),
    }


def _window_sizes(n, season):
    holdout = min(max(season, 8), 24, max(n // 5, 1))
    validation = holdout
    if n - holdout - validation < max(season * 2, 18):
        raise SystemExit("This file does not have enough history for an honest holdout.")
    return validation, holdout


def _seasonal_naive(series, window, season):
    return series.shift(season).loc[window].to_numpy(dtype=float)


def _last_value(series, window, season):
    return series.shift(1).loc[window].to_numpy(dtype=float)


def _residual_boost(series, train_end, window, season):
    features = _features(series, season)
    columns = ["lag_1", "lag_season", "roll_3", "calendar", "trend"]
    change = series - series.shift(season)
    trainable = features.index[features.index <= train_end]
    usable = trainable[features.loc[trainable, columns].notna().all(axis=1) & change.loc[trainable].notna()]
    model = HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.08,
        max_iter=200,
        l2_regularization=1.0,
        random_state=0,
    )
    model.fit(features.loc[usable, columns], change.loc[usable])
    predicted_change = model.predict(features.loc[window, columns])
    return series.shift(season).loc[window].to_numpy(dtype=float) + predicted_change


def _holt_winters(series, window, season, seasonal):
    preds = []
    for stamp in window:
        history = series.loc[:stamp].iloc[:-1]
        if len(history) < season * 2:
            raise ValueError("not enough history")
        if seasonal == "mul" and (history <= 0).any():
            raise ValueError("multiplicative seasonality needs positive sales")
        model = ExponentialSmoothing(
            history,
            trend="add",
            seasonal=seasonal,
            seasonal_periods=season,
            initialization_method="estimated",
        )
        fitted = model.fit(optimized=True)
        preds.append(float(np.asarray(fitted.forecast(1)).ravel()[0]))
    return np.asarray(preds, dtype=float)


def _sarima(series, train_end, window, season):
    history = series.loc[:train_end]
    fitted = SARIMAX(
        history,
        order=(0, 1, 1),
        seasonal_order=(0, 1, 1, season),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=40)
    preds = []
    for stamp in window:
        value = float(np.asarray(fitted.forecast(1)).ravel()[0])
        preds.append(value)
        fitted = fitted.append(series.loc[[stamp]], refit=False)
    return np.asarray(preds, dtype=float)


def _score(actual, predicted):
    predicted = np.asarray(predicted, dtype=float)
    if len(predicted) != len(actual) or not np.isfinite(predicted).all():
        raise ValueError("forecast was not finite")
    return _metrics(actual, predicted)


def _candidates(series, season):
    options = [
        ("seasonal_naive", "baseline", lambda train_end, window: _seasonal_naive(series, window, season)),
        ("last_value", "baseline", lambda train_end, window: _last_value(series, window, season)),
        ("residual_boosting", "gradient boosting", lambda train_end, window: _residual_boost(series, train_end, window, season)),
    ]
    if len(series) >= season * 3:
        options.append(
            ("holt_winters_add", "Holt-Winters", lambda train_end, window: _holt_winters(series, window, season, "add"))
        )
        options.append(
            ("holt_winters_mul", "Holt-Winters", lambda train_end, window: _holt_winters(series, window, season, "mul"))
        )
    if len(series) >= season * 4:
        options.append(
            ("sarima_lite", "SARIMA", lambda train_end, window: _sarima(series, train_end, window, season))
        )
    return options


def run_forecast(frame, target, date_column, log, output_dir):
    series, season, frequency_label = build_series(frame, target, date_column, log)
    validation_n, holdout_n = _window_sizes(len(series), season)
    holdout_index = series.index[-holdout_n:]
    validation_index = series.index[-(holdout_n + validation_n) : -holdout_n]
    train_end_for_choice = series.index[-(holdout_n + validation_n) - 1]
    train_end_for_holdout = series.index[-holdout_n - 1]

    def judge(train_end, window):
        rows = []
        predictions = {}
        actual = series.loc[window].to_numpy(dtype=float)
        for name, family, predictor in _candidates(series, season):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    predicted = predictor(train_end, window)
                score = _score(actual, predicted)
                rows.append({"name": name, "family": family, **score})
                predictions[name] = predicted
            except Exception as exc:
                rows.append({"name": name, "family": family, "error": f"{type(exc).__name__}: {exc}"})
        return rows, predictions

    validation_rows, _ = judge(train_end_for_choice, validation_index)
    usable = [row for row in validation_rows if "mae" in row]
    if not usable:
        raise SystemExit("No forecast method produced a number on the validation months.")
    selected = min(usable, key=lambda row: (row["mae"], row["mape"]))
    selected_name = selected["name"]
    holdout_rows, holdout_predictions = judge(train_end_for_holdout, holdout_index)
    by_name = {row["name"]: row for row in holdout_rows if "mae" in row}
    baseline_name = "seasonal_naive" if "seasonal_naive" in by_name else "last_value"
    baseline = by_name[baseline_name]
    chosen = by_name[selected_name]
    beats = selected_name != baseline_name and chosen["mae"] < baseline["mae"] and chosen["mape"] <= baseline["mape"] + 1e-9
    # A model is better only when the holdout error is smaller. A validation win is not enough.
    if selected_name == baseline_name:
        beats = False

    recommended_name = selected_name if beats else baseline_name
    chart = output_dir / "forecast.png"
    _chart(series, holdout_index, holdout_predictions, baseline_name, selected_name, target, chart)

    predictions = pd.DataFrame(
        {
            "date": holdout_index,
            "actual": series.loc[holdout_index].to_numpy(),
            "baseline": holdout_predictions[baseline_name],
            "model": holdout_predictions[selected_name],
        }
    )
    predictions["recommended"] = predictions["model"] if beats else predictions["baseline"]
    next_date = holdout_index[-1] + series.index.freq
    next_value = _next_forecast(series, season, recommended_name)
    next_row = pd.DataFrame(
        [{"date": next_date, "actual": np.nan, "baseline": np.nan, "model": np.nan, "recommended": next_value}]
    )
    predictions = pd.concat([predictions, next_row], ignore_index=True)
    predictions_path = output_dir / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    return {
        "task": "forecast",
        "target": target,
        "date_column": date_column,
        "frequency": frequency_label,
        "season": season,
        "validation_start": str(validation_index.min().date()),
        "validation_end": str(validation_index.max().date()),
        "holdout_start": str(holdout_index.min().date()),
        "holdout_end": str(holdout_index.max().date()),
        "validation": validation_rows,
        "selected_model": selected_name,
        "baseline_name": baseline_name,
        "baseline_mae": baseline["mae"],
        "baseline_mape": baseline["mape"],
        "model_mae": chosen["mae"],
        "model_mape": chosen["mape"],
        "beats_baseline": beats,
        "recommended_name": recommended_name,
        "next_period": str(next_date.date()),
        "next_value": next_value,
        "chart": chart.name,
        "predictions": predictions_path.name,
        "changes": log_entries(log),
    }


def log_entries(log):
    return log.as_dicts()


def _next_forecast(series, season, name):
    if name == "seasonal_naive":
        return float(series.iloc[-season])
    if name == "last_value":
        return float(series.iloc[-1])
    if name == "residual_boosting":
        extended = series.copy()
        next_stamp = series.index[-1] + series.index.freq
        extended.loc[next_stamp] = np.nan
        predicted = _residual_boost(extended, series.index[-1], pd.DatetimeIndex([next_stamp]), season)
        return float(predicted[0])
    if name.startswith("holt_winters"):
        seasonal = "mul" if name.endswith("mul") else "add"
        model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal=seasonal,
            seasonal_periods=season,
            initialization_method="estimated",
        )
        return float(np.asarray(model.fit(optimized=True).forecast(1)).ravel()[0])
    if name == "sarima_lite":
        fitted = SARIMAX(
            series,
            order=(0, 1, 1),
            seasonal_order=(0, 1, 1, season),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=40)
        return float(np.asarray(fitted.forecast(1)).ravel()[0])
    return float(series.iloc[-season])


def _chart(series, holdout_index, predictions, baseline_name, selected_name, target, path):
    shown = series.iloc[-max(36, len(holdout_index) + 12) :]
    figure, axis = plt.subplots(figsize=(10.5, 5.2))
    axis.plot(shown.index, shown.to_numpy(), color="#1f2933", linewidth=2.2, label="Actual")
    axis.plot(holdout_index, predictions[baseline_name], color="#7b8794", linestyle="--", linewidth=2, label="Baseline")
    if selected_name != baseline_name and selected_name in predictions:
        axis.plot(holdout_index, predictions[selected_name], color="#c2410c", linewidth=2.2, label=selected_name.replace("_", " "))
    axis.axvspan(holdout_index.min(), holdout_index.max(), color="#c2410c", alpha=0.08)
    axis.set_title(f"Next-period forecast for {target}")
    axis.set_ylabel(target)
    axis.legend(frameon=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#e6e8eb")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
