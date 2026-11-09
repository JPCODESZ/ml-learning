"""Lead scoring for a phone campaign.

The question is: before the call, who is likely to say yes?
Call length is excluded on purpose. You do not know it until the call is over,
so using it would be cheating.
"""

import json
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"
OUT_DIR = HERE / "output"
SOURCE_URL = "https://archive.ics.uci.edu/static/public/222/data.csv"
DATA_FILE = DATA_DIR / "bank-marketing.csv"
# On the training set, keep at least this share of the eventual yeses,
# then take the call list with the highest precision.
MINIMUM_RECALL = 0.40


def download_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_FILE.exists() and DATA_FILE.stat().st_size > 0:
        print(f"Using saved file {DATA_FILE.name}")
        return
    print("Downloading the UCI Bank Marketing file...")
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "ml-learning-fiverr-sample/1.0 (educational portfolio)"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        DATA_FILE.write_bytes(response.read())


def clean(frame):
    report = {"rows_read": int(len(frame)), "columns": list(frame.columns)}
    working = frame.copy()
    working.columns = [column.strip() for column in working.columns]
    report["renamed_columns"] = {}
    # This file calls the column day_of_week, but the values run from 1 to 31.
    if "day_of_week" in working.columns and pd.to_numeric(working["day_of_week"], errors="coerce").max() > 7:
        working = working.rename(columns={"day_of_week": "day_of_month"})
        report["renamed_columns"]["day_of_week"] = "day_of_month"
    # Blank strings are missing values, not a real category.
    working = working.replace(r"^\s*$", np.nan, regex=True)
    report["missing_cells"] = int(working.isna().sum().sum())
    report["missing_by_column"] = {
        column: int(count)
        for column, count in working.isna().sum().items()
        if int(count) > 0
    }
    # Blank contact or previous-campaign outcome means "unknown", not a broken row.
    # Dropping those rows would throw away most of the list.
    for column in working.select_dtypes(exclude="number").columns:
        working[column] = working[column].fillna("unknown")
    before = len(working)
    working = working.drop_duplicates()
    report["duplicate_rows_removed"] = int(before - len(working))
    if "duration" not in working.columns or "y" not in working.columns:
        raise SystemExit("Expected columns duration and y were not in the file.")
    # Duration is only known after the call. A lead list cannot use it.
    working = working.drop(columns=["duration"])
    report["dropped_leakage_column"] = "duration"
    working["y"] = working["y"].astype(str).str.strip().str.lower()
    working = working[working["y"].isin(["yes", "no"])]
    report["rows_kept"] = int(len(working))
    report["yes_rate"] = float((working["y"] == "yes").mean())
    return working, report


def make_matrix(frame):
    features = frame.drop(columns=["y"])
    target = (frame["y"] == "yes").astype(int)
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    return features, target, numeric, categorical


def build_logistic(numeric, categorical):
    transform = ColumnTransformer(
        [
            ("number", StandardScaler(), numeric),
            (
                "category",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ]
    )
    return Pipeline(
        [
            ("prepare", transform),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=0,
                ),
            ),
        ]
    )


def build_boosting(numeric, categorical):
    transform = ColumnTransformer(
        [
            ("number", "passthrough", numeric),
            (
                "category",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ]
    )
    return Pipeline(
        [
            ("prepare", transform),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_depth=4,
                    learning_rate=0.08,
                    max_iter=300,
                    l2_regularization=0.1,
                    class_weight="balanced",
                    random_state=0,
                ),
            ),
        ]
    )


def rates_at_threshold(target, scores, threshold):
    called = scores >= threshold
    if called.sum() == 0:
        return {"precision": None, "recall": 0.0, "calls": 0}
    return {
        "precision": float(precision_score(target, called, zero_division=0)),
        "recall": float(recall_score(target, called, zero_division=0)),
        "calls": int(called.sum()),
    }


def choose_threshold(target, scores):
    """Pick a cutoff on the training scores, not on the holdout."""
    best = None
    for threshold in np.round(np.linspace(0.05, 0.95, 37), 4):
        result = rates_at_threshold(target, scores, threshold)
        if result["recall"] < MINIMUM_RECALL:
            continue
        if best is None or result["precision"] > best["precision"]:
            best = {"threshold": float(threshold), **result}
    if best is None:
        # If 40% recall is impossible, take the cutoff with the best recall.
        for threshold in np.round(np.linspace(0.05, 0.95, 37), 4):
            result = rates_at_threshold(target, scores, threshold)
            if best is None or result["recall"] > best["recall"]:
                best = {"threshold": float(threshold), **result}
        best["note"] = "No cutoff kept 40% of the yeses. This is the highest recall found."
    return best


def pretty_name(raw):
    labels = {
        "poutcome": "previous campaign result",
        "contact": "phone or mobile",
        "month": "month of the call",
        "day_of_month": "day of the month",
        "housing": "housing loan",
        "balance": "account balance",
        "campaign": "times contacted",
        "age": "age",
    }
    return labels.get(str(raw), str(raw).replace("_", " "))


def main():
    download_data()
    raw = pd.read_csv(DATA_FILE)
    cleaned, cleanup = clean(raw)
    features, target, numeric, categorical = make_matrix(cleaned)
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=0,
        stratify=target,
    )

    logistic = build_logistic(numeric, categorical)
    boosting = build_boosting(numeric, categorical)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    logistic_cv = cross_val_score(logistic, train_x, train_y, cv=folds, scoring="roc_auc")
    boosting_cv = cross_val_score(boosting, train_x, train_y, cv=folds, scoring="roc_auc")

    logistic.fit(train_x, train_y)
    boosting.fit(train_x, train_y)
    logistic_test = logistic.predict_proba(test_x)[:, 1]
    boosting_test = boosting.predict_proba(test_x)[:, 1]
    boosting_train = boosting.predict_proba(train_x)[:, 1]

    # A one-column rule: call people whose previous campaign already succeeded.
    simple_test = (test_x["poutcome"].astype(str) == "success").astype(float).to_numpy()
    base_rate = float(test_y.mean())
    chosen = choose_threshold(train_y.to_numpy(), boosting_train)
    holdout_at_choice = rates_at_threshold(test_y.to_numpy(), boosting_test, chosen["threshold"])
    holdout_at_half = rates_at_threshold(test_y.to_numpy(), boosting_test, 0.5)
    simple_rates = rates_at_threshold(test_y.to_numpy(), simple_test, 0.5)

    # Shuffle one original column at a time. That answers "which fact matters",
    # instead of splitting a fact like job into many tiny yes/no pieces.
    base_auc = float(roc_auc_score(test_y, boosting_test))
    generator = np.random.default_rng(0)
    driver_rows = []
    for column in test_x.columns:
        drops = []
        for _ in range(5):
            shuffled = test_x.copy()
            shuffled[column] = generator.permutation(shuffled[column].to_numpy())
            score = roc_auc_score(test_y, boosting.predict_proba(shuffled)[:, 1])
            drops.append(base_auc - float(score))
        driver_rows.append((column, float(np.mean(drops))))
    driver_rows.sort(key=lambda item: item[1], reverse=True)
    drivers = [
        {"feature": pretty_name(name), "auc_drop": drop}
        for name, drop in driver_rows[:8]
    ]

    result = {
        "rows_after_cleanup": cleanup["rows_kept"],
        "duplicate_rows_removed": cleanup["duplicate_rows_removed"],
        "missing_cells": cleanup["missing_cells"],
        "missing_by_column": cleanup["missing_by_column"],
        "dropped_leakage_column": cleanup["dropped_leakage_column"],
        "renamed_columns": cleanup["renamed_columns"],
        "train_yes_rate": float(train_y.mean()),
        "test_yes_rate": base_rate,
        "logistic_cv_auc_mean": float(logistic_cv.mean()),
        "logistic_cv_auc_std": float(logistic_cv.std()),
        "boosting_cv_auc_mean": float(boosting_cv.mean()),
        "boosting_cv_auc_std": float(boosting_cv.std()),
        "test_auc": {
            "constant_baseline": 0.5,
            "previous_success_rule": float(roc_auc_score(test_y, simple_test)),
            "logistic_regression": float(roc_auc_score(test_y, logistic_test)),
            "gradient_boosting": float(roc_auc_score(test_y, boosting_test)),
        },
        "previous_success_rule_on_test": simple_rates,
        "threshold_chosen_on_training": chosen,
        "boosting_on_test_at_chosen_threshold": holdout_at_choice,
        "boosting_on_test_at_0_5": holdout_at_half,
        "top_drivers": drivers,
        "boosting_beats_logistic_on_test_auc": bool(
            roc_auc_score(test_y, boosting_test) > roc_auc_score(test_y, logistic_test)
        ),
        "boosting_beats_simple_rule_on_test_auc": bool(
            roc_auc_score(test_y, boosting_test) > roc_auc_score(test_y, simple_test)
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 5.2))
    curve_axis, driver_axis = axes
    for scores, label, color in [
        (simple_test, "Previous success only", "#7b8794"),
        (logistic_test, "Logistic regression", "#1d4e89"),
        (boosting_test, "Gradient boosting", "#c2410c"),
    ]:
        false_rate, true_rate, _ = roc_curve(test_y, scores)
        curve_axis.plot(false_rate, true_rate, color=color, linewidth=2.2, label=label)
    curve_axis.plot([0, 1], [0, 1], color="#d9e2ec", linewidth=1.2, label="Call at random")
    curve_axis.set_title("Who is worth calling?")
    curve_axis.set_xlabel("Share of no's that we still call")
    curve_axis.set_ylabel("Share of yeses that we reach")
    curve_axis.legend(frameon=False, fontsize=8)
    curve_axis.spines["top"].set_visible(False)
    curve_axis.spines["right"].set_visible(False)

    driver_names = [item["feature"] for item in drivers][::-1]
    driver_values = [item["auc_drop"] for item in drivers][::-1]
    driver_axis.barh(driver_names, driver_values, color="#1d4e89")
    driver_axis.set_title("What moves the score")
    driver_axis.set_xlabel("How much the ranking slips if this is hidden")
    driver_axis.spines["top"].set_visible(False)
    driver_axis.spines["right"].set_visible(False)
    figure.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    chart = FIG_DIR / "lead_scores.png"
    figure.savefig(chart, dpi=140)
    plt.close(figure)

    print(json.dumps(result, indent=2))
    print(f"Chart: {chart}")


if __name__ == "__main__":
    main()
