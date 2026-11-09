"""Score which rows are worth acting on."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MINIMUM_RECALL = 0.40


def _split_columns(frame, feature_columns):
    numeric = [name for name in feature_columns if pd.api.types.is_numeric_dtype(frame[name])]
    categorical = [name for name in feature_columns if name not in numeric]
    return numeric, categorical


def _logistic(numeric, categorical):
    transform = ColumnTransformer(
        [
            ("number", StandardScaler(), numeric),
            ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ]
    )
    return Pipeline(
        [
            ("prepare", transform),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0)),
        ]
    )


def _boosting(numeric, categorical):
    transform = ColumnTransformer(
        [
            ("number", "passthrough", numeric),
            ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ]
    )
    return Pipeline(
        [
            ("prepare", transform),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_depth=3,
                    learning_rate=0.08,
                    max_iter=150,
                    l2_regularization=1.0,
                    class_weight="balanced",
                    random_state=0,
                ),
            ),
        ]
    )


def _simple_rule(train_x, train_y, test_x):
    """One yes/no fact from the training rows, such as a single category."""
    overall = float(train_y.mean())
    best = None
    for column in train_x.columns:
        values = train_x[column]
        if pd.api.types.is_numeric_dtype(values) and values.nunique(dropna=True) > 12:
            continue
        for level, index in values.groupby(values).groups.items():
            if len(index) < 25:
                continue
            rate = float(train_y.loc[index].mean())
            if rate <= overall:
                continue
            if best is None or rate > best["rate"]:
                best = {"column": column, "level": level, "rate": rate}
    if best is None:
        scores = np.full(len(test_x), overall)
        return scores, "no single category beat the average", overall
    column, level = best["column"], best["level"]
    scores = (test_x[column] == level).astype(float).to_numpy()
    return scores, f"{column} = {level}", best["rate"]


def _at_threshold(target, scores, threshold):
    called = np.asarray(scores) >= threshold
    if called.sum() == 0:
        return {"precision": None, "recall": 0.0, "calls": 0}
    return {
        "precision": float(precision_score(target, called, zero_division=0)),
        "recall": float(recall_score(target, called, zero_division=0)),
        "calls": int(called.sum()),
    }


def _choose_threshold(target, scores):
    best = None
    for threshold in np.round(np.linspace(0.05, 0.95, 37), 4):
        result = _at_threshold(target, scores, threshold)
        if result["precision"] is None or result["recall"] < MINIMUM_RECALL:
            continue
        if best is None or result["precision"] > best["precision"]:
            best = {"threshold": float(threshold), **result}
    if best is None:
        best = {"threshold": 0.5, **_at_threshold(target, scores, 0.5)}
        best["note"] = "No cutoff kept 40% of the events. 0.5 was used instead."
    return best


def _drivers(model, frame, target):
    base = roc_auc_score(target, model.predict_proba(frame)[:, 1])
    generator = np.random.default_rng(0)
    rows = []
    for column in frame.columns:
        drops = []
        for _ in range(3):
            shuffled = frame.copy()
            shuffled[column] = generator.permutation(shuffled[column].to_numpy())
            score = roc_auc_score(target, model.predict_proba(shuffled)[:, 1])
            drops.append(base - float(score))
        rows.append({"feature": str(column).replace("_", " "), "auc_drop": float(np.mean(drops))})
    rows.sort(key=lambda item: item["auc_drop"], reverse=True)
    return rows[:8]


def _chart(target, curves, drivers, path):
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 5.2))
    curve_axis, driver_axis = axes
    for scores, label, color in curves:
        false_rate, true_rate, _ = roc_curve(target, scores)
        curve_axis.plot(false_rate, true_rate, color=color, linewidth=2.2, label=label)
    curve_axis.plot([0, 1], [0, 1], color="#d9e2ec", linewidth=1.2, label="No skill")
    curve_axis.set_title("Who is worth acting on?")
    curve_axis.set_xlabel("Share of non-events still selected")
    curve_axis.set_ylabel("Share of events reached")
    curve_axis.legend(frameon=False, fontsize=8)
    curve_axis.spines["top"].set_visible(False)
    curve_axis.spines["right"].set_visible(False)
    names = [item["feature"] for item in drivers][::-1]
    values = [item["auc_drop"] for item in drivers][::-1]
    driver_axis.barh(names, values, color="#1d4e89")
    driver_axis.set_title("What moves the score")
    driver_axis.set_xlabel("How much the ranking slips if this is hidden")
    driver_axis.spines["top"].set_visible(False)
    driver_axis.spines["right"].set_visible(False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def run_classify(frame, target, id_columns, feature_columns, log, output_dir):
    if len(frame) < 80:
        raise SystemExit("Need at least 80 rows to hold some back for an honest test.")
    features = frame[feature_columns].copy()
    labels = frame[target].astype(int)
    if labels.nunique() < 2:
        raise SystemExit("The target has only one value, so there is nothing to score.")
    index = frame.index
    train_x, test_x, train_y, test_y, train_index, test_index = train_test_split(
        features,
        labels,
        index,
        test_size=0.20,
        random_state=0,
        stratify=labels,
    )
    numeric, categorical = _split_columns(frame, feature_columns)
    for column in categorical:
        train_x[column] = train_x[column].astype(str)
        test_x[column] = test_x[column].astype(str)

    logistic = _logistic(numeric, categorical)
    boosting = _boosting(numeric, categorical)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    logistic_cv = cross_val_score(logistic, train_x, train_y, cv=folds, scoring="roc_auc")
    boosting_cv = cross_val_score(boosting, train_x, train_y, cv=folds, scoring="roc_auc")
    selected_name = "gradient_boosting" if boosting_cv.mean() >= logistic_cv.mean() else "logistic_regression"
    selected = boosting if selected_name == "gradient_boosting" else logistic

    logistic.fit(train_x, train_y)
    boosting.fit(train_x, train_y)
    selected.fit(train_x, train_y)
    logistic_test = logistic.predict_proba(test_x)[:, 1]
    boosting_test = boosting.predict_proba(test_x)[:, 1]
    selected_test = boosting_test if selected_name == "gradient_boosting" else logistic_test
    selected_train = selected.predict_proba(train_x)[:, 1]
    simple_test, simple_description, simple_train_rate = _simple_rule(train_x, train_y, test_x)
    chosen = _choose_threshold(train_y.to_numpy(), selected_train)
    holdout_choice = _at_threshold(test_y.to_numpy(), selected_test, chosen["threshold"])
    simple_rates = _at_threshold(test_y.to_numpy(), simple_test, 0.5)
    simple_auc = float(roc_auc_score(test_y, simple_test)) if np.unique(simple_test).size > 1 else 0.5
    selected_auc = float(roc_auc_score(test_y, selected_test))
    beats = selected_auc > max(simple_auc, 0.5) + 1e-9

    drivers = _drivers(selected, test_x, test_y)
    chart = output_dir / "scores.png"
    _chart(
        test_y,
        [
            (simple_test, "Simple rule", "#7b8794"),
            (logistic_test, "Logistic regression", "#1d4e89"),
            (boosting_test, "Gradient boosting", "#c2410c"),
        ],
        drivers,
        chart,
    )

    features_for_score = frame[feature_columns].copy()
    for column in categorical:
        features_for_score[column] = features_for_score[column].astype(str)
    all_scores = selected.predict_proba(features_for_score)[:, 1]
    identifier = id_columns[0] if id_columns else None
    export = pd.DataFrame(
        {
            "row_id": frame[identifier] if identifier else frame.index,
            "actual": labels.to_numpy(),
            "score": all_scores,
            "act": (all_scores >= chosen["threshold"]).astype(int) if beats else np.zeros(len(frame), dtype=int),
            "split": np.where(frame.index.isin(test_index), "holdout", "train"),
        }
    )
    predictions_path = output_dir / "scored_rows.csv"
    export.to_csv(predictions_path, index=False)
    log.add(
        "chose a cutoff on the training rows",
        f"threshold {chosen['threshold']} was picked before looking at the holdout",
    )

    return {
        "task": "classify",
        "target": target,
        "rows": int(len(frame)),
        "event_rate": float(labels.mean()),
        "simple_rule": simple_description,
        "simple_rule_train_rate": simple_train_rate,
        "logistic_cv_auc": float(logistic_cv.mean()),
        "logistic_cv_auc_std": float(logistic_cv.std()),
        "boosting_cv_auc": float(boosting_cv.mean()),
        "boosting_cv_auc_std": float(boosting_cv.std()),
        "selected_model": selected_name,
        "test_auc": {
            "constant_baseline": 0.5,
            "simple_rule": simple_auc,
            "logistic_regression": float(roc_auc_score(test_y, logistic_test)),
            "gradient_boosting": float(roc_auc_score(test_y, boosting_test)),
        },
        "simple_rule_holdout": simple_rates,
        "threshold": chosen,
        "holdout_at_threshold": holdout_choice,
        "beats_baseline": beats,
        "top_drivers": drivers,
        "chart": chart.name,
        "predictions": predictions_path.name,
        "changes": log.as_dicts(),
        "leakage_note": "Possible leakage columns were left out of the score.",
    }
